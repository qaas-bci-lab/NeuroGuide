"""Conservative allocation estimates and stage events for individual workers."""
from contextlib import contextmanager
import math
import os
import time


def is_one_mm_image(path):
    import nibabel as nib
    import numpy as np
    basis=nib.load(str(path)).affine[:3,:3]
    return bool(np.allclose(basis.T@basis,np.eye(3),atol=1e-3,rtol=0)
                and np.all(np.sum(np.abs(basis)>1e-3,axis=0)==1))


def estimate_brainlfnet_bytes(shape, base_channels=16):
    """Heuristic, not an allocation guarantee: dense skips, decoder and FP32 norm.

    Twenty-four bytes per base feature cover retained FP32 encoder skips,
    promoted decoder concatenations and normalization temporaries. The
    caller also reserves CUDA workspace and space for other applications.
    """
    if len(shape) != 3 or any(int(n) < 1 for n in shape):
        raise ValueError('Invalid prediction grid')
    return math.prod(int(n) for n in shape) * (24 * int(base_channels) + 40) + 128 * 1024**2


def estimate_low_memory_bytes(shape,base_channels=16):
    # Full-volume FP16 activations, owned residual buffers, plus cuDNN/cache room.
    if len(shape)!=3 or any(int(n)<1 for n in shape):raise ValueError('Invalid prediction grid')
    return math.prod(int(n) for n in shape)*(10*int(base_channels)+64)+128*1024**2


def plan_leadfield_device(shape, base_channels, candidate, progress=lambda _: None, prefer_low_memory=True):
    """Default to low-memory GPU; False retains standard-first selection."""
    import torch
    estimate = estimate_brainlfnet_bytes(shape, base_channels)
    low_estimate=estimate_low_memory_bytes(shape,base_channels)
    progress('推荐设备：GPU 显存 ≥ 4 GB（大体积 MRI 建议 ≥ 8 GB；显存不足时自动使用 CPU）')
    plan = {'estimated_peak_bytes': estimate,'standard_estimated_peak_bytes':estimate,
            'low_memory_estimated_peak_bytes':low_estimate,'grid_shape':list(shape),'estimate_kind':'conservative_heuristic'}
    if candidate.type != 'cuda':
        torch.set_num_threads(max(1,min(8,os.cpu_count() or 1)))
        plan['cpu_threads']=torch.get_num_threads()
        plan.update(device='cpu',inference_mode='float32'); return torch.device('cpu'), plan
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(candidate)
    reserve = max(512 * 1024**2, int(total * .15))
    plan.update(free_bytes=free, total_bytes=total, reserved_headroom_bytes=reserve)
    available=max(0,free-reserve)
    if estimate<=available and not prefer_low_memory:
        progress(f'导联场显存预估 {estimate/2**30:.1f} GB，当前可用 {free/2**30:.1f} GB，使用 GPU')
        plan.update(device=str(candidate),inference_mode='cuda_autocast_fp16',allocator_limit_bytes=available)
        return candidate,plan
    if low_estimate<=available:
        progress(f'使用低显存 GPU 推理：预计约 {low_estimate/2**30:.1f} GB，当前可用 {free/2**30:.1f} GB；中间特征由系统内存暂存')
        plan.update(device=str(candidate),inference_mode='cuda_low_memory_fp16',estimated_peak_bytes=low_estimate,
                    allocator_limit_bytes=min(available,int(low_estimate*1.1)))
        return candidate,plan
    if low_estimate>available:
        progress(f'低显存推理预计需要约 {low_estimate/2**30:.1f} GB，当前可用 {free/2**30:.1f} GB；自动使用 CPU')
        torch.set_num_threads(max(1,min(8,os.cpu_count() or 1)))
        plan.update(device='cpu',inference_mode='float32',cpu_threads=torch.get_num_threads())
        return torch.device('cpu'), plan


def is_cuda_oom(error):
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'cuda out of memory' in str(error).casefold())


@contextmanager
def allocator_budget(plan,device):
    """Budget the isolated prediction worker; restore its allocator on every exit."""
    import torch
    limit=plan.get('allocator_limit_bytes')
    if device.type!='cuda' or not limit:
        yield;return
    getter=getattr(torch.cuda,'get_per_process_memory_fraction',None)
    previous=getter(device) if getter else 1.
    torch.cuda.set_per_process_memory_fraction(min(1.,limit/plan['total_bytes']),device)
    try:yield
    finally:
        try:torch.cuda.set_per_process_memory_fraction(previous,device)
        except RuntimeError as error:
            from core.cuda_compat import is_cuda_compatibility_error
            if not is_cuda_compatibility_error(error):raise


@contextmanager
def timed_stage(stage, callback=None, device=None):
    start=time.perf_counter()
    if callback: callback({'stage':stage,'state':'running','device':device})
    record={}
    try:
        yield record
    except BaseException:
        record.update(seconds=time.perf_counter()-start, state='failed')
        record.setdefault('device',device)
        if callback: callback(dict(record,stage=stage))
        raise
    else:
        record.update(seconds=time.perf_counter()-start,state='completed')
        record.setdefault('device',device)
        if callback: callback(dict(record,stage=stage))
