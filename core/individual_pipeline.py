"""nnU-Net -> individual electrode masks -> BrainLFNet, without SPM/MATLAB."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from core.code_assets import code_asset_path


TRAINING_SPACING_MM = (1.2, 0.9375, 0.9375)


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with code_asset_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_asset(path):
    path = Path(path)
    actual = code_asset_path(path)
    if not actual.is_file():
        raise FileNotFoundError(f"缺少模型／模板文件: {path}")
    with actual.open("rb") as stream:
        if stream.read(60).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"仍是 Git LFS 指针，请下载真实文件: {path}")
    return path


def nnunet_package(brain_root):
    root = Path(brain_root).resolve() / "SegandElec_nnUNet/model"
    for name in ("checkpoint_final.pth", "plans.json", "dataset.json", "inference_config.json", "postprocessing.json"):
        require_asset(root / name)
    return root, root / "checkpoint_final.pth"


def validate_tissues(labels):
    """The audited checkpoint already produces WM=1, GM=2. Never swap them."""
    labels = np.asarray(labels)
    if not np.isfinite(labels).all() or not np.all(labels == np.rint(labels)) or not np.isin(labels, np.arange(7)).all():
        raise ValueError("组织分割必须包含整数标签 0..6")
    return labels.astype(np.uint8)


def preprocessing_settings(padding=10):
    from importlib.metadata import version
    return {"version": 5, "backend": "SegandElec_nnUNet", "padding": int(padding),
             "spatial_mode": "legacy", "grid_policy": "training_spacing_native_directions_ras_padding",
            "target_spacing_mm":list(TRAINING_SPACING_MM), "resampling_order":1,
            "resample_grid":"native", "resample_interpolation":"linear", "resample_antialias":True,
            "inferior_crop_grid":"native",
            "field_of_view_policy": "bulk_gmwm_inferior_margin_v1", "inferior_margin_mm": 20.0,
            "cleanup": "roast-neck", "model_label_order": "verified",
            "landmark_method": "python-affine", "calibration": "extracranial_preserve_intracranial",
            "mirroring": True, "step_size": .5, "threads": 4,
            "nnunet_version": version("nnunetv2"), "torch_version": version("torch"),
            "adapter_sha256": file_hash(__file__)}


def select_device(mode="auto", progress=print):
    """Execute representative kernels in a disposable process before selecting GPU."""
    import torch
    from core.cuda_compat import run_gpu_probe
    if mode not in ("auto", "cuda", "cpu"):
        raise ValueError("Unknown inference device")
    if mode != "cpu":
        progress("正在检查显卡与 GPU 运行库兼容性…")
        probe = run_gpu_probe()
        if probe.get("usable"):
            progress("显卡兼容性检查通过，正在使用 GPU 加速计算")
            return torch.device("cuda:0")
        if mode == "cuda":
            raise RuntimeError("GPU 兼容性检查未通过：" + str(probe.get("message", probe.get("reason", ""))))
        if probe.get("reason") == "unavailable":
            progress("未检测到可用 GPU，自动使用 CPU")
        else:
            name = probe.get("gpu_name", "当前显卡")
            progress(f"{name} 的 GPU 运算自检未通过，自动使用 CPU；可检查显卡驱动与运行库兼容性。")
    progress("正在使用 CPU 计算，完成时间可能较长")
    return torch.device("cpu")


def preprocessing_backend(brain_root):
    root = Path(brain_root).resolve()
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    from SegandElec import pipeline_common
    # Python modules and packaged data have different resource roots.
    pipeline_common.PACKAGE_ROOT = root / "SegandElec"
    from SegandElec_nnUNet.pipeline import run_nnunet_pipeline
    return run_nnunet_pipeline


def prepare_subject(t1, brain_root, output_dir, subject, padding, device, progress, timing=None):
    package, _ = nnunet_package(brain_root)
    backend = preprocessing_backend(brain_root)
    translations = (("Preparing", "统一 MRI 方向并补零"),
                    ("Running full-head", "nnU-Net 正在进行全头部分割"),
                    ("Building SegandElec", "正在清理组织并校准头皮与颅骨"),
                    ("Placing", "Python 仿射标志点定位及 72 电极放置"),
                    ("preprocessing complete", "组织分割与电极放置完成"))
    def report(message):
        progress(next((text for token, text in translations if token in message), message))
    options=dict(t1=t1, output_dir=output_dir, subject=subject, padding=padding, target_spacing_mm=TRAINING_SPACING_MM,
                   resample_grid="native", resample_interpolation="linear", resample_antialias=True,
                   inferior_crop_grid="native", inferior_margin_mm=20.0,
                   model_root=package, device=str(device), cleanup="roast-neck",
                   model_label_order="verified", landmark_method="python-affine",
                   probability_calibration=Path(brain_root)/"SegandElec_nnUNet/configs/extracranial_calibration.json",
                   probability_template=Path(brain_root)/"SegandElec/assets/eTPM.nii",
                   mirroring=True, step_size=.5, threads=4,
                   save_probabilities=False, save_intermediates=False, overwrite=True, progress=report, timing_callback=timing)
    from core.runtime import is_cuda_oom
    from core.cuda_compat import is_cuda_compatibility_error, safe_cuda_cleanup
    fallback=False
    compatibility_failure=False
    try:
        result=backend(**options)
    except RuntimeError as error:
        kind=getattr(device,'type',str(device).split(':')[0])
        compatibility_failure=is_cuda_compatibility_error(error)
        if kind!='cuda' or not (is_cuda_oom(error) or compatibility_failure): raise
        fallback=True
    if fallback:
        import torch
        gc.collect();safe_cuda_cleanup()
        progress('nnU-Net GPU 运行库不兼容，自动使用 CPU 继续组织分割' if compatibility_failure
                 else 'nnU-Net 显存不足，自动使用 CPU 继续组织分割')
        options['device']='cpu'
        result=backend(**options)
    return result



def validate_preprocessing(prepared, task_dir):
    """Validate anatomy, label meaning, all cap channels and artifact ownership."""
    task_dir = Path(task_dir).resolve()
    files = ("padded_t1", "tissue_masks", "electrode_mask", "gel_mask",
             "electrode_geometry", "electrode_coordinates", "manifest")
    for key in files:
        path = Path(prepared[key]).resolve()
        if not path.is_relative_to(task_dir) or not path.is_file():
            raise ValueError(f"预处理结果文件缺失或不属于当前任务：{key}")
    image = nib.load(prepared["padded_t1"])
    if len(image.shape) != 3 or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-8:
        raise ValueError("无效的个体 MRI 网格")
    # Legacy training padding can retain oblique/nonorthogonal direction cosines.
    # Compare column lengths in the output R/A/S voxel-axis order, not diagonals.
    spacing = np.linalg.norm(image.affine[:3,:3],axis=0)
    if nib.aff2axcodes(image.affine) != ("R","A","S") or not np.allclose(
            spacing,TRAINING_SPACING_MM,atol=1e-4,rtol=0):
        raise ValueError("预处理必须使用 R/A/S 体素轴 1.2×0.9375×0.9375 mm 网格")
    volumes = {}
    for key in ("tissue_masks", "electrode_mask", "gel_mask"):
        volume = nib.load(prepared[key])
        if volume.shape != image.shape or not np.allclose(volume.affine, image.affine, atol=1e-4):
            raise ValueError("组织、电极、凝胶与 padded T1 必须处于同一空间")
        values = np.asarray(volume.dataobj)
        if not np.isfinite(values).all() or np.any(values != np.rint(values)):
            raise ValueError("预处理掩膜必须为有限整数标签")
        volumes[key] = values
    tissue = validate_tissues(volumes['tissue_masks'])
    if not np.isin([1,2,3,4,5], np.unique(tissue)).all():
        raise ValueError("分割缺少 WM/GM/CSF/骨/皮肤，请检查 MRI 视野")
    for key in ('electrode_mask', 'gel_mask'):
        if not np.array_equal(np.unique(volumes[key]), np.arange(73)):
            raise ValueError("电极与凝胶掩膜必须完整包含标签 1..72")
    if np.any((tissue > 0) & ((volumes['electrode_mask'] > 0) | (volumes['gel_mask'] > 0))) or np.any((volumes['electrode_mask'] > 0) & (volumes['gel_mask'] > 0)):
        raise ValueError("组织、电极或凝胶掩膜相互重叠")
    if prepared.get('model_label_order') != 'verified' or prepared.get('landmark_method') != 'python-affine' or prepared.get('requires_spm_runtime') is not False or prepared.get('cleanup') != 'roast-neck' or prepared.get('preserve_uncalibrated_intracranial') is not True:
        raise ValueError("预处理后端或组织标签约定与当前集成不一致")
    fov = prepared.get('field_of_view') or {}
    if (fov.get('policy') != 'bulk_gmwm_inferior_margin_v1' or fov.get('margin_mm') != 20.0
            or fov.get('grid_policy') != 'native_voxel_slices_without_interpolation'):
        raise ValueError('预处理未按当前脑下缘 20 mm 扫描范围规则核对。')


def _predict_brainlfnet_once(tissue_path, electrode_path, padded_t1, brain_root, output_dir, device, progress, inference_mode=None, memory_plan=None):
    import torch
    from contextlib import nullcontext
    from brainlfnet import BrainLFNet

    image = nib.load(str(tissue_path))
    electrode_image = nib.load(str(electrode_path))
    tissue = validate_tissues(np.asarray(image.dataobj))
    electrodes = np.asarray(electrode_image.dataobj, dtype=np.uint8)
    if tissue.shape != electrodes.shape or not np.allclose(image.affine, electrode_image.affine, atol=1e-4):
        raise ValueError("组织和电极掩膜不在同一空间")
    if not np.array_equal(np.unique(electrodes[electrodes > 0]), np.arange(1, 73)):
        raise ValueError("电极掩膜必须完整包含标签 1..72")
    checkpoint_path = require_asset(Path(brain_root) / "checkpoints/best_ema.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = dict(checkpoint.get("args", {}))
    companion = checkpoint_path.with_name("args.json")
    if companion.is_file():
        configuration.update(json.loads(companion.read_text(encoding="utf-8")))
    inference_mode=inference_mode or ('cuda_autocast_fp16' if device.type=='cuda' else 'float32')
    low_memory=inference_mode=='cuda_low_memory_fp16'
    model=BrainLFNet(base_channels=int(configuration.get('base_channels',16)),activation_checkpointing=False).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    if low_memory:model.half()
    model=model.to(device)
    if low_memory:
        from core.low_memory_inference import LowMemoryInference
        predictor=LowMemoryInference(model)
    else:predictor=model
    del checkpoint
    scale = float(configuration.get("e_scale_value", configuration.get("e_scale", 0.11916366219520569)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid checkpoint field scaling")
    ijk = np.argwhere((tissue == 1) | (tissue == 2)).astype(np.int32)
    if not len(ijk):
        raise ValueError("分割中没有灰／白质")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    state_path = output / "prediction_progress.json"
    partial = output / "leadfield_gmwm.partial.npy"
    shape = (len(ijk), 71, 3)
    input_signature={role:file_hash(path) for role,path in zip(('tissue','electrodes','padded_t1','checkpoint'),(tissue_path,electrode_path,padded_t1,checkpoint_path))}
    recipe_hash=file_hash(Path(__file__).with_name('low_memory_inference.py'))
    column_precisions={}
    complete = []
    prior_device = None
    resumed_columns = False
    if partial.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        leadfield = np.load(partial, mmap_mode="r+")
        if leadfield.shape != shape:
            leadfield._mmap.close()
            raise ValueError("Partial leadfield geometry mismatch")
        complete=state.get('completed_labels',[])
        if state.get('input_signature') and state['input_signature']!=input_signature:
            leadfield._mmap.close();raise ValueError('续算导联场的输入文件与已保存列不同')
        prior_device=state.get('last_device')
        column_precisions=dict(state.get('column_precisions',{}))
        previous_mode=state.get('last_precision') or ('cuda_autocast_fp16' if str(prior_device).startswith('cuda') else 'float32' if prior_device=='cpu' else 'unknown_legacy')
        for label in complete:column_precisions.setdefault(str(label),previous_mode)
        if 'cuda_low_memory_fp16' in column_precisions.values() and state.get('inference_recipe_sha256')!=recipe_hash:
            leadfield._mmap.close();raise ValueError('低显存推理实现已变化，不能混用此前未完成的列')
        resumed_columns = bool(complete)
    else:
        leadfield = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape, fortran_order=True)
    try:
        use_half = device.type == "cuda"
        dtype = torch.float16 if use_half else torch.float32
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        storage_device=torch.device('cpu') if low_memory else device
        # Avoid int64 one-hot expansion; low-memory mode keeps reusable anatomy on CPU.
        anatomy = torch.stack([torch.from_numpy((tissue == k) | ((tissue == 6) & (k == 0))) for k in range(6)]).to(device=storage_device, dtype=dtype)
        reference = torch.from_numpy(electrodes == 72).to(device=storage_device, dtype=dtype)
        query = torch.from_numpy(ijk.astype(np.int64)).unsqueeze(0).to(device)
        for active in range(1, 72):
            if active in complete:
                progress(f"复用导联场列 {active}/71")
                continue
            progress(f"BrainLFNet 预测 {active}/71")
            mask = torch.from_numpy(electrodes == active).to(device=storage_device, dtype=dtype)
            network_input = torch.cat([anatomy, mask[None], reference[None]], dim=0).unsqueeze(0).to(device)
            with torch.inference_mode(), (torch.autocast("cuda",dtype=torch.float16) if use_half and not low_memory else nullcontext()):
                prediction = predictor(network_input, query)[0].float() * scale
            values = prediction.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"第 {active} 列出现非有限数值")
            leadfield[:, active - 1] = values
            leadfield.flush()
            complete.append(active)
            column_precisions[str(active)]=inference_mode
            save_json(state_path,{'completed_labels':complete,'shape':list(shape),'last_device':str(device),
                'last_precision':inference_mode,'column_precisions':column_precisions,
                'input_signature':input_signature,'inference_recipe_sha256':recipe_hash})
            del prediction, values, mask, network_input
        leadfield_fortran = leadfield.flags.f_contiguous
        leadfield.flush()
        leadfield._mmap.close()
        del leadfield
        partial.replace(output / "leadfield_gmwm.npy")
        np.savez_compressed(output / "leadfield_geometry.npz", valid_ijk=ijk, affine=image.affine,
                            grid_shape=tissue.shape, voxel_size_mm=image.header.get_zooms()[:3],
                            active_electrode_labels=np.arange(1, 72, dtype=np.int16), reference_electrode_label=72)
        result = {"leadfield": str(output / "leadfield_gmwm.npy"), "geometry": str(output / "leadfield_geometry.npz"),
                  "shape": list(shape), "units": "V/m for +1 mA active and -1 mA at Iz", "padded_t1": str(padded_t1),
                  "tissue_masks": str(tissue_path), "electrode_masks": str(electrode_path), "checkpoint": str(checkpoint_path),
                  'device':str(device),'inference_mode':inference_mode,
                  'precision': next(iter(set(column_precisions.values()))) if len(set(column_precisions.values()))==1 else 'mixed_precision_columns',
                  'column_precisions':column_precisions,'inference_recipe_sha256':recipe_hash,
                  'storage_order':'Fortran' if leadfield_fortran else 'C','continued_from_device':prior_device}
        if device.type=='cuda':
            result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            result['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
        save_json(output / "BrainLFNet_prediction.json", result)
        del predictor,model,anatomy,reference,query
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    finally:
        mapping = getattr(locals().get('leadfield'), '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.flush(); mapping.close()


def predict_brainlfnet(tissue_path, electrode_path, padded_t1, brain_root, output_dir, device, progress):
    import torch
    from core.runtime import plan_leadfield_device, is_cuda_oom,allocator_budget
    from core.cuda_compat import is_cuda_compatibility_error,safe_cuda_cleanup
    shape=nib.load(str(tissue_path)).shape
    config_path=Path(brain_root)/'checkpoints/args.json'
    base=int(json.loads(config_path.read_text(encoding='utf-8')).get('base_channels',16))
    chosen,plan=plan_leadfield_device(shape,base,device,progress)
    attempts=[];fallback=False
    while True:
        mode=plan.get('inference_mode','cuda_autocast_fp16' if chosen.type=='cuda' else 'float32')
        attempts.append({'device':str(chosen),'inference_mode':mode})
        failed=False;compatibility_failure=False
        try:
            with allocator_budget(plan,chosen):
                result=_predict_brainlfnet_once(tissue_path,electrode_path,padded_t1,brain_root,output_dir,chosen,progress,
                    inference_mode=mode,memory_plan=plan)
        except RuntimeError as error:
            compatibility_failure=is_cuda_compatibility_error(error)
            if chosen.type!='cuda' or not (is_cuda_oom(error) or compatibility_failure):raise
            failed=True;fallback=True
            attempts[-1]['outcome']='cuda_compatibility_error' if compatibility_failure else 'cuda_oom'
        if not failed:
            attempts[-1]['outcome']='completed';break
        # Outside exception scope: discard traceback-held activations before another attempt.
        gc.collect();safe_cuda_cleanup()
        if compatibility_failure:
            progress('导联场 GPU 运行库不兼容，自动切换 CPU 并继续已保存列')
            chosen,plan=plan_leadfield_device(shape,base,torch.device('cpu'),progress)
            plan['runtime_compatibility_fallback']=True
        elif mode=='cuda_autocast_fp16':
            progress('标准 GPU 推理显存不足，正在尝试低显存模式并继续已保存列')
            chosen,plan=plan_leadfield_device(shape,base,device,progress,prefer_low_memory=True)
        else:
            progress('低显存 GPU 推理仍不足，自动切换 CPU 并继续已保存列')
            chosen,plan=plan_leadfield_device(shape,base,torch.device('cpu'),progress)
        if not compatibility_failure:plan['runtime_oom_fallback']=True
    result['memory_plan']=plan;result['inference_attempts']=attempts
    result['cuda_fallback']=device.type=='cuda' and chosen.type=='cpu'
    result['runtime_gpu_retry']=fallback
    save_json(Path(output_dir)/'BrainLFNet_prediction.json',result)
    return result


def prediction_assets(t1, brain_root):
    root = Path(brain_root).resolve()
    package, checkpoint = nnunet_package(root)
    assets = [require_asset(Path(t1).resolve()), checkpoint,
              root / "checkpoints/best_ema.pt", root / "checkpoints/args.json",
              root / "SegandElec/assets/eTPM.nii", root / "SegandElec/assets/capInfo.xlsx",
              root / "SegandElec_nnUNet/configs/extracranial_calibration.json"]
    assets += [package/name for name in ("plans.json", "dataset.json", "inference_config.json", "postprocessing.json")]
    assets += [root/"SegandElec_nnUNet"/name for name in
               ("__init__.py", "pipeline.py", "field_of_view.py", "inference.py", "labels.py", "landmarks.py", "python_affine.py")]
    assets += [root/"SegandElec"/name for name in
               ("pipeline_common.py", "spatial_compat.py", "roast_touchup.py", "electrodes.py")]
    assets += [root/"brainlfnet/model.py"]
    return [require_asset(p) for p in assets]


def run_pipeline(config, progress=print, timing=None):
    root = Path(config["brain_root"]).resolve()
    padding = int(config.get("padding", 10))
    assets = prediction_assets(config["t1"], root)
    progress("正在核对 MRI、模型、校准参数与模板文件")
    device = select_device(config.get("device", "auto"), progress)
    from core.cuda_compat import last_gpu_probe
    gpu_check=last_gpu_probe() if config.get('device','auto')!='cpu' else None
    signature = preprocessing_settings(padding)
    signature.update(hashes={str(p):file_hash(p) for p in assets}, device=device.type)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(config["t1"]).name.split(".")[0])[:50] or "subject"
    task = Path(config["output_root"]).resolve() / f"{stem}_{key[:12]}"
    from core.cache_contract import signatures_match, assets_match, normalized
    from core.runtime import is_one_mm_image
    current_hashes={normalized(p.relative_to(root)):signature['hashes'][str(p)] for p in assets[1:]}
    matches=[]
    output_root=Path(config['output_root']).resolve()
    if output_root.is_dir():
        for path in output_root.rglob('manifest.json'):
            try:
                old=json.loads(path.read_text(encoding='utf-8'))
                if not signatures_match(old['signature'],signature,allow_native_1mm=is_one_mm_image(config['t1'])): continue
                old_hashes={normalized(p):h for p,h in old['signature']['hashes'].items()}
                if old_hashes.pop(normalized(old['t1']))!=signature['hashes'][str(assets[0])]: continue
                if not assets_match(old_hashes,current_hashes): continue
                if old.get('stages',{}).get('preprocessing'):
                    matches.append((bool(old['stages'].get('prediction')),path.stat().st_mtime,path))
            except (OSError,ValueError,KeyError,TypeError): continue
    if matches: task=max(matches,key=lambda item:item[:2])[2].parent.resolve()
    task.mkdir(parents=True, exist_ok=True)
    manifest_path = task / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest['fingerprint'] != key:
            manifest['previous_fingerprint']=manifest['fingerprint']
            manifest['fingerprint']=key; manifest['signature']=signature
    else:
        manifest = {"fingerprint":key, "signature":signature, "t1":str(Path(config["t1"]).resolve()), "stages":{}}
    manifest['device_check']=gpu_check
    save_json(manifest_path, manifest)
    prepared = manifest['stages'].get('preprocessing')
    if prepared:
        try:
            validate_preprocessing(prepared, task)
        except (OSError, ValueError, KeyError, TypeError, nib.filebasedimages.ImageFileError) as exc:
            from uuid import uuid4
            progress(f"已保存预处理不完整，将重新生成：{exc}")
            old_prediction = (task/'prediction').resolve()
            if old_prediction.exists():
                if not old_prediction.is_relative_to(task): raise ValueError('Prediction path outside task')
                old_prediction.rename(task/f'prediction_incomplete_{uuid4().hex[:8]}')
            manifest['stages'] = {}
            manifest.pop('atlases', None)
            save_json(manifest_path, manifest)
            prepared = None
        else:
            progress("复用已完成的组织分割与电极放置")
            if timing:
                for stage in ('mri_preparation','tissue_segmentation','electrode_placement'):
                    timing(dict(prepared.get('timings',{}).get(stage,{}),stage=stage,state='cached'))
    if not prepared:
        progress("正在准备 nnU-Net 个体组织及电极模型")
        prepared = prepare_subject(config['t1'], root, task/'preprocessing', stem, padding, device, progress, timing=timing)
        validate_preprocessing(prepared, task)
        manifest['stages'].update(preprocessing=prepared, padded=prepared['padded_t1'],
                                  segmentation=prepared['tissue_masks'], electrodes=prepared)
        save_json(manifest_path, manifest)
    prediction = task / 'prediction'
    if manifest['stages'].get('prediction'):
        from core.ti import Leadfield
        lf = Leadfield(prediction)
        try:
            image = nib.load(prepared['padded_t1'])
            if lf.shape != image.shape or not np.allclose(lf.affine,image.affine,atol=1e-4):
                raise ValueError('Cached leadfield grid mismatch')
            json.loads((prediction/'BrainLFNet_prediction.json').read_text(encoding='utf-8'))
        finally: lf.close()
        progress('复用已完成的个体导联场')
        if timing: timing(dict(manifest.get('timings',{}).get('leadfield_prediction',{}),stage='leadfield_prediction',state='cached'))
    else:
        preprocessing_backend(root)  # Make BrainLFNet importable in source and frozen workers.
        gc.collect()
        if device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
        from core.runtime import timed_stage
        with timed_stage('leadfield_prediction',timing,device.type) as duration:
            result = predict_brainlfnet(prepared['tissue_masks'], prepared['electrode_mask'],
                                       prepared['padded_t1'], root, prediction, device, progress)
            duration['device']=result['device']
        manifest.setdefault('timings',{}).update(prepared.get('timings',{}))
        manifest['timings']['leadfield_prediction']=duration
        result['preprocessing_manifest'] = prepared['manifest']
        save_json(prediction/'BrainLFNet_prediction.json', result)
        manifest['stages']['prediction'] = result
        save_json(manifest_path, manifest)
    return {'task_dir':str(task), 'manifest':str(manifest_path), 't1':manifest['t1'],
            'padded_t1':prepared['padded_t1'], 'tissue_masks':prepared['tissue_masks'],
            'prediction_dir':str(prediction), 'electrodes':prepared, 'timings':manifest.get('timings',prepared.get('timings',{}))}
