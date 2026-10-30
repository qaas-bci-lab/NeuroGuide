"""Isolated CUDA execution probe, so failed GPU contexts never enter inference."""
from pathlib import Path
import json
import os
import subprocess
import sys

PREFIX = "NEUROGUIDE_CUDA_PROBE "
_last_probe = None


def is_cuda_compatibility_error(error):
    if not isinstance(error, RuntimeError):
        return False
    text = str(error).casefold()
    return any(token in text for token in (
        "no kernel image is available", "invalid device function",
        "not compatible with the current pytorch", "cuda driver version is insufficient",
        "found no nvidia driver", "provided ptx was compiled", "unsupported ptx version",
        "cuda_error_unsupported_ptx_version", "cuda_error_system_driver_mismatch",
        "cuda initialization error", "cuda driver initialization failed",
        "cudnn_status_arch_mismatch", "cudaerrorcallrequiresnewerdriver",
        "call requires newer driver", "cuda error: operation not supported",
        "cudnn_status_not_supported", "cublas_status_not_supported",
        "unable to find a valid cudnn algorithm",
        "get was unable to find an engine to execute this computation",
    ))


def safe_cuda_cleanup():
    import torch
    try:
        torch.cuda.empty_cache()
    except RuntimeError as error:
        if not is_cuda_compatibility_error(error):
            raise


def probe_main():
    info = {"usable": False}
    try:
        import torch
        import torch.nn.functional as functional
        info.update(torch_version=torch.__version__, cuda_build=torch.version.cuda)
        if not torch.cuda.is_available():
            info.update(reason="unavailable", message="CUDA is not available")
        else:
            info.update(gpu_name=torch.cuda.get_device_name(0),
                        capability=list(torch.cuda.get_device_capability(0)),
                        compiled_arches=torch.cuda.get_arch_list())
            operations = []
            with torch.inference_mode():
                for dtype in (torch.float32, torch.float16):
                    volume = torch.randn((1, 8, 12, 12, 12), device="cuda:0", dtype=dtype)
                    weights = torch.randn((8, 8, 3, 3, 3), device="cuda:0", dtype=dtype)
                    result = functional.conv3d(volume, weights, padding=1)
                    result = functional.group_norm(result, 4)
                    result = functional.gelu(result)
                    result = functional.interpolate(result, size=(15, 15, 15), mode="trilinear", align_corners=False)
                    context = result.flatten(2).mean(-1)
                    attention = functional.softmax(context @ context.T, dim=-1)
                    torch.cuda.synchronize()
                    if not (torch.isfinite(result).all().item() and torch.isfinite(attention).all().item()):
                        raise RuntimeError("GPU probe produced non-finite values")
                    operations.append(str(dtype))
                    del volume, weights, result, context, attention
            info.update(usable=True, reason="passed",
                        operations=["conv3d", "group_norm", "gelu", "trilinear", "matmul", "softmax"],
                        precisions=operations)
    except Exception as error:
        info.update(reason="compatibility" if is_cuda_compatibility_error(error) else "probe_failed",
                    error_type=type(error).__name__, message=str(error)[:4000])
    print(PREFIX + json.dumps(info, ensure_ascii=False), flush=True)
    return 0


def run_gpu_probe(timeout=90):
    global _last_probe
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--cuda-probe"]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--cuda-probe"]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        records = [line[len(PREFIX):] for line in process.stdout.splitlines() if line.startswith(PREFIX)]
        if process.returncode != 0 or len(records) != 1:
            result = {"usable": False, "reason": "probe_process_failed", "returncode": process.returncode,
                      "message": (process.stderr or process.stdout)[-2000:]}
        else:
            result = json.loads(records[0])
            if not isinstance(result, dict) or type(result.get("usable")) is not bool:
                raise ValueError("Invalid CUDA probe result")
    except subprocess.TimeoutExpired:
        result = {"usable": False, "reason": "probe_timeout", "message": "GPU compatibility check timed out"}
    except (OSError, ValueError, TypeError):
        result = {"usable": False, "reason": "probe_process_failed", "message": "GPU compatibility check could not run"}
    _last_probe = result
    return result


def last_gpu_probe():
    return dict(_last_probe) if _last_probe is not None else None


if __name__ == "__main__":
    raise SystemExit(probe_main())
