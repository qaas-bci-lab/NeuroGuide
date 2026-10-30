"""CUDA compatibility checks must run kernels outside the inference process."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from core.cuda_compat import PREFIX, run_gpu_probe, is_cuda_compatibility_error
from core.individual_pipeline import select_device, prepare_subject, predict_brainlfnet


class CudaCompatibilityTests(unittest.TestCase):
    def test_probe_parses_result_and_uses_disposable_process_without_shell(self):
        payload={"usable":True,"gpu_name":"test","capability":[12,0]}
        completed=subprocess.CompletedProcess([],0,"noise\n"+PREFIX+json.dumps(payload)+"\n","")
        with patch("core.cuda_compat.subprocess.run",return_value=completed) as run:
            self.assertEqual(run_gpu_probe(),payload)
            self.assertNotIn("shell",run.call_args.kwargs)
            self.assertIn("--cuda-probe",run.call_args.args[0])

    def test_probe_timeout_crash_and_malformed_result_fall_back(self):
        cases=[subprocess.TimeoutExpired(["probe"],90),
               subprocess.CompletedProcess([],17,"","driver crash"),
               subprocess.CompletedProcess([],0,PREFIX+'{"usable":"yes"}',"")]
        for case in cases:
            with self.subTest(case=case):
                kwargs={"side_effect":case} if isinstance(case,Exception) else {"return_value":case}
                with patch("core.cuda_compat.subprocess.run",**kwargs):
                    self.assertFalse(run_gpu_probe()["usable"])

    def test_compatibility_auto_fallback_strict_cuda_and_explicit_cpu(self):
        logs=[]
        with patch("core.cuda_compat.run_gpu_probe",return_value={"usable":False,"reason":"compatibility","gpu_name":"RTX 5060","message":"no kernel image"}):
            self.assertEqual(select_device("auto",logs.append).type,"cpu")
            with self.assertRaisesRegex(RuntimeError,"GPU"):select_device("cuda",logs.append)
        self.assertTrue(any("CPU" in message for message in logs))
        with patch("core.cuda_compat.run_gpu_probe",side_effect=AssertionError("CPU must not access CUDA")):
            self.assertEqual(select_device("cpu",logs.append).type,"cpu")

    def test_error_classification_does_not_hide_checkpoint_or_memory_errors(self):
        for message in ("CUDA error: no kernel image is available for execution on the device",
                        "CUDA error: invalid device function","CUDA driver version is insufficient",
                        "The provided PTX was compiled with an unsupported toolchain",
                        "cuDNN error: CUDNN_STATUS_ARCH_MISMATCH",
                        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
                        "GET was unable to find an engine to execute this computation"):
            self.assertTrue(is_cuda_compatibility_error(RuntimeError(message)))
        for message in ("bad checkpoint","size mismatch","index out of bounds","CUDA out of memory"):
            self.assertFalse(is_cuda_compatibility_error(RuntimeError(message)))

    def test_segmentation_runtime_kernel_failure_retries_cpu_once(self):
        logs=[]
        with patch("core.individual_pipeline.nnunet_package",return_value=(Path("model"),Path("weights"))), \
             patch("core.individual_pipeline.preprocessing_backend") as backend,patch("torch.cuda.empty_cache"):
            backend.return_value.side_effect=[RuntimeError("CUDA error: no kernel image is available"),{"done":True}]
            result=prepare_subject("t1","brain","output","subject",10,torch.device("cuda:0"),logs.append)
            self.assertTrue(result["done"])
            self.assertEqual([call.kwargs["device"] for call in backend.return_value.call_args_list],["cuda:0","cpu"])
        self.assertTrue(any("不兼容" in message for message in logs))

    def test_leadfield_runtime_kernel_failure_skips_low_memory_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"checkpoints").mkdir();(root/"checkpoints/args.json").write_text('{"base_channels":16}')
            path=root/"T1.nii";nib.save(nib.Nifti1Image(np.ones((4,4,4)),np.eye(4)),path)
            plans=[(torch.device("cuda:0"),{"inference_mode":"cuda_autocast_fp16"}),
                   (torch.device("cpu"),{"inference_mode":"float32"})]
            with patch("core.runtime.plan_leadfield_device",side_effect=plans),patch("torch.cuda.empty_cache"), \
                 patch("core.individual_pipeline._predict_brainlfnet_once",side_effect=[
                     RuntimeError("CUDA error: no kernel image is available"),{"device":"cpu"}]) as predict:
                result=predict_brainlfnet(path,path,path,root,root,torch.device("cuda:0"),lambda _:None)
            self.assertEqual([call.args[5].type for call in predict.call_args_list],["cuda","cpu"])
            self.assertEqual(result["inference_attempts"][0]["outcome"],"cuda_compatibility_error")
            self.assertTrue(result["memory_plan"]["runtime_compatibility_fallback"])
            self.assertTrue(result["cuda_fallback"])
