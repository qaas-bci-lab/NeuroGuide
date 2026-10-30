import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch
import nibabel as nib
import numpy as np
import torch
from core.runtime import estimate_brainlfnet_bytes, plan_leadfield_device, timed_stage,allocator_budget
from core.individual_pipeline import predict_brainlfnet


class RuntimeTests(unittest.TestCase):
    def test_memory_plan_uses_available_memory_and_sample_grid(self):
        candidate=torch.device('cuda:0')
        with patch('torch.cuda.empty_cache'), patch('torch.cuda.mem_get_info',return_value=(7*2**30,8*2**30)):
            small,plan=plan_leadfield_device((202,238,202),16,candidate)
            standard,standard_plan=plan_leadfield_device((202,238,202),16,candidate,prefer_low_memory=False)
            large,large_plan=plan_leadfield_device((260,532,532),16,candidate)
        self.assertEqual(small.type,'cuda'); self.assertEqual(large.type,'cpu')
        self.assertEqual(plan['inference_mode'],'cuda_low_memory_fp16')
        self.assertEqual(standard.type,'cuda')
        self.assertEqual(standard_plan['inference_mode'],'cuda_autocast_fp16')
        self.assertGreater(large_plan['estimated_peak_bytes'],8*2**30)
        with patch('torch.cuda.empty_cache'), patch('torch.cuda.mem_get_info',return_value=(1*2**30,8*2**30)):
            busy,_=plan_leadfield_device((202,238,202),16,candidate)
        self.assertEqual(busy.type,'cpu')

    def test_runtime_oom_retries_cpu_but_other_errors_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'checkpoints').mkdir(); (root/'checkpoints/args.json').write_text('{"base_channels":16}')
            t1=root/'t1.nii'; nib.save(nib.Nifti1Image(np.ones((4,4,4)),np.eye(4)),t1)
            args=(t1,t1,t1,root,root,torch.device('cuda:0'),lambda _:None)
            with patch('core.runtime.plan_leadfield_device',side_effect=[(torch.device('cuda:0'),{}),(torch.device('cpu'),{})]), \
                 patch('torch.cuda.empty_cache'), patch('core.individual_pipeline._predict_brainlfnet_once',side_effect=[torch.cuda.OutOfMemoryError('CUDA out of memory'),{'device':'cpu'}]) as predict:
                result=predict_brainlfnet(*args)
                self.assertEqual(predict.call_count,2)
                self.assertEqual(predict.call_args.args[5].type,'cpu')
                self.assertTrue(result['cuda_fallback'])
            with patch('core.runtime.plan_leadfield_device',return_value=(torch.device('cuda:0'),{})), \
                 patch('core.individual_pipeline._predict_brainlfnet_once',side_effect=RuntimeError('bad checkpoint')) as predict:
                with self.assertRaisesRegex(RuntimeError,'bad checkpoint'): predict_brainlfnet(*args)
                self.assertEqual(predict.call_count,1)

    def test_timing_reports_completion_failure_and_actual_device(self):
        events=[]
        with timed_stage('leadfield_prediction',events.append,'cuda') as record:
            record['device']='cpu'
        self.assertEqual(events[-1]['device'],'cpu')
        self.assertEqual(events[-1]['state'],'completed')
        self.assertGreaterEqual(events[-1]['seconds'],0)
        with self.assertRaises(ValueError):
            with timed_stage('field_synthesis',events.append): raise ValueError('stop')
        self.assertEqual(events[-1]['state'],'failed')

    def test_oom_closes_mmap_and_preserves_committed_gpu_column(self):
        import sys,types
        class Model(torch.nn.Module):
            instances=[]
            def __init__(self,**kwargs):
                super().__init__(); self.calls=0; self.number=len(self.instances); self.instances.append(self)
            def load_state_dict(self,*args,**kwargs): pass
            def forward(self,volume,query):
                self.calls+=1
                if self.number==0 and self.calls==2: raise torch.cuda.OutOfMemoryError('CUDA out of memory')
                return torch.ones((1,query.shape[1],3))*(1 if self.number==0 else 2)
        module=types.ModuleType('brainlfnet'); module.BrainLFNet=Model
        original_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            kwargs.pop('device',None)
            args=tuple(a for a in args if not isinstance(a,torch.device))
            return original_to(tensor,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'checkpoints').mkdir(); (root/'checkpoints/args.json').write_text('{"base_channels":16,"e_scale":1}')
            (root/'checkpoints/best_ema.pt').write_bytes(b'fixture')
            t1=root/'t1.nii'; elec=root/'elec.nii'
            nib.save(nib.Nifti1Image(np.ones((6,6,6),dtype=np.uint8),np.eye(4)),t1)
            data=np.zeros((6,6,6),dtype=np.uint8);data.ravel()[:72]=np.arange(1,73)
            nib.save(nib.Nifti1Image(data,np.eye(4)),elec)
            output=root/'prediction'
            with patch.dict(sys.modules,{'brainlfnet':module}), patch('torch.load',return_value={'model':{}}), \
                 patch('torch.Tensor.to',cpu_to), patch('torch.autocast',return_value=nullcontext()), \
                 patch('torch.cuda.empty_cache'),patch('torch.cuda.reset_peak_memory_stats'), \
                 patch('torch.cuda.max_memory_allocated',return_value=0),patch('torch.cuda.max_memory_reserved',return_value=0), \
                 patch('core.runtime.plan_leadfield_device',side_effect=[(torch.device('cuda:0'),{}),(torch.device('cpu'),{})]):
                result=predict_brainlfnet(t1,elec,t1,root,output,torch.device('cuda:0'),lambda _:None)
            array=np.load(output/'leadfield_gmwm.npy')
            self.assertTrue(np.all(array[:,0]==1)); self.assertTrue(np.all(array[:,1:]==2))
            self.assertEqual(Model.instances[1].calls,70)
            self.assertEqual(result['continued_from_device'],'cuda:0')
            self.assertTrue(result['cuda_fallback'])
            self.assertFalse((output/'leadfield_gmwm.partial.npy').exists())

    def test_low_memory_is_selected_for_five_point_nine_gib_free(self):
        with patch('torch.cuda.empty_cache'),patch('torch.cuda.mem_get_info',return_value=(int(5.9*2**30),8*2**30)):
            device,plan=plan_leadfield_device((219,290,292),16,torch.device('cuda:0'),prefer_low_memory=False)
        self.assertEqual(device.type,'cuda');self.assertEqual(plan['inference_mode'],'cuda_low_memory_fp16')
        self.assertLess(plan['estimated_peak_bytes'],plan['allocator_limit_bytes'])
        self.assertLessEqual(plan['allocator_limit_bytes'],plan['free_bytes']-plan['reserved_headroom_bytes'])

    def test_allocator_budget_is_restored_after_failure(self):
        with patch('torch.cuda.get_per_process_memory_fraction',create=True,return_value=.85), \
             patch('torch.cuda.set_per_process_memory_fraction') as setter:
            with self.assertRaises(ValueError):
                with allocator_budget({'allocator_limit_bytes':4*2**30,'total_bytes':8*2**30},torch.device('cuda:0')):
                    raise ValueError('test')
        self.assertEqual([call.args[0] for call in setter.call_args_list],[.5,.85])

    def test_resume_rejects_changed_anatomy_and_closes_mapping(self):
        import sys,types
        from core.individual_pipeline import _predict_brainlfnet_once
        class Model(torch.nn.Module):
            def __init__(self,**kwargs):super().__init__();self.calls=0
            def load_state_dict(self,*args,**kwargs):pass
            def forward(self,volume,query):
                self.calls+=1
                if self.calls==2:raise torch.cuda.OutOfMemoryError('interrupted probe')
                return torch.ones(1,query.shape[1],3)
        module=types.ModuleType('brainlfnet');module.BrainLFNet=Model
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir();(root/'checkpoints/args.json').write_text('{"base_channels":16}')
            (root/'checkpoints/best_ema.pt').write_bytes(b'fixture')
            t1=root/'t1.nii';elec=root/'elec.nii'
            tissue=np.ones((6,6,6),np.uint8);nib.save(nib.Nifti1Image(tissue,np.eye(4)),t1)
            labels=np.zeros_like(tissue);labels.ravel()[:72]=np.arange(1,73)
            nib.save(nib.Nifti1Image(labels,np.eye(4)),elec)
            output=root/'prediction';args=(t1,elec,t1,root,output,torch.device('cpu'),lambda _:None)
            with patch.dict(sys.modules,{'brainlfnet':module}),patch('torch.load',return_value={'model':{}}):
                with self.assertRaises(torch.cuda.OutOfMemoryError):_predict_brainlfnet_once(*args)
                tissue[0,0,0]=2;nib.save(nib.Nifti1Image(tissue,np.eye(4)),t1)
                with self.assertRaisesRegex(ValueError,'输入文件'):_predict_brainlfnet_once(*args)
            partial=output/'leadfield_gmwm.partial.npy';partial.rename(output/'unlocked.npy')

    def test_standard_oom_then_low_memory_oom_then_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir();(root/'checkpoints/args.json').write_text('{"base_channels":16}')
            path=root/'t1.nii';nib.save(nib.Nifti1Image(np.ones((4,4,4)),np.eye(4)),path)
            plans=[(torch.device('cuda:0'),{'inference_mode':'cuda_autocast_fp16'}),
                   (torch.device('cuda:0'),{'inference_mode':'cuda_low_memory_fp16'}),
                   (torch.device('cpu'),{'inference_mode':'float32'})]
            with patch('core.runtime.plan_leadfield_device',side_effect=plans),patch('torch.cuda.empty_cache'), \
                 patch('core.individual_pipeline._predict_brainlfnet_once',side_effect=[torch.cuda.OutOfMemoryError('CUDA out of memory'),
                     torch.cuda.OutOfMemoryError('CUDA out of memory'),{'device':'cpu'}]) as predictor:
                result=predict_brainlfnet(path,path,path,root,root,torch.device('cuda:0'),lambda _:None)
            self.assertEqual([call.kwargs['inference_mode'] for call in predictor.call_args_list],
                ['cuda_autocast_fp16','cuda_low_memory_fp16','float32'])
            self.assertTrue(result['cuda_fallback']);self.assertTrue(result['runtime_gpu_retry'])
