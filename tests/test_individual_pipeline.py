import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
from core.individual_pipeline import prepare_subject, run_pipeline, validate_preprocessing, TRAINING_SPACING_MM
from test_individual_cache import cache_fixture
from test_individual_ti import fixture


def prepared_fixture(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    shape=(8,8,8); affine=np.diag([*TRAINING_SPACING_MM,1.])
    data=np.zeros(shape,dtype=np.uint8); data.ravel()[:5]=np.arange(1,6)
    electrode=np.zeros(shape,dtype=np.uint8); electrode.ravel()[8:80]=np.arange(1,73)
    gel=np.zeros(shape,dtype=np.uint8); gel.ravel()[100:172]=np.arange(1,73)
    result={}
    for key,values in (('padded_t1',np.ones(shape)),('tissue_masks',data),('electrode_mask',electrode),('gel_mask',gel)):
        path=folder/(key+'.nii.gz'); nib.save(nib.Nifti1Image(values,affine),path); result[key]=str(path)
    for key in ('electrode_geometry','electrode_coordinates','manifest'):
        path=folder/(key+'.json'); path.write_text('{}'); result[key]=str(path)
    result.update(model_label_order='verified',landmark_method='python-affine',requires_spm_runtime=False,
                  cleanup='roast-neck',preserve_uncalibrated_intracranial=True,
                  field_of_view={'policy':'bulk_gmwm_inferior_margin_v1','margin_mm':20.0,'applied':False,'grid_policy':'native_voxel_slices_without_interpolation'})
    return result


class PipelineTests(unittest.TestCase):
    def test_segmentation_cuda_oom_retries_cpu(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            config,_,_=cache_fixture(d)
            backend=Mock(side_effect=[torch.cuda.OutOfMemoryError('CUDA out of memory'),{'inference':{'device':'cpu'}}])
            with patch('core.individual_pipeline.preprocessing_backend',return_value=backend), patch('torch.cuda.empty_cache'):
                result=prepare_subject(config['t1'],config['brain_root'],Path(d)/'pre','subject',10,torch.device('cuda:0'),lambda _:None)
            self.assertEqual(backend.call_count,2)
            self.assertEqual(backend.call_args.kwargs['device'],'cpu')
            self.assertEqual(backend.call_args.kwargs['target_spacing_mm'],TRAINING_SPACING_MM)
            self.assertEqual(result['inference']['device'],'cpu')

    def test_backend_options_and_packaged_probability_template(self):
        with tempfile.TemporaryDirectory() as d:
            config,_,_=cache_fixture(d)
            backend=Mock(return_value={'ok':True}); messages=[]
            with patch('core.individual_pipeline.preprocessing_backend',return_value=backend):
                result=prepare_subject(config['t1'],config['brain_root'],Path(d)/'pre','subject',10,'cpu',messages.append)
            self.assertEqual(result,{'ok':True})
            args=backend.call_args.kwargs
            self.assertEqual(args['model_label_order'],'verified')
            self.assertEqual(args['cleanup'],'roast-neck')
            self.assertEqual(args['inferior_margin_mm'],20.)
            self.assertEqual(args['target_spacing_mm'],TRAINING_SPACING_MM)
            self.assertEqual(args['resample_grid'],'native');self.assertTrue(args['resample_antialias'])
            self.assertEqual(args['resample_interpolation'],'linear');self.assertEqual(args['inferior_crop_grid'],'native')
            self.assertEqual(args['landmark_method'],'python-affine')
            self.assertEqual(args['probability_template'],Path(config['brain_root'])/'SegandElec/assets/eTPM.nii')
            self.assertEqual(args['probability_calibration'],Path(config['brain_root'])/'SegandElec_nnUNet/configs/extracranial_calibration.json')
            self.assertFalse(args['save_probabilities']); self.assertFalse(args['save_intermediates'])
            args['progress']('Placing the shared 72-channel 10-10 electrode array')
            self.assertIn('72 电极',messages[-1])

    def test_same_native_grid_labels_and_overlap_validation(self):
        with tempfile.TemporaryDirectory() as d:
            prepared=prepared_fixture(d); validate_preprocessing(prepared,d)
            image=nib.load(prepared['electrode_mask']); values=np.asarray(image.dataobj).copy()
            values.ravel()[0]=1
            nib.save(nib.Nifti1Image(values,image.affine),prepared['electrode_mask'])
            with self.assertRaisesRegex(ValueError,'重叠'): validate_preprocessing(prepared,d)
            changed=image.affine.copy(); changed[0,3]+=1
            nib.save(nib.Nifti1Image(np.asarray(image.dataobj),changed),prepared['electrode_mask'])
            with self.assertRaisesRegex(ValueError,'同一空间'): validate_preprocessing(prepared,d)

    def test_prediction_retry_reuses_completed_preprocessing(self):
        with tempfile.TemporaryDirectory() as d:
            config,_,_=cache_fixture(d); config['output_root']=str(Path(d)/'new_tasks')
            calls=[]
            def prepare(*args,**kwargs):
                calls.append(args)
                return prepared_fixture(args[2])
            with patch('core.individual_pipeline.select_device',return_value=type('Device',(),{'type':'cpu'})()), \
                 patch('core.individual_pipeline.prepare_subject',side_effect=prepare), \
                 patch('core.individual_pipeline.preprocessing_backend'), \
                 patch('core.individual_pipeline.predict_brainlfnet',side_effect=RuntimeError('interrupted prediction')):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'interrupted prediction'): run_pipeline(config,lambda _:None)
            self.assertEqual(len(calls),1)
            manifest=json.loads(next(Path(config['output_root']).rglob('manifest.json')).read_text())
            self.assertNotIn('prediction',manifest['stages'])
            self.assertIn('preprocessing',manifest['stages'])
            self.assertEqual(manifest['signature']['version'],5)
            self.assertEqual(manifest['signature']['grid_policy'],'training_spacing_native_directions_ras_padding')
            # A damaged completed stage must archive any old prediction and
            # rebuild the stage before resuming, rather than mix geometries.
            path=Path(calls[0][2]); (path.parent/'prediction').mkdir()
            (path.parent/'prediction/leadfield_gmwm.partial.npy').write_bytes(b'old columns')
            Path(manifest['stages']['preprocessing']['tissue_masks']).unlink()
            with patch('core.individual_pipeline.select_device',return_value=type('Device',(),{'type':'cpu'})()), \
                 patch('core.individual_pipeline.prepare_subject',side_effect=prepare), \
                 patch('core.individual_pipeline.preprocessing_backend'), \
                 patch('core.individual_pipeline.predict_brainlfnet',side_effect=RuntimeError('interrupted prediction')):
                with self.assertRaises(RuntimeError): run_pipeline(config,lambda _:None)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(list(path.parent.glob('prediction_incomplete_*'))),1)

    def test_missing_electrode_and_unverified_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            prepared=prepared_fixture(d)
            prepared['model_label_order']='metadata'
            with self.assertRaisesRegex(ValueError,'标签约定'): validate_preprocessing(prepared,d)
            prepared['model_label_order']='verified'
            image=nib.load(prepared['electrode_mask']); values=np.asarray(image.dataobj).copy();values[values==72]=0
            nib.save(nib.Nifti1Image(values,image.affine),prepared['electrode_mask'])
            with self.assertRaisesRegex(ValueError,'1..72'): validate_preprocessing(prepared,d)
