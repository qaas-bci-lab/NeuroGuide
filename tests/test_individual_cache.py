import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.individual_cache import find_reusable_task
from core.individual_pipeline import prediction_assets, file_hash, nnunet_package, preprocessing_settings
from test_individual_ti import fixture


def cache_fixture(directory):
    root = Path(directory)
    task = root/'individualized'/'old_model'; task.mkdir(parents=True)
    prediction, t1 = fixture(task)
    brain = root/'BrainLFNet'
    package = brain/'SegandElec_nnUNet/model'
    resources = [package/name for name in ('checkpoint_final.pth','plans.json','dataset.json','inference_config.json','postprocessing.json')]
    resources += [brain/name for name in ('checkpoints/best_ema.pt','checkpoints/args.json',
                  'SegandElec/assets/eTPM.nii','SegandElec/assets/capInfo.xlsx',
                  'SegandElec_nnUNet/configs/extracranial_calibration.json','brainlfnet/model.py')]
    resources += [brain/'SegandElec_nnUNet'/name for name in ('__init__.py','pipeline.py','field_of_view.py','inference.py','labels.py','landmarks.py','python_affine.py')]
    resources += [brain/'SegandElec'/name for name in ('pipeline_common.py','spatial_compat.py','roast_touchup.py','electrodes.py')]
    for p in resources:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(p.name.encode())
    electrode_files = {}
    for key in ('electrode_mask', 'electrode_geometry', 'electrode_coordinates'):
        p = task/key; p.write_bytes(b'fixture'); electrode_files[key] = str(p)
    signature = preprocessing_settings(10)
    signature.update(device='cuda', hashes={str(p):file_hash(p) for p in prediction_assets(t1,brain)})
    manifest = {'t1':str(t1), 'signature':signature, 'stages':{'padded':str(t1),
                'segmentation':str(t1), 'electrodes':electrode_files,
                'preprocessing':{'model_label_order':'verified','landmark_method':'python-affine','requires_spm_runtime':False,'cleanup':'roast-neck','preserve_uncalibrated_intracranial':True}, 'prediction':{'complete':True}}}
    path = task/'manifest.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
    config = {'t1':str(t1), 'brain_root':str(brain), 'output_root':str(root/'individualized'), 'padding':10}
    return config, path, manifest


class CacheTests(unittest.TestCase):
    def test_device_check_only_update_reuses_completed_model_but_framework_change_does_not(self):
        from core.cache_contract import PREVIOUS_DEVICE_CHECK_ADAPTER
        with tempfile.TemporaryDirectory() as directory:
            config,path,manifest=cache_fixture(directory)
            manifest['signature']['adapter_sha256']=PREVIOUS_DEVICE_CHECK_ADAPTER
            path.write_text(json.dumps(manifest))
            self.assertIsNotNone(find_reusable_task(config)['task'])
            manifest['signature']['torch_version']='different_framework'
            path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_previous_grid_recipes_require_rebuilding_for_inferior_crop(self):
        import nibabel as nib
        import numpy as np
        from core.cache_contract import PREVIOUS_ADAPTER
        with tempfile.TemporaryDirectory() as d:
            config,path,manifest=cache_fixture(d)
            signature=manifest['signature']
            signature.update(version=2,grid_policy='native_ras_padding',adapter_sha256=PREVIOUS_ADAPTER)
            signature.pop('target_spacing_mm');signature.pop('resampling_order')
            path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])
            source=Path(config['t1']);image=nib.load(source)
            affine=image.affine.copy();affine[:3,:3]*=.5
            nib.save(nib.Nifti1Image(np.asarray(image.dataobj),affine),source)
            signature['hashes'][str(source)]=file_hash(source)
            path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_matching_model_and_moved_mri(self):
        with tempfile.TemporaryDirectory() as d:
            config, path, _ = cache_fixture(d)
            relocated = Path(d)/'relocated.nii.gz'; shutil.copyfile(config['t1'], relocated)
            config['t1'] = str(relocated)
            task = find_reusable_task(config)['task']
            self.assertEqual(task['manifest'], str(path))
            self.assertEqual(task['t1'], str(relocated))
            self.assertEqual(task['prediction_dir'], str(path.parent/'prediction'))

    def test_other_mri_or_changed_model_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            config, _, _ = cache_fixture(d)
            source = Path(config['t1']); other = Path(d)/source.name; other.write_bytes(b'another MRI')
            self.assertIsNone(find_reusable_task(dict(config, t1=str(other)))['task'])
            (Path(config['brain_root'])/'checkpoints/best_ema.pt').write_bytes(b'new model')
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_incompatible_parameters_and_missing_prediction_are_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            config, path, manifest = cache_fixture(d)
            self.assertIsNone(find_reusable_task(dict(config, padding=12))['task'])
            manifest['stages']['prediction'] = None; path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_corrupt_leadfield_and_grid_mismatch_are_not_reused(self):
        import nibabel as nib
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            config, path, manifest = cache_fixture(d)
            lf = path.parent/'prediction/leadfield_gmwm.npy'; original = lf.read_bytes()
            lf.write_bytes(b'truncated')
            self.assertIsNone(find_reusable_task(config)['task'])
            lf.write_bytes(original)
            tissue = path.parent/'wrong_grid.nii'; nib.save(nib.Nifti1Image(np.ones((2,2,2)), np.eye(4)), tissue)
            manifest['stages']['segmentation'] = str(tissue); path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_old_backend_and_changed_recipe_are_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            config,path,manifest=cache_fixture(d)
            manifest['signature']['version']=1
            manifest['signature']['landmark_method']='legacy-registration'
            path.write_text(json.dumps(manifest))
            self.assertIsNone(find_reusable_task(config)['task'])
            config,path,manifest=None,None,None
        with tempfile.TemporaryDirectory() as d:
            config,path,manifest=cache_fixture(d)
            (Path(config['brain_root'])/'SegandElec_nnUNet/configs/extracranial_calibration.json').write_bytes(b'new calibration')
            self.assertIsNone(find_reusable_task(config)['task'])

    def test_moved_installation_signature_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            config, path, manifest = cache_fixture(d)
            original_root = str(Path(config['brain_root']))
            manifest['signature']['hashes'] = {p.replace(original_root, 'Z:\\old\\BrainLFNet'):h
                                               for p,h in manifest['signature']['hashes'].items()}
            path.write_text(json.dumps(manifest))
            self.assertIsNotNone(find_reusable_task(config)['task'])
