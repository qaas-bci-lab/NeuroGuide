import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from core.cache_contract import signatures_match,PREVIOUS_ADAPTER
from core.individual_pipeline import preprocessing_settings
from core.runtime import is_one_mm_image

brain=Path(__file__).resolve().parents[1]/'BrainLFNet'
sys.path.insert(0,str(brain))
from SegandElec_nnUNet.pipeline import prepare_input_grid


class ResamplingTests(unittest.TestCase):
    def test_half_mm_mri_is_resampled_before_padding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input.nii.gz'; output=root/'one_mm.nii.gz'
            affine=np.diag([.5,.5,.8,1.]);affine[:3,3]=[-3,-4,-5]
            data=np.full((15,17,11),37,dtype=np.float32)
            nib.save(nib.Nifti1Image(data,affine),source)
            path,resampled,spacing=prepare_input_grid(source,output,1.,grid_policy="world")
            self.assertEqual(path,output);self.assertEqual(resampled,str(output))
            self.assertTrue(is_one_mm_image(output));self.assertFalse(is_one_mm_image(source))
            image=nib.load(output)
            np.testing.assert_allclose(image.header.get_zooms(),[1,1,1])
            np.testing.assert_allclose(image.affine[:3,3],affine[:3,3])
            self.assertLess(np.prod(image.shape),np.prod(data.shape))
            self.assertAlmostEqual(float(image.get_fdata()[3,4,4]),37)
            np.testing.assert_allclose(spacing,[.5,.5,.8],atol=1e-6)

    def test_one_mm_mri_skips_interpolation_and_standalone_native_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input.nii'; output=root/'one_mm.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((5,6,7)),np.eye(4)),source)
            path,resampled,_=prepare_input_grid(source,output,1.,grid_policy="world")
            self.assertEqual(path,source);self.assertIsNone(resampled);self.assertFalse(output.exists())
            path,resampled,_=prepare_input_grid(source,output,None)
            self.assertEqual(path,source);self.assertIsNone(resampled)

    def test_nonorthogonal_grid_is_resampled(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input.nii'; output=root/'one_mm.nii.gz'
            affine=np.eye(4);affine[0,1]=.2
            nib.save(nib.Nifti1Image(np.ones((5,6,7)),affine),source)
            path,_,_=prepare_input_grid(source,output,1.,grid_policy="world")
            self.assertTrue(is_one_mm_image(path))

    def test_old_native_caches_require_one_mm_input_and_known_adapter(self):
        expected=preprocessing_settings(10)
        old=dict(expected,version=2,grid_policy='native_ras_padding',adapter_sha256=PREVIOUS_ADAPTER)
        old.pop('target_spacing_mm');old.pop('resampling_order')
        self.assertFalse(signatures_match(old,expected,allow_native_1mm=False))
        self.assertFalse(signatures_match(old,expected,allow_native_1mm=True))
        old['adapter_sha256']='unknown change'
        self.assertFalse(signatures_match(old,expected,allow_native_1mm=True))

    def test_oblique_one_mm_mri_gets_axis_aligned_before_legacy_padding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input.nii'; output=root/'one_mm.nii.gz'
            affine=np.eye(4); theta=.15
            affine[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
            nib.save(nib.Nifti1Image(np.ones((5,6,7)),affine),source)
            self.assertFalse(is_one_mm_image(source))
            path,_,_=prepare_input_grid(source,output,1.,grid_policy="world")
            self.assertEqual(path,output);self.assertTrue(is_one_mm_image(path))
