"""Training-grid regression tests: spacing order, native directions and geometry."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from core.cache_contract import signatures_match
from core.individual_pipeline import TRAINING_SPACING_MM, preprocessing_settings, validate_preprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"BrainLFNet"))
from SegandElec_nnUNet.field_of_view import (
    DEFAULT_TARGET_SPACING_MM, normalize_target_spacing, prepare_input_grid, crop_inferior_fov)
from test_individual_pipeline import prepared_fixture

def oblique_basis(spacing):
    angle=.15
    rotation=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
    return rotation@np.diag(spacing)

class TrainingGridTests(unittest.TestCase):
    def test_spacing_contract_and_invalid_values(self):
        self.assertEqual(TRAINING_SPACING_MM,DEFAULT_TARGET_SPACING_MM)
        self.assertEqual(normalize_target_spacing(1.),(1.,1.,1.))
        for invalid in ([1,2], [1,2,3,4], [0,1,1], [-1,1,1], [np.nan,1,1], [[1,1,1]]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                normalize_target_spacing(invalid)

    def test_matching_oblique_psr_input_skips_all_interpolation_and_keeps_file(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";output=Path(d)/"resampled.nii.gz"
            canonical=oblique_basis(TRAINING_SPACING_MM)
            # Storage axes P,S,R: anisotropic spacing belongs to the THIRD axis.
            affine=np.eye(4);affine[:3,:3]=canonical[:,[1,2,0]]*[-1,1,1]
            affine[:3,3]=[13,-19,22]
            data=np.arange(7*8*9,dtype=np.float32).reshape((7,8,9))
            nib.save(nib.Nifti1Image(data,affine),source)
            before=source.read_bytes()
            with patch("nibabel.processing.resample_from_to",side_effect=AssertionError("must skip")),patch(
                "scipy.ndimage.gaussian_filter",side_effect=AssertionError("must skip")):
                path,resampled,spacing=prepare_input_grid(source,output,TRAINING_SPACING_MM)
            self.assertEqual(path,source);self.assertIsNone(resampled);self.assertFalse(output.exists())
            self.assertEqual(before,source.read_bytes())
            np.testing.assert_allclose(spacing,[.9375,.9375,1.2],atol=1e-5)
            self.assertEqual(nib.aff2axcodes(nib.load(source).affine),("P","S","R"))

    def test_matching_tolerance_accepts_header_roundoff_not_different_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";out=Path(d)/"out.nii.gz"
            for first,should_skip in [(1.2000033855,True),(1.21,False)]:
                affine=np.diag([first,.9375,.9375,1])
                nib.save(nib.Nifti1Image(np.ones((8,9,10)),affine),source)
                _,resampled,_=prepare_input_grid(source,out,TRAINING_SPACING_MM)
                self.assertEqual(resampled is None,should_skip)

    def test_same_numbers_in_wrong_ras_axis_order_are_resampled(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";out=Path(d)/"out.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((8,9,10)),np.diag([.9375,.9375,1.2,1])),source)
            _,resampled,_=prepare_input_grid(source,out,TRAINING_SPACING_MM)
            self.assertEqual(resampled,str(out))
            np.testing.assert_allclose(nib.affines.voxel_sizes(nib.load(out).affine),TRAINING_SPACING_MM,atol=1e-6)

    def test_half_mm_native_resampling_preserves_direction_and_physical_center(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";out=Path(d)/"out.nii.gz"
            affine=np.eye(4);affine[:3,:3]=oblique_basis([.5,.5,.5]);affine[:3,3]=[-33,11,-17]
            # Also exercise reordering and flipping instead of assuming storage RAS.
            canonical=nib.Nifti1Image(np.full((23,27,31),37,dtype=np.float32),affine)
            oriented=canonical.as_reoriented(np.array([[1,-1],[2,1],[0,1]]))
            nib.save(oriented,source)
            path,resampled,_=prepare_input_grid(source,out,TRAINING_SPACING_MM)
            self.assertEqual(path,out);self.assertEqual(resampled,str(out))
            actual=nib.load(out);original=nib.as_closest_canonical(nib.load(source))
            old_spacing=nib.affines.voxel_sizes(original.affine);new_spacing=nib.affines.voxel_sizes(actual.affine)
            np.testing.assert_allclose(new_spacing,TRAINING_SPACING_MM,atol=1e-6)
            np.testing.assert_allclose(actual.affine[:3,:3]/new_spacing,original.affine[:3,:3]/old_spacing,atol=1e-6)
            np.testing.assert_allclose(
                nib.affines.apply_affine(actual.affine,(np.array(actual.shape)-1)/2),
                nib.affines.apply_affine(original.affine,(np.array(original.shape)-1)/2),atol=1e-5)
            self.assertEqual(actual.shape,(10,15,17))
            np.testing.assert_allclose(actual.get_fdata(),37,atol=1e-5)

    def test_antialias_only_downsampled_axes_with_expected_sigma(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";out=Path(d)/"out.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((10,11,12),np.float32),np.diag([.5,1.5,.5,1])),source)
            with patch("scipy.ndimage.gaussian_filter",wraps=gaussian_filter) as filter:
                prepare_input_grid(source,out,TRAINING_SPACING_MM)
            filter.assert_called_once()
            np.testing.assert_allclose(filter.call_args.kwargs["sigma"],[.7,0,.4375])

    def test_antialias_reduces_checkerboard_aliasing(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";aa=Path(d)/"aa.nii";plain=Path(d)/"plain.nii"
            values=(np.indices((41,43,45)).sum(axis=0)%2).astype(np.float32)
            nib.save(nib.Nifti1Image(values,np.diag([.5,.5,.5,1])),source)
            prepare_input_grid(source,aa,TRAINING_SPACING_MM)
            prepare_input_grid(source,plain,TRAINING_SPACING_MM,anti_alias=False)
            smooth=nib.load(aa).get_fdata()[2:-2,2:-2,2:-2]
            aliased=nib.load(plain).get_fdata()[2:-2,2:-2,2:-2]
            self.assertLess(smooth.std(),aliased.std()*.5)
            self.assertAlmostEqual(smooth.mean(),.5,places=2)

    def test_upsampling_does_not_smooth(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"input.nii.gz";out=Path(d)/"out.nii"
            nib.save(nib.Nifti1Image(np.ones((5,6,7)),np.diag([2,2,2,1])),source)
            with patch("scipy.ndimage.gaussian_filter",side_effect=AssertionError("upsampling must not smooth")):
                prepare_input_grid(source,out,TRAINING_SPACING_MM)

    def test_oblique_crop_preserves_basis_data_and_world_landmarks(self):
        affine=np.eye(4);affine[:3,:3]=oblique_basis(TRAINING_SPACING_MM);affine[:3,3]=[-20,8,-90]
        labels=np.zeros((16,18,130),np.uint8);labels[2:14,3:15,80:110]=2
        values=np.arange(labels.size,dtype=np.float32).reshape(labels.shape)
        reference=nib.Nifti1Image(values,affine)
        image,tissue,record=crop_inferior_fov(reference,labels,10,20.,grid_policy="native")
        # ceil(20/.9375)=22 voxel slices below the bulk, then ten zero-padding slices.
        self.assertEqual(record["crop_start_z"],58);self.assertEqual(record["offset_voxels"],[0,0,48])
        self.assertAlmostEqual(record["removed_inferior_mm"],45.)
        self.assertEqual(record["grid_policy"],"native_voxel_slices_without_interpolation")
        np.testing.assert_allclose(image.affine[:3,:3],affine[:3,:3])
        np.testing.assert_allclose(nib.affines.apply_affine(image.affine,[4,5,32]),
                                   nib.affines.apply_affine(affine,[4,5,80]),atol=1e-5)
        np.testing.assert_array_equal(np.asarray(image.dataobj)[:,:,10:],values[:,:,58:])
        self.assertEqual(np.count_nonzero(tissue==2),np.count_nonzero(labels==2))

    def test_oblique_training_grid_is_validated_by_column_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            prepared=prepared_fixture(d)
            for key in ("padded_t1","tissue_masks","electrode_mask","gel_mask"):
                image=nib.load(prepared[key]);affine=image.affine.copy();affine[:3,:3]=oblique_basis(TRAINING_SPACING_MM)
                nib.save(nib.Nifti1Image(np.asarray(image.dataobj),affine),prepared[key])
            validate_preprocessing(prepared,d)
            path=prepared["padded_t1"];image=nib.load(path)
            nib.save(nib.Nifti1Image(np.asarray(image.dataobj),np.eye(4)),path)
            with self.assertRaisesRegex(ValueError,"1.2"):
                validate_preprocessing(prepared,d)

    def test_v4_one_mm_recipe_and_axis_aligned_recipe_are_not_reused(self):
        expected=preprocessing_settings()
        self.assertEqual(expected["version"],5)
        old=dict(expected,version=4,grid_policy="resampled_1mm_ras_padding",target_spacing_mm=1.)
        self.assertFalse(signatures_match(old,expected,allow_native_1mm=True))
        wrong_policy=dict(expected,resample_grid="world")
        self.assertFalse(signatures_match(wrong_policy,expected))
        no_filter=dict(expected,resample_antialias=False)
        self.assertFalse(signatures_match(no_filter,expected))
        self.assertTrue(signatures_match(expected,expected))

if __name__=="__main__":unittest.main()
