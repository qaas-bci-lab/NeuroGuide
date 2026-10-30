import unittest
import numpy as np
import nibabel as nib
from ui.volume_geometry import slice_plane_coordinates,volume_world_bounds,slice_plane_pose
from core.atlas import nifti_to_ants,ants_on_reference
from core.ti import Leadfield

class NativeGeometryTests(unittest.TestCase):
    def affine(self):
        affine=np.eye(4);affine[:3,:3]=[[1.2,.003,-.001],[.004,.929,.124],[.0005,.124,.929]]
        affine[:3,3]=[-30,10,-60]
        return affine

    def test_plane_points_bounds_normals_and_drag_inverse_agree(self):
        affine=self.affine();shape=(11,13,17)
        expected_corners=nib.affines.apply_affine(affine,np.array([
            [0,0,0],[10,0,0],[0,12,0],[0,0,16],[10,12,0],[10,0,16],[0,12,16],[10,12,16]]))
        bounds=np.asarray(volume_world_bounds(shape,affine)).reshape(3,2)
        np.testing.assert_allclose(bounds[:,0],expected_corners.min(axis=0))
        np.testing.assert_allclose(bounds[:,1],expected_corners.max(axis=0))
        for axis in range(3):
            xyz=np.stack(slice_plane_coordinates(shape,affine,axis,4,2),axis=-1)
            ijk=nib.affines.apply_affine(np.linalg.inv(affine),xyz)
            np.testing.assert_allclose(ijk[...,axis],4,atol=1e-10)
            origin,normal=slice_plane_pose(shape,affine,axis,4)
            np.testing.assert_allclose((xyz-np.array(origin))@normal,0,atol=1e-10)
            self.assertAlmostEqual(nib.affines.apply_affine(np.linalg.inv(affine),origin)[axis],4)
            # Along X, physical distance is 1.2 mm, not one display unit.
            other=[value for value in range(3) if value!=axis]
            np.testing.assert_allclose(np.linalg.norm(xyz[1,0]-xyz[0,0]),2*np.linalg.norm(affine[:3,other[0]]))

    def test_ants_native_geometry_roundtrip_retains_shear_and_ras_lps_coordinates(self):
        affine=self.affine();data=np.arange(7*8*9,dtype=np.float32).reshape(7,8,9)
        reference=nib.Nifti1Image(data,affine)
        actual=nifti_to_ants(reference)
        np.testing.assert_array_equal(actual.numpy(),data)
        point=np.array([3,4,5])
        actual_lps=np.array(actual.origin)+actual.direction@(point*np.array(actual.spacing))
        np.testing.assert_allclose(actual_lps,nib.affines.apply_affine(affine,point)*[-1,-1,1])
        restored=ants_on_reference(actual,reference)
        np.testing.assert_allclose(restored.affine,affine,atol=1e-5)
        np.testing.assert_array_equal(np.asarray(restored.dataobj),data)
        actual.set_origin(tuple(np.array(actual.origin)+[1,0,0]))
        with self.assertRaisesRegex(ValueError,"physical grid"):
            ants_on_reference(actual,reference)

    def test_ants_identity_interpolation_preserves_nonorthogonal_grid(self):
        import ants
        affine=self.affine();data=np.ones((9,10,11),np.float32);data[2:6,3:7,4:8]=41
        reference=nib.Nifti1Image(data,affine)
        original=nifti_to_ants(reference)
        actual=ants.resample_image_to_target(original,original,interp_type="nearestNeighbor")
        np.testing.assert_array_equal(ants_on_reference(actual,reference).get_fdata(),data)

    def test_roi_radius_is_physical_mm_on_nonorthogonal_anisotropic_grid(self):
        lf=Leadfield.__new__(Leadfield)
        lf.affine=self.affine();lf.shape=(11,13,17)
        lf.ijk=np.indices(lf.shape).reshape(3,-1).T
        center=nib.affines.apply_affine(lf.affine,[5,6,8])
        selected=lf.roi({"center_ras_mm":center.tolist(),"radius_mm":3.})
        points=nib.affines.apply_affine(lf.affine,lf.ijk)
        np.testing.assert_array_equal(selected,np.linalg.norm(points-center,axis=1)<=3.)
        self.assertFalse(selected[np.flatnonzero(np.all(lf.ijk==[8,6,8],axis=1))[0]])
        self.assertTrue(selected[np.flatnonzero(np.all(lf.ijk==[7,6,8],axis=1))[0]])


    def test_atlas_working_grid_is_world_aligned_but_native_reference_is_unchanged(self):
        from core.atlas import atlas_working_grid
        affine=self.affine();data=np.ones((15,16,17),np.float32)
        reference=nib.Nifti1Image(data,affine)
        before=reference.affine.copy()
        mask=np.ones(reference.shape,np.uint8)
        fixed,working_mask=atlas_working_grid(reference,mask)
        np.testing.assert_allclose(fixed.direction,np.diag([-1.,-1.,1.]))
        np.testing.assert_allclose(fixed.spacing,[2,2,2])
        self.assertEqual(fixed.shape,working_mask.shape)
        np.testing.assert_allclose(fixed.origin,working_mask.origin)
        self.assertTrue(set(np.unique(working_mask.numpy())).issubset({0,1}))
        np.testing.assert_allclose(reference.affine,before)
        np.testing.assert_array_equal(np.asarray(reference.dataobj),data)



    def test_subject_to_mni_initialization_ras_lps_conversion(self):
        from core.atlas import subject_to_mni_lps
        matrix=np.eye(4);matrix[:3,:3]=[[1.1,.03,.01],[-.02,.9,.2],[.01,-.3,1.2]]
        matrix[:3,3]=[-1,-20,-25]
        linear,offset=subject_to_mni_lps(matrix)
        point=np.array([4,20,35.])
        np.testing.assert_allclose(linear@(point*[-1,-1,1])+offset,
                                   nib.affines.apply_affine(matrix,point)*[-1,-1,1])

    def test_initialization_must_belong_to_current_mri(self):
        import tempfile,json
        from pathlib import Path
        from core.atlas import preprocessing_affine
        with tempfile.TemporaryDirectory() as folder:
            task=Path(folder);pre=task/"preprocessing";pre.mkdir()
            reference=pre/"input.nii.gz"
            (task/"manifest.json").write_text(json.dumps({"stages":{"preprocessing":{"subject":"case"}}}))
            path=pre/"case_landmark_affine.json"
            record={"subject":str(reference),"Affine":np.eye(4).tolist()}
            path.write_text(json.dumps(record))
            source,matrix=preprocessing_affine(task,reference)
            self.assertEqual(source,path);np.testing.assert_array_equal(matrix,np.eye(4))
            record["subject"]=str(pre/"other.nii.gz");path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"MRI"):
                preprocessing_affine(task,reference)


if __name__=="__main__":unittest.main()
