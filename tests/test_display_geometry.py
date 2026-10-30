"""Checks for a single scan-native display and camera-only pose."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation
from core.display_geometry import DisplayGrid, report_display_engine
from core.engine import NiiEngine
from ui.volume_geometry import volume_camera_pose, slice_plane_coordinates
from tools.report_page import generate_report, slice_image


def phantom():
    shape=(19,23,27)
    affine=np.eye(4)
    affine[:3,:3]=Rotation.from_euler("x",3,degrees=True).as_matrix() @ np.diag([1.2,.9375,.9375])
    affine[:3,3]=[-11,-12,-13]
    coords=np.indices(shape)
    data=(100*np.exp(-sum(((coords[i]-(shape[i]-1)/2)/6)**2 for i in range(3)))).astype(np.float32)
    return data,affine


class DisplayTests(unittest.TestCase):
    def test_oblique_scan_arrays_affine_and_registration_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,affine=phantom();path=root/"T1.nii"
            nib.save(nib.Nifti1Image(data,affine),path)
            engine=NiiEngine()
            with patch("core.atlas.preprocessing_affine",side_effect=AssertionError("Display must not use registration")) as read:
                self.assertTrue(engine.load_image(path)[0])
                read.assert_not_called()
            self.assertEqual(engine.view.mode,"original")
            self.assertIs(engine.view.data,engine.data)
            np.testing.assert_array_equal(engine.view.affine,engine.affine)
            np.testing.assert_array_equal(engine.view.native_to_view,np.eye(4))

    def test_sheared_native_coordinate_roundtrip_and_exact_voxel_mapping(self):
        from itertools import product
        data,affine=phantom();affine[:3,1]+=affine[:3,2]*.2
        before=affine.copy();view=DisplayGrid(data,affine)
        for point in product(*[(0,size-1) for size in data.shape]):
            ras=(affine@np.r_[point,1])[:3]
            voxel=view.native_ras_to_voxel(ras)
            np.testing.assert_allclose(voxel,point,atol=1e-11)
            np.testing.assert_allclose(view.voxel_to_native_ras(voxel),ras,atol=1e-11)
            np.testing.assert_array_equal(view.native_voxel_to_view(point),point)
        self.assertIs(view.data,data);np.testing.assert_array_equal(affine,before)

    def test_fields_labels_and_replaced_roi_are_used_without_interpolation_or_stale_cache(self):
        data,affine=phantom();field=np.zeros(data.shape,np.float32);field[7:12,9:14,11:16]=.8
        labels=np.zeros(data.shape,np.float32);labels[7:12,9:14,11:16]=41
        view=DisplayGrid(data,affine)
        target={"data":labels,"is_atlas":True};scalar={"data":field,"is_atlas":False}
        self.assertIs(view.overlay_data(target),labels)
        self.assertIs(view.overlay_data(scalar),field)
        self.assertEqual(set(np.unique(view.overlay_data(target))),{0.,41.})
        target["data"]=np.zeros(data.shape,np.float32)
        self.assertIs(view.overlay_data(target),target["data"])
        self.assertFalse(view.overlay_data(target).any())
        self.assertAlmostEqual(float(view.overlay_data(scalar).max()),.8,places=6)

    def test_camera_pose_tracks_scan_directions_without_changing_slice_geometry(self):
        data,affine=phantom()
        affine[:3,1]+=affine[:3,2]*.2
        before=affine.copy()
        pose=np.asarray(volume_camera_pose(data.shape,affine))
        center=(affine@np.r_[(np.asarray(data.shape)-1)/2,1])[:3]
        np.testing.assert_allclose(pose[1],center,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(pose[2]),1.)
        self.assertGreater(np.dot(pose[2],affine[:3,2]/np.linalg.norm(affine[:3,2])),.99)
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler("xyz",[12,-5,7],degrees=True).as_matrix()
        transform[:3,3]=[50,-40,20]
        moved=np.asarray(volume_camera_pose(data.shape,transform@affine))
        np.testing.assert_allclose(moved[1],(transform@np.r_[pose[1],1])[:3],atol=1e-11)
        np.testing.assert_allclose(moved[2],transform[:3,:3]@pose[2],atol=1e-11)
        direction=(pose[0]-pose[1])/np.linalg.norm(pose[0]-pose[1])
        moved_direction=(moved[0]-moved[1])/np.linalg.norm(moved[0]-moved[1])
        np.testing.assert_allclose(moved_direction,transform[:3,:3]@direction,atol=1e-11)
        np.testing.assert_array_equal(affine,before)
        points=np.stack(slice_plane_coordinates(data.shape,affine,0,9),axis=-1)
        ijk=nib.affines.apply_affine(np.linalg.inv(affine),points)
        np.testing.assert_allclose(ijk[...,0],9,atol=1e-12)

    def test_report_keeps_scan_planes_and_native_roi_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,affine=phantom();path=root/"T1.nii.gz"
            nib.save(nib.Nifti1Image(data,affine),path)
            engine=NiiEngine();self.assertTrue(engine.load_image(path)[0])
            field=root/"field.nii.gz";nib.save(nib.Nifti1Image(data/100,affine),field)
            engine.load_overlay(field,kind="field")
            center=affine@np.r_[9,11,13,1]
            stimulation={"roi":{"center_ras_mm":center[:3].tolist(),"radius_mm":3},"statistics":{"roi":{"mean_v_m":.23,"max_v_m":.89}}}
            before=engine.overlays[0]["data"].copy()
            with patch("tools.report_page.slice_image",wraps=slice_image) as capture:
                success,error=generate_report(engine,str(path),"","ROI",str(root/"report.pdf"),patient_info={"stimulation":stimulation})
            self.assertTrue(success,error)
            adapter=report_display_engine(engine)
            self.assertIs(adapter.data,engine.data)
            np.testing.assert_array_equal(adapter.affine,engine.affine)
            for call in capture.call_args_list:
                np.testing.assert_array_equal(call.args[0].affine,engine.affine)
                self.assertEqual(call.args[2],(9,11,13))
            np.testing.assert_array_equal(engine.overlays[0]["data"],before)
            import pymupdf
            with pymupdf.open(root/"report.pdf") as pdf:
                self.assertEqual(len(pdf),1);self.assertEqual(len(pdf[0].get_images()),3)
                self.assertIn("0.2300",pdf[0].get_text());self.assertIn("0.8900",pdf[0].get_text())
