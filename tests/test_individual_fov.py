import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'BrainLFNet'))
from SegandElec_nnUNet.pipeline import crop_inferior_fov,run_nnunet_pipeline
from core.individual_pipeline import preprocessing_settings
from core.cache_contract import signatures_match


class FieldOfViewTests(unittest.TestCase):
    def test_spinal_tail_is_removed_but_bulk_and_twenty_mm_margin_are_preserved(self):
        labels=np.zeros((100,100,160),np.uint8)
        labels[10:90,10:90,80:140]=2
        labels[49:51,49:51,10:80]=1
        image=nib.Nifti1Image(np.ones(labels.shape,np.float32),np.eye(4))
        cropped,tissue,record=crop_inferior_fov(image,labels,10,20.)
        self.assertTrue(record['applied']);self.assertFalse(record['guarded_fallback'])
        self.assertEqual(record['removed_inferior_mm'],50.)
        self.assertEqual(record['removed_inferior_gmwm_voxels'],200)
        self.assertEqual(np.count_nonzero(tissue==2),np.count_nonzero(labels==2))
        np.testing.assert_allclose(nib.affines.apply_affine(cropped.affine,[50,50,30]),[50,50,80])

    def test_large_inferior_label_mass_is_kept_when_bulk_cut_is_ambiguous(self):
        labels=np.zeros((100,100,180),np.uint8)
        labels[10:90,10:90,150:165]=2
        labels[47:53,47:53,10:150]=1
        image=nib.Nifti1Image(np.ones(labels.shape,np.float32),np.eye(4))
        unchanged,tissue,record=crop_inferior_fov(image,labels,10,20.)
        self.assertTrue(record['guarded_fallback']);self.assertFalse(record['applied'])
        self.assertIs(unchanged,image);np.testing.assert_array_equal(tissue,labels)

    def test_different_neck_lengths_produce_same_head_grid_and_world_coordinates(self):
        outputs=[]
        for extra in [0,31,70]:
            shape=(16,18,80+extra); values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            labels=np.zeros(shape,np.uint8);labels[2:14,3:15,30+extra:60+extra]=2
            labels[3:13,4:14,33+extra:57+extra]=1
            affine=np.diag([1.,1.,1.,1.]);affine[:3,3]=[-10,-11,-30-extra]
            image=nib.Nifti1Image(values,affine)
            cropped,tissue,record=crop_inferior_fov(image,labels,10,20.)
            self.assertEqual(tissue.shape,(16,18,80))
            np.testing.assert_allclose(cropped.affine[:3,3],[-10,-11,-30])
            self.assertEqual(np.count_nonzero(np.isin(tissue,[1,2])),np.count_nonzero(np.isin(labels,[1,2])))
            self.assertEqual(record['applied'],extra>0)
            self.assertEqual(record['removed_inferior_mm'],float(extra))
            if extra:
                np.testing.assert_array_equal(np.asarray(cropped.dataobj)[:,:,10:],values[:,:,extra+10:])
            np.testing.assert_allclose(nib.affines.apply_affine(image.affine,[5,6,30+extra]),
                                       nib.affines.apply_affine(cropped.affine,[5,6,30]))
            outputs.append(tissue)
        for tissue in outputs[1:]: np.testing.assert_array_equal(tissue,outputs[0])

    def test_limited_coverage_preserves_input_and_nonbrain_scans_are_rejected(self):
        labels=np.zeros((8,9,40),np.uint8);labels[2:6,2:7,15:30]=2
        image=nib.Nifti1Image(np.ones(labels.shape),np.eye(4))
        unchanged,tissue,record=crop_inferior_fov(image,labels,10,20.)
        self.assertIs(unchanged,image);self.assertIs(tissue,labels)
        self.assertFalse(record['applied']);self.assertEqual(record['available_inferior_mm'],5.)
        with self.assertRaisesRegex(ValueError,'脑灰质'):crop_inferior_fov(image,np.zeros_like(labels),10)
        tilted=np.eye(4);tilted[0,2]=.1
        with self.assertRaisesRegex(ValueError,'axis-aligned'):
            crop_inferior_fov(nib.Nifti1Image(np.ones(labels.shape),tilted),labels,10)

    def test_backend_crops_before_electrode_placement_and_updates_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.nii';model=root/'model';model.mkdir()
            (model/'checkpoint_final.pth').write_bytes(b'fixture');(model/'postprocessing.json').write_text('{}')
            (model/'inference_config.json').write_text('{}')
            nib.save(nib.Nifti1Image(np.ones((8,9,100),np.float32),np.eye(4)),source)
            before={}
            def predict(path,*args,**kwargs):
                image=nib.load(path);before['shape']=image.shape;before['affine']=image.affine.copy()
                labels=np.zeros(image.shape,np.uint8);labels[11:16,11:17,70:95]=2;labels[12:15,12:16,75:90]=1
                probabilities=np.eye(7,dtype=np.float32)[labels].transpose(3,0,1,2)
                return labels,probabilities
            def place(**options):
                image=nib.load(options['padded_t1']);labels=nib.load(options['tissue_path'])
                self.assertEqual(image.shape,labels.shape);np.testing.assert_allclose(image.affine,labels.affine)
                self.assertLess(image.shape[2],before['shape'][2])
                np.testing.assert_allclose(nib.affines.apply_affine(image.affine,[12,12,30]),
                                           nib.affines.apply_affine(before['affine'],[12,12,70]))
                return {}
            with patch('SegandElec_nnUNet.pipeline.predict_tissues',side_effect=predict),patch('SegandElec_nnUNet.pipeline.place_consistent_electrodes',side_effect=place) as placed:
                result=run_nnunet_pipeline(t1=source,output_dir=root/'out',model_root=model,device='cpu',cleanup='raw',
                                          target_spacing_mm=1.,resample_grid="world",inferior_margin_mm=20.,model_label_order='metadata',
                                          save_intermediates=True,save_probabilities=True,progress=lambda _:None)
            placed.assert_called_once();self.assertTrue(result['field_of_view']['applied'])
            reference=nib.load(result['padded_t1'])
            for key in ('tissue_masks','raw_model_masks','remapped_masks','model_postprocessed_masks'):
                image=nib.load(result[key]);self.assertEqual(image.shape,reference.shape)
                np.testing.assert_allclose(image.affine,reference.affine)
            with np.load(result['probabilities']) as archive:
                self.assertEqual(tuple(archive['probabilities'].shape[1:]),reference.shape)
                np.testing.assert_allclose(archive['affine'],reference.affine)
                np.testing.assert_allclose(archive['probabilities'].sum(axis=0),1.)
            self.assertEqual(json.loads(Path(result['manifest']).read_text())['field_of_view'],result['field_of_view'])

    def test_previous_no_crop_caches_are_incompatible(self):
        expected=preprocessing_settings(10);previous=dict(expected,version=3)
        previous.pop('field_of_view_policy');previous.pop('inferior_margin_mm')
        self.assertFalse(signatures_match(previous,expected,allow_native_1mm=True))


if __name__=='__main__':unittest.main()
