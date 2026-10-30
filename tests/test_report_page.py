import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pymupdf as fitz
import numpy as np
import nibabel as nib
from core.engine import NiiEngine
from core.ti import synthesize
from tools.report_page import generate_report,report_center,slice_image
from test_individual_ti import fixture

class ReportTests(unittest.TestCase):
    def test_single_page_three_parallel_target_centered_views_and_only_requested_content(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory)
            result=synthesize(prediction,Path(directory)/'result',[[1,2],[3,72]],[2,2],
                roi_spec={'center_ras_mm':[0,0,0],'radius_mm':1.5},progress=lambda _:None)
            engine=NiiEngine();engine.load_image(str(t1));engine.load_overlay(result['overlays'][0],kind='field')
            pdf=Path(directory)/'report.pdf'
            with patch('tools.report_page.slice_image',wraps=slice_image) as capture:
                success,error=generate_report(engine,str(t1),'','个体靶区',output_path=str(pdf),
                    patient_info={'name':'测试患者 <示例>','age':'36','gender':'男','head_circumference':'56 cm',
                                  'disease_type':'示例','stimulation':result,'slice_indices':[0,0,0]})
            self.assertTrue(success,error)
            self.assertEqual([call.args[2] for call in capture.call_args_list],[(3,2,1)]*3)
            document=fitz.open(pdf);self.assertEqual(len(document),1)
            text=document[0].get_text()
            for required in ('患者基本信息','仿真结果方案','ROI 场强','频率设定需要结合实际情况'):
                self.assertIn(required,text)
            for removed in ('头围','56 cm','载频','差频','解剖底图信息','叠加图层概览','电极位置图'):
                self.assertNotIn(removed,text)
            pictures=document[0].get_images();self.assertEqual(len(pictures),3)
            rectangles=[document[0].get_image_rects(picture[0])[0] for picture in pictures]
            self.assertLess(max(rect.y0 for rect in rectangles)-min(rect.y0 for rect in rectangles),1)
            self.assertEqual(len(set(round(rect.x0) for rect in rectangles)),3)
            document.close()

    def test_selected_atlas_center_and_affine_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory);engine=NiiEngine();engine.load_image(str(t1))
            data=np.zeros(engine.data.shape,np.int16);data[1:3,2:4,1:3]=41;data[4:]=42
            mask=Path(directory)/'atlas.nii.gz';nib.save(nib.Nifti1Image(data,engine.affine),mask)
            self.assertEqual(report_center(engine,{'roi':{'mask':str(mask),'labels':[41]}}),(2,2,2))
            nib.save(nib.Nifti1Image(data,np.eye(4)),mask)
            with self.assertRaises(ValueError):report_center(engine,{'roi':{'mask':str(mask),'labels':[41]}})
