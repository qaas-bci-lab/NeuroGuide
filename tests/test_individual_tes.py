import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from core.ti import Leadfield, synthesize as synthesize_ti
from core.tes import synthesize, optimize, validate_montage
from test_individual_ti import fixture


class TESPhysicsTests(unittest.TestCase):
    def test_signed_superposition_reference_return_scaling_and_polarity(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);labels=[1,2,3,4,72];currents=[2,-.5,-.5,-.5,-.5]
            a=synthesize(prediction,Path(directory)/'a',labels,currents,progress=lambda _:None)
            b=synthesize(prediction,Path(directory)/'b',labels,[-c for c in currents],progress=lambda _:None)
            c=synthesize(prediction,Path(directory)/'c',labels,[v*.5 for v in currents],progress=lambda _:None)
            bank=np.load(prediction/'leadfield_gmwm.npy')
            expected=2*bank[:,0]-.5*(bank[:,1]+bank[:,2]+bank[:,3])
            np.testing.assert_allclose(nib.load(a['vector_field']).get_fdata().reshape(-1,3),expected,atol=1e-7)
            va=nib.load(a['overlays'][0]).get_fdata()
            np.testing.assert_allclose(va,nib.load(b['overlays'][0]).get_fdata(),atol=1e-7)
            np.testing.assert_allclose(va*.5,nib.load(c['overlays'][0]).get_fdata(),atol=1e-7)
            np.testing.assert_allclose(nib.load(a['vector_field']).get_fdata(),-nib.load(b['vector_field']).get_fdata(),atol=1e-7)
            self.assertEqual(a['current_summary'],{'signed_sum_ma':0.,'absolute_sum_ma':4.,'total_injected_ma':2.})
            self.assertTrue((Path(directory)/'a/electrode_currents.csv').is_file())

    def test_constraints_reject_unbalanced_over_cap_over_budget_duplicates_and_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);lf=Leadfield(prediction)
            try:
                for labels,currents in [([1,2],[1,-.9]),([1,2],[2.1,-2.1]),([1,2,3,4],[1.5,1.5,-1.5,-1.5]),([1,1],[1,-1]),([1,2,3],[1,-1,0])]:
                    with self.subTest(currents=currents),self.assertRaises(ValueError):validate_montage(lf,labels,currents)
                validate_montage(lf,[1,2,3,4,72],[2,-.5,-.5,-.5,-.5],electrode_count=5)
            finally:lf.close()

    def test_reproducible_five_electrode_search_and_every_finalist_feasible(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            options={'population':24,'generations':5,'seed':42,'finalist_count':8,'sample_per_region':16}
            a=optimize(prediction,Path(directory)/'a',roi,options,lambda _:None)
            b=optimize(prediction,Path(directory)/'b',roi,options,lambda _:None)
            self.assertEqual(a['electrode_labels'],b['electrode_labels']);self.assertEqual(a['currents_ma'],b['currents_ma'])
            self.assertEqual(len(a['electrode_labels']),5);self.assertEqual(a['optimization']['current_step_ma'],.05)
            lf=Leadfield(prediction)
            try:
                for item in a['optimization']['ranking']:
                    validate_montage(lf,item['electrode_labels'],item['currents_ma'],electrode_count=5)
                    r=synthesize(prediction,Path(directory)/'check',item['electrode_labels'],item['currents_ma'],roi,lambda _:None)
                    stats=r['statistics']
                    score=stats['roi']['mean_v_m']-stats['non_roi']['mean_v_m']
                    self.assertAlmostEqual(score,item['full_score'],places=6)
                self.assertEqual(a['statistics']['all_gmwm']['voxels'],len(lf.ijk))
            finally:lf.close()
            self.assertEqual(a['optimization']['ranking'][0]['full_score'],max(v['full_score'] for v in a['optimization']['ranking']))

    def test_two_electrode_reduction_matches_standard_pair_and_known_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            a=optimize(prediction,Path(directory)/'a',roi,{'electrode_count':2,'electrodes':[1,72],
                'avoidance_weight':0,'population':8,'generations':2},lambda _:None)
            self.assertEqual(a['currents_ma'],[2.,-2.])
            expected=2*np.linalg.norm(np.load(prediction/'leadfield_gmwm.npy')[:,0],axis=1)
            np.testing.assert_allclose(nib.load(a['overlays'][0]).get_fdata().ravel(),expected,atol=1e-7)

    def test_focality_full_score_and_invalid_discretization(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            result=optimize(prediction,Path(directory)/'a',roi,{'goal':'focality','population':8,'generations':2},lambda _:None)
            stats=result['statistics'];target=stats['roi'];avoid=stats['non_roi']
            score=target['fraction_above_threshold']-avoid['fraction_above_threshold']-2*max(0,.2-target['max_v_m'])/.2
            self.assertAlmostEqual(score,result['optimization']['ranking'][0]['full_score'],places=6)
            with self.assertRaises(ValueError):optimize(prediction,Path(directory)/'invalid',roi,{'ratio_steps':2},lambda _:None)

    def test_custom_caps_counts_and_nondecimal_current_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory);roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            lf=Leadfield(prediction)
            try:
                for count in [2,3,5,6]:
                    options={'electrode_count':count,'electrode_limit_ma':1.333,'absolute_sum_limit_ma':2.7,
                             'ratio_steps':31,'population':8,'generations':2}
                    result=optimize(prediction,Path(directory)/str(count),roi,options,lambda _:None)
                    for item in result['optimization']['ranking']:
                        validate_montage(lf,item['electrode_labels'],item['currents_ma'],1.333,2.7,count)
            finally:lf.close()

    def test_real_worker_synthesis_and_optimization_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory)
            for action in ['synthesize','optimize']:
                config={'action':action,'stimulation_type':'tES','prediction_dir':str(prediction),'output_dir':str(Path(directory)/action),
                        'roi':{'center_ras_mm':[0,0,0],'radius_mm':1.5}}
                if action=='synthesize':config.update(electrode_labels=[1,2,3,4,72],currents=[2,-.5,-.5,-.5,-.5])
                else:config['options']={'population':8,'generations':2}
                request=Path(directory)/'request.json';request.write_text(json.dumps(config),encoding='utf-8')
                process=subprocess.run([sys.executable,'tools/individual_worker.py',str(request)],capture_output=True,text=True,encoding='utf-8',timeout=60)
                self.assertEqual(process.returncode,0,process.stdout+process.stderr)
                events=[json.loads(line.split('NEUROGUIDE_EVENT ',1)[1]) for line in process.stdout.splitlines() if line.startswith('NEUROGUIDE_EVENT ')]
                result=next(e['value'] for e in events if e['kind']=='result')
                self.assertEqual(result['stimulation_type'],'tES');self.assertIn('run_timing',result)


class TESGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt6.QtWidgets import QApplication
        import ui.canvas
        ui.canvas.PYVISTA_AVAILABLE=False
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from main import ZjuMriViewer
        self.viewer=ZjuMriViewer();self.viewer.on_individual_mode_clicked()
        self.panel=self.viewer.individual_dialog;self.panel.stimulation_mode.setCurrentIndex(1)

    def tearDown(self):
        self.viewer.close();self.viewer.deleteLater();self.app.processEvents()

    def test_defaults_ui_dispatch_saved_result_field_and_ti_switch(self):
        panel=self.panel
        self.assertEqual(panel.tes_count.value(),5);self.assertEqual(panel.tes_cap.value(),2);self.assertEqual(panel.tes_total.value(),4)
        self.assertEqual([c.value() for c in panel.tes_currents],[2,-.5,-.5,-.5,-.5])
        self.assertTrue(panel.cap.isHidden());self.assertFalse(panel.tes_cap.isHidden())
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory);panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            panel.roi_mode.setCurrentIndex(0);panel.target_selected=True;panel.radius.setValue(1.5)
            panel.run=Mock();panel.start_optimization();config=panel.run.call_args.args[0]
            self.assertEqual(config['stimulation_type'],'tES');self.assertEqual(config['options']['electrode_count'],5)
            self.assertNotIn('channel_limit_ma',config['options'])
            result=optimize(prediction,Path(directory)/'result',config['roi'],{'population':8,'generations':2},lambda _:None)
            panel.apply_result(result);self.viewer.on_individual_field_ready(result)
            self.assertIn('tES',self.viewer.current_label.text());self.assertIn('5 电极',self.viewer.current_label.text())
            self.assertEqual(self.viewer.engine.overlays[0]['name'],'tES 电场幅值（峰值）(V/m)')
            self.assertEqual(self.viewer.engine.overlays[0]['cmap'],'jet')
            for _ in range(6):self.app.processEvents()
            self.assertGreaterEqual(self.viewer.current_label.height(),self.viewer.current_label.heightForWidth(self.viewer.current_label.width()))
            panel.run.reset_mock();panel.start_synthesis();manual=panel.run.call_args.args[0]
            self.assertEqual(manual['electrode_labels'],result['electrode_labels'])
            np.testing.assert_allclose(manual['currents'],result['currents_ma'],atol=1e-12)
            ti=synthesize_ti(prediction,Path(directory)/'ti',[[1,2],[3,72]],[1,.5],progress=lambda _:None)
            panel.apply_result(ti);self.assertFalse(panel.is_tes());self.assertEqual([c.value() for c in panel.currents],[1,.5])

    def test_rounded_optimized_currents_remain_balanced_in_manual_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory);self.panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            result=optimize(prediction,Path(directory)/'result',{'center_ras_mm':[0,0,0],'radius_mm':1.5},
                {'population':8,'generations':2,'electrode_limit_ma':1.333,'absolute_sum_limit_ma':2.7,'ratio_steps':31},lambda _:None)
            self.panel.apply_result(result)
            self.panel.run=Mock();self.panel.show_error=Mock();self.panel.start_synthesis()
            self.panel.show_error.assert_not_called();self.panel.run.assert_called_once()
            currents=self.panel.run.call_args.args[0]['currents']
            self.assertLess(abs(sum(currents)),1e-8)

    def test_manual_constraint_error_is_shown_without_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory);self.panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            for combo,label in zip(self.panel.tes_electrodes,[1,2,3,4,72]):combo.setCurrentIndex(combo.findData(label))
            self.panel.tes_currents[0].setValue(1.9)
            self.panel.show_error=Mock();self.panel.run=Mock();self.panel.start_synthesis()
            self.panel.run.assert_not_called();self.assertIn('电流不平衡',self.panel.show_error.call_args.args[1])

    def test_one_page_tes_report_has_five_signed_currents_and_three_views(self):
        from core.engine import NiiEngine
        from tools.report_page import generate_report
        import pymupdf as fitz
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory);result=synthesize(prediction,Path(directory)/'result',[1,2,3,4,72],[2,-.5,-.5,-.5,-.5],
                {'center_ras_mm':[0,0,0],'radius_mm':1.5},lambda _:None)
            engine=NiiEngine();engine.load_image(str(t1));engine.load_overlay(result['overlays'][0],kind='field')
            path=Path(directory)/'report.pdf'
            success,error=generate_report(engine,str(t1),'','tES',str(path),patient_info={'name':'测试','stimulation':result})
            self.assertTrue(success,error)
            with fitz.open(path) as document:
                self.assertEqual(len(document),1);page=document[0];text=page.get_text()
                self.assertIn('tES',text);self.assertIn('+2.00000',text);self.assertEqual(text.count('-0.50000'),4)
                self.assertIn('交流',text);self.assertEqual(len(page.get_images()),3)


if __name__=='__main__':unittest.main()
