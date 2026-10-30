"""Offscreen GUI and actual subprocess handshake regression checks."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import nibabel as nib
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt6.QtCore import QEventLoop, QTimer, Qt
import ui.canvas
ui.canvas.PYVISTA_AVAILABLE = False  # These tests exercise 2D/worker UI, not VTK drivers.
from main import ZjuMriViewer
from core.ti import synthesize
from test_individual_ti import fixture


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = ZjuMriViewer()

    def tearDown(self):
        self.viewer.close(); self.viewer.deleteLater(); self.app.processEvents()

    def test_single_workflow_startup_and_default_jet(self):
        self.viewer.on_individual_mode_clicked()
        self.assertTrue(self.viewer.individual_mode)
        self.assertFalse(hasattr(self.viewer, "disease_combo"))
        self.assertFalse(hasattr(self.viewer, "btn_std_mode"))
        self.assertFalse(self.viewer.individual_dialog.isWindow())
        self.assertEqual(self.viewer.cmap_combo.currentText(), 'jet')
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = fixture(directory)
            result = synthesize(prediction, Path(directory)/'result', [[1, 2], [3, 72]], [1, .5], [2000, 2010], progress=lambda _: None)
            self.viewer.on_individual_field_ready(result)
            self.assertIn('1 mA', self.viewer.current_label.text())
            self.assertIn('0.5 mA', self.viewer.current_label.text())
            self.assertNotIn('Hz',self.viewer.current_label.text())
            self.assertIn('目标区域场强',self.viewer.current_label.text())
            self.assertEqual(self.viewer.engine.overlays[0]['cmap'], 'jet')

    def test_builtin_mri_uses_the_same_model_inspection_and_prediction(self):
        self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        self.assertFalse(hasattr(self.viewer, 'target_combo'))
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory)
            panel.task={'task_dir':'old'}; panel.atlas_record={'atlas_path':'old'}; panel.target_selected=True
            with patch('ui.individual.builtin_mri_path',return_value=t1), patch.object(panel,'run') as run:
                panel.builtin_button.click()
                self.assertEqual(run.call_args.args[0]['action'],'inspect')
                self.assertEqual(run.call_args.args[0]['t1'],str(t1))
                self.assertEqual(self.viewer.current_base_path,str(t1))
                self.assertIsNone(panel.task); self.assertIsNone(panel.atlas_record)
                self.assertFalse(panel.target_selected); self.assertFalse(panel.tabs.isTabEnabled(1))
                self.assertIn('内置参考 MRI',panel.source_note.text())
                panel.start_prediction()
                config=run.call_args.args[0]
                self.assertEqual(config['action'],'predict'); self.assertEqual(config['t1'],str(t1))
                self.assertEqual(config['padding'],10); self.assertEqual(config['device'],'auto')

    def test_failed_mri_selection_preserves_existing_model_and_field(self):
        panel=self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory)
            panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            result=synthesize(prediction,Path(directory)/'result',[[1,2],[3,72]],[1,.5],progress=lambda _:None)
            self.viewer.on_individual_field_ready(result)
            engine=self.viewer.engine; task=panel.task; text=self.viewer.current_label.text()
            with patch.object(QMessageBox,'warning'),patch.object(panel,'run') as run:
                self.assertFalse(panel.select_mri(str(Path(directory)/'missing.nii')))
                run.assert_not_called()
            self.assertIs(self.viewer.engine,engine); self.assertIs(panel.task,task)
            self.assertEqual(self.viewer._individual_result,result)
            self.assertEqual(self.viewer.current_label.text(),text)

    def test_builtin_resource_resolves_packaged_mri_from_another_directory(self):
        from ui.individual import builtin_mri_path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); app=root/'program'; packed=app/'_internal'; (packed/'mri').mkdir(parents=True)
            reference=packed/'mri'/'T1.nii'; reference.touch()
            with patch('ui.individual.app_root',return_value=app),patch.object(sys,'_MEIPASS',str(packed),create=True):
                self.assertEqual(builtin_mri_path(),reference)
                (app/'mri').mkdir(); external=app/'mri'/'TI.nii'; external.touch()
                self.assertEqual(builtin_mri_path(),external)
            self.assertEqual(self.viewer.current_stimulation, self.viewer.individual_dialog.owner.current_stimulation)

    def test_independent_channel_limit_and_floating_advanced(self):
        self.viewer.show();self.viewer.on_individual_mode_clicked();panel=self.viewer.individual_dialog
        self.assertFalse(hasattr(panel,'budget'))
        self.assertEqual([c.value() for c in panel.currents],[2.,2.])
        self.assertTrue(self.viewer.display_settings.isHidden())
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory)
            panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            panel.set_busy(False);panel.tabs.setCurrentIndex(1);panel.roi_mode.setCurrentIndex(0)
            panel.target_selected=True;panel.radius.setValue(1.5)
            panel.run=Mock();panel.start_optimization()
            options=panel.run.call_args.args[0]['options']
            self.assertNotIn('total_peak_ma',options);self.assertEqual(options['channel_limit_ma'],2.)
            self.app.processEvents()
            before=self.viewer.current_panel.mapTo(self.viewer,self.viewer.current_panel.rect().topLeft())
            panel.advanced.toggle.click();self.app.processEvents()
            after=self.viewer.current_panel.mapTo(self.viewer,self.viewer.current_panel.rect().topLeft())
            self.assertEqual(before,after);self.assertTrue(panel.advanced.drawer.isVisible())
            self.assertEqual(self.viewer.sidebar_scroll.verticalScrollBar().maximum(),0)
            panel.cap.setValue(1.5)
            self.assertEqual([c.maximum() for c in panel.currents],[1.5,1.5])
            panel.advanced.set_expanded(False)
            self.assertFalse(panel.advanced.drawer.isVisible())
            panel.advanced.set_expanded(True);panel.set_busy(True)
            self.assertFalse(panel.advanced.drawer.isVisible());self.assertFalse(panel.advanced.body.isEnabled())
            panel.set_busy(False)

    def test_target_location_for_point_and_selected_atlas_labels(self):
        self.viewer.on_individual_mode_clicked();panel=self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory)
            panel.task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1)}
            self.viewer.execute_base_mri_load(str(t1));panel.set_busy(False)
            panel.roi_mode.setCurrentIndex(0);panel.target_selected=True
            panel.locate_button.click()
            self.assertEqual([self.viewer.sliders[a].value() for a in ('X','Y','Z')],[3,2,1])
            image=nib.load(t1);data=np.zeros(image.shape,np.int16);data[1:3,2:4,1:3]=41;data[4:]=42
            path=Path(directory)/'atlas.nii.gz';nib.save(nib.Nifti1Image(data,image.affine),path)
            self.viewer.locate_roi({'mask':str(path),'labels':[41]},str(t1))
            self.assertEqual([self.viewer.sliders[a].value() for a in ('X','Y','Z')],[2,2,2])

    def test_frequency_free_ui_opaque_fields_and_roi_result_summary(self):
        self.viewer.on_individual_mode_clicked();panel=self.viewer.individual_dialog
        self.assertEqual(panel.roi_mode.count(),2)
        self.assertEqual([panel.roi_mode.itemText(i) for i in range(2)],['在 MRI 上点选','按 AAL3 脑区选择'])
        self.assertFalse(hasattr(panel,'frequencies'));self.assertFalse(hasattr(panel,'roi_mask'))
        self.assertEqual(self.viewer.alpha_slider.value(),100)
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory);panel.task={'prediction_dir':str(prediction),'padded_t1':str(t1),'task_dir':directory}
            config=panel.job_config('synthesize');self.assertNotIn('frequencies',config)
            roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            result=synthesize(prediction,Path(directory)/'result',[[1,2],[3,72]],[2,2],roi_spec=roi,progress=lambda _:None)
            panel.apply_result(result);self.viewer.on_individual_field_ready(result)
            self.assertEqual(self.viewer.engine.overlays[0]['alpha'],1.)
            self.assertEqual(self.viewer.get_render_params()[-1][0]['alpha'],1.)
            self.assertIn('目标区域均值',self.viewer.current_label.text())
            self.assertIn(f"{result['statistics']['roi']['max_v_m']:.4f}",self.viewer.current_label.text())
            self.assertNotIn('差频',self.viewer.current_label.text())
            self.assertNotIn('frequencies_hz',result)

    def test_qprocess_worker_result(self):
        self.viewer.show()
        self.viewer.on_individual_mode_clicked(); dialog = self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction, t1 = fixture(directory)
            dialog.task = {'prediction_dir': str(prediction), 'task_dir': directory, 'padded_t1': str(t1)}
            dialog.storage_root = Path(directory)
            loop = QEventLoop(); completed = []
            dialog.field_ready.connect(lambda result: (completed.append(result), loop.quit()))
            dialog.run({'action': 'synthesize', 'prediction_dir': str(prediction), 'output_dir': str(Path(directory)/'result'),
                        'pairs': [[1, 2], [3, 72]], 'currents': [1, .5]})
            self.assertFalse(dialog.builtin_button.isEnabled())
            QTimer.singleShot(30000, loop.quit); loop.exec()
            if dialog.process:
                dialog.cancel(); dialog.process.waitForFinished(3000)
            self.assertEqual(len(completed), 1, dialog.log.toPlainText())
            self.assertEqual(completed[0]['currents_peak_ma'], [1., .5])
            self.assertNotIn('frequencies_hz',completed[0])
            self.assertFalse(dialog.isHidden())
            self.assertTrue(dialog.isVisible())
            self.assertFalse(dialog.cancel_button.isVisible())
            self.assertTrue(dialog.builtin_button.isEnabled())


    def test_fixed_runtime_settings_and_collapsed_defaults(self):
        self.viewer.on_individual_mode_clicked(); dialog = self.viewer.individual_dialog
        for field in ('brain_root', 'output_root', 'python', 'device', 'padding'):
            self.assertFalse(hasattr(dialog, field))
        self.assertTrue(dialog.advanced.body.isHidden())
        self.assertFalse(hasattr(dialog,"manual_advanced"));self.assertFalse(hasattr(dialog,"frequencies"))
        self.assertTrue(dialog.details.body.isHidden())
        self.assertTrue(self.viewer.display_settings.body.isHidden())
        self.assertFalse(hasattr(self.viewer, "standard_target_section"))
        dialog.task = {'task_dir': 'test'}
        dialog.set_busy(False)
        dialog.advanced.toggle.click()
        self.assertFalse(dialog.advanced.body.isHidden())
        with tempfile.TemporaryDirectory() as directory:
            _, t1 = fixture(directory)
            dialog.t1.setText(str(t1)); dialog.run = Mock()
            dialog.start_prediction()
            request = dialog.run.call_args.args[0]
            self.assertEqual(request['padding'], 10)
            self.assertEqual(request['device'], 'auto')
            self.assertEqual(request['brain_root'], str(dialog.models))
            self.assertEqual(request['output_root'], str(dialog.storage_root))

    def test_crosshair_target_roundtrip_and_advanced_parameters(self):
        self.viewer.on_individual_mode_clicked(); dialog = self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction, t1 = fixture(directory)
            self.viewer.execute_base_mri_load(str(t1))
            dialog.task = {'prediction_dir': str(prediction), 'task_dir': directory, 'padded_t1': str(t1)}
            dialog.set_busy(False); dialog.tabs.setCurrentIndex(1); dialog.roi_mode.setCurrentIndex(0)
            self.assertFalse(dialog.advanced.body.isAncestorOf(dialog.sphere_panel))
            self.assertFalse(hasattr(dialog, 'pick_button'))
            self.assertEqual(dialog.goal.currentText(), '优先提高目标区域场强')
            with self.assertRaises(ValueError):
                dialog.roi_spec()
            self.viewer.handle_canvas_click('axial', 3, 2)
            self.assertFalse(dialog.target_selected)
            dialog.crosshair_button.click()
            self.assertTrue(dialog.target_selected)
            expected=self.viewer.engine.voxel_to_mni(*(self.viewer.sliders[a].value() for a in ('X','Y','Z')))
            np.testing.assert_allclose(dialog.roi_spec()['center_ras_mm'], expected, atol=.01)
            self.assertEqual(dialog.tabs.currentIndex(), 1)
            dialog.finalists.setValue(3); dialog.ratio_steps.setValue(11); dialog.sample_size.setValue(128)
            dialog.run = Mock(); dialog.start_optimization()
            request = dialog.run.call_args.args[0]
            self.assertEqual(request['options']['finalist_count'], 3)
            self.assertEqual(request['options']['ratio_steps'], 11)
            self.assertEqual(request['options']['sample_per_region'], 128)
            self.assertEqual(request['options']['goal'], 'intensity')
            dialog.center[0].setValue(4.25)
            self.assertEqual(dialog.roi_spec()['center_ras_mm'][0], 4.25)
            dialog.roi_mode.setCurrentIndex(1)
            self.assertTrue(dialog.sphere_panel.isHidden())

    def test_aal_regions_search_preview_and_optimization_roi(self):
        self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction,t1=fixture(directory); image=nib.load(t1)
            labels=np.full(image.shape,41,dtype=np.int16); labels[3:]=42
            atlas_path=Path(directory)/'AAL3_native.nii.gz'; nib.save(nib.Nifti1Image(labels,image.affine),atlas_path)
            record={'atlas_path':str(atlas_path),'atlas_version':'AAL3v1','labels':{'41':'Hippocampus_L','42':'Hippocampus_R'},'gmwm_counts':{'41':60,'42':60}}
            task={'prediction_dir':str(prediction),'task_dir':directory,'padded_t1':str(t1),'tissue_masks':str(t1),'atlas':record}
            panel.attach_task(task); panel.tabs.setCurrentIndex(1)
            self.assertEqual(panel.roi_mode.currentIndex(),1)
            panel.region_search.setText('海马')
            self.assertFalse(panel.region_list.item(0).isHidden())
            for i in range(2): panel.region_list.item(i).setCheckState(Qt.CheckState.Checked)
            roi=panel.roi_spec(); self.assertEqual(roi['labels'],[41,42]); self.assertEqual(roi['atlas_id'],'aal3')
            self.assertEqual(self.viewer.engine.overlays[-1]['active_labels'],[41,42])
            panel.run=Mock(); panel.start_optimization()
            self.assertEqual(panel.run.call_args.args[0]['roi']['mask'],str(atlas_path))
            result={'pairs':[[1,2],[3,72]],'currents_peak_ma':[1.,1.],'frequencies_hz':[2000,2010],'roi':roi}
            panel.clear_regions(); panel.apply_result(result)
            self.assertEqual(panel.selected_regions(),[41,42])
            panel.clear_regions()
            with self.assertRaises(ValueError): panel.roi_spec()

    def test_new_mri_discards_old_atlas_and_plan(self):
        self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        panel.task={'task_dir':'old'}; panel.atlas_record={'atlas_path':'old'}; panel.target_selected=True
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory)
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(t1),'MRI')), patch.object(panel,'run') as run:
                panel.browse_t1()
                self.assertEqual(run.call_args.args[0]['action'],'inspect')
            self.assertIsNone(panel.task); self.assertIsNone(panel.atlas_record)
            self.assertFalse(panel.tabs.isTabEnabled(1)); self.assertFalse(panel.target_selected)
            self.assertEqual(panel.t1_name.text(),t1.name)
            self.assertEqual(self.viewer.current_base_path,str(t1))

    def test_field_window_bounds_peak_and_independent_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory); image=nib.load(t1); self.viewer.execute_base_mri_load(str(t1))
            for k in (1,2):
                path=Path(directory)/f'field{k}.nii.gz'
                data=np.linspace(0,k,np.prod(image.shape),dtype=np.float32).reshape(image.shape)
                nib.save(nib.Nifti1Image(data,image.affine),path)
                ok,_=self.viewer.engine.load_overlay(path,kind='field'); self.assertTrue(ok)
            self.viewer.refresh_layer_list_ui(schedule=False); self.viewer.layer_list.setCurrentRow(0)
            self.assertAlmostEqual(self.viewer.engine.overlays[0]['display_max'],.95)
            self.assertIn('1 V/m',self.viewer.field_peak_label.text())
            self.viewer.min_slider.setValue(200); self.viewer.max_slider.setValue(700)
            self.assertAlmostEqual(self.viewer.engine.overlays[0]['display_min'],.2)
            self.assertAlmostEqual(self.viewer.engine.overlays[0]['display_max'],.7)
            self.viewer.layer_list.setCurrentRow(1)
            self.assertAlmostEqual(self.viewer.engine.overlays[1]['display_max'],1.9)
            params=self.viewer.get_render_params()[-1]
            self.assertAlmostEqual(params[0]['max'],.7)
            self.viewer.layer_list.setCurrentRow(0); self.assertEqual(self.viewer.max_slider.value(),700)
            self.viewer.min_slider.setValue(950)
            self.assertLess(self.viewer.min_slider.value(),self.viewer.max_slider.value())
            self.viewer.reset_field_range(); self.assertEqual(self.viewer.max_slider.value(),950)
            self.assertEqual(self.viewer.engine.overlays[0]['data_max'],1.)

    def test_saved_result_restores_target_and_currents(self):
        self.viewer.on_individual_mode_clicked(); panel = self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as directory:
            prediction, t1 = fixture(directory)
            panel.task = {'prediction_dir':str(prediction), 'task_dir':directory, 'padded_t1':str(t1)}
            roi = {'center_ras_mm':[0,0,0], 'radius_mm':1.5}
            result = synthesize(prediction, Path(directory)/'result', [[1,2],[3,72]], [.8,.5], [2000,2010], roi_spec=roi, progress=lambda _:None)
            panel.apply_result(result)
            self.assertEqual(panel.roi_spec(), roi)
            self.assertFalse(hasattr(panel,"budget"))
            self.assertEqual([x.value() for x in panel.currents], [.8,.5])
            self.viewer.on_individual_field_ready(result)
            self.assertEqual([self.viewer.sliders[a].value() for a in ('X','Y','Z')],[3,2,1])
            self.assertTrue(self.viewer.engine.overlays[-1]['is_roi'])
            self.assertEqual(self.viewer.layer_list.item(1).checkState(),Qt.CheckState.Checked)

    def test_embedded_layout_layers_and_workspace_reentry(self):
        self.viewer.show(); self.viewer.on_individual_mode_clicked()
        self.app.processEvents()
        panel = self.viewer.individual_dialog
        self.assertFalse(panel.isWindow())
        self.assertIs(panel.parentWidget(), self.viewer.individual_host)
        self.assertTrue(panel.isVisible())
        self.assertGreaterEqual(self.viewer.sidebar_scroll.width(), 400)
        self.assertLessEqual(self.viewer.sidebar_scroll.width(), 680)
        self.assertEqual(self.viewer.layer_list.height(), 42)
        panel.task = {'task_dir':'test'}; panel.set_busy(False)
        for index, button in enumerate((panel.predict_button, panel.optimize_button, panel.synthesize_button)):
            panel.tabs.setCurrentIndex(index); self.app.processEvents()
            self.assertTrue(button.isVisible())
            self.assertTrue(panel.rect().contains(button.mapTo(panel, button.rect().center())))
        panel.tabs.setCurrentIndex(1); panel.set_busy(True)
        self.assertEqual(panel.tabs.currentIndex(),1)
        panel.set_busy(False); self.assertEqual(panel.tabs.currentIndex(),1)
        self.assertEqual(self.viewer.sidebar_scroll.verticalScrollBar().maximum(), 0)
        for i in range(8): self.viewer.layer_list.addItem(f'Layer {i}')
        self.assertLessEqual(self.viewer.layer_list.height(), 160)
        self.viewer.layer_list.clear(); self.assertEqual(self.viewer.layer_list.height(), 42)
        with tempfile.TemporaryDirectory() as directory:
            prediction, t1 = fixture(directory)
            panel.task = {'prediction_dir':str(prediction), 'task_dir':directory, 'padded_t1':str(t1)}
            panel.set_busy(False); panel.tabs.setCurrentIndex(1)
            self.viewer.execute_base_mri_load(str(t1))
            panel.begin_target_selection(); panel.tabs.setCurrentIndex(2)
            self.assertFalse(panel.selection_mode)
            self.assertEqual(self.viewer.statusBar().currentMessage(), '')
            panel.begin_target_selection()
            self.viewer.show_home()
            self.viewer.on_individual_mode_clicked()
            self.assertIs(panel, self.viewer.individual_dialog)
            self.assertTrue(panel.isVisible())
            self.assertEqual(panel.task['prediction_dir'], str(prediction))
            panel.roi_mode.setCurrentIndex(1)
            self.assertFalse(panel.selection_mode)


    def test_mri_check_worker_offers_and_reuses_only_matching_model(self):
        from test_individual_cache import cache_fixture
        self.viewer.show(); self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        self.assertFalse(hasattr(panel,'saved_models'))
        self.assertFalse(hasattr(panel,'history'))
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, _ = cache_fixture(directory)
            panel.storage_root=Path(config['output_root']); panel.models=Path(config['brain_root'])
            panel.root=Path(__file__).resolve().parents[1]
            loop=QEventLoop(); completed=[]
            panel.task_ready.connect(lambda task:(completed.append(task),loop.quit()))
            def accept(box):
                box.buttons()[0].click()
                return 0
            with patch.object(QMessageBox,'exec',accept), patch.object(QFileDialog,'getOpenFileName',return_value=(config['t1'],'MRI')):
                panel.browse_t1()
                self.assertIsNotNone(panel.process)
                QTimer.singleShot(30000,loop.quit); loop.exec()
            if panel.process:
                panel.cancel(); panel.process.waitForFinished(3000)
            self.assertEqual(len(completed),1,panel.log.toPlainText())
            self.assertEqual(panel.task['manifest'],str(manifest))
            self.assertEqual(panel.tabs.currentIndex(),1)
            self.assertTrue(panel.tabs.isTabEnabled(2))
            # Choosing the same source again must also inspect it, and clear
            # the prior model until the user accepts a verified match.
            with patch.object(QFileDialog,'getOpenFileName',return_value=(config['t1'],'MRI')), patch.object(panel,'run') as run:
                panel.browse_t1()
                self.assertEqual(run.call_args.args[0]['action'],'inspect')
            self.assertIsNone(panel.task)
            self.assertFalse(panel.tabs.isTabEnabled(1))

    def test_declining_reuse_keeps_raw_mri_and_no_optimization(self):
        self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        panel._inspected_t1='chosen.nii'; panel.t1.setText('chosen.nii')
        def decline(box):
            box.buttons()[1].click()
            return 0
        with patch.object(QMessageBox,'exec',decline), patch.object(panel,'attach_task') as attach:
            panel.offer_existing_model({'task':{'task_dir':'saved'}})
            attach.assert_not_called()
        self.assertIsNone(panel.task); self.assertFalse(panel.tabs.isTabEnabled(1))

    def test_atlas_select_all_clear_and_six_complete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory); image=nib.load(t1)
            self.viewer.execute_base_mri_load(str(t1))
            atlas=Path(directory)/'regions.nii.gz'
            labels=(np.arange(np.prod(image.shape)).reshape(image.shape)%8+1).astype(np.int16)
            nib.save(nib.Nifti1Image(labels,image.affine),atlas)
            ok,_=self.viewer.engine.load_overlay(str(atlas),kind='atlas'); self.assertTrue(ok)
            self.viewer.refresh_layer_list_ui(schedule=False); self.viewer.layer_list.setCurrentRow(0)
            self.viewer.show(); self.viewer.stack.setCurrentIndex(1)
            self.viewer.display_settings.toggle.click(); self.app.processEvents()
            self.viewer.clear_all_labels_button.click()
            self.assertEqual(self.viewer.engine.overlays[0]['active_labels'],[])
            self.assertTrue(all(self.viewer.label_list.item(i).checkState()==Qt.CheckState.Unchecked for i in range(8)))
            self.viewer.select_all_labels_button.click()
            self.assertEqual(self.viewer.engine.overlays[0]['active_labels'],list(range(1,9)))
            sixth=self.viewer.label_list.visualItemRect(self.viewer.label_list.item(5))
            self.assertLessEqual(sixth.bottom(),self.viewer.label_list.viewport().height())
            self.assertTrue(self.viewer.display_settings.drawer.isVisible())
            self.assertTrue(self.viewer.display_settings.overlay)

    def test_plane_sliders_coordinates_and_left_display_controls(self):
        self.viewer.show(); self.viewer.stack.setCurrentIndex(1)
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory); self.viewer.execute_base_mri_load(str(t1))
            self.app.processEvents()
            self.assertIs(self.viewer.cmap_combo.parentWidget(),self.viewer.info_panel)
            self.assertIs(self.viewer.alpha_slider.parentWidget(),self.viewer.info_panel)
            for axis in ('X','Y','Z'):
                self.assertIs(self.viewer.sliders[axis].parentWidget(),self.viewer.slice_panels[axis])
                self.assertTrue(self.viewer.sliders[axis].isEnabled())
            self.viewer.sliders['X'].setValue(4)
            self.assertIn('1.0 mm',self.viewer.slice_coord_labels['X'].text())
            self.assertIn('5/6',self.viewer.slice_coord_labels['X'].text())
            self.viewer.handle_3d_slider_change(1,2,3)
            self.assertIn('-2.0 mm',self.viewer.slice_coord_labels['X'].text())
            self.assertIn('2/6',self.viewer.slice_coord_labels['X'].text())
            self.viewer.alpha_slider.setValue(45)
            self.assertEqual(self.viewer.alpha_value_label.text(),'45%')

    def test_stage_times_and_only_unused_nnunet_path_messages_filtered(self):
        from ui.runtime_status import duration_text
        self.viewer.on_individual_mode_clicked(); panel=self.viewer.individual_dialog
        self.assertTrue(panel.irrelevant_nnunet_message('nnUNet_raw is not defined and nnU-Net can only be used...'))
        self.assertTrue(panel.irrelevant_nnunet_message('nnUNet_preprocessed is not defined and nnU-Net can not...'))
        self.assertTrue(panel.irrelevant_nnunet_message('nnUNet_results is not defined and nnU-Net cannot...'))
        self.assertFalse(panel.irrelevant_nnunet_message('CUDA out of memory'))
        self.assertFalse(panel.irrelevant_nnunet_message('nnUNet_results checkpoint is missing'))
        panel.timings.update_stage({'stage':'tissue_segmentation','state':'running','device':'cuda:0'})
        self.assertIn('运行中',panel.timings.text())
        panel.timings.update_stage({'stage':'tissue_segmentation','state':'completed','device':'cuda:0','seconds':72})
        self.assertIn('1 分 12 秒',panel.timings.text())
        panel.timings.update_stage({'stage':'leadfield_prediction','state':'cached'})
        self.assertIn('已复用',panel.timings.text())
        self.assertEqual(duration_text(3661),'1 小时 01 分')
        self.assertIs(panel.timings.parentWidget(),self.viewer.run_details.body)
        import codecs
        panel._decoder=codecs.getincrementaldecoder('utf-8')()
        output=('nnUNet_raw is not defined and nnU-Net can only be used...\n'
                'nnUNet_preprocessed is not defined and nnU-Net can not...\n'
                'nnUNet_results is not defined and nnU-Net cannot...\n'
                'Real diagnostic: CUDA out of memory\n').encode('utf-8')
        panel.process=Mock(); panel.process.readAllStandardOutput.return_value=output
        try: panel.read_output()
        finally: panel.process=None
        self.assertNotIn('nnUNet_raw is not defined',panel.log.toPlainText())
        self.assertIn('Real diagnostic: CUDA out of memory',panel.log.toPlainText())

    def test_top_alignment_and_fixed_action_with_left_logs(self):
        self.viewer.show(); self.viewer.stack.setCurrentIndex(1); self.app.processEvents()
        self.assertFalse(hasattr(self.viewer, "btn_std_mode"))
        self.viewer.on_individual_mode_clicked(); self.app.processEvents()
        panel=self.viewer.individual_dialog
        position=lambda:panel.predict_button.mapTo(self.viewer.viewer_page,panel.predict_button.rect().topLeft()).y()
        self.viewer.field_range_panel.show()
        self.viewer.current_label.setText('通道 1：±1 mA\n通道 2：±1 mA\n差频：10 Hz')
        self.app.processEvents();self.app.processEvents();initial=position()
        self.viewer.run_details.toggle.click()
        panel.set_busy(True); self.app.processEvents()
        self.assertEqual(position(),initial)
        panel.set_busy(False); self.app.processEvents(); self.assertEqual(position(),initial)
        self.assertLess(self.viewer.coord_label.mapTo(self.viewer.viewer_page,self.viewer.coord_label.rect().topLeft()).y(),self.viewer.canvases['sagittal'].mapTo(self.viewer.viewer_page,self.viewer.canvases['sagittal'].rect().topLeft()).y())
        self.assertEqual(self.viewer.coord_label.parentWidget(),self.viewer.val_label.parentWidget())
        self.assertIs(self.viewer.current_label.parentWidget(),self.viewer.current_panel)
        self.assertIs(panel.details,self.viewer.run_details)

    def test_mouse_wheel_moves_only_the_hovered_plane_and_clamps(self):
        from PyQt6.QtCore import QPoint, QPointF
        from PyQt6.QtGui import QWheelEvent
        self.viewer.show(); self.viewer.on_individual_mode_clicked()
        with tempfile.TemporaryDirectory() as directory:
            _,t1=fixture(directory); self.viewer.execute_base_mri_load(str(t1))
            self.app.processEvents(); self.viewer.update_2d_views()
            def wheel(plane, delta):
                canvas=self.viewer.canvases[plane]; canvas.draw()
                box=canvas.ax.bbox; ratio=canvas.devicePixelRatioF()
                local=QPointF((box.x0+box.x1)/2/ratio,canvas.height()-(box.y0+box.y1)/2/ratio)
                event=QWheelEvent(local,QPointF(canvas.mapToGlobal(local.toPoint())),QPoint(),QPoint(0,delta),
                                  Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
                self.app.sendEvent(canvas,event)
            for plane,axis in [('sagittal','X'),('coronal','Y'),('axial','Z')]:
                self.viewer.sliders[axis].setValue(1)
                before={a:s.value() for a,s in self.viewer.sliders.items()}
                wheel(plane,120)
                self.assertEqual(self.viewer.sliders[axis].value(),2)
                for other in set(before)-{axis}: self.assertEqual(self.viewer.sliders[other].value(),before[other])
                wheel(plane,-120);self.assertEqual(self.viewer.sliders[axis].value(),1)
                self.viewer.sliders[axis].setValue(0);wheel(plane,-120)
                self.assertEqual(self.viewer.sliders[axis].value(),0)
                self.viewer.sliders[axis].setValue(self.viewer.sliders[axis].maximum());wheel(plane,120)
                self.assertEqual(self.viewer.sliders[axis].value(),self.viewer.sliders[axis].maximum())

    def test_ti_electrode_pairs_have_matching_shapes_tags_and_connections(self):
        from ui.electrode_map import electrode_dialog
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'electrodes.tsv'
            path.write_text('name\tras_x\tras_y\tras_z\n'+'\n'.join(f'{name}\t0\t0\t0' for name in ['F3','F4','P3','P4','Iz']),encoding='utf-8')
            result={'pairs':[[1,2],[3,4]],'currents_peak_ma':[1.5,2.]}
            dialog=electrode_dialog(self.viewer,str(path),result)
            self.assertEqual(len(dialog.electrode_pair_lines),2)
            for pair,line in zip([['F3','F4'],['P3','P4']],dialog.electrode_pair_lines):
                np.testing.assert_allclose(np.c_[line.get_xdata(),line.get_ydata()],
                                           [dialog.electrode_positions[name] for name in pair])
            self.assertEqual([line.get_linestyle() for line in dialog.electrode_pair_lines],['-','--'])
            labels=[text.get_text() for text in dialog.figure_canvas.figure.axes[0].texts]
            for tag in ('1+','1-','2+','2-'): self.assertTrue(any(tag in label for label in labels))
            self.assertIn('第1对',dialog.scheme_legend.text());self.assertIn('第2对',dialog.scheme_legend.text())
            markers=[len(dot.get_paths()[0].vertices) for dot in dialog.figure_canvas.figure.axes[0].collections]
            self.assertNotEqual(markers[0],markers[2]);self.assertEqual(markers[0],markers[1]);self.assertEqual(markers[2],markers[3])
            dialog.close()
            tes=electrode_dialog(self.viewer,str(path),{'stimulation_type':'tES','electrode_labels':[1,2,3,4,5],'currents_ma':[2.,-.5,-.5,-.5,-.5]})
            self.assertEqual(tes.electrode_pair_lines,[]);self.assertNotIn('第1对',tes.scheme_legend.text());tes.close()



    def test_portable_task_directory_does_not_depend_on_localappdata(self):
        from ui.individual import task_root
        with tempfile.TemporaryDirectory() as folder:
            with patch('ui.individual.app_root',return_value=Path(folder)),patch.object(sys,'frozen',True,create=True),patch.dict(os.environ,{'LOCALAPPDATA':'C:/unrelated'}):
                self.assertEqual(task_root(),Path(folder)/'NeuroGuide/individualized')

    def test_shared_target_controls_are_visible_in_manual_and_automatic_pages(self):
        panel=self.viewer.individual_dialog
        with tempfile.TemporaryDirectory() as folder:
            prediction,t1=fixture(folder);self.viewer.execute_base_mri_load(str(t1))
            panel.task={'task_dir':folder,'prediction_dir':str(prediction),'padded_t1':str(t1)}
            panel.set_busy(False);panel.tabs.setCurrentIndex(1);panel.roi_mode.setCurrentIndex(0)
            panel.crosshair_button.click();roi=panel.roi_spec()
            self.assertTrue(panel.tabs.widget(1).isAncestorOf(panel.shared_target_controls))
            panel.tabs.setCurrentIndex(2)
            self.assertTrue(panel.tabs.widget(2).isAncestorOf(panel.shared_target_controls))
            self.assertEqual(panel.roi_spec(),roi)
            panel.center[0].setValue(1.25);panel.tabs.setCurrentIndex(1)
            self.assertEqual(panel.roi_spec()['center_ras_mm'][0],1.25)
            from PyQt6.QtWidgets import QHBoxLayout
            self.assertIsInstance(panel.manual_stack.widget(0).layout(),QHBoxLayout)
            self.assertEqual(panel.manual_stack.widget(0).layout().count(),2)

    def test_hide_locator_keeps_mouse_navigation_and_removes_crosshair_artists(self):
        with tempfile.TemporaryDirectory() as folder:
            _,t1=fixture(folder);self.viewer.execute_base_mri_load(str(t1));self.viewer.update_2d_views()
            self.assertTrue(all(len(canvas.ax.lines)==2 for canvas in self.viewer.canvases.values()))
            self.viewer.chk_show_locator.setChecked(False)
            self.assertTrue(all(len(canvas.ax.lines)==0 and len(canvas.ax.collections)==0 for canvas in self.viewer.canvases.values()))
            self.viewer.handle_canvas_click('axial',1,2)
            self.assertEqual(self.viewer.sliders['X'].value(),1);self.assertEqual(self.viewer.sliders['Y'].value(),2)
            self.assertFalse(self.viewer.engine.show_locator)

    def test_target_crop_updates_two_and_three_dimensional_fields_without_showing_roi(self):
        with tempfile.TemporaryDirectory() as folder:
            _,t1=fixture(folder);image=nib.load(t1);self.viewer.execute_base_mri_load(str(t1))
            shape=image.shape;values=np.linspace(.1,1.,np.prod(shape),dtype=np.float32).reshape(shape)
            field=Path(folder)/'field.nii.gz';nib.save(nib.Nifti1Image(values,image.affine),field)
            self.viewer.engine.load_overlay(str(field),kind='field');self.viewer.refresh_layer_list_ui(schedule=False)
            atlas=np.zeros(shape,np.int16);atlas[1:4,:,:]=41;atlas[4:,:,:]=42
            path=Path(folder)/'atlas.nii.gz';nib.save(nib.Nifti1Image(atlas,image.affine),path)
            self.viewer.set_target_roi({'mask':str(path),'labels':[41]},'靶区轮廓 · AAL3')
            mask=atlas==41;self.viewer.chk_show_roi.setChecked(False)
            with patch.object(self.viewer,'update_2d_views',wraps=self.viewer.update_2d_views) as render2d,patch.object(self.viewer.canvas_3d,'render_3d') as render3d:
                self.viewer.chk_apply_mask.setChecked(True)
                self.assertTrue(render2d.called);self.assertTrue(render3d.called)
                volume=render3d.call_args.kwargs['overlays'][0]['volume']
                self.assertTrue(np.isnan(volume[~mask]).all());self.assertTrue(np.isfinite(volume[mask]).all())
                params=self.viewer.get_render_params()
                self.assertEqual(len(params[-1]),1);np.testing.assert_array_equal(params[3],mask)
                canvas=self.viewer.canvases['axial'];z=self.viewer.sliders['Z'].value()
                display_mask=canvas._format_slice_for_display(mask[:,:,z])
                self.assertTrue(np.ma.getmaskarray(canvas.ax.images[-1].get_array())[~display_mask].all())
                self.viewer.chk_apply_mask.setChecked(False)
                self.assertTrue(np.isfinite(render3d.call_args.kwargs['overlays'][0]['volume']).all())
            self.viewer.layer_list.setCurrentRow(1)
            self.assertTrue(self.viewer.display_settings.isHidden());self.assertTrue(self.viewer.mask_panel.isHidden())

    def test_single_scan_display_point_selection_and_roi_crop_keep_native_coordinates(self):
        from scipy.spatial.transform import Rotation
        with tempfile.TemporaryDirectory() as folder:
            shape=(19,23,27);affine=np.eye(4)
            affine[:3,:3]=Rotation.from_euler('x',3,degrees=True).as_matrix()@np.diag([1.2,.9375,.9375])
            affine[:3,3]=[-11,-12,-13]
            path=Path(folder)/'T1.nii';nib.save(nib.Nifti1Image(np.ones(shape,np.float32)*100,affine),path)
            self.viewer.execute_base_mri_load(str(path))
            self.assertFalse(hasattr(self.viewer,'view_direction'))
            self.assertIs(self.viewer.engine.view.data,self.viewer.engine.data)
            np.testing.assert_array_equal(self.viewer.engine.view.affine,self.viewer.engine.affine)
            self.assertEqual(self.viewer.engine.view.mode,'original')
            ras=(affine@np.r_[9,11,13,1])[:3]
            self.viewer.locate_roi({'center_ras_mm':ras.tolist(),'radius_mm':3})
            np.testing.assert_allclose(self.viewer.current_cursor_ras(),ras,atol=1.)
            panel=self.viewer.individual_dialog;panel.task={'padded_t1':str(path)}
            panel.use_crosshair()
            np.testing.assert_allclose([control.value() for control in panel.center],self.viewer.current_cursor_ras(),atol=.051)
            field=Path(folder)/'field.nii';nib.save(nib.Nifti1Image(np.ones(shape,np.float32),affine),field)
            self.viewer.engine.load_overlay(str(field),kind='field');self.viewer.refresh_layer_list_ui(schedule=False)
            self.viewer.set_target_roi({'center_ras_mm':ras.tolist(),'radius_mm':3})
            self.viewer.chk_show_roi.setChecked(False)
            with patch.object(self.viewer.canvas_3d,'render_3d') as render:
                self.viewer.chk_apply_mask.setChecked(True)
                mask=self.viewer.get_render_params()[3]
                values=render.call_args.kwargs['overlays'][0]['volume']
                self.assertTrue(np.isnan(values[~mask]).all());self.assertTrue(np.isfinite(values[mask]).all())
                np.testing.assert_allclose(render.call_args.kwargs['affine'],self.viewer.engine.view.affine)
            self.assertIs(self.viewer.engine.view.data,self.viewer.engine.data)
            cursor=self.viewer.current_cursor_ras().copy();volume=self.viewer.engine.view.data
            self.viewer.reset_3d_button.click()
            np.testing.assert_array_equal(self.viewer.current_cursor_ras(),cursor)
            self.assertIs(self.viewer.engine.view.data,volume)
            for axis in ('X','Y','Z'):self.assertEqual(self.viewer.sliders[axis].maximum(),shape[('X','Y','Z').index(axis)]-1)


if __name__ == '__main__':
    unittest.main()
