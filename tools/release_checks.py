"""Hidden startup checks used to validate the actual distributed GUI."""
import json
from pathlib import Path


def check_dicom_import(viewer, config):
    """Exercise the actual import button, background jobs and MRI load in an EXE."""
    import time
    import numpy as np
    import nibabel as nib
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from ui.dicom_import import DicomImportDialog
    from core.dicom_import import converter_path
    viewer.on_individual_mode_clicked()
    panel = viewer.individual_dialog
    if config.get('output_root'):
        panel.storage_root = Path(config['output_root']) / 'individualized'
    original = QFileDialog.getExistingDirectory
    original_question = QMessageBox.question
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: config['dicom_folder']
    phase, errors = ['find'], []
    deadline = time.monotonic() + 120
    timer = QTimer(viewer); timer.setInterval(30)
    def advance():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError('DICOM GUI check timed out: ' + phase[0])
            dialog = next((w for w in QApplication.topLevelWidgets() if isinstance(w, DicomImportDialog) and w.isVisible()), None)
            if dialog is None:
                return
            if phase[0] == 'find':
                phase[0] = 'scan'; dialog.browse.click()
            elif phase[0] == 'scan' and dialog.job is None:
                assert len(dialog.records) == 2, dialog.status.text()
                assert dialog.selected_record() is None and not dialog.convert.isEnabled()
                index = next(i for i, r in enumerate(dialog.records) if r['number'] == '1')
                phase[0] = 'preview'; dialog.table.selectRow(index)
            elif phase[0] == 'preview' and dialog.job is None:
                assert not dialog.confirm.isChecked() and not dialog.convert.isEnabled()
                if config.get('dialog_screenshot'):
                    dialog.grab().save(config['dialog_screenshot'])
                dialog.confirm.setChecked(True)
                assert dialog.convert.isEnabled()
                phase[0] = 'convert'; dialog.convert.click()
            elif phase[0] == 'convert' and dialog.job is None and dialog.selected_path is None:
                raise RuntimeError('Frozen DICOM conversion failed: ' + dialog.status.text())
        except Exception as exc:
            errors.append(str(exc)); timer.stop()
            if dialog is not None:
                dialog.reject()
    timer.timeout.connect(advance); timer.start()
    try:
        panel.dicom_button.click()
    finally:
        timer.stop(); QFileDialog.getExistingDirectory = original
        QMessageBox.question = original_question
    assert not errors, errors
    assert phase[0] == 'convert'
    assert panel.t1.text() and 'dicom_imports' in panel.t1.text()
    # Model inspection is allowed, but prediction requires the doctor's next click.
    while panel.process is not None and time.monotonic() < deadline:
        QApplication.processEvents(); time.sleep(.02)
    assert panel.process is None
    assert panel.task is None and panel._action == 'inspect'
    image = nib.load(panel.t1.text()); values = image.get_fdata()
    expected = np.load(config['expected'])
    np.testing.assert_array_equal(np.sort(values.ravel()), np.sort(expected.ravel()))
    ijk = np.indices(values.shape).reshape(3, -1).T
    lps = nib.affines.apply_affine(image.affine, ijk) * [-1, -1, 1]
    original_ijk = np.rint((lps - [10, 20, 30]) / [.8, 1.2, 2]).astype(int)
    np.testing.assert_array_equal(values.ravel(), expected[original_ijk[:, 2], original_ijk[:, 1], original_ijk[:, 0]])
    if config.get('screenshot'):
        viewer.resize(1440, 1000); viewer.grab().save(config['screenshot'])
    Path(config['check_record']).write_text(json.dumps({
        'dicom_import_complete': True, 'explicit_series_confirmation': True,
        'only_selected_series_converted': True, 'physical_geometry_verified': True,
        'prediction_not_started': True, 'nifti': panel.t1.text(),
        'bundled_converter': str(converter_path()),
    }, indent=2), encoding='utf-8')


def check_field_result(viewer, config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('dicom_folder'):
        return check_dicom_import(viewer, config)
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import QEventLoop, QTimer
    from tools.report_page import generate_report
    from ui.electrode_map import electrode_dialog
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    result=json.loads(Path(config['result_file']).read_text(encoding='utf-8'))
    def fail(title, text, *args, **kwargs):
        raise RuntimeError(f'{title}: {text}')
    QMessageBox.warning=fail
    QMessageBox.information=fail
    viewer.resize(1440,1080)
    viewer.on_individual_mode_clicked()
    panel=viewer.individual_dialog
    manifest=Path(result['prediction_dir']).parent/'manifest.json'
    panel.attach_task(panel.task_from_manifest(manifest))
    panel.apply_result(result)
    viewer.on_individual_field_ready(result)
    assert viewer._individual_result is result
    assert not hasattr(viewer,"view_direction")
    assert viewer.engine.view.mode=="original"
    assert viewer.engine.view.data is viewer.engine.data
    tes=result.get('stimulation_type')=='tES'
    assert panel.is_tes()==tes
    assert viewer.engine.overlays[0]['cmap']=='jet'
    for current in result.get('currents_ma',result.get('currents_peak_ma',[])):
        assert f'{abs(float(current)):.4f}' in viewer.current_label.text() if tes else f'{float(current):g}' in viewer.current_label.text()
    dialog=electrode_dialog(viewer,panel.task['electrodes']['electrode_coordinates'],result)
    expected=len(result['electrode_labels']) if tes else 4
    assert len(dialog.electrode_currents)==expected
    from ui.individual import app_root
    assert panel.storage_root == app_root()/'NeuroGuide/individualized'
    assert panel.population.value()==256 and panel.generations.value()==160
    peak=viewer.engine.overlays[0]['max']
    assert abs(viewer.engine.overlays[0]['display_min']-.05*peak)<1e-5
    assert abs(viewer.engine.overlays[0]['display_max']-.95*peak)<1e-5
    viewer.chk_show_locator.setChecked(False)
    assert all(len(canvas.ax.lines)==0 for canvas in viewer.canvases.values())
    panel.tabs.setCurrentIndex(2)
    assert panel.tabs.widget(2).isAncestorOf(panel.shared_target_controls)
    panel.tabs.setCurrentIndex(1)
    assert panel.tabs.widget(1).isAncestorOf(panel.shared_target_controls)
    viewer.chk_show_roi.setChecked(False);viewer.chk_apply_mask.setChecked(True)
    rendered=viewer.get_render_params()
    assert rendered[3] is not None and rendered[3].any()
    assert all(not ov.get('is_roi') for ov in rendered[-1])
    viewer.chk_apply_mask.setChecked(False);viewer.chk_show_roi.setChecked(True)
    if not tes:
        tags=[text.get_text() for text in dialog.figure_canvas.figure.axes[0].texts]
        assert all(any(tag in text for text in tags) for tag in ('1+','1-','2+','2-'))
    panel.tabs.setCurrentIndex(2)
    success,error=generate_report(viewer.engine,result['padded_t1'],'','个体靶区',config['report'],
                                 patient_info={'name':'发布验证','stimulation':result})
    assert success,error
    loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec()
    # Verify real VTK planes use the display grid; prediction space stays native.
    widget=viewer.canvas_3d
    assert widget.available and widget._plotter is not None
    viewer.update_3d_view()
    affine=viewer.engine.view.affine
    import numpy as np
    import nibabel as nib
    from ui.volume_geometry import slice_plane_pose
    assert np.allclose(widget._affine,affine,atol=1e-8)
    for axis,name in enumerate(("sagittal","coronal","axial")):
        mapper=widget._plotter.actors["plane_"+name].mapper
        mapper.Update()
        points=mapper.dataset.points
        assert len(points)>0, "VTK slice mesh must be materialized"
        voxel=nib.affines.apply_affine(np.linalg.inv(affine),points)
        assert np.allclose(voxel[:,axis],widget._slice[("x","y","z")[axis]],atol=1e-5)
    original=tuple(widget._slice[name] for name in ("x","y","z"))
    next_x=(original[0]+1)%viewer.engine.view.shape[0]
    origin,normal=slice_plane_pose(viewer.engine.view.shape,affine,0,next_x)
    widget._on_drag("x",normal,origin)
    assert widget._slice["x"]==next_x
    origin,normal=slice_plane_pose(viewer.engine.view.shape,affine,0,original[0])
    widget._on_drag("x",normal,origin)
    assert widget._slice["x"]==original[0]
    # Materialize lazy VTK inputs after drag replacement, before comparing.
    for name in ("sagittal","coronal","axial"):
        widget._plotter.actors["plane_"+name].mapper.Update()
    # Rotate the camera, then exercise the user-visible reset button.
    cursor_before=viewer.current_cursor_ras().copy()
    slices_before=widget._slice.copy()
    plane_points={name:widget._plotter.actors["plane_"+name].mapper.dataset.points.copy()
                  for name in ("sagittal","coronal","axial")}
    widget._plotter.camera.azimuth+=27
    viewer.reset_3d_button.click()
    from ui.volume_geometry import volume_camera_pose
    pose=volume_camera_pose(viewer.engine.view.shape,affine)
    camera=widget._plotter.camera
    assert np.allclose(np.asarray(camera.up)/np.linalg.norm(camera.up),pose[2],atol=1e-8)
    wanted=np.asarray(pose[0])-np.asarray(pose[1]);wanted/=np.linalg.norm(wanted)
    actual=np.asarray(camera.position)-np.asarray(camera.focal_point);actual/=np.linalg.norm(actual)
    assert np.allclose(actual,wanted,atol=1e-8)
    assert slices_before==widget._slice
    assert np.allclose(viewer.current_cursor_ras(),cursor_before,atol=1e-10)
    for name,points in plane_points.items():
        mapper=widget._plotter.actors["plane_"+name].mapper;mapper.Update()
        after_points=mapper.dataset.points
        assert len(points)>0 and np.array_equal(points,after_points), name
    if config.get('screenshot'):
        viewer.grab().save(config['screenshot'])
    if config.get('electrode_screenshot'):
        dialog.grab().save(config['electrode_screenshot'])
    check={'version':QApplication.applicationVersion(),'mode':'tES' if tes else 'TI',
           'defaults':{'tes_count':panel.tes_count.value(),'tes_cap':panel.tes_cap.value(),'tes_total':panel.tes_total.value()},
           'current_text':viewer.current_label.text(),'overlay':viewer.engine.overlays[0]['name'],
           'electrode_map_count':len(dialog.electrode_currents),'report':config['report'],
           'report_success':True,'gui_loaded':True,'storage_root':str(panel.storage_root),
           'display_3d_grid_verified':True,'physical_drag_verified':True,
           'scan_default_verified':True,'camera_reset_preserves_planes_verified':True,
           'display_direction':'native_scan','display_direction_selector_removed':True,
           'display_shape':list(viewer.engine.view.shape),'native_shape':list(viewer.engine.data.shape),
           'display_transform':viewer.engine.view.native_to_view.tolist(),
           'display_affine':viewer.engine.view.affine.tolist(),
           'shared_target_pages_verified':True,'hidden_locator_verified':True,'roi_clip_verified':True,
           'search_defaults':{'population':panel.population.value(),'generations':panel.generations.value(),'samples':panel.sample_size.value(),'finalists':panel.finalists.value(),'ratio_steps':panel.ratio_steps.value()}}
    Path(config['check_record']).write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    dialog.close()
