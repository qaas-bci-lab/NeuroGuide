"""Mixed-series import, real converter geometry, and explicit doctor confirmation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PyQt6.QtWidgets import QApplication

from core.dicom_import import scan_series, convert_series, preview_series
from ui.dicom_import import DicomImportDialog


def write_series(root, number=1, offset=0, patient='SYNTHETIC'):
    """Generated pixels only; never real patient data."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    values = (np.arange(12 * 20 * 24).reshape(12, 20, 24) + offset).astype(np.int16)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((.8, 1.2, 2.)); image.SetOrigin((10., 20., 30.))
    writer = sitk.ImageFileWriter(); writer.SetImageIO('GDCMImageIO'); writer.KeepOriginalImageUIDOn()
    for index in range(12):
        part = image[:, :, index]
        tags = {'0008|0016': '1.2.840.10008.5.1.4.1.1.4', '0008|0018': f'1.2.826.0.1.3680043.2.1125.456.{number}.{index+1}',
            '0008|0020': '20260101', '0008|0030': '120000', '0008|0060': 'MR', '0008|0070': 'SYNTHETIC',
            '0008|103e': 'T1 test' if number == 1 else 'T2 test', '0008|0008': 'ORIGINAL\\PRIMARY\\OTHER',
            '0010|0010': patient, '0010|0020': patient, '0018|0050': '2', '0018|0088': '2',
            '0018|0020': 'SE', '0018|0021': 'NONE', '0018|0022': 'NONE', '0018|0023': '2D',
            '0020|000d': '1.2.826.0.1.3680043.2.1125.123',
            '0020|000e': f'1.2.826.0.1.3680043.2.1125.124.{number}',
            '0020|0011': str(number), '0020|0013': str(index + 1),
            '0020|0032': '\\'.join(map(str, image.TransformIndexToPhysicalPoint((0, 0, index)))),
            '0020|0037': '1\\0\\0\\0\\1\\0'}
        for tag, value in tags.items():
            part.SetMetaData(tag, value)
        writer.SetFileName(str(root / f'{number}_{index:03d}.bin')); writer.Execute(part)
    return values


class DicomTests(unittest.TestCase):
    def test_mixed_folder_conversion_only_confirmed_series_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = write_series(root / 'mixed', 1)
            write_series(root / 'mixed', 2, 10000)
            result = scan_series(root / 'mixed')
            self.assertEqual(len(result['series']), 2)
            chosen = next(r for r in result['series'] if r['number'] == '1')
            self.assertEqual(len(chosen['files']), 12)
            self.assertEqual(preview_series(chosen).ndim, 2)
            converted = convert_series(chosen, root / 'outputs')
            self.assertEqual(len(converted['candidates']), 1)
            image = nib.load(converted['candidates'][0]['path'])
            actual = image.get_fdata()
            np.testing.assert_array_equal(np.sort(actual.ravel()), np.sort(expected.ravel()))
            # Check every output voxel at its physical coordinate, including signs/orientation.
            ijk = np.indices(actual.shape).reshape(3, -1).T
            ras = nib.affines.apply_affine(image.affine, ijk)
            lps = ras * [-1, -1, 1]
            original = np.rint((lps - [10, 20, 30]) / [.8, 1.2, 2]).astype(int)
            np.testing.assert_array_equal(actual.ravel(), expected[original[:, 2], original[:, 1], original[:, 0]])
            self.assertEqual(len(list((root / 'mixed').glob('*.bin'))), 24)

    def test_changed_series_rejected_and_cancel_cleans_own_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_series(root / 'input')
            record = scan_series(root / 'input')['series'][0]
            record['patient_id'] = 'WRONG'
            with self.assertRaisesRegex(ValueError, '混有'):
                convert_series(record, root / 'outputs')
            self.assertEqual(list((root / 'outputs').iterdir()), [])
            with self.assertRaises(InterruptedError):
                convert_series(record, root / 'outputs', lambda: True)
            self.assertEqual(list((root / 'outputs').iterdir()), [])

    def test_doctor_must_select_and_confirm_and_reconfirm_on_change(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            write_series(Path(directory) / 'input', 1)
            write_series(Path(directory) / 'input', 2, 10000)
            dialog = DicomImportDialog(Path(directory) / 'outputs')
            dialog.show_series(scan_series(Path(directory) / 'input'))
            self.assertIsNone(dialog.selected_record())
            self.assertFalse(dialog.convert.isEnabled())
            dialog.start_conversion()
            self.assertIsNone(dialog.job)
            for row in (0, 1):
                dialog.table.selectRow(row)
                deadline = time.monotonic() + 10
                while dialog.job is not None and time.monotonic() < deadline:
                    app.processEvents(); time.sleep(.01)
                self.assertIsNone(dialog.job)
                self.assertFalse(dialog.confirm.isChecked())
                self.assertFalse(dialog.convert.isEnabled())
                dialog.confirm.setChecked(True)
                self.assertTrue(dialog.convert.isEnabled())
            dialog.reject(); dialog.deleteLater(); app.processEvents()

    def test_multiple_outputs_require_choice_and_cancel_preserves_current_mri(self):
        app = QApplication.instance() or QApplication([])
        dialog = DicomImportDialog('unused')
        result = {'candidates': [{'path': 'a.nii.gz', 'shape': (10, 20, 30)},
                                {'path': 'b.nii.gz', 'shape': (10, 20, 30)}],
                  'rejected': [], 'output': 'outputs'}
        with patch('ui.dicom_import.QInputDialog.getItem', return_value=('', False)):
            dialog.conversion_done(result)
        self.assertIsNone(dialog.selected_path)
        with patch('ui.dicom_import.QInputDialog.getItem', return_value=('b.nii.gz  10 × 20 × 30', True)):
            dialog.conversion_done(result)
        self.assertEqual(dialog.selected_path, 'b.nii.gz')
        dialog.deleteLater(); app.processEvents()

    def test_converter_warnings_require_further_confirmation(self):
        app = QApplication.instance() or QApplication([])
        from PyQt6.QtWidgets import QMessageBox
        dialog = DicomImportDialog('unused')
        result = {'candidates': [{'path': 'a.nii.gz', 'shape': (10, 20, 30)}],
                  'rejected': [], 'output': 'outputs', 'warnings': ['Warning: Missing images']}
        with patch('ui.dicom_import.QMessageBox.question', return_value=QMessageBox.StandardButton.Cancel):
            dialog.conversion_done(result)
        self.assertIsNone(dialog.selected_path)
        dialog.deleteLater(); app.processEvents()


if __name__ == '__main__':
    unittest.main()
