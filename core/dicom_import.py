"""Local DICOM discovery and isolated dcm2niix conversion of an explicit series."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def check_cancel(cancel):
    if cancel():
        raise InterruptedError('已取消 DICOM 导入。')


def header(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO('GDCMImageIO')
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    def get(tag):
        return reader.GetMetaData(tag).strip().strip('\x00') if reader.HasMetaDataKey(tag) else ''
    return {
        'patient': get('0010|0010'), 'patient_id': get('0010|0020'),
        'study': get('0020|000d'), 'uid': get('0020|000e'),
        'date': get('0008|0020'), 'description': get('0008|103e'),
        'protocol': get('0018|1030'), 'number': get('0020|0011'),
        'modality': get('0008|0060'), 'image_type': get('0008|0008'),
        'frames': get('0028|0008') or '1',
        'matrix': ' × '.join(str(n) for n in reader.GetSize()),
    }


def identity(record):
    return record['patient_id'], record['study'], record['uid']


def scan_series(folder, cancel=lambda: False):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError('请选择包含 DICOM 文件的文件夹。')
    records, unreadable = {}, 0
    for directory, _, filenames in os.walk(folder):
        check_cancel(cancel)
        if not filenames:
            continue
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or ():
            check_cancel(cancel)
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uid)
            if not files:
                continue
            try:
                record = header(files[0])
            except RuntimeError:
                unreadable += 1
                continue
            if not record['uid'] or not record['study']:
                unreadable += 1
                continue
            key = identity(record)
            if key not in records:
                records[key] = dict(record, files=set())
            records[key]['files'].update(str(Path(p).resolve()) for p in files)
    results = []
    for record in records.values():
        record['files'] = sorted(record['files'])
        results.append(record)
    results.sort(key=lambda r: (r['patient_id'], r['date'], r['study'], r['number'], r['uid']))
    return {'series': results, 'unreadable': unreadable}


def preview_series(record, cancel=lambda: False):
    """Preview one DICOM instance; the UI explicitly labels this as a sample."""
    check_cancel(cancel)
    files = record['files']
    image = sitk.ReadImage(files[len(files) // 2])
    values = sitk.GetArrayFromImage(image)
    while values.ndim > 2:
        values = values[values.shape[0] // 2]
    if values.ndim != 2:
        raise ValueError('该序列无法生成二维预览，请核对原始影像。')
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError('预览影像没有有效像素。')
    low, high = np.percentile(finite, [1, 99])
    pixels = np.clip((np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-8), 0, 1)
    return np.ascontiguousarray(pixels * 255, dtype=np.uint8)


def converter_path():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    path = root / 'assets/dcm2niix/dcm2niix.exe'
    if not path.is_file():
        raise FileNotFoundError('缺少内置 DICOM 转换工具，请检查软件文件是否完整。')
    return path


def validate_nifti(path):
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError('该输出不是三维体积影像，不能用于当前 T1 建模。')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10:
        raise ValueError('转换影像的空间信息无效。')
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all() or np.ptp(values) <= 0:
        raise ValueError('转换影像存在无效像素或没有有效灰度变化。')
    return image.shape


def convert_series(record, output_root, cancel=lambda: False):
    if record['modality'] != 'MR':
        raise ValueError('当前建模需要 MR 的 T1 序列，不能导入其他模态。')
    executable = converter_path()
    output_root = Path(output_root).resolve()
    output = output_root / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    try:
        # Copy only the confirmed series. Never pass the original mixed folder to dcm2niix.
        # ASCII temporary paths also avoid legacy Windows converter path limitations.
        with tempfile.TemporaryDirectory(prefix='ng_dicom_') as temporary:
            source = Path(temporary) / 'input'; source.mkdir()
            target = Path(temporary) / 'output'; target.mkdir()
            for index, filename in enumerate(record['files']):
                check_cancel(cancel)
                current = header(filename)
                if identity(current) != identity(record) or current['modality'] != 'MR':
                    raise ValueError('序列中混有不同患者、检查或序列的文件，请重新整理影像。')
                shutil.copyfile(filename, source / f'{index:07d}.dcm')
            command = [str(executable), '-g', 'i', '-z', 'i', '-b', 'n', '-i', 'n',
                       '-m', 'n', '-x', 'n', '-f', 'T1_%s',
                       '-o', str(target), str(source)]
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    while process.poll() is None:
                        check_cancel(cancel)
                        time.sleep(.1)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait()
                log.seek(0)
                message = log.read().decode('utf-8', errors='replace')
            check_cancel(cancel)
            if process.returncode:
                raise RuntimeError('DICOM 转换失败：\n' + message[-3000:])
            generated = sorted(target.glob('*.nii*'))
            if not generated:
                raise ValueError('未生成可用影像。所选序列可能是定位像、派生图像或不支持的数据。')
            candidates, rejected = [], []
            for path in generated:
                check_cancel(cancel)
                try:
                    shape = validate_nifti(path)
                except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
                    rejected.append(path.name + '：' + str(exc)); continue
                destination = output / path.name
                shutil.copyfile(path, destination)
                candidates.append({'path': str(destination), 'shape': shape})
            if not candidates:
                raise ValueError('转换结果不符合三维 T1 导入要求：\n' + '\n'.join(rejected))
            (output / 'conversion.log').write_text(message, encoding='utf-8')
            warnings = [line.strip() for line in message.splitlines()
                        if line.strip().lower().startswith(('warning:', 'error:'))]
            return {'candidates': candidates, 'rejected': rejected, 'output': str(output),
                    'warnings': warnings}
    except BaseException:
        # This UUID directory was created by this call and never contains original DICOMs.
        if output.resolve().parent == output_root:
            shutil.rmtree(output)
        raise
