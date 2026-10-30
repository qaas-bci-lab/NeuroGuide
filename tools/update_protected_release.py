"""Publish a verified DICOM update and refresh ZIP entries without recompressing models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_protected import ROOT, RELEASE, audit_distribution, digest
from tools.update_release_manual import replace_members


def crc(path):
    value = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value = zlib.crc32(block, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    candidate, validation = args.candidate.resolve(), args.validation.resolve()
    stage = ROOT / 'build/protected_V4_5'
    manifest = json.loads((stage / 'native/protection_manifest.json').read_text(encoding='utf-8'))
    native_tests = json.loads((stage / 'native_test_result.json').read_text(encoding='utf-8'))
    assert native_tests['successful']
    dicom = json.loads((validation / 'validation/dicom_check.json').read_text(encoding='utf-8'))
    smoke = json.loads((validation / 'smoke/validation.json').read_text(encoding='utf-8'))
    assert Path(smoke['distribution']).resolve() == candidate
    assert (validation / 'validation/dicom.ready').read_text() == 'ready'
    assert dicom['dicom_import_complete'] and dicom['physical_geometry_verified'] and dicom['prediction_not_started']
    assert smoke['console_startup'] and smoke['gui_startup'] and smoke['gpu_probe'] and smoke['hidden_gpu_probe']
    audit_distribution(candidate, manifest)
    manual = candidate / '使用说明.txt'
    manual.write_text((ROOT / 'docs/使用说明_V4.5_医生版.txt').read_text(encoding='utf-8-sig'), encoding='utf-8-sig')
    # Refresh third party notices fetched after PyInstaller's resource scan.
    shutil.copytree(ROOT / 'assets/dcm2niix', candidate / '_internal/assets/dcm2niix', dirs_exist_ok=True)
    distribution = ROOT / 'dist' / RELEASE
    archive_path = ROOT / 'dist' / (RELEASE + '.zip')
    backup = validation / 'previous_release_files'
    changes = {}
    with zipfile.ZipFile(archive_path) as archive:
        old = {item.filename: item for item in archive.infolist()}
        for source in sorted(candidate.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(candidate)
            if relative.parts[0].casefold() == 'neuroguide' or relative.name.casefold() == 'patient_records.json':
                continue
            member = (Path(RELEASE) / relative).as_posix()
            info = old.get(member)
            target = distribution / relative
            if (info and source.stat().st_size == info.file_size and crc(source) == info.CRC
                    and target.is_file() and target.stat().st_size == info.file_size and crc(target) == info.CRC):
                continue
            if target.exists():
                previous = backup / relative
                previous.parent.mkdir(parents=True, exist_ok=True)
                if not previous.exists():
                    shutil.copy2(target, previous)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.update-tmp')
            shutil.copy2(source, temporary); temporary.replace(target)
            changes[member] = target
    audit = audit_distribution(distribution, manifest)
    print('UPDATED DISTRIBUTION', len(changes), 'files; updating ZIP', flush=True)
    replace_members(archive_path, changes)
    with zipfile.ZipFile(archive_path) as archive:
        files = archive.infolist()
        assert not any('/NeuroGuide/' in f.filename or f.filename.endswith('/patient_records.json') for f in files)
        total = sum(f.file_size for f in files)
    record = {'version': '4.5', 'created_at': datetime.now(timezone.utc).isoformat(),
        'directory': str(distribution), 'exe': str(distribution / (RELEASE + '.exe')),
        'exe_sha256': digest(distribution / (RELEASE + '.exe')),
        'worker_sha256': digest(distribution / 'NeuroGuideWorker.exe'),
        'zip': str(archive_path), 'zip_sha256': digest(archive_path),
        'zip_bytes': archive_path.stat().st_size, 'files': len(files), 'distribution_bytes': total,
        'native_modules': audit['native_modules_verified'], 'zip_crc_verified': True,
        'manual': manual.name, 'manual_sha256': digest(distribution / manual.name),
        'dicom_import_verified': True, 'dicom_validation': str(validation),
        'updated_files': list(changes)}
    for path in (stage / 'artifacts.json', ROOT / 'docs/release_V4_5_artifacts.json'):
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    archive_path.with_suffix('.zip.sha256').write_text(record['zip_sha256'] + '  ' + archive_path.name + '\n', encoding='ascii')
    acceptance_path = ROOT / 'docs/release_V4_5_validation.json'
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    acceptance.update(protection_audit=audit, native_tests=native_tests,
        dicom_update={'created_at': record['created_at'], 'standalone_import': dicom,
                      'standalone_smoke': smoke, 'exe_sha256': record['exe_sha256'],
                      'worker_sha256': record['worker_sha256'],
                      'data_scope': 'Synthetic classic MR DICOM; vendor clinical data not tested.'})
    acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding='utf-8')
    (stage / 'protection_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('PROTECTED DICOM RELEASE UPDATED; ZIP CRC VERIFIED', archive_path, flush=True)


if __name__ == '__main__':
    main()
