"""Archive an audited and accepted protected release, excluding user task data."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_protected import audit_distribution, digest, RELEASE, ROOT, STAGE, VERSION, VERSION_TAG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--distribution', type=Path, default=ROOT / 'dist' / RELEASE)
    args = parser.parse_args()
    stage, dist = args.stage.resolve(), args.distribution.resolve()
    validation = json.loads((stage / 'validation/validation.json').read_text(encoding='utf-8'))
    if not validation.get('validation_complete') or Path(validation['distribution']).resolve() != dist:
        raise RuntimeError('Full executable acceptance is required before archiving.')
    manifest = json.loads((stage / 'native/protection_manifest.json').read_text(encoding='utf-8'))
    audit = audit_distribution(dist, manifest)
    (stage / 'protection_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    instructions = (ROOT / 'docs' / f'使用说明_V{VERSION}_医生版.txt').read_text(encoding='utf-8-sig')
    (dist / '使用说明.txt').write_text(instructions, encoding='utf-8-sig')
    files = []
    for path in sorted(dist.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(dist)
        if relative.parts[0].casefold() == 'neuroguide' or relative.name.casefold() == 'patient_records.json':
            continue
        files.append((path, relative))
    archive_path = dist.parent / (RELEASE + '.zip')
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with TemporaryDirectory(prefix=RELEASE + '-', dir=dist.parent) as temporary:
        staged_archive = Path(temporary) / archive_path.name
        with zipfile.ZipFile(staged_archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for index, (path, relative) in enumerate(files):
                archive.write(path, (Path(RELEASE) / relative).as_posix())
                if index % 1000 == 0:
                    print('ARCHIVE', index, '/', len(files), flush=True)
        with zipfile.ZipFile(staged_archive) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError('ZIP CRC mismatch: ' + bad)
            assert len(archive.infolist()) == len(files)
        staged_archive.rename(archive_path)
    record = {'version': VERSION, 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'directory': str(dist), 'exe': str(dist / (RELEASE + '.exe')),
              'exe_sha256': digest(dist / (RELEASE + '.exe')),
              'worker_sha256': digest(dist / 'NeuroGuideWorker.exe'),
              'zip': str(archive_path), 'zip_sha256': digest(archive_path),
              'zip_bytes': archive_path.stat().st_size, 'files': len(files),
              'distribution_bytes': sum(path.stat().st_size for path, _ in files),
              'native_modules': audit['native_modules_verified'], 'zip_crc_verified': True}
    for path in (stage / 'artifacts.json', ROOT / 'docs' / f'release_{VERSION_TAG}_artifacts.json'):
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    archive_path.with_suffix('.zip.sha256').write_text(
        record['zip_sha256'] + '  ' + archive_path.name + '\n', encoding='ascii')
    acceptance = {'version': VERSION, 'protection_audit': audit, 'standalone': validation,
                  'native_tests': json.loads((stage / 'native_test_result.json').read_text(encoding='utf-8'))}
    (ROOT / 'docs' / f'release_{VERSION_TAG}_validation.json').write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ARCHIVE COMPLETE', archive_path, record['zip_bytes'], flush=True)


if __name__ == '__main__':
    main()
