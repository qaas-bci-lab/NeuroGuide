"""Replace the release manual without recompressing unchanged runtime binaries."""
from __future__ import annotations
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_protected import ROOT, RELEASE, digest


def replace_member(archive_path, member, replacement):
    with zipfile.ZipFile(archive_path) as archive:
        if sum(info.filename == member for info in archive.infolist()) != 1:
            raise ValueError('Expected exactly one manual entry in the release ZIP')
    replace_members(archive_path, {member: replacement})


def replace_members(archive_path, replacements):
    """Copy intact local ZIP records; rebuild offsets using Python's ZIP writer."""
    temporary = archive_path.with_name(archive_path.stem + '.manual-update.tmp.zip')
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(archive_path) as original, zipfile.ZipFile(
            temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as updated:
        infos = sorted(original.infolist(), key=lambda info: info.header_offset)
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError('Duplicate ZIP entries')
        updated.comment = original.comment
        for index, info in enumerate(infos):
            if info.filename in replacements:
                continue
            end = infos[index + 1].header_offset if index + 1 < len(infos) else original.start_dir
            original.fp.seek(info.header_offset)
            new_info = copy.copy(info)
            new_info.header_offset = updated.fp.tell()
            remaining = end - info.header_offset
            while remaining:
                block = original.fp.read(min(8 * 1024**2, remaining))
                if not block:
                    raise EOFError('Truncated source ZIP record: ' + info.filename)
                updated.fp.write(block)
                remaining -= len(block)
            updated.filelist.append(new_info)
            updated.NameToInfo[new_info.filename] = new_info
            updated.start_dir = updated.fp.tell()
            if index % 1000 == 0:
                print('COPIED', index, '/', len(infos), flush=True)
        for member, replacement in replacements.items():
            updated.write(replacement, member)
    with zipfile.ZipFile(archive_path) as original, zipfile.ZipFile(temporary) as updated:
        old = {info.filename: info for info in original.infolist()}
        new = {info.filename: info for info in updated.infolist()}
        assert len(updated.infolist()) == len(new)
        assert set(old) | set(replacements) == set(new)
        for name in old.keys() - replacements.keys():
            assert (old[name].CRC, old[name].compress_size, old[name].file_size) == (
                new[name].CRC, new[name].compress_size, new[name].file_size), name
        for member, replacement in replacements.items():
            assert updated.read(member) == replacement.read_bytes()
        print('VERIFYING ZIP CRC', flush=True)
        bad = updated.testzip()
        if bad:
            raise RuntimeError('ZIP CRC mismatch: ' + bad)
    # Replace only after the new archive has passed all checks.
    temporary.replace(archive_path)


def main():
    stage = ROOT / 'build/protected_V4_5'
    distribution = ROOT / 'dist' / RELEASE
    archive_path = ROOT / 'dist' / (RELEASE + '.zip')
    manual = distribution / '使用说明.txt'
    canonical = ROOT / 'docs/使用说明_V4.5_医生版.txt'
    manual.write_text(canonical.read_text(encoding='utf-8-sig'), encoding='utf-8-sig')
    record = json.loads((ROOT / 'docs/release_V4_5_artifacts.json').read_text(encoding='utf-8'))
    # A documentation refresh must preserve the accepted executables.
    assert digest(distribution / (RELEASE + '.exe')) == record['exe_sha256']
    assert digest(distribution / 'NeuroGuideWorker.exe') == record['worker_sha256']
    member = RELEASE + '/使用说明.txt'
    with zipfile.ZipFile(archive_path) as archive:
        (stage / 'previous_release_manual.txt').write_bytes(archive.read(member))
    replace_member(archive_path, member, manual)
    record.update(zip_sha256=digest(archive_path), zip_bytes=archive_path.stat().st_size,
                  manual='使用说明.txt', manual_sha256=digest(manual),
                  manual_updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  zip_crc_verified=True)
    with zipfile.ZipFile(archive_path) as archive:
        record['files'] = len(archive.infolist())
        record['distribution_bytes'] = sum(info.file_size for info in archive.infolist())
    for path in (ROOT / 'docs/release_V4_5_artifacts.json', stage / 'artifacts.json'):
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    archive_path.with_suffix('.zip.sha256').write_text(
        record['zip_sha256'] + '  ' + archive_path.name + '\n', encoding='ascii')
    print('DOCTOR MANUAL UPDATED; ZIP CRC VERIFIED', archive_path, flush=True)


if __name__ == '__main__':
    main()
