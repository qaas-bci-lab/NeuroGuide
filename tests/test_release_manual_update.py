import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location(
    '_release_manual_update', Path(__file__).resolve().parents[1] / 'tools/update_release_manual.py')
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


class ManualUpdateTests(unittest.TestCase):
    def test_updates_multiple_entries_and_adds_converter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / 'release.zip'
            replacement = root / 'new'; replacement.write_bytes(b'new native binary')
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('app/keep', b'unchanged')
                output.writestr('app/update', b'old')
            updater.replace_members(archive, {'app/update': replacement, 'app/converter': replacement})
            with zipfile.ZipFile(archive) as updated:
                self.assertEqual(updated.read('app/keep'), b'unchanged')
                self.assertEqual(updated.read('app/update'), replacement.read_bytes())
                self.assertEqual(updated.read('app/converter'), replacement.read_bytes())
                self.assertEqual(len(updated.infolist()), 3)

    def test_replaces_one_entry_and_preserves_other_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, replacement = root / 'release.zip', root / 'manual.txt'
            replacement.write_text('医生操作流程：影像 → 靶区 → 方案 → 报告', encoding='utf-8-sig')
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr('app/before.bin', b'before' * 1000)
                output.writestr('app/使用说明.txt', 'old')
                # Exercise a local ZIP64 header and an entry after the replaced file.
                with output.open('app/after.bin', 'w', force_zip64=True) as binary:
                    binary.write(b'after' * 1000)
                output.comment = b'preserved archive comment'
            updater.replace_member(archive, 'app/使用说明.txt', replacement)
            with zipfile.ZipFile(archive) as updated:
                self.assertEqual(len(updated.infolist()), 3)
                self.assertEqual(updated.read('app/使用说明.txt'), replacement.read_bytes())
                self.assertEqual(updated.read('app/before.bin'), b'before' * 1000)
                self.assertEqual(updated.read('app/after.bin'), b'after' * 1000)
                self.assertEqual(updated.comment, b'preserved archive comment')
