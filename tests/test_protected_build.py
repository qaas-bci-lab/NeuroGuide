"""Security boundaries of the protected-release audit, without rebuilding EXEs."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    '_neuroguide_build_audit', Path(__file__).resolve().parents[1] / 'tools/build_protected.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class ProtectedBuildTests(unittest.TestCase):
    def test_native_brain_package_layout_avoids_windows_case_collision_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internal = root / '_internal'
            weights = internal / 'BrainLFNet/checkpoints/model.pt'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'unchanged weights')
            binary = internal / 'brainlfnet/model.cp310-win_amd64.pyd'
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b'native model')
            item = {'module': 'brainlfnet.model', 'binary': 'brainlfnet/model.cp310-win_amd64.pyd',
                    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
            manifest = {'modules': [item]}
            moved = build.arrange_native_packages(root, manifest)
            self.assertEqual(len(moved), 1)
            target = internal / 'BrainLFNet/brainlfnet/model.cp310-win_amd64.pyd'
            self.assertEqual(target.read_bytes(), b'native model')
            self.assertFalse(binary.exists())
            self.assertEqual(weights.read_bytes(), b'unchanged weights')
            self.assertEqual(build.arrange_native_packages(root, manifest), [])

    def fixture(self, root, bytecode=None):
        internal = root / '_internal'
        binary = internal / 'core/recipe.cp310-win_amd64.pyd'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'fixture native module')
        manifest = {'modules': [{'module': 'core.recipe', 'binary': 'core/recipe.cp310-win_amd64.pyd',
                                  'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}]}
        archive = SimpleNamespace(toc={'PYZ.pyz': (0, 'z')},
                                  open_embedded_archive=lambda _: SimpleNamespace(toc=bytecode or {}))
        return internal, binary, manifest, archive

    def test_native_only_distribution_passes_and_modified_binary_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, binary, manifest, archive = self.fixture(root)
            with patch('PyInstaller.archive.readers.CArchiveReader', return_value=archive):
                self.assertTrue(build.audit_distribution(root, manifest)['private_sources_absent'])
                binary.write_bytes(b'changed native module')
                with self.assertRaisesRegex(RuntimeError, 'Missing/changed binary'):
                    build.audit_distribution(root, manifest)

    def test_plaintext_source_and_executable_bytecode_copies_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internal, _, manifest, archive = self.fixture(root)
            with patch('PyInstaller.archive.readers.CArchiveReader', return_value=archive):
                source = internal / 'neuroguide_app.py'
                source.write_text('private application code')
                with self.assertRaisesRegex(RuntimeError, 'Private application source'):
                    build.audit_distribution(root, manifest)
                source.unlink()
                for module in ('core.recipe', 'core.new_module'):
                    archive.open_embedded_archive = lambda _: SimpleNamespace(toc={module: (0, 0, 0)})
                    with self.assertRaisesRegex(RuntimeError, 'private bytecode in PYZ'):
                        build.audit_distribution(root, manifest)

    def test_uninspectable_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, manifest, archive = self.fixture(root)
            archive.toc = {}
            with patch('PyInstaller.archive.readers.CArchiveReader', return_value=archive):
                with self.assertRaisesRegex(RuntimeError, 'inspectable PYZ'):
                    build.audit_distribution(root, manifest)
