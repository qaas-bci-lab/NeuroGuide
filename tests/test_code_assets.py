import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import tempfile
import unittest

from core.code_assets import code_asset_path
from core.individual_pipeline import file_hash, require_asset


class CodeAssetTests(unittest.TestCase):
    def test_native_distribution_hashes_actual_binary_and_retains_logical_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            logical = Path(directory) / 'BrainLFNet/brainlfnet/model.py'
            native = Path(directory) / 'brainlfnet' / ('model' + EXTENSION_SUFFIXES[0])
            native.parent.mkdir(parents=True)
            native.write_bytes(b'native binary v1')
            self.assertEqual(code_asset_path(logical), native)
            self.assertEqual(require_asset(logical), logical)
            first = file_hash(logical)
            self.assertEqual(first, hashlib.sha256(native.read_bytes()).hexdigest())
            native.write_bytes(b'native binary v2')
            self.assertNotEqual(file_hash(logical), first)

    def test_source_run_uses_source_and_missing_recipe_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'recipe.py'
            native = source.with_name('recipe' + EXTENSION_SUFFIXES[0])
            native.write_bytes(b'native')
            source.write_bytes(b'source')
            self.assertEqual(code_asset_path(source), source)
            source.unlink()
            self.assertEqual(code_asset_path(source), native)
            native.unlink()
            with self.assertRaises(FileNotFoundError):
                require_asset(source)
