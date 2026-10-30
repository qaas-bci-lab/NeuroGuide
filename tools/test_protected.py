"""Run the existing regression suite against native modules, never source fallbacks."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=root / 'build/protected_V4_5/native')
    args = parser.parse_args()
    native = args.native.resolve()
    sys.path.insert(0, str(native))
    manifest = json.loads((native / 'protection_manifest.json').read_text(encoding='utf-8'))
    # Pin package search paths before tests add their own source paths to sys.path.
    for name in ('core', 'ui', 'tools', 'brainlfnet', 'SegandElec', 'SegandElec_nnUNet'):
        package = importlib.import_module(name)
        assert Path(package.__file__).is_relative_to(native), (name, package.__file__)
    app = importlib.import_module('neuroguide_app')
    sys.modules['main'] = app
    # This build-time resource utility is deliberately excluded from the EXE.
    for name in ('package_resources', 'build_protected'):
        spec = importlib.util.spec_from_file_location('tools.' + name, root / 'tools' / (name + '.py'))
        utility = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = utility
        spec.loader.exec_module(utility)
    # The native staging tree contains code only. Reuse original read-only data
    # for this pre-freeze test; the standalone tests verify packaged resources.
    sys._MEIPASS = str(root)
    individual = importlib.import_module('ui.individual')
    individual.app_root = lambda: root
    individual.worker_command = lambda _root, config: (
        sys.executable, ['-u', str(native / 'launcher.py'), '--individual-worker', str(config)])
    from PyQt6.QtWidgets import QMessageBox
    def unexpected_message(_parent, title, message, *args, **kwargs):
        raise RuntimeError(f'Unexpected GUI message: {title}: {message}')
    QMessageBox.warning = unexpected_message
    QMessageBox.critical = unexpected_message
    individual.IndividualPanel.show_error = unexpected_message
    slot_errors = []
    def slot_exception(kind, value, trace):
        slot_errors.append(str(value))
        traceback.print_exception(kind, value, trace)
    sys.excepthook = slot_exception
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checked = []
    for item in manifest['modules']:
        module = sys.modules.get(item['module'])
        if module is not None:
            path = Path(module.__file__)
            assert path.is_relative_to(native) and path.suffix == '.pyd', (item['module'], path)
            checked.append(item['module'])
    print('NATIVE IMPORTS VERIFIED', len(checked), flush=True)
    (native.parent / 'native_test_result.json').write_text(json.dumps({
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'successful': result.wasSuccessful() and not slot_errors,
        'slot_errors': slot_errors, 'verified_native_imports': checked,
    }, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() and not slot_errors else 1)


if __name__ == '__main__':
    main()
