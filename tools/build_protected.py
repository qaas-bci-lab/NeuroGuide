"""Compile private modules, freeze V4.5, and audit both executables.

Run with the project's CUDA-compatible build interpreter. Cython and MSVC are
build dependencies only; the distribution does not need either of them.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'NeuroGuide-V4.5'
PRIVATE_PACKAGES = ('core', 'ui', 'tools', 'brainlfnet', 'SegandElec', 'SegandElec_nnUNet')
BRAIN_PACKAGES = ('brainlfnet', 'SegandElec', 'SegandElec_nnUNet')
RUNTIME_TOOLS = ('__init__.py', 'individual_worker.py', 'report_page.py',
                 'generate_pdf.py', 'release_checks.py')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def selected_sources():
    sources = [(ROOT / 'main.py', Path('neuroguide_app.py'))]
    for folder in ('core', 'ui', 'BrainLFNet/brainlfnet',
                   'BrainLFNet/SegandElec', 'BrainLFNet/SegandElec_nnUNet'):
        sources.extend((path, path.relative_to(ROOT)) for path in sorted((ROOT / folder).glob('*.py')))
    sources.extend((ROOT / 'tools' / name, Path('tools') / name) for name in RUNTIME_TOOLS)
    return sources


def module_name(relative):
    parts = relative.with_suffix('').parts
    if parts[0] == 'BrainLFNet':
        parts = parts[1:]
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def imports_from_sources(sources):
    """Native modules hide imports from PyInstaller; collect them before compilation."""
    imports = set()
    for source, relative in sources:
        module = module_name(relative)
        package = module if relative.name == '__init__.py' else module.rpartition('.')[0]
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    name = importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                else:
                    name = node.module
                if name:
                    imports.add(name)
                    if not node.module or name == 'PyQt6':
                        imports.update(name + '.' + alias.name for alias in node.names if alias.name != '*')
    return sorted(imports - {'__future__'})


def compile_modules(stage, jobs):
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    from setuptools import Extension, setup

    source_dir, native_dir = stage / 'sources', stage / 'native'
    sources = selected_sources()
    extensions = []
    for source, relative in sources:
        staged = source_dir / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.is_file() or digest(source) != digest(staged):
            shutil.copyfile(source, staged)
        name = module_name(relative)
        extension_name = name + '.__init__' if relative.name == '__init__.py' else name
        extensions.append(Extension(extension_name, [str(staged)],
                                    define_macros=[('CYTHON_CLINE_IN_TRACEBACK', '0')],
                                    extra_compile_args=['/O2'] if os.name == 'nt' else ['-O2']))
    Options.docstrings = False
    Options.emit_code_comments = False
    started = time.perf_counter()
    setup(name='neuroguide-private-runtime', version='4.5.0',
          script_args=['build_ext', '--build-lib', str(native_dir),
                       '--build-temp', str(stage / 'objects'), '--parallel', str(jobs)],
          ext_modules=cythonize(extensions, build_dir=str(stage / 'generated_c'),
                               nthreads=jobs, compiler_directives={
                                   'language_level': 3, 'annotation_typing': False,
                                   'infer_types': False, 'binding': True,
                                   'embedsignature': False, 'linetrace': False,
                                   'profile': False}))
    records = []
    for source, relative in sources:
        module = module_name(relative)
        source_hash = digest(source_dir / relative)
        if digest(source) != source_hash:
            raise RuntimeError('Source changed during compilation: ' + str(source))
        binary_parent = native_dir.joinpath(*module.split('.')[:-1])
        stem = module.rsplit('.', 1)[-1]
        if relative.name == '__init__.py':
            binary_parent = native_dir.joinpath(*module.split('.'))
            stem = '__init__'
        binaries = list(binary_parent.glob(stem + '*.pyd'))
        if len(binaries) != 1:
            raise RuntimeError(f'Expected one native module for {module}: {binaries}')
        records.append({'module': module, 'source': source.relative_to(ROOT).as_posix(),
                        'source_sha256': source_hash,
                        'binary': binaries[0].relative_to(native_dir).as_posix(),
                        'binary_sha256': digest(binaries[0])})
    (native_dir / 'launcher.py').write_text(
        'from neuroguide_app import main\n\nif __name__ == "__main__":\n    main()\n', encoding='utf-8')
    manifest = {'version': '4.5', 'python': sys.version, 'interpreter': sys.executable,
                'cython': __import__('Cython').__version__, 'modules': records,
                'hiddenimports': imports_from_sources(sources),
                'compilation_seconds': round(time.perf_counter() - started, 2)}
    (native_dir / 'protection_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return native_dir


def packaged_binary(item):
    relative = Path(item['binary'])
    # Windows merges BrainLFNet and brainlfnet at the same directory depth.
    # Match the source resource layout: BrainLFNet/brainlfnet and siblings.
    return Path('BrainLFNet') / relative if item['module'].split('.')[0] in BRAIN_PACKAGES else relative


def arrange_native_packages(distribution, manifest):
    """Relocate only verified native files; preserve every model/data resource."""
    internal = distribution / '_internal'
    moved = []
    for item in manifest['modules']:
        relative = packaged_binary(item)
        if relative == Path(item['binary']):
            continue
        source, destination = internal / item['binary'], internal / relative
        if source.is_file():
            if digest(source) != item['binary_sha256']:
                raise RuntimeError('Unexpected native binary before relocation: ' + str(source))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != item['binary_sha256']:
                raise RuntimeError('Native binary changed during relocation: ' + str(destination))
            source.unlink()
            moved.append({'from': item['binary'], 'to': relative.as_posix()})
        elif not destination.is_file() or digest(destination) != item['binary_sha256']:
            raise RuntimeError('Missing native package binary: ' + str(destination))
    return moved


def audit_distribution(distribution, manifest):
    """Reject source/bytecode copies and require the expected native binaries."""
    from PyInstaller.archive.readers import CArchiveReader

    internal = distribution / '_internal'
    errors, checked = [], []
    modules = {item['module'] for item in manifest['modules']} | {'main'}
    for item in manifest['modules']:
        binary = internal / packaged_binary(item)
        if not binary.is_file() or digest(binary) != item['binary_sha256']:
            errors.append('Missing/changed binary: ' + item['module'])
        checked.append(item['module'])
    for folder in ('core', 'ui', 'tools', 'brainlfnet', 'SegandElec', 'SegandElec_nnUNet',
                   'BrainLFNet/brainlfnet', 'BrainLFNet/SegandElec', 'BrainLFNet/SegandElec_nnUNet'):
        for path in (internal / folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.py', '.pyc', '.pyx', '.c', '.cpp', '.pdb', '.html'}:
                errors.append('Private source/build artifact: ' + str(path.relative_to(distribution)))
    for path in internal.iterdir():
        if path.stem.split('.')[0] in {'main', 'neuroguide_app'} and path.suffix.lower() in {'.py', '.pyc', '.pyx', '.c', '.cpp', '.pdb'}:
            errors.append('Private application source: ' + path.name)
    for filename in (RELEASE + '.exe', 'NeuroGuideWorker.exe'):
        archive = CArchiveReader(str(distribution / filename))
        if modules.intersection(archive.toc):
            errors.append(filename + ': private code in executable script entries')
        pyz_count = 0
        for name, entry in archive.toc.items():
            if entry[-1] == 'z':
                pyz_count += 1
                pyz = archive.open_embedded_archive(name)
                leaks = {name for name in pyz.toc if name in modules or any(
                    name == package or name.startswith(package + '.') for package in PRIVATE_PACKAGES)}
                if leaks:
                    errors.append(filename + ': private bytecode in PYZ: ' + ', '.join(sorted(leaks)))
        if pyz_count != 1:
            errors.append(filename + ': expected exactly one inspectable PYZ archive')
    report = {'version': '4.5', 'distribution': str(distribution),
              'native_modules_verified': len(checked), 'modules': checked,
              'private_sources_absent': not errors, 'errors': errors,
              'protection': 'Cython native compilation; not encryption or absolute reverse-engineering prevention',
              'weights_encrypted': False}
    if errors:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=ROOT / 'build/protected_V4_5')
    parser.add_argument('--distpath', type=Path, default=ROOT / 'dist')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    for key, folder in (('TEMP', 'temp'), ('TMP', 'temp'), ('MPLCONFIGDIR', 'mpl_cache'),
                        ('PYINSTALLER_CONFIG_DIR', 'pyinstaller_cache'), ('CUDA_CACHE_PATH', 'cuda_cache'),
                        ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('TRITON_CACHE_DIR', 'triton'),
                        ('HF_HOME', 'hf_cache'), ('TORCH_HOME', 'torch_cache')):
        env[key] = str(stage / folder)
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    os.environ.update({key: env[key] for key in ('TEMP', 'TMP')})
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    native = stage / 'native' if args.skip_compile else compile_modules(stage, args.jobs)
    if args.compile_only:
        print('NATIVE MODULES READY', native, flush=True)
        return
    manifest = json.loads((native / 'protection_manifest.json').read_text(encoding='utf-8'))
    if {item['source'] for item in manifest['modules']} != {source.relative_to(ROOT).as_posix() for source, _ in selected_sources()}:
        raise RuntimeError('Runtime module selection changed; recompile before freezing.')
    for item in manifest['modules']:
        if digest(ROOT / item['source']) != item['source_sha256']:
            raise RuntimeError('Sources changed since compilation: ' + item['source'])
    env.update(NEUROGUIDE_RELEASE_NAME=RELEASE, NEUROGUIDE_PROTECTED_ROOT=str(native))
    base = Path(sys.base_prefix)
    env['PATH'] = os.pathsep.join([str(base / 'Library/bin'), str(base / 'DLLs'),
                                 str(base), env.get('PATH', '')])
    command = [sys.executable, '-u', '-X', 'utf8', '-m', 'PyInstaller', '--noconfirm', '--clean',
               '--workpath', str(stage / 'pyinstaller'), '--distpath', str(args.distpath.resolve()),
               str(ROOT / 'NeuroGuide_revise.spec')]
    print('FREEZING', RELEASE, flush=True)
    subprocess.run(command, cwd=native, env=env, check=True)
    distribution = args.distpath.resolve() / RELEASE
    moved = arrange_native_packages(distribution, manifest)
    (stage / 'native_package_layout.json').write_text(json.dumps(moved, indent=2), encoding='utf-8')
    report = audit_distribution(distribution, manifest)
    (stage / 'protection_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PROTECTED RELEASE VERIFIED', report['native_modules_verified'], 'native modules', flush=True)


if __name__ == '__main__':
    main()
