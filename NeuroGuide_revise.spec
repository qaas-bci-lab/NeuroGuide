# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
import sys
import time
import json
import pefile
from PyInstaller.utils.hooks import collect_submodules, collect_all, copy_metadata
from PyInstaller.config import CONF

root = Path(SPECPATH)
release_name = os.environ.get('NEUROGUIDE_RELEASE_NAME', 'NeuroGuide-V4.5')
if Path(release_name).name != release_name or '/' in release_name or '\\' in release_name or release_name in {'.', '..'}:
    raise ValueError('Invalid NeuroGuide release name')
brain = root / 'BrainLFNet'
protected = os.environ.get('NEUROGUIDE_PROTECTED_ROOT')
if release_name.startswith('NeuroGuide-V4.5') and not protected:
    raise RuntimeError('V4.5 requires the protected build: python tools/build_protected.py')
code_root = Path(protected).resolve() if protected else root
protection = json.loads((code_root / 'protection_manifest.json').read_text(encoding='utf-8')) if protected else None
sys.path.insert(0, str(code_root))
# Only runtime resources: the previous disease/target field bank is excluded.
datas = [(str(root / 'mri'), 'mri'),
         (str(root / 'assets/dcm2niix'), 'assets/dcm2niix'),
         (str(root / 'assets/atlases'), 'assets/atlases'),
         (str(root / 'assets/branding/neuroguide_mark.png'), 'assets/branding'),
         (str(root / 'assets/branding/SELECTED_LOGO.md'), 'assets/branding')]

if not protected:
    datas.append((str(root / 'core/individual_pipeline.py'), 'core'))
if not protected:
    datas.append((str(root / 'core/low_memory_inference.py'), 'core'))
datas.extend([
    (str(brain / 'LICENSE.md'), 'BrainLFNet'),
    (str(brain / 'THIRD_PARTY_NOTICES.md'), 'BrainLFNet'),
    (str(brain / 'SegandElec/LICENSE_GPL3.md'), 'BrainLFNet/SegandElec'),
    (str(brain / 'checkpoints/best_ema.pt'), 'BrainLFNet/checkpoints'),
    (str(brain / 'checkpoints/args.json'), 'BrainLFNet/checkpoints'),
    (str(brain / 'SegandElec/assets'), 'BrainLFNet/SegandElec/assets'),
])
package = brain / 'SegandElec_nnUNet/model'
relative = 'BrainLFNet/SegandElec_nnUNet/model'
for name in ('checkpoint_final.pth', 'plans.json', 'dataset.json', 'inference_config.json', 'postprocessing.json', 'README.md'):
    datas.append((str(package / name), relative))
datas.append((str(brain / 'SegandElec_nnUNet/configs'), 'BrainLFNet/SegandElec_nnUNet/configs'))
# Include source files too: task fingerprints cover the actual preprocessing recipe.
if not protected:
    for folder in ('SegandElec_nnUNet', 'SegandElec', 'brainlfnet'):
        for source in (brain / folder).glob('*.py'):
            datas.append((str(source), 'BrainLFNet/' + folder))
hidden = ['brainlfnet', 'brainlfnet.model', 'SegandElec.pipeline_common',
          'SegandElec.spatial_compat', 'SegandElec.electrodes', 'SegandElec.roast_touchup',
          'SegandElec.legacy_landmark_registration', 'SegandElec.landmark_registration',
          'SegandElec_nnUNet', 'SegandElec_nnUNet.pipeline', 'SegandElec_nnUNet.inference',
          'SegandElec_nnUNet.field_of_view', 'SegandElec_nnUNet.labels', 'SegandElec_nnUNet.landmarks', 'SegandElec_nnUNet.python_affine',
          'matplotlib.backends.backend_qtagg', 'nibabel.processing', 'core.tes', 'tools.release_checks']
if protected:
    hidden.extend(protection['hiddenimports'])
    hidden.extend(item['module'] for item in protection['modules'])
ants_datas, ants_binaries, ants_hidden = collect_all('ants')
# MKL loads its CPU kernels dynamically. The inherited Conda environment is
# not identified correctly by the default torch hook, so collect them explicitly.
mkl_bin = Path(sys.base_prefix) / 'Library/bin'
mkl_binaries = [(str(path), '.') for pattern in ('mkl*.dll', 'libiomp5md.dll', 'tbb*.dll')
                for path in mkl_bin.glob(pattern)]
datas.extend(ants_datas); hidden.extend(ants_hidden)
for package_name in ('nnunetv2', 'dynamic_network_architectures'):
    hidden.extend(collect_submodules(package_name))
for distribution in ('nnunetv2', 'dynamic-network-architectures', 'torch', 'antspyx'):
    datas.extend(copy_metadata(distribution))
entry = code_root / 'launcher.py' if protected else root / 'main.py'
a = Analysis([str(entry)], pathex=[str(code_root), str(code_root / 'BrainLFNet')], binaries=ants_binaries+mkl_binaries, datas=datas,
             hiddenimports=hidden, hookspath=[], hooksconfig={}, runtime_hooks=[str(root / 'tools/frozen_runtime.py')],
             excludes=['PyQt5'], noarchive=False, optimize=0)
if protected:
    protected_names = {item['module'] for item in protection['modules']}
    private_packages = ('core', 'ui', 'tools', 'brainlfnet', 'SegandElec', 'SegandElec_nnUNet')
    leaked = [name for name, *_ in a.pure if name in protected_names or name == 'main'
              or any(name == package or name.startswith(package + '.') for package in private_packages)]
    if leaked:
        raise RuntimeError('Protected modules collected as bytecode: ' + ', '.join(leaked))
    for item in protection['modules']:
        if not any(name.replace('\\', '/') == item['binary'] for name, *_ in a.binaries):
            raise RuntimeError('Missing protected binary: ' + item['module'])
pyz = PYZ(a.pure)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name=release_name,
          debug=False, strip=False, upx=False, console=False,
          version=str(root / 'assets/branding/version_info.txt'),
          icon=str(root / 'assets/branding/neuroguide_mark.png'))
# Same entry point, dispatched before any Qt import by --individual-worker.
worker = EXE(pyz, a.scripts, [], exclude_binaries=True, name='NeuroGuideWorker',
             debug=False, strip=False, upx=False, console=True,
             version=str(root / 'assets/branding/version_info.txt'))
# Frozen imports add stack frames around pandas' nested native imports.
# Reserve address space for the main thread; committed stack grows on demand.
for executable in (gui, worker):
    for attempt in range(60):
        try:
            image = pefile.PE(executable.name, fast_load=True)
            image.OPTIONAL_HEADER.SizeOfStackReserve = 16 * 1024**2
            image.OPTIONAL_HEADER.CheckSum = 0
            patched_image = image.write()
            image.close()
            Path(executable.name).write_bytes(patched_image)
            break
        except PermissionError:
            if attempt == 59: raise
            time.sleep(.5)
coll = COLLECT(gui, worker, a.binaries, a.datas, strip=False, upx=False, name=release_name)
