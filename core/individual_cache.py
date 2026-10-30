"""Read-only matching of completed individual models, without loading networks."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from core.individual_pipeline import file_hash, prediction_assets, preprocessing_settings
from core.ti import Leadfield
from core.cache_contract import signatures_match, assets_match


def normalized(path):
    return str(path).replace('\\', '/').casefold()


def compatible_signature(signature, padding, t1=None):
    expected = preprocessing_settings(padding)
    # A completed GPU/CPU result can be reused on either device; its precision
    # remains recorded in prediction metadata and no inference is run here.
    from core.runtime import is_one_mm_image
    return signatures_match(signature,expected,allow_native_1mm=signature.get('version')==2 and t1 is not None and is_one_mm_image(t1))


def completed_task(path, manifest, t1):
    stages = manifest['stages']
    if not stages.get('prediction'): raise ValueError('Incomplete prediction')
    prepared = stages['preprocessing']
    if prepared.get('model_label_order') != 'verified' or prepared.get('landmark_method') != 'python-affine' or prepared.get('requires_spm_runtime') is not False or prepared.get('cleanup') != 'roast-neck' or prepared.get('preserve_uncalibrated_intracranial') is not True:
        raise ValueError('Cached preprocessing backend mismatch')
    padded, tissue = Path(stages['padded']), Path(stages['segmentation'])
    electrodes = stages['electrodes']
    if not all(Path(electrodes[k]).is_file() for k in ('electrode_mask', 'electrode_geometry', 'electrode_coordinates')):
        raise ValueError('Incomplete electrode placement')
    prediction = path.parent/'prediction'
    json.loads((prediction/'BrainLFNet_prediction.json').read_text(encoding='utf-8'))
    lf = Leadfield(prediction)
    try:
        for image_path in (padded, tissue):
            image = nib.load(image_path)
            if image.shape != lf.shape or not np.allclose(image.affine, lf.affine, atol=1e-4):
                raise ValueError('Cached anatomy and leadfield grids differ')
    finally:
        lf.close()
    return {'task_dir': str(path.parent), 'manifest': str(path), 't1': str(t1),
            'padded_t1': str(padded), 'tissue_masks': str(tissue),
            'prediction_dir': str(prediction), 'electrodes': electrodes,
            'atlas': manifest.get('atlases', {}).get('aal3'), 'timings':manifest.get('timings',prepared.get('timings',{}))}


def find_reusable_task(config, progress=lambda _: None):
    t1 = Path(config['t1']).resolve()
    storage = Path(config['output_root']).resolve()
    if not t1.is_file(): raise FileNotFoundError('MRI file is missing')
    if not storage.is_dir(): return {'task': None}
    candidates = []
    for path in storage.rglob('manifest.json'):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            signature = manifest.get('signature', {})
            if not manifest.get('stages', {}).get('prediction') or not compatible_signature(signature, int(config.get('padding', 10)),t1):
                continue
            hashes = {normalized(p): h for p, h in signature['hashes'].items()}
            source_hash = hashes.pop(normalized(manifest['t1']))
            candidates.append((path, manifest, source_hash, hashes))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    if not candidates: return {'task': None}
    progress('正在核对 MRI 与已保存模型')
    source_hash = file_hash(t1)
    candidates = [c for c in candidates if c[2] == source_hash]
    if not candidates: return {'task': None}
    brain = Path(config['brain_root']).resolve()
    progress('正在核对模型权重、模板与处理参数')
    assets = prediction_assets(t1, brain)[1:]
    current = {normalized(p.relative_to(brain)): file_hash(p) for p in assets}
    for path, manifest, _, hashes in sorted(candidates, key=lambda c: c[0].stat().st_mtime, reverse=True):
        # Relative resource paths allow the installation and MRI to be moved;
        # compare content hashes rather than filenames or modification times.
        if not assets_match(hashes,current): continue
        try:
            task = completed_task(path, manifest, t1)
        except Exception as exc:
            progress(f'跳过不完整的已保存模型：{path.parent.name}（{exc}）')
            continue
        return {'task': task}
    return {'task': None}
