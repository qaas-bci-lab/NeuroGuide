"""Validate the actual V4.5 executables with Python/Conda removed from PATH."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare_predictions(baseline_path, current_path, require_identical_grid=False, max_relative_l2=.01):
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    current = json.loads(Path(current_path).read_text(encoding='utf-8'))
    image = nib.load(current['padded_t1'])
    with np.load(baseline['geometry']) as old_geometry, np.load(current['geometry']) as new_geometry:
        np.testing.assert_allclose(old_geometry['affine'], new_geometry['affine'], atol=1e-6)
        old_ijk, new_ijk = old_geometry['valid_ijk'], new_geometry['valid_ijk']
        identical = np.array_equal(old_ijk, new_ijk)
        if require_identical_grid:
            np.testing.assert_array_equal(old_ijk, new_ijk)
        if identical:
            old_rows = new_rows = np.arange(len(old_ijk))
        else:
            _, old_rows, new_rows = np.intersect1d(
                np.ravel_multi_index(old_ijk.T, image.shape),
                np.ravel_multi_index(new_ijk.T, image.shape), assume_unique=True, return_indices=True)
        coverage = len(old_rows) / max(len(old_ijk), len(new_ijk))
        assert coverage > .9999, coverage
        counts = (len(old_ijk), len(new_ijk))
    old, new = np.load(baseline['leadfield'], mmap_mode='r'), np.load(current['leadfield'], mmap_mode='r')
    signal = error = peak = 0.
    try:
        assert old.shape[1:] == new.shape[1:] == (71, 3)
        for column in range(71):
            a = np.asarray(old[old_rows, column], dtype=np.float64)
            b = np.asarray(new[new_rows, column], dtype=np.float64)
            signal += float(np.square(a).sum())
            error += float(np.square(a-b).sum())
            peak = max(peak, float(np.abs(a-b).max()))
    finally:
        old._mmap.close(); new._mmap.close()
    relative = float(np.sqrt(error / max(signal, 1e-30)))
    # Keep the historical full-pipeline release limit (1%). Compilation itself
    # is checked separately on identical preprocessed inputs in the same env.
    assert np.isfinite(relative) and relative < max_relative_l2, relative
    old_tissue = np.asarray(nib.load(baseline['tissue_masks']).dataobj)
    new_tissue = np.asarray(nib.load(current['tissue_masks']).dataobj)
    assert old_tissue.shape == new_tissue.shape
    return {'relative_l2': relative, 'relative_l2_limit': max_relative_l2,
            'max_absolute_error': peak, 'same_geometry': True,
            'same_valid_voxels': identical, 'baseline_voxels': counts[0], 'current_voxels': counts[1],
            'shared_voxels': len(old_rows), 'shared_voxel_fraction': coverage,
            'changed_tissue_voxels': int(np.count_nonzero(old_tissue != new_tissue)),
            'baseline': str(baseline_path), 'current': str(current_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path, default=ROOT / 'dist/NeuroGuide-V4.5')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/protected_V4_5/validation')
    parser.add_argument('--t1', type=Path, default=ROOT / 'mri/T1.nii')
    parser.add_argument('--baseline-prediction', type=Path)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--comparison-only', action='store_true',
                        help='Resume the numerical comparison after all executable functional checks passed.')
    parser.add_argument('--identical-input-prediction', type=Path)
    parser.add_argument('--source-reference-prediction', type=Path)
    args = parser.parse_args()
    dist, out = args.distribution.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.comparison_only:
        record = json.loads((out / 'validation.json').read_text(encoding='utf-8'))
        assert Path(record['distribution']).resolve() == dist
        for key in ('console_startup', 'gui_startup', 'all_71_columns_finite', 'compiled_cache_reuse',
                    'ti_result', 'tes_result', 'ti_gui_report', 'tes_gui_report'):
            assert record.get(key), 'Missing completed check: ' + key
        assert args.baseline_prediction is not None
        current = Path(record['prediction_task']['prediction_dir']) / 'BrainLFNet_prediction.json'
        record['baseline_comparison'] = compare_predictions(args.baseline_prediction, current)
        if args.identical_input_prediction:
            record['identical_input_comparison'] = compare_predictions(
                args.baseline_prediction, args.identical_input_prediction,
                require_identical_grid=True, max_relative_l2=.001)
        if args.source_reference_prediction:
            assert args.identical_input_prediction is not None
            record['source_vs_compiled_comparison'] = compare_predictions(
                args.source_reference_prediction, args.identical_input_prediction,
                require_identical_grid=True, max_relative_l2=1e-6)
        record['validation_complete'] = True
        (out / 'validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('V4.5 STANDALONE ACCEPTANCE COMPLETE', out / 'validation.json', flush=True)
        return
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV',
                'CONDA_SHLVL', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM',
                'QT_API', 'MPLBACKEND'):
        env.pop(key, None)
    windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
    env.update(PATH=os.pathsep.join([str(windows / 'System32'), str(windows), str(dist)]),
               PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    for key, folder in (('TEMP', 'temp'), ('TMP', 'temp'), ('MPLCONFIGDIR', 'mpl_cache'),
                        ('CUDA_CACHE_PATH', 'cuda_cache'), ('TORCH_HOME', 'torch_cache'),
                        ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('TRITON_CACHE_DIR', 'triton'),
                        ('HF_HOME', 'hf_cache')):
        env[key] = str(out / folder)
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    worker = dist / 'NeuroGuideWorker.exe'
    gui = dist / 'NeuroGuide-V4.5.exe'
    record = {'version': '4.5', 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'distribution': str(dist), 'clean_path': env['PATH']}
    def save():
        (out / 'validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')

    def run(name, command, timeout=240, environment=None):
        started = time.perf_counter()
        with (out / (name + '.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([str(value) for value in command], cwd=out, env=environment or env,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            print(name, 'PID', process.pid, flush=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                raise RuntimeError(f'{name} timed out; see {out / (name + ".log")}')
        text = (out / (name + '.log')).read_text(encoding='utf-8', errors='replace')
        if code:
            raise RuntimeError(f'{name} exited {code}: {text[-10000:]}')
        record[name + '_seconds'] = round(time.perf_counter() - started, 2)
        print(name, 'PASS', record[name + '_seconds'], 's', flush=True)
        return text

    def job(name, config, timeout=1200):
        path = out / (name + '.json')
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        text = run(name, [worker, '--individual-worker', path], timeout)
        events = [json.loads(line.split('NEUROGUIDE_EVENT ', 1)[1]) for line in text.splitlines()
                  if line.startswith('NEUROGUIDE_EVENT ')]
        values = [event['value'] for event in events if event['kind'] == 'result']
        assert len(values) == 1, events[-3:]
        return values[0]

    def gui_check(name, result_path=None, executable=gui):
        ready = out / (name + '.ready')
        ready.unlink(missing_ok=True)
        command = [executable, '--startup-check']
        if result_path:
            config = {'result_file': str(result_path), 'report': str(out / (name + '.pdf')),
                      'screenshot': str(out / (name + '.png')),
                      'electrode_screenshot': str(out / (name + '_electrodes.png')),
                      'check_record': str(out / (name + '_check.json'))}
            path = out / (name + '_gui.json')
            path.write_text(json.dumps(config), encoding='utf-8')
            command.extend(['--field-check', path])
        run(name, command, environment=dict(env, NEUROGUIDE_STARTUP_READY=str(ready)))
        assert ready.read_text() == 'ready'
        if result_path:
            check = json.loads((out / (name + '_check.json')).read_text(encoding='utf-8'))
            assert check['version'] == '4.5'
            for key in ('report_success', 'gui_loaded', 'display_3d_grid_verified',
                        'physical_drag_verified', 'camera_reset_preserves_planes_verified', 'roi_clip_verified'):
                assert check[key], key
            import pymupdf
            with pymupdf.open(config['report']) as pdf:
                assert len(pdf) == 1 and len(pdf[0].get_images()) == 3
            return check
        return True

    # Console startup first makes any native-import failure visible in a log.
    record['console_startup'] = gui_check('console_startup', executable=worker)
    record['gui_startup'] = gui_check('gui_startup')
    for name, environment in (('gpu_probe', env), ('hidden_gpu_probe', dict(env, CUDA_VISIBLE_DEVICES='-1'))):
        text = run(name, [worker, '--cuda-probe'], environment=environment)
        rows = [line.split('NEUROGUIDE_CUDA_PROBE ', 1)[1] for line in text.splitlines()
                if line.startswith('NEUROGUIDE_CUDA_PROBE ')]
        assert len(rows) == 1, text
        record[name] = json.loads(rows[0])
    assert record['gpu_probe']['torch_version'] == '2.7.1+cu128'
    assert 'sm_120' in record['gpu_probe'].get('compiled_arches', []) or not record['gpu_probe']['usable']
    assert not record['hidden_gpu_probe']['usable']
    save()
    if args.smoke_only:
        return
    config = {'action': 'predict', 't1': str(args.t1.resolve()),
              'brain_root': str(dist / '_internal/BrainLFNet'),
              'output_root': str(out / 'tasks'), 'padding': 10, 'device': 'auto'}
    result = job('real_prediction', config)
    record['prediction_task'] = result
    metadata = json.loads((Path(result['prediction_dir']) / 'BrainLFNet_prediction.json').read_text(encoding='utf-8'))
    image = nib.load(result['padded_t1'])
    np.testing.assert_allclose(nib.affines.voxel_sizes(image.affine), [1.2, .9375, .9375], atol=1e-4)
    array = np.load(metadata['leadfield'], mmap_mode='r')
    assert array.shape[1:] == (71, 3)
    for column in range(71):
        assert np.isfinite(array[:, column]).all()
    array._mmap.close()
    record['all_71_columns_finite'] = True
    inspect = job('cache_inspection', dict(config, action='inspect'), 180)
    assert inspect['task'] is not None
    record['compiled_cache_reuse'] = True
    save()
    with np.load(metadata['geometry']) as geometry:
        ijk = geometry['valid_ijk'].copy()
    center = ijk[np.argmin(np.square(ijk - ijk.mean(axis=0)).sum(axis=1))]
    roi = {'center_ras_mm': nib.affines.apply_affine(image.affine, center).tolist(), 'radius_mm': 10.}
    common = {'population': 256, 'generations': 160, 'ratio_steps': 41, 'sample_per_region': 4096,
              'finalist_count': 16, 'goal': 'intensity', 'seed': 20261012}
    for mode, options in (('ti', dict(common, channel_limit_ma=2., min_channel_ma=.1)),
                          ('tes', dict(common, electrode_limit_ma=2., absolute_sum_limit_ma=4., electrode_count=5))):
        optimization = {'action': 'optimize', 'prediction_dir': result['prediction_dir'],
                        'output_dir': str(out / mode), 'roi': roi, 'options': options}
        if mode == 'tes':
            optimization['stimulation_type'] = 'tES'
        optimized = job(mode + '_optimization', optimization)
        record[mode + '_result'] = optimized
        record[mode + '_gui_report'] = gui_check(mode + '_gui', optimized['result_path'])
        save()
    if args.baseline_prediction:
        record['baseline_comparison'] = compare_predictions(
            args.baseline_prediction, Path(result['prediction_dir']) / 'BrainLFNet_prediction.json')
    record['validation_complete'] = True
    save()
    print('V4.5 STANDALONE ACCEPTANCE COMPLETE', out / 'validation.json', flush=True)


if __name__ == '__main__':
    main()
