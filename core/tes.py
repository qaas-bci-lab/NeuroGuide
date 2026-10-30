"""Balanced multi-electrode tES synthesis and constrained elitist search.

Signed currents are DC mA, or peak mA for common-frequency 0/180-degree AC.
The objective is field magnitude, not a cortical-normal or polarity objective.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from core.ti import Leadfield, statistics


def validate_montage(lf, labels, currents, electrode_limit_ma=2., absolute_sum_limit_ma=4., electrode_count=None):
    labels = list(map(int, labels))
    currents = np.asarray(currents, dtype=float)
    limits = np.asarray([electrode_limit_ma, absolute_sum_limit_ma], dtype=float)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('电流上限必须是大于零的有限值。')
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('tES 至少需要两个不同的电极。')
    if any(label not in lf.labels + [lf.reference] for label in labels):
        raise ValueError('方案包含未知电极。')
    if currents.shape != (len(labels),) or not np.isfinite(currents).all():
        raise ValueError('每个电极必须对应一个有限的带符号电流。')
    if np.any(np.abs(currents) < 1e-9):
        raise ValueError('所选电极的电流不能为零；请减少电极数量或设置非零电流。')
    if electrode_count is not None and len(labels) != int(electrode_count):
        raise ValueError('方案的有效电极数量与设置不一致。')
    if abs(float(currents.sum())) > 1e-8:
        raise ValueError('电流不平衡：所有电极的带符号电流之和必须为 0 mA。')
    if np.max(np.abs(currents)) > electrode_limit_ma + 1e-8:
        raise ValueError('单电极电流绝对值超过上限。')
    if np.abs(currents).sum() > absolute_sum_limit_ma + 1e-8:
        raise ValueError('所有电极电流绝对值之和超过上限。')


def field(lf, labels, currents, indices):
    # Reference column is zero; explicit balance ensures the implicit return
    # equals the requested reference current even when Iz is selected.
    result = np.zeros((len(lf.ijk[indices]), 3), dtype=np.float64)
    for label, current in zip(labels, currents):
        result += float(current) * lf.column(label, indices)
    return result


def synthesize(prediction_dir, output_dir, labels, currents, roi_spec=None, progress=print,
               threshold=0.2, electrode_limit_ma=2., absolute_sum_limit_ma=4., electrode_count=None):
    lf = Leadfield(prediction_dir)
    try:
        validate_montage(lf, labels, currents, electrode_limit_ma, absolute_sum_limit_ma, electrode_count)
        roi = lf.roi(roi_spec) if roi_spec else None
        vectors = np.empty((len(lf.ijk), 3), dtype=np.float32)
        values = np.empty(len(lf.ijk), dtype=np.float32)
        for start in range(0, len(values), 65536):
            sl = slice(start, min(start + 65536, len(values)))
            e = field(lf, labels, currents, sl)
            vectors[sl] = e
            values[sl] = np.linalg.norm(e, axis=1)
            progress(f'tES 电场合成 {sl.stop}/{len(values)}')
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, data in [('tES_magnitude', values), ('tES_vector', vectors)]:
            volume = np.zeros(lf.shape + data.shape[1:], dtype=np.float32)
            volume[tuple(lf.ijk.T)] = data
            image = nib.Nifti1Image(volume, lf.affine)
            image.header.set_xyzt_units('mm')
            image.header['descrip'] = b'NeuroGuide | tES | V/m | signed DC or AC peak mA'
            path = output / f'{name}.nii.gz'
            temporary = output / f'{name}.partial.nii.gz'
            nib.save(image, str(temporary)); temporary.replace(path)
            paths[name] = str(path)
        if roi is not None:
            volume = np.zeros(lf.shape, dtype=np.uint8)
            volume[tuple(lf.ijk[roi].T)] = 1
            nib.save(nib.Nifti1Image(volume, lf.affine), str(output / 'ROI.nii.gz'))
        manifest_path = lf.directory / 'BrainLFNet_prediction.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        currents = list(map(float, currents))
        result = {'stimulation_type': 'tES', 'prediction_dir': str(lf.directory),
                  'padded_t1': manifest.get('padded_t1'), 'electrode_labels': list(map(int, labels)),
                  'currents_ma': currents, 'current_convention': 'signed_dc_or_common_frequency_ac_peak',
                  'quantity': 'tes_electric_field_magnitude', 'units': 'V/m', 'colormap': 'jet',
                  'field_assumption': 'quasi_static_common_waveform_0_or_180_degree_phase',
                  'roi': roi_spec, 'overlays': [paths['tES_magnitude']], 'vector_field': paths['tES_vector'],
                  'constraints': {'electrode_limit_ma': float(electrode_limit_ma),
                                  'absolute_sum_limit_ma': float(absolute_sum_limit_ma),
                                  'electrode_count': len(labels)},
                  'current_summary': {'signed_sum_ma': float(sum(currents)),
                                      'absolute_sum_ma': float(sum(map(abs, currents))),
                                      'total_injected_ma': float(sum(max(0., c) for c in currents))},
                  'statistics': statistics(values, roi, threshold)}
        with (output / 'electrode_currents.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['electrode_label', 'signed_current_ma'])
            writer.writerows(zip(labels, currents))
        result['result_path'] = str(output / 'result.json')
        Path(result['result_path']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result
    finally:
        lf.close()


def optimize(prediction_dir, output_dir, roi_spec, options, progress=print):
    lf = Leadfield(prediction_dir)
    try:
        roi = lf.roi(roi_spec)
        if not (~roi).any():
            raise ValueError('优化需要靶区之外的脑组织用于评价。')
        seed = int(options.get('seed', 20261012)); rng = np.random.default_rng(seed)
        candidates = list(map(int, options.get('electrodes', lf.labels + [lf.reference])))
        n = int(options.get('electrode_count', 5))
        if n < 2 or len(set(candidates)) != len(candidates) or len(candidates) < n or any(x not in lf.labels + [lf.reference] for x in candidates):
            raise ValueError('候选电极必须有效、互不重复，且数量不少于所需电极数。')
        cap = float(options.get('electrode_limit_ma', 2.))
        budget = float(options.get('absolute_sum_limit_ma', 4.))
        from core.search_defaults import DEFAULT_SEARCH
        steps = int(options.get('ratio_steps', DEFAULT_SEARCH['ratio_steps']))
        if not np.isfinite([cap, budget]).all() or min(cap, budget) <= 0 or steps < 2:
            raise ValueError('请检查电流上限和搜索档数。')
        cap_units = steps - 1; delta = cap / cap_units
        budget_units = int(np.floor((budget / 2) / delta + 1e-10))
        partitions = [k for k in range(1, n) if max(k, n-k) <= min(budget_units, k*cap_units, (n-k)*cap_units)]
        if not partitions:
            raise ValueError('当前电流档位和总电流上限无法容纳所需非零电极；请增加搜索档数、提高总上限或减少电极数。')
        size = int(options.get('population', DEFAULT_SEARCH['population'])); generations = int(options.get('generations', DEFAULT_SEARCH['generations']))
        finalists = int(options.get('finalist_count', DEFAULT_SEARCH['finalist_count']))
        threshold = float(options.get('threshold_v_m', .2)); penalty = float(options.get('avoidance_weight', 1.))
        goal = options.get('goal', 'intensity')
        if size < 8 or generations < 1 or finalists < 1 or goal not in ('intensity', 'focality'):
            raise ValueError('请检查搜索种群、代数、复算数和优化方向。')
        if not np.isfinite([threshold, penalty]).all() or threshold <= 0 or penalty < 0:
            raise ValueError('评价阈值必须大于零，惩罚权重不能为负。')
        sample_size = max(1, int(options.get('sample_per_region', DEFAULT_SEARCH['sample_per_region'])))
        sampled = [np.sort(rng.choice(g, min(sample_size, len(g)), replace=False)) for g in [np.flatnonzero(roi), np.flatnonzero(~roi)]]
        indices = np.concatenate(sampled); split = len(sampled[0])
        bank = np.stack([lf.column(label, indices) for label in candidates], axis=1).astype(np.float64)
        if not np.isfinite(bank).all():
            raise ValueError('导联场包含无效数值。')
        cache = {}

        def allocate(total, count):
            allocation = np.ones(count, dtype=int)
            weights = rng.dirichlet(np.ones(count))
            for _ in range(total - count):
                probability = weights * (allocation < cap_units)
                probability /= probability.sum()
                allocation[rng.choice(count, p=probability)] += 1
            return allocation

        def random_currents(full=False):
            k = int(rng.choice(partitions))
            upper = min(budget_units, k*cap_units, (n-k)*cap_units)
            total = upper if full else int(rng.integers(max(k, n-k), upper + 1))
            units = np.r_[allocate(total, k), -allocate(total, n-k)]
            return tuple(map(int, rng.permutation(units)))

        def canonical(electrodes, units):
            ordered = sorted(zip(electrodes, units))
            # Magnitude is invariant under a global polarity reversal. Keep one
            # representative; this is not an optimization of tDCS polarity.
            sign = 1 if ordered[0][1] > 0 else -1
            return tuple(e for e, _ in ordered) + tuple(sign*u for _, u in ordered)

        def score(target, avoid):
            if goal == 'intensity':
                return float(target.mean() - penalty * avoid.mean())
            shortfall = max(0., threshold - float(target.max())) / threshold
            return float(np.mean(target >= threshold) - penalty*np.mean(avoid >= threshold) - 2*shortfall)

        def evaluate(g):
            if g not in cache:
                e = np.einsum('vnc,n->vc', bank[:, g[:n], :], np.asarray(g[n:])*delta, optimize=False)
                v = np.linalg.norm(e, axis=1)
                cache[g] = score(v[:split], v[split:])
            return cache[g]

        population = [canonical(rng.choice(len(candidates), n, replace=False), random_currents(full=i==0)) for i in range(size)]
        history = []
        for generation in range(generations):
            population.sort(key=evaluate, reverse=True)
            history.append(evaluate(population[0]))
            progress(f'tES 遗传搜索 {generation+1}/{generations}；当前采样得分 {history[-1]:.5f}；已评估 {len(cache)} 个方案')
            next_population = list(population[:max(2, size//8)])
            while len(next_population) < size:
                def tournament():
                    return max((population[i] for i in rng.integers(size, size=3)), key=evaluate)
                p, q = tournament(), tournament()
                child = [p[i] if rng.random() < .5 else q[i] for i in range(n)]
                used = set()
                for i in range(n):
                    if child[i] in used or rng.random() < .2:
                        available = [j for j in range(len(candidates)) if j not in used and j not in child[i+1:]]
                        if not available: available = [j for j in range(len(candidates)) if j not in used]
                        child[i] = int(rng.choice(available))
                    used.add(child[i])
                units = list((p if rng.random() < .5 else q)[n:])
                if rng.random() < .3:
                    units = random_currents()
                else:
                    # Transfer one unit between same-polarity electrodes. This
                    # preserves balance, absolute sum, count, and the cap.
                    transfers = [(i,j) for i in range(n) for j in range(n) if i!=j and units[i]*units[j]>0 and abs(units[i])>1 and abs(units[j])<cap_units]
                    if transfers:
                        i,j = transfers[int(rng.integers(len(transfers)))]; sign = 1 if units[i]>0 else -1
                        units[i] -= sign; units[j] += sign
                next_population.append(canonical(child, units))
            population = next_population
        for g in population: evaluate(g)
        ranking = []
        for k,g in enumerate(sorted(cache, key=cache.get, reverse=True)[:finalists]):
            labels = [candidates[i] for i in g[:n]]; currents = (np.asarray(g[n:])*delta).tolist()
            validate_montage(lf, labels, currents, cap, budget, n)
            values = np.empty(len(lf.ijk), dtype=np.float32)
            for start in range(0, len(values), 65536):
                sl = slice(start, min(start+65536, len(values)))
                values[sl] = np.linalg.norm(field(lf, labels, currents, sl), axis=1)
            ranking.append({'electrode_labels': labels, 'currents_ma': currents, 'sample_score': cache[g],
                            'full_score': score(values[roi], values[~roi]), 'statistics': statistics(values, roi, threshold)})
            progress(f'tES 完整导联场复算 {k+1}/{min(finalists,len(cache))}')
        ranking.sort(key=lambda item: item['full_score'], reverse=True)
        winner = ranking[0]
        result = synthesize(prediction_dir, output_dir, winner['electrode_labels'], winner['currents_ma'], roi_spec,
                            progress, threshold, cap, budget, n)
        result['optimization'] = {'method': 'leadfield_constrained_elitist_genetic_algorithm',
                                  'options': options, 'seed': seed, 'goal': goal, 'evaluated': len(cache),
                                  'current_step_ma': delta, 'current_constraint': 'balanced_signed_l1_and_individual_limits',
                                  'sample_roi_indices': sampled[0].tolist(), 'sample_non_roi_indices': sampled[1].tolist(),
                                  'history': history, 'ranking': ranking, 'global_optimum_guaranteed': False,
                                  'polarity_objective': 'magnitude_only_global_sign_equivalent'}
        Path(result['result_path']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result
    finally:
        lf.close()
