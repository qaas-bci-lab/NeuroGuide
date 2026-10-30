"""Leadfield-based two-channel TI synthesis and reproducible genetic search.

Currents are peak mA. Leadfield columns are V/m per +1/-1 mA with Iz return.
The GA follows the published two-pair search approach, with explicit local
engineering choices documented in docs/individual_ti.md (not an exact replay).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def max_ti(e1, e2):
    """Maximum directional modulation amplitude, consistent with SimNIBS.

    Handles zero fields and equal parallel fields without divide-by-zero.
    """
    a, b = np.asarray(e1, dtype=np.float64), np.asarray(e2, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("Electric fields must have matching [...,3] shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Electric fields contain non-finite values")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    swap = nb > na
    large, small = np.where(swap[..., None], b, a), np.where(swap[..., None], a, b)
    dot = np.sum(large * small, axis=-1)
    small = np.where((dot < 0)[..., None], -small, small)
    lo, hi = np.minimum(na, nb), np.maximum(na, nb)
    difference = large - small
    denominator = np.linalg.norm(difference, axis=-1)
    cross = 2 * np.linalg.norm(np.cross(small, difference), axis=-1)
    oblique = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    return np.where(lo * lo <= np.abs(dot), 2 * lo, oblique).clip(0).astype(np.float32)


class Leadfield:
    def close(self):
        array = getattr(self, "array", None)
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def __del__(self):
        self.close()

    def __init__(self, prediction_dir):
        self.directory = Path(prediction_dir).resolve()
        with np.load(self.directory / "leadfield_geometry.npz", allow_pickle=False) as g:
            self.ijk = np.asarray(g["valid_ijk"], dtype=np.int64)
            self.affine = np.asarray(g["affine"], dtype=np.float64)
            self.shape = tuple(int(x) for x in g["grid_shape"])
            self.labels = np.asarray(g["active_electrode_labels"], dtype=int).tolist()
            self.reference = int(g["reference_electrode_label"])
        self.array = np.load(self.directory / "leadfield_gmwm.npy", mmap_mode="r", allow_pickle=False)
        if self.array.shape != (len(self.ijk), len(self.labels), 3):
            raise ValueError("Leadfield shape does not match geometry")
        if self.ijk.shape != (len(self.ijk), 3) or not len(self.ijk):
            raise ValueError("Invalid or empty leadfield voxel indices")
        if len(self.shape) != 3 or np.any(self.ijk < 0) or np.any(self.ijk >= self.shape):
            raise ValueError("Leadfield indices are outside the volume")
        if self.affine.shape != (4, 4) or not np.isfinite(self.affine).all():
            raise ValueError("Invalid leadfield affine")
        if len(set(self.labels)) != len(self.labels) or self.reference in self.labels:
            raise ValueError("Invalid active/reference electrode labels")
        self.columns = {label: i for i, label in enumerate(self.labels)}

    def column(self, label, indices):
        label = int(label)
        if label == self.reference:
            return np.zeros((len(self.ijk[indices]), 3), dtype=np.float32)
        if label not in self.columns:
            raise ValueError(f"Unknown electrode label: {label}")
        return np.array(self.array[indices, self.columns[label], :], dtype=np.float32, copy=True)

    def pair(self, pair, current, indices):
        return float(current) * (self.column(pair[0], indices) - self.column(pair[1], indices))

    def roi(self, spec):
        if spec.get("mask"):
            image = nib.load(str(spec["mask"]))
            if image.shape != self.shape or not np.allclose(image.affine, self.affine, atol=1e-4):
                raise ValueError("ROI mask must match the task's padded T1 grid and affine")
            data = np.asarray(image.dataobj)
            selected = spec.get("labels")
            roi = np.isin(data[tuple(self.ijk.T)], selected) if selected else data[tuple(self.ijk.T)] > 0
        else:
            center = np.asarray(spec["center_ras_mm"], dtype=float)
            radius = float(spec["radius_mm"])
            if center.shape != (3,) or not np.isfinite(center).all() or not 0 < radius:
                raise ValueError("Invalid ROI center/radius")
            xyz = nib.affines.apply_affine(self.affine, self.ijk)
            roi = np.sum((xyz - center) ** 2, axis=1) <= radius ** 2
        if not roi.any():
            raise ValueError("ROI does not intersect predicted GM/WM voxels")
        return roi


def validate_montage(lf, pairs, currents, frequencies=None):
    if len(pairs) != 2 or any(len(p) != 2 for p in pairs):
        raise ValueError("Two electrode pairs are required")
    labels = [int(x) for p in pairs for x in p]
    if len(set(labels)) != 4:
        raise ValueError("The two channels require four distinct electrodes")
    if any(x not in lf.labels + [lf.reference] for x in labels):
        raise ValueError("Montage contains an unknown electrode")
    if len(currents) != 2 or not np.isfinite(currents).all() or min(currents) <= 0:
        raise ValueError("Both current amplitudes must be positive finite peak mA")
    if frequencies is not None and (len(frequencies) != 2 or not np.isfinite(frequencies).all() or min(frequencies) <= 0 or frequencies[0] == frequencies[1]):
        raise ValueError("TI requires two distinct positive carrier frequencies")


def statistics(values, roi=None, threshold=0.2):
    def stats(v):
        return {"mean_v_m": float(v.mean()), "max_v_m": float(v.max()),
                "p95_v_m": float(np.percentile(v, 95)), "voxels": int(v.size),
                "fraction_above_threshold": float(np.mean(v >= threshold))}
    result = {"all_gmwm": stats(values), "threshold_v_m": float(threshold)}
    if roi is not None:
        result["roi"] = stats(values[roi])
        if (~roi).any():
            result["non_roi"] = stats(values[~roi])
            result["mean_roi_non_roi_ratio"] = result["roi"]["mean_v_m"] / max(result["non_roi"]["mean_v_m"], 1e-12)
    return result


def synthesize(prediction_dir, output_dir, pairs, currents, frequencies=None, roi_spec=None, progress=print, threshold=0.2):
    lf = Leadfield(prediction_dir)
    validate_montage(lf, pairs, currents, frequencies)
    roi = lf.roi(roi_spec) if roi_spec else None
    values = np.empty(len(lf.ijk), dtype=np.float32)
    amplitudes = [np.empty_like(values), np.empty_like(values)]
    for start in range(0, len(values), 65536):
        sl = slice(start, min(start + 65536, len(values)))
        fields = [lf.pair(p, c, sl) for p, c in zip(pairs, currents)]
        values[sl] = max_ti(*fields)
        for k in range(2):
            amplitudes[k][sl] = np.linalg.norm(fields[k], axis=1)
        progress(f"TI 合成 {min(start + 65536, len(values))}/{len(values)}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in [("TI_max", values), ("channel_1_magnitude", amplitudes[0]), ("channel_2_magnitude", amplitudes[1])]:
        volume = np.zeros(lf.shape, dtype=np.float32)
        volume[tuple(lf.ijk.T)] = data
        image = nib.Nifti1Image(volume, lf.affine)
        image.header.set_xyzt_units("mm")
        image.header["descrip"] = b"NeuroGuide | V/m | currents: peak mA"
        path = output / f"{name}.nii.gz"
        temporary = output / f"{name}.partial.nii.gz"
        nib.save(image, str(temporary))
        temporary.replace(path)
        paths.append(str(path))
    if roi is not None:
        volume = np.zeros(lf.shape, dtype=np.uint8)
        volume[tuple(lf.ijk[roi].T)] = 1
        nib.save(nib.Nifti1Image(volume, lf.affine), str(output / "ROI.nii.gz"))
    manifest_path = lf.directory / "BrainLFNet_prediction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    result = {"prediction_dir": str(lf.directory), "padded_t1": manifest.get("padded_t1"),
              "pairs": [[int(x) for x in p] for p in pairs], "currents_peak_ma": list(map(float, currents)),
              "quantity": "max_directional_TI_envelope", "units": "V/m", "colormap": "jet",
              "field_assumption": "quasi_static_frequency_independent_leadfield", "roi": roi_spec,
              "overlays": paths, "statistics": statistics(values, roi, threshold)}
    if frequencies is not None:
        # Accept historical explicit metadata; amplitude calculation does not use it.
        result.update(frequencies_hz=list(map(float,frequencies)),beat_hz=abs(float(frequencies[0])-float(frequencies[1])))
    path = output / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(path)
    lf.close()
    return result


def optimize(prediction_dir, output_dir, roi_spec, options, frequencies=None, progress=print):
    """Elitist GA on distinct electrode labels and a discretized current ratio.

    Search evaluates a fixed stratified sample; top candidates are reranked on
    ALL predicted voxels. Seed, sampling and constraints are saved for replay.
    """
    from core.search_defaults import DEFAULT_SEARCH
    lf = Leadfield(prediction_dir)
    roi = lf.roi(roi_spec)
    if not (~roi).any():
        raise ValueError("Optimization requires a non-ROI avoidance region")
    seed = int(options.get("seed", 20261012))
    rng = np.random.default_rng(seed)
    candidates = list(map(int, options.get("electrodes", lf.labels + [lf.reference])))
    if len(set(candidates)) != len(candidates) or len(candidates) < 4 or any(x not in lf.labels + [lf.reference] for x in candidates):
        raise ValueError("Choose at least four distinct valid candidate electrodes")
    cap = float(options["channel_limit_ma"])
    budget = float(options['total_peak_ma']) if 'total_peak_ma' in options else None
    minimum = float(options.get("min_channel_ma", 0.1))
    if not np.isfinite([cap, minimum]).all() or minimum <= 0 or cap < minimum:
        raise ValueError('Each channel needs a finite positive current range')
    if budget is not None and (not np.isfinite(budget) or budget < 2*minimum or budget>2*cap):
        raise ValueError('Historical fixed-total request is outside the channel limits')
    lower, upper = (max(minimum,budget-cap),min(cap,budget-minimum)) if budget is not None else (minimum,cap)
    validate_montage(lf, [candidates[:2], candidates[2:4]], [cap,cap], frequencies)
    ratio_steps = int(options.get("ratio_steps", DEFAULT_SEARCH['ratio_steps']))
    if ratio_steps < 2:
        raise ValueError("At least two current-ratio steps are required")
    ratios = np.linspace(lower, upper, ratio_steps)
    threshold = float(options.get("threshold_v_m", 0.2))
    penalty = float(options.get("avoidance_weight", 1.0))
    if not np.isfinite([threshold, penalty]).all() or threshold <= 0 or penalty < 0:
        raise ValueError("Invalid target threshold or avoidance weight")
    goal = options.get("goal", "intensity")
    if goal not in ("focality", "intensity"):
        raise ValueError("Unknown optimization goal")
    sample_size = max(1, int(options.get("sample_per_region", DEFAULT_SEARCH['sample_per_region'])))
    groups = [np.flatnonzero(roi), np.flatnonzero(~roi)]
    sampled = [np.sort(rng.choice(g, min(sample_size, len(g)), replace=False)) for g in groups]
    indices = np.concatenate(sampled)
    split = len(sampled[0])
    bank = np.stack([lf.column(label, indices) for label in candidates], axis=1)
    if not np.isfinite(bank).all():
        raise ValueError("Leadfield sample contains non-finite values")
    cache = {}

    def canonical(g):
        a, b = sorted(g[:2]); c, d = sorted(g[2:4]); r = int(g[4])
        second = int(g[5]) if budget is None else None
        if (a, b) > (c, d):
            a,b,c,d=c,d,a,b
            if budget is None:r,second=second,r
            else:r=ratio_steps-1-r
        return (a,b,c,d,r,second) if budget is None else (a,b,c,d,r)

    def gene_currents(g):
        return [float(ratios[g[4]]),float(ratios[g[5]]) if budget is None else float(budget-ratios[g[4]])]

    def fitness(target, avoid):
        if goal == "intensity":
            return float(target.mean() - penalty * avoid.mean())
        # Threshold coverage difference with a penalty below desired ROI peak.
        shortfall = max(0.0, threshold - float(target.max())) / threshold
        return float(np.mean(target >= threshold) - penalty * np.mean(avoid >= threshold) - 2 * shortfall)

    def evaluate(g):
        g = canonical(g)
        if g not in cache:
            a,b,c,d=g[:4];i1,i2=gene_currents(g)
            v = max_ti(i1*(bank[:,a]-bank[:,b]),i2*(bank[:,c]-bank[:,d]))
            cache[g] = fitness(v[:split], v[split:])
        return cache[g]

    size, generations = int(options.get("population", DEFAULT_SEARCH['population'])), int(options.get("generations", DEFAULT_SEARCH['generations']))
    finalist_count = int(options.get("finalist_count", DEFAULT_SEARCH['finalist_count']))
    if finalist_count < 1:
        raise ValueError("At least one full-volume finalist is required")
    if size < 8 or generations < 1:
        raise ValueError("Population >=8 and generations >=1 are required")
    population = [canonical((*rng.choice(len(candidates),4,replace=False),*rng.integers(ratio_steps,size=2 if budget is None else 1))) for _ in range(size)]
    if budget is None:
        # Include full per-channel amplitude at initialization, without a total-current constraint.
        population[0]=canonical((*population[0][:4],ratio_steps-1,ratio_steps-1))
    history = []
    for generation in range(generations):
        population.sort(key=evaluate, reverse=True)
        history.append(float(evaluate(population[0])))
        progress(f"遗传搜索 {generation + 1}/{generations}；当前采样得分 {history[-1]:.5f}；已评估 {len(cache)} 个方案")
        elites = population[:max(2, size // 8)]
        next_population = list(elites)
        while len(next_population) < size:
            def tournament():
                return max((population[i] for i in rng.integers(size, size=3)), key=evaluate)
            p, q = tournament(), tournament()
            child = [p[i] if rng.random() < 0.5 else q[i] for i in range(4)]
            used = set()
            for i in range(4):
                if child[i] in used or rng.random() < 0.2:
                    available = [j for j in range(len(candidates)) if j not in used and j not in child[i+1:]]
                    if not available:
                        available = [j for j in range(len(candidates)) if j not in used]
                    child[i] = int(rng.choice(available))
                used.add(child[i])
            amplitudes=[]
            for index in range(4,len(p)):
                value=p[index] if rng.random()<.5 else q[index]
                if rng.random()<.3:value=int(rng.integers(ratio_steps))
                amplitudes.append(value)
            next_population.append(canonical((*child,*amplitudes)))
        population = next_population
    for g in population:
        evaluate(g)
    finalists = sorted(cache, key=cache.get, reverse=True)[:min(finalist_count, len(cache))]
    ranking = []
    for k, g in enumerate(finalists):
        a,b,c,d=g[:4]
        pairs = [[candidates[a], candidates[b]], [candidates[c], candidates[d]]]
        currents = gene_currents(g)
        values = np.empty(len(lf.ijk), dtype=np.float32)
        for start in range(0, len(values), 65536):
            sl = slice(start, min(start + 65536, len(values)))
            values[sl] = max_ti(*(lf.pair(p, c, sl) for p, c in zip(pairs, currents)))
        ranking.append({"pairs": pairs, "currents_peak_ma": currents,
                        "sample_score": cache[g], "full_score": fitness(values[roi], values[~roi]),
                        "statistics": statistics(values, roi, threshold)})
        progress(f"完整导联场复算 {k + 1}/{len(finalists)}")
    ranking.sort(key=lambda x: x["full_score"], reverse=True)
    winner = ranking[0]
    result = synthesize(prediction_dir, output_dir, winner["pairs"], winner["currents_peak_ma"], frequencies, roi_spec, progress, threshold)
    result["optimization"] = {"method": "leadfield_elitist_genetic_algorithm", "basis_doi": "10.1088/1741-2552/ac89b3",
                              "reproduction_scope": "method-level adaptation; operators and objective documented locally",
                              "options": options, "current_constraint":'independent_channel_limits' if budget is None else 'historical_fixed_total',
                              "seed": seed, "goal": goal, "evaluated": len(cache),
                              "sample_roi_indices": sampled[0].tolist(), "sample_non_roi_indices": sampled[1].tolist(),
                              "history": history, "ranking": ranking, "global_optimum_guaranteed": False}
    Path(result["result_path"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lf.close()
    return result
