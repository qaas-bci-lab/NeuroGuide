import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from core.ti import Leadfield, max_ti, synthesize, optimize, validate_montage
from core.individual_pipeline import validate_tissues, require_asset, select_device


def fixture(root):
    root = Path(root); prediction = root / 'prediction'; prediction.mkdir()
    shape = (6, 5, 4)
    ijk = np.argwhere(np.ones(shape, dtype=bool))
    affine = np.diag([1., 1., 1., 1.]); affine[:3, 3] = [-3, -2, -1]
    values = np.random.default_rng(8).normal(0, 0.25, size=(len(ijk), 5, 3)).astype(np.float32)
    np.save(prediction / 'leadfield_gmwm.npy', values)
    np.savez(prediction / 'leadfield_geometry.npz', valid_ijk=ijk, affine=affine,
             grid_shape=shape, active_electrode_labels=[1, 2, 3, 4, 5], reference_electrode_label=72)
    t1 = root / 'T1.nii.gz'; nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), t1)
    (prediction / 'BrainLFNet_prediction.json').write_text(json.dumps({'padded_t1': str(t1)}))
    return prediction, t1


class TIPhysicsTests(unittest.TestCase):
    def test_parallel_antiparallel_orthogonal_and_zero(self):
        a = np.array([[2, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        b = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        np.testing.assert_allclose(max_ti(a, b), [2, 2, np.sqrt(2), 0, 2], rtol=1e-6)

    def test_against_directional_brute_force(self):
        rng = np.random.default_rng(15)
        a, b = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
        directions = rng.normal(size=(40000, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
        brute = (2 * np.minimum(np.abs(a @ directions.T), np.abs(b @ directions.T))).max(axis=1)
        exact = max_ti(a, b)
        self.assertTrue(np.all(exact >= brute - 1e-5))
        np.testing.assert_allclose(exact, brute, atol=0.02, rtol=0.01)
        np.testing.assert_allclose(exact, max_ti(b, a), rtol=1e-6)
        np.testing.assert_allclose(exact, max_ti(a, -b), rtol=1e-6)
        np.testing.assert_allclose(2 * exact, max_ti(2*a, 2*b), rtol=1e-6)

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            max_ti([[float('nan'), 0, 0]], [[0, 0, 0]])


class TaskTests(unittest.TestCase):
    def test_auto_gpu_and_cpu_fallback(self):
        with patch('core.cuda_compat.run_gpu_probe', return_value={'usable':False,'reason':'unavailable'}):
            self.assertEqual(select_device('auto', lambda _: None).type, 'cpu')
        with patch('core.cuda_compat.run_gpu_probe', return_value={'usable':True}):
            self.assertEqual(select_device('auto', lambda _: None).type, 'cuda')
        with patch('core.cuda_compat.run_gpu_probe', return_value={'usable':False,'reason':'probe_failed'}):
            self.assertEqual(select_device('auto', lambda _: None).type, 'cpu')

    def test_labels_and_lfs_pointer(self):
        np.testing.assert_array_equal(validate_tissues(np.arange(7)), [0, 1, 2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            validate_tissues([1.5, 2])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'checkpoint.pt'
            p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123')
            with self.assertRaises(ValueError):
                require_asset(p)

    def test_pair_reference_and_current_units(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = fixture(directory); lf = Leadfield(prediction)
            np.testing.assert_allclose(lf.pair([1, 72], 2, slice(None)), 2 * lf.array[:, 0])
            np.testing.assert_allclose(lf.pair([1, 2], 1, slice(None)), lf.array[:, 0] - lf.array[:, 1])
            with self.assertRaises(ValueError):
                validate_montage(lf, [[1, 2], [2, 3]], [1, 1], [2000, 2010])
            with self.assertRaises(ValueError):
                validate_montage(lf, [[1, 2], [3, 4]], [1, 1], [2000, 2000])
            lf.close()

    def test_two_channels_can_each_reach_two_ma_without_total_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory)
            opts={'channel_limit_ma':2.,'min_channel_ma':.1,'ratio_steps':3,
                  'population':64,'generations':8,'seed':12,'goal':'intensity','avoidance_weight':0.,
                  'electrodes':[1,2,3,72],'sample_per_region':1000}
            roi={'center_ras_mm':[0,0,0],'radius_mm':1.5}
            result=optimize(prediction,Path(directory)/'independent',roi,opts,[2000,2010],lambda _:None)
            self.assertEqual(result['currents_peak_ma'],[2.,2.])
            self.assertEqual(result['optimization']['current_constraint'],'independent_channel_limits')
            self.assertNotIn('total_peak_ma',result['optimization']['options'])
            for ranked in result['optimization']['ranking']:
                self.assertTrue(all(.1<=c<=2 for c in ranked['currents_peak_ma']))

    def test_amplitude_is_independent_of_optional_frequency_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction,_=fixture(directory)
            a=synthesize(prediction,Path(directory)/'without',[[1,2],[3,72]],[2,1],progress=lambda _:None)
            b=synthesize(prediction,Path(directory)/'with',[[1,2],[3,72]],[2,1],[1900,2000],progress=lambda _:None)
            np.testing.assert_array_equal(nib.load(a['overlays'][0]).get_fdata(),nib.load(b['overlays'][0]).get_fdata())
            self.assertNotIn('frequencies_hz',a);self.assertNotIn('beat_hz',a)

    def test_configurable_full_volume_finalists(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = fixture(directory)
            options = {'total_peak_ma': 2., 'channel_limit_ma': 2., 'population': 8,
                       'generations': 1, 'finalist_count': 1, 'seed': 6}
            result = optimize(prediction, Path(directory)/'result', {'center_ras_mm': [0,0,0], 'radius_mm': 1.5},
                              options, [2000,2010], lambda _: None)
            self.assertEqual(len(result['optimization']['ranking']), 1)

    def test_synthesis_geometry_and_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = fixture(directory)
            a = synthesize(prediction, Path(directory)/'a', [[1, 2], [3, 72]], [1, 0.5], [2000, 2010], progress=lambda _: None)
            b = synthesize(prediction, Path(directory)/'b', [[1, 2], [3, 72]], [2, 1], [2000, 2010], progress=lambda _: None)
            i, j = nib.load(a['overlays'][0]), nib.load(b['overlays'][0])
            lf = Leadfield(prediction)
            np.testing.assert_allclose(i.affine, lf.affine)
            self.assertEqual(i.shape, lf.shape)
            np.testing.assert_allclose(j.get_fdata(), 2 * i.get_fdata(), rtol=1e-6)
            self.assertEqual(a['beat_hz'], 10)
            self.assertEqual(a['colormap'], 'jet')
            lf.close()

    def test_roi_affine_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, t1 = fixture(directory); lf = Leadfield(prediction)
            wrong = Path(directory) / 'wrong.nii.gz'
            nib.save(nib.Nifti1Image(np.ones(lf.shape), np.eye(4)), wrong)
            with self.assertRaises(ValueError):
                lf.roi({'mask': str(wrong)})
            with self.assertRaises(ValueError):
                lf.roi({'center_ras_mm': [100, 100, 100], 'radius_mm': 1})
            lf.close()

    def test_genetic_search_replay_constraints_and_small_exhaustive_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = fixture(directory)
            roi = {'center_ras_mm': [0, 0, 0], 'radius_mm': 1.5}
            opts = {'total_peak_ma': 2., 'channel_limit_ma': 1.5, 'min_channel_ma': .5,
                    'population': 24, 'generations': 6, 'ratio_steps': 3, 'seed': 12,
                    'electrodes': [1, 2, 3, 72], 'sample_per_region': 1000, 'goal': 'intensity', 'avoidance_weight': .5}
            a = optimize(prediction, Path(directory)/'opt_a', roi, opts, [2000, 2010], lambda _: None)
            b = optimize(prediction, Path(directory)/'opt_b', roi, opts, [2000, 2010], lambda _: None)
            self.assertEqual(a['pairs'], b['pairs']); self.assertEqual(a['currents_peak_ma'], b['currents_peak_ma'])
            self.assertEqual(len({v for pair in a['pairs'] for v in pair}), 4)
            self.assertAlmostEqual(sum(a['currents_peak_ma']), 2)
            self.assertLessEqual(max(a['currents_peak_ma']), 1.5)
            lf = Leadfield(prediction); selected = lf.roi(roi)
            best = -np.inf
            for other in [2, 3, 72]:
                second = [x for x in [2, 3, 72] if x != other]
                for current in [.5, 1., 1.5]:
                    v = max_ti(lf.pair([1, other], current, slice(None)), lf.pair(second, 2-current, slice(None)))
                    best = max(best, float(v[selected].mean() - .5*v[~selected].mean()))
            self.assertAlmostEqual(a['optimization']['ranking'][0]['full_score'], best, places=6)
            lf.close()


if __name__ == '__main__':
    unittest.main()
