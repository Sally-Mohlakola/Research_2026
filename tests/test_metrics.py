import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.evaluate_render import (block_bootstrap_correlation, bootstrap_correlation, noise_corrected,
                                      noise_free_contrast)
from analysis.compare_runs import paired
from analysis.quality_vs_time import curves, error_model, summary
from utils.run_record import geometric_facets, mesh_record


# a converged "true" image and noisy independent halves of it, as luminance-like positive values
def noisy_halves(truth, noise, rng):
    return truth + rng.normal(0, noise, truth.shape), truth + rng.normal(0, noise, truth.shape)


# a small synthetic stone: 32 x 32 pixels, every pixel on the stone, eight facets of 128 pixels each
def synthetic_stone():
    mask = np.ones((32, 32), bool)
    triangle = (np.arange(32*32)//128).reshape(32, 32)
    return mask, triangle, np.arange(8)


# an RGB image whose luminance is `values` (equal channels, so luminance equals the value)
def grey(values):
    return np.repeat(values[..., None], 3, axis=-1)


class NoiseCorrections(unittest.TestCase):
    # with noise far below the signal, the noise-free contrast recovers the true contrast
    def test_noise_free_contrast_recovers_truth(self):
        rng = np.random.default_rng(1)
        truth = rng.gamma(.5, 1., 200000)
        y1, y2 = noisy_halves(truth, .8, rng)
        true_contrast = truth.std()/truth.mean()
        self.assertAlmostEqual(noise_free_contrast(y1, y2), true_contrast, delta=.02*true_contrast)
        self.assertGreater(y1.std()/y1.mean(), 1.2*true_contrast)

    # two halves with no shared signal give zero contrast rather than a noise-inflated one
    def test_noise_free_contrast_of_pure_noise_is_zero(self):
        rng = np.random.default_rng(2)
        y1, y2 = 1 + rng.normal(0, .3, 100000), 1 + rng.normal(0, .3, 100000)
        self.assertLess(noise_free_contrast(y1, y2), .02)

    # the disattenuated correlation and the debiased error recover the converged images' values
    def test_disattenuation_and_debiasing(self):
        rng = np.random.default_rng(3)
        a = rng.gamma(.5, 1., 200000)
        b = .6*a + .4*rng.gamma(.5, 1., 200000)
        a1, a2 = noisy_halves(a, .5, rng)
        n1, n2 = noisy_halves(b, .5, rng)
        result = noise_corrected(a1, a2, n1, n2)
        self.assertAlmostEqual(result['disattenuated_correlation'], np.corrcoef(a, b)[0, 1], delta=.02)
        self.assertAlmostEqual(result['debiased_mse'], ((a - b)**2).mean(), delta=.02)
        self.assertGreater(((a1 - n1)**2).mean(), 2*((a - b)**2).mean())


class Intervals(unittest.TestCase):
    # a bootstrap interval brackets its own point estimate and the truth on a large sample
    def test_facet_bootstrap_brackets_estimate(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=400)
        y = .5*x + rng.normal(size=400)
        low, high = bootstrap_correlation(x, y, repeats=500)
        r = np.corrcoef(x, y)[0, 1]
        self.assertLess(low, r)
        self.assertGreater(high, r)

    # the pixel-block bootstrap gives an ordered interval around the point estimate
    def test_block_bootstrap_brackets_estimate(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(64, 64))
        y = x + rng.normal(size=(64, 64))
        mask = np.ones((64, 64), bool)
        low, high = block_bootstrap_correlation(x, y, mask, block=8, repeats=300)
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        self.assertLess(low, r)
        self.assertGreater(high, r)


class PairedComparison(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.mask, self.triangle, self.facets = synthetic_stone()
        truth = rng.gamma(1., 1., (32, 32)) + 1.
        self.analytic = {k: grey(v) for k, v in zip(('a1', 'a2'), noisy_halves(truth, .2, rng))}
        good = noisy_halves(truth, .2, rng)
        bad = noisy_halves(rng.permutation(truth.ravel()).reshape(32, 32), .2, rng)
        self.good = {k: grey(v) for k, v in zip(('n1', 'n2'), good)}
        self.bad = {k: grey(v) for k, v in zip(('n1', 'n2'), bad)}

    def run_pair(self, first, second):
        rows, _ = paired(dict(self.analytic, **first), second, self.mask, self.triangle, self.facets,
                         block=8, repeats=300)
        return {r['metric']: r for r in rows}

    # a run compared with itself differs by exactly zero, with a zero-width interval
    def test_run_against_itself_is_zero(self):
        rows = self.run_pair(self.good, self.good)
        for row in rows.values():
            self.assertEqual(row['difference'], 0.)
            self.assertEqual(row['ci95_low'], 0.)
            self.assertEqual(row['ci95_high'], 0.)
            self.assertFalse(row['interval_excludes_zero'])

    # a run that matches the reference beats a scrambled one, and the interval says so
    def test_clear_difference_is_detected(self):
        rows = self.run_pair(self.good, self.bad)
        self.assertGreater(rows['pixel_correlation']['difference'], .5)
        self.assertTrue(rows['pixel_correlation']['interval_excludes_zero'])


class QualityModel(unittest.TestCase):
    # the error model recovers a known per-seed variance and a known bias, and the curves behave
    def test_error_model_recovers_variance_and_bias(self):
        rng = np.random.default_rng(7)
        mask = np.ones((64, 64), bool)
        truth = rng.gamma(1., 1., (64, 64)) + 1.
        offset = .3
        analytic = np.stack([grey(truth + rng.normal(0, .5, truth.shape)) for _ in range(8)])
        neural = np.stack([grey(truth + offset + rng.normal(0, .2, truth.shape)) for _ in range(8)])
        model = error_model(analytic, neural, mask)
        self.assertAlmostEqual(model['analytic_variance_per_seed'], .25, delta=.02)
        self.assertAlmostEqual(model['neural_variance_per_seed'], .04, delta=.005)
        self.assertAlmostEqual(model['neural_bias_squared'], offset**2, delta=.01)

        seconds = dict(analytic=2., neural=1.)
        rows = curves(model, seconds, spp=8, max_seeds=512)
        analytic_errors = [r['relative_rmse'] for r in rows if r['mode'] == 'analytic']
        neural_errors = [r['relative_rmse'] for r in rows if r['mode'] == 'neural']
        self.assertTrue(all(b < a for a, b in zip(analytic_errors, analytic_errors[1:])))
        self.assertGreaterEqual(min(neural_errors), offset/model['reference_mean'] - 1e-9)
        result = summary(model, seconds)
        self.assertAlmostEqual(result['neural_error_floor'], offset/model['reference_mean'], delta=.01)
        self.assertAlmostEqual(result['analytic_seconds_to_reach_neural_floor'],
                               2.*model['analytic_variance_per_seed']/model['neural_bias_squared'], places=6)


class Geometry(unittest.TestCase):
    # the round brilliant's 64 triangles make the 57 facets of a round brilliant
    def test_round_brilliant_facets(self):
        record = mesh_record('round_diamond_gia')
        self.assertEqual(record['triangles'], 64)
        self.assertEqual(record['geometric_facets'], 57)

    # splitting one flat square into two triangles still counts as one facet
    def test_coplanar_triangles_merge(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], float)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]])
        labels = geometric_facets(vertices, faces)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == '__main__':
    unittest.main()
