import unittest
import numpy as np
from gather_boundary import build_scene, gather, dielectric
from config.parameters import get_diamond_parameters
from neural.boundary_data import split_entries
import mitsuba as mi


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = get_diamond_parameters('round_diamond_gia')
        cls.scene, v, _ = build_scene(cls.params)
        cls.radius = float(np.linalg.norm(v, axis=1).max())
        cls.data, _ = gather(cls.scene, cls.radius, cls.params, 16, 4, 64, 17)

    def test_escape_physics(self):
        a = self.data
        escaped = a['status'] == 0
        self.assertTrue(escaped.any())
        self.assertFalse((a['status'] == 2).any())
        np.testing.assert_allclose(a['throughput'][escaped], 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(a['exit_direction'][escaped], axis=1), 1, atol=1e-5)
        self.assertTrue((np.sum(a['entry_direction']*a['entry_normal'], axis=1)<0).all())
        self.assertTrue((np.sum(a['exit_direction'][escaped]*a['exit_normal'][escaped], axis=1)>0).all())

    def test_reproducible(self):
        other, _ = gather(self.scene, self.radius, self.params, 16, 4, 64, 17)
        for key in self.data:
            np.testing.assert_array_equal(self.data[key], other[key])

    def test_truncation_is_not_escape(self):
        a, _ = gather(self.scene, self.radius, self.params, 16, 4, 2, 17)
        self.assertTrue((a['status'] == 1).any())
        self.assertTrue((a['throughput'][a['status'] != 0] == 0).all())
        self.assertTrue(np.isnan(a['exit_position'][a['status'] != 0]).all())

    def test_grouped_split(self):
        train, val = split_entries(self.data['entry_id'])
        self.assertFalse(set(self.data['entry_id'][train]) & set(self.data['entry_id'][val]))
        self.assertTrue(train.any() and val.any())

    def test_fresnel_normal_incidence_and_tir(self):
        si = mi.SurfaceInteraction3f()
        si.wi = mi.Vector3f(0, 0, 1)
        f, _, _, _ = dielectric(si, 2.4)
        self.assertAlmostEqual(f, ((2.4-1)/(2.4+1))**2, places=6)
        si.wi = mi.Vector3f(np.sqrt(.75), 0, -.5)
        f, _, _, _ = dielectric(si, 2.4)
        self.assertEqual(f, 1.)


if __name__ == '__main__':
    unittest.main()
