"""Checks for the learned dispersion head and its four-lane rendering path."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from neural.dispersion_model import DispersionModel, coordinates, decode, diamond_index


class DispersionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.model = DispersionModel(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                                     np.array([[0, 1, 2]]), width=16, depth=3)
        self.buffers = (self.model.triangles, self.model.tangent, self.model.bitangent, self.model.normal)

    def test_coordinates_invert_decode(self):
        y = torch.randn(256, 4)*torch.tensor([1., 1., .5, .5])
        facet = torch.zeros(256, dtype=torch.long)
        position, direction = decode(*self.buffers, facet, y)
        recovered, clipped = coordinates(*self.buffers, facet, position, direction)
        self.assertFalse(clipped.any())
        torch.testing.assert_close(recovered, y, atol=2e-4, rtol=1e-4)

    def test_index_matches_published_values(self):
        # Sellmeier fit quoted in bsdf/dispersion.py against reference indices.
        for wavelength, expected in ((430.8, 2.4515), (589.3, 2.4175), (686.7, 2.4073)):
            self.assertAlmostEqual(float(diamond_index(torch.tensor(wavelength))), expected, places=3)

    def test_shift_loss_only_on_usable_pairs(self):
        n = 64
        args = (torch.zeros(n, 10), torch.zeros(n, dtype=torch.long), torch.zeros(n, 4),
                torch.full((n,), 450.), torch.ones(n, dtype=torch.bool), torch.full((n, 4), float('nan')))
        gate, shift = self.model.losses(*args, torch.zeros(n, dtype=torch.bool))
        self.assertTrue(torch.isfinite(gate))
        self.assertEqual(float(shift.detach()), 0.)

    def test_samples_stay_on_the_facet_and_outward(self):
        n = 128
        s = self.model.sample(torch.randn(n, 10), torch.zeros(n, dtype=torch.long), torch.randn(n, 4),
                              torch.full((n,), 700.), torch.Generator().manual_seed(4))
        torch.testing.assert_close(s['exit_position'][:, 2], torch.zeros(n))
        self.assertTrue((s['exit_position'][:, :2] >= 0).all())
        self.assertTrue((s['exit_position'][:, :2].sum(-1) <= 1 + 1e-6).all())
        self.assertTrue((s['exit_direction'][:, 2] > 0).all())


class DispersionRenderTests(unittest.TestCase):
    def test_four_lanes_conserve_energy_under_constant_light(self):
        # White furnace: under unit constant radiance every lane, whether it
        # stays, splits or reflects at the first surface, must still carry one.
        # This checks the lane masks, the Fresnel ratios and the lane average.
        script = r"""
import render_boundary as rb
import mitsuba as mi
import numpy as np
import torch
from ground_truth.brilliant_geometry import make_round_brilliant
from config.parameters import get_diamond_parameters
from neural.boundary_model import BoundaryModel
from neural.dispersion_model import DispersionModel

torch.set_num_threads(2)
torch.manual_seed(5)
v,f=make_round_brilliant()
model=BoundaryModel(v,f)
checkpoint=dict(vertices=torch.from_numpy(v),faces=torch.from_numpy(f.astype(np.int64)),
                input_radius=float(np.linalg.norm(v,axis=1).max()),
                metadata=dict(parameters=get_diamond_parameters('round_diamond_gia'),dispersion=True,max_depth=128))
full,stone,mesh=rb.make_scene(checkpoint,8,8)
scene=mi.load_dict(dict(type='scene',diamond=mesh,sensor=full.sensors()[0],environment=dict(type='constant',radiance=1.)))
class UnitExit:
    triangles,tangent,bitangent,normal=model.triangles,model.tangent,model.bitangent,model.normal
    def sample(self,x,generator=None,deterministic=False,facet=None):
        n=len(x)
        facet=int(torch.argmax(model.normal[:,2]))
        return dict(escaped=torch.ones(n,dtype=torch.bool),exit_facet=torch.full((n,),facet),
                    exit_position=model.triangles[facet].mean(0).repeat(n,1),
                    exit_direction=model.normal[facet].repeat(n,1),throughput=torch.ones(n))
head=DispersionModel(v,f,width=16,depth=3).eval()
image,stats,_=rb.render(scene,stone,mesh,UnitExit(),checkpoint,8,8,128,5,scene_depth=4,dispersion=head)
assert np.max(np.abs(image.mean((0,1))-1))<.1, image.mean((0,1))
assert stats['dispersion_lanes']>0 and stats['dispersion_stays']>0 and stats['dispersion_splits']>0
"""
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
