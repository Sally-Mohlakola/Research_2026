"""Run renderer checks in isolation: Mitsuba variants are process-global."""
import subprocess
import sys
import unittest
from pathlib import Path


class RenderTests(unittest.TestCase):
    def test_constant_illumination_and_neural_exit(self):
        script = r"""
import render_boundary as rb
import mitsuba as mi
import numpy as np
import torch
from ground_truth.brilliant_geometry import make_round_brilliant
from config.parameters import get_diamond_parameters
from neural.boundary_model import BoundaryModel

torch.set_num_threads(2)
v,f=make_round_brilliant()
model=BoundaryModel(v,f)
checkpoint=dict(vertices=torch.from_numpy(v),faces=torch.from_numpy(f.astype(np.int64)),
                input_radius=float(np.linalg.norm(v,axis=1).max()),
                metadata=dict(parameters=get_diamond_parameters('round_diamond_gia'),dispersion=True,max_depth=128))
full,stone,mesh=rb.make_scene(checkpoint,8,8)
scene=mi.load_dict(dict(type='scene',diamond=mesh,sensor=full.sensors()[0],environment=dict(type='constant',radiance=1.)))
image,stats,_=rb.render(scene,stone,mesh,model,checkpoint,8,8,128,5,mode='analytic',scene_depth=4)
assert np.isfinite(image).all()
assert np.max(np.abs(image.mean((0,1))-1))<.1
assert stats['analytic_escaped']>0
class UnitExit:
    normal=model.normal
    def sample(self,x,generator=None):
        n=len(x)
        facet=int(torch.argmax(model.normal[:,2]))
        return dict(escaped=torch.ones(n,dtype=torch.bool),exit_facet=torch.full((n,),facet),
                    exit_position=model.triangles[facet].mean(0).repeat(n,1),
                    exit_direction=model.normal[facet].repeat(n,1),throughput=torch.ones(n))
image,stats,_=rb.render(scene,stone,mesh,UnitExit(),checkpoint,8,8,128,5,scene_depth=4)
assert np.max(np.abs(image.mean((0,1))-1))<.1
assert stats['neural_queries']>0 and stats['predicted_reentry']==0
assert stats['neural_escaped']==stats['neural_queries']
"""
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
