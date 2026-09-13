import math
import unittest
import numpy as np
import torch
from neural.boundary_model import BoundaryModel
from neural.boundary_prior import BoundaryPrior, sample_mixture
from ground_truth.brilliant_geometry import make_round_brilliant


class PriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        v,f=make_round_brilliant()
        self.model=BoundaryModel(v,f,width=16,components=2)
        nt,np_=4,8
        edges=torch.linspace(0,math.pi,nt+1)
        sa=((edges[:-1].cos()-edges[1:].cos())[:,None].expand(-1,np_)*2*math.pi/np_).flatten()
        self.prior=BoundaryPrior((sa/(4*math.pi))[None].expand(nt*np_,-1),nt,np_,self.model)

    def test_prior_is_uniform_in_solid_angle(self):
        x=torch.zeros(12000,10)
        x[:,5]=-1
        f,p,d=self.prior.sample(x,torch.Generator().manual_seed(12))
        self.assertLess(abs(float(d[:,2].mean())),.02)
        self.assertLess(abs(float(d[:,2].square().mean())-1/3),.02)
        self.assertTrue(((d*self.model.normal[f]).sum(-1)>0).all())
        self.assertTrue(torch.isfinite(self.prior.log_pdf(x,f,d)).all())

    def test_jacobians(self):
        tri=self.model.triangles[0].double()
        z=torch.tensor([.3,-.4],dtype=torch.double)
        def point(z):
            bary=torch.cat([z,torch.zeros(1,dtype=z.dtype)]).softmax(-1)
            return (tri*bary[:,None]).sum(0)
        jac=torch.autograd.functional.jacobian(point,z)
        actual=torch.linalg.vector_norm(torch.cross(jac[:,0],jac[:,1],dim=0))
        bary=torch.cat([z,torch.zeros(1,dtype=z.dtype)]).softmax(-1)
        expected=torch.linalg.vector_norm(torch.cross(tri[0]-tri[2],tri[1]-tri[2],dim=0))*bary.prod()
        torch.testing.assert_close(actual,expected)
        def direction(s):
            return torch.nn.functional.normalize(torch.cat([s,torch.ones(1,dtype=s.dtype)]),dim=0)
        jac=torch.autograd.functional.jacobian(direction,z)
        actual=torch.linalg.vector_norm(torch.cross(jac[:,0],jac[:,1],dim=0))
        torch.testing.assert_close(actual,(1+z.square().sum()).pow(-1.5))

    def test_mixture_preserves_mass(self):
        x=torch.zeros(16000,10)
        x[:,5]=-1
        result=sample_mixture(self.model,self.prior,x,.25,torch.Generator().manual_seed(4))
        expected=float(self.model.escape(self.model.encoder(x[:1])).sigmoid().detach())
        self.assertLess(abs(float(result['throughput'].mean())-expected),.025)
        self.assertTrue((result['throughput']<=1/.75+1e-5).all())

    def test_zero_fraction_is_original_sampler(self):
        x=torch.zeros(32,10)
        a=sample_mixture(self.model,self.prior,x,0,torch.Generator().manual_seed(2))
        b=self.model.sample(x,torch.Generator().manual_seed(2))
        for key in a:
            torch.testing.assert_close(a[key],b[key])
        with self.assertRaises(ValueError):
            sample_mixture(self.model,self.prior,x,1)


if __name__=='__main__':
    unittest.main()
