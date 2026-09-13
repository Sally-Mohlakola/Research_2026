"""World-direction RDM proposal for boundary transport, with area/solid-angle PDF.

Unlike the legacy first-hit-local depth>k RDM, this table is conditioned on entry
transmission and escape. It must be built from the boundary model TRAIN split.
The table ignores position and wavelength; importance correction preserves the
learned spatial/spectral target. This does not correct the target toward physics.
"""
import math
import torch


class BoundaryPrior:
    def __init__(self, pmf, theta_bins, phi_bins, model):
        self.pmf = torch.as_tensor(pmf, dtype=torch.float32)
        self.theta_bins, self.phi_bins = int(theta_bins), int(phi_bins)
        self.model = model
        edges = torch.linspace(0, math.pi, self.theta_bins+1)
        self.solid_angle = ((edges[:-1].cos()-edges[1:].cos())[:,None].expand(-1,self.phi_bins)*2*math.pi/self.phi_bins).flatten()
        t = model.triangles
        self.area = torch.linalg.vector_norm(torch.cross(t[:,1]-t[:,0],t[:,2]-t[:,0],dim=-1),dim=-1)/2
        size = self.theta_bins*self.phi_bins
        if self.pmf.shape != (size,size) or not torch.isfinite(self.pmf).all() or not (self.pmf>0).all():
            raise ValueError('Prior must be a finite, positive square directional PMF')
        if not torch.allclose(self.pmf.sum(-1),torch.ones(size),atol=1e-5):
            raise ValueError('Prior rows must sum to one')

    def bins(self, direction):
        theta = torch.acos(direction[:,2].clamp(-1,1))
        phi = torch.atan2(direction[:,1],direction[:,0])
        ti = (theta/math.pi*self.theta_bins).long().clamp(0,self.theta_bins-1)
        pi = ((phi+math.pi)/(2*math.pi)*self.phi_bins).long().clamp(0,self.phi_bins-1)
        return ti*self.phi_bins+pi

    def log_pdf(self, x, facet, direction):
        row, column = self.bins(-x[:,3:6]), self.bins(direction)
        cosine = (direction @ self.model.normal.T).clamp_min(0)
        projected_area = (cosine*self.area).sum(-1)
        # Choose facet by projected area, then sample uniformly over its area.
        cosine_f = cosine.gather(1,facet[:,None]).squeeze(-1)
        return (self.pmf[row,column]/self.solid_angle[column]).log()+cosine_f.log()-projected_area.log()

    def sample(self, x, generator=None):
        n = len(x)
        row = self.bins(-x[:,3:6])
        cell = torch.multinomial(self.pmf[row],1,generator=generator).squeeze(-1)
        u = torch.rand((n,4),generator=generator)
        lo = (cell//self.phi_bins)*math.pi/self.theta_bins
        hi = lo+math.pi/self.theta_bins
        c = (1-u[:,0])*lo.cos()+u[:,0]*hi.cos()
        phi = -math.pi+((cell%self.phi_bins)+u[:,1])*2*math.pi/self.phi_bins
        sint = (1-c.square()).clamp_min(0).sqrt()
        direction = torch.stack([sint*phi.cos(),sint*phi.sin(),c],-1)
        facet_mass = (direction @ self.model.normal.T).clamp_min(0)*self.area
        facet = torch.multinomial(facet_mass,1,generator=generator).squeeze(-1)
        root = u[:,2].sqrt()
        bary = torch.stack([1-root,root*(1-u[:,3]),root*u[:,3]],-1)
        position = (self.model.triangles[facet]*bary[:,:,None]).sum(1)
        return facet, position, direction


def sample_mixture(model, prior, x, alpha, generator=None):
    if not 0 <= alpha < 1:
        raise ValueError('Prior fraction must be in [0,1) to retain neural support')
    result = model.sample(x,generator)
    if alpha == 0:
        return result
    use_prior = torch.rand(len(x),generator=generator) < alpha
    facet, position, direction = prior.sample(x,generator)
    for key, values in [('exit_facet',facet),('exit_position',position),('exit_direction',direction)]:
        mask = use_prior if values.ndim == 1 else use_prior[:,None]
        result[key] = torch.where(mask,values,result[key])
    log_target = model.log_exit_pdf(x,result['exit_facet'],result['exit_position'],result['exit_direction'])
    log_prior = prior.log_pdf(x,result['exit_facet'],result['exit_direction'])
    log_proposal = torch.logaddexp(log_target+math.log1p(-alpha),log_prior+math.log(alpha))
    weight = (log_target-log_proposal).exp()
    if not torch.isfinite(weight).all():
        raise RuntimeError('Nonfinite boundary importance weight')
    result['throughput'] = result['escaped'].float()*weight
    # Original barycentrics no longer describe replaced samples.
    result.pop('barycentric',None)
    return result
