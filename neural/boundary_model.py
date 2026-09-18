"""Conditional finite-depth escape model. Densities are in transformed coordinates.

Factorization: escape Bernoulli, exit-triangle categorical, then a joint Gaussian
mixture over two barycentric logits and two outgoing tangent slopes. Sampling
always yields a surface point and an outward unit direction. This smooth model
approximates singular specular transport; it is not an unbiased physical BSDF.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def frames(vertices, faces):
    triangles = np.asarray(vertices)[np.asarray(faces)]
    edge = triangles[:, 1] - triangles[:, 0]
    tangent = edge / np.linalg.norm(edge, axis=1, keepdims=True)
    normal = np.cross(edge, triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return triangles, tangent, np.cross(normal, tangent), normal


def encode_data(data, vertices, faces):
    radius = float(np.linalg.norm(vertices, axis=1).max())
    x = np.concatenate([data['entry_position']/radius, data['entry_direction'],
                        data['entry_normal'], (data['wavelength_nm'][:, None]-595)/235], axis=1)
    valid = data['status'] != 2
    escaped = data['status'] == 0
    facet = np.where(escaped, data['exit_facet'], 0).astype(np.int64)
    if ((facet < 0) | (facet >= len(faces))).any():
        raise ValueError('Invalid exit facet')
    tri, t, b, n = frames(vertices, faces)
    y = np.zeros((len(x), 4), dtype=np.float32)
    ids = np.flatnonzero(escaped)
    clipped = 0
    if len(ids):
        # Only len(faces) distinct triangles exist, so one pseudo-inverse per
        # triangle replaces the per-record least-squares solve exactly.
        basis = (tri[:, :2]-tri[:, 2:3]).transpose(0, 2, 1)
        pseudo_inverse = np.linalg.pinv(basis)
        f = facet[ids]
        uv = np.einsum('nij,nj->ni', pseudo_inverse[f], data['exit_position'][ids]-tri[f, 2])
        bary = np.concatenate([uv, 1-uv.sum(1, keepdims=True)], axis=1)
        d = data['exit_direction'][ids]
        cosine = np.einsum('nj,nj->n', d, n[f])
        if bary.min() < -1e-3 or cosine.min() < -1e-5:
            raise ValueError('Exit does not match its outward triangle')
        clipped = int(((bary.min(1) < 1e-5) | (cosine < 1e-5)).sum())
        bary = np.maximum(bary, 1e-5)
        safe_cosine = np.maximum(cosine, 1e-5)
        y[ids, :2] = np.log(bary[:, :2]/bary[:, 2:3])
        y[ids, 2] = np.einsum('nj,nj->n', d, t[f])/safe_cosine
        y[ids, 3] = np.einsum('nj,nj->n', d, b[f])/safe_cosine
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite training coordinates')
    return x.astype(np.float32), facet, y, valid, escaped, clipped


class BoundaryModel(nn.Module):
    def __init__(self, vertices, faces, width=64, components=4):
        super().__init__()
        if width < 1 or components < 1:
            raise ValueError('width and components must be positive')
        tri, t, b, n = frames(vertices, faces)
        for name, value in [('triangles', tri), ('tangent', t), ('bitangent', b), ('normal', n)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.width, self.components = width, components
        self.encoder = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.escape = nn.Linear(width, 1)
        self.facet = nn.Linear(width, len(faces))
        self.embedding = nn.Embedding(len(faces), 16)
        self.mixture = nn.Sequential(nn.Linear(width+16, width), nn.SiLU(), nn.Linear(width, components*9))

    def distribution(self, h, facet):
        params = self.mixture(torch.cat([h, self.embedding(facet)], dim=-1)).reshape(-1, self.components, 9)
        return params[:, :, 0], params[:, :, 1:5], params[:, :, 5:9].clamp(-4, 4)

    def losses(self, x, facet, y, escaped):
        h = self.encoder(x)
        escape_loss = F.binary_cross_entropy_with_logits(self.escape(h).squeeze(-1), escaped.float())
        if not escaped.any():
            return escape_loss, escape_loss*0, escape_loss*0
        facet_loss = F.cross_entropy(self.facet(h[escaped]), facet[escaped])
        logits, means, log_std = self.distribution(h[escaped], facet[escaped])
        z = (y[escaped, None, :] - means)*torch.exp(-log_std)
        log_density = (-.5*z.square()-log_std-.5*math.log(2*math.pi)).sum(-1)
        mixture_loss = -torch.logsumexp(F.log_softmax(logits, -1)+log_density, -1).mean()
        return escape_loss, facet_loss, mixture_loss

    @torch.no_grad()
    def log_exit_pdf(self, x, facet, position, direction):
        """Conditional escaped density in surface area times solid angle.

        Includes categorical triangle mass and both coordinate Jacobians.
        Does not include the finite-depth escape Bernoulli probability.
        """
        triangle = self.triangles[facet]
        edges = triangle[:, :2]-triangle[:, 2:3]
        delta = position-triangle[:, 2]
        gram = edges @ edges.transpose(1, 2)
        uv = torch.linalg.solve(gram, (edges @ delta[:, :, None])).squeeze(-1)
        bary = torch.cat([uv, 1-uv.sum(-1, keepdim=True)], -1)
        normal = self.normal[facet]
        cosine = (direction*normal).sum(-1)
        plane_error = ((position-triangle[:, 2])*normal).sum(-1).abs()
        on_surface = plane_error <= 1e-5*torch.linalg.vector_norm(edges[:, 0],dim=-1)
        unit_direction = (torch.linalg.vector_norm(direction,dim=-1)-1).abs() < 1e-4
        supported = (bary > 0).all(-1) & (cosine > 0) & on_surface & unit_direction
        safe_bary = bary.clamp_min(1e-30)
        safe_cosine = cosine.clamp_min(1e-30)
        z = torch.log(safe_bary[:, :2]/safe_bary[:, 2:3])
        slope = torch.stack([(direction*self.tangent[facet]).sum(-1),
                             (direction*self.bitangent[facet]).sum(-1)], -1)/safe_cosine[:, None]
        y = torch.cat([z, slope], -1)
        h = self.encoder(x)
        logits, means, log_std = self.distribution(h, facet)
        residual = (y[:, None]-means)*torch.exp(-log_std)
        log_mixture = torch.logsumexp(F.log_softmax(logits, -1)+
            (-.5*residual.square()-log_std-.5*math.log(2*math.pi)).sum(-1), -1)
        double_area = torch.linalg.vector_norm(torch.cross(edges[:,0], edges[:,1], dim=-1), dim=-1)
        log_jacobian = double_area.log()+safe_bary.log().sum(-1)+3*safe_cosine.log()
        log_pdf = F.log_softmax(self.facet(h), -1).gather(1, facet[:,None]).squeeze(-1)+log_mixture-log_jacobian
        return torch.where(supported, log_pdf, torch.full_like(log_pdf, -torch.inf))

    @torch.no_grad()
    def sample(self, x, generator=None, deterministic=False, facet=None):
        """Draw an escape event.

        `deterministic` keeps the discrete structure -- the exit facet and the
        mixture component are still sampled -- but decodes each draw at its
        component mean instead of adding the Gaussian spread. Measured transport
        is a handful of discrete branches each with 0.00 degrees of within-branch
        spread, so this is the limiting case that matches the physics, and it
        isolates the learned spread as the only thing that changes.

        `facet` overrides the exit-facet categorical with a caller-supplied
        branch. Passing the true facet turns the model into a within-facet
        decoder and isolates branch selection as the only error source, which
        is the upper bound the oracle experiment needs.
        """
        h = self.encoder(x)
        escaped = torch.rand(len(x), device=x.device, generator=generator) < torch.sigmoid(self.escape(h).squeeze(-1))
        if facet is None:
            facet = torch.multinomial(self.facet(h).softmax(-1), 1, generator=generator).squeeze(-1)
        logits, means, log_std = self.distribution(h, facet)
        component = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
        index = torch.arange(len(x), device=x.device)
        y = means[index, component]
        if not deterministic:
            y = y + log_std[index, component].exp()*torch.randn((len(x), 4), device=x.device, generator=generator)
        bary = torch.cat([y[:, :2], torch.zeros_like(y[:, :1])], -1).softmax(-1)
        position = (self.triangles[facet]*bary[:, :, None]).sum(1)
        direction = F.normalize(self.normal[facet]+y[:, 2:3]*self.tangent[facet]+y[:, 3:4]*self.bitangent[facet], dim=-1)
        return dict(escaped=escaped, exit_facet=facet, exit_position=position,
                    exit_direction=direction, barycentric=bary,
                    throughput=escaped.float())


class BoundaryCloneModel(nn.Module):
    """Behavioural cloning baseline: deterministic regression of the exit.

    Shares BoundaryModel's inputs, encoder, escape head, facet head and decode,
    and replaces the Gaussian mixture over exit coordinates with direct
    regression. Only the coordinate head and its loss differ, so a comparison
    against BoundaryModel isolates one variable: distributional against
    deterministic.

    Regression is conditioned on the exit facet, which matters. Measured
    transport reaches a median of two distinct facets per entry state, so
    regressing without that conditioning would average across branches and
    reproduce the very blur this baseline exists to test. Within a single facet
    the measured spread of real exit directions is 0.00 degrees, so the
    within-facet target is effectively a deterministic function and least
    squares is well posed.

    The loss is smooth L1 rather than plain squared error. Barycentric logits are
    unbounded and reach magnitude ~17 near facet edges, and tangent slopes blow
    up at grazing angles, so squared error would be dominated by a few extreme
    targets. Smooth L1 is quadratic near zero and linear in the tails, which
    gives the baseline its best chance rather than handicapping it.
    """

    def __init__(self, vertices, faces, width=64, components=4):
        super().__init__()
        if width < 1:
            raise ValueError('width must be positive')
        tri, t, b, n = frames(vertices, faces)
        for name, value in [('triangles', tri), ('tangent', t), ('bitangent', b), ('normal', n)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        # `components` is accepted and stored so checkpoints stay interchangeable;
        # a deterministic head has no mixture components.
        self.width, self.components = width, components
        self.encoder = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.escape = nn.Linear(width, 1)
        self.facet = nn.Linear(width, len(faces))
        self.embedding = nn.Embedding(len(faces), 16)
        self.regressor = nn.Sequential(nn.Linear(width+16, width), nn.SiLU(), nn.Linear(width, 4))

    def coordinates(self, h, facet):
        return self.regressor(torch.cat([h, self.embedding(facet)], dim=-1))

    def losses(self, x, facet, y, escaped):
        """Returns (escape BCE, facet cross-entropy, coordinate smooth L1).

        The third term is a regression error, not a negative log likelihood, so
        the combined figure reported during training is not a joint NLL and is
        not comparable with BoundaryModel's. Compare the two on exit-prediction
        error or on rendered images instead.
        """
        h = self.encoder(x)
        escape_loss = F.binary_cross_entropy_with_logits(self.escape(h).squeeze(-1), escaped.float())
        if not escaped.any():
            return escape_loss, escape_loss*0, escape_loss*0
        facet_loss = F.cross_entropy(self.facet(h[escaped]), facet[escaped])
        # Teacher forcing: regress against the true facet during training, and
        # against the predicted facet at inference.
        predicted = self.coordinates(h[escaped], facet[escaped])
        coordinate_loss = F.smooth_l1_loss(predicted, y[escaped])
        return escape_loss, facet_loss, coordinate_loss

    @torch.no_grad()
    def sample(self, x, generator=None, deterministic=False, facet=None):
        """Predict one exit per query.

        The exit coordinates are always deterministic. `deterministic` selects
        how the facet is chosen: False samples it from the categorical, keeping
        the discrete branch multiplicity that real transport has; True takes the
        most likely facet, which is pure behavioural cloning.

        `facet` overrides that choice with a caller-supplied branch, which is
        the inference-time counterpart of the teacher forcing `losses` already
        uses during training.
        """
        h = self.encoder(x)
        escaped = torch.rand(len(x), device=x.device, generator=generator) < torch.sigmoid(self.escape(h).squeeze(-1))
        probability = self.facet(h).softmax(-1)
        if facet is None:
            facet = (probability.argmax(-1) if deterministic
                     else torch.multinomial(probability, 1, generator=generator).squeeze(-1))
        y = self.coordinates(h, facet)
        bary = torch.cat([y[:, :2], torch.zeros_like(y[:, :1])], -1).softmax(-1)
        position = (self.triangles[facet]*bary[:, :, None]).sum(1)
        direction = F.normalize(self.normal[facet]+y[:, 2:3]*self.tangent[facet]+y[:, 3:4]*self.bitangent[facet], dim=-1)
        return dict(escaped=escaped, exit_facet=facet, exit_position=position,
                    exit_direction=direction, barycentric=bary,
                    throughput=escaped.float())


HEADS = {'mixture': BoundaryModel, 'clone': BoundaryCloneModel}


def load_model(path):
    """Load the self-contained CPU model and its data conventions."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Unsupported model schema')
    head = checkpoint.get('head', 'mixture')
    if head not in HEADS:
        raise ValueError('Unknown coordinate head %r' % head)
    model = HEADS[head](checkpoint['vertices'].numpy(), checkpoint['faces'].numpy(),
                        checkpoint['width'], checkpoint['components'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
