"""Learned dispersion: how one path's other colours leave, given the hero's exit.

The boundary operator ignores wavelength because the colour-dependent part of
an exit is far below its own error. The paired-wavelength pilot
(gather_dispersion.py) showed why, and what a model of it needs: dispersion has
two regimes. On short internal paths the colours share the hero's route and fan
out by a fraction of a degree to a few degrees -- ordered fire. On long paths
each extra bounce amplifies the colour difference until the colours leave by
unrelated routes, so their exits are effectively independent.

So this head does not predict an exit. It predicts, for a second wavelength
given the hero's entry state, exit facet and exit coordinates:

- a gate: the probability that this colour stays on the hero's route;
- a shift: the change in the four exit coordinates when it does, in the same
  per-facet frame the operator decodes in (two barycentric logits, two tangent
  slopes), so the shifted exit is always on the facet and always outward.

A colour the gate sends away is given its own independent draw from the
boundary operator at its own wavelength. Both branches are sampled, never
weighted, so the head is a generative model of the other lanes and adds no
importance weights to the renderer.

The head is trained on true hero exits and applied to the operator's sampled
ones. That is a deliberate mismatch: the shift is a local derivative of the
exit with respect to wavelength, so it should transfer to nearby exits even
where the operator misplaces them.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neural.boundary_model import frames

WAVELENGTH_CENTRE, WAVELENGTH_SCALE = 595., 235.


def diamond_index(wavelength_nm):
    """Sellmeier index of diamond, the same fit as bsdf.dispersion.diamond_ior.

    Duplicated as plain arithmetic so the head can be used without importing a
    Mitsuba variant. The index difference between two colours, not their
    wavelength gap, is what physically drives the shift: n(lambda) is steep in
    the blue and flat in the red, so equal gaps disperse very differently.
    """
    l2 = (wavelength_nm/1000.)**2
    return (1 + .3306*l2/(l2 - .175**2) + 4.3356*l2/(l2 - .106**2))**.5


def coordinates(triangles, tangent, bitangent, normal, facet, position, direction):
    """Exit coordinates in the operator's convention; inverse of `decode`.

    Returns (y, clipped): y is barycentric logits against the third vertex and
    outgoing tangent slopes, and `clipped` flags exits on a facet edge or at a
    grazing angle, where the coordinates were clamped to stay finite.
    """
    tri = triangles[facet]
    edges = tri[:, :2] - tri[:, 2:3]
    delta = position - tri[:, 2]
    gram = edges @ edges.transpose(1, 2)
    uv = torch.linalg.solve(gram, (edges @ delta[..., None])).squeeze(-1)
    bary = torch.cat([uv, 1 - uv.sum(-1, keepdim=True)], -1)
    cosine = (direction*normal[facet]).sum(-1)
    clipped = (bary.min(-1).values < 1e-5) | (cosine < 1e-5)
    bary = bary.clamp_min(1e-5)
    cosine = cosine.clamp_min(1e-5)
    y = torch.stack([torch.log(bary[:, 0]/bary[:, 2]), torch.log(bary[:, 1]/bary[:, 2]),
                     (direction*tangent[facet]).sum(-1)/cosine,
                     (direction*bitangent[facet]).sum(-1)/cosine], -1)
    return y, clipped


def decode(triangles, tangent, bitangent, normal, facet, y):
    """Surface point and outward unit direction, exactly as the operator decodes."""
    bary = torch.cat([y[:, :2], torch.zeros_like(y[:, :1])], -1).softmax(-1)
    position = (triangles[facet]*bary[:, :, None]).sum(1)
    direction = F.normalize(normal[facet] + y[:, 2:3]*tangent[facet]
                            + y[:, 3:4]*bitangent[facet], dim=-1)
    return position, direction


class DispersionModel(nn.Module):
    """Gate and shift for a second wavelength, given the hero's escape.

    Inputs are the operator's own ten entry features at the hero wavelength,
    a learned facet embedding, the hero's exit coordinates (squashed with asinh,
    since slopes blow up at grazing exits), the other wavelength, the
    wavelength gap and the refractive-index gap, scaled so diamond's full
    visible dispersion of about 0.05 spans about five units. Shift targets are standardised per coordinate with scales
    stored in the checkpoint, so the four outputs are learned on equal terms.
    """

    def __init__(self, vertices, faces, width=128, depth=4):
        super().__init__()
        tri, t, b, n = frames(vertices, faces)
        for name, value in [('triangles', tri), ('tangent', t), ('bitangent', b), ('normal', n)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer('shift_scale', torch.ones(4))
        self.width, self.depth = width, depth
        self.embedding = nn.Embedding(len(faces), 16)
        layers, size = [], 10 + 16 + 4 + 3
        for _ in range(depth - 1):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(width, 1)
        self.shift = nn.Linear(width, 4)

    def forward(self, x, facet, y, wavelength):
        """x: operator features at the hero wavelength; y: hero exit coordinates;
        wavelength: the other lane's wavelength in nm. Returns (gate logit, shift)."""
        hero = x[:, 9:10]
        other = ((wavelength - WAVELENGTH_CENTRE)/WAVELENGTH_SCALE)[:, None]
        # Clamp to the spectral range: the Sellmeier fit has poles in the UV.
        index_gap = 100*(diamond_index(wavelength[:, None].clamp(360., 830.))
                         - diamond_index((hero*WAVELENGTH_SCALE + WAVELENGTH_CENTRE).clamp(360., 830.)))
        h = self.trunk(torch.cat([x, self.embedding(facet), torch.asinh(y), other, other - hero,
                                  index_gap], -1))
        return self.gate(h).squeeze(-1), self.shift(h)*self.shift_scale

    def losses(self, x, facet, y, wavelength, stays, shift, usable):
        """Gate cross-entropy on every pair; shift smooth-L1 on the pairs that
        stay and whose coordinates were not clamped at an edge or grazing exit."""
        logit, predicted = self(x, facet, y, wavelength)
        gate_loss = F.binary_cross_entropy_with_logits(logit, stays.float())
        if not usable.any():
            return gate_loss, gate_loss*0
        shift_loss = F.smooth_l1_loss(predicted[usable]/self.shift_scale,
                                      shift[usable]/self.shift_scale)
        return gate_loss, shift_loss

    @torch.no_grad()
    def sample(self, x, facet, y, wavelength, generator=None):
        """Per query: whether this colour stays on the hero's route, and if so its
        exit position and direction on the hero's facet."""
        logit, shift = self(x, facet, y, wavelength)
        stays = torch.rand(len(x), generator=generator) < torch.sigmoid(logit)
        position, direction = decode(self.triangles, self.tangent, self.bitangent,
                                     self.normal, facet, y + shift)
        return dict(stays=stays, exit_position=position, exit_direction=direction,
                    stay_probability=torch.sigmoid(logit))


def hero_coordinates(model, facet, position, direction):
    """Exit coordinates of an operator sample, for feeding the dispersion head."""
    y, _ = coordinates(model.triangles, model.tangent, model.bitangent, model.normal,
                       facet, position, direction)
    return y


def load_dispersion(path, operator_checkpoint=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1 or checkpoint.get('kind') != 'dispersion':
        raise ValueError('Not a dispersion checkpoint: %s' % path)
    if operator_checkpoint is not None and not torch.equal(
            checkpoint['faces'], operator_checkpoint['faces'].to(checkpoint['faces'].dtype)):
        raise ValueError('Dispersion head and boundary operator use different geometry')
    model = DispersionModel(checkpoint['vertices'].numpy(), checkpoint['faces'].numpy(),
                            checkpoint['width'], checkpoint['depth'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
