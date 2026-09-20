"""Can the within-facet decode be fixed? It is the half nobody has attacked.

Given the true exit facet, the shipped clone places the exit 24.66 degrees from
the truth while an untrained nearest-neighbour lookup manages 8.87. A 2.8x gap
on a smooth four-output regression, lost to a method with no parameters at all,
is underfitting rather than an intrinsic limit.

Two suspects, both in the architecture rather than the data.

The regressor is two layers and 5,444 parameters, which is small for a function
that has to place an exit anywhere on a facet at any incidence.

It reads a trunk trained jointly for escape, facet and coordinates, and the
64-way facet cross-entropy dominates that trunk's gradient. The features it
learns are the ones that separate branches, which are not necessarily the ones
that locate a point inside one.

The arms separate those: depth alone, an independent trunk alone, both, and
simply weighting the coordinate loss up against a shared trunk. Every arm keeps
the same escape and facet heads, so branch selection is a control -- an arm that
improves the decode by wrecking the branch predictor has not helped.

Selection is on validation coordinate error, not on the joint loss, because the
joint loss is dominated by the term this experiment is not trying to move.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neural.boundary_data import load_boundary, split_entries
from neural.boundary_model import BoundaryCloneModel, encode_data


class DecoderVariant(BoundaryCloneModel):
    """Clone with a configurable coordinate head, everything else unchanged."""

    def __init__(self, vertices, faces, width=64, components=4,
                 depth=2, separate=False):
        super().__init__(vertices, faces, width, components)
        layers, size = [], width+16
        for _ in range(depth-1):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        layers.append(nn.Linear(size, 4))
        self.regressor = nn.Sequential(*layers)
        # An independent trunk lets the coordinate task learn its own features
        # instead of inheriting ones shaped by the facet cross-entropy.
        self.trunk = (nn.Sequential(nn.Linear(10, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
                      if separate else None)

    def features(self, x, h):
        return self.trunk(x) if self.trunk is not None else h

    def losses(self, x, facet, y, escaped):
        h = self.encoder(x)
        escape_loss = F.binary_cross_entropy_with_logits(
            self.escape(h).squeeze(-1), escaped.float())
        if not escaped.any():
            return escape_loss, escape_loss*0, escape_loss*0
        facet_loss = F.cross_entropy(self.facet(h[escaped]), facet[escaped])
        g = self.features(x, h)
        predicted = self.coordinates(g[escaped], facet[escaped])
        return escape_loss, facet_loss, F.smooth_l1_loss(predicted, y[escaped])

    @torch.no_grad()
    def sample(self, x, generator=None, deterministic=False, facet=None):
        h = self.encoder(x)
        escaped = torch.rand(len(x), generator=generator) < torch.sigmoid(
            self.escape(h).squeeze(-1))
        probability = self.facet(h).softmax(-1)
        if facet is None:
            facet = (probability.argmax(-1) if deterministic
                     else torch.multinomial(probability, 1, generator=generator).squeeze(-1))
        y = self.coordinates(self.features(x, h), facet)
        bary = torch.cat([y[:, :2], torch.zeros_like(y[:, :1])], -1).softmax(-1)
        position = (self.triangles[facet]*bary[:, :, None]).sum(1)
        direction = F.normalize(self.normal[facet] + y[:, 2:3]*self.tangent[facet]
                                + y[:, 3:4]*self.bitangent[facet], dim=-1)
        return dict(escaped=escaped, exit_facet=facet, exit_position=position,
                    exit_direction=direction, barycentric=bary,
                    throughput=escaped.float())


ARMS = {
    'baseline':      dict(depth=2, separate=False, coord_weight=1.),
    'deep':          dict(depth=4, separate=False, coord_weight=1.),
    'separate':      dict(depth=2, separate=True, coord_weight=1.),
    'separate_deep': dict(depth=4, separate=True, coord_weight=1.),
    'weighted':      dict(depth=2, separate=False, coord_weight=5.),
}


def load_pool(path):
    data, _ = load_boundary(path)
    with np.load(path, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, facet, y, valid, escaped, _ = encode_data(data, vertices, faces)
    return data, vertices, faces, x, facet, y, valid, escaped


@torch.no_grad()
def coordinate_error(model, x, facet, y, index, chunk=65536):
    """Validation smooth-L1 in transformed coordinates, the selection metric."""
    total, count = 0., 0
    for start in range(0, len(index), chunk):
        idx = index[start:start+chunk]
        h = model.encoder(x[idx])
        predicted = model.coordinates(model.features(x[idx], h), facet[idx])
        total += F.smooth_l1_loss(predicted, y[idx], reduction='sum').item()
        count += idx.numel()*4
    return total/count


@torch.no_grad()
def score(model, x, facet, truth, chunk=65536):
    hits, angles = [], []
    generator = torch.Generator().manual_seed(0)
    for start in range(0, len(x), chunk):
        block, true_facet = x[start:start+chunk], facet[start:start+chunk]
        hits.append(model.facet(model.encoder(block)).argmax(-1) == true_facet)
        predicted = model.sample(block, generator, facet=true_facet)['exit_direction']
        cosine = (predicted*truth[start:start+chunk]).sum(-1).clamp(-1., 1.)
        angles.append(torch.rad2deg(torch.arccos(cosine)))
    hits, angles = torch.cat(hits), torch.cat(angles)
    return dict(facet_top1=hits.float().mean().item(),
                exit_angle_mean=angles.mean().item(),
                exit_angle_median=angles.median().item(),
                exit_angle_p90=torch.quantile(angles, .9).item())


def train_arm(name, config, tensors, train_idx, val_idx, vertices, faces, args):
    torch.manual_seed(args.seed)
    model = DecoderVariant(vertices, faces, args.width, args.components,
                           config['depth'], config['separate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x, facet, y, escaped = tensors
    escaped_val = val_idx[escaped[val_idx]]
    best, best_state = float('inf'), None
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for idx in order.split(args.batch_size):
            e, f, c = model.losses(x[idx], facet[idx], y[idx], escaped[idx])
            loss = e + escaped[idx].float().mean()*(f + config['coord_weight']*c)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss in arm %s' % name)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        model.eval()
        error = coordinate_error(model, x, facet, y, escaped_val)
        if error < best:
            best, best_state = error, copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch+1) % 10 == 0:
            print('  %-14s epoch %2d  validation coordinate error %.5f'
                  % (name, epoch+1, error), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, time.perf_counter()-began


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=sorted(ARMS), choices=sorted(ARMS))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--save_best', type=Path,
                        help='Write the winning arm as a loadable checkpoint')
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data, vertices, faces, x, facet, y, valid, escaped = load_pool(args.data)
    train, validation = split_entries(data['entry_id'], seed=args.seed)
    train &= valid
    validation &= valid
    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    train_idx = torch.from_numpy(np.flatnonzero(train))
    val_idx = torch.from_numpy(np.flatnonzero(validation))

    test_data, tv, tf, tx, tfacet, _, _, tescaped = load_pool(args.test)
    if not np.array_equal(tv, vertices) or not np.array_equal(tf, faces):
        raise ValueError('Training and test pools use different geometry')
    test_x = torch.from_numpy(tx[tescaped])
    test_facet = torch.from_numpy(tfacet[tescaped])
    test_direction = torch.from_numpy(test_data['exit_direction'][tescaped])

    print('%d training records, %d held-out escaped records, %d epochs per arm\n'
          % (len(train_idx), len(test_x), args.epochs))

    results, models = {}, {}
    for name in args.arms:
        model, seconds = train_arm(name, ARMS[name], tensors, train_idx, val_idx,
                                   vertices, faces, args)
        parameters = sum(p.numel() for p in model.parameters())
        results[name] = dict(seconds=seconds, parameters=parameters, **ARMS[name],
                             **score(model, test_x, test_facet, test_direction))
        models[name] = model
        print('  %-14s exit angle %.2f deg   facet top-1 %.4f   %d params   (%.0f s)\n'
              % (name, results[name]['exit_angle_mean'],
                 results[name]['facet_top1'], parameters, seconds), flush=True)

    report = dict(epochs=args.epochs, width=args.width, lr=args.lr, seed=args.seed,
                  train_records=len(train_idx), test_records=len(test_x),
                  reference=dict(shipped_clone_exit_angle=24.66,
                                 nearest_neighbour_exit_angle=8.87),
                  arms=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    base = results.get('baseline', {}).get('exit_angle_mean')
    print('\n%-16s %12s %10s %14s %10s' % ('arm', 'exit angle', 'vs base',
                                           'facet top-1', 'params'))
    for name in args.arms:
        r = results[name]
        delta = ('%+.2f' % (r['exit_angle_mean']-base)) if base and name != 'baseline' else '-'
        print('%-16s %9.2f deg %10s %14.4f %10d'
              % (name, r['exit_angle_mean'], delta, r['facet_top1'], r['parameters']))
    print('\nnearest-neighbour lookup reaches 8.87 deg on the same conditional')

    if args.save_best:
        winner = min(results, key=lambda n: results[n]['exit_angle_mean'])
        torch.save(dict(schema_version=1, head='clone', width=args.width,
                        components=args.components, arm=winner,
                        state_dict=models[winner].state_dict()), args.save_best)
        print('saved %s (%s)' % (args.save_best, winner))
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
