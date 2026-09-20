"""Which inputs does the boundary operator actually need?

The project's central design decision is that a radiance distribution map
conditioned on angle alone cannot locate facet detail, and that entry position
must be conditioned on. That was argued from the geometry -- light enters the
crown and exits the pavilion -- and supported indirectly by lookup tables, where
a position proxy predicts the exit facet roughly three times better than
direction. This runs the direct version: take an input away from the network
itself and measure what breaks.

Inputs are removed by zeroing their slice of the feature vector rather than by
narrowing the first layer. The architecture, parameter count and optimisation
are then identical across arms, so a difference in the result is a difference in
information and not in capacity. A constant-zero input carries nothing, which is
what "removed" has to mean here.

Every arm shares one training budget, one seed and one entry-grouped split, so
the arms are comparable with each other. The `full` arm is the control: compare
the others against it, not against numbers from longer runs elsewhere in the
project, which used a different budget.

Two things are measured per arm. Exit-facet top-1 is branch selection. Exit
angle given the TRUE facet is the within-facet decode, isolated by handing the
model the correct branch -- so the two halves of the diagnosis are read
separately rather than through a single image metric that confounds them.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from neural.boundary_data import load_boundary, split_entries
from neural.boundary_model import BoundaryCloneModel, encode_data

GROUPS = {'position': slice(0, 3), 'direction': slice(3, 6),
          'normal': slice(6, 9), 'wavelength': slice(9, 10)}

ARMS = {
    'full': (),
    'no_position': ('position',),
    'no_direction': ('direction',),
    'no_normal': ('normal',),
    'no_wavelength': ('wavelength',),
    'direction_only': ('position', 'normal', 'wavelength'),
}


def load_pool(path):
    data, metadata = load_boundary(path)
    with np.load(path, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, facet, y, valid, escaped, _ = encode_data(data, vertices, faces)
    return data, vertices, faces, x, facet, y, valid, escaped


def mask_for(dropped, width=10):
    keep = torch.ones(width, dtype=torch.float32)
    for name in dropped:
        keep[GROUPS[name]] = 0.
    return keep


def train_arm(name, dropped, tensors, train_idx, val_idx, vertices, faces, args):
    torch.manual_seed(args.seed)
    keep = mask_for(dropped)
    model = BoundaryCloneModel(vertices, faces, args.width, args.components)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x, facet, y, escaped = tensors
    best, best_state = float('inf'), None
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for idx in order.split(args.batch_size):
            losses = model.losses(x[idx]*keep, facet[idx], y[idx], escaped[idx])
            loss = losses[0] + escaped[idx].float().mean()*(losses[1]+losses[2])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss in arm %s' % name)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        # Model selection on the held-in validation split, never on the test
        # pool the arms are finally compared on.
        with torch.no_grad():
            model.eval()
            sub = val_idx[escaped[val_idx]]
            logits = model.facet(model.encoder(x[sub]*keep))
            nll = torch.nn.functional.cross_entropy(logits, facet[sub]).item()
        if nll < best:
            best, best_state = nll, copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch+1) % 5 == 0:
            print('  %-15s epoch %2d  validation facet NLL %.4f'
                  % (name, epoch+1, nll), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, keep, time.perf_counter()-began


@torch.no_grad()
def score(model, keep, x, facet, truth_direction, chunk=65536):
    """Branch accuracy, and exit angle given the true branch."""
    hits, angles = [], []
    generator = torch.Generator().manual_seed(0)
    for start in range(0, len(x), chunk):
        block = x[start:start+chunk]*keep
        true_facet = facet[start:start+chunk]
        logits = model.facet(model.encoder(block))
        hits.append(logits.argmax(-1) == true_facet)
        # Hand the model the correct facet so only the within-facet decode is
        # its own; this is the same isolation the oracle-facet render uses.
        predicted = model.sample(block, generator, facet=true_facet)['exit_direction']
        cosine = (predicted*truth_direction[start:start+chunk]).sum(-1).clamp(-1., 1.)
        angles.append(torch.rad2deg(torch.arccos(cosine)))
    hits, angles = torch.cat(hits), torch.cat(angles)
    return dict(facet_top1=hits.float().mean().item(),
                exit_angle_mean=angles.mean().item(),
                exit_angle_median=angles.median().item(),
                exit_angle_p90=torch.quantile(angles, .9).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Training pool')
    parser.add_argument('--test', type=Path, required=True, help='Held-out pool')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=sorted(ARMS),
                        choices=sorted(ARMS))
    parser.add_argument('--epochs', type=int, default=20,
                        help='Matched budget for every arm')
    parser.add_argument('--entries', type=int, default=0,
                        help='Subsample this many training entry states; 0 uses all')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--threads', type=int, default=0,
                        help='torch thread cap; 0 leaves the default')
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data, vertices, faces, x, facet, y, valid, escaped = load_pool(args.data)
    train, validation = split_entries(data['entry_id'], seed=args.seed)
    train &= valid
    validation &= valid
    if args.entries:
        unique = np.unique(data['entry_id'][train])
        keep_ids = np.random.default_rng(args.seed).permutation(unique)[:args.entries]
        train &= np.isin(data['entry_id'], keep_ids)
    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    train_idx = torch.from_numpy(np.flatnonzero(train))
    val_idx = torch.from_numpy(np.flatnonzero(validation))

    test_data, tv, tf, tx, tfacet, _, _, tescaped = load_pool(args.test)
    if not np.array_equal(tv, vertices) or not np.array_equal(tf, faces):
        raise ValueError('Training and test pools use different geometry')
    test_x = torch.from_numpy(tx[tescaped])
    test_facet = torch.from_numpy(tfacet[tescaped])
    test_direction = torch.from_numpy(test_data['exit_direction'][tescaped])

    print('%d training records from %d entry states, %d held-out escaped records'
          % (len(train_idx), len(np.unique(data['entry_id'][train])), len(test_x)))
    print('%d epochs per arm, width %d, batch %d, lr %g, seed %d\n'
          % (args.epochs, args.width, args.batch_size, args.lr, args.seed))

    results = {}
    for name in args.arms:
        dropped = ARMS[name]
        model, keep, seconds = train_arm(name, dropped, tensors, train_idx, val_idx,
                                         vertices, faces, args)
        results[name] = dict(dropped=list(dropped), seconds=seconds,
                             **score(model, keep, test_x, test_facet, test_direction))
        print('  %-15s facet top-1 %.4f   exit angle %.2f deg   (%.0f s)\n'
              % (name, results[name]['facet_top1'],
                 results[name]['exit_angle_mean'], seconds), flush=True)

    report = dict(epochs=args.epochs, width=args.width, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed,
                  train_records=len(train_idx), test_records=len(test_x),
                  train_entries=int(len(np.unique(data['entry_id'][train]))),
                  arms=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    base = results.get('full', {}).get('facet_top1')
    print('\n%-16s %12s %10s %14s' % ('arm', 'facet top-1', 'vs full', 'exit angle'))
    for name in args.arms:
        r = results[name]
        delta = ('%+.4f' % (r['facet_top1']-base)) if base is not None and name != 'full' else '-'
        print('%-16s %12.4f %10s %11.2f deg'
              % (name, r['facet_top1'], delta, r['exit_angle_mean']))
    print('\nwrote %s' % args.output)


if __name__ == '__main__':
    main()
