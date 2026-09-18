"""Compare exit-prediction quality across boundary operators, without rendering.

Different coordinate heads optimise different losses -- a Gaussian mixture
reports a negative log likelihood, a regression head reports a squared or
smooth-L1 error -- so their training figures cannot be compared. This tool
scores every operator on the same geometric quantities instead: where does the
predicted exit land, and which way does it point.

Real transport is multimodal: a single entry state reaches a median of two
distinct exit facets. Comparing a prediction against one arbitrary recorded
outcome would therefore penalise a model for picking a different but equally
valid branch. Two complementary errors are reported:

  nearest  -- angle and distance to the CLOSEST recorded exit for that entry.
              Asks whether the prediction lands on some real branch at all.
  matched  -- angle and distance to a randomly chosen recorded exit.
              Asks whether it reproduces the branch distribution.

A model that is sharp but picks the wrong branch scores well on `nearest` and
badly on `matched`. One that hedges across branches scores moderately on both.
The gap between them is informative, so neither is reported alone.

Branch structure is also measured: distinct exit facets per 16 draws, against
the same count for real transport, and the angular spread within a single
predicted facet, which isolates how much the operator blurs each branch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_rgb')
from neural.boundary_data import load_boundary
from neural.boundary_model import load_model, encode_data

DRAWS = 16  # matches the recorded repeats per entry state


def unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def branch_spread(directions, facets):
    """Mean angle from the group mean, within each predicted facet."""
    out = []
    for value in np.unique(facets):
        group = directions[facets == value]
        if len(group) < 4:
            continue
        mean = group.mean(0)
        norm = np.linalg.norm(mean)
        if norm < 1e-9:
            continue
        out.append(np.degrees(np.arccos(np.clip(group @ (mean/norm), -1, 1))).mean())
    return out


def evaluate(model, x, entries, truth, rng, generator, deterministic):
    angles_near, angles_matched, distance_near = [], [], []
    facet_hit, distinct, spread = [], [], []
    for entry, (true_position, true_direction, true_facet) in zip(entries, truth):
        inputs = torch.from_numpy(np.repeat(x[entry][None], DRAWS, axis=0))
        prediction = model.sample(inputs, generator, deterministic=deterministic)
        direction = unit(prediction['exit_direction'].numpy())
        position = prediction['exit_position'].numpy()
        facet = prediction['exit_facet'].numpy()
        # Angle from every draw to every recorded exit for this entry state.
        cosine = np.clip(direction @ true_direction.T, -1, 1)
        angle = np.degrees(np.arccos(cosine))
        angles_near.extend(angle.min(1))
        angles_matched.extend(angle[np.arange(len(angle)),
                                    rng.integers(0, len(true_direction), len(angle))])
        gap = np.linalg.norm(position[:, None, :]-true_position[None], axis=2)
        distance_near.extend(gap.min(1))
        facet_hit.extend(np.isin(facet, true_facet))
        distinct.append(len(np.unique(facet)))
        spread.extend(branch_spread(direction, facet))
    return dict(
        angle_nearest_deg=float(np.median(angles_near)),
        angle_matched_deg=float(np.median(angles_matched)),
        position_nearest=float(np.median(distance_near)),
        facet_hit_rate=float(np.mean(facet_hit)),
        distinct_facets_per_16=float(np.mean(distinct)),
        within_branch_spread_deg=float(np.median(spread)) if spread else None,
        entry_states=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True,
                        help='Held-out gathered pool, with repeats per entry state')
    parser.add_argument('--models', nargs='+', required=True, metavar='NAME=PATH',
                        help='Operators to score, e.g. mixture=checkpoints/a/model.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entries', type=int, default=400)
    parser.add_argument('--deterministic', action='store_true',
                        help='Ask each operator for its deterministic decode')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.entries < 1:
        parser.error('--entries must be positive')

    data, _ = load_boundary(args.data)
    with np.load(args.data, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, _, _, valid, escaped, _ = encode_data(data, vertices, faces)
    ids = data['entry_id']

    rng = np.random.default_rng(args.seed)
    candidates = np.unique(ids)
    chosen = rng.choice(candidates, min(args.entries, len(candidates)), replace=False)
    entries, truth, real_distinct, real_spread = [], [], [], []
    for entry in chosen:
        rows = np.flatnonzero((ids == entry) & escaped & valid)
        if len(rows) < DRAWS:
            continue
        rows = rows[:DRAWS]
        direction = unit(data['exit_direction'][rows])
        facet = data['exit_facet'][rows]
        entries.append(rows[0])
        truth.append((data['exit_position'][rows], direction, facet))
        real_distinct.append(len(np.unique(facet)))
        real_spread.extend(branch_spread(direction, facet))
    if not entries:
        raise ValueError('No entry states with enough escaped repeats')

    report = dict(data=str(args.data), entry_states=len(entries),
                  draws_per_entry=DRAWS, deterministic=args.deterministic,
                  reference=dict(distinct_facets_per_16=float(np.mean(real_distinct)),
                                 within_branch_spread_deg=float(np.median(real_spread))),
                  operators={})
    for item in args.models:
        if '=' not in item:
            parser.error('--models entries must look like NAME=PATH, got %r' % item)
        name, path = item.split('=', 1)
        model, checkpoint = load_model(Path(path))
        generator = torch.Generator().manual_seed(args.seed + 1)
        scores = evaluate(model, x, entries, truth, np.random.default_rng(args.seed + 2),
                          generator, args.deterministic)
        scores['head'] = checkpoint.get('head', 'mixture')
        scores['path'] = path
        report['operators'][name] = scores

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    reference = report['reference']
    print('%d entry states, %d draws each, deterministic=%s\n'
          % (len(entries), DRAWS, args.deterministic))
    header = ('%-16s %8s %9s %9s %9s %9s %9s'
              % ('operator', 'head', 'near deg', 'match deg', 'facet hit', 'facets/16', 'blur deg'))
    print(header)
    print('-'*len(header))
    print('%-16s %8s %9s %9s %9s %9.2f %9.2f'
          % ('real transport', '-', '0.00', '-', '1.000',
             reference['distinct_facets_per_16'], reference['within_branch_spread_deg']))
    for name, s in report['operators'].items():
        # A deterministic coordinate head gives exactly 0.0 spread, which is a
        # real measurement rather than a missing one, so test for None.
        blur = s['within_branch_spread_deg']
        print('%-16s %8s %9.2f %9.2f %9.3f %9.2f %9s'
              % (name, s['head'], s['angle_nearest_deg'], s['angle_matched_deg'],
                 s['facet_hit_rate'], s['distinct_facets_per_16'],
                 'n/a' if blur is None else '%.2f' % blur))
    print('\nWrote %s' % args.output)


if __name__ == '__main__':
    main()
