"""Nearest-neighbour oracle: is the exit predictable from the entry state at all?

Every architecture experiment so far has asked how well a network learns the
boundary operator. This asks a prior question. Given the exact entry state of a
held-out path, look up the most similar entry states in the training pool and
take their exits. No training, no capacity, no optimisation -- just the data
answering whether the mapping it encodes is a function of the conditioning.

That distinction is the conclusion, not a detail:

  oracle succeeds, model fails   the exit IS predictable from entry state and
                                 the network is the bottleneck. Encoding,
                                 loss weighting and class balance are then
                                 worth the effort.
  oracle fails too               the exit is not determined by this 10-dim
                                 conditioning. No architecture fixes that, and
                                 the finding is about the representation, not
                                 about the model.

The oracle is generous on purpose. It sees the true entry state at full
precision, it may consult a million training paths, and it is never asked to
generalise beyond interpolation. Whatever it scores is a ceiling on what any
learner reading the same 10 inputs can do.

Three things are measured.

Facet accuracy, directly comparable with the facet_accuracy a trained
checkpoint reports, so the oracle and the network are read on one scale.

Top-k facet coverage: whether the true exit facet appears among the k nearest
neighbours' facets at all. A large gap between top-1 and top-k means the
branch is present in the neighbourhood but not resolved by proximity, which is
a different failure from the branch being absent.

Neighbour distance, reported as a distribution. An oracle that fails because
its nearest neighbour is far away is a coverage problem and more data fixes
it. An oracle that fails with a near neighbour in hand is telling you the map
itself is discontinuous there, and more data does not.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural.boundary_model import encode_data

FEATURE_NAMES = ('position', 'direction', 'normal', 'wavelength')
FEATURE_SLICES = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 10))


def load_pool(path):
    data = dict(np.load(path, allow_pickle=False))
    vertices, faces = data.pop('vertices'), data.pop('faces')
    data.pop('metadata_json', None)
    x, facet, y, _, escaped, _ = encode_data(data, vertices, faces)
    return (torch.from_numpy(x[escaped]), torch.from_numpy(facet[escaped]),
            torch.from_numpy(data['exit_direction'][escaped]), vertices, faces)


def neighbours(query, reference, k, weights, query_chunk=256, reference_chunk=131072):
    """Brute-force k nearest neighbours; no spatial index, scipy is absent.

    Chunked on both sides so peak memory stays at query_chunk*reference_chunk
    floats rather than the full product, which would be hundreds of gigabytes.
    """
    scaled_reference = reference*weights
    distances = torch.empty(len(query), k)
    indices = torch.empty(len(query), k, dtype=torch.long)
    for start in range(0, len(query), query_chunk):
        block = query[start:start+query_chunk]*weights
        best_d = torch.full((len(block), 0), float('inf'))
        best_i = torch.empty(len(block), 0, dtype=torch.long)
        for offset in range(0, len(scaled_reference), reference_chunk):
            chunk = scaled_reference[offset:offset+reference_chunk]
            d = torch.cdist(block, chunk)
            take = min(k, d.shape[1])
            d, i = torch.topk(d, take, dim=1, largest=False)
            best_d = torch.cat([best_d, d], dim=1)
            best_i = torch.cat([best_i, i+offset], dim=1)
            keep = min(k, best_d.shape[1])
            best_d, order = torch.topk(best_d, keep, dim=1, largest=False)
            best_i = best_i.gather(1, order)
        distances[start:start+query_chunk] = best_d
        indices[start:start+query_chunk] = best_i
    return distances, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', type=Path, required=True, help='Lookup pool')
    parser.add_argument('--test', type=Path, required=True, help='Held-out queries')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--queries', type=int, default=16384,
                        help='Random subset of the test pool to evaluate')
    parser.add_argument('--reference', type=int, default=0,
                        help='Cap the lookup pool; 0 uses all of it')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wavelength_weight', type=float, default=1.0,
                        help='Relative weight of the wavelength input in the metric')
    args = parser.parse_args()

    train_x, train_facet, train_direction, _, faces = load_pool(args.train)
    test_x, test_facet, test_direction, _, _ = load_pool(args.test)
    generator = torch.Generator().manual_seed(args.seed)
    if args.reference and args.reference < len(train_x):
        pick = torch.randperm(len(train_x), generator=generator)[:args.reference]
        train_x, train_facet, train_direction = train_x[pick], train_facet[pick], train_direction[pick]
    if args.queries and args.queries < len(test_x):
        pick = torch.randperm(len(test_x), generator=generator)[:args.queries]
        test_x, test_facet, test_direction = test_x[pick], test_facet[pick], test_direction[pick]

    weights = torch.ones(train_x.shape[1])
    weights[9] = args.wavelength_weight
    print('oracle over %d training paths, %d held-out queries, k=%d'
          % (len(train_x), len(test_x), args.k))
    distance, index = neighbours(test_x, train_x, args.k, weights)

    predicted = train_facet[index]
    top1 = (predicted[:, 0] == test_facet).float().mean().item()
    coverage = [(predicted[:, :j+1] == test_facet[:, None]).any(1).float().mean().item()
                for j in range(args.k)]
    # Majority vote over the neighbourhood, ties broken by the nearest member.
    votes = torch.zeros(len(test_x), len(faces))
    votes.scatter_add_(1, predicted, torch.ones_like(predicted, dtype=torch.float))
    majority = (votes.argmax(1) == test_facet).float().mean().item()

    cosine = (train_direction[index[:, 0]]*test_direction).sum(1).clamp(-1., 1.)
    angle = torch.rad2deg(torch.arccos(cosine))
    hit = predicted[:, 0] == test_facet
    # The most diagnostic split: among queries whose nearest neighbour is very
    # close, does the oracle still miss? A near neighbour with the wrong exit
    # is evidence of discontinuity rather than of sparse coverage.
    near = distance[:, 0] <= torch.quantile(distance[:, 0], 0.10)

    report = dict(
        training_paths=len(train_x), queries=len(test_x), k=args.k,
        facet_accuracy_top1=top1, facet_accuracy_majority=majority,
        facet_coverage_topk=coverage,
        exit_angle_deg=dict(mean=angle.mean().item(), median=angle.median().item(),
                            p10=torch.quantile(angle, .1).item(),
                            p90=torch.quantile(angle, .9).item()),
        exit_angle_deg_when_facet_correct=angle[hit].mean().item() if hit.any() else None,
        neighbour_distance=dict(
            mean=distance[:, 0].mean().item(), median=distance[:, 0].median().item(),
            p10=torch.quantile(distance[:, 0], .1).item(),
            p90=torch.quantile(distance[:, 0], .9).item()),
        facet_accuracy_nearest_decile=hit[near].float().mean().item(),
        facet_accuracy_farthest_decile=hit[distance[:, 0] >= torch.quantile(
            distance[:, 0], 0.90)].float().mean().item(),
        wavelength_weight=args.wavelength_weight)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print()
    print('facet accuracy   top-1 %.4f   majority-of-%d %.4f' % (top1, args.k, majority))
    print('facet coverage   ' + '  '.join('top%d %.3f' % (j+1, c)
                                          for j, c in enumerate(coverage)))
    print('exit angle       mean %.2f deg  median %.2f deg  (when facet correct %.2f deg)'
          % (report['exit_angle_deg']['mean'], report['exit_angle_deg']['median'],
             report['exit_angle_deg_when_facet_correct'] or float('nan')))
    print('neighbour dist   median %.5f  p10 %.5f  p90 %.5f'
          % (report['neighbour_distance']['median'], report['neighbour_distance']['p10'],
             report['neighbour_distance']['p90']))
    print('accuracy by distance   nearest decile %.4f   farthest decile %.4f'
          % (report['facet_accuracy_nearest_decile'], report['facet_accuracy_farthest_decile']))
    print('\nwrote %s' % args.output)


if __name__ == '__main__':
    main()
