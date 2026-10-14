import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch

from neural.learned_interior_data import load_learned_interior
from neural.learned_interior_model import HEADS, encode_data
from pipeline.train_learned_interior import evaluate


# load and encode a group of shards, keeping only valid records
def load_group(paths):
    xs, facets, ys, escapes = [], [], [], []
    vertices = faces = None

    for path in paths:
        data, _ = load_learned_interior(path)
        with np.load(path, allow_pickle=False) as archive:
            vertices, faces = archive['vertices'].copy(), archive['faces'].copy()

        x, facet, y, valid, escaped, _ = encode_data(data, vertices, faces)
        if not np.allclose(data['throughput'][escaped], 1, atol=1e-5):
            raise ValueError('Only lossless conditional transport is supported')

        xs.append(x[valid]); facets.append(facet[valid])
        ys.append(y[valid]); escapes.append(escaped[valid])

    return [torch.from_numpy(np.concatenate(a)) for a in (xs, facets, ys, escapes)], vertices, faces


# split each shard directory into training shards and its last few as validation, so validation covers every pool
def split_shards(args, parser):
    if args.use_shards and len(args.use_shards) != len(args.shards):
        parser.error('--use_shards needs one count per --shards directory')

    train, validation, geometry = [], [], set()
    for index, directory in enumerate(args.shards):
        found = sorted(directory.glob('shard_*.npz'))
        if len(found) <= args.validation_shards:
            raise ValueError('Not enough shards in %s' % directory)

        validation += found[-args.validation_shards:]
        pool = found[:-args.validation_shards]
        if args.use_shards and args.use_shards[index]:
            pool = pool[:args.use_shards[index]]
        train += pool
        geometry.add(load_learned_interior(found[0])[1]['geometry_sha256'])

    if len(geometry) != 1:
        raise ValueError('Shard directories use different geometry')
    return train, validation


# the pool's metadata, naming every pool when several are mixed
def training_metadata(args, first_shard):
    _, metadata = load_learned_interior(first_shard)
    metadata = {k: v for k, v in metadata.items() if k not in ('shard', 'attempts')}
    if len(args.shards) > 1:
        metadata['entry_sampling'] = 'mixed: ' + ', '.join(
            '%s (%s)' % (d.name, load_learned_interior(sorted(d.glob('shard_*.npz'))[0])[1]
                         .get('entry_sampling')) for d in args.shards)
    return metadata


# warm start from an already trained operator with the same head and geometry
def warm_start(model, path, head, faces):
    initial = torch.load(path, map_location='cpu', weights_only=True)
    if initial.get('head', 'mixture') != head:
        raise ValueError('--init checkpoint uses a different head')
    if not torch.equal(initial['faces'], torch.from_numpy(faces.astype(np.int64))):
        raise ValueError('--init checkpoint uses different geometry')
    model.load_state_dict(initial['state_dict'])
    print('warm start from %s' % path, flush=True)


# train an operator by streaming shard groups from disk, resumable from state.pt after every group
def main():
    parser = argparse.ArgumentParser(description='Train the learned interior operator on sharded pools.')
    parser.add_argument('--shards', type=Path, nargs='+', required=True,
                        help='One or more gather_sharded output directories, mixed together')
    parser.add_argument('--use_shards', type=int, nargs='+',
                        help='Training shards to take from each directory, in order; '
                             '0 means all')
    parser.add_argument('--init', type=Path,
                        help='Warm-start from this checkpoint (same head and geometry)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', choices=sorted(HEADS), default='clone_deep')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--validation_shards', type=int, default=2)
    parser.add_argument('--group', type=int, default=4, help='Shards shuffled together')
    parser.add_argument('--validate_every', type=int, default=20,
                        help='Shard groups between validation passes')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max_seconds', type=float, default=0,
                        help='Save and stop after this long; rerun to resume')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    train, validation = split_shards(args, parser)
    val_tensors, vertices, faces = load_group(validation)
    val_idx = torch.arange(len(val_tensors[0]))
    metadata = training_metadata(args, train[0])

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output/'state.pt'
    torch.manual_seed(args.seed)
    model = HEADS[args.head](vertices, faces, args.width, args.components)
    if args.init and not state_path.exists():
        warm_start(model, args.init, args.head, faces)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        print('resuming at epoch %d, group %d, %d records seen'
              % (state['epoch']+1, state['group'], state['visits']), flush=True)
    else:
        state = dict(epoch=0, group=0, visits=0, best=float('inf'), best_state=None,
                     best_visits=0, history=[],
                     initial=evaluate(model, val_tensors, val_idx))
        print('%d training shards, %d validation shards (%d records), %d epochs'
              % (len(train), len(validation), len(val_idx), args.epochs), flush=True)

    # write weights, optimiser, random state and position, so a killed run can resume
    def save():
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     torch_rng=torch.get_rng_state())
        torch.save(state, state_path)

    # score on the validation shards and keep the best weights so far
    def validate():
        metrics = evaluate(model, val_tensors, val_idx)
        state['history'].append(dict(visits=state['visits'], **metrics))
        if metrics['joint_nll'] < state['best']:
            state['best'], state['best_visits'] = metrics['joint_nll'], state['visits']
            state['best_state'] = copy.deepcopy(model.state_dict())
        print('  epoch %d, %.1fM records seen: validation joint NLL %.4f, facet top-1 %.4f'
              % (state['epoch']+1, state['visits']/1e6, metrics['joint_nll'],
                 metrics['facet_accuracy']), flush=True)

    began = time.perf_counter()
    while state['epoch'] < args.epochs:
        order = np.random.default_rng(args.seed+state['epoch']).permutation(len(train))
        groups = [order[i:i+args.group] for i in range(0, len(order), args.group)]

        while state['group'] < len(groups):
            if args.max_seconds and time.perf_counter()-began > args.max_seconds:
                save()
                print('paused at epoch %d, group %d/%d; rerun to resume'
                      % (state['epoch']+1, state['group'], len(groups)))
                return

            (x, facet, y, escaped), _, _ = load_group([train[i] for i in groups[state['group']]])
            model.train()
            for idx in torch.randperm(len(x)).split(args.batch_size):
                losses = model.losses(x[idx], facet[idx], y[idx], escaped[idx])
                loss = losses[0] + escaped[idx].float().mean()*(losses[1]+losses[2])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()

            state['visits'] += len(x)
            state['group'] += 1
            if state['group'] % args.validate_every == 0 or state['group'] == len(groups):
                validate()
            save()

        state['epoch'] += 1
        state['group'] = 0
        save()

    model.load_state_dict(state['best_state'])
    metrics = evaluate(model, val_tensors, val_idx)

    manifest_bytes = b''.join((d/'manifest.json').read_bytes() for d in args.shards)
    checkpoint = dict(schema_version=1, state_dict=state['best_state'], width=args.width,
                      components=args.components, head=args.head, metadata=metadata,
                      vertices=torch.from_numpy(vertices),
                      faces=torch.from_numpy(faces.astype(np.int64)),
                      input_radius=float(np.linalg.norm(vertices, axis=1).max()),
                      seed=args.seed, best_visits=state['best_visits'],
                      source_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                      source=[str(d) for d in args.shards],
                      initialised_from=str(args.init) if args.init else None,
                      validation_shards=[str(p) for p in validation],
                      density_measure='barycentric logits and outgoing tangent slopes; '
                                      'not solid-angle/area PDF')
    torch.save(checkpoint, args.output/'model.pt')

    report = dict(initial_validation=state['initial'], best_validation=metrics,
                  best_visits=state['best_visits'], total_visits=state['visits'],
                  train_shards=len(train), validation_records=len(val_idx),
                  history=state['history'])
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(args.output), best_visits=state['best_visits'],
                          validation=metrics), indent=2))


if __name__ == '__main__':
    main()
