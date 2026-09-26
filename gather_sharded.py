"""Gather boundary transport at scales that do not fit in memory.

gather_boundary.py holds every record as a Python dictionary until the end,
which is fine at a million records and impossible at fifty million on a 15 GB
machine. This drives the same `gather` function, unchanged, over many small
shards and writes each one to disk as soon as it is finished, so peak memory is
one shard per worker regardless of the total.

Shards are independent and resumable. Shard k always covers entry IDs
[k*entries_per_shard, (k+1)*entries_per_shard) and draws from the k-th child of
one SeedSequence, so its contents depend on the seed and its index and not on
how many workers ran it or in what order. A shard that already exists is
skipped, so an interrupted gather resumes where it stopped, and `--max_seconds`
lets a gather be run in bounded steps.

Every shard carries the same metadata and geometry as a gather_boundary.py pool,
so `load_boundary` and the rest of the pipeline read shards unchanged.
"""
import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import gather_boundary as gb
from config.parameters import get_diamond_parameters


def _run_shard(task):
    (index, diamond_name, entries, paths, max_depth, child, dispersion,
     sampling, azimuth_range, directory) = task
    arrays, attempts = gb._worker((diamond_name, index*entries, entries, paths,
                                   max_depth, child, dispersion, sampling, azimuth_range))
    return index, arrays, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--diamond_name', default='pear_brilliant')
    parser.add_argument('--records', type=int, required=True,
                        help='Total paths to gather, rounded up to whole shards')
    parser.add_argument('--paths_per_entry', type=int, default=16)
    parser.add_argument('--entries_per_shard', type=int, default=8192)
    parser.add_argument('--max_depth', type=int, default=128)
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--no_dispersion', action='store_true')
    parser.add_argument('--entry_sampling', choices=['uniform', 'camera'], default='camera')
    parser.add_argument('--azimuth_range', type=float, nargs=2, default=[0., 360.])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_seconds', type=float, default=0,
                        help='Stop starting new shards after this long; 0 = no limit')
    args = parser.parse_args()

    per_shard = args.entries_per_shard*args.paths_per_entry
    shards = -(-args.records//per_shard)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = get_diamond_parameters(args.diamond_name)
    _, vertices, faces = gb.build_scene(parameters)
    geometry = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
    metadata = dict(schema_version=1, diamond_name=args.diamond_name, parameters=parameters,
                    seed=args.seed, max_depth=args.max_depth,
                    dispersion=not args.no_dispersion, geometry_sha256=geometry,
                    coordinates='object_space',
                    direction_convention='entry points into stone; exit points out',
                    transport_mode='radiance',
                    conditioning='entry transmission forced; multiply entry_transmittance once',
                    status_codes={'escaped': 0, 'truncated': 1, 'invalid': 2},
                    entry_sampling=args.entry_sampling, azimuth_range=list(args.azimuth_range),
                    sharding=dict(shards=shards, entries_per_shard=args.entries_per_shard,
                                  paths_per_entry=args.paths_per_entry,
                                  seed_scheme='SeedSequence(seed).spawn(shards)[index]'))
    (args.output_dir/'manifest.json').write_text(json.dumps(metadata, indent=2))

    children = np.random.SeedSequence(args.seed).spawn(shards)
    todo = [k for k in range(shards) if not (args.output_dir/('shard_%04d.npz' % k)).exists()]
    print('%d shards of %d records = %d records; %d already on disk, %d to gather'
          % (shards, per_shard, shards*per_shard, shards-len(todo), len(todo)), flush=True)
    if not todo:
        return

    began, done = time.perf_counter(), 0
    tasks = iter([(k, args.diamond_name, args.entries_per_shard, args.paths_per_entry,
                   args.max_depth, children[k], not args.no_dispersion,
                   args.entry_sampling, tuple(args.azimuth_range), args.output_dir)
                  for k in todo])
    with multiprocessing.Pool(args.workers) as pool:
        pending = set()

        def submit():
            if args.max_seconds and time.perf_counter()-began > args.max_seconds:
                return False
            task = next(tasks, None)
            if task is None:
                return False
            pending.add(pool.apply_async(_run_shard, (task,)))
            return True

        for _ in range(args.workers):
            if not submit():
                break
        while pending:
            finished = [p for p in pending if p.ready()]
            if not finished:
                time.sleep(.2)
                continue
            for p in finished:
                pending.discard(p)
                index, arrays, attempts = p.get()
                target = args.output_dir/('shard_%04d.npz' % index)
                partial = target.with_suffix('.tmp.npz')
                shard_meta = dict(metadata, shard=index, attempts=attempts)
                np.savez_compressed(partial, **arrays, vertices=vertices, faces=faces,
                                    metadata_json=np.array(json.dumps(shard_meta)))
                partial.replace(target)
                done += 1
                submit()
            elapsed = time.perf_counter()-began
            print('  %d/%d shards this run, %.0f s, %.0f records/s'
                  % (done, len(todo), elapsed, done*per_shard/max(elapsed, 1e-9)), flush=True)
    remaining = sum(1 for k in range(shards)
                    if not (args.output_dir/('shard_%04d.npz' % k)).exists())
    print('finished this run: %d shards remaining' % remaining)


if __name__ == '__main__':
    main()
