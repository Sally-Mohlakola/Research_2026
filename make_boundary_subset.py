"""Write a nested entry-grouped subset of a gathered boundary dataset.

Subsets select whole entry states, never individual paths, so every rung of a
data ladder keeps the repeated outcomes of an entry together and remains
compatible with the entry-grouped split used for training. Geometry, metadata
and gather seed are carried through unchanged; only the record count differs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def subset(archive, entries):
    """Keep the `entries` lowest distinct entry_id values from a gathered pool."""
    entry_id = archive['entry_id']
    unique = np.unique(entry_id)
    if entries > len(unique):
        raise ValueError(f'Pool holds {len(unique)} entry states, requested {entries}')
    keep = np.isin(entry_id, unique[:entries])
    count = len(entry_id)
    data = {name: (value[keep] if getattr(value, 'shape', ()) and value.shape[0] == count else value)
            for name, value in archive.items() if name != 'metadata_json'}
    return data, int(keep.sum()), len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entries', type=int, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace {args.output}')
    if args.entries < 2:
        parser.error('--entries must be at least two for a grouped split')
    with np.load(args.pool, allow_pickle=False) as archive:
        contents = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(contents['metadata_json']))
    data, records, available = subset(contents, args.entries)
    metadata['subset_of'] = str(args.pool)
    metadata['subset_entries'] = args.entries
    metadata['pool_entries'] = available
    metadata['subset_rule'] = 'lowest entry_id values; nested across rungs; whole entry states only'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data,
                        metadata_json=np.array(json.dumps(metadata)))
    print(f'{records} records from {args.entries}/{available} entry states -> {args.output}')


if __name__ == '__main__':
    main()
