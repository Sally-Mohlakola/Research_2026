import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np

from neural.learned_interior_data import load_learned_interior


# join gathered pools of one stone into one, renumbering entry states so each pool's stay distinct
def merge(paths):
    merged, sources, offset = {}, [], 0
    geometry = None
    for path in paths:
        data, metadata = load_learned_interior(path)
        with np.load(path, allow_pickle=False) as archive:
            vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
        if geometry is None:
            geometry = (metadata['geometry_sha256'], vertices, faces, metadata)
        elif metadata['geometry_sha256'] != geometry[0]:
            raise ValueError('%s was gathered on a different stone' % path)
        data['entry_id'] = data['entry_id'] + offset
        offset = int(data['entry_id'].max()) + 1
        for name, values in data.items():
            merged.setdefault(name, []).append(values)
        sources.append(dict(path=str(path), entry_sampling=metadata['entry_sampling'], seed=metadata['seed'],
                            records=int(len(data['entry_id'])),
                            entries=int(len(np.unique(data['entry_id'])))))
    return {name: np.concatenate(values) for name, values in merged.items()}, geometry, sources


# one pool from several gathers of the same stone, e.g. camera-matched plus uniform with the aim offset
def main():
    parser = argparse.ArgumentParser(description='Merge gathered learned interior pools of one stone.')
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    data, (_, vertices, faces, first), sources = merge(args.inputs)
    metadata = dict(first)
    metadata['entry_sampling'] = 'mixed'
    metadata['entry_sampling_detail'] = ' + '.join('%s (%d entries)' % (s['entry_sampling'], s['entries'])
                                                   for s in sources)
    metadata['seed'] = [s['seed'] for s in sources]
    metadata['merged_from'] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data, vertices=vertices, faces=faces,
                        metadata_json=np.array(json.dumps(metadata)))
    print('merged %d pools: %d records, %d entry states -> %s'
          % (len(sources), len(data['entry_id']), len(np.unique(data['entry_id'])), args.output))


if __name__ == '__main__':
    main()
