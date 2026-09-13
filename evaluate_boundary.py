"""Evaluate a saved boundary model on an independently gathered dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from neural.boundary_data import load_boundary
from neural.boundary_model import load_model, encode_data
from train_boundary import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    model, checkpoint = load_model(args.model)
    data, metadata = load_boundary(args.data)
    for key in ('geometry_sha256', 'max_depth', 'dispersion', 'parameters', 'conditioning', 'transport_mode'):
        if metadata[key] != checkpoint['metadata'][key]:
            raise ValueError(f'Test dataset differs in {key}')
    if metadata['seed'] == checkpoint['metadata']['seed']:
        raise ValueError('Use an independently gathered seed for testing')
    with np.load(args.data, allow_pickle=False) as archive:
        v, f = archive['vertices'].copy(), archive['faces'].copy()
    if hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest() != metadata['geometry_sha256']:
        raise ValueError('Geometry hash mismatch')
    x, facet, y, valid, escaped, clipped = encode_data(data, v, f)
    if not valid.any() or not escaped.any():
        raise ValueError('No valid escaped test records')
    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    indices = torch.from_numpy(np.flatnonzero(valid))
    report = dict(test=evaluate(model, tensors, indices), records=int(valid.sum()),
                  invalid_records=int((~valid).sum()), coordinate_clipped_records=clipped,
                  seed=metadata['seed'], model=str(args.model),
                  note='Transformed-coordinate likelihood, not physical area/solid-angle density. No render comparison.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
