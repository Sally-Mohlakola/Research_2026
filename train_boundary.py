"""Train and evaluate a conditional escape distribution on grouped entry states."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from neural.boundary_data import load_boundary, split_entries
from neural.boundary_model import BoundaryModel, encode_data


def evaluate(model, tensors, indices):
    model.eval()
    x, facet, y, escaped = [t[indices] for t in tensors]
    with torch.no_grad():
        losses = model.losses(x, facet, y, escaped)
        h = model.encoder(x)
        probability = model.escape(h).squeeze(-1).sigmoid()
        result = dict(escape_bce=float(losses[0]), facet_nll=float(losses[1]),
                      transformed_mixture_nll=float(losses[2]),
                      escape_brier=float((probability-escaped.float()).square().mean()),
                      measured_escape_rate=float(escaped.float().mean()),
                      predicted_escape_rate=float(probability.mean()))
        result['facet_accuracy'] = (float((model.facet(h[escaped]).argmax(-1)==facet[escaped]).float().mean())
                                    if escaped.any() else None)
        # Joint likelihood, accounting for continuous targets only on escaped records.
        result['joint_nll'] = result['escape_bce'] + result['measured_escape_rate']*(result['facet_nll']+result['transformed_mixture_nll'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch_size and lr must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data, metadata = load_boundary(args.data)
    if metadata.get('transport_mode') != 'radiance':
        raise ValueError('Only radiance-mode records are supported')
    with np.load(args.data, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    digest = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
    if digest != metadata['geometry_sha256']:
        raise ValueError('Geometry hash mismatch')
    x, facet, y, valid, escaped, clipped = encode_data(data, vertices, faces)
    if not np.allclose(data['throughput'][escaped], 1, atol=1e-5):
        raise ValueError('This first model supports only lossless conditional transport')
    train, validation = split_entries(data['entry_id'], seed=args.seed)
    train &= valid
    validation &= valid
    if not train.any() or not validation.any() or not escaped[train].any() or not escaped[validation].any():
        raise ValueError('Need valid escaped records in both entry-grouped splits')
    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    train_idx = torch.from_numpy(np.flatnonzero(train))
    val_idx = torch.from_numpy(np.flatnonzero(validation))
    model = BoundaryModel(vertices, faces, args.width, args.components)
    initial = evaluate(model, tensors, val_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for idx in order.split(args.batch_size):
            losses = model.losses(*[t[idx] for t in tensors])
            loss = losses[0] + tensors[3][idx].float().mean()*(losses[1]+losses[2])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        metrics = evaluate(model, tensors, val_idx)
        history.append(dict(epoch=epoch+1, **metrics))
        if metrics['joint_nll'] < best:
            best, best_epoch = metrics['joint_nll'], epoch+1
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: validation joint NLL {metrics['joint_nll']:.4f}", flush=True)
    model.load_state_dict(best_state)
    metrics = evaluate(model, tensors, val_idx)
    # Sampling constraints on held-out inputs, independent of target fitting.
    sample = model.sample(tensors[0][val_idx], torch.Generator().manual_seed(91))
    outward = (sample['exit_direction']*model.normal[sample['exit_facet']]).sum(-1)
    assert torch.isfinite(sample['exit_position']).all() and (outward > 0).all()
    args.output.mkdir(parents=True)
    checkpoint = dict(schema_version=1, state_dict=best_state, width=args.width,
                      components=args.components, metadata=metadata,
                      vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces.astype(np.int64)),
                      input_radius=float(np.linalg.norm(vertices, axis=1).max()),
                      seed=args.seed, best_epoch=best_epoch,
                      source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      train_entry_ids=np.unique(data['entry_id'][train]).tolist(),
                      validation_entry_ids=np.unique(data['entry_id'][validation]).tolist(),
                      density_measure='barycentric logits and outgoing tangent slopes; not solid-angle/area PDF')
    torch.save(checkpoint, args.output/'model.pt')
    report = dict(initial_validation=initial, best_validation=metrics,
                  best_epoch=best_epoch, train_records=int(train.sum()),
                  validation_records=int(validation.sum()), invalid_records=int((~valid).sum()),
                  coordinate_clipped_records=clipped, history=history,
                  limitations=['Validation is used for model selection, not an independent test set.',
                               'Smooth distribution approximates singular specular transport.',
                               'Escape probability is for the recorded finite depth cap.',
                               'No render-quality or speedup claim; no RDM prior integrated yet.'])
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), best_epoch=best_epoch, validation=metrics), indent=2))


if __name__ == '__main__':
    main()
