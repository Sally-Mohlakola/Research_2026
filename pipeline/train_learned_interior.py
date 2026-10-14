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
from neural.learned_interior_data import load_learned_interior, split_entries
from neural.learned_interior_model import HEADS, encode_data
from utils.run_record import write_csv


LIMITATIONS = ['Validation is used for model selection, not an independent test set.',
               'Smooth distribution approximates singular specular transport.',
               'Escape probability is for the recorded finite depth cap.',
               'No render-quality or speedup claim; no RDM prior integrated yet.']


# score the model on the validation records: escape, facet and exit losses, and facet accuracy
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
        result['joint_nll'] = result['escape_bce'] + result['measured_escape_rate']*(result['facet_nll']+result['transformed_mixture_nll'])

    return result


# load a gathered pool, check its geometry hash and encode it into network inputs and targets
def load_pool(path):
    data, metadata = load_learned_interior(path)
    if metadata.get('transport_mode') != 'radiance':
        raise ValueError('Only radiance-mode records are supported')

    with np.load(path, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    digest = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
    if digest != metadata['geometry_sha256']:
        raise ValueError('Geometry hash mismatch')

    x, facet, y, valid, escaped, clipped = encode_data(data, vertices, faces)
    if not np.allclose(data['throughput'][escaped], 1, atol=1e-5):
        raise ValueError('This first model supports only lossless conditional transport')

    return data, metadata, vertices, faces, (x, facet, y, valid, escaped, clipped)


# train and keep the weights with the best validation joint loss
# with `epochs` set, train exactly that long (the old behaviour); otherwise train until the best
# validation loss has not improved by more than `tolerance` for `patience` epochs in a row
def train_model(model, tensors, train_idx, val_idx, batch_size, lr, epochs=None,
                tolerance=1e-3, patience=5, max_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    last_real_gain, stalled = float('inf'), 0
    limit = epochs if epochs is not None else max_epochs
    stop = 'fixed epochs' if epochs is not None else 'hit max_epochs before converging'

    for epoch in range(limit):
        began = time.perf_counter()
        model.train()
        totals, seen = np.zeros(4), 0
        order = train_idx[torch.randperm(len(train_idx))]
        for idx in order.split(batch_size):
            losses = model.losses(*[t[idx] for t in tensors])
            loss = losses[0] + tensors[3][idx].float().mean()*(losses[1]+losses[2])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            totals += len(idx)*np.array([losses[0].item(), losses[1].item(), losses[2].item(), loss.item()])
            seen += len(idx)
        train_seconds = time.perf_counter() - began

        began = time.perf_counter()
        metrics = evaluate(model, tensors, val_idx)
        validation_seconds = time.perf_counter() - began

        train = totals/seen
        history.append(dict(epoch=epoch+1, **metrics,
                            train_escape_bce=float(train[0]), train_facet_nll=float(train[1]),
                            train_coordinate_loss=float(train[2]), train_joint_loss=float(train[3]),
                            train_seconds=train_seconds, validation_seconds=validation_seconds))
        if metrics['joint_nll'] < best:
            best, best_epoch = metrics['joint_nll'], epoch+1
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: train {train[3]:.4f}, validation {metrics['joint_nll']:.4f} "
                  f"({train_seconds:.1f} s + {validation_seconds:.1f} s)", flush=True)

        if epochs is None:
            if metrics['joint_nll'] < last_real_gain - tolerance:
                last_real_gain, stalled = metrics['joint_nll'], 0
            else:
                stalled += 1
            if stalled >= patience:
                stop = 'converged'
                break

    convergence = dict(stop_reason=stop, epochs_run=len(history), best_epoch=best_epoch,
                       tolerance=tolerance if epochs is None else None,
                       patience=patience if epochs is None else None,
                       max_epochs=max_epochs if epochs is None else None,
                       rule=('best validation joint loss improved by no more than tolerance '
                             'for patience consecutive epochs' if epochs is None else 'fixed epoch count'))
    print(f"Stopped after {len(history)} epochs: {stop} (best epoch {best_epoch})", flush=True)
    return best_state, best_epoch, history, convergence


# check that sampled exits from held-out inputs are finite and point out of the stone
def check_samples(model, inputs):
    sample = model.sample(inputs, torch.Generator().manual_seed(91))
    outward = (sample['exit_direction']*model.normal[sample['exit_facet']]).sum(-1)
    assert torch.isfinite(sample['exit_position']).all() and (outward > 0).all()


# train one learned interior operator on a gathered pool and save model.pt and metrics.json
def main():
    parser = argparse.ArgumentParser(description='Train the learned interior operator on a gathered pool.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Train exactly this many epochs (how every earlier model was trained). '
                             'Leave it out to train until the validation loss has converged.')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Convergence: an improvement smaller than this does not count')
    parser.add_argument('--patience', type=int, default=5,
                        help='Convergence: epochs in a row without a real improvement before stopping')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Safety ceiling when training to convergence; reaching it is reported as not converged')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--head', choices=sorted(HEADS), default='mixture',
                        help="mixture is the Gaussian-mixture density; clone is the "
                             "deterministic regression baseline. For clone the third "
                             "loss term is a regression error, so the reported joint "
                             "figure is not a likelihood and is not comparable across "
                             "heads. Compare on exit-prediction error or on renders.")
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()

    if (args.epochs is not None and args.epochs < 1) or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch_size and lr must be positive')
    if args.tolerance <= 0 or args.patience < 1 or args.max_epochs < 1:
        parser.error('tolerance, patience and max_epochs must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)

    data, metadata, vertices, faces, encoded = load_pool(args.data)
    x, facet, y, valid, escaped, clipped = encoded

    train, validation = split_entries(data['entry_id'], seed=args.seed)
    train &= valid
    validation &= valid
    if not train.any() or not validation.any() or not escaped[train].any() or not escaped[validation].any():
        raise ValueError('Need valid escaped records in both entry-grouped splits')

    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    train_idx = torch.from_numpy(np.flatnonzero(train))
    val_idx = torch.from_numpy(np.flatnonzero(validation))

    model = HEADS[args.head](vertices, faces, args.width, args.components)
    initial = evaluate(model, tensors, val_idx)
    began = time.perf_counter()
    best_state, best_epoch, history, convergence = train_model(
        model, tensors, train_idx, val_idx, args.batch_size, args.lr, epochs=args.epochs,
        tolerance=args.tolerance, patience=args.patience, max_epochs=args.max_epochs)
    wall_seconds = time.perf_counter() - began

    model.load_state_dict(best_state)
    metrics = evaluate(model, tensors, val_idx)
    check_samples(model, tensors[0][val_idx])

    args.output.mkdir(parents=True)
    checkpoint = dict(schema_version=1, state_dict=best_state, width=args.width,
                      components=args.components, head=args.head, metadata=metadata,
                      vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces.astype(np.int64)),
                      input_radius=float(np.linalg.norm(vertices, axis=1).max()),
                      seed=args.seed, best_epoch=best_epoch,
                      source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      train_entry_ids=np.unique(data['entry_id'][train]).tolist(),
                      validation_entry_ids=np.unique(data['entry_id'][validation]).tolist(),
                      density_measure='barycentric logits and outgoing tangent slopes; not solid-angle/area PDF',
                      head_note=('Gaussian mixture density over exit coordinates'
                                 if args.head == 'mixture' else
                                 'deterministic smooth-L1 regression of exit coordinates; '
                                 'the third loss term is not a likelihood'))
    torch.save(checkpoint, args.output/'model.pt')

    timing = dict(wall_seconds=wall_seconds,
                  train_seconds=sum(h['train_seconds'] for h in history),
                  validation_seconds=sum(h['validation_seconds'] for h in history),
                  mean_train_seconds_per_epoch=float(np.mean([h['train_seconds'] for h in history])),
                  mean_validation_seconds_per_epoch=float(np.mean([h['validation_seconds'] for h in history])),
                  threads=torch.get_num_threads())
    report = dict(initial_validation=initial, best_validation=metrics,
                  best_epoch=best_epoch, convergence=convergence, timing=timing,
                  trainable_parameters=int(sum(p.numel() for p in model.parameters())),
                  optimiser=dict(name='Adam', lr=args.lr, batch_size=args.batch_size, gradient_clip_norm=10.),
                  train_records=int(train.sum()),
                  validation_records=int(validation.sum()), invalid_records=int((~valid).sum()),
                  coordinate_clipped_records=clipped, history=history,
                  limitations=LIMITATIONS)
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_csv(args.output/'history.csv', history)
    print(json.dumps(dict(output=str(args.output), best_epoch=best_epoch, validation=metrics), indent=2))


if __name__ == '__main__':
    main()
