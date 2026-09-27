"""Train the dispersion head on a paired-wavelength gather.

Every entry whose hero lane escaped yields one training pair per other lane:
the gate learns whether that colour stayed on the hero's route, and on the
pairs that did, the shift learns how its exit coordinates moved. Validation is
a separate gather with its own seed, so no entry state appears on both sides.

Reported alongside the losses are the numbers that decide whether the head is
useful: gate accuracy against always predicting the majority outcome, and, for
colours that stay, the angle between predicted and true exit directions
against the zero-shift baseline -- which is what the boundary operator does
today, since it ignores wavelength.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from neural.dispersion_model import DispersionModel, coordinates, decode


def load_pairs(path, model):
    """Flatten a paired gather into (x, facet, y, wavelength, stays, shift, usable)."""
    with np.load(path, allow_pickle=False) as data:
        d = {k: data[k] for k in data.files}
    metadata = json.loads(str(d['metadata_json']))
    radius = float(np.linalg.norm(d['vertices'], axis=1).max())
    hero = d['status'][:, 0] == 0
    lanes = d['status'].shape[1]
    rows = np.flatnonzero(hero)
    x = np.concatenate([d['entry_position'][rows]/radius, d['entry_direction'][rows],
                        d['entry_normal'][rows], (d['wavelength_nm'][rows, :1]-595)/235], 1)
    facet = torch.from_numpy(d['exit_facet'][rows, 0].astype(np.int64))
    buffers = (model.triangles, model.tangent, model.bitangent, model.normal)
    y0, clipped0 = coordinates(*buffers, facet, torch.from_numpy(d['exit_position'][rows, 0]),
                               torch.from_numpy(d['exit_direction'][rows, 0]))
    out = {k: [] for k in ('x', 'facet', 'y', 'wavelength', 'stays', 'shift', 'usable',
                           'hero_direction', 'direction')}
    for k in range(1, lanes):
        stays = (d['status'][rows, k] == 0) & (d['route'][rows, k] == d['route'][rows, 0])
        position = np.where(stays[:, None], d['exit_position'][rows, k], d['exit_position'][rows, 0])
        direction = np.where(stays[:, None], d['exit_direction'][rows, k], d['exit_direction'][rows, 0])
        yk, clipped = coordinates(*buffers, facet, torch.from_numpy(position), torch.from_numpy(direction))
        out['x'].append(torch.from_numpy(x.astype(np.float32)))
        out['facet'].append(facet)
        out['y'].append(y0)
        out['wavelength'].append(torch.from_numpy(d['wavelength_nm'][rows, k]))
        out['stays'].append(torch.from_numpy(stays))
        out['shift'].append(yk - y0)
        # Clamped coordinates at facet edges or grazing exits are not the true
        # shift; they still count for the gate but not for the shift loss.
        out['usable'].append(torch.from_numpy(stays) & ~clipped & ~clipped0)
        out['hero_direction'].append(torch.from_numpy(d['exit_direction'][rows, 0]))
        out['direction'].append(torch.from_numpy(d['exit_direction'][rows, k]))
    pairs = {k: torch.cat(v) for k, v in out.items()}
    return pairs, metadata, d['vertices'], d['faces']


@torch.no_grad()
def evaluate(model, pairs, batch=65536):
    gate_loss = shift_loss = 0.
    correct = 0
    errors, baseline = [], []
    n = len(pairs['stays'])
    for start in range(0, n, batch):
        s = slice(start, start+batch)
        stays, usable = pairs['stays'][s], pairs['usable'][s]
        logit, shift = model(pairs['x'][s], pairs['facet'][s], pairs['y'][s], pairs['wavelength'][s])
        gate_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            logit, stays.float(), reduction='sum').item()
        correct += ((logit > 0) == stays).sum().item()
        if usable.any():
            shift_loss += torch.nn.functional.smooth_l1_loss(
                shift[usable]/model.shift_scale, pairs['shift'][s][usable]/model.shift_scale,
                reduction='sum').item()/4
            _, predicted = _decode(model, pairs['facet'][s][usable], pairs['y'][s][usable] + shift[usable])
            truth = pairs['direction'][s][usable]
            errors.append(_angle(predicted, truth))
            baseline.append(_angle(pairs['hero_direction'][s][usable], truth))
    errors, baseline = torch.cat(errors), torch.cat(baseline)
    rate = pairs['stays'].float().mean().item()
    usable_count = int(pairs['usable'].sum())
    return dict(pairs=n, stay_rate=rate,
                gate_bce=gate_loss/n,
                gate_bce_constant=float(-(rate*np.log(rate)+(1-rate)*np.log(1-rate))),
                gate_accuracy=correct/n, gate_accuracy_majority=max(rate, 1-rate),
                shift_loss=shift_loss/max(usable_count, 1), shift_pairs=usable_count,
                exit_angle_error_deg_median=float(errors.median()),
                exit_angle_error_deg_mean=float(errors.mean()),
                zero_shift_angle_deg_median=float(baseline.median()),
                zero_shift_angle_deg_mean=float(baseline.mean()),
                spread_explained=float(1 - errors.square().mean()/baseline.square().mean()))


def _decode(model, facet, y):
    return decode(model.triangles, model.tangent, model.bitangent, model.normal, facet, y)


def _angle(a, b):
    return torch.rad2deg(torch.arccos((a*b).sum(-1).clamp(-1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    with np.load(args.train, allow_pickle=False) as data:
        vertices, faces = data['vertices'], data['faces']
    model = DispersionModel(vertices, faces, args.width, args.depth)
    train, metadata, _, _ = load_pairs(args.train, model)
    validation, vmeta, _, vfaces = load_pairs(args.validation, model)
    if not np.array_equal(faces, vfaces):
        raise ValueError('Training and validation gathers use different geometry')
    # Robust per-coordinate scale: the shift has heavy tails at grazing exits.
    usable = train['shift'][train['usable']]
    scale = 1.4826*(usable - usable.median(0).values).abs().median(0).values
    model.shift_scale.copy_(scale.clamp_min(1e-4))
    print('%d training pairs (%.1f%% stay), %d validation pairs; shift scale %s'
          % (len(train['stays']), 100*train['stays'].float().mean(), len(validation['stays']),
             np.round(model.shift_scale.numpy(), 5).tolist()), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs*((len(train['stays'])+args.batch_size-1)//args.batch_size))
    initial = evaluate(model.eval(), validation)
    best, best_state, history = float('inf'), None, []
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(train['stays'])).split(args.batch_size):
            gate_loss, shift_loss = model.losses(train['x'][idx], train['facet'][idx], train['y'][idx],
                                                 train['wavelength'][idx], train['stays'][idx],
                                                 train['shift'][idx], train['usable'][idx])
            loss = gate_loss + shift_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            schedule.step()
        metrics = evaluate(model.eval(), validation)
        history.append(dict(epoch=epoch+1, **metrics))
        score = metrics['gate_bce'] + metrics['shift_loss']
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
        print('epoch %d  gate BCE %.4f (constant %.4f)  accuracy %.3f (majority %.3f)  '
              'exit error %.3f deg (zero shift %.3f)  spread explained %.3f'
              % (epoch+1, metrics['gate_bce'], metrics['gate_bce_constant'], metrics['gate_accuracy'],
                 metrics['gate_accuracy_majority'], metrics['exit_angle_error_deg_median'],
                 metrics['zero_shift_angle_deg_median'], metrics['spread_explained']), flush=True)

    model.load_state_dict(best_state)
    final = evaluate(model.eval(), validation)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version=1, kind='dispersion', state_dict=best_state,
                    width=args.width, depth=args.depth,
                    vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces.astype(np.int64)),
                    metadata=dict(diamond_name=metadata['diamond_name'],
                                  geometry_sha256=metadata['geometry_sha256'],
                                  train=str(args.train), validation=str(args.validation),
                                  train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest()),
                    seed=args.seed), args.output/'model.pt')
    report = dict(initial=initial, best=final, history=history,
                  seconds=time.perf_counter()-began, arguments={k: str(v) for k, v in vars(args).items()})
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
