"""Convergence traces of the trained operators, from their saved training histories.

Every trainer evaluates on its held-out validation split after each epoch (or
each group of shards) and writes the history to the model's `metrics.json`.
This plots those histories into `figures/plots/`; nothing is retrained.
Training loss was not logged, so the traces show held-out error only -- the
quantity that decides model selection, but not a train/validation gap.

Three figures:

- `convergence_operators.png`: one row per cut (GIA round, pear, step cut),
  every head trained on that cut's pool. Exit-facet NLL and top-1 accuracy are
  the same quantity for every head, so heads share those axes. The coordinate
  term is not: the mixture's is a negative log likelihood, the clones' a
  smooth-L1 regression error, so each head gets its own axis in the third
  column. The selected (best) epoch is marked on every curve.
- `convergence_streaming.png`: the pear operators trained over shards, by
  records seen rather than epochs.
- `convergence_dispersion.png`: the dispersion head's gate and shift.

Models are looked up in `checkpoints/`, then in `../checkpoint_history/`, where
older ones were moved.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEARCH = (REPO/'checkpoints', REPO.parent/'checkpoint_history')
# (row title, [(label, model folder, head)]); every model in a row shares one pool.
CUTS = [
    ('GIA round (camera_pool)', [('mixture', 'camera_model', 'mixture'),
                                 ('clone', 'clone_model', 'clone'),
                                 ('clone_deep', 'gia_clone_deep', 'clone_deep')]),
    ('pear (pear_pool)', [('mixture', 'pear_model', 'mixture'),
                          ('clone_deep', 'pear_deep', 'clone_deep')]),
    ('step cut (step_pool)', [('mixture', 'step_model', 'mixture'),
                              ('clone_deep', 'step_clone_deep', 'clone_deep')]),
]
STREAMING = [('clone_deep, 50M camera records', 'pear_deep_50m'),
             ('clone_deep, 50M camera + 25M uniform (warm start)', 'pear_deep_mixed')]
COLOURS = {'mixture': '#2a6fb0', 'clone': '#d08a2c', 'clone_deep': '#b5446e'}


def load(folder):
    for root in SEARCH:
        path = root/folder/'metrics.json'
        if path.exists():
            return json.loads(path.read_text()), path
    return None, None


def operators(output):
    fig, axes = plt.subplots(len(CUTS), 3, figsize=(16, 4.2*len(CUTS)), squeeze=False)
    used = []
    for row, (title, models) in enumerate(CUTS):
        coord_axis = axes[row, 2]
        twin = None
        for label, folder, head in models:
            report, path = load(folder)
            if report is None:
                print('missing %s, skipped' % folder)
                continue
            used.append(str(path))
            history = report['history']
            epochs = [h['epoch'] for h in history]
            best = report.get('best_epoch')
            colour = COLOURS[head]
            for col, key in ((0, 'facet_nll'), (1, 'facet_accuracy')):
                values = [h[key] for h in history]
                axes[row, col].plot(epochs, values, color=colour, label='%s (%s)' % (label, folder))
                if best:
                    axes[row, col].plot(best, values[best-1], 'o', color=colour, ms=6)
            values = [h['transformed_mixture_nll'] for h in history]
            if head == 'mixture':
                axis = coord_axis
            else:
                twin = twin or coord_axis.twinx()
                axis = twin
            axis.plot(epochs, values, color=colour, label=label,
                      linestyle='-' if head == 'mixture' else '--')
            if best:
                axis.plot(best, values[best-1], 'o', color=colour, ms=6)
        axes[row, 0].set_ylabel(title, fontsize=11)
        axes[row, 0].set_title('exit-facet NLL (lower is better)' if row == 0 else '')
        axes[row, 1].set_title('exit-facet top-1 accuracy' if row == 0 else '')
        coord_axis.set_title('exit coordinates: mixture NLL (left, solid),\n'
                             'clone smooth-L1 (right, dashed)' if row == 0 else '')
        coord_axis.set_ylabel('mixture NLL')
        if twin is not None:
            twin.set_ylabel('clone smooth-L1')
        axes[row, 0].legend(fontsize=8)
        for col in range(3):
            axes[row, col].set_xlabel('epoch')
            axes[row, col].grid(alpha=.3)
    fig.suptitle('Validation traces of the boundary operators (dots mark the selected epoch)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output, dpi=130, facecolor='white')
    return used


def streaming(output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    used = []
    for (label, folder), colour in zip(STREAMING, ('#b5446e', '#6a3d9a')):
        report, path = load(folder)
        if report is None:
            print('missing %s, skipped' % folder)
            continue
        used.append(str(path))
        history = report['history']
        seen = [h['visits']/1e6 for h in history]
        for axis, key in zip(axes, ('facet_nll', 'facet_accuracy')):
            axis.plot(seen, [h[key] for h in history], 'o-', color=colour, label=label)
    axes[0].set_title('exit-facet NLL'); axes[1].set_title('exit-facet top-1 accuracy')
    for axis in axes:
        axis.set_xlabel('training records seen (millions)'); axis.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Streaming training of the pear operators (validation on held-out shards)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=130, facecolor='white')
    return used


def dispersion(output):
    report, path = load('dispersion_pear/head')
    if report is None:
        print('missing dispersion_pear/head, skipped')
        return []
    history = report['history']
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(epochs, [h['gate_bce'] for h in history], 'o-', color='#2a6fb0', label='gate')
    axes[0].axhline(history[0]['gate_bce_constant'], color='grey', ls=':', label='constant predictor')
    axes[0].set_title('gate cross-entropy (stays on the hero route?)')
    axes[1].plot(epochs, [h['exit_angle_error_deg_median'] for h in history], 'o-', color='#b5446e',
                 label='dispersion head')
    axes[1].axhline(history[0]['zero_shift_angle_deg_median'], color='grey', ls=':',
                    label='no shift (the operator alone)')
    axes[1].set_title('median exit error of a colour that stays (degrees)')
    for axis in axes:
        axis.set_xlabel('epoch'); axis.grid(alpha=.3); axis.legend(fontsize=8)
    fig.suptitle('Dispersion head, pear (validation on a separate gather)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=130, facecolor='white')
    return [str(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_dir', type=Path, default=REPO/'figures'/'plots')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in (('convergence_operators.png', operators),
                       ('convergence_streaming.png', streaming),
                       ('convergence_dispersion.png', dispersion)):
        used = draw(args.output_dir/name)
        print('wrote %s from %d histories' % (args.output_dir/name, len(used)))


if __name__ == '__main__':
    main()
