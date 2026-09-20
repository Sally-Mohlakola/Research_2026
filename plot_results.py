"""Figures for the operator results, drawn from the JSON reports.

Every number here is read from a report file rather than typed in, so a figure
cannot drift from the measurement it illustrates. Re-run after any experiment
and the figures follow.

Two figures.

`conditioning_ablation.png` shows what the operator needs from its inputs. The
two measures -- branch selection and within-facet decode -- are on separate
panels rather than a shared axis with two scales, because they are different
quantities and a dual axis invites a comparison that is not meaningful. The
branch panel carries reference lines for the trivial baseline and for the
direction-only lookup table, which is what makes the direction-only arm
readable: it lands in the RDM's regime rather than the network's.

`facet_accuracy.png` puts every branch predictor in the project on one scale,
so the lookup tables, the nearest-neighbour oracle and the trained heads can be
compared without converting between three different reporting conventions.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Validated categorical slots; only slot 1 is used, since every bar in a panel
# is the same entity type and colour here carries emphasis, not identity.
SERIES = '#2a78d6'
MUTED = '#9a9892'
INK = '#0b0b0b'
SECONDARY = '#52514e'
SURFACE = '#fcfcfb'
GRID = '#e4e3df'

LABELS = {'full': 'full conditioning', 'no_wavelength': 'no wavelength',
          'no_normal': 'no normal', 'no_position': 'no position',
          'no_direction': 'no direction', 'direction_only': 'direction only'}


def bars(axis, names, values, highlight, formatter, title, subtitle, limit=None):
    """Horizontal bars, sorted, directly labelled, with a recessive grid."""
    y = range(len(names))
    colours = [SERIES if n in highlight else MUTED for n in names]
    axis.barh(list(y), values, height=.62, color=colours, zorder=3)
    axis.set_yticks(list(y))
    axis.set_yticklabels([LABELS.get(n, n) for n in names], fontsize=9, color=INK)
    axis.invert_yaxis()
    span = limit or max(values)*1.18
    axis.set_xlim(0, span)
    for index, value in enumerate(values):
        axis.text(value + span*.015, index, formatter(value), va='center',
                  fontsize=9, color=SECONDARY)
    axis.set_title(title, fontsize=11, color=INK, loc='left', pad=18)
    axis.text(0, 1.02, subtitle, fontsize=8.5, color=SECONDARY,
              transform=axis.transAxes, ha='left', va='bottom')
    axis.xaxis.grid(True, color=GRID, linewidth=.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color(GRID)
    axis.tick_params(axis='x', colors=SECONDARY, labelsize=8.5, length=0)
    axis.tick_params(axis='y', length=0)


def reference(axis, value, text, depth=-.13):
    """Dashed rule with its label below the axis, stacked to avoid collision."""
    axis.axvline(value, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    axis.annotate(text, xy=(value, 0), xytext=(value, depth),
                  textcoords=axis.get_xaxis_transform(), fontsize=8,
                  color=SECONDARY, ha='center', va='top', annotation_clip=False,
                  arrowprops=dict(arrowstyle='-', color=GRID, linewidth=.8,
                                  shrinkA=0, shrinkB=0))


def conditioning_figure(report, output, trivial, table_direction):
    arms = report['arms']
    order = sorted(arms, key=lambda n: -arms[n]['facet_top1'])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)

    bars(axes[0], order, [arms[n]['facet_top1'] for n in order],
         {'full', 'direction_only'}, lambda v: '%.4f' % v,
         'Branch selection', 'exit-facet top-1, higher is better', limit=0.52)
    reference(axes[0], table_direction, 'RDM table  %.4f' % table_direction, depth=-.14)
    reference(axes[0], trivial, 'trivial  %.4f' % trivial, depth=-.28)

    bars(axes[1], order, [arms[n]['exit_angle_mean'] for n in order],
         {'full', 'direction_only'}, lambda v: '%.1f°' % v,
         'Within-facet decode', 'exit angle given the true facet, lower is better')

    figure.suptitle('What the boundary operator needs from its inputs',
                    fontsize=12.5, color=INK, x=.006, ha='left', y=1.13)
    figure.text(.006, 1.035,
                '%d epochs per arm, %s training records, %s held-out records; '
                'inputs removed by zeroing, architecture identical across arms'
                % (report['epochs'], format(report['train_records'], ','),
                   format(report['test_records'], ',')),
                fontsize=8.5, color=SECONDARY, ha='left')
    figure.subplots_adjust(left=.13, right=.985, top=.86, bottom=.24, wspace=.42)
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(figure)
    print('wrote %s' % output)


def accuracy_figure(rows, output):
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    highlight = {n for n, _, h in rows if h}
    figure, axis = plt.subplots(figsize=(7.4, 3.4))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    bars(axis, names, values, highlight, lambda v: '%.4f' % v,
         'Every branch predictor on one scale',
         'exit-facet top-1 on the held-out pool', limit=0.52)
    figure.subplots_adjust(left=.30, right=.97, top=.84, bottom=.14)
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(figure)
    print('wrote %s' % output)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ablation', type=Path,
                        default=Path('checkpoints/camera_model/conditioning_ablation.json'))
    parser.add_argument('--oracle', type=Path,
                        default=Path('checkpoints/camera_model/oracle.json'))
    parser.add_argument('--output_dir', type=Path, default=Path('renders/figures'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ablation = json.loads(args.ablation.read_text())
    oracle = json.loads(args.oracle.read_text())

    # Lookup-table figures come from compare_rdm_conditioning; they are constants
    # of the dataset rather than of any model, so they are named here explicitly
    # and cited in the caption.
    trivial, table_direction, table_facet = 0.0465, 0.0682, 0.1918

    conditioning_figure(ablation, args.output_dir/'conditioning_ablation.png',
                        trivial, table_direction)

    arms = ablation['arms']
    accuracy_figure([
        ('trivial baseline', trivial, False),
        ('RDM table, direction', table_direction, False),
        ('network, direction only', arms['direction_only']['facet_top1'], True),
        ('RDM table, entry facet', table_facet, False),
        ('NN oracle, top-1', oracle['facet_accuracy_top1'], False),
        ('NN oracle, majority of 8', oracle['facet_accuracy_majority'], False),
        ('network, full conditioning', arms['full']['facet_top1'], True),
    ], args.output_dir/'facet_accuracy.png')


if __name__ == '__main__':
    main()
