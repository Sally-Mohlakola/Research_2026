import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from neural.learned_interior_data import load_learned_interior
from neural.learned_interior_model import encode_data, load_model
from utils.run_record import geometric_facets, now, write_csv, write_json

# facets a path hits inside the stone, the exit included (the pool's depth counts the entry as 1)
HIT_BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 127))
REGION_COLOURS = dict(table='#e9b44c', crown='#3d5a80', pavilion='#d1495b', girdle='#8a98a8')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25})


# the operator's answers for every escaped test path: its top facet, and its exit decoded on the true facet
@torch.no_grad()
def predictions(model, x, true_facet, batch=65536):
    top, direction = [], []
    generator = torch.Generator().manual_seed(0)
    for start in range(0, len(x), batch):
        xb, fb = x[start:start+batch], true_facet[start:start+batch]
        top.append(model.facet(model.encoder(xb)).argmax(-1))
        direction.append(model.sample(xb, generator, deterministic=True, facet=fb)['exit_direction'])
    return torch.cat(top).numpy(), torch.cat(direction).numpy()


# which part of the stone each triangle belongs to, from its outward normal
def regions(vertices, faces):
    corners = np.asarray(vertices, float)[np.asarray(faces)]
    normal = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    up = normal[:, 2]/np.linalg.norm(normal, axis=1)
    return np.where(up > .999, 'table', np.where(up > .05, 'crown', np.where(up > -.05, 'girdle', 'pavilion')))


# held-out accuracy and exit-angle error for paths grouped by how many facets they hit inside
def by_path_length(hits, correct, angle):
    rows = []
    for low, high in HIT_BINS:
        pick = (hits >= low) & (hits <= high)
        if not pick.any():
            continue
        label = str(low) if low == high else ('%d+' % low if high >= 127 else '%d-%d' % (low, high))
        rows.append(dict(facet_hits=label, paths=int(pick.sum()), share=float(pick.mean()),
                         facet_accuracy=float(correct[pick].mean()),
                         exit_angle_median_deg=float(np.median(angle[pick])),
                         exit_angle_p90_deg=float(np.percentile(angle[pick], 90))))
    return rows


# per exit triangle: how often it is predicted correctly, its exit error, and how much training data it had
def by_facet(exit_facet, correct, angle, train, faces, region, geometric):
    train_exits = np.bincount(train['exit_facet'][train['escaped']], minlength=len(faces))
    first = np.unique(train['entry_id'], return_index=True)[1]
    train_entries = np.bincount(train['entry_facet'][first], minlength=len(faces))
    rows = []
    for k in range(len(faces)):
        pick = exit_facet == k
        rows.append(dict(triangle=k, geometric_facet=int(geometric[k]), region=str(region[k]),
                         test_exits=int(pick.sum()),
                         recall=float(correct[pick].mean()) if pick.any() else None,
                         exit_angle_median_deg=float(np.median(angle[pick])) if pick.any() else None,
                         train_exits=int(train_exits[k]), train_entries=int(train_entries[k])))
    return rows


# Spearman correlation between two lists, for "does more data mean better accuracy"
def rank_correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


# two panels by path length: how often the exit facet is right, and how far the exit lands given the right facet
def plot_path_length(rows, output, title, overall=None):
    x = np.arange(len(rows))
    labels = ['%s\n%s' % (r['facet_hits'], '%.0f%%' % (100*r['share']) if r['share'] >= .01 else '%d paths' % r['paths'])
              for r in rows]
    sparse = np.array([r['paths'] < 1000 for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    accuracy = [100*r['facet_accuracy'] for r in rows]
    ax.bar(x, accuracy, .6, color=['#c9ced6' if s else '#3d5a80' for s in sparse])
    for i, a in enumerate(accuracy):
        ax.text(i, a + 1.5, '%.0f%%' % a, ha='center', fontsize=8)
    if overall is not None:
        ax.axhline(100*overall, color='#444', linewidth=1, linestyle='--')
        ax.text(len(rows) - .5, 100*overall + 1.5, 'all paths %.0f%%' % (100*overall), ha='right', fontsize=7.5,
                color='#444')
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel='exit facet predicted correctly (%)',
           xlabel='facets the light meets inside the stone, the exit included\n(below each: share of held-out paths)',
           title='Does the operator pick the right exit facet?')

    ax = axes[1]
    median = [r['exit_angle_median_deg'] for r in rows]
    p90 = [r['exit_angle_p90_deg'] for r in rows]
    ax.vlines(x, median, p90, color='#d1495b', linewidth=2, alpha=.5)
    ax.scatter(x, median, s=40, color='#d1495b', zorder=3, label='median')
    ax.scatter(x[sparse], np.array(median)[sparse], s=40, color='#c9ced6', zorder=4)
    ax.scatter(x, p90, s=40, marker='_', color='#d1495b', linewidth=2, zorder=3, label='90th percentile')
    for i, m in enumerate(median):
        ax.text(i + .12, m, '%.0f°' % m, va='center', fontsize=8)
    ax.set(xticks=x, xticklabels=labels, xlim=(-.5, len(rows) - .5), ylim=(0, 90),
           ylabel='angle from the recorded exit (degrees)',
           xlabel='facets the light meets inside the stone, the exit included\n(below each: share of held-out paths)',
           title='Given the right facet, how far off is the exit direction?')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# a tick written with a power of ten, e.g. 2x10^4, and plain 0 at the origin
def power_of_ten(value, _):
    if value <= 0:
        return '0'
    exponent = int(np.floor(np.log10(value) + 1e-9))
    digit = value / 10**exponent
    digit = ('%d' % round(digit)) if abs(digit - round(digit)) < 1e-6 else ('%g' % digit)
    return '$10^{%d}$' % exponent if digit == '1' else r'$%s{\times}10^{%d}$' % (digit, exponent)


# two panels by exit triangle: accuracy per part of the stone, and accuracy against the training data it had
def plot_facets(rows, output, title, minimum=20):
    shown = [r for r in rows if r['recall'] is not None and r['test_exits'] >= minimum]
    regions_present = [g for g in REGION_COLOURS if any(r['region'] == g for r in shown)]
    exits = sum(r['test_exits'] for r in rows) or 1
    drawn = {g: sum(r['region'] == g for r in shown) for g in regions_present}
    total = {g: sum(r['region'] == g for r in rows) for g in regions_present}
    count = {g: '%d' % drawn[g] if drawn[g] == total[g] else '%d of %d' % (drawn[g], total[g])
             for g in regions_present}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    jitter = np.random.default_rng(0)
    for i, g in enumerate(regions_present):
        mine = [r for r in shown if r['region'] == g]
        weight = sum(r['test_exits'] for r in mine)
        pooled = sum(r['recall']*r['test_exits'] for r in mine)/weight
        ax.bar(i, 100*pooled, .6, color=REGION_COLOURS[g], alpha=.3)
        ax.scatter(i + jitter.uniform(-.2, .2, len(mine)), [100*r['recall'] for r in mine], s=16,
                   color=REGION_COLOURS[g], edgecolor='white', linewidth=.5, zorder=3, clip_on=False)
        ax.text(i + .32, 100*pooled, ' %.0f%%' % (100*pooled), va='center', fontsize=8)
    ax.set(xticks=range(len(regions_present)), ylim=(0, 100), xlim=(-.5, len(regions_present) - .2),
           xticklabels=['%s\n%s triangles\n%.0f%% of exits'
                        % (g, count[g], 100*sum(r['test_exits'] for r in rows if r['region'] == g)/exits)
                        for g in regions_present],
           ylabel='exits that were predicted correctly (%)',
           title='Which exit facets does the operator get right?')
    key = [Line2D([], [], marker='o', linestyle='', color='#777', markeredgecolor='white', markersize=5,
                  label='one triangle'),
           Patch(color='#777', alpha=.3, label='all exits through that part')]
    ax.legend(handles=key, frameon=False, fontsize=7.5, loc='upper right')

    ax = axes[1]
    for g in regions_present:
        mine = [r for r in shown if r['region'] == g and r['train_exits'] > 0]
        ax.scatter([r['train_exits'] for r in mine], [100*r['recall'] for r in mine], s=18,
                   color=REGION_COLOURS[g], label='%s (%d triangles)' % (g, len(mine)), edgecolor='white',
                   linewidth=.5, zorder=3, clip_on=False)
    ax.set(xlim=(0, None), ylim=(0, 100), xlabel='how many training paths left through the triangle',
           ylabel='exits that were predicted correctly (%)',
           title='Does more training data mean better accuracy?')
    most = max(r['train_exits'] for r in shown)
    crowded = [r for r in shown if r['train_exits'] < .02*most]
    if len(crowded) > 1:
        x = np.mean([r['train_exits'] for r in crowded])
        y = np.mean([100*r['recall'] for r in crowded])
        ax.annotate('%d triangles overlap here\n(%d-%d training paths each)'
                    % (len(crowded), min(r['train_exits'] for r in crowded), max(r['train_exits'] for r in crowded)),
                    (x, y), xytext=(.02*most, 45), fontsize=7.5, color='#444',
                    arrowprops=dict(arrowstyle='-', color='#888', linewidth=.8))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8, steps=(1, 2, 5, 10)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(power_of_ten))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', length=3)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.suptitle(title, fontsize=10)
    fig.text(.5, .01, 'one dot per triangle with at least %d held-out exits (%d of %d)' % (minimum, len(shown), len(rows)),
             ha='center', fontsize=7.5, color='#444')
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)


# a thin box around a table drawn under an axis: a heading column, then one column centred on each bar
def outline_table(fig, ax, cells, pad=3, colour='#bbb'):
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    box = lambda c: to_figure.transform(c.get_window_extent().get_points())
    px, py = pad/72/fig.get_figwidth(), pad/72/fig.get_figheight()
    at = lambda x: to_figure.transform(ax.transData.transform((x, 0)))[0]
    columns = len(cells[0]) - 1
    left = min(box(row[0])[0][0] for row in cells) - px
    edges = [at(i - .5) for i in range(columns + 1)]
    tops = [max(box(c)[1][1] for c in row) + py for row in cells]
    bottom = min(box(c)[0][1] for c in cells[-1]) - py
    line = dict(color=colour, linewidth=.6, transform=fig.transFigure, figure=fig)
    fig.add_artist(matplotlib.patches.Rectangle((left, bottom), edges[-1] - left, tops[0] - bottom, fill=False,
                                                edgecolor=colour, linewidth=.6, transform=fig.transFigure))
    for top in tops[1:]:
        fig.add_artist(Line2D([left, edges[-1]], [top, top], **line))
    for x in edges[:-1]:
        fig.add_artist(Line2D([x, x], [bottom, tops[0]], **line))


# the left panel alone, sized for one IEEE column (3.5 inches) and saved as png and pdf
def plot_facets_column(rows, output, minimum=20):
    shown = [r for r in rows if r['recall'] is not None and r['test_exits'] >= minimum]
    regions_present = [g for g in REGION_COLOURS if any(r['region'] == g for r in shown)]
    exits = sum(r['test_exits'] for r in rows) or 1
    with plt.rc_context({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        jitter = np.random.default_rng(0)
        labels = []
        for i, g in enumerate(regions_present):
            mine = [r for r in shown if r['region'] == g]
            pooled = sum(r['recall']*r['test_exits'] for r in mine)/sum(r['test_exits'] for r in mine)
            ax.bar(i, 100*pooled, .6, color=REGION_COLOURS[g], alpha=.3)
            ax.scatter(i + jitter.uniform(-.2, .2, len(mine)), [100*r['recall'] for r in mine], s=9,
                       color=REGION_COLOURS[g], edgecolor='white', linewidth=.4, zorder=3, clip_on=False)
            ax.text(i + .32, 100*pooled, ' %.0f%%' % (100*pooled), va='center', fontsize=7)
            total = sum(r['region'] == g for r in rows)
            count = '%d' % len(mine) if len(mine) == total else '%d of %d' % (len(mine), total)
            labels.append((g, count, '%.0f%%' % (100*sum(r['test_exits'] for r in rows if r['region'] == g)/exits)))
        ax.set(xticks=range(len(regions_present)), xticklabels=[name for name, _, _ in labels], ylim=(0, 100),
               xlim=(-.5, len(regions_present) - .2), ylabel='exits predicted correctly (%)')
        key = [Line2D([], [], marker='o', linestyle='', color='#777', markeredgecolor='white', markersize=4,
                      label='one triangle'),
               Patch(color='#777', alpha=.3, label='all exits through that part')]
        ax.legend(handles=key, frameon=False, loc='upper right', handlelength=1.2)
        ax.tick_params(axis='x', labelsize=8, length=0, pad=4)
        under = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        cells = []
        for row, (heading, column) in enumerate((('triangles', 1), ('share of exits', 2))):
            y = -.15 - .095*row
            cells.append([ax.text(-.6, y, heading, transform=under, ha='right', va='top', fontsize=6.5,
                                  color='#666')]
                         + [ax.text(i, y, label[column], transform=under, ha='center', va='top', fontsize=7,
                                    color='#333') for i, label in enumerate(labels)])
        fig.tight_layout(pad=.3)
        fig.subplots_adjust(bottom=.26, left=max(fig.subplotpars.left, .25))
        outline_table(fig, ax, cells)
        fig.savefig(output.with_suffix('.png'), dpi=300)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)


# operator errors by path length and by exit facet on a held-out pool, with training coverage per facet
def main():
    parser = argparse.ArgumentParser(description='Held-out operator errors by path length and by facet.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--test_pool', type=Path, required=True, help='Independently gathered pool')
    parser.add_argument('--train_pool', type=Path, required=True, help='The pool the model was trained on')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--title', default='')
    args = parser.parse_args()
    torch.set_num_threads(2)

    model, checkpoint = load_model(args.model)
    data, metadata = load_learned_interior(args.test_pool)
    if metadata['geometry_sha256'] != checkpoint['metadata']['geometry_sha256']:
        raise ValueError('The test pool is for a different stone')
    with np.load(args.test_pool, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, facet, _, valid, escaped, _ = encode_data(data, vertices, faces)
    keep = valid & escaped
    x, facet = torch.from_numpy(x[keep]), torch.from_numpy(facet[keep])
    top, direction = predictions(model, x, facet)
    truth = data['exit_direction'][keep]
    angle = np.degrees(np.arccos(np.clip((direction*truth).sum(1)/np.linalg.norm(truth, axis=1), -1, 1)))
    correct = top == facet.numpy()
    hits = data['depth'][keep] - 1

    train_data, _ = load_learned_interior(args.train_pool)
    train = dict(exit_facet=np.where(train_data['status'] == 0, train_data['exit_facet'], 0),
                 escaped=train_data['status'] == 0, entry_facet=train_data['entry_facet'],
                 entry_id=train_data['entry_id'])

    path_rows = by_path_length(hits, correct, angle)
    facet_rows = by_facet(facet.numpy(), correct, angle, train, faces, regions(vertices, faces),
                          geometric_facets(vertices, faces))
    seen = [r for r in facet_rows if r['recall'] is not None and r['test_exits'] >= 20]
    summary = dict(created=now(), model=str(args.model), test_pool=str(args.test_pool),
                   train_pool=str(args.train_pool), held_out_paths=int(keep.sum()),
                   facet_accuracy=float(correct.mean()), exit_angle_median_deg=float(np.median(angle)),
                   accuracy_vs_training_exits_rank_correlation=rank_correlation(
                       [r['train_exits'] for r in seen], [r['recall'] for r in seen]),
                   triangles_with_20_or_more_test_exits=len(seen),
                   method='Facet accuracy: the top choice of the facet head equals the recorded exit triangle. Exit '
                          'angle: the operator decodes its exit on the recorded triangle (deterministic decode), '
                          'compared with the recorded exit direction. Escaped, valid held-out paths only.',
                   by_path_length=path_rows, by_facet=facet_rows)

    out = args.output_dir
    write_json(out/'breakdown.json', summary)
    write_csv(out/'by_path_length.csv', path_rows)
    write_csv(out/'by_facet.csv', facet_rows)
    plot_path_length(path_rows, out/'by_path_length.png', args.title, summary['facet_accuracy'])
    plot_facets(facet_rows, out/'by_facet.png', args.title)
    plot_facets_column(facet_rows, out/'by_facet_column')

    print('%d held-out paths: facet accuracy %.3f, median exit error %.1f deg'
          % (summary['held_out_paths'], summary['facet_accuracy'], summary['exit_angle_median_deg']))
    for r in path_rows:
        print('  %-7s %5.1f%% of paths  accuracy %.3f  exit error %.1f deg (p90 %.1f)'
              % (r['facet_hits'], 100*r['share'], r['facet_accuracy'], r['exit_angle_median_deg'],
                 r['exit_angle_p90_deg']))
    print('  accuracy against training exits per triangle (rank correlation): %s'
          % summary['accuracy_vs_training_exits_rank_correlation'])


if __name__ == '__main__':
    main()
