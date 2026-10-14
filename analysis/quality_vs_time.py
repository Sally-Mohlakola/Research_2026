import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import csv
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.evaluate_render import LUMINANCE, load_renders, stone_mask
from neural.learned_interior_model import load_model
from utils.run_record import now, write_csv, write_json

ANALYTIC, NEURAL = '#3d5a80', '#d1495b'


# mean seconds per seed render for one mode, from the run's timing table
def seconds_per_render(folder, mode):
    with open(folder/'render'/'timing.csv', newline='', encoding='utf-8') as handle:
        values = [float(r['render_seconds']) for r in csv.DictReader(handle) if r['mode'] == mode]
    return float(np.mean(values))


# the error model, per pixel over the stone, against the converged analytic image:
# analytic error is pure noise, variance / seeds; neural error is a fixed bias plus its own variance / seeds
def error_model(analytic, neural, mask):
    ya = np.stack([image[mask] @ LUMINANCE for image in analytic])
    yn = np.stack([image[mask] @ LUMINANCE for image in neural])
    half = len(ya)//2
    a1, a2 = ya[:half].mean(0), ya[half:].mean(0)
    n1, n2 = yn[:half].mean(0), yn[half:].mean(0)
    return dict(reference_mean=float(ya.mean()),
                analytic_variance_per_seed=float(ya.var(0, ddof=1).mean()),
                neural_variance_per_seed=float(yn.var(0, ddof=1).mean()),
                neural_bias_squared=max(float(((n1 - a1)*(n2 - a2)).mean()), 0.),
                seeds_measured=int(len(ya)))


# relative RMSE against the converged reference after k seeds, for both modes
def curves(model, seconds, spp, max_seeds):
    mu = model['reference_mean']
    rows = []
    for k in np.unique(np.round(np.geomspace(1, max_seeds, 60)).astype(int)):
        for mode in ('analytic', 'neural'):
            if mode == 'analytic':
                mse = model['analytic_variance_per_seed']/k
            else:
                mse = model['neural_bias_squared'] + model['neural_variance_per_seed']/k
            rows.append(dict(mode=mode, seeds=int(k), spp=int(k*spp), seconds=k*seconds[mode],
                             relative_rmse=math.sqrt(mse)/mu, measured=bool(k <= model['seeds_measured'])))
    return rows


# the numbers a reader wants from the curves: the neural floor, and the equal-time comparison
def summary(model, seconds):
    mu, n = model['reference_mean'], model['seeds_measured']
    neural_time = n*seconds['neural']
    neural_error = math.sqrt(model['neural_bias_squared'] + model['neural_variance_per_seed']/n)/mu
    analytic_seeds_same_time = neural_time/seconds['analytic']
    analytic_error_same_time = math.sqrt(model['analytic_variance_per_seed']/analytic_seeds_same_time)/mu
    floor = math.sqrt(model['neural_bias_squared'])/mu
    seeds_to_floor = (model['analytic_variance_per_seed']/model['neural_bias_squared']
                      if model['neural_bias_squared'] > 0 else None)
    return dict(neural_error_floor=floor,
                neural_error_at_its_time=neural_error,
                neural_seconds=neural_time,
                analytic_error_at_equal_time=analytic_error_same_time,
                equal_time_error_ratio_neural_over_analytic=neural_error/analytic_error_same_time,
                analytic_seconds_to_reach_neural_floor=(seeds_to_floor*seconds['analytic']
                                                        if seeds_to_floor is not None else None),
                analytic_seconds_per_render=seconds['analytic'],
                analytic_below_neural_floor_after_one_render=(seeds_to_floor is not None and seeds_to_floor <= 1))


# the plot: relative error against render time, measured range solid, extrapolation dashed
def plot(sections, output, title):
    fig, axes = plt.subplots(1, len(sections), figsize=(4.6*len(sections), 3.6), squeeze=False)
    for ax, (section, block) in zip(axes[0], sections.items()):
        for mode, colour in (('analytic', ANALYTIC), ('neural', NEURAL)):
            mine = [r for r in block['curve'] if r['mode'] == mode]
            measured = [r for r in mine if r['measured']]
            ax.plot([r['seconds'] for r in mine], [r['relative_rmse'] for r in mine], '--', color=colour, lw=1)
            ax.plot([r['seconds'] for r in measured], [r['relative_rmse'] for r in measured], 'o-',
                    color=colour, ms=3, label=mode)
        # the floor sits exactly on the neural curve's tail; draw it 2% higher so both stay visible
        ax.axhline(block['summary']['neural_error_floor']*1.02, color=NEURAL, ls=':', lw=1.2,
                   label='neural floor (bias), drawn 2% high to stay visible')
        ax.set(xscale='log', yscale='log', xlabel='render time (s)',
               ylabel='relative RMSE (vs converged reference)',
               title=section.replace('_', ' '))
        ax.grid(True, which='both', alpha=.2)
    fig.suptitle(title + '  (solid: within the rendered seeds, dashed: extrapolated by the error model)', fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)


# quality against render time for one recorded run, written into figures/<name>/render/
def main():
    parser = argparse.ArgumentParser(description='Quality against render time for one recorded run.')
    parser.add_argument('--run', required=True, help='Run name under figures/')
    parser.add_argument('--extrapolate', type=float, default=64.,
                        help='Draw the error model out to this many times the measured seeds')
    args = parser.parse_args()
    folder = REPO/'figures'/args.run
    record = json.loads((folder/'run.json').read_text())
    inputs = record['evaluation_inputs']

    analytic, reports, analytic_parts = load_renders([Path(p) for p in inputs['analytic']])
    neural, _, neural_parts = load_renders([Path(p) for p in inputs['neural']])
    _, checkpoint = load_model(record['model']['path'])
    first = reports[0]
    mask = stone_mask(checkpoint, first['width'], first['height'], first['azimuth'],
                      first.get('rotation_deg'), first.get('camera', 'studio'))
    analytic_source = REPO/'figures'/record['reference_run'] if record.get('reference_run') else folder
    seconds = dict(analytic=seconds_per_render(analytic_source, 'analytic'),
                   neural=seconds_per_render(folder, 'neural'))

    sections = {}
    pairs = {'whole_stone': (analytic, neural),
             'interior_only': (analytic_parts.get('operator'), neural_parts.get('operator'))}
    for section, (a, n) in pairs.items():
        if a is None or n is None:
            continue
        model = error_model(a, n, mask)
        sections[section] = dict(error_model=model,
                                 curve=curves(model, seconds, first['spp'], len(a)*args.extrapolate),
                                 summary=summary(model, seconds))

    result = dict(created=now(), run=args.run, seconds_per_render=seconds, spp_per_render=first['spp'],
                  method='Expected relative RMSE against the converged analytic image. Analytic: per-pixel '
                         'variance across seeds divided by the seed count. Neural: the noise-free squared '
                         'difference from the analytic image (independent halves, debiased) plus its own '
                         'variance divided by the seed count. Seeds are independent; times are mean seconds '
                         'per seed render.',
                  sections=sections)
    write_json(folder/'render'/'quality_vs_time.json', result)
    write_csv(folder/'render'/'quality_vs_time.csv',
              [dict(section=s, **row) for s, block in sections.items() for row in block['curve']])
    plot(sections, folder/'render'/'quality_vs_time.png', args.run)

    for section, block in sections.items():
        s = block['summary']
        print('%s: neural floor %.3f, at equal time neural %.3f vs analytic %.3f (ratio %.2f); '
              'analytic reaches the neural floor after %s s'
              % (section, s['neural_error_floor'], s['neural_error_at_its_time'],
                 s['analytic_error_at_equal_time'], s['equal_time_error_ratio_neural_over_analytic'],
                 'n/a' if s['analytic_seconds_to_reach_neural_floor'] is None
                 else '%.0f' % s['analytic_seconds_to_reach_neural_floor']))


if __name__ == '__main__':
    main()
