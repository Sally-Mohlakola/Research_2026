import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json
import math

import numpy as np

from utils.run_record import now, write_csv, write_json


# the numbers to compare across seeds, gathered from one run's record as (name, value) pairs
def run_values(record):
    values = {}
    for section in ('whole_stone', 'interior_only'):
        block = (record.get('headline') or {}).get(section)
        if block:
            for key in ('pixel_correlation', 'disattenuated_correlation', 'pixel_rank_correlation',
                        'pixel_log_correlation', 'facet_correlation', 'relative_energy_error',
                        'noise_free_contrast_neural'):
                values['%s.%s' % (section, key)] = block.get(key)
        quality = (record.get('quality_vs_time') or {}).get(section)
        if quality:
            values['%s.neural_error_floor' % section] = quality['neural_error_floor']
            values['%s.equal_time_error_ratio' % section] = quality['equal_time_error_ratio_neural_over_analytic']
    for key in ('facet_accuracy', 'exit_angle_nearest_deg', 'exit_angle_matched_deg', 'facet_hit_rate',
                'distinct_facets_per_16', 'within_branch_spread_deg'):
        if record.get('test') and record['test'].get(key) is not None:
            values['test.' + key] = record['test'][key]
    for row in (record.get('breakdown') or {}).get('by_path_length', []):
        values['operator.accuracy_%s_facet_hits' % row['facet_hits']] = row['facet_accuracy']
        values['operator.exit_error_deg_%s_facet_hits' % row['facet_hits']] = row['exit_angle_median_deg']
    convergence = record.get('convergence') or {}
    if convergence:
        values['training.epochs_run'] = convergence['epochs_run']
        values['training.best_epoch'] = convergence['best_epoch']
    timing = (record.get('render_timing') or {}).get('neural')
    if timing:
        values['timing.neural_us_per_decision'] = timing['microseconds_per_interior_decision']
        values['timing.neural_mean_render_seconds'] = timing['mean_render_seconds']
    return values


# mean, sample standard deviation and range of one metric across the runs that have it
def spread(name, values):
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if not len(v):
        return dict(metric=name, n=0)
    return dict(metric=name, n=int(len(v)), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                min=float(v.min()), max=float(v.max()), values=json.dumps([round(float(x), 6) for x in v]))


# per-seed paired differences against partner runs, from each run's comparisons folder
def paired_differences(runs, partners):
    by_metric = {}
    for run, partner in zip(runs, partners):
        path = REPO/'figures'/run/'comparisons'/('%s_vs_%s.json' % (run, partner))
        flipped = False
        if not path.exists():
            path = REPO/'figures'/partner/'comparisons'/('%s_vs_%s.json' % (partner, run))
            flipped = True
        if not path.exists():
            raise FileNotFoundError('No paired comparison between %s and %s' % (run, partner))
        result = json.loads(path.read_text())
        for section, block in result['sections'].items():
            if not block:
                continue
            for row in block['rows']:
                sign = -1 if flipped else 1
                entry = by_metric.setdefault('%s.%s' % (section, row['metric']), dict(differences=[], clear=0))
                entry['differences'].append(sign*row['difference'])
                entry['clear'] += int(row['interval_excludes_zero'])
    rows = []
    for name, entry in by_metric.items():
        row = spread(name, entry['differences'])
        row['seeds_with_clear_difference'] = entry['clear']
        rows.append(row)
    return rows


# summarise repeated runs (different training seeds) into one folder under figures/
def main():
    parser = argparse.ArgumentParser(description='Mean and spread of repeated runs, e.g. three training seeds.')
    parser.add_argument('--name', required=True, help='Folder name under figures/ for the summary')
    parser.add_argument('--runs', nargs='+', required=True, help='Run names under figures/')
    parser.add_argument('--against', nargs='+',
                        help='Partner run for each run, in the same order, for paired differences (run minus partner)')
    args = parser.parse_args()
    if args.against and len(args.against) != len(args.runs):
        parser.error('--against needs one partner per run')

    records = [json.loads((REPO/'figures'/r/'run.json').read_text()) for r in args.runs]
    heads = {r['model']['head'] for r in records}
    stones = {r['mesh']['diamond_name'] for r in records}
    if len(heads) != 1 or len(stones) != 1:
        raise ValueError('Repeats should share one head and one stone, found %s on %s' % (heads, stones))
    per_run = [run_values(r) for r in records]
    names = list(dict.fromkeys(k for values in per_run for k in values))
    summary = [spread(name, [values.get(name) for values in per_run]) for name in names]

    folder = REPO/'figures'/args.name
    write_csv(folder/'summary.csv', summary)
    result = dict(created=now(), runs=args.runs, head=heads.pop(), stone=stones.pop(),
                  seeds=[r['model']['seed'] for r in records], summary=summary,
                  note='sd is the sample standard deviation across runs (ddof = 1); with three seeds it is a rough '
                       'guide to run-to-run variation, not a precise estimate')
    lines = ['# %s' % args.name, '',
             '%s on %s, %d runs (training seeds %s): %s.' % (result['head'], result['stone'], len(args.runs),
                                                            ', '.join(str(s) for s in result['seeds']),
                                                            ', '.join(args.runs)),
             '', '| metric | n | mean | sd | min | max |', '|---|---|---|---|---|---|']
    fmt = lambda v: 'n/a' if v is None else '%.4f' % v
    for row in summary:
        if row['n']:
            lines.append('| %s | %d | %s | %s | %s | %s |' % (row['metric'], row['n'], fmt(row['mean']),
                                                              fmt(row['sd']), fmt(row['min']), fmt(row['max'])))
    if args.against:
        differences = paired_differences(args.runs, args.against)
        write_csv(folder/'paired_differences.csv', differences)
        result['paired_differences'] = dict(against=args.against, rows=differences)
        lines += ['', '## Paired differences (this group minus its partner, seed by seed)', '',
                  '| metric | seeds | mean difference | sd | seeds where the 95% interval excludes zero |',
                  '|---|---|---|---|---|']
        for row in differences:
            if not row['n']:
                lines.append('| %s | 0 | n/a | n/a | undefined in every seed |' % row['metric'])
                continue
            lines.append('| %s | %d | %s | %s | %d of %d |' % (row['metric'], row['n'], fmt(row['mean']),
                                                               fmt(row['sd']), row['seeds_with_clear_difference'],
                                                               row['n']))
    write_json(folder/'summary.json', result)
    (folder/'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
