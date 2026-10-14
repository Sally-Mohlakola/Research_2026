import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json
import math

import numpy as np

from analysis.evaluate_render import LUMINANCE, facet_means, load_renders, stone_hits
from neural.learned_interior_model import load_model
from utils.run_record import geometric_facets, now, write_csv, write_json

METRICS = ('pixel_correlation', 'disattenuated_correlation', 'facet_correlation', 'relative_energy_error',
           'noise_free_contrast_neural')


# one run's evaluation inputs: which analytic and neural seed renders it was scored on
def run_inputs(name):
    record = json.loads((REPO/'figures'/name/'run.json').read_text())
    inputs = record.get('evaluation_inputs')
    if not inputs:
        raise ValueError('figures/%s/run.json has no evaluation_inputs; rerun it with run_experiment.py' % name)
    return record, [Path(p) for p in inputs['analytic']], [Path(p) for p in inputs['neural']]


# both runs must be scored against the very same analytic pixels, or the comparison is not paired
def check_same_reference(first, second):
    if [p.resolve() for p in first] == [p.resolve() for p in second]:
        return 'same files'
    a, _, _ = load_renders(first)
    b, _, _ = load_renders(second)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError('The two runs were scored against different analytic renders; they cannot be paired')
    return 'identical pixels'


# per-block sums for every pair of signals the metrics need, so a bootstrap only re-adds blocks
def block_sums(signals, blocks):
    names = list(signals)
    sums = {'n': np.bincount(blocks)}
    for i, a in enumerate(names):
        sums[a] = np.bincount(blocks, weights=signals[a])
        for b in names[i:]:
            sums[a + '*' + b] = np.bincount(blocks, weights=signals[a]*signals[b])
    return sums


# correlation, covariance and means from summed block statistics
class Stats:
    def __init__(self, sums, pick):
        self.s = {k: v[pick].sum() for k, v in sums.items()}
        self.n = self.s['n']

    def mean(self, a):
        return self.s[a]/self.n

    def cov(self, a, b):
        key = a + '*' + b if a + '*' + b in self.s else b + '*' + a
        return self.s[key]/self.n - self.mean(a)*self.mean(b)

    def corr(self, a, b):
        va, vb = self.cov(a, a), self.cov(b, b)
        return self.cov(a, b)/math.sqrt(va*vb) if va > 0 and vb > 0 else float('nan')


# the pixel metrics of one run, from its halves a1 a2 (shared analytic) and n1 n2 (this run's neural)
def pixel_metrics(stats, n1, n2):
    cross = .5*(stats.corr('a1', n2) + stats.corr('a2', n1))
    reliability = stats.corr('a1', 'a2')*stats.corr(n1, n2)
    mean_reference = .5*(stats.mean('a1') + stats.mean('a2'))
    mean_other = .5*(stats.mean(n1) + stats.mean(n2))
    return dict(pixel_correlation=stats.corr('a1', n1),
                disattenuated_correlation=cross/math.sqrt(reliability) if reliability > 0 else float('nan'),
                relative_energy_error=(mean_other - mean_reference)/mean_reference,
                noise_free_contrast_neural=math.sqrt(max(stats.cov(n1, n2), 0.))/mean_other)


# facet correlation of one run from per-facet means
def facet_metric(analytic, neural, pick):
    a, n = analytic[pick], neural[pick]
    return float(np.corrcoef(a, n)[0, 1]) if a.std() > 0 and n.std() > 0 else float('nan')


# the five headline metrics for both runs and their differences, with a paired bootstrap
def paired(images_a, images_b, mask, triangle, facet_of_triangle, block=16, repeats=2000, seed=0, min_pixels=20):
    lum = {k: (v @ LUMINANCE) for k, v in images_a.items()}
    lum.update({k.replace('n', 'm'): (v @ LUMINANCE) for k, v in images_b.items() if k.startswith('n')})
    rows, columns = np.nonzero(mask)
    blocks = np.unique((rows//block)*(mask.shape[1]//block + 1) + columns//block, return_inverse=True)[1]
    sums = block_sums({k: v[mask] for k, v in lum.items()}, blocks)

    label = facet_of_triangle[triangle[triangle >= 0]]
    facets = np.flatnonzero(np.bincount(label, minlength=facet_of_triangle.max()+1) >= min_pixels)
    full = lambda first, second: .5*(facet_means(first, triangle, facet_of_triangle, facets)
                                    + facet_means(second, triangle, facet_of_triangle, facets))
    fa = full(images_a['a1'], images_a['a2'])
    fn, fm = full(images_a['n1'], images_a['n2']), full(images_b['n1'], images_b['n2'])

    def evaluate(block_pick, facet_pick):
        stats = Stats(sums, block_pick)
        first, second = pixel_metrics(stats, 'n1', 'n2'), pixel_metrics(stats, 'm1', 'm2')
        first['facet_correlation'] = facet_metric(fa, fn, facet_pick)
        second['facet_correlation'] = facet_metric(fa, fm, facet_pick)
        return first, second

    every_block, every_facet = np.arange(len(sums['n'])), np.arange(len(facets))
    point_a, point_b = evaluate(every_block, every_facet)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in METRICS}
    for _ in range(repeats):
        first, second = evaluate(rng.integers(len(every_block), size=len(every_block)),
                                 rng.integers(len(every_facet), size=len(every_facet)))
        for k in METRICS:
            draws[k].append(first[k] - second[k])

    result = []
    for k in METRICS:
        d = np.array(draws[k])
        d = d[np.isfinite(d)]
        difference = point_a[k] - point_b[k]
        result.append(dict(metric=k, run_a=point_a[k], run_b=point_b[k], difference=difference,
                           ci95_low=float(np.percentile(d, 2.5)), ci95_high=float(np.percentile(d, 97.5)),
                           share_same_sign=float(np.mean(np.sign(d) == np.sign(difference))) if difference else None,
                           interval_excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)))
    return result, dict(blocks=int(len(every_block)), facets=int(len(facets)), repeats=repeats, block_size=block)


# the four half images of one run: first and second half of the analytic and the neural seeds
def halves(stack):
    half = len(stack)//2
    return stack[:half].mean(0), stack[half:].mean(0)


# compare two recorded runs on the same reference and write the result into both run folders
def main():
    parser = argparse.ArgumentParser(description='Paired comparison of two recorded runs on the same analytic reference.')
    parser.add_argument('--runs', nargs=2, required=True, metavar=('A', 'B'), help='Run names under figures/')
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    name_a, name_b = args.runs

    record_a, analytic_a, neural_a = run_inputs(name_a)
    record_b, analytic_b, neural_b = run_inputs(name_b)
    pairing = check_same_reference(analytic_a, analytic_b)

    analytic, reports, analytic_parts = load_renders(analytic_a)
    neural_one, _, parts_one = load_renders(neural_a)
    neural_two, _, parts_two = load_renders(neural_b)
    if not (analytic.shape == neural_one.shape == neural_two.shape):
        raise ValueError('The runs differ in resolution or seed count')

    _, checkpoint = load_model(record_a['model']['path'])
    first = reports[0]
    mask, triangle = stone_hits(checkpoint, first['width'], first['height'], first['azimuth'],
                                first.get('rotation_deg'), first.get('camera', 'studio'))
    facet_of_triangle = geometric_facets(checkpoint['vertices'].numpy(), checkpoint['faces'].numpy())

    sections = {}
    for section, (a_stack, one, two) in {
            'whole_stone': (analytic, neural_one, neural_two),
            'interior_only': (analytic_parts.get('operator'), parts_one.get('operator'), parts_two.get('operator'))}.items():
        if a_stack is None or one is None or two is None:
            sections[section] = None
            continue
        a1, a2 = halves(a_stack)
        n1, n2 = halves(one)
        m1, m2 = halves(two)
        rows, design = paired(dict(a1=a1, a2=a2, n1=n1, n2=n2), dict(n1=m1, n2=m2),
                              mask, triangle, facet_of_triangle, repeats=args.repeats)
        sections[section] = dict(rows=rows, design=design)

    result = dict(created=now(), run_a=name_a, run_b=name_b, head_a=record_a['model']['head'],
                  head_b=record_b['model']['head'], pairing=pairing, sections=sections,
                  method='paired bootstrap: the same 16 x 16 pixel blocks and the same facets are resampled for '
                         'both runs in every repeat; the interval is the 2.5th to 97.5th percentile of the '
                         'difference (A minus B)')
    for name, other in ((name_a, name_b), (name_b, name_a)):
        out = REPO/'figures'/name/'comparisons'
        write_json(out/('%s_vs_%s.json' % (name_a, name_b)), result)
        flat = [dict(section=s, **row) for s, block in sections.items() if block for row in block['rows']]
        write_csv(out/('%s_vs_%s.csv' % (name_a, name_b)), flat)

    print('%s (A) against %s (B), %s' % (name_a, name_b, pairing))
    for section, block in sections.items():
        if not block:
            continue
        print('\n%s' % section)
        print('%-28s %9s %9s %9s %21s %8s' % ('metric', 'A', 'B', 'A - B', '95% interval', 'clear?'))
        for r in block['rows']:
            print('%-28s %9.4f %9.4f %9.4f   [%8.4f, %8.4f] %8s'
                  % (r['metric'], r['run_a'], r['run_b'], r['difference'], r['ci95_low'], r['ci95_high'],
                     'yes' if r['interval_excludes_zero'] else 'no'))


if __name__ == '__main__':
    main()
