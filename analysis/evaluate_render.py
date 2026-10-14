"""Masked render-space comparison of a learned interior operator against analytic transport.

Every statistic is computed on linear EXR values over primary-hit stone pixels
only, so background and ground plane cannot dilute an error.

Independent-reference noise is measured, not assumed. Each mode is rendered
under several independent seeds; splitting those seeds in half gives two
independent estimates of the SAME image, and the difference between them is
pure Monte Carlo noise. The neural-versus-analytic comparison is then repeated
on matching half-splits, so signal and noise are read at equal sample counts.
A difference smaller than the noise floor is not evidence of anything.

The analytic path is the reference transport here. It is not a converged
physical ground truth, so these numbers bound agreement with that reference.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import math
import json
from pathlib import Path

import numpy as np

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_spectral')
from pipeline.render_learned_interior import make_scene, tumble_matrix
from neural.learned_interior_model import load_model
from utils.run_record import geometric_facets

LUMINANCE = np.array([.2126, .7152, .0722])


def load_renders(directories):
    """Load per-seed renders, requiring a shared azimuth, resolution and spp."""
    images, reports, components = [], [], {}
    for directory in directories:
        exr = Path(directory)/'frames'/'frame_0000.exr'
        report = Path(directory)/'render.json'
        if not exr.exists() or not report.exists():
            raise FileNotFoundError('Incomplete render in %s' % directory)
        images.append(np.array(mi.Bitmap(str(exr)), dtype=np.float64))
        reports.append(json.loads(report.read_text()))
        for name in ('operator', 'untouched'):
            piece = Path(directory)/'frames'/('component_%s.exr' % name)
            if piece.exists():
                components.setdefault(name, []).append(
                    np.array(mi.Bitmap(str(piece)), dtype=np.float64))
    if not images:
        raise ValueError('No renders given')
    for key in ('azimuth', 'width', 'height', 'spp'):
        if len({r[key] for r in reports}) != 1:
            raise ValueError('Renders disagree on %s' % key)
    seeds = [r['seed'] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated seeds are not independent samples')
    stacked = {k: np.stack(v) for k, v in components.items() if len(v) == len(images)}
    return np.stack(images), reports, stacked


def stone_hits(checkpoint, width, height, azimuth, rotation_deg=None, camera='studio'):
    """Primary-hit stone pixels and the triangle each one sees first (-1 off the stone)."""
    rotation = (None if rotation_deg is None
                else tumble_matrix(*[math.radians(v) for v in rotation_deg]))
    scene, _, mesh = make_scene(checkpoint, width, height, azimuth, rotation=rotation, camera=camera)
    sensor = scene.sensors()[0]
    triangle = np.full((height, width), -1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            ray, _ = sensor.sample_ray(0., .5, mi.Point2f((x+.5)/width, (y+.5)/height),
                                       mi.Point2f(.5))
            si = scene.ray_intersect(ray)
            if bool(si.is_valid()) and si.shape == mesh:
                triangle[y, x] = int(si.prim_index)
    return triangle >= 0, triangle


def stone_mask(checkpoint, width, height, azimuth, rotation_deg=None, camera='studio'):
    """Primary-hit stone pixels; `rotation_deg` matches render_learned_interior's tumble."""
    return stone_hits(checkpoint, width, height, azimuth, rotation_deg, camera)[0]


def _correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


def _ranks(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def noise_corrected(a1, a2, n1, n2):
    """Agreement with Monte Carlo noise removed, from two independent halves per mode.

    a1, a2 are analytic and n1, n2 neural values (pixels or facets) from disjoint
    seed sets. Debiased MSE averages (n1-a1)(n2-a2): the halves' noise is
    independent, so it cancels in expectation and what remains estimates the
    squared difference of the converged images. Cross-half correlation pairs
    different seeds, so noise the two modes share through common seeds cannot
    pose as agreement; dividing by the geometric mean of each mode's split-half
    reliability (Spearman's correction for attenuation) estimates the
    correlation at infinite samples.
    """
    reference_mean = float(np.concatenate([a1, a2]).mean())
    debiased_mse = float(((n1-a1)*(n2-a2)).mean())
    cross = [_correlation(a1, n2), _correlation(a2, n1)]
    cross = None if None in cross else float(np.mean(cross))
    reliability_analytic, reliability_neural = _correlation(a1, a2), _correlation(n1, n2)
    disattenuated = None
    if (cross is not None and reliability_analytic is not None and reliability_neural is not None
            and reliability_analytic > 0 and reliability_neural > 0):
        disattenuated = cross/math.sqrt(reliability_analytic*reliability_neural)
    debiased_rmse = math.sqrt(max(debiased_mse, 0.))
    return dict(debiased_mse=debiased_mse,
                debiased_rmse=debiased_rmse,
                debiased_relative_rmse=debiased_rmse/reference_mean if reference_mean else None,
                cross_half_correlation=cross,
                reliability_analytic=reliability_analytic,
                reliability_neural=reliability_neural,
                disattenuated_correlation=disattenuated)


def facet_means(image, triangle, facet_of_triangle, facets):
    """Mean stone luminance seen through each geometric facet."""
    y = image[triangle >= 0] @ LUMINANCE
    label = facet_of_triangle[triangle[triangle >= 0]]
    return np.array([y[label == f].mean() for f in facets])


def facet_comparison(images, triangle, facet_of_triangle, min_pixels=20, top_fraction=.1):
    """Compare the two modes facet by facet: which facets are lit, and how brightly.

    Each facet averages every pixel it covers, so per-pixel noise largely cancels;
    facets covering fewer than `min_pixels` pixels are left out. Every facet
    counts once, however large.
    """
    label = facet_of_triangle[triangle[triangle >= 0]]
    counts = np.bincount(label, minlength=facet_of_triangle.max()+1)
    facets = np.flatnonzero(counts >= min_pixels)
    a, n, a1, a2, n1, n2 = (facet_means(images[k], triangle, facet_of_triangle, facets)
                            for k in ('analytic', 'neural', 'a1', 'a2', 'n1', 'n2'))
    top = max(1, int(round(top_fraction*len(facets))))
    brightest_reference = set(np.argsort(-a)[:top].tolist())
    brightest_neural = set(np.argsort(-n)[:top].tolist())
    reference_mean = float(a.mean())
    rmse = float(np.sqrt(((n-a)**2).mean()))
    result = dict(
        triangles=int(len(facet_of_triangle)), geometric_facets=int(facet_of_triangle.max()+1),
        visible_facets=int(len(facets)), min_pixels=min_pixels,
        correlation=_correlation(a, n),
        rank_correlation=_correlation(_ranks(a), _ranks(n)),
        relative_rmse=rmse/reference_mean if reference_mean else None,
        relative_energy_error=float((n.sum()-a.sum())/a.sum()) if a.sum() else None,
        brightest_facets_counted=top,
        brightest_facet_overlap=len(brightest_reference & brightest_neural)/top,
        analytic_noise_floor_correlation=_correlation(a1, a2),
        neural_noise_floor_correlation=_correlation(n1, n2))
    result.update(noise_corrected(a1, a2, n1, n2))
    return result


# contrast (std / mean) with the render noise taken out: the covariance of two independent
# halves of the same image estimates the variance of the converged image
def noise_free_contrast(y1, y2):
    mean = .5*(y1.mean() + y2.mean())
    covariance = float(((y1 - y1.mean())*(y2 - y2.mean())).mean())
    return math.sqrt(max(covariance, 0.))/mean if mean > 0 else None


# 95% interval for a correlation by resampling the paired values (facets) with replacement
def bootstrap_correlation(x, y, repeats=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        pick = rng.integers(len(x), size=len(x))
        r = _correlation(x[pick], y[pick])
        if r is not None:
            values.append(r)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


# 95% interval for a pixel correlation, resampling square blocks of pixels, since neighbours are not independent
def block_bootstrap_correlation(x_image, y_image, mask, block=16, repeats=1000, seed=0):
    rows, columns = np.nonzero(mask)
    labels = (rows//block)*(mask.shape[1]//block + 1) + columns//block
    x, y = x_image[mask], y_image[mask]
    _, blocks = np.unique(labels, return_inverse=True)
    sums = np.stack([np.bincount(blocks, weights=w) for w in (np.ones_like(x), x, y, x*x, y*y, x*y)], 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        n, sx, sy, sxx, syy, sxy = sums[rng.integers(len(sums), size=len(sums))].sum(0)
        vx, vy = sxx/n - (sx/n)**2, syy/n - (sy/n)**2
        if vx > 0 and vy > 0:
            values.append((sxy/n - sx*sy/n**2)/math.sqrt(vx*vy))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


# the six numbers the paper reports, from two independent halves of each mode
def headline(a1, a2, n1, n2, mask, triangle, facet_of_triangle, min_pixels=20):
    la1, la2, ln1, ln2 = [(image @ LUMINANCE) for image in (a1, a2, n1, n2)]
    ya1, ya2, yn1, yn2 = [image[mask] for image in (la1, la2, ln1, ln2)]
    corrected = noise_corrected(ya1, ya2, yn1, yn2)

    label = facet_of_triangle[triangle[triangle >= 0]]
    counts = np.bincount(label, minlength=facet_of_triangle.max()+1)
    facets = np.flatnonzero(counts >= min_pixels)
    fa1, fa2, fn1, fn2 = (facet_means(image, triangle, facet_of_triangle, facets) for image in (a1, a2, n1, n2))
    fa, fn = .5*(fa1 + fa2), .5*(fn1 + fn2)

    reference, other = .5*(ya1 + ya2), .5*(yn1 + yn2)
    floor = .01*max(float(reference.mean()), 1e-12)
    log = lambda y: np.log(np.maximum(y, 0.) + floor)
    return dict(
        pixel_correlation=_correlation(ya1, yn1),
        pixel_rank_correlation=_correlation(_ranks(ya1), _ranks(yn1)),
        pixel_log_correlation=_correlation(log(ya1), log(yn1)),
        pixel_correlation_ci95=block_bootstrap_correlation(la1, ln1, mask),
        pixel_noise_floor=_correlation(ya1, ya2),
        disattenuated_correlation=corrected['disattenuated_correlation'],
        reliability_analytic=corrected['reliability_analytic'],
        reliability_neural=corrected['reliability_neural'],
        facet_correlation=_correlation(fa, fn),
        facet_correlation_ci95=bootstrap_correlation(fa, fn),
        facet_noise_floor=_correlation(fa1, fa2),
        facets_used=int(len(facets)),
        relative_energy_error=float((other.sum() - reference.sum())/reference.sum()) if reference.sum() else None,
        noise_free_contrast_analytic=noise_free_contrast(ya1, ya2),
        noise_free_contrast_neural=noise_free_contrast(yn1, yn2),
        pixels=int(mask.sum()))


HEADLINE_DEFINITIONS = {
    'pixel_correlation': 'Pearson correlation of linear luminance over stone pixels, analytic seeds of the first '
                         'half against neural seeds of the first half (equal sample counts)',
    'pixel_correlation_ci95': '95% interval, bootstrap over 16 x 16 pixel blocks',
    'pixel_noise_floor': 'the same correlation between the two analytic halves: the best any method can reach',
    'pixel_rank_correlation': 'Spearman (rank) correlation of the same pixels; every pixel counts equally, so a few '
                              'very bright sparkles cannot decide it',
    'pixel_log_correlation': 'Pearson correlation of log luminance (plus 1% of the mean reference luminance), which '
                             'weighs dark and bright regions more evenly',
    'disattenuated_correlation': 'cross-half correlation divided by the root of both modes\' split-half '
                                 'reliabilities; estimates the correlation at infinite samples',
    'facet_correlation': 'Pearson correlation of mean luminance per geometric facet (coplanar triangles merged), '
                         'facets covering at least 20 pixels',
    'facet_correlation_ci95': '95% interval, bootstrap over facets',
    'relative_energy_error': '(neural - analytic) / analytic, summed luminance over stone pixels',
    'noise_free_contrast': 'sqrt(covariance of the two halves) / mean: standard deviation over mean with the '
                           'render noise removed',
    'interior_only': 'the same metrics on light that went through the stone interior; the exact first-surface '
                     'reflection, identical in both modes, is excluded',
}


def highlight_agreement(reference, other, fraction=.01):
    """Do the brightest pixels land in the same places?"""
    count = max(1, int(round(fraction*reference.size)))
    top_reference = set(np.argpartition(-reference, count-1)[:count].tolist())
    top_other = set(np.argpartition(-other, count-1)[:count].tolist())
    return len(top_reference & top_other)/len(top_reference)


def centroid(values, coordinates):
    total = values.sum()
    if total <= 0:
        return None
    return (values[:, None]*coordinates).sum(0)/total


def compare(reference, other, mask, anchor=None):
    """Reference-relative agreement over stone pixels, in linear units."""
    a, b = reference[mask], other[mask]
    ya, yb = a @ LUMINANCE, b @ LUMINANCE
    finite = np.isfinite(ya) & np.isfinite(yb)
    ya, yb, a, b = ya[finite], yb[finite], a[finite], b[finite]
    mean_a = float(ya.mean())
    result = dict(pixels=int(ya.size), reference_mean=mean_a, other_mean=float(yb.mean()))
    result['relative_energy_error'] = float((yb.sum()-ya.sum())/ya.sum()) if ya.sum() else None
    rmse = float(np.sqrt(((yb-ya)**2).mean()))
    result['rmse'] = rmse
    result['relative_rmse'] = rmse/mean_a if mean_a else None
    if ya.std() > 0 and yb.std() > 0:
        result['correlation'] = float(np.corrcoef(ya, yb)[0, 1])
    else:
        result['correlation'] = None
    result['highlight_overlap_top1pct'] = highlight_agreement(ya, yb)
    # Brightness-weighted centroid drift, in pixels, over masked coordinates.
    rows, columns = np.nonzero(mask)
    coordinates = np.stack([rows, columns], 1).astype(float)[finite]
    ca, cb = centroid(np.maximum(ya, 0), coordinates), centroid(np.maximum(yb, 0), coordinates)
    result['highlight_centroid_shift_px'] = (float(np.linalg.norm(ca-cb))
                                             if ca is not None and cb is not None else None)
    # Chromaticity distance on pixels bright enough for colour to be meaningful.
    floor = max(mean_a, 1e-12)
    bright = (ya > floor) & (yb > floor)
    if bright.any():
        ca_ = np.maximum(a[bright], 0); cb_ = np.maximum(b[bright], 0)
        ca_ = ca_/np.maximum(ca_.sum(1, keepdims=True), 1e-30)
        cb_ = cb_/np.maximum(cb_.sum(1, keepdims=True), 1e-30)
        result['chromaticity_error'] = float(np.linalg.norm(ca_-cb_, axis=1).mean())
        result['chromaticity_pixels'] = int(bright.sum())
    else:
        result['chromaticity_error'] = None
        result['chromaticity_pixels'] = 0
    if anchor is not None:
        result['flash_fraction_reference'] = float((ya > anchor).mean())
        result['flash_fraction_other'] = float((yb > anchor).mean())
    return result


def shape_statistics(image, mask, anchor=None):
    """Within-image highlight distribution, independent of any comparison."""
    y = (image[mask] @ LUMINANCE)
    y = y[np.isfinite(y)]
    mean = float(y.mean())
    result = dict(mean=mean, median=float(np.median(y)), peak=float(y.max()),
                  p99=float(np.percentile(y, 99)),
                  contrast=float(y.std()/mean) if mean > 0 else None,
                  peak_over_mean=float(y.max()/mean) if mean > 0 else None)
    if anchor is not None:
        result['flash_fraction'] = float((y > anchor).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True,
                        help='Checkpoint supplying geometry for the stone mask')
    parser.add_argument('--analytic', type=Path, nargs='+', required=True,
                        help='Per-seed analytic render directories')
    parser.add_argument('--neural', type=Path, nargs='+', required=True,
                        help='Per-seed neural render directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flash_multiple', type=float, default=32.,
                        help='Flash threshold as a multiple of the analytic masked median')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.analytic) < 2 or len(args.neural) < 2:
        parser.error('Give at least two seeds per mode so the noise floor is measurable')

    analytic, analytic_reports, analytic_parts = load_renders(args.analytic)
    neural, neural_reports, neural_parts = load_renders(args.neural)
    if analytic.shape != neural.shape:
        raise ValueError('Modes differ in resolution or seed count')
    for key in ('azimuth', 'width', 'height', 'spp', 'rotation_deg', 'camera'):
        default = 'studio' if key == 'camera' else None
        if analytic_reports[0].get(key, default) != neural_reports[0].get(key, default):
            raise ValueError('Modes disagree on %s; comparison would be unpaired' % key)
    if analytic_reports[0]['mode'] != 'analytic' or neural_reports[0]['mode'] != 'neural':
        raise ValueError('Render modes are mislabelled')

    _, checkpoint = load_model(args.model)
    height, width = analytic.shape[1], analytic.shape[2]
    azimuth = analytic_reports[0]['azimuth']
    rotation_deg = analytic_reports[0].get('rotation_deg')
    print('computing stone mask at azimuth %g%s' % (
        azimuth, '' if rotation_deg is None else ', rotation %s' % (rotation_deg,)), flush=True)
    mask, triangle = stone_hits(checkpoint, width, height, azimuth, rotation_deg,
                                analytic_reports[0].get('camera', 'studio'))
    if not mask.any():
        raise ValueError('Empty stone mask')

    mean_analytic, mean_neural = analytic.mean(0), neural.mean(0)
    anchor = args.flash_multiple*float(np.median((mean_analytic[mask] @ LUMINANCE)))

    half = len(analytic)//2
    if half < 1:
        raise ValueError('Need at least two seeds per mode to split')
    a1, a2 = analytic[:half].mean(0), analytic[half:].mean(0)
    n1, n2 = neural[:half].mean(0), neural[half:].mean(0)

    report = dict(
        azimuth=azimuth, width=width, height=height,
        spp_each=analytic_reports[0]['spp'], seeds_per_mode=len(analytic),
        effective_spp=analytic_reports[0]['spp']*len(analytic),
        model=str(args.model), flash_multiple=args.flash_multiple,
        flash_threshold=anchor, mask_pixels=int(mask.sum()),
        analytic_shape=shape_statistics(mean_analytic, mask, anchor),
        neural_shape=shape_statistics(mean_neural, mask, anchor),
        full_comparison=compare(mean_analytic, mean_neural, mask, anchor),
        half_split_comparison=compare(a1, n1, mask, anchor),
        analytic_noise_floor=compare(a1, a2, mask, anchor),
        neural_noise_floor=compare(n1, n2, mask, anchor),
        noise_corrected=noise_corrected(*[(image[mask] @ LUMINANCE) for image in (a1, a2, n1, n2)]),
        facet_comparison=facet_comparison(
            dict(analytic=mean_analytic, neural=mean_neural, a1=a1, a2=a2, n1=n1, n2=n2),
            triangle, geometric_facets(checkpoint['vertices'].numpy().astype(np.float64),
                                       checkpoint['faces'].numpy())),
        limitations=[
            'Analytic mode is the reference transport path, not converged physical ground truth.',
            'Single view and single lighting rig; no held-out illumination.',
            'Noise floor and half-split comparison use half the seeds, so both sit at half the effective spp.',
            'Highlight overlap uses the brightest one percent of stone pixels.',
        ])
    # Isolating the learned component removes the analytic first-surface
    # reflection that both modes compute identically and that therefore
    # inflates whole-image agreement.
    shared = sorted(set(analytic_parts) & set(neural_parts))
    if shared:
        report['components'] = {}
        total = float((mean_analytic[mask] @ LUMINANCE).sum())
        for name in shared:
            a_part, n_part = analytic_parts[name].mean(0), neural_parts[name].mean(0)
            entry = compare(a_part, n_part, mask, anchor)
            entry['analytic_share_of_stone_radiance'] = float(
                (a_part[mask] @ LUMINANCE).sum()/total) if total else None
            entry['neural_share_of_stone_radiance'] = float(
                (n_part[mask] @ LUMINANCE).sum()/
                max(float((mean_neural[mask] @ LUMINANCE).sum()), 1e-30))
            report['components'][name] = entry

    facet_of_triangle = geometric_facets(checkpoint['vertices'].numpy(), checkpoint['faces'].numpy())
    interior = None
    if 'operator' in analytic_parts and 'operator' in neural_parts:
        parts_a, parts_n = analytic_parts['operator'], neural_parts['operator']
        interior = headline(parts_a[:half].mean(0), parts_a[half:].mean(0),
                            parts_n[:half].mean(0), parts_n[half:].mean(0), mask, triangle, facet_of_triangle)
    report['headline'] = dict(
        whole_stone=headline(a1, a2, n1, n2, mask, triangle, facet_of_triangle),
        interior_only=interior,
        definitions=HEADLINE_DEFINITIONS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nazimuth %g, %dx%d, %d effective spp, %d stone pixels'
          % (azimuth, width, height, report['effective_spp'], report['mask_pixels']))
    print('\n%-28s %14s %14s' % ('image shape', 'analytic', 'neural'))
    for key in ('mean', 'contrast', 'peak_over_mean', 'flash_fraction'):
        print('%-28s %14.5g %14.5g'
              % (key, report['analytic_shape'][key], report['neural_shape'][key]))
    print('\n%-28s %14s %14s %14s' % ('agreement (half-split)', 'neural', 'noise floor', 'ratio'))
    for key in ('relative_rmse', 'correlation', 'highlight_overlap_top1pct',
                'relative_energy_error', 'chromaticity_error'):
        signal = report['half_split_comparison'][key]
        noise = report['analytic_noise_floor'][key]
        if signal is None or noise is None:
            continue
        if key == 'correlation':
            ratio = '-'
        else:
            ratio = '%.4g' % (abs(signal)/abs(noise)) if noise else 'inf'
        print('%-28s %14.5g %14.5g %14s' % (key, signal, noise, ratio))
    corrected, facet = report['noise_corrected'], report['facet_comparison']
    print('\n%-28s %14s' % ('noise corrected (pixels)', 'neural'))
    for key in ('cross_half_correlation', 'disattenuated_correlation', 'debiased_relative_rmse'):
        value = corrected[key]
        print('%-28s %14s' % (key, 'n/a' if value is None else '%.5g' % value))
    print('\n%-28s %14s   (%d of %d facets visible, %d triangles)'
          % ('facets', 'neural', facet['visible_facets'], facet['geometric_facets'], facet['triangles']))
    for key in ('correlation', 'rank_correlation', 'disattenuated_correlation',
                'brightest_facet_overlap', 'relative_rmse', 'debiased_relative_rmse',
                'analytic_noise_floor_correlation'):
        value = facet[key]
        print('%-28s %14s' % (key, 'n/a' if value is None else '%.5g' % value))
    if report.get('components'):
        # The learned operator in isolation, with the shared analytic
        # first-surface reflection removed from both sides.
        print('\n%-28s %14s %14s %14s'
              % ('component', 'correlation', 'neural share', 'energy err'))
        for name, entry in report['components'].items():
            correlation = entry['correlation']
            print('%-28s %14s %14.4f %14.5g'
                  % (name,
                     'n/a' if correlation is None else '%.5g' % correlation,
                     entry['neural_share_of_stone_radiance'],
                     entry['relative_energy_error']))
    print('\n%-30s %16s %16s' % ('headline', 'whole stone', 'interior only'))
    for key in ('pixel_correlation', 'pixel_noise_floor', 'disattenuated_correlation', 'facet_correlation',
                'facet_noise_floor', 'relative_energy_error', 'noise_free_contrast_analytic',
                'noise_free_contrast_neural'):
        cells = []
        for block in (report['headline']['whole_stone'], report['headline']['interior_only']):
            value = None if block is None else block[key]
            cells.append('n/a' if value is None else '%.4f' % value)
        print('%-30s %16s %16s' % (key, *cells))
    print('\nWrote %s' % args.output)


if __name__ == '__main__':
    main()
