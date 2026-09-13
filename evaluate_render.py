"""Masked render-space comparison of a learned boundary operator against analytic transport.

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
import argparse
import json
from pathlib import Path

import numpy as np

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_spectral')
from render_boundary import make_scene
from neural.boundary_model import load_model

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


def stone_mask(checkpoint, width, height, azimuth):
    scene, _, mesh = make_scene(checkpoint, width, height, azimuth)
    sensor = scene.sensors()[0]
    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            ray, _ = sensor.sample_ray(0., .5, mi.Point2f((x+.5)/width, (y+.5)/height),
                                       mi.Point2f(.5))
            si = scene.ray_intersect(ray)
            mask[y, x] = bool(si.is_valid()) and si.shape == mesh
    return mask


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
    for key in ('azimuth', 'width', 'height', 'spp'):
        if analytic_reports[0][key] != neural_reports[0][key]:
            raise ValueError('Modes disagree on %s; comparison would be unpaired' % key)
    if analytic_reports[0]['mode'] != 'analytic' or neural_reports[0]['mode'] != 'neural':
        raise ValueError('Render modes are mislabelled')

    _, checkpoint = load_model(args.model)
    height, width = analytic.shape[1], analytic.shape[2]
    azimuth = analytic_reports[0]['azimuth']
    print('computing stone mask at azimuth %g' % azimuth, flush=True)
    mask = stone_mask(checkpoint, width, height, azimuth)
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
    print('\nWrote %s' % args.output)


if __name__ == '__main__':
    main()
