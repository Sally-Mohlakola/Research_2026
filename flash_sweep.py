"""Scintillation (flash) sweep: paired neural/analytic renders over a camera orbit.

A round brilliant flashes when facets swing through specular alignment, so the
diagnostic is angular: sample azimuth finely enough to resolve individual
flashes, then ask whether highlights are sparse and intense (flashing) or broad
and steady (over-smoothed).

The camera orbits while the studio lights stay fixed, so pixel (x, y) views
different geometry in every frame. No per-pixel temporal variance is therefore
computed; every statistic here is correspondence-free, either a within-frame
distribution shape or the across-frame variation of such a statistic.

Each configuration is rendered under two independent seeds. The seed-to-seed
spread is the Monte Carlo noise floor and is reported alongside the neural
versus analytic difference, because sparse-highlight statistics are exactly the
ones that path-tracing noise inflates.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_spectral')
from render_boundary import make_scene
from neural.boundary_model import load_model

LUMINANCE = np.array([.2126, .7152, .0722])


def diamond_mask(checkpoint, width, height, azimuth):
    """Pixels whose primary ray hits the stone, at this azimuth."""
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


def frame_statistics(image, mask, thresholds):
    """Within-frame highlight-distribution shape over stone pixels only."""
    y = (image @ LUMINANCE)[mask]
    y = y[np.isfinite(y)]
    if not y.size:
        return None
    mean = float(y.mean())
    result = dict(pixels=int(y.size), mean=mean, median=float(np.median(y)),
                  peak=float(y.max()), p99=float(np.percentile(y, 99)),
                  contrast=float(y.std()/mean) if mean > 0 else None,
                  peak_over_mean=float(y.max()/mean) if mean > 0 else None)
    result['flash_fraction'] = {'%g' % t: float((y > t).mean()) for t in thresholds}
    return result


def run_render(task):
    directory, model, mode, azimuth, seed, width, height, spp, scene_depth = task
    command = [sys.executable, '-B', 'render_boundary.py', '--model', str(model),
               '--output_dir', str(directory), '--mode', mode,
               '--azimuth', '%g' % azimuth, '--seed', str(seed),
               '--width', str(width), '--height', str(height), '--spp', str(spp),
               '--scene_depth', str(scene_depth)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError('Render failed (%s az=%g seed=%d):\n%s'
                           % (mode, azimuth, seed, done.stderr[-2000:]))
    return directory


def load_exr(directory):
    return np.array(mi.Bitmap(str(Path(directory)/'frames'/'frame_0000.exr')), dtype=np.float64)


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    return dict(mean=mean, std=float(array.std()), min=float(array.min()),
                max=float(array.max()),
                coefficient_of_variation=float(array.std()/mean) if mean else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--azimuth_start', type=float, default=0.)
    parser.add_argument('--azimuth_stop', type=float, default=45.)
    parser.add_argument('--azimuth_step', type=float, default=1.5)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=96)
    parser.add_argument('--spp', type=int, default=16)
    parser.add_argument('--scene_depth', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 72])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--flash_multiple', type=float, default=32.,
                        help='Headline flash threshold, as a multiple of the analytic '
                             'masked median luminance. Every multiple in the ladder is '
                             'recorded regardless; this only selects the reported one.')
    args = parser.parse_args()
    if args.azimuth_step <= 0 or args.azimuth_stop <= args.azimuth_start:
        parser.error('Require azimuth_start < azimuth_stop and a positive step')
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Give at least two distinct seeds so the noise floor is measurable')
    if args.workers < 1:
        parser.error('--workers must be at least one')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    azimuths = [float(a) for a in np.arange(args.azimuth_start, args.azimuth_stop,
                                            args.azimuth_step)]
    modes = ['analytic', 'neural']
    _, checkpoint = load_model(args.model)

    tasks, index = [], {}
    for mode in modes:
        for azimuth in azimuths:
            for seed in args.seeds:
                directory = args.output_dir/'frames'/('%s_az%06.2f_s%d' % (mode, azimuth, seed))
                index[(mode, azimuth, seed)] = directory
                tasks.append((directory, args.model, mode, azimuth, seed,
                              args.width, args.height, args.spp, args.scene_depth))
    print('%d renders: %d modes x %d azimuths x %d seeds'
          % (len(tasks), len(modes), len(azimuths), len(args.seeds)), flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_render, tasks):
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print('  %d/%d renders done' % (completed, len(tasks)), flush=True)

    print('computing stone masks', flush=True)
    masks = {a: diamond_mask(checkpoint, args.width, args.height, a) for a in azimuths}

    # One absolute threshold set, shared by both modes and every frame, anchored
    # to the analytic reference so neural highlights are judged on its scale.
    reference = np.concatenate([(load_exr(index[('analytic', a, args.seeds[0])]) @ LUMINANCE)[masks[a]]
                                for a in azimuths])
    anchor = float(np.median(reference[np.isfinite(reference)]))
    multiples = sorted({4., 8., 16., 32., 64., 128., float(args.flash_multiple)})
    thresholds = [anchor*k for k in multiples]
    print('analytic masked median luminance %.6g; thresholds %s'
          % (anchor, ['%.4g' % t for t in thresholds]), flush=True)

    per_frame = {mode: {seed: [] for seed in args.seeds} for mode in modes}
    for mode in modes:
        for seed in args.seeds:
            for a in azimuths:
                stats = frame_statistics(load_exr(index[(mode, a, seed)]), masks[a], thresholds)
                per_frame[mode][seed].append(dict(azimuth=a, **(stats or {})))

    key = '%g' % (anchor*float(args.flash_multiple))

    def pick(entry, name):
        if name == 'flash_fraction':
            return entry.get('flash_fraction', {}).get(key)
        return entry.get(name)

    report = dict(model=str(args.model), azimuths=azimuths, seeds=args.seeds,
                  width=args.width, height=args.height, spp=args.spp,
                  scene_depth=args.scene_depth, luminance_anchor=anchor,
                  thresholds=thresholds, threshold_multiples=multiples,
                  flash_multiple=float(args.flash_multiple), flash_threshold_used=key,
                  per_frame=per_frame, summary={}, noise_floor={})
    for mode in modes:
        first = per_frame[mode][args.seeds[0]]
        report['summary'][mode] = {name: summarize([pick(f, name) for f in first])
                                   for name in ('contrast', 'peak_over_mean',
                                                'flash_fraction', 'mean')}
        # Seed-to-seed spread of the same statistics: the Monte Carlo noise floor.
        spread = {}
        for name in ('contrast', 'peak_over_mean', 'flash_fraction', 'mean'):
            gaps = []
            for i in range(len(azimuths)):
                values = [pick(per_frame[mode][s][i], name) for s in args.seeds]
                values = [v for v in values if v is not None]
                if len(values) == len(args.seeds):
                    gaps.append(max(values)-min(values))
            spread[name] = summarize(gaps)
        report['noise_floor'][mode] = spread
    report['limitations'] = [
        'Camera orbits, so pixels do not correspond across frames; no per-pixel temporal variance.',
        'Statistics use linear EXR luminance over primary-hit stone pixels only.',
        'Sparse-highlight statistics are inflated by Monte Carlo noise; read them against the noise floor.',
        'Analytic mode is the reference transport path here, not a converged physical ground truth.',
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'flash_sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\n%16s %26s %26s' % ('statistic', 'analytic', 'neural'))
    for name in ('contrast', 'peak_over_mean', 'flash_fraction', 'mean'):
        row = '%16s' % name
        for mode in modes:
            s = report['summary'][mode][name]
            row += ('  %11.5g +-%10.4g' % (s['mean'], s['std'])) if s else '%26s' % 'n/a'
        print(row)
    print('\nseed-to-seed noise floor (mean gap over azimuths):')
    for name in ('contrast', 'flash_fraction'):
        print('  %-16s analytic %-12.5g neural %.5g'
              % (name, report['noise_floor']['analytic'][name]['mean'],
                 report['noise_floor']['neural'][name]['mean']))
    print('\nWrote %s' % (args.output_dir/'flash_sweep.json'))


if __name__ == '__main__':
    main()
