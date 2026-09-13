"""Average independent equal-spp renders into one lower-variance image.

Averaging K independent renders of S samples per pixel is exactly the same
estimator as one render of K*S samples per pixel, provided the seeds differ, so
this is a parallel route to high sample counts rather than an approximation.
Every input must share resolution and spp or the average would be a weighted
mix with unequal, unrecorded weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_rgb')
from utils.studio_env import display_exposure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    directories = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    images, reports = [], []
    for directory in directories:
        exr = directory/'frames'/'frame_0000.exr'
        report = directory/'render.json'
        if not exr.exists() or not report.exists():
            raise FileNotFoundError(f'Incomplete render in {directory}')
        images.append(np.array(mi.Bitmap(str(exr)), dtype=np.float64))
        reports.append(json.loads(report.read_text()))
    if not images:
        raise ValueError('No renders found')
    if len({i.shape for i in images}) != 1:
        raise ValueError('Renders differ in resolution')
    if len({r['spp'] for r in reports}) != 1:
        raise ValueError('Renders differ in spp; a plain mean would be wrong')
    seeds = [r['seed'] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated seeds are not independent samples')

    stack = np.stack(images)
    mean = stack.mean(0)
    if not np.isfinite(mean).all():
        raise ValueError('Nonfinite merged image')
    # Spread across independent renders is the Monte Carlo error of the mean.
    standard_error = stack.std(0, ddof=1)/np.sqrt(len(stack))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(mean.astype(np.float32)).write(str(args.output.with_suffix('.exr')))
    mi.Bitmap(np.maximum(mean, 0).astype(np.float32)*display_exposure()).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write(
        str(args.output.with_suffix('.png')))

    total = sum(r['spp'] for r in reports)
    luminance = np.array([.2126, .7152, .0722])
    summary = dict(renders=len(images), seeds=seeds, spp_each=reports[0]['spp'],
                   effective_spp=total, mode=reports[0]['mode'],
                   model=reports[0]['model'], azimuth=reports[0]['azimuth'],
                   width=reports[0]['width'], height=reports[0]['height'],
                   checkpoint_sha256=reports[0]['checkpoint_sha256'],
                   mean_luminance=float((mean @ luminance).mean()),
                   relative_standard_error=float(
                       (standard_error @ luminance).sum()/max((mean @ luminance).sum(), 1e-30)),
                   seconds_each=[r['seconds'] for r in reports])
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
