import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import config
import mitsuba as mi

mi.set_variant('scalar_rgb')
from utils.studio_env import display_exposure


# read every seed's frame and render.json from the input folder
def load_renders(input_dir):
    directories = sorted(p for p in input_dir.iterdir() if p.is_dir())
    images, reports = [], []

    for directory in directories:
        exr = directory/'frames'/'frame_0000.exr'
        report = directory/'render.json'
        if not exr.exists() or not report.exists():
            raise FileNotFoundError(f'Incomplete render in {directory}')

        images.append(np.array(mi.Bitmap(str(exr)), dtype=np.float64))
        reports.append(json.loads(report.read_text()))

    return images, reports


# the seeds can only be averaged if they match in size and spp and never repeat
def check_mergeable(images, reports):
    if not images:
        raise ValueError('No renders found')
    if len({i.shape for i in images}) != 1:
        raise ValueError('Renders differ in resolution')
    if len({r['spp'] for r in reports}) != 1:
        raise ValueError('Renders differ in spp; a plain mean would be wrong')

    seeds = [r['seed'] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated seeds are not independent samples')


# average the seeds into one image, and write it as exr, png and a json summary
def main():
    parser = argparse.ArgumentParser(description='Merge independent seed renders into one image.')
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    images, reports = load_renders(args.input_dir)
    check_mergeable(images, reports)

    stack = np.stack(images)
    mean = stack.mean(0)
    if not np.isfinite(mean).all():
        raise ValueError('Nonfinite merged image')
    standard_error = stack.std(0, ddof=1)/np.sqrt(len(stack))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(mean.astype(np.float32)).write(str(args.output.with_suffix('.exr')))
    mi.Bitmap(np.maximum(mean, 0).astype(np.float32)*display_exposure()).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write(
        str(args.output.with_suffix('.png')))

    luminance = np.array([.2126, .7152, .0722])
    first = reports[0]
    summary = dict(
        renders=len(images),
        seeds=[r['seed'] for r in reports],
        spp_each=first['spp'],
        effective_spp=sum(r['spp'] for r in reports),
        mode=first['mode'],
        model=first['model'],
        azimuth=first['azimuth'],
        width=first['width'],
        height=first['height'],
        checkpoint_sha256=first['checkpoint_sha256'],
        mean_luminance=float((mean @ luminance).mean()),
        relative_standard_error=float(
            (standard_error @ luminance).sum()/max((mean @ luminance).sum(), 1e-30)),
        seconds_each=[r['seconds'] for r in reports],
    )

    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
