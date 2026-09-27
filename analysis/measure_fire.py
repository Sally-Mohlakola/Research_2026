"""Measure fire: the colour in a render that is really there, not sampling noise.

Spectral renders are speckled with colour from one-wavelength-per-sample noise,
so "how colourful is it" does not measure dispersion. Fire is colour that
repeats: an independent render of the same frame shows the same spectral band
in the same place, while noise colour does not.

Each run's seeds are split into two independent halves. On the brightest
stone pixels, where fire lives, the chromatic offset of each pixel (its colour
minus its own grey, relative to its brightness) is computed in both halves:

- coherent colour: sqrt(mean(cA . cB)), an unbiased estimate of the colour
  that survives averaging -- fire;
- colour noise: sqrt(mean|cA - cB|^2 / 2), the colour that does not;
- agreement: the correlation of cA and cB, 1 for pure fire, 0 for pure noise.

Stone pixels come from the same mask evaluate_render.py uses.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config  # noqa: F401  resolve runtime before importing Mitsuba
import mitsuba as mi
from analysis.evaluate_render import stone_mask


def load(path):
    return np.asarray(mi.Bitmap(str(path)))[..., :3].astype(np.float64)


def halves(directory):
    frames = sorted(Path(directory).glob('s*/frames/frame_0000.exr'))
    if len(frames) < 2 or len(frames) % 2:
        raise ValueError('%s needs an even number (>= 2) of seed renders' % directory)
    images = [load(f) for f in frames]
    h = len(images)//2
    return np.mean(images[:h], 0), np.mean(images[h:], 0), len(frames)


def chroma(image):
    grey = image.mean(-1, keepdims=True)
    return (image - grey)/np.maximum(grey, 1e-12)


def measure(a, b, mask, fraction):
    full = (a + b)/2
    luminance = full.mean(-1)
    threshold = np.quantile(luminance[mask], 1 - fraction)
    bright = mask & (luminance >= threshold)
    ca, cb = chroma(a)[bright], chroma(b)[bright]
    shared = (ca*cb).sum(-1).mean()
    noise = ((ca - cb)**2).sum(-1).mean()/2
    agreement = shared/np.sqrt((ca**2).sum(-1).mean()*(cb**2).sum(-1).mean())
    return dict(pixels=int(bright.sum()), coherent_colour=float(np.sqrt(max(shared, 0.))),
                colour_noise=float(np.sqrt(noise)), agreement=float(agreement),
                mean_luminance=float(luminance[mask].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='Checkpoint for the stone mask')
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('LABEL', 'DIR'),
                        help='A label and a directory of seed renders (<dir>/s*/frames/). Repeatable.')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--azimuth', type=float, default=0.)
    parser.add_argument('--rotation_deg', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--bright_fraction', type=float, default=.1,
                        help='Measure on this brightest fraction of stone pixels')
    parser.add_argument('--output', type=Path, required=True, help='JSON report')
    parser.add_argument('--figure', type=Path, help='Optional crop comparison PNG')
    parser.add_argument('--crop', type=int, nargs=3, metavar=('X', 'Y', 'SIZE'))
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    mask = np.asarray(stone_mask(checkpoint, args.width, args.height, args.azimuth,
                                 args.rotation_deg)).astype(bool)
    report, images = {}, {}
    for label, directory in args.run:
        a, b, seeds = halves(directory)
        report[label] = dict(directory=directory, seeds=seeds, **measure(a, b, mask, args.bright_fraction))
        images[label] = (a + b)/2
        r = report[label]
        print('%-28s coherent colour %.4f   colour noise %.4f   agreement %.3f   (%d px, %d seeds)'
              % (label, r['coherent_colour'], r['colour_noise'], r['agreement'], r['pixels'], seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(bright_fraction=args.bright_fraction,
                                           rotation_deg=args.rotation_deg, azimuth=args.azimuth,
                                           runs=report), indent=2))
    if args.figure:
        figure(images, report, args)


def figure(images, report, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.studio_env import display_exposure
    labels = list(images)
    exposure = display_exposure()
    srgb = lambda x: np.clip(np.where(x <= .0031308, 12.92*x, 1.055*np.power(np.maximum(x, 0), 1/2.4) - .055), 0, 1)
    rows = 2 if args.crop else 1
    fig, axes = plt.subplots(rows + 1, len(labels), figsize=(4.6*len(labels), 4.6*(rows + 1) - 1),
                             squeeze=False)
    for col, label in enumerate(labels):
        img = srgb(images[label]*exposure)
        axes[0, col].imshow(img); axes[0, col].set_title(label, fontsize=12); axes[0, col].axis('off')
        if args.crop:
            x, y, s = args.crop
            axes[0, col].add_patch(plt.Rectangle((x, y), s, s, fill=False, ec='yellow', lw=1.2))
            axes[1, col].imshow(img[y:y+s, x:x+s], interpolation='nearest'); axes[1, col].axis('off')
        axes[-1, col].axis('off')
    ax = fig.add_subplot(rows + 1, 1, rows + 1)
    x = np.arange(len(labels))
    for i, (key, name, colour) in enumerate((('coherent_colour', 'coherent colour (fire)', '#b5446e'),
                                             ('colour_noise', 'colour noise', '#888888'))):
        values = [report[l][key] for l in labels]
        bars = ax.bar(x + (i - .5)*.36, values, .36, label=name, color=colour)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v, '%.3f' % v, ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, ['%s\nagreement %.2f' % (l, report[l]['agreement']) for l in labels])
    ax.set_ylabel('chromatic offset / brightness'); ax.legend()
    ax.set_title('Colour on the brightest %d%% of stone pixels, from independent halves'
                 % round(100*args.bright_fraction))
    fig.tight_layout()
    fig.savefig(args.figure, dpi=100, bbox_inches='tight')
    print('wrote %s' % args.figure)


if __name__ == '__main__':
    main()
