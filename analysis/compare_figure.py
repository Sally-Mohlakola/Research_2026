"""Side-by-side comparison figure for a paired neural/analytic render.

Draws the neural render, the analytic reference and their difference on one
row, over stone pixels only, captioned with the correlation, samples per pixel
and stone-pixel count.

The difference panel is signed and symmetric about zero, so light placed where
the reference has none reads opposite to light the reference has and the model
misses. That distinction matters: a blurred version of the right image and a
sharp version of the wrong one both score badly on RMSE but look entirely
different here.

Exposure matches render_boundary.py's display convention, and is identical
across panels so brightness is comparable by eye.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi

mi.set_variant('scalar_spectral')
# One stone mask for every figure and evaluation, so pixels always agree.
from analysis.evaluate_render import stone_mask
from neural.boundary_model import load_model
from utils.studio_env import display_exposure

LUMINANCE = np.array([.2126, .7152, .0722])


def tonemap(image, exposure):
    return np.clip(np.maximum(image, 0)*exposure, 0, 1)**(1/2.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True,
                        help='Checkpoint supplying geometry for the stone mask')
    parser.add_argument('--neural', type=Path, required=True, help='Merged neural EXR')
    parser.add_argument('--analytic', type=Path, required=True, help='Merged analytic EXR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--azimuth', type=float, default=0.)
    parser.add_argument('--rotation_deg', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Stone rotation of a tumbled pose, as given to render_boundary')
    parser.add_argument('--evaluation', type=Path,
                        help='evaluation.json to annotate the figure with')
    parser.add_argument('--title', default='')
    args = parser.parse_args()

    neural = np.array(mi.Bitmap(str(args.neural)), dtype=np.float64)
    analytic = np.array(mi.Bitmap(str(args.analytic)), dtype=np.float64)
    if neural.shape != analytic.shape:
        raise ValueError('Renders differ in resolution')
    height, width = neural.shape[:2]

    _, checkpoint = load_model(args.model)
    mask = stone_mask(checkpoint, width, height, args.azimuth, args.rotation_deg)
    exposure = display_exposure()

    yn, ya = neural @ LUMINANCE, analytic @ LUMINANCE
    difference = np.where(mask, yn-ya, 0.)
    limit = float(np.percentile(np.abs(difference[mask]), 99)) or 1e-6

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for axis, image, label in ((axes[0], neural, 'neural render'),
                               (axes[1], analytic, 'analytic reference')):
        axis.imshow(tonemap(image, exposure))
        axis.set_title(label, fontsize=12)
        axis.set_axis_off()
    handle = axes[2].imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[2].set_title('difference (neural minus reference)\nred = light the model adds, '
                      'blue = light it misses', fontsize=10)
    axes[2].set_axis_off()
    figure.colorbar(handle, ax=axes[2], fraction=0.046, shrink=0.85)

    # The caption carries only what a reader needs to place the image: how well
    # it agrees with the reference and how it was sampled. The full set of
    # statistics is in evaluation.json and is printed below.
    lines = []
    if args.evaluation and args.evaluation.exists():
        report = json.loads(args.evaluation.read_text())
        lines.append('correlation %.4f   |   %d effective spp   |   %d stone pixels'
                     % (report['full_comparison']['correlation'], report['effective_spp'],
                        report['mask_pixels']))
    if args.title:
        figure.suptitle(args.title, fontsize=13)
    if lines:
        figure.text(0.5, 0.015, '\n'.join(lines), ha='center', fontsize=9, family='monospace')
    figure.tight_layout(rect=(0, 0.07 if lines else 0, 1, 0.96 if args.title else 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor='white')
    print('wrote %s' % args.output)
    for line in lines:
        print('  ' + line)


if __name__ == '__main__':
    main()
