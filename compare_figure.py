"""Side-by-side comparison figure for a paired neural/analytic render.

Draws the learned render, the analytic reference and their difference on one
row, over stone pixels only, with the agreement statistics printed beneath.

The difference panel is signed and symmetric about zero, so light placed where
the reference has none reads opposite to light the reference has and the model
misses. That distinction matters: a blurred version of the right image and a
sharp version of the wrong one both score badly on RMSE but look entirely
different here.

Exposure matches render_boundary.py's display convention, and is identical
across panels so brightness is comparable by eye.
"""
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
from render_boundary import make_scene
from neural.boundary_model import load_model
from utils.studio_env import display_exposure

LUMINANCE = np.array([.2126, .7152, .0722])


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
    mask = stone_mask(checkpoint, width, height, args.azimuth)
    exposure = display_exposure()

    yn, ya = neural @ LUMINANCE, analytic @ LUMINANCE
    difference = np.where(mask, yn-ya, 0.)
    limit = float(np.percentile(np.abs(difference[mask]), 99)) or 1e-6

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for axis, image, label in ((axes[0], neural, 'learned operator'),
                               (axes[1], analytic, 'analytic reference')):
        axis.imshow(tonemap(image, exposure))
        axis.set_title(label, fontsize=12)
        axis.set_axis_off()
    handle = axes[2].imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[2].set_title('difference (neural minus reference)\nred = light the model adds, '
                      'blue = light it misses', fontsize=10)
    axes[2].set_axis_off()
    figure.colorbar(handle, ax=axes[2], fraction=0.046, shrink=0.85)

    lines = []
    if args.evaluation and args.evaluation.exists():
        report = json.loads(args.evaluation.read_text())
        full, floor = report['full_comparison'], report['analytic_noise_floor']
        shape_a, shape_n = report['analytic_shape'], report['neural_shape']
        lines.append('correlation %.4f  (noise floor %.4f)   |   highlight overlap %.4f  '
                     '(floor %.4f)   |   relative energy %+.1f%%'
                     % (full['correlation'], floor['correlation'],
                        full['highlight_overlap_top1pct'],
                        floor['highlight_overlap_top1pct'],
                        100*full['relative_energy_error']))
        lines.append('contrast  neural %.3f  vs  reference %.3f      '
                     'peak/mean  neural %.1f  vs  reference %.1f      '
                     '%d stone pixels, %d effective spp'
                     % (shape_n['contrast'], shape_a['contrast'],
                        shape_n['peak_over_mean'], shape_a['peak_over_mean'],
                        report['mask_pixels'], report['effective_spp']))
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
