"""Draw how gather_boundary.py chooses entry states, from the real sampling code.

Top row: the rays themselves. Bottom row: where they land on the stone.

  uniform          a random direction on the sphere, with the ray origin offset
                   across a disc behind the stone, so every incoming direction
                   and every part of the silhouette is equally likely
  camera, one view a pinhole camera: one fixed origin, directions spread across
                   the square film, exactly as render_boundary samples pixels
  camera, orbit    the same camera at random azimuths over the full range,
                   which is what the training pools actually used

Rays are drawn only if they hit the stone from outside, the same rejection the
gather applies, so the figure shows the distribution the network trains on.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gather_boundary as gb
from config.parameters import get_diamond_parameters

RAY = '#2a78d6'
CAMERA = '#eb6834'
INK = '#0b0b0b'
SECONDARY = '#52514e'
SURFACE = '#fcfcfb'


def uniform_ray(rng, radius):
    """Lines 99-107 of gather_boundary.gather, reproduced verbatim."""
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    axis = np.array([0., 0., 1.]) if abs(u[2]) < .9 else np.array([1., 0., 0.])
    tangent = np.cross(u, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(u, tangent)
    r, phi = radius*np.sqrt(rng.random()), 2*np.pi*rng.random()
    origin = 3*radius*u + r*(np.cos(phi)*tangent + np.sin(phi)*bitangent)
    return origin, -u


def sample(scene, sampler, count, rng):
    """Keep rays that hit the stone from outside, as the gather does."""
    origins, hits = [], []
    while len(hits) < count:
        origin, travel = sampler(rng)
        si = scene.ray_intersect(gb.mi.Ray3f(gb.mi.Point3f(origin), gb.mi.Vector3f(travel)))
        if si.is_valid() and si.wi.z > 0:
            origins.append(origin)
            hits.append(np.array(si.p))
    return np.array(origins), np.array(hits)


def stone(axis, vertices, faces):
    axis.add_collection3d(Poly3DCollection(vertices[faces], facecolors=(.80, .82, .86, .35),
                                           edgecolors=(.35, .38, .45, .35), linewidths=.3))


def frame(axis, title, subtitle, limit=4.6):
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit*.55, limit)
    axis.set_box_aspect((1, 1, .78))
    axis.view_init(elev=18, azim=-38)
    axis.set_axis_off()
    axis.set_title(title, fontsize=11, color=INK, pad=0)
    axis.text2D(.5, .94, subtitle, transform=axis.transAxes, ha='center',
                fontsize=8.5, color=SECONDARY)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diamond_name', default='round_diamond_gia')
    parser.add_argument('--output', type=Path, default=Path('renders/figures/entry_sampling.png'))
    parser.add_argument('--rays', type=int, default=110)
    parser.add_argument('--points', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()

    parameters = get_diamond_parameters(args.diamond_name)
    scene, vertices, faces = gb.build_scene(parameters)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    samplers = [
        ('uniform', 'every direction; origin offset across a disc',
         lambda rng: uniform_ray(rng, radius)),
        ('camera, one view', 'one pinhole origin; directions offset across the film',
         lambda rng: gb.camera_ray(rng, (0., 0.))),
        ('camera, full orbit', 'the pinhole at random azimuths — what training used',
         lambda rng: gb.camera_ray(rng, (0., 360.))),
    ]

    figure = plt.figure(figsize=(15, 9.6))
    figure.patch.set_facecolor(SURFACE)
    for column, (name, subtitle, sampler) in enumerate(samplers):
        rng = np.random.default_rng(args.seed)
        origins, hits = sample(scene, sampler, args.rays, rng)
        axis = figure.add_subplot(2, 3, column+1, projection='3d')
        axis.set_facecolor(SURFACE)
        stone(axis, vertices, faces)
        for o, h in zip(origins, hits):
            axis.plot(*zip(o, h), color=RAY, linewidth=.55, alpha=.55)
        axis.scatter(*hits.T, s=5, color=RAY, depthshade=False)
        if name != 'uniform':
            unique = np.unique(np.round(origins, 6), axis=0)
            axis.scatter(*unique.T, s=34 if len(unique) == 1 else 9, color=CAMERA,
                         depthshade=False, zorder=5)
        frame(axis, name, subtitle)

        rng = np.random.default_rng(args.seed+1)
        _, points = sample(scene, sampler, args.points, rng)
        crown = (points[:, 2] > 1e-4).mean()
        axis = figure.add_subplot(2, 3, column+4, projection='3d')
        axis.set_facecolor(SURFACE)
        stone(axis, vertices, faces)
        axis.scatter(*points.T, s=1.6, color=RAY, alpha=.7, depthshade=False)
        frame(axis, 'where entries land',
              '%.0f%% on the crown, %.0f%% on the pavilion' % (100*crown, 100*(1-crown)),
              limit=1.25)

    figure.text(.012, .972, 'How gather_boundary.py chooses entry states', fontsize=13.5,
                color=INK, ha='left')
    figure.text(.012, .947, 'Orange = camera origin. Blue = rays that hit the stone from '
                'outside, and where they land. Real samples from the gather code, '
                '%s.' % args.diamond_name, fontsize=9, color=SECONDARY, ha='left')
    figure.subplots_adjust(left=0, right=1, top=.92, bottom=0, wspace=-.08, hspace=.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=SURFACE)
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
