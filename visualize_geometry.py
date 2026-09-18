"""Draw the diamond meshes: facet structure for every configured variant.

Shades each triangle by its normal against a fixed light so facet boundaries are
legible, and reports the watertightness of the real mesh.

Note on watertightness: run the check on the mesh `make_round_brilliant`
returns, not on the output of `make_flat_shaded`. Flat shading gives every
triangle its own three vertices, so every edge appears unshared and a naive
check reports one boundary edge per edge. That is a property of the shaded
copy, not a hole in the geometry.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ground_truth.cuts import make_diamond
from ground_truth.pear_geometry import make_pear_brilliant
from ground_truth.step_geometry import make_step_cut
from config.parameters import DIAMOND_VARIANTS

LIGHT = np.array([0.4, 0.5, 0.75])


def watertight(vertices, faces):
    """Boundary and non-manifold edge counts for the unshaded mesh."""
    edges = defaultdict(int)
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[(min(a, b), max(a, b))] += 1
    boundary = sum(1 for n in edges.values() if n == 1)
    non_manifold = sum(1 for n in edges.values() if n > 2)
    return boundary, non_manifold


def view_direction(elevation, azimuth):
    """Unit vector from the origin toward the camera, in matplotlib's convention."""
    e, a = np.radians(elevation), np.radians(azimuth)
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])


def feature_edges(vertices, faces, normals, visible, tolerance=1e-5):
    """Edges where the surface actually creases, not every triangle edge.

    A flat facet is triangulated, so drawing every edge scores a table with the
    diagonals of its own fan -- the cross that a real step cut's table does not
    have. Those interior edges join two coplanar triangles and carry no
    geometry, so only edges whose two faces disagree on a normal are drawn, and
    only where at least one of those faces is turned toward the camera.

    The tolerance has to sit below the smallest real crease. A step cut at
    step_angle_spread=0.06 puts only about two degrees between neighbouring
    terraces, so a loose tolerance erases the very step lines the cut is
    named for; 1e-5 is roughly a quarter of a degree, well under that and
    well over float32 noise on a genuinely flat facet.
    """
    shared = defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            shared[(min(a, b), max(a, b))].append(index)
    return [[vertices[a], vertices[b]] for (a, b), owners in shared.items()
            if visible[owners].any()
            and (len(owners) != 2
                 or normals[owners[0]] @ normals[owners[1]] < 1.-tolerance)]


def draw(axis, vertices, faces, elevation, azimuth, title):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    shade = np.clip(normals @ (LIGHT/np.linalg.norm(LIGHT)), 0, 1)
    colours = plt.cm.bone(0.25 + 0.7*shade)
    # Matplotlib does not hide surfaces behind other surfaces, so without
    # culling the culet and the far facets read through the table as phantom
    # edges. Drop the back faces and the silhouette is what the eye expects.
    visible = normals @ view_direction(elevation, azimuth) > 0.
    # Edge colour matches the fill so adjacent coplanar triangles do not leave
    # hairline seams where one flat facet has been triangulated.
    axis.add_collection3d(Poly3DCollection(triangles[visible], facecolors=colours[visible],
                                           edgecolors=colours[visible], linewidths=0.4))
    axis.add_collection3d(Line3DCollection(
        feature_edges(vertices, faces, normals, visible),
        colors=[(0.15, 0.2, 0.3, 0.9)], linewidths=0.7))
    extent = float(np.abs(vertices).max())*1.05
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    axis.set_title(title, fontsize=9, pad=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('renders/diamond_variants.png'))
    parser.add_argument('--pear', type=int, nargs='*', metavar='POINTS',
                        help='Also draw pear brilliants with these girdle point '
                             'counts, e.g. --pear 16 32')
    parser.add_argument('--pear_taper', type=float, default=1.0,
                        help='Higher values sharpen the point of the pear.')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    views = [(22, -60, 'three-quarter'), (88, -90, 'top (table)'), (2, -90, 'side (profile)')]
    meshes = []
    for name in DIAMOND_VARIANTS:
        # Dispatch on the variant's 'cut' key; round, pear and step cuts all
        # live in DIAMOND_VARIANTS now, so a single generator will not do.
        meshes.append((name, make_diamond(DIAMOND_VARIANTS[name])))
    if args.pear:
        for points in args.pear:
            meshes.append(('pear_brilliant (n=%d)' % points,
                           make_pear_brilliant(num_girdle_points=points,
                                               taper=args.pear_taper)))

    figure = plt.figure(figsize=(3.1*len(views), 3.4*len(meshes)))
    for row, (name, (vertices, faces)) in enumerate(meshes):
        boundary, non_manifold = watertight(vertices, faces)
        for column, (elevation, azimuth, label) in enumerate(views):
            axis = figure.add_subplot(len(meshes), len(views),
                                      row*len(views)+column+1, projection='3d')
            heading = label if column else '%s\n%d verts / %d faces / %s' % (
                name, len(vertices), len(faces),
                'watertight' if boundary == 0 and non_manifold == 0 else 'LEAKS')
            draw(axis, vertices, faces, elevation, azimuth, heading)
        print('%-28s %3d vertices %3d faces  boundary %d  non-manifold %d'
              % (name, len(vertices), len(faces), boundary, non_manifold))

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print('\nWrote %s' % args.output)


if __name__ == '__main__':
    main()
