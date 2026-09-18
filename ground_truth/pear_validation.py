"""Watertightness and orientation checks for generated diamond meshes.

Written to replace the checks in geometry_leakage_validation.py, which runs
against make_flat_shaded's output. Flat shading gives every triangle its own
three vertices, so every edge appears unshared and that script reports one
boundary edge per edge on a perfectly sound mesh. These checks run on the mesh
the generator returns.

Four properties are tested, not one:

  boundary edges      an edge used by a single triangle is a hole
  non-manifold edges  an edge used by more than two is a self-intersection
  Euler characteristic  V - E + F must equal 2 for a closed genus-0 surface.
                      Stronger than edge counting alone: a mesh can be closed
                      and still topologically wrong, and only this catches it.
  orientation         every normal must face away from the interior, and the
                      signed volume must be positive. This matters to the
                      renderer, which decides inside from outside by the sign
                      of the incident direction against the surface normal.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ground_truth.brilliant_geometry import make_round_brilliant
from ground_truth.pear_geometry import make_pear_brilliant
from ground_truth.step_geometry import make_step_cut


def validate(vertices, faces):
    edges = defaultdict(int)
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[(min(a, b), max(a, b))] += 1
    boundary = sum(1 for n in edges.values() if n == 1)
    non_manifold = sum(1 for n in edges.values() if n > 2)

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    outward = int(((normals*(triangles.mean(1)-vertices.mean(0))).sum(1) <= 0).sum())
    volume = float(np.einsum('ij,ij->i', triangles[:, 0],
                             np.cross(triangles[:, 1], triangles[:, 2])).sum()/6.)
    degenerate = int((np.linalg.norm(normals, axis=1) < 1e-12).sum())
    duplicates = len(vertices)-len(np.unique(np.round(vertices, 9), axis=0))

    return dict(vertices=len(vertices), faces=len(faces), edges=len(edges),
                boundary_edges=boundary, non_manifold_edges=non_manifold,
                euler=len(vertices)-len(edges)+len(faces),
                inward_normals=outward, signed_volume=volume,
                degenerate_faces=degenerate, duplicate_vertices=duplicates)


def report(name, result):
    checks = [
        ('boundary edges', result['boundary_edges'] == 0, result['boundary_edges']),
        ('non-manifold edges', result['non_manifold_edges'] == 0, result['non_manifold_edges']),
        ('Euler characteristic = 2', result['euler'] == 2, result['euler']),
        ('all normals outward', result['inward_normals'] == 0, result['inward_normals']),
        ('positive signed volume', result['signed_volume'] > 0, round(result['signed_volume'], 4)),
        ('no degenerate faces', result['degenerate_faces'] == 0, result['degenerate_faces']),
        ('no duplicate vertices', result['duplicate_vertices'] == 0, result['duplicate_vertices']),
    ]
    passed = all(ok for _, ok, _ in checks)
    print('\n%s' % name)
    print('  %d vertices, %d edges, %d faces' % (result['vertices'], result['edges'], result['faces']))
    for label, ok, value in checks:
        print('    [%s] %-26s %s' % ('PASS' if ok else 'FAIL', label, value))
    print('  => %s' % ('WATERTIGHT AND CORRECTLY ORIENTED' if passed else 'FAILED'))
    return passed


def edge_diagnostic(vertices, faces, path, title):
    """Wireframe with faulty edges highlighted, in the style of boundary_edges.png.

    A sound mesh draws no red or orange, so the image is evidence rather than
    decoration: any hole or self-intersection would be visible and located.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shared = defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            shared[(min(a, b), max(a, b))].append(index)
    boundary = [e for e, f in shared.items() if len(f) == 1]
    non_manifold = [e for e, f in shared.items() if len(f) > 2]

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection='3d')
    for edge in shared:
        a, b = vertices[edge[0]], vertices[edge[1]]
        axis.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color='#2b5d9e', alpha=0.45, linewidth=1.0)
    for edge in boundary:
        a, b = vertices[edge[0]], vertices[edge[1]]
        axis.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'r-', linewidth=3)
    for edge in non_manifold:
        a, b = vertices[edge[0]], vertices[edge[1]]
        axis.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color='darkorange', linewidth=3)
    axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                 s=9, c='#123', depthshade=False)

    extent = float(np.abs(vertices).max())*1.05
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.view_init(elev=22, azim=-60)
    verdict = ('no boundary or non-manifold edges: WATERTIGHT'
               if not boundary and not non_manifold
               else '%d boundary (red), %d non-manifold (orange)'
                    % (len(boundary), len(non_manifold)))
    axis.set_title('%s\n%d vertices, %d edges, %d faces\n%s'
                   % (title, len(vertices), len(shared), len(faces), verdict), fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor='white')
    plt.close(figure)
    print('  wrote %s' % path)


def step_counts(spec):
    """Parse a 'crown,pavilion' step specification."""
    try:
        crown, pavilion = (int(part) for part in spec.split(','))
    except ValueError:
        raise ValueError('Step specification %r must read CROWN,PAVILION' % spec)
    return crown, pavilion


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pear_points', type=int, nargs='+', default=[12, 16, 24, 32])
    parser.add_argument('--taper', type=float, default=1.0)
    parser.add_argument('--png', type=Path, default=Path(__file__).with_name('pear_edges.png'),
                        help='Edge diagnostic image for the pear cut')
    parser.add_argument('--png_points', type=int, default=16,
                        help='Girdle point count used for the diagnostic image')
    parser.add_argument('--step_steps', nargs='+', default=['3,3', '2,1', '1,2', '4,4'],
                        metavar='CROWN,PAVILION',
                        help='Step cut tessellations to validate')
    parser.add_argument('--step_png', type=Path,
                        default=Path(__file__).with_name('step_edges.png'),
                        help='Edge diagnostic image for the step cut')
    parser.add_argument('--step_png_steps', default='3,3',
                        metavar='CROWN,PAVILION',
                        help='Step counts used for the step diagnostic image')
    args = parser.parse_args()

    print('='*64)
    print('DIAMOND MESH VALIDATION')
    print('='*64)

    everything = [('round_brilliant (num_main_facets=8)', make_round_brilliant())]
    for points in args.pear_points:
        everything.append(('pear_brilliant (num_girdle_points=%d)' % points,
                           make_pear_brilliant(num_girdle_points=points, taper=args.taper)))
    for spec in args.step_steps:
        crown, pavilion = step_counts(spec)
        everything.append(('step_cut (crown_steps=%d, pavilion_steps=%d)' % (crown, pavilion),
                           make_step_cut(crown_steps=crown, pavilion_steps=pavilion)))

    results = [report(name, validate(v, f)) for name, (v, f) in everything]

    print('\nEDGE DIAGNOSTIC IMAGES')
    vertices, faces = make_pear_brilliant(num_girdle_points=args.png_points,
                                          taper=args.taper)
    edge_diagnostic(vertices, faces, args.png,
                    'pear_brilliant (num_girdle_points=%d)' % args.png_points)
    crown, pavilion = step_counts(args.step_png_steps)
    vertices, faces = make_step_cut(crown_steps=crown, pavilion_steps=pavilion)
    edge_diagnostic(vertices, faces, args.step_png,
                    'step_cut (crown_steps=%d, pavilion_steps=%d)' % (crown, pavilion))

    print('\n' + '='*64)
    print('%d of %d meshes passed every check' % (sum(results), len(results)))
    print('='*64)
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
