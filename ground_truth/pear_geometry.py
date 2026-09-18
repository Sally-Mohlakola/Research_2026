"""Pear (teardrop) brilliant geometry, matching make_round_brilliant's conventions.

The girdle outline is the classical teardrop curve

    x(t) = cos t,    y(t) = sin t * sin(t/2) ** taper

which is cusped at t = 0, giving the point of the pear, and smoothly rounded at
t = pi, giving the bowl. Raising `taper` makes the point sharper.

Everything else follows the round brilliant: a planar girdle at z = 0, a planar
table above it scaled toward the outline centroid, and a pavilion converging to
a single apex. Returned in the same (vertices, faces) form so the mesh drops
into the existing pipeline unchanged.
"""
import numpy as np


def pear_outline(count, taper=1.0):
    """Girdle outline points, normalised so the maximum radius is 1."""
    # Skip t = 0 exactly: the curve is cusped there and a vertex sitting on the
    # cusp gives two coincident edges.
    t = np.linspace(0., 2.*np.pi, count, endpoint=False) + np.pi/count
    x = np.cos(t)
    y = np.sin(t)*np.abs(np.sin(t/2.))**taper
    points = np.stack([x, y], axis=1)
    return points/np.linalg.norm(points, axis=1).max()


def make_pear_brilliant(girdle_radius=1.0, crown_angle_deg=34.5,
                        pavilion_angle_deg=40.75, table_frac=0.56,
                        num_girdle_points=16, taper=1.0):
    """Watertight pear brilliant as (vertices, faces).

    `num_girdle_points` plays the role `num_main_facets` plays for the round
    cut: it sets how finely the girdle outline is sampled, and therefore the
    triangle count.
    """
    if num_girdle_points < 6:
        raise ValueError('num_girdle_points must be at least six')
    if not 0. < table_frac < 1.:
        raise ValueError('table_frac must lie strictly between zero and one')
    if taper <= 0.:
        raise ValueError('taper must be positive')

    outline = pear_outline(num_girdle_points, taper)*girdle_radius
    centroid = outline.mean(axis=0)
    mean_radius = float(np.linalg.norm(outline-centroid, axis=1).mean())

    # Planar girdle and table, as in a real cut; heights come from the mean
    # radius so the crown and pavilion angles hold on average around the stone.
    crown_height = mean_radius*(1.-table_frac)*np.tan(np.radians(crown_angle_deg))
    pavilion_depth = mean_radius*np.tan(np.radians(pavilion_angle_deg))

    girdle = np.column_stack([outline, np.zeros(num_girdle_points)])
    table_xy = centroid + table_frac*(outline-centroid)
    table = np.column_stack([table_xy, np.full(num_girdle_points, crown_height)])
    apex = np.array([centroid[0], centroid[1], -pavilion_depth])

    vertices = np.vstack([girdle, table, apex]).astype(np.float32)
    girdle_base, table_base, apex_index = 0, num_girdle_points, 2*num_girdle_points

    faces = []
    for i in range(num_girdle_points):
        j = (i+1) % num_girdle_points
        g0, g1 = girdle_base+i, girdle_base+j
        t0, t1 = table_base+i, table_base+j
        faces.append([g0, g1, t1])          # crown band
        faces.append([g0, t1, t0])
        faces.append([g1, g0, apex_index])  # pavilion
    for i in range(1, num_girdle_points-1):  # table cap, fanned
        faces.append([table_base, table_base+i, table_base+i+1])
    faces = np.array(faces, dtype=np.uint32)

    # Orient every triangle outward from the solid's centre.
    centre = vertices.mean(axis=0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = (normals*(triangles.mean(axis=1)-centre)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces
