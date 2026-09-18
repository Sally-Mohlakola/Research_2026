"""Step cut (emerald cut) geometry, matching make_round_brilliant's conventions.

A step cut differs from a brilliant in two structural ways, and both matter to
a transport model rather than only to the eye.

The outline is an octagon -- a rectangle with chamfered corners -- instead of a
circle or a teardrop, so the stone has four long parallel facet runs rather
than a ring of small triangular facets.

The pavilion converges to a KEEL LINE, not to a point. Every other generator
here ends the pavilion at a single apex, so the closure below the last ring is
the one piece of topology a step cut cannot borrow from them: eight ring
vertices reduce to a two-vertex ridge, six triangles and two quads.

Facets are arranged in concentric steps. Real step cuts vary the angle from one
step to the next -- steepest at the girdle on the crown, steepest at the keel on
the pavilion -- and without that variation adjacent bands would be coplanar and
the cut would read as a plain frustum. `step_angle_spread` controls it.

Overall proportions are held to the same envelope as the other cuts: the same
girdle radius, the same nominal crown height and pavilion depth from
`crown_angle_deg` and `pavilion_angle_deg`. Only the facet structure changes,
which is what makes a cross-cut comparison a controlled one.
"""
import numpy as np


def step_outline(length_ratio=1.35, corner_frac=0.18):
    """Octagonal girdle outline, normalised so the maximum radius is 1.

    A rectangle of half-extents (length_ratio, 1) with each corner chamfered by
    `corner_frac` of the half-extent it cuts into. Returned counter-clockwise
    starting from the +x side, so index order matches the other generators.
    """
    if length_ratio < 1.:
        raise ValueError('length_ratio must be at least one')
    if not 0. < corner_frac < 0.5:
        raise ValueError('corner_frac must lie strictly between zero and one half')
    a, b = float(length_ratio), 1.
    ta, tb = corner_frac*a, corner_frac*b
    points = np.array([[a, b-tb], [a-ta, b], [-(a-ta), b], [-a, b-tb],
                       [-a, -(b-tb)], [-(a-ta), -b], [a-ta, -b], [a, -(b-tb)]])
    return points/np.linalg.norm(points, axis=1).max()


def _band_angles(nominal_deg, count, spread, steepest_first):
    """Per-step facet angles, fanning out symmetrically about the nominal."""
    if count == 1:
        return [np.radians(nominal_deg)]
    fraction = (np.arange(count)+.5)/count
    factor = 1.+spread*(1.-2.*fraction) if steepest_first else 1.+spread*(2.*fraction-1.)
    return [np.radians(nominal_deg*f) for f in factor]


def _cumulative(scales, angles, mean_radius, target):
    """Heights for a run of steps, rescaled to land exactly on `target`."""
    increments = [mean_radius*(scales[k]-scales[k+1])*np.tan(angles[k])
                  for k in range(len(angles))]
    total = float(sum(increments))
    if total <= 0.:
        raise ValueError('Degenerate step run')
    factor = target/total
    heights, running = [], 0.
    for increment in increments:
        running += increment*factor
        heights.append(running)
    return heights


def make_step_cut(girdle_radius=1.0, crown_angle_deg=34.5, pavilion_angle_deg=40.75,
                  table_frac=0.56, length_ratio=1.35, corner_frac=0.18,
                  crown_steps=3, pavilion_steps=3, keel_ring_frac=0.30,
                  keel_length_frac=0.18, step_angle_spread=0.35):
    """Watertight step cut as (vertices, faces).

    `crown_steps` and `pavilion_steps` set the tessellation the way
    `num_main_facets` does for the round cut. The mesh carries
    8*(crown_steps+pavilion_steps+1)+2 vertices and
    16*(crown_steps+pavilion_steps)+16 faces, so crown_steps+pavilion_steps == 3
    reproduces the round brilliant's 34 vertices and 64 faces exactly.
    """
    if crown_steps < 1 or pavilion_steps < 1:
        raise ValueError('crown_steps and pavilion_steps must be at least one')
    if not 0. < table_frac < 1.:
        raise ValueError('table_frac must lie strictly between zero and one')
    if not 0. < keel_ring_frac < 1.:
        raise ValueError('keel_ring_frac must lie strictly between zero and one')
    if step_angle_spread < 0. or step_angle_spread >= 1.:
        raise ValueError('step_angle_spread must lie in [0, 1)')

    outline = step_outline(length_ratio, corner_frac)*girdle_radius
    sides = len(outline)
    centroid = outline.mean(axis=0)
    mean_radius = float(np.linalg.norm(outline-centroid, axis=1).mean())
    crown_height = mean_radius*(1.-table_frac)*np.tan(np.radians(crown_angle_deg))
    pavilion_depth = mean_radius*np.tan(np.radians(pavilion_angle_deg))

    # Crown: girdle up to table, steepest step at the girdle.
    crown_scales = np.linspace(1., table_frac, crown_steps+1)
    crown_heights = _cumulative(crown_scales,
                                _band_angles(crown_angle_deg, crown_steps,
                                             step_angle_spread, True),
                                mean_radius, crown_height)
    # Pavilion: girdle down through the rings and on to the keel, which is the
    # scale-zero end of the same run, so the keel drop obeys the progression.
    pavilion_scales = np.concatenate([np.linspace(1., keel_ring_frac, pavilion_steps+1), [0.]])
    pavilion_heights = _cumulative(pavilion_scales,
                                   _band_angles(pavilion_angle_deg, pavilion_steps+1,
                                                step_angle_spread, False),
                                   mean_radius, pavilion_depth)

    def ring(scale, height):
        return np.column_stack([centroid+scale*(outline-centroid),
                                np.full(sides, height)])

    # Rings run top to bottom: table, crown intermediates, girdle, pavilion.
    rings = [ring(table_frac, crown_height)]
    for k in range(crown_steps-1, 0, -1):
        rings.append(ring(crown_scales[k], crown_heights[k-1]))
    rings.append(ring(1., 0.))
    for i in range(pavilion_steps):
        rings.append(ring(pavilion_scales[i+1], -pavilion_heights[i]))

    extent = float(np.abs(outline[:, 0]).max())
    keel_half = keel_length_frac*extent
    if keel_half >= keel_ring_frac*extent:
        raise ValueError('keel_length_frac must be smaller than keel_ring_frac')
    keel = np.array([[keel_half, centroid[1], -pavilion_depth],
                     [-keel_half, centroid[1], -pavilion_depth]])

    vertices = np.vstack(rings+[keel]).astype(np.float32)
    last = (len(rings)-1)*sides
    plus, minus = len(rings)*sides, len(rings)*sides+1

    faces = []
    for i in range(1, sides-1):                     # table cap, fanned
        faces.append([0, i, i+1])
    for r in range(len(rings)-1):                   # step bands
        upper, lower = r*sides, (r+1)*sides
        for i in range(sides):
            j = (i+1) % sides
            faces.append([upper+i, upper+j, lower+j])
            faces.append([upper+i, lower+j, lower+i])
    # Keel closure: each ring vertex belongs to the nearer keel end, so an edge
    # whose ends agree spans a triangle and an edge that straddles spans a quad.
    side = [plus if point[0] >= centroid[0] else minus for point in outline]
    for i in range(sides):
        j = (i+1) % sides
        a, b = last+i, last+j
        if side[i] == side[j]:
            faces.append([a, b, side[i]])
        else:
            faces.append([a, b, side[j]])
            faces.append([a, side[j], side[i]])
    faces = np.array(faces, dtype=np.uint32)

    centre = vertices.mean(axis=0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = (normals*(triangles.mean(axis=1)-centre)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces
