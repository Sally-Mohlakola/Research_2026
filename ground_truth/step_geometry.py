"""Step cut (emerald cut) geometry, matching make_round_brilliant's conventions.

Built to the standard trade diagram: an octagonal outline, concentric
rectangular step facets on both crown and pavilion, and a small flat CULET
facet at the bottom.

A step cut differs from a brilliant in two structural ways, and both matter to
a transport model rather than only to the eye.

The outline is an octagon -- a rectangle with chamfered corners -- instead of a
circle or a teardrop, so the stone has four long parallel facet runs and four
corner runs rather than a ring of small triangular facets.

The facets are concentric terraces. Real step cuts vary the angle from one step
to the next -- steepest at the girdle on the crown, steepest at the culet on
the pavilion -- and without that variation adjacent bands would be coplanar and
the cut would read as a plain frustum. `step_angle_spread` controls it, and
the default is deliberately small: on a real step cut the terraces show as
lines on the face, not as bends in the silhouette. At 0.06 the mesh keeps 50
distinct facet planes with a straight-sided profile; at 0 it collapses to 18
and becomes a frustum, and at 0.35 the pavilion visibly funnels.

The pavilion tapers through its steps to a flat culet rather than to a point,
which is the same arrangement `make_round_brilliant` uses for `culet_radius`:
a small ring capped by a fan around a centre vertex, lying in one plane.

Overall proportions are held to the same envelope as the other cuts: the same
girdle radius, the same nominal crown height and pavilion depth from
`crown_angle_deg` and `pavilion_angle_deg`. Only the facet structure changes,
which is what makes a cross-cut comparison a controlled one.
"""
import numpy as np


def step_outline(length_ratio=1.35, corner_frac=0.20):
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
                  table_frac=0.56, length_ratio=1.35, corner_frac=0.20,
                  crown_steps=3, pavilion_steps=3, culet_frac=0.15,
                  step_angle_spread=0.06):
    """Watertight step cut as (vertices, faces).

    `crown_steps` and `pavilion_steps` set the tessellation the way
    `num_main_facets` does for the round cut. The mesh carries
    8*(crown_steps+pavilion_steps+1)+2 vertices and
    16*(crown_steps+pavilion_steps)+16 faces, so crown_steps+pavilion_steps == 3
    reproduces the round brilliant's 34 vertices and 64 faces exactly.

    `culet_frac` is the culet's width as a fraction of the girdle outline. It
    plays the role `culet_radius` plays for the round cut, but it cannot be
    zero here: collapsing the bottom ring to a point would change the face
    count and break the tessellation contract above.
    """
    if crown_steps < 1 or pavilion_steps < 1:
        raise ValueError('crown_steps and pavilion_steps must be at least one')
    if not 0. < table_frac < 1.:
        raise ValueError('table_frac must lie strictly between zero and one')
    if not 0. < culet_frac < 1.:
        raise ValueError('culet_frac must lie strictly between zero and one')
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
    # Pavilion: girdle down to the culet ring, steepest step at the culet.
    pavilion_scales = np.linspace(1., culet_frac, pavilion_steps+1)
    pavilion_heights = _cumulative(pavilion_scales,
                                   _band_angles(pavilion_angle_deg, pavilion_steps,
                                                step_angle_spread, False),
                                   mean_radius, pavilion_depth)

    def ring(scale, height):
        return np.column_stack([centroid+scale*(outline-centroid),
                                np.full(sides, height)])

    # Rings run top to bottom: table, crown intermediates, girdle, pavilion,
    # with the last pavilion ring being the culet.
    rings = [ring(table_frac, crown_height)]
    for k in range(crown_steps-1, 0, -1):
        rings.append(ring(crown_scales[k], crown_heights[k-1]))
    rings.append(ring(1., 0.))
    for i in range(pavilion_steps):
        rings.append(ring(pavilion_scales[i+1], -pavilion_heights[i]))

    # Flat caps get a centre vertex so table and culet are fans rather than
    # polygon triangulations, matching make_round_brilliant.
    table_centre = np.array([[centroid[0], centroid[1], crown_height]])
    culet_centre = np.array([[centroid[0], centroid[1], -pavilion_depth]])
    vertices = np.vstack(rings+[table_centre, culet_centre]).astype(np.float32)
    culet_base = (len(rings)-1)*sides
    table_apex, culet_apex = len(rings)*sides, len(rings)*sides+1

    faces = []
    for i in range(sides):                          # table cap
        faces.append([table_apex, i, (i+1) % sides])
    for r in range(len(rings)-1):                   # step bands
        upper, lower = r*sides, (r+1)*sides
        for i in range(sides):
            j = (i+1) % sides
            faces.append([upper+i, upper+j, lower+j])
            faces.append([upper+i, lower+j, lower+i])
    for i in range(sides):                          # culet cap
        faces.append([culet_apex, culet_base+i, culet_base+(i+1) % sides])
    faces = np.array(faces, dtype=np.uint32)

    centre = vertices.mean(axis=0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = (normals*(triangles.mean(axis=1)-centre)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces
