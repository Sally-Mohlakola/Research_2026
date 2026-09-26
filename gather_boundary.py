"""Gather spatial, monochromatic boundary transport using scalar Mitsuba.

Entry reflection is excluded analytically. Samples describe transport conditional
on entry transmission. No lighting is baked in; weights are radiance-mode
throughput, not radiance or a continuous exit PDF. Geometry remains object-space.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import config  # resolve runtime before importing Mitsuba
import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("scalar_rgb")
from ground_truth.cuts import make_diamond
from config.parameters import get_diamond_parameters
from bsdf.dispersion import diamond_ior


def build_scene(parameters):
    vertices, faces = make_diamond(parameters)
    mesh = mi.Mesh("boundary_diamond", len(vertices), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = vertices.ravel()
    params["faces"] = faces.ravel()
    params.update()
    scene = mi.load_dict({"type": "scene", "diamond": mesh})
    return scene, vertices, faces


def dielectric(si, eta):
    """Return Fresnel probability and exact local delta directions."""
    f, ct, _, eta_ti = mi.fresnel(float(si.wi.z), eta)
    return float(f), mi.reflect(si.wi), mi.refract(si.wi, ct, eta_ti), float(eta_ti)


def camera_ray(rng, azimuth_range, fov=30.):
    """A primary camera ray from the studio rig, matching render_boundary.make_scene.

    Uniform sphere sampling spreads entry states over every incoming direction,
    but a renderer only queries the narrow cone its camera sees: at azimuth 0
    with a 30 degree field of view, under 3 percent of uniformly sampled entry
    directions fall within 15 degrees of the camera axis. Training density in the
    region that actually gets queried is therefore far lower than the nominal
    entry count suggests. This sampler draws entry states from the query
    distribution instead.

    Only primary camera rays are reproduced. Secondary rays, after a ground
    bounce or a first-surface reflection, reach the stone from wider angles and
    are not covered here.
    """
    low, high = azimuth_range
    a = math.radians(rng.uniform(low, high))
    origin = np.array([-1.8*math.sin(a), 1.8*math.cos(a), 4.])
    forward = -origin/np.linalg.norm(origin)
    right = np.cross(forward, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    extent = math.tan(math.radians(fov)/2.)
    # Film point in [-1, 1]^2; the rig uses a square film.
    px, py = rng.uniform(-1., 1., size=2)
    direction = forward + px*extent*right + py*extent*up
    direction /= np.linalg.norm(direction)
    return origin, direction


def gather(scene, radius, parameters, entries, paths_per_entry, max_depth, seed,
           dispersion=True, entry_offset=0, seed_sequence=None,
           entry_sampling='uniform', azimuth_range=(0., 360.)):
    """Trace paths_per_entry paths from each of `entries` sampled entry states.

    `entry_offset` shifts the emitted entry_id range so parallel chunks stay
    globally distinct, and `seed_sequence` replaces `seed` for a chunked stream.
    `entry_sampling` selects uniform-sphere entry states or the camera query
    distribution. Defaults reproduce the original single-process behaviour
    exactly.
    """
    if entry_sampling not in ('uniform', 'camera'):
        raise ValueError("entry_sampling must be 'uniform' or 'camera'")
    if entries < 1 or paths_per_entry < 1 or max_depth < 2:
        raise ValueError("Require entries >= 1, paths_per_entry >= 1, max_depth >= 2")
    rng = np.random.default_rng(seed if seed_sequence is None else seed_sequence)
    # Preallocated columns rather than one dict per path: a dict of Python
    # objects costs about 1.2 KB per record against 152 bytes here, which is
    # what made fifty-million-record gathers impossible in memory. Rows are
    # filled in the order paths were appended before, and values are rounded
    # to float32 on assignment exactly as np.asarray rounded them afterwards,
    # so output is bit-identical to the dict-based version for the same seed.
    total = entries*paths_per_entry
    arrays = {}
    for name in ('entry_id', 'repeat_id'):
        arrays[name] = np.empty(total, dtype=np.int64)
    for name in ('entry_position', 'entry_normal', 'entry_direction', 'entry_local_wi'):
        arrays[name] = np.empty((total, 3), dtype=np.float32)
    arrays['entry_facet'] = np.empty(total, dtype=np.int64)
    for name in ('wavelength_nm', 'wavelength_pdf', 'entry_transmittance'):
        arrays[name] = np.empty(total, dtype=np.float32)
    for name in ('exit_position', 'exit_direction', 'exit_normal'):
        arrays[name] = np.empty((total, 3), dtype=np.float32)
    arrays['exit_facet'] = np.empty(total, dtype=np.int64)
    arrays['throughput'] = np.empty(total, dtype=np.float32)
    arrays['depth'] = np.empty(total, dtype=np.int64)
    arrays['status'] = np.empty(total, dtype=np.int64)
    arrays['branch_log_probability'] = np.empty(total, dtype=np.float32)
    row = 0
    attempts = 0
    for local_id in range(entries):
        entry_id = entry_offset + local_id
        while True:
            attempts += 1
            if attempts > entries * 10000:
                raise RuntimeError("Too many misses while sampling entry states")
            if entry_sampling == 'camera':
                origin, travel = camera_ray(rng, azimuth_range)
            else:
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                axis = np.array([0., 0., 1.]) if abs(u[2]) < .9 else np.array([1., 0., 0.])
                tangent = np.cross(u, axis)
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(u, tangent)
                r, phi = radius * np.sqrt(rng.random()), 2 * np.pi * rng.random()
                origin = 3 * radius * u + r * (np.cos(phi)*tangent + np.sin(phi)*bitangent)
                travel = -u
            ray = mi.Ray3f(mi.Point3f(origin), mi.Vector3f(travel))
            first = scene.ray_intersect(ray)
            if first.is_valid() and first.wi.z > 0:
                break
        wavelength = float(rng.uniform(360., 830.))
        ior = (float(diamond_ior(wavelength))
               if dispersion else parameters['int_ior'])
        eta = ior / parameters['ext_ior']
        entry_f, _, entry_t, eta_in = dielectric(first, eta)
        if entry_f >= 1.:
            raise RuntimeError("Degenerate fully reflecting entry state")
        for repeat in range(paths_per_entry):
            si = first
            ray = first.spawn_ray(dr.normalize(first.to_world(entry_t)))
            weight = eta_in * eta_in
            log_probability = 0.
            status = 1  # depth truncated
            depth = 1
            exit_position = [np.nan] * 3
            exit_direction = [np.nan] * 3
            exit_normal = [np.nan] * 3
            exit_facet = -1
            while depth < max_depth:
                si = scene.ray_intersect(ray)
                if not si.is_valid():
                    status = 2  # numerical/geometry failure: do not label as escape
                    break
                if si.wi.z >= 0:
                    status = 2
                    break
                f, reflected, transmitted, eta_out = dielectric(si, eta)
                choose_r = rng.random() < f
                probability = f if choose_r else 1. - f
                log_probability += np.log(max(probability, np.finfo(float).tiny))
                depth += 1
                if choose_r:
                    ray = si.spawn_ray(dr.normalize(si.to_world(reflected)))
                else:
                    weight *= eta_out * eta_out
                    out = dr.normalize(si.to_world(transmitted))
                    # For the closed stone, confirm that this is a true exit.
                    if scene.ray_intersect(si.spawn_ray(out)).is_valid():
                        status = 2
                        break
                    status = 0
                    exit_position = list(si.p)
                    exit_direction = list(out)
                    exit_normal = list(si.n)
                    exit_facet = int(si.prim_index)
                    break
            arrays['entry_id'][row] = entry_id
            arrays['repeat_id'][row] = repeat
            arrays['entry_position'][row] = list(first.p)
            arrays['entry_normal'][row] = list(first.n)
            arrays['entry_direction'][row] = list(mi.Vector3f(travel))
            arrays['entry_local_wi'][row] = list(first.wi)
            arrays['entry_facet'][row] = int(first.prim_index)
            arrays['wavelength_nm'][row] = wavelength
            arrays['wavelength_pdf'][row] = 1./470.
            arrays['entry_transmittance'][row] = 1.-entry_f
            arrays['exit_position'][row] = exit_position
            arrays['exit_direction'][row] = exit_direction
            arrays['exit_normal'][row] = exit_normal
            arrays['exit_facet'][row] = exit_facet
            arrays['throughput'][row] = weight if status == 0 else 0.
            arrays['depth'][row] = depth
            arrays['status'][row] = status
            arrays['branch_log_probability'][row] = log_probability
            row += 1
    return arrays, attempts


def _chunk_bounds(entries, workers):
    """Contiguous (offset, count) entry ranges; earlier chunks absorb the remainder."""
    base, extra = divmod(entries, workers)
    bounds, start = [], 0
    for index in range(workers):
        count = base + (1 if index < extra else 0)
        if count:
            bounds.append((start, count))
            start += count
    return bounds


def _worker(task):
    """Build an independent scalar scene per process; Mitsuba objects do not pickle."""
    (diamond_name, offset, count, paths_per_entry, max_depth, child, dispersion,
     entry_sampling, azimuth_range) = task
    parameters = get_diamond_parameters(diamond_name)
    scene, vertices, _ = build_scene(parameters)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    return gather(scene, radius, parameters, count, paths_per_entry, max_depth,
                  None, dispersion, entry_offset=offset, seed_sequence=child,
                  entry_sampling=entry_sampling, azimuth_range=azimuth_range)


def gather_parallel(diamond_name, entries, paths_per_entry, max_depth, seed,
                    dispersion, workers, entry_sampling='uniform',
                    azimuth_range=(0., 360.)):
    """Chunked gather. Deterministic given (seed, workers, entries, paths_per_entry).

    Changing the worker count repartitions the random streams and therefore
    changes the sampled entry states, so reproduction requires the same value.
    """
    bounds = _chunk_bounds(entries, workers)
    children = np.random.SeedSequence(seed).spawn(len(bounds))
    tasks = [(diamond_name, offset, count, paths_per_entry, max_depth, child,
              dispersion, entry_sampling, azimuth_range)
             for (offset, count), child in zip(bounds, children)]
    with multiprocessing.Pool(len(bounds)) as pool:
        results = pool.map(_worker, tasks)
    merged = {key: np.concatenate([chunk[key] for chunk, _ in results])
              for key in results[0][0]}
    return merged, sum(attempts for _, attempts in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diamond_name', default='round_diamond_gia')
    parser.add_argument('--entries', type=int, default=1024)
    parser.add_argument('--paths_per_entry', type=int, default=8)
    parser.add_argument('--max_depth', type=int, default=128)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--no_dispersion', action='store_true')
    parser.add_argument('--entry_sampling', choices=['uniform', 'camera'],
                        default='uniform',
                        help='uniform spreads entry states over every incoming '
                             'direction; camera draws them from the distribution a '
                             'render actually queries. Under 3 percent of uniform '
                             'entry states fall within the camera cone, so a render '
                             'sees far lower training density than the entry count '
                             'suggests.')
    parser.add_argument('--azimuth_range', type=float, nargs=2, default=[0., 360.],
                        metavar=('LOW', 'HIGH'),
                        help='Camera azimuth range for --entry_sampling camera.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel gather processes. The stream partition '
                             'depends on this count, so reproducing a dataset '
                             'requires the same --workers value.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least one')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace {args.output}')
    params = get_diamond_parameters(args.diamond_name)
    scene, vertices, faces = build_scene(params)
    workers = min(args.workers, args.entries)
    if workers > 1:
        data, attempts = gather_parallel(args.diamond_name, args.entries,
                                         args.paths_per_entry, args.max_depth,
                                         args.seed, not args.no_dispersion, workers,
                                         args.entry_sampling, tuple(args.azimuth_range))
    else:
        data, attempts = gather(scene, float(np.linalg.norm(vertices, axis=1).max()),
                               params, args.entries, args.paths_per_entry,
                               args.max_depth, args.seed, not args.no_dispersion,
                               entry_sampling=args.entry_sampling,
                               azimuth_range=tuple(args.azimuth_range))
    metadata = dict(schema_version=1, diamond_name=args.diamond_name,
                    parameters=params, seed=args.seed, max_depth=args.max_depth,
                    dispersion=not args.no_dispersion, attempts=attempts,
                    workers=workers,
                    reproducibility='entry streams are partitioned by worker count; '
                                    'reproduce with the same seed AND workers',
                    geometry_sha256=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest(),
                    coordinates='object_space', direction_convention='entry points into stone; exit points out',
                    transport_mode='radiance', conditioning='entry transmission forced; multiply entry_transmittance once',
                    status_codes={'escaped':0, 'truncated':1, 'invalid':2},
                    entry_sampling=args.entry_sampling,
                    entry_sampling_detail=(
                        'uniform sphere direction and uniform projected bounding disc, conditioned on hit'
                        if args.entry_sampling == 'uniform' else
                        'primary studio camera rays over azimuth range %s, conditioned on hit; '
                        'secondary rays after ground bounce or first-surface reflection are not covered'
                        % (tuple(args.azimuth_range),)),
                    azimuth_range=list(args.azimuth_range),
                    pdf_note='branch_log_probability is conditional discrete path probability, NOT exit density',
                    normalization='throughput already contains physical/proposal branch ratios; do not divide by branch probability again')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data, vertices=vertices, faces=faces,
                        metadata_json=np.array(json.dumps(metadata)))
    print(f"Saved {len(data['status'])} paths: "
          f"{np.sum(data['status']==0)} escaped, {np.sum(data['status']==1)} truncated, "
          f"{np.sum(data['status']==2)} invalid -> {args.output}")


if __name__ == '__main__':
    main()
