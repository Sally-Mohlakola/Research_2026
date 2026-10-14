import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import hashlib
import json
import math
import multiprocessing

import config 
import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("scalar_rgb")
from ground_truth.cuts import make_diamond
from config.parameters import get_diamond_parameters
from bsdf.dispersion import diamond_ior


# how every traced path ended
ESCAPED, TRUNCATED, INVALID = 0, 1, 2
NO_EXIT = ([np.nan] * 3, [np.nan] * 3, [np.nan] * 3, -1)


# build the diamond mesh and place it alone in a mitsuba scene
def build_scene(parameters):
    vertices, faces = make_diamond(parameters)
    mesh = mi.Mesh("learned_interior_diamond", len(vertices), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = vertices.ravel()
    params["faces"] = faces.ravel()
    params.update()
    scene = mi.load_dict({"type": "scene", "diamond": mesh})
    return scene, vertices, faces


# return the Fresnel probability and the exact local delta directions
def dielectric(si, eta):
    f, ct, _, eta_ti = mi.fresnel(float(si.wi.z), eta)
    return float(f), mi.reflect(si.wi), mi.refract(si.wi, ct, eta_ti), float(eta_ti)


# camera sampling: a primary ray from the studio camera, so entries match what a render sees
def camera_ray(rng, azimuth_range, fov=30.):
    low, high = azimuth_range
    a = math.radians(rng.uniform(low, high))
    origin = np.array([-1.8*math.sin(a), 1.8*math.cos(a), 4.])

    forward = -origin/np.linalg.norm(origin)
    right = np.cross(forward, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    extent = math.tan(math.radians(fov)/2.)
    px, py = rng.uniform(-1., 1., size=2)
    direction = forward + px*extent*right + py*extent*up
    direction /= np.linalg.norm(direction)
    return origin, direction


# uniform sampling with an aim offset: a random direction, shifted sideways across a disc radius the size of the stone
def uniform_ray(rng, radius):
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)

    axis = np.array([0., 0., 1.]) if abs(u[2]) < .9 else np.array([1., 0., 0.])
    tangent = np.cross(u, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(u, tangent)

    r, phi = radius * np.sqrt(rng.random()), 2 * np.pi * rng.random()
    origin = 3 * radius * u + r * (np.cos(phi)*tangent + np.sin(phi)*bitangent)
    return origin, -u


# keep firing rays until one hits the stone from outside, that hit is the entry
def sample_entry(scene, rng, radius, entry_sampling, azimuth_range, attempts, limit):
    while True:
        attempts += 1
        if attempts > limit:
            raise RuntimeError("Too many misses while sampling entry states")

        if entry_sampling == 'camera':
            origin, travel = camera_ray(rng, azimuth_range)
        else:
            origin, travel = uniform_ray(rng, radius)

        first = scene.ray_intersect(mi.Ray3f(mi.Point3f(origin), mi.Vector3f(travel)))
        if first.is_valid() and first.wi.z > 0:
            return first, travel, attempts


# follow one path inside the stone, reflecting or refracting by Fresnel probability at every facet, until it escapes
def trace_path(scene, rng, first, entry_t, eta_in, eta, max_depth):
    ray = first.spawn_ray(dr.normalize(first.to_world(entry_t)))
    weight = eta_in * eta_in
    log_probability = 0.
    depth = 1

    while depth < max_depth:
        si = scene.ray_intersect(ray)
        if not si.is_valid() or si.wi.z >= 0:
            return INVALID, depth, 0., log_probability, NO_EXIT

        f, reflected, transmitted, eta_out = dielectric(si, eta)
        choose_r = rng.random() < f
        probability = f if choose_r else 1. - f
        log_probability += np.log(max(probability, np.finfo(float).tiny))
        depth += 1

        if choose_r:
            ray = si.spawn_ray(dr.normalize(si.to_world(reflected)))
            continue

        weight *= eta_out * eta_out
        out = dr.normalize(si.to_world(transmitted))
        if scene.ray_intersect(si.spawn_ray(out)).is_valid():
            return INVALID, depth, 0., log_probability, NO_EXIT

        exit = (list(si.p), list(out), list(si.n), int(si.prim_index))
        return ESCAPED, depth, weight, log_probability, exit

    return TRUNCATED, depth, 0., log_probability, NO_EXIT


# one empty column per recorded quantity, filled row by row as paths are traced
def empty_columns(total):
    columns = {}
    for name in ('entry_id', 'repeat_id', 'entry_facet', 'exit_facet', 'depth', 'status'):
        columns[name] = np.empty(total, dtype=np.int64)
    for name in ('wavelength_nm', 'wavelength_pdf', 'entry_transmittance',
                 'throughput', 'branch_log_probability'):
        columns[name] = np.empty(total, dtype=np.float32)
    for name in ('entry_position', 'entry_normal', 'entry_direction', 'entry_local_wi',
                 'exit_position', 'exit_direction', 'exit_normal'):
        columns[name] = np.empty((total, 3), dtype=np.float32)
    return columns


# trace paths_per_entry paths from each of `entries` entry states, and record where each one leaves
def gather(scene, radius, parameters, entries, paths_per_entry, max_depth, seed,
           dispersion=True, entry_offset=0, seed_sequence=None,
           entry_sampling='uniform', azimuth_range=(0., 360.)):
    if entry_sampling not in ('uniform', 'camera'):
        raise ValueError("entry_sampling must be 'uniform' or 'camera'")
    if entries < 1 or paths_per_entry < 1 or max_depth < 2:
        raise ValueError("Require entries >= 1, paths_per_entry >= 1, max_depth >= 2")

    rng = np.random.default_rng(seed if seed_sequence is None else seed_sequence)
    columns = empty_columns(entries * paths_per_entry)
    row = 0
    attempts = 0

    for local_id in range(entries):
        first, travel, attempts = sample_entry(scene, rng, radius, entry_sampling,
                                               azimuth_range, attempts, entries * 10000)

        wavelength = float(rng.uniform(360., 830.))
        ior = float(diamond_ior(wavelength)) if dispersion else parameters['int_ior']
        eta = ior / parameters['ext_ior']

        entry_f, _, entry_t, eta_in = dielectric(first, eta)
        if entry_f >= 1.:
            raise RuntimeError("Degenerate fully reflecting entry state")

        for repeat in range(paths_per_entry):
            status, depth, throughput, log_probability, exit = trace_path(
                scene, rng, first, entry_t, eta_in, eta, max_depth)
            exit_position, exit_direction, exit_normal, exit_facet = exit

            columns['entry_id'][row] = entry_offset + local_id
            columns['repeat_id'][row] = repeat

            columns['entry_position'][row] = list(first.p)
            columns['entry_normal'][row] = list(first.n)
            columns['entry_direction'][row] = list(mi.Vector3f(travel))
            columns['entry_local_wi'][row] = list(first.wi)
            columns['entry_facet'][row] = int(first.prim_index)
            columns['entry_transmittance'][row] = 1. - entry_f

            columns['wavelength_nm'][row] = wavelength
            columns['wavelength_pdf'][row] = 1. / 470.

            columns['exit_position'][row] = exit_position
            columns['exit_direction'][row] = exit_direction
            columns['exit_normal'][row] = exit_normal
            columns['exit_facet'][row] = exit_facet

            columns['throughput'][row] = throughput
            columns['depth'][row] = depth
            columns['status'][row] = status
            columns['branch_log_probability'][row] = log_probability
            row += 1

    return columns, attempts


# split the entries into one chunk per worker, the first chunks take the remainder
def _chunk_bounds(entries, workers):
    base, extra = divmod(entries, workers)
    bounds, start = [], 0
    for index in range(workers):
        count = base + (1 if index < extra else 0)
        if count:
            bounds.append((start, count))
            start += count
    return bounds


# each worker builds its own scene, since mitsuba objects can't be sent between processes
def _worker(task):
    (diamond_name, offset, count, paths_per_entry, max_depth, child, dispersion,
     entry_sampling, azimuth_range) = task

    parameters = get_diamond_parameters(diamond_name)
    scene, vertices, _ = build_scene(parameters)
    radius = float(np.linalg.norm(vertices, axis=1).max())

    return gather(scene, radius, parameters, count, paths_per_entry, max_depth,
                  None, dispersion, entry_offset=offset, seed_sequence=child,
                  entry_sampling=entry_sampling, azimuth_range=azimuth_range)


# run the gather across several processes, reproducible only with the same seed and worker count
def gather_parallel(diamond_name, entries, paths_per_entry, max_depth, seed,
                    dispersion, workers, entry_sampling='uniform',
                    azimuth_range=(0., 360.)):
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


# reading the options, gather the paths, and save them with everything needed to reproduce them
def main():
    parser = argparse.ArgumentParser(description='Gather true interior transport paths '
                                                 'to train the learned interior operator.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diamond_name', default='round_diamond_gia')
    parser.add_argument('--entries', type=int, default=1024)
    parser.add_argument('--paths_per_entry', type=int, default=8)
    parser.add_argument('--max_depth', type=int, default=128)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--no_dispersion', action='store_true')
    parser.add_argument('--entry_sampling', choices=['uniform', 'camera'], default='uniform',
                        help='uniform sampling spreads entries over incoming directions; '
                             'camera draws them from what a render actually sees')
    parser.add_argument('--azimuth_range', type=float, nargs=2, default=[0., 360.],
                        metavar=('LOW', 'HIGH'),
                        help='camera azimuth range for --entry_sampling camera')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel processes; reproducing a pool needs the same value')
    args = parser.parse_args()

    if args.workers < 1:
        parser.error('--workers must be at least one')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace {args.output}')

    params = get_diamond_parameters(args.diamond_name)
    scene, vertices, faces = build_scene(params)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    workers = min(args.workers, args.entries)

    if workers > 1:
        data, attempts = gather_parallel(args.diamond_name, args.entries,
                                         args.paths_per_entry, args.max_depth,
                                         args.seed, not args.no_dispersion, workers,
                                         args.entry_sampling, tuple(args.azimuth_range))
    else:
        data, attempts = gather(scene, radius, params, args.entries, args.paths_per_entry,
                                args.max_depth, args.seed, not args.no_dispersion,
                                entry_sampling=args.entry_sampling,
                                azimuth_range=tuple(args.azimuth_range))

    if args.entry_sampling == 'uniform':
        sampling_detail = 'uniform sphere direction and uniform projected bounding disc, conditioned on hit'
    else:
        sampling_detail = ('primary studio camera rays over azimuth range %s, conditioned on hit; '
                           'secondary rays after ground bounce or first-surface reflection are not covered'
                           % (tuple(args.azimuth_range),))

    metadata = dict(
        schema_version=1,
        diamond_name=args.diamond_name,
        parameters=params,
        seed=args.seed,
        max_depth=args.max_depth,
        dispersion=not args.no_dispersion,
        attempts=attempts,
        workers=workers,
        reproducibility='entry streams are sectioned by worker count; '
                        'reproduce with the same seed AND workers',
        geometry_sha256=hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
        coordinates='object_space',
        direction_convention='entry points into stone, and the exit points out',
        transport_mode='radiance',
        conditioning='entry transmission forced; multiply entry_transmittance once',
        status_codes={'escaped': ESCAPED, 'truncated': TRUNCATED, 'invalid': INVALID},
        entry_sampling=args.entry_sampling,
        entry_sampling_detail=sampling_detail,
        azimuth_range=list(args.azimuth_range),
        pdf_note='branch_log_probability is conditional discrete path probability, NOT exit density',
        normalization='throughput already contains physical/proposal branch ratios; '
                      'do not divide by branch probability again',
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data, vertices=vertices, faces=faces,
                        metadata_json=np.array(json.dumps(metadata)))

    status = data['status']
    print(f"Saved {len(status)} paths: {np.sum(status == ESCAPED)} escaped, "
          f"{np.sum(status == TRUNCATED)} truncated, {np.sum(status == INVALID)} invalid "
          f"-> {args.output}")


if __name__ == '__main__':
    main()
