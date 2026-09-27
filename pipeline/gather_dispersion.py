"""Paired-wavelength boundary gather: the go/no-go pilot for learned dispersion.

The boundary operator ignores wavelength (conditioning ablation), because the
colour-dependent part of an exit -- a few degrees -- is far below the operator's
own exit error. The proposed fix learns that part separately: a small head
predicting how an exit shifts with wavelength, given the route the light took.
That only works if the colours of one path mostly take the same route and
differ by a small, smooth shift. This gather measures whether they do.

Each entry state is traced once per wavelength. The hero wavelength samples
its reflect/refract choices; the other lanes replay them and carry the ratio of
branch probabilities, as a hero-wavelength renderer does, so any route split
recorded here is geometric rather than an artefact of sampling.
Wavelengths are either a hero drawn uniformly on [360, 830] nm plus its
rotations by a quarter of the range, the way a spectral renderer assigns four
lanes (`rotated`, used for the pilot), or independent uniform draws
(`independent`, the default), which cover every wavelength gap and are what
train_dispersion.py needs: rotated lanes only ever differ by multiples of
117.5 nm.

The pilot in checkpoints/dispersion_pilot/pear_100k was gathered before the
`independent` mode existed; reproduce it with --wavelengths rotated --seed 31
--entries 100000 --workers 4 (its archive also predates the entry position and
normal fields).

Per path and wavelength it records the exit (facet, position, direction), the
status and a hash of the full facet sequence, which defines the route. Nothing
here touches gather_boundary.py or its datasets.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the repo root, for config/, neural/ and the stage folders
import argparse
import hashlib
import json
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np

from pipeline.gather_boundary import build_scene, camera_ray, dielectric
from config.parameters import get_diamond_parameters
from bsdf.dispersion import diamond_ior

import mitsuba as mi
import drjit as dr

LOW, HIGH = 360., 830.


def lane_wavelengths(hero, lanes):
    """Hero plus rotations by (HIGH-LOW)/lanes, wrapped into [LOW, HIGH)."""
    span = HIGH - LOW
    return LOW + np.mod(hero - LOW + np.arange(lanes)*span/lanes, span)


def draw_wavelengths(rng, lanes, mode):
    if mode == 'rotated':
        return lane_wavelengths(rng.uniform(LOW, HIGH), lanes)
    return rng.uniform(LOW, HIGH, size=lanes)


def entry_state(scene, radius, rng, sampling):
    """One entry ray that hits the stone from outside, as gather_boundary samples it."""
    while True:
        mode = sampling if sampling != 'mixed' else ('camera' if rng.random() < .5 else 'uniform')
        if mode == 'camera':
            origin, travel = camera_ray(rng, (0., 360.))
        else:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            axis = np.array([0., 0., 1.]) if abs(u[2]) < .9 else np.array([1., 0., 0.])
            tangent = np.cross(u, axis); tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(u, tangent)
            r, phi = radius*np.sqrt(rng.random()), 2*np.pi*rng.random()
            origin = 3*radius*u + r*(np.cos(phi)*tangent + np.sin(phi)*bitangent)
            travel = -u
        first = scene.ray_intersect(mi.Ray3f(mi.Point3f(origin), mi.Vector3f(travel)))
        if first.is_valid() and first.wi.z > 0:
            return first, travel, mode


def trace(scene, first, eta, uniforms=None, decisions=None):
    """Transmitted journey at one wavelength.

    The hero lane decides reflect/transmit with `uniforms`; the other lanes
    replay the hero's `decisions` and carry the probability ratio instead, as a
    hero-wavelength renderer would. Letting each lane decide for itself would
    split routes wherever a shared uniform falls between two lanes' Fresnel
    probabilities -- a sampling artefact, not dispersion. Replaying leaves only
    geometric divergence: a different facet hit, or a transmission that is
    impossible at this wavelength (total internal reflection), status 3.

    Returns status (0 escaped, 1 truncated, 2 invalid, 3 blocked), exit facet,
    position, direction, route hash, the decisions made and the log ratio of
    this lane's branch probability to the hero's.
    """
    _, _, entry_t, _ = dielectric(first, eta)
    ray = first.spawn_ray(dr.normalize(first.to_world(entry_t)))
    route = [int(first.prim_index)]
    made, log_probability = [], 0.
    steps = len(uniforms) if decisions is None else len(decisions)
    for j in range(steps):
        si = scene.ray_intersect(ray)
        if not si.is_valid() or si.wi.z >= 0:
            return 2, -1, None, None, 0, made, log_probability
        route.append(int(si.prim_index))
        f, reflected, transmitted, _ = dielectric(si, eta)
        reflect = uniforms[j] < f if decisions is None else decisions[j]
        probability = f if reflect else 1. - f
        if probability <= 0.:
            return 3, -1, None, None, 0, made, log_probability
        log_probability += math.log(probability)
        made.append(reflect)
        if reflect:
            ray = si.spawn_ray(dr.normalize(si.to_world(reflected)))
            continue
        out = dr.normalize(si.to_world(transmitted))
        if scene.ray_intersect(si.spawn_ray(out)).is_valid():
            return 2, -1, None, None, 0, made, log_probability
        digest = hashlib.blake2b(np.asarray(route, np.int32).tobytes(), digest_size=8)
        return (0, int(si.prim_index), list(si.p), list(out),
                int.from_bytes(digest.digest(), 'little', signed=True), made, log_probability)
    return 1, -1, None, None, 0, made, log_probability


def gather(diamond_name, entries, lanes, max_depth, seed_sequence, sampling,
           wavelength_mode='independent'):
    parameters = get_diamond_parameters(diamond_name)
    scene, vertices, _ = build_scene(parameters)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    rng = np.random.default_rng(seed_sequence)
    out = dict(wavelength_nm=np.empty((entries, lanes), np.float32),
               status=np.empty((entries, lanes), np.int8),
               exit_facet=np.empty((entries, lanes), np.int32),
               exit_position=np.full((entries, lanes, 3), np.nan, np.float32),
               exit_direction=np.full((entries, lanes, 3), np.nan, np.float32),
               route=np.empty((entries, lanes), np.int64),
               log_ratio=np.zeros((entries, lanes), np.float32),
               depth=np.zeros(entries, np.int16),
               entry_direction=np.empty((entries, 3), np.float32),
               entry_position=np.empty((entries, 3), np.float32),
               entry_normal=np.empty((entries, 3), np.float32),
               entry_facet=np.empty(entries, np.int32),
               camera_entry=np.empty(entries, bool))
    for i in range(entries):
        first, travel, mode = entry_state(scene, radius, rng, sampling)
        uniforms = rng.random(max_depth - 1)
        wavelengths = draw_wavelengths(rng, lanes, wavelength_mode)
        out['wavelength_nm'][i] = wavelengths
        out['entry_direction'][i] = travel
        out['entry_position'][i] = list(first.p)
        out['entry_normal'][i] = list(first.n)
        out['entry_facet'][i] = first.prim_index
        out['camera_entry'][i] = mode == 'camera'
        hero_decisions = hero_log = None
        for k, wavelength in enumerate(wavelengths):
            eta = float(diamond_ior(float(wavelength)))/parameters['ext_ior']
            if k == 0:
                status, facet, position, direction, route, hero_decisions, hero_log = trace(
                    scene, first, eta, uniforms=uniforms)
            else:
                status, facet, position, direction, route, _, log_p = trace(
                    scene, first, eta, decisions=hero_decisions)
                out['log_ratio'][i, k] = log_p - hero_log
            if k == 0:
                out['depth'][i] = len(hero_decisions)
            out['status'][i, k], out['exit_facet'][i, k], out['route'][i, k] = status, facet, route
            if status == 0:
                out['exit_position'][i, k], out['exit_direction'][i, k] = position, direction
    return out


def _worker(task):
    return gather(*task)


def summarise(data, faces_normal):
    """The go/no-go numbers."""
    status, route, facet = data['status'], data['route'], data['exit_facet']
    direction, wavelength = data['exit_direction'], data['wavelength_nm']
    hero_escaped = status[:, 0] == 0
    lanes = status.shape[1]
    report = dict(entries=int(len(status)), hero_escaped=float(hero_escaped.mean()))
    same_all = hero_escaped & (status == 0).all(1) & (route == route[:, :1]).all(1)
    report['all_lanes_same_route'] = float(same_all[hero_escaped].mean())
    pairs = []
    for k in range(1, lanes):
        a, b = 0, k
        both = hero_escaped
        same = both & (status[:, b] == 0) & (route[:, b] == route[:, a])
        lost = both & (status[:, b] != 0)
        blocked = both & (status[:, b] == 3)
        other_facet = both & (status[:, b] == 0) & (facet[:, b] != facet[:, a])
        same_facet_new_route = both & (status[:, b] == 0) & (facet[:, b] == facet[:, a]) & ~same
        cosine = np.clip((direction[:, a]*direction[:, b]).sum(-1), -1, 1)
        angle = np.degrees(np.arccos(cosine[same]))
        dl = np.abs(wavelength[:, b] - wavelength[:, a])
        pairs.append(dict(
            lane=k, mean_delta_nm=float(dl[both].mean()),
            same_route=float(same[both].mean()),
            other_lane_not_escaping=float(lost[both].mean()),
            other_lane_total_internal_reflection=float(blocked[both].mean()),
            weight_ratio_same_route_median=float(np.median(np.exp(data['log_ratio'][same, b]))),
            weight_ratio_same_route_p01_p99=[float(v) for v in np.percentile(
                np.exp(data['log_ratio'][same, b]), [1, 99])],
            different_exit_facet=float(other_facet[both].mean()),
            same_facet_different_route=float(same_facet_new_route[both].mean()),
            shift_deg_median=float(np.median(angle)), shift_deg_p90=float(np.percentile(angle, 90)),
            shift_deg_mean=float(angle.mean()),
            shift_deg_per_100nm_median=float(np.median(angle/np.maximum(dl[same], 1e-6)*100))))
    report['pairs'] = pairs
    # Same route as a function of how far apart the colours are.
    dl = np.abs(wavelength[:, 1:] - wavelength[:, :1])
    same = (status[:, 1:] == 0) & (route[:, 1:] == route[:, :1]) & hero_escaped[:, None]
    edges = np.arange(0, 480, 60)
    report['same_route_by_delta_nm'] = {
        '%d-%d' % (lo, lo+60): float(same[(dl >= lo) & (dl < lo+60) & hero_escaped[:, None]].mean())
        for lo in edges if ((dl >= lo) & (dl < lo+60) & hero_escaped[:, None]).any()}
    # Every escaped pair, whatever its route: how far apart the colours leave.
    both = hero_escaped[:, None] & (status[:, 1:] == 0)
    cosine = np.clip((direction[:, 1:]*direction[:, :1]).sum(-1), -1, 1)
    angle = np.degrees(np.arccos(cosine))
    split = both & (route[:, 1:] != route[:, :1])
    same_pair = both & ~split
    report['exit_separation_deg'] = dict(
        same_route_median=float(np.median(angle[same_pair])),
        split_route_median=float(np.median(angle[split])),
        split_route_p90=float(np.percentile(angle[split], 90)))
    # Route agreement against the number of internal interactions.
    depth = data['depth']
    report['all_lanes_same_route_by_interactions'] = {
        '%d-%d' % (lo, hi): dict(share_of_paths=float((hero_escaped & (depth >= lo) & (depth < hi)).mean()
                                                      / hero_escaped.mean()),
                                 same_route=float(same_all[hero_escaped & (depth >= lo) & (depth < hi)].mean()))
        for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 129))
        if (hero_escaped & (depth >= lo) & (depth < hi)).any()}
    # Split by entry sampling.
    for name, sel in (('camera', data['camera_entry']), ('uniform', ~data['camera_entry'])):
        h = hero_escaped & sel
        if h.any():
            report['all_lanes_same_route_' + name] = float(same_all[h].mean())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diamond_name', default='pear_brilliant')
    parser.add_argument('--entries', type=int, default=20000)
    parser.add_argument('--lanes', type=int, default=4)
    parser.add_argument('--max_depth', type=int, default=128)
    parser.add_argument('--entry_sampling', choices=['uniform', 'camera', 'mixed'], default='mixed')
    parser.add_argument('--wavelengths', choices=['independent', 'rotated'],
                        default='independent',
                        help='independent uniform lanes (training) or a hero plus '
                             'quarter-range rotations (the pilot)')
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        raise FileExistsError('Refusing to replace %s' % args.output)

    base, extra = divmod(args.entries, args.workers)
    counts = [base + (i < extra) for i in range(args.workers) if base + (i < extra)]
    children = np.random.SeedSequence(args.seed).spawn(len(counts))
    began = time.perf_counter()
    with multiprocessing.Pool(len(counts)) as pool:
        chunks = pool.map(_worker, [(args.diamond_name, n, args.lanes, args.max_depth, c,
                                     args.entry_sampling, args.wavelengths)
                                    for n, c in zip(counts, children)])
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    seconds = time.perf_counter() - began

    parameters = get_diamond_parameters(args.diamond_name)
    _, vertices, faces = build_scene(parameters)
    report = summarise(data, None)
    report.update(seconds=seconds, diamond_name=args.diamond_name, lanes=args.lanes,
                  wavelengths=args.wavelengths,
                  entry_sampling=args.entry_sampling, seed=args.seed, workers=args.workers)
    metadata = dict(schema_version=1, diamond_name=args.diamond_name, parameters=parameters,
                    lanes=args.lanes, max_depth=args.max_depth, seed=args.seed,
                    workers=args.workers, entry_sampling=args.entry_sampling,
                    wavelengths=('hero uniform on [360, 830] nm plus rotations by a quarter range'
                                 if args.wavelengths == 'rotated' else
                                 'every lane independent uniform on [360, 830] nm; lane 0 is the hero'),
                    wavelength_mode=args.wavelengths,
                    shared_randomness='hero lane samples reflect/transmit; other lanes replay '
                                      'those decisions and store log_ratio of branch probabilities',
                    route='blake2b hash of entry facet then every facet hit, exit included',
                    status_codes={'escaped': 0, 'truncated': 1, 'invalid': 2,
                                  'blocked_total_internal_reflection': 3},
                    geometry_sha256=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data, vertices=vertices, faces=faces,
                        metadata_json=np.array(json.dumps(metadata)))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
