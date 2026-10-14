import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import hashlib
import json
import math
import time

import config
import mitsuba as mi
import drjit as dr
import numpy as np
import torch

mi.set_variant('scalar_spectral')
from neural.learned_interior_model import load_model
from neural.dispersion_model import hero_coordinates, load_dispersion
from bsdf.dispersion import diamond_ior
from utils.studio_env import studio_lighting, display_exposure


LIMITATIONS = ['Experimental learned transport; RDM proposal preserves the learned target, not physical ground truth.',
               'BSDF-only scene tracing; finite depth; no MIS or next-event estimation.',
               'Same seeds do not ensure identical paths between modes.',
               'Predicted exits reentering the stone are rejected and counted.']


# object-to-world rotation about X, then Y, then Z, matching eval.py's tumble (including its Y sign convention)
def tumble_matrix(angle_x, angle_y, angle_z):
    ca, sa = math.cos(angle_x), math.sin(angle_x)
    cb, sb = math.cos(angle_y), math.sin(angle_y)
    cc, sc = math.cos(angle_z), math.sin(angle_z)

    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float64)


# where the camera sits: studio looks down onto the table (the training view), spin is ground_truth/diamond.py's side view
# spin keeps diamond.py's line of sight but sits at 0.8 of its distance with a wider lens, since the studio rig's side_light_2 panel now stands where diamond.py's camera was
CAMERAS = ('studio', 'spin')


# camera origin, target, up, field of view, and ground height and size, for each camera
def camera_setup(camera, azimuth):
    a = math.radians(azimuth)
    if camera == 'spin':
        return [3.68*math.sin(a), -3.68*math.cos(a), 1.91], [0, 0, -.05], [0, 0, 1], 39.4, -.86, 6
    return [-1.8*math.sin(a), 1.8*math.cos(a), 4.], [0, 0, 0], [0, 1, 0], 30, -.9, 8


# the studio scene (camera, ground, lights, stone) and a second scene holding only the stone for the interior trace
def make_scene(checkpoint, width, height, azimuth=0., ambient=None, rfilter='box',
               rotation=None, camera='studio'):
    vertices = checkpoint['vertices'].numpy()
    if rotation is not None:
        vertices = (vertices @ np.asarray(rotation, dtype=np.float32).T).astype(np.float32)
    faces = checkpoint['faces'].numpy().astype(np.uint32)

    mesh = mi.Mesh('learned_interior_diamond', len(vertices), len(faces))
    params = mi.traverse(mesh)
    params['vertex_positions'], params['faces'] = vertices.ravel(), faces.ravel()
    params.update()

    origin, target, up, fov, ground_z, ground_size = camera_setup(camera, azimuth)
    scene = mi.load_dict(dict(type='scene', diamond=mesh,
        sensor=dict(type='perspective', fov=fov,
                    to_world=mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up),
                    film=dict(type='hdrfilm', width=width, height=height, pixel_format='rgb',
                              rfilter=dict(type=rfilter)),
                    sampler=dict(type='independent', sample_count=1)),
        ground=dict(type='rectangle', to_world=mi.ScalarTransform4f.translate([0,0,ground_z]).scale([ground_size,ground_size,1]),
                    bsdf=dict(type='diffuse', reflectance=dict(type='rgb', value=[.12,.12,.14]))),
        **studio_lighting(**({} if ambient is None else {'ambient':ambient}))))

    stone = mi.load_dict(dict(type='scene', diamond=mesh))
    return scene, stone, mesh


# relative index of refraction at one wavelength
def eta_at(wavelength, metadata):
    p = metadata['parameters']
    return ((float(diamond_ior(wavelength)) if metadata['dispersion'] else p['int_ior']) / p['ext_ior'])


# trace the true interior journey, returning the exit ray, its weight, how it ended and the facet it left through
def analytic_exit(first, stone, eta, max_depth, rng):
    _, ct, _, eta_ti = mi.fresnel(float(first.wi.z), eta)
    ray = first.spawn_ray(dr.normalize(first.to_world(mi.refract(first.wi, ct, eta_ti))))
    weight = float(eta_ti)**2

    for _ in range(1, max_depth):
        si = stone.ray_intersect(ray)
        if not si.is_valid() or si.wi.z >= 0:
            return None, 0., 'invalid', None

        f, ct, _, eta_ti = mi.fresnel(float(si.wi.z), eta)
        if rng.random() < f:
            ray = si.spawn_ray(dr.normalize(si.to_world(mi.reflect(si.wi))))
            continue

        facet = int(si.prim_index)
        ray = si.spawn_ray(dr.normalize(si.to_world(mi.refract(si.wi, ct, eta_ti))))
        if stone.ray_test(ray):
            return None, 0., 'invalid', None
        return ray, weight*float(eta_ti)**2, 'escaped', facet

    return None, 0., 'truncated', None


# one camera ray per sample in this batch, carrying one hero wavelength, or all four lanes with a dispersion head
# a path's state is [pixel, ray, throughput, radiance, used the operator, lane (-1 for all four)]
def camera_samples(sensor, rng, start, stop, width, height, spp, stratify_wavelength, dispersion):
    states = []
    for sample_id in range(start, stop):
        pixel = sample_id//spp
        x, y = pixel % width, pixel//width

        if stratify_wavelength:
            wavelength_sample = ((sample_id % spp) + rng.random())/spp
        else:
            wavelength_sample = float(rng.random())
        ray, sensor_weight = sensor.sample_ray(0., wavelength_sample,
            mi.Point2f((x+rng.random())/width, (y+rng.random())/height), mi.Point2f(.5))

        if dispersion is None:
            beta = sensor_weight * mi.Spectrum([4.,0.,0.,0.])
            states.append([pixel, ray, beta, mi.Spectrum(0.), False, 0])
        else:
            states.append([pixel, ray, sensor_weight*mi.Spectrum(1.), mi.Spectrum(0.), False, -1])
    return states


# the network inputs for one stone hit, mapped back into object space when the stone is tumbling
def query_features(si, ray, hero, inverse_rotation, input_radius):
    p_q = np.asarray(si.p); d_q = np.asarray(ray.d); n_q = np.asarray(si.n)
    if inverse_rotation is not None:
        p_q = inverse_rotation @ p_q
        d_q = inverse_rotation @ d_q
        n_q = inverse_rotation @ n_q
    return list(p_q/input_radius) + list(d_q) + list(n_q) + [(float(ray.wavelengths[hero])-595)/235]


# path trace the studio scene, handing every refraction into the stone to the learned operator (or the analytic trace)
# returns the image, counters, and the image split by whether each path went through the operator
def render(scene, stone, mesh, model, checkpoint, width, height, spp, seed,
           mode='neural', scene_depth=8, batch_size=512, prior=None, prior_fraction=.25,
           stratify_wavelength=False, rotation=None, deterministic_exit=False,
           oracle_facet=False, dispersion=None):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    forward_rotation = None if rotation is None else np.asarray(rotation, dtype=np.float64)
    inverse_rotation = None if rotation is None else forward_rotation.T
    metadata = checkpoint['metadata']

    image = np.zeros((height*width, 3), dtype=np.float64)
    operator_image = np.zeros_like(image)
    untouched_image = np.zeros_like(image)
    sensor = scene.sensors()[0]

    stats = dict(neural_queries=0, neural_escaped=0, predicted_reentry=0,
                 analytic_escaped=0, analytic_truncated=0, analytic_invalid=0,
                 inside_scene_hit=0, scene_truncated=0,
                 operator_seconds=0., operator_calls=0, dispersion_seconds=0.,
                 analytic_interior_seconds=0., oracle_trace_seconds=0.)
    if dispersion is not None:
        if mode != 'neural' or prior is not None or oracle_facet:
            raise ValueError('The dispersion head needs plain neural mode')
        stats.update(dispersion_lanes=0, dispersion_stays=0, dispersion_splits=0,
                     dispersion_split_escaped=0, dispersion_lane_reentry=0)
    lane_masks = [mi.Spectrum([1. if j == k else 0. for j in range(4)]) for k in range(4)]

    # world-space ray leaving a predicted object-space exit, or None if it would go back into the stone
    def exit_ray(template, facet, position, direction):
        exit_si = mi.SurfaceInteraction3f()
        exit_n = model.normal[facet].numpy()
        if rotation is not None:
            position, exit_n, direction = (forward_rotation @ position, forward_rotation @ exit_n,
                                           forward_rotation @ direction)

        exit_si.p = mi.Point3f(position)
        exit_si.n = mi.Normal3f(exit_n)
        exit_si.wavelengths = template.wavelengths
        exit_si.time = template.time
        out = exit_si.spawn_ray(mi.Vector3f(direction))
        return None if stone.ray_test(out) else out

    # split all-lane paths at the operator into one path per lane
    # lane 0 keeps the operator's exit; each other lane either stays on the hero's route with a shifted exit, or leaves on its own
    def disperse(group, inputs, prediction, rows, spawned):
        rows = torch.as_tensor(rows)
        facet = prediction['exit_facet'][rows]
        position, direction = prediction['exit_position'][rows], prediction['exit_direction'][rows]
        y = hero_coordinates(model, facet, position, direction)
        wavelength = torch.tensor([float(state[1].wavelengths[k])
                                   for state in group for k in range(1, 4)])

        x = inputs.repeat_interleave(3, 0)
        began = time.perf_counter()
        shifted = dispersion.sample(x, facet.repeat_interleave(3), y.repeat_interleave(3, 0),
                                    wavelength, generator)
        own = x.clone()
        own[:, 9] = (wavelength - 595)/235
        independent = model.sample(own, generator, deterministic=deterministic_exit)
        stats['dispersion_seconds'] += time.perf_counter() - began

        stats['dispersion_lanes'] += 3*len(group)
        stats['dispersion_stays'] += int(shifted['stays'].sum())
        stats['dispersion_splits'] += int((~shifted['stays']).sum())

        survivors = []
        for g, state in enumerate(group):
            beta, template = state[2], state[1]
            out = exit_ray(template, int(facet[g]), position[g].numpy(), direction[g].numpy())
            if out is None:
                stats['predicted_reentry'] += 1
            else:
                state[1], state[2], state[4], state[5] = out, beta*lane_masks[0], True, 0
                survivors.append(state)

            for k in range(1, 4):
                j = 3*g + k - 1
                if bool(shifted['stays'][j]):
                    lane_out = exit_ray(template, int(facet[g]), shifted['exit_position'][j].numpy(),
                                        shifted['exit_direction'][j].numpy())
                elif bool(independent['escaped'][j]):
                    stats['dispersion_split_escaped'] += 1
                    lane_out = exit_ray(template, int(independent['exit_facet'][j]),
                                        independent['exit_position'][j].numpy(),
                                        independent['exit_direction'][j].numpy())
                else:
                    continue

                if lane_out is None:
                    stats['dispersion_lane_reentry'] += 1
                    continue
                child = [state[0], lane_out, beta*lane_masks[k], mi.Spectrum(0.), True, k]
                spawned.append(child)
                survivors.append(child)

        return survivors

    total = width*height*spp
    for start in range(0, total, batch_size):
        states = camera_samples(sensor, rng, start, min(start+batch_size, total), width, height,
                                spp, stratify_wavelength, dispersion)
        active = states
        spawned = []

        for step in range(scene_depth):
            next_active, pending, features, oracle = [], [], [], []

            for state in active:
                pixel, ray, beta, radiance = state[:4]
                si = scene.ray_intersect(ray)

                if not si.is_valid():
                    environment = scene.environment()
                    if environment is not None:
                        si.wi = -ray.d
                        si.wavelengths = ray.wavelengths
                        state[3] += beta*environment.eval(si)
                    continue

                emitter = si.emitter(scene)
                if emitter is not None:
                    state[3] += beta*emitter.eval(si)
                    continue

                if si.shape != mesh:
                    bs, weight = si.bsdf().sample(mi.BSDFContext(), si, float(rng.random()),
                                                mi.Point2f(*rng.random(2)))
                    if bs.pdf > 0 and float(dr.max(weight)) > 0:
                        state[1], state[2] = si.spawn_ray(si.to_world(bs.wo)), beta*weight
                        next_active.append(state)
                    continue

                if si.wi.z <= 0:
                    stats['inside_scene_hit'] += 1
                    continue

                lane = state[5]
                hero = max(lane, 0)
                eta = eta_at(float(ray.wavelengths[hero]), metadata)
                f, _, _, _ = mi.fresnel(float(si.wi.z), eta)
                reflect = rng.random() < f

                if lane < 0:
                    ratios = []
                    for k in range(4):
                        fk = float(mi.fresnel(float(si.wi.z),
                                              eta_at(float(ray.wavelengths[k]), metadata))[0])
                        ratios.append(fk/f if reflect else (1-fk)/(1-f))
                    state[2] = state[2]*mi.Spectrum(ratios)

                if reflect:
                    state[1] = si.spawn_ray(dr.normalize(si.to_world(mi.reflect(si.wi))))
                    next_active.append(state)

                elif mode == 'analytic':
                    began = time.perf_counter()
                    out, weight, status, _ = analytic_exit(si, stone, eta, metadata['max_depth'], rng)
                    stats['analytic_interior_seconds'] += time.perf_counter() - began
                    stats['analytic_'+status] += 1
                    if out is not None:
                        state[1], state[2] = out, beta*weight
                        state[4] = True
                        next_active.append(state)

                else:
                    features.append(query_features(si, ray, hero, inverse_rotation,
                                                   checkpoint['input_radius']))
                    pending.append(state)
                    if oracle_facet:
                        began = time.perf_counter()
                        _, _, status, true_facet = analytic_exit(
                            si, stone, eta, metadata['max_depth'], rng)
                        stats['oracle_trace_seconds'] += time.perf_counter() - began
                        oracle.append((status == 'escaped', true_facet or 0))

            if pending:
                inputs = torch.tensor(features, dtype=torch.float32)
                forced = (torch.tensor([f for _, f in oracle], dtype=torch.long)
                          if oracle_facet else None)

                began = time.perf_counter()
                if prior is None:
                    prediction = model.sample(inputs, generator,
                                              deterministic=deterministic_exit,
                                              facet=forced)
                    if oracle_facet:
                        prediction['escaped'] = torch.tensor([e for e, _ in oracle])
                        prediction['throughput'] = prediction['escaped'].float()
                else:
                    from neural.learned_interior_prior import sample_mixture
                    prediction = sample_mixture(model, prior, inputs, prior_fraction, generator)
                stats['operator_seconds'] += time.perf_counter() - began
                stats['operator_calls'] += 1
                stats['neural_queries'] += len(pending)

                spread = []
                for i, state in enumerate(pending):
                    if not bool(prediction['escaped'][i]):
                        continue
                    stats['neural_escaped'] += 1
                    if state[5] < 0:
                        spread.append(i)
                        continue

                    out = exit_ray(state[1], int(prediction['exit_facet'][i]),
                                   prediction['exit_position'][i].numpy(),
                                   prediction['exit_direction'][i].numpy())
                    if out is None:
                        stats['predicted_reentry'] += 1
                        continue
                    state[1] = out
                    state[2] *= float(prediction['throughput'][i])
                    state[4] = True
                    next_active.append(state)

                if spread:
                    next_active += disperse([pending[i] for i in spread], inputs[spread],
                                            prediction, spread, spawned)

            active = next_active
            if not active:
                break

        stats['scene_truncated'] += len(active)
        for pixel, ray, _, radiance, used_operator, _ in states + spawned:
            value = np.asarray(mi.spectrum_to_srgb(radiance, ray.wavelengths))
            image[pixel] += value
            (operator_image if used_operator else untouched_image)[pixel] += value

        if start == 0 or (start//batch_size) % 20 == 0:
            print(f'{min(start+batch_size,total)}/{total} camera samples', flush=True)

    shape = (height, width, 3)
    components = dict(
        operator=(operator_image.reshape(shape)/spp).astype(np.float32),
        untouched=(untouched_image.reshape(shape)/spp).astype(np.float32))
    return (image.reshape(shape)/spp).astype(np.float32), stats, components


# the render options and the checks on how they combine
def parse_arguments():
    parser = argparse.ArgumentParser(description='Spectral renderer for the learned interior operator.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['neural','analytic'], default='neural')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--spp', type=int, default=16)
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--scene_depth', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--azimuth', type=float, default=0.)
    parser.add_argument('--camera', choices=CAMERAS, default='studio',
                        help='studio is the view the operator was trained for; spin is '
                             'the side view of ground_truth/diamond.py (analytic_spin.gif)')
    parser.add_argument('--rotation_deg', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Tumble the stone by these degrees about X, then Y, '
                             'then Z, leaving camera, lights and ground fixed. '
                             'Matches the convention eval.py animates with. Model '
                             'queries are mapped into object space automatically.')
    parser.add_argument('--oracle_facet', action='store_true',
                        help='Neural mode only. Take the exit facet and the escape '
                             'decision from the analytic trace and leave the network '
                             'only the coordinates within that facet. This is an upper '
                             'bound, not a renderer: it measures how much of the gap to '
                             'the reference is branch selection alone.')
    parser.add_argument('--deterministic_exit', action='store_true',
                        help='Decode each learned exit at its mixture component '
                             'mean instead of sampling the Gaussian spread. The '
                             'exit facet and component are still sampled, so the '
                             'discrete branch structure is kept and only the '
                             'within-branch blur is removed.')
    parser.add_argument('--stratify_wavelength', action='store_true',
                        help='Stratify the hero wavelength across a pixel samples '
                             'instead of drawing it independently. Unbiased, free, '
                             'and reduces chromatic speckle. Off by default so that '
                             'existing noise measurements stay comparable.')
    parser.add_argument('--rfilter', choices=['box','gaussian','tent'], default='box',
                        help='Film reconstruction filter. Box is the default so that '
                             'existing measurements stay comparable; gaussian spreads '
                             'each sample over neighbouring pixels and looks markedly '
                             'cleaner, but correlates adjacent pixels, so do not mix '
                             'filters within one comparison.')
    parser.add_argument('--dispersion', type=Path,
                        help='Learned dispersion head (train_dispersion.py). Neural mode '
                             'only: traces all four spectral lanes and lets the head '
                             'decide per lane whether it follows the hero\'s route with '
                             'a shifted exit or leaves independently.')
    parser.add_argument('--rdm_prior', type=Path)
    parser.add_argument('--prior_fraction', type=float, default=.25)
    parser.add_argument('--split_components', action='store_true',
                        help='Also write component_operator.exr and component_untouched.exr, '
                             'an exact split of the frame by whether each path invoked the '
                             'learned operator. In analytic mode the operator component is zero.')
    args = parser.parse_args()

    if not 0 <= args.prior_fraction < 1:
        parser.error('--prior_fraction must be in [0,1)')
    if args.mode == 'analytic' and args.rdm_prior is not None:
        parser.error('RDM prior applies only to neural mode')
    if args.dispersion is not None and (args.mode != 'neural' or args.rdm_prior is not None
                                        or args.oracle_facet):
        parser.error('--dispersion needs plain neural mode (no RDM prior, no oracle facet)')
    if min(args.width,args.height,args.spp,args.scene_depth,args.batch_size) < 1:
        parser.error('Image dimensions, spp, depths and batch size must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    return args


# load the operator and check it uses the radiance convention and the geometry it was trained on
def load_checked_model(path):
    model, checkpoint = load_model(path)
    m = checkpoint['metadata']
    if m.get('transport_mode') != 'radiance' or m.get('coordinates') != 'object_space':
        raise ValueError('Unsupported transport convention')

    v, f = checkpoint['vertices'].numpy(), checkpoint['faces'].numpy().astype(np.uint32)
    if hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest() != m['geometry_sha256']:
        raise ValueError('Checkpoint geometry hash mismatch')
    return model, checkpoint


# load an RDM prior, only if it was built for exactly this checkpoint
def load_prior(path, model_path, model):
    from neural.learned_interior_prior import LearnedInteriorPrior
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(str(archive['metadata_json']))
        if info.get('schema_version') != 1 or info['model_sha256'] != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError('RDM prior does not match this checkpoint')
        return LearnedInteriorPrior(archive['pmf'], info['theta_bins'], info['phi_bins'], model)


# write the frame as exr and png, plus the operator / untouched split if asked for
def write_frames(directory, image, components, split_components):
    frames = directory/'frames'
    frames.mkdir(parents=True)
    mi.Bitmap(image).write(str(frames/'frame_0000.exr'))
    if split_components:
        for name, value in components.items():
            mi.Bitmap(value).write(str(frames/('component_%s.exr' % name)))
    mi.Bitmap(np.maximum(image,0)*display_exposure()).convert(mi.Bitmap.PixelFormat.RGB,
                mi.Struct.Type.UInt8, True).write(str(frames/'frame_0000.png'))


# render one frame and save it with a render.json of every setting and counter
def main():
    args = parse_arguments()
    torch.set_num_threads(2)

    model, checkpoint = load_checked_model(args.model)
    m = checkpoint['metadata']
    rotation = (None if args.rotation_deg is None
                else tumble_matrix(*[math.radians(v) for v in args.rotation_deg]))
    scene, stone, mesh = make_scene(checkpoint,args.width,args.height,args.azimuth,
                                    rfilter=args.rfilter, rotation=rotation, camera=args.camera)

    prior = None
    if args.rdm_prior is not None:
        prior = load_prior(args.rdm_prior, args.model, model)
    dispersion = None
    if args.dispersion is not None:
        dispersion, _ = load_dispersion(args.dispersion, checkpoint)

    begin = time.perf_counter()
    image, stats, components = render(scene,stone,mesh,model,checkpoint,args.width,args.height,args.spp,args.seed,
                          args.mode,args.scene_depth,args.batch_size,prior,args.prior_fraction,
                          args.stratify_wavelength,rotation,args.deterministic_exit,
                          args.oracle_facet, dispersion)
    render_seconds = time.perf_counter()-begin
    if not np.isfinite(image).all():
        raise RuntimeError('Nonfinite render')
    write_frames(args.output_dir, image, components, args.split_components)

    queries = stats['neural_queries']
    interior_seconds = stats['operator_seconds'] if args.mode == 'neural' else stats['analytic_interior_seconds']
    interior_count = queries if args.mode == 'neural' else sum(
        stats['analytic_' + s] for s in ('escaped', 'truncated', 'invalid'))
    timing = dict(render_seconds=render_seconds,
                  interior_decision_seconds=interior_seconds,
                  interior_decisions=interior_count,
                  microseconds_per_interior_decision=(1e6*interior_seconds/interior_count
                                                      if interior_count else None),
                  interior_share_of_render=interior_seconds/render_seconds if render_seconds else None,
                  network_batches=stats['operator_calls'],
                  mean_network_batch_size=queries/stats['operator_calls'] if stats['operator_calls'] else None,
                  dispersion_seconds=stats['dispersion_seconds'],
                  oracle_trace_seconds=stats['oracle_trace_seconds'],
                  torch_threads=torch.get_num_threads(),
                  note='interior decision = one learned operator query in neural mode, one exact '
                       'interior trace in analytic mode; batched queries are timed per batch and divided')

    report = dict(rdm_prior=str(args.rdm_prior) if prior is not None else None,
                  prior_fraction=args.prior_fraction if prior is not None else 0.,
                  mode=args.mode, model=str(args.model),
                  width=args.width, height=args.height,
                  spp=args.spp, seed=args.seed, scene_depth=args.scene_depth, batch_size=args.batch_size,
                  azimuth=args.azimuth, camera=args.camera, rfilter=args.rfilter,
                  rotation_deg=args.rotation_deg,
                  deterministic_exit=args.deterministic_exit,
                  oracle_facet=args.oracle_facet,
                  stratify_wavelength=args.stratify_wavelength,
                  dispersion=str(args.dispersion) if dispersion is not None else None,
                  dispersion_sha256=(hashlib.sha256(args.dispersion.read_bytes()).hexdigest()
                                     if dispersion is not None else None),
                  internal_max_depth=m['max_depth'], seconds=time.perf_counter()-begin,
                  timing=timing, stats=stats, geometry_sha256=m['geometry_sha256'],
                  checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  exposure=display_exposure(), limitations=LIMITATIONS)
    (args.output_dir/'render.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
