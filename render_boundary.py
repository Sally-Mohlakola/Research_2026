"""Experimental scalar spectral renderer for a learned boundary operator.

The model replaces the complete journey conditional on entry transmission.
Entry reflection/transmission is sampled with Fresnel probabilities, so entry T
must NOT also multiply the transmitted branch. No learned PDF is used for MIS.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import config
import mitsuba as mi
import drjit as dr
import numpy as np
import torch

mi.set_variant('scalar_spectral')
from neural.boundary_model import load_model
from bsdf.dispersion import diamond_ior
from utils.studio_env import studio_lighting, display_exposure


def make_scene(checkpoint, width, height, azimuth=0., ambient=None):
    vertices = checkpoint['vertices'].numpy()
    faces = checkpoint['faces'].numpy().astype(np.uint32)
    mesh = mi.Mesh('boundary_diamond', len(vertices), len(faces))
    params = mi.traverse(mesh)
    params['vertex_positions'], params['faces'] = vertices.ravel(), faces.ravel()
    params.update()
    a = math.radians(azimuth)
    camera = [-1.8*math.sin(a), 1.8*math.cos(a), 4.]
    scene = mi.load_dict(dict(type='scene', diamond=mesh,
        sensor=dict(type='perspective', fov=30,
                    to_world=mi.ScalarTransform4f.look_at(origin=camera, target=[0,0,0], up=[0,1,0]),
                    film=dict(type='hdrfilm', width=width, height=height, pixel_format='rgb',
                              rfilter=dict(type='box')),
                    sampler=dict(type='independent', sample_count=1)),
        ground=dict(type='rectangle', to_world=mi.ScalarTransform4f.translate([0,0,-.9]).scale([8,8,1]),
                    bsdf=dict(type='diffuse', reflectance=dict(type='rgb', value=[.12,.12,.14]))),
        **studio_lighting(**({} if ambient is None else {'ambient':ambient}))))
    # Separate scene for the internal reference journey. Same mesh, no lights.
    stone = mi.load_dict(dict(type='scene', diamond=mesh))
    return scene, stone, mesh


def eta_at(wavelength, metadata):
    p = metadata['parameters']
    return ((float(diamond_ior(wavelength)) if metadata['dispersion'] else p['int_ior']) / p['ext_ior'])


def analytic_exit(first, stone, eta, max_depth, rng):
    _, ct, _, eta_ti = mi.fresnel(float(first.wi.z), eta)
    ray = first.spawn_ray(dr.normalize(first.to_world(mi.refract(first.wi, ct, eta_ti))))
    weight = float(eta_ti)**2
    for _ in range(1, max_depth):
        si = stone.ray_intersect(ray)
        if not si.is_valid() or si.wi.z >= 0:
            return None, 0., 'invalid'
        f, ct, _, eta_ti = mi.fresnel(float(si.wi.z), eta)
        if rng.random() < f:
            ray = si.spawn_ray(dr.normalize(si.to_world(mi.reflect(si.wi))))
        else:
            ray = si.spawn_ray(dr.normalize(si.to_world(mi.refract(si.wi, ct, eta_ti))))
            if stone.ray_test(ray):
                return None, 0., 'invalid'
            return ray, weight*float(eta_ti)**2, 'escaped'
    return None, 0., 'truncated'


def render(scene, stone, mesh, model, checkpoint, width, height, spp, seed,
           mode='neural', scene_depth=8, batch_size=512, prior=None, prior_fraction=.25):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    metadata = checkpoint['metadata']
    image = np.zeros((height*width, 3), dtype=np.float64)
    # Exact split of the final image by whether the path used the learned operator.
    operator_image = np.zeros_like(image)
    untouched_image = np.zeros_like(image)
    sensor = scene.sensors()[0]
    stats = dict(neural_queries=0, neural_escaped=0, predicted_reentry=0,
                 analytic_escaped=0, analytic_truncated=0, analytic_invalid=0,
                 inside_scene_hit=0, scene_truncated=0)
    total = width*height*spp
    for start in range(0, total, batch_size):
        states = []
        for sample_id in range(start, min(start+batch_size, total)):
            pixel = sample_id//spp
            x, y = pixel % width, pixel//width
            ray, sensor_weight = sensor.sample_ray(0., float(rng.random()),
                mi.Point2f((x+rng.random())/width, (y+rng.random())/height), mi.Point2f(.5))
            # Trace one hero wavelength. Sensor weights retain its wavelength PDF.
            beta = sensor_weight * mi.Spectrum([4.,0.,0.,0.])
            states.append([pixel, ray, beta, mi.Spectrum(0.), False])
        active = states
        for step in range(scene_depth):
            next_active, pending, features = [], [], []
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
                if si.shape == mesh:
                    if si.wi.z <= 0:
                        stats['inside_scene_hit'] += 1
                        continue
                    eta = eta_at(float(ray.wavelengths[0]), metadata)
                    f, _, _, _ = mi.fresnel(float(si.wi.z), eta)
                    if rng.random() < f:
                        state[1] = si.spawn_ray(dr.normalize(si.to_world(mi.reflect(si.wi))))
                        next_active.append(state)
                    elif mode == 'analytic':
                        out, weight, status = analytic_exit(si, stone, eta, metadata['max_depth'], rng)
                        stats['analytic_'+status] += 1
                        if out is not None:
                            state[1], state[2] = out, beta*weight
                            # Same flag as the neural branch: this path went
                            # through the stone interior rather than reflecting
                            # off the first surface, so both modes split alike.
                            state[4] = True
                            next_active.append(state)
                    else:
                        features.append(list(np.asarray(si.p)/checkpoint['input_radius'])+
                                        list(ray.d)+list(si.n)+[(float(ray.wavelengths[0])-595)/235])
                        pending.append(state)
                else:
                    bs, weight = si.bsdf().sample(mi.BSDFContext(), si, float(rng.random()),
                                                mi.Point2f(*rng.random(2)))
                    if bs.pdf > 0 and float(dr.max(weight)) > 0:
                        state[1], state[2] = si.spawn_ray(si.to_world(bs.wo)), beta*weight
                        next_active.append(state)
            if pending:
                inputs = torch.tensor(features, dtype=torch.float32)
                if prior is None:
                    prediction = model.sample(inputs, generator)
                else:
                    from neural.boundary_prior import sample_mixture
                    prediction = sample_mixture(model, prior, inputs, prior_fraction, generator)
                stats['neural_queries'] += len(pending)
                for i, state in enumerate(pending):
                    if not bool(prediction['escaped'][i]):
                        continue
                    stats['neural_escaped'] += 1
                    facet = int(prediction['exit_facet'][i])
                    # spawn_ray uses Mitsuba's surface-offset convention.
                    exit_si = mi.SurfaceInteraction3f()
                    exit_si.p = mi.Point3f(prediction['exit_position'][i].numpy())
                    exit_si.n = mi.Normal3f(model.normal[facet].numpy())
                    exit_si.wavelengths = state[1].wavelengths
                    exit_si.time = state[1].time
                    out = exit_si.spawn_ray(mi.Vector3f(prediction['exit_direction'][i].numpy()))
                    if stone.ray_test(out):
                        # Do not tunnel through the stone or invent a corrected exit.
                        stats['predicted_reentry'] += 1
                        continue
                    state[1] = out
                    state[2] *= float(prediction['throughput'][i])
                    state[4] = True
                    next_active.append(state)
            active = next_active
            if not active:
                break
        stats['scene_truncated'] += len(active)
        for pixel, ray, _, radiance, used_operator in states:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    if min(args.width,args.height,args.spp,args.scene_depth,args.batch_size) < 1:
        parser.error('Image dimensions, spp, depths and batch size must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(2)
    model, checkpoint = load_model(args.model)
    m = checkpoint['metadata']
    if m.get('transport_mode') != 'radiance' or m.get('coordinates') != 'object_space':
        raise ValueError('Unsupported transport convention')
    v, f = checkpoint['vertices'].numpy(), checkpoint['faces'].numpy().astype(np.uint32)
    if hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest() != m['geometry_sha256']:
        raise ValueError('Checkpoint geometry hash mismatch')
    scene, stone, mesh = make_scene(checkpoint,args.width,args.height,args.azimuth)
    prior = None
    if args.rdm_prior is not None:
        from neural.boundary_prior import BoundaryPrior
        with np.load(args.rdm_prior, allow_pickle=False) as archive:
            info = json.loads(str(archive['metadata_json']))
            if info.get('schema_version') != 1 or info['model_sha256'] != hashlib.sha256(args.model.read_bytes()).hexdigest():
                raise ValueError('RDM prior does not match this checkpoint')
            prior = BoundaryPrior(archive['pmf'], info['theta_bins'], info['phi_bins'], model)
    begin = time.perf_counter()
    image, stats, components = render(scene,stone,mesh,model,checkpoint,args.width,args.height,args.spp,args.seed,
                          args.mode,args.scene_depth,args.batch_size,prior,args.prior_fraction)
    if not np.isfinite(image).all():
        raise RuntimeError('Nonfinite render')
    frames = args.output_dir/'frames'
    frames.mkdir(parents=True)
    mi.Bitmap(image).write(str(frames/'frame_0000.exr'))
    if args.split_components:
        for name, value in components.items():
            mi.Bitmap(value).write(str(frames/('component_%s.exr' % name)))
    mi.Bitmap(np.maximum(image,0)*display_exposure()).convert(mi.Bitmap.PixelFormat.RGB,
                mi.Struct.Type.UInt8, True).write(str(frames/'frame_0000.png'))
    report = dict(rdm_prior=str(args.rdm_prior) if prior is not None else None,
                  prior_fraction=args.prior_fraction if prior is not None else 0.,mode=args.mode, model=str(args.model), width=args.width,height=args.height,
                  spp=args.spp,seed=args.seed,scene_depth=args.scene_depth,batch_size=args.batch_size,
                  azimuth=args.azimuth,internal_max_depth=m['max_depth'],seconds=time.perf_counter()-begin,
                  stats=stats,geometry_sha256=m['geometry_sha256'],
                  checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  exposure=display_exposure(),limitations=['Experimental learned transport; RDM proposal preserves the learned target, not physical ground truth.',
                  'BSDF-only scene tracing; finite depth; no MIS or next-event estimation.',
                  'Same seeds do not ensure identical paths between modes.',
                  'Predicted exits reentering the stone are rejected and counted.'])
    (args.output_dir/'render.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
