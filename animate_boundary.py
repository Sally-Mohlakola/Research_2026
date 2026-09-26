"""Rotation animation for the learned boundary operator.

Renders a camera orbit around the fixed stone, one frame per azimuth, in
parallel across frames, then encodes MP4 and GIF with ffmpeg.

Frames reuse render_boundary.py unchanged, so every frame carries the same
camera rig, spectral handling and fixed display exposure. The exposure is a
constant rather than per-frame auto-exposure, which matters for an animation:
adaptive exposure would make the stone appear to pulse as facets brighten and
darken, hiding exactly the scintillation an orbit is meant to reveal.

Cost scales with frames x width x height x spp, so prefer a modest resolution
for a long orbit. This is a qualitative viewing tool; use flash_sweep.py when
scintillation needs to be measured rather than watched.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, '-B'] + [str(c) for c in command],
                          capture_output=True, text=True)
    log.write_text((done.stdout or '') + (done.stderr or ''), encoding='utf-8')
    if done.returncode != 0:
        raise RuntimeError('Frame failed; see %s\n%s' % (log, (done.stderr or '')[-2000:]))


def tumble_angles(index, frames, speed=1.0):
    """eval.py's tumbling motion: 2, 1.5 and 0.5 turns about X, Y and Z."""
    t = index/frames
    return (360.*2.0*t*speed, 360.*1.5*t*speed, 360.*0.5*t*speed)


def render_frame(job):
    (index, azimuth, directory, model, mode, seed, width, height, spp,
     scene_depth, logs, rotation) = job
    command = ['render_boundary.py', '--model', model, '--output_dir', directory,
               '--mode', mode, '--azimuth', '%g' % azimuth, '--seed', seed,
               '--width', width, '--height', height, '--spp', spp,
               '--scene_depth', scene_depth]
    if rotation is not None:
        command += ['--rotation_deg'] + ['%g' % v for v in rotation]
    run(command, logs/('frame_%04d.log' % index))
    return index, azimuth


def encode(sequence, output, fps):
    """Write MP4 and GIF from a numbered PNG sequence, if ffmpeg is present."""
    if shutil.which('ffmpeg') is None:
        print('ffmpeg not found; PNG frames written but no video encoded')
        return []
    pattern = str(sequence/'frame_%04d.png')
    written = []
    video = output/'rotation.mp4'
    subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', pattern,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                    str(video)], check=True, capture_output=True)
    written.append(video)
    animated = output/'rotation.gif'
    # Never upscale: enlarging a noisy frame magnifies the grain without adding
    # information, which reads as a far worse render than the source actually is.
    subprocess.run(['ffmpeg', '-y', '-framerate', str(min(fps, 15)), '-i', pattern,
                    '-vf', "scale='min(480,iw)':-1:flags=lanczos,split[a][b];"
                           '[a]palettegen[p];[b][p]paletteuse',
                    str(animated)], check=True, capture_output=True)
    written.append(animated)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['neural', 'analytic'], default='neural')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--azimuth_start', type=float, default=0.)
    parser.add_argument('--azimuth_sweep', type=float, default=360.,
                        help='Total degrees travelled. A round brilliant has '
                             'eight-fold symmetry, so 45 degrees already shows '
                             'every distinct pose.')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--spp', type=int, default=32)
    parser.add_argument('--scene_depth', type=int, default=8)
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted animation in --output_dir: '
                             'frames already rendered are kept, half-finished ones '
                             'are redone. Settings must match the original run.')
    parser.add_argument('--vary_seed', action='store_true',
                        help='Use a different seed per frame. Off by default: a '
                             'shared seed keeps the noise pattern stable between '
                             'frames so the eye reads facet motion rather than '
                             'flicker.')
    parser.add_argument('--motion', choices=['orbit', 'tumble'], default='orbit',
                        help='orbit moves the camera around a fixed stone. tumble '
                             'rotates the stone with camera and lights fixed, which '
                             'is what eval.py does and what shows scintillation.')
    parser.add_argument('--rotation_speed', type=float, default=1.0,
                        help='Turns multiplier for --motion tumble.')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent frame renders, about 250 MB each.')
    args = parser.parse_args()

    if args.frames < 2:
        parser.error('--frames must be at least two')
    if min(args.width, args.height, args.spp, args.scene_depth, args.workers, args.fps) < 1:
        parser.error('Dimensions, spp, depth, workers and fps must be positive')
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError('Refusing to replace %s (pass --resume to continue it)'
                              % args.output_dir)

    logs = args.output_dir/'logs'
    step = args.azimuth_sweep/args.frames
    jobs = []
    skipped = 0
    for index in range(args.frames):
        target = args.output_dir/'raw'/('f%04d' % index)
        if args.resume and (target/'frames'/'frame_0000.png').exists():
            skipped += 1
            continue
        if target.exists():
            # A frame directory without its image was interrupted mid-render;
            # render_boundary refuses existing directories, so clear it.
            shutil.rmtree(target)
        azimuth = args.azimuth_start + index*step
        rotation = (tumble_angles(index, args.frames, args.rotation_speed)
                    if args.motion == 'tumble' else None)
        jobs.append((index, azimuth if args.motion == 'orbit' else args.azimuth_start,
                     args.output_dir/'raw'/('f%04d' % index),
                     args.model, args.mode,
                     args.seed + (index if args.vary_seed else 0),
                     args.width, args.height, args.spp, args.scene_depth, logs,
                     rotation))

    print('%d frames, %s, %gx%g degrees at %g deg/frame, %dx%d at %d spp, %d workers'
          % (args.frames, args.mode, args.azimuth_start,
             args.azimuth_start+args.azimuth_sweep, step,
             args.width, args.height, args.spp, args.workers), flush=True)
    if skipped:
        print('resuming: %d frames already rendered, %d to go' % (skipped, len(jobs)),
              flush=True)

    begin = time.perf_counter()
    done = skipped
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, azimuth in pool.map(render_frame, jobs):
            done += 1
            print('  [%d/%d] frame %d' % (done, args.frames, index), flush=True)
    seconds = time.perf_counter()-begin

    # ffmpeg needs a contiguous numbered sequence in one directory.
    sequence = args.output_dir/'frames'
    sequence.mkdir(parents=True, exist_ok=True)
    for index in range(args.frames):
        source = args.output_dir/'raw'/('f%04d' % index)/'frames'/'frame_0000.png'
        if not source.exists():
            raise FileNotFoundError('Missing frame %d at %s' % (index, source))
        shutil.copyfile(source, sequence/('frame_%04d.png' % index))

    written = encode(sequence, args.output_dir, args.fps)
    (args.output_dir/'animation.json').write_text(json.dumps(dict(
        model=str(args.model), mode=args.mode, frames=args.frames,
        azimuth_start=args.azimuth_start, azimuth_sweep=args.azimuth_sweep,
        degrees_per_frame=step, width=args.width, height=args.height,
        spp=args.spp, seed=args.seed, vary_seed=args.vary_seed, fps=args.fps,
        motion=args.motion, rotation_speed=args.rotation_speed,
        render_seconds=seconds, outputs=[str(p) for p in written]),
        indent=2), encoding='utf-8')

    print('\n%d frames in %.0f s -> %s' % (args.frames, seconds, args.output_dir))
    for path in written:
        print('  %s' % path)


if __name__ == '__main__':
    main()
