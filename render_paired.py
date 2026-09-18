"""One command for a paired neural/analytic comparison at high sample counts.

Renders both modes over several independent seeds in parallel, merges each mode
into a single low-variance image, and runs the masked render evaluation. This
replaces the manual sequence of launching renders, waiting, merging each set and
then evaluating.

Averaging K independent renders of S samples per pixel is exactly the same
estimator as one render of K*S samples per pixel when the seeds differ, so the
seed fan-out is a parallel route to a high sample count, not an approximation.

Stages run as subprocesses rather than imports on purpose: the Mitsuba variant
is process-global, and these tools do not agree on one. render_boundary and
evaluate_render need scalar_spectral while merge_renders uses scalar_rgb, so
importing them into a single process would fight over the variant.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODES = ('analytic', 'neural')


def run(command, log):
    """Run one stage, capturing output to a log file beside the results."""
    log.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, '-B'] + [str(c) for c in command],
                          capture_output=True, text=True)
    log.write_text((done.stdout or '') + (done.stderr or ''), encoding='utf-8')
    if done.returncode != 0:
        raise RuntimeError('Stage failed (%s); see %s\n%s'
                           % (' '.join(str(c) for c in command[:2]), log,
                              (done.stderr or '')[-2000:]))
    return done.stdout


def render_task(job):
    (directory, model, mode, azimuth, seed, width, height, spp, scene_depth,
     logs, split, rfilter, deterministic, oracle) = job
    command = ['render_boundary.py', '--model', model, '--output_dir', directory,
               '--mode', mode, '--azimuth', '%g' % azimuth, '--seed', seed,
               '--width', width, '--height', height, '--spp', spp,
               '--scene_depth', scene_depth, '--rfilter', rfilter]
    if deterministic:
        command.append('--deterministic_exit')
    if oracle and mode == 'neural':
        command.append('--oracle_facet')
    if split:
        command.append('--split_components')
    run(command, logs/('render_%s_s%s.log' % (mode, seed)))
    return mode, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--azimuth', type=float, default=0.)
    parser.add_argument('--width', type=int, default=728)
    parser.add_argument('--height', type=int, default=728)
    parser.add_argument('--spp', type=int, default=8,
                        help='Samples per pixel PER SEED. Effective spp is this '
                             'times the number of seeds.')
    parser.add_argument('--scene_depth', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[101, 102, 103, 104, 105, 106, 107, 108])
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent render processes. Each holds Mitsuba and '
                             'PyTorch at roughly 250 MB, so raise this only when '
                             'nothing else heavy is running.')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--flash_multiple', type=float, default=32.)
    parser.add_argument('--rfilter', choices=['box','gaussian','tent'], default='box',
                        help='Film reconstruction filter, applied to both modes so the '
                             'comparison stays paired. Gaussian looks cleaner but '
                             'correlates neighbouring pixels.')
    parser.add_argument('--oracle_facet', action='store_true',
                        help='Neural renders take the exit facet and escape decision '
                             'from the analytic trace; an upper bound isolating branch '
                             'selection, not a usable renderer.')
    parser.add_argument('--deterministic_exit', action='store_true',
                        help='Decode learned exits at the mixture component mean, '
                             'keeping the discrete branch structure but removing '
                             'the within-branch Gaussian spread.')
    parser.add_argument('--split_components', action='store_true',
                        help='Split each frame by whether the path went through the '
                             'stone interior, and report the learned component in '
                             'isolation from the analytic first-surface reflection.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Render and merge only; do not compare the modes.')
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be distinct; repeats are not independent samples')
    if not args.skip_evaluation and len(args.modes) != len(MODES):
        parser.error('Evaluation needs both modes; pass --skip_evaluation for one')
    if not args.skip_evaluation and len(args.seeds) < 2:
        parser.error('Evaluation needs at least two seeds to measure a noise floor')
    if min(args.width, args.height, args.spp, args.scene_depth, args.workers) < 1:
        parser.error('Dimensions, spp, depth and workers must be positive')
    if args.output_dir.exists():
        raise FileExistsError('Refusing to replace %s' % args.output_dir)

    logs = args.output_dir/'logs'
    jobs, directories = [], {mode: [] for mode in args.modes}
    for mode in args.modes:
        for seed in args.seeds:
            directory = args.output_dir/mode/('s%d' % seed)
            directories[mode].append(directory)
            jobs.append((directory, args.model, mode, args.azimuth, seed,
                         args.width, args.height, args.spp, args.scene_depth, logs,
                         args.split_components, args.rfilter, args.deterministic_exit,
                         args.oracle_facet))

    effective = args.spp*len(args.seeds)
    print('%d renders: %s x %d seeds at %d spp each (%d effective spp), %d workers'
          % (len(jobs), '+'.join(args.modes), len(args.seeds), args.spp,
             effective, args.workers), flush=True)

    begin = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for mode, seed in pool.map(render_task, jobs):
            completed += 1
            print('  [%d/%d] %s seed %d' % (completed, len(jobs), mode, seed), flush=True)
    render_seconds = time.perf_counter()-begin

    merged, results = {}, {}
    for mode in args.modes:
        output = args.output_dir/('%s_az%03.0f' % (mode, args.azimuth))
        results[mode] = output.with_suffix('.png')
        print('merging %s' % mode, flush=True)
        run(['merge_renders.py', '--input_dir', args.output_dir/mode,
             '--output', output], logs/('merge_%s.log' % mode))
        merged[mode] = json.loads(output.with_suffix('.json').read_text())

    summary = dict(model=str(args.model), azimuth=args.azimuth,
                   width=args.width, height=args.height, spp_each=args.spp,
                   seeds=args.seeds, effective_spp=effective,
                   modes=args.modes, workers=args.workers, rfilter=args.rfilter,
                   deterministic_exit=args.deterministic_exit,
                   oracle_facet=args.oracle_facet,
                   render_seconds=render_seconds, merged=merged,
                   result_images={m: str(p) for m, p in results.items()},
                   note='Per-seed renders under <mode>/s<seed>/ are ingredients at '
                        '--spp each, not results. The finished images are '
                        'result_images, at the top level, at the full effective spp.')

    if not args.skip_evaluation:
        evaluation = args.output_dir/'evaluation.json'
        print('evaluating', flush=True)
        command = ['evaluate_render.py', '--model', args.model,
                   '--flash_multiple', args.flash_multiple, '--output', evaluation]
        command += ['--analytic'] + directories['analytic']
        command += ['--neural'] + directories['neural']
        print(run(command, logs/'evaluate.log'))
        summary['evaluation'] = str(evaluation)

    (args.output_dir/'paired.json').write_text(json.dumps(summary, indent=2),
                                               encoding='utf-8')
    print('\nrendered in %.0f s' % render_seconds)
    print('\nFINISHED IMAGES (%d effective spp):' % effective)
    for mode in args.modes:
        print('  %-8s %s' % (mode, results[mode]))
        print('  %-8s mean luminance %.6g' % ('', merged[mode]['mean_luminance']))
    print('\nThe per-seed renders under <mode>/s<seed>/ are ingredients at %d spp '
          'each,\nnot results. Use the files listed above.' % args.spp)


if __name__ == '__main__':
    main()
