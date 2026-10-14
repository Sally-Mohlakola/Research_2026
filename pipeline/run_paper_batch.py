import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import datetime
import shlex
import subprocess
import time

HISTORY = REPO.parent/'checkpoint_history'

# the pools each stone is trained and tested on
POOLS = {
    'gia': dict(train=HISTORY/'camera_pool'/'train.npz', test=HISTORY/'camera_pool'/'test.npz'),
    'pear': dict(train=HISTORY/'pear_pool'/'train.npz', test=HISTORY/'pear_pool'/'test.npz'),
    'step': dict(train=REPO/'checkpoints'/'step_pool'/'train.npz', test=REPO/'checkpoints'/'step_pool'/'test.npz'),
}

# the step cut never had a held-out pool; this gathers one the same way as its training pool, with a new seed
STEP_TEST_GATHER = ['pipeline/gather_learned_interior.py', '--output', POOLS['step']['test'],
                    '--diamond_name', 'step_cut', '--entries', 16384, '--paths_per_entry', 16,
                    '--max_depth', 128, '--seed', 402, '--workers', 8,
                    '--entry_sampling', 'camera', '--azimuth_range', 0, 360]

# render settings shared by every run, so each run can reuse the first run's analytic reference
RENDER = ['--image_width', 512, '--image_height', 512, '--spp', 32, '--seeds', 101, 102, 103, 104, '--workers', 4]
HEADS = dict(stochastic='mixture', deep='clone_deep', clone='clone')


# one recorded run: which stone, which head, which training seed, and what it is paired with
def job(stone, decoder, seed, reference=None, compare=(), note=''):
    name = '%s_%s_s%d' % (stone, decoder, seed)
    command = ['pipeline/run_experiment.py', '--name', name, '--note', note,
               '--pool', POOLS[stone]['train'], '--test_pool', POOLS[stone]['test'],
               '--head', HEADS[decoder], '--seed', seed, *RENDER]
    if reference:
        command += ['--reference_run', reference]
    if compare:
        command += ['--compare_with', *compare]
    return name, command


# the whole batch, in an order where every reference run exists before the runs that reuse it
def plan():
    runs = []
    for stone in ('gia', 'pear', 'step'):
        first, _ = job(stone, 'stochastic', 23)
        runs.append(job(stone, 'stochastic', 23, note='%s, stochastic decoder, seed 23; the reference renders' % stone))
        runs.append(job(stone, 'deep', 23, reference=first, note='%s, deep decoder, seed 23' % stone))
    runs.append(job('gia', 'clone', 23, reference='gia_stochastic_s23', note='gia, clone baseline, seed 23'))
    for seed in (24, 25):
        stochastic = 'gia_stochastic_s%d' % seed
        runs.append(job('gia', 'stochastic', seed, reference='gia_stochastic_s23',
                        note='gia, stochastic decoder, repeat with training seed %d' % seed))
        runs.append(job('gia', 'deep', seed, reference='gia_stochastic_s23', compare=(stochastic,),
                        note='gia, deep decoder, repeat with training seed %d' % seed))
    summaries = [
        ['analysis/aggregate_runs.py', '--name', 'gia_stochastic_seeds',
         '--runs', 'gia_stochastic_s23', 'gia_stochastic_s24', 'gia_stochastic_s25'],
        ['analysis/aggregate_runs.py', '--name', 'gia_deep_seeds',
         '--runs', 'gia_deep_s23', 'gia_deep_s24', 'gia_deep_s25',
         '--against', 'gia_stochastic_s23', 'gia_stochastic_s24', 'gia_stochastic_s25'],
    ]
    return runs, summaries


# a run is finished once its README exists; anything else in its folders is a half-finished attempt
def finished(name):
    return (REPO/'figures'/name/'README.md').exists()


# move a half-finished run's folders aside rather than deleting them, so nothing is lost
def set_aside(name, log):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    for parent in ('figures', 'checkpoints', 'renders'):
        folder = REPO/parent/name
        if folder.exists():
            target = folder.with_name('%s.incomplete-%s' % (name, stamp))
            folder.rename(target)
            log('  moved half-finished %s to %s' % (folder.relative_to(REPO), target.name))


# run one command from the repository root, echoing it and its duration into the batch log
def execute(command, log, dry_run):
    line = ' '.join(shlex.quote(str(c)) for c in [sys.executable, '-B', *command])
    log('$ ' + line)
    if dry_run:
        return True
    began = time.perf_counter()
    done = subprocess.run([sys.executable, '-B', *[str(c) for c in command]], cwd=REPO)
    log('  exit %d after %.1f min' % (done.returncode, (time.perf_counter() - began)/60))
    return done.returncode == 0


# run the paper's recorded experiments overnight; rerun the same command to resume after a stop
def main():
    parser = argparse.ArgumentParser(description="Run every recorded experiment the paper reports, resumably.")
    parser.add_argument('--dry_run', action='store_true', help='Print the plan and commands without running them')
    parser.add_argument('--only', nargs='+', help='Run just these run names (their references must exist)')
    args = parser.parse_args()

    log_path = REPO/'figures'/('batch_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not args.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        stamped = '[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), message)
        print(stamped, flush=True)
        if not args.dry_run:
            with open(log_path, 'a', encoding='utf-8') as handle:
                handle.write(stamped + '\n')

    missing = [str(p) for stone in POOLS.values() for key, p in stone.items()
               if not p.exists() and p != POOLS['step']['test']]
    if missing:
        raise FileNotFoundError('Missing pools: %s' % ', '.join(missing))

    runs, summaries = plan()
    log('%d runs and %d summaries planned' % (len(runs), len(summaries)))
    if not POOLS['step']['test'].exists():
        log('gathering a held-out pool for the step cut')
        if not execute(STEP_TEST_GATHER, log, args.dry_run):
            raise SystemExit('Could not gather the step-cut test pool')

    failed = []
    for name, command in runs:
        if args.only and name not in args.only:
            continue
        if finished(name):
            log('%s already finished, skipping' % name)
            continue
        set_aside(name, log) if not args.dry_run else None
        log('starting %s' % name)
        if not execute(command, log, args.dry_run):
            failed.append(name)
            log('%s FAILED; see figures/%s/logs/' % (name, name))

    for command in summaries:
        needed = [c for c in command if isinstance(c, str) and c.startswith('gia_') and c.endswith(('s23', 's24', 's25'))]
        if not args.dry_run and not all(finished(n) for n in needed):
            log('skipping %s: not all of its runs finished' % command[2])
            continue
        execute(command, log, args.dry_run)

    log('done; %d failed: %s' % (len(failed), ', '.join(failed) or 'none'))
    if not args.dry_run:
        log('log written to %s' % log_path)


if __name__ == '__main__':
    main()
