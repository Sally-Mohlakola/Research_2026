import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import csv
import json
import shutil
import statistics
import time

import numpy as np

from pipeline.render_paired import run
from utils.run_record import (SCHEMA_VERSION, command, environment, file_sha256, git_state, mesh_record,
                              model_record, now, pool_record, write_csv, write_json)

HEADLINE_KEYS = ('pixel_correlation', 'pixel_noise_floor', 'disattenuated_correlation', 'pixel_rank_correlation',
                 'pixel_log_correlation', 'facet_correlation', 'facet_noise_floor', 'relative_energy_error',
                 'noise_free_contrast_analytic', 'noise_free_contrast_neural')

# how each metric is named for a reader; the csv and json keep the code names beside these
LABELS = dict(
    pixel_correlation='Pixel correlation (half-split)',
    pixel_noise_floor='Pixel noise floor (analytic against itself)',
    disattenuated_correlation='Pixel correlation, noise removed (disattenuated)',
    pixel_rank_correlation='Pixel rank correlation (robustness check)',
    pixel_log_correlation='Pixel log-luminance correlation (robustness check)',
    facet_correlation='Facet brightness correlation',
    facet_noise_floor='Facet noise floor (analytic against itself)',
    relative_energy_error='Relative energy error',
    noise_free_contrast_analytic='Noise-free contrast, analytic',
    noise_free_contrast_neural='Noise-free contrast, neural',
    pixel_correlation_ci95='Pixel correlation, 95% interval',
    facet_correlation_ci95='Facet correlation, 95% interval')


# the options: what to train (or which model to use), how to render it, and what to call the run
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='One recorded experiment: train (optional), time the network, render both modes, '
                    'evaluate, and write everything to figures/<name>/.')
    parser.add_argument('--name', required=True, help='Run name; used for figures/, checkpoints/ and renders/')
    parser.add_argument('--note', default='', help='One line on why this run exists')
    parser.add_argument('--no_index', action='store_true',
                        help='Do not add this run to figures/runs.csv (for test runs)')

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool', type=Path, help='Gathered pool to train a new operator on')
    source.add_argument('--model', type=Path, help='Existing checkpoint to evaluate instead of training')

    training = parser.add_argument_group('training (with --pool)')
    training.add_argument('--head', default='mixture', choices=('mixture', 'clone', 'clone_deep'))
    training.add_argument('--epochs', type=int, help='Fixed epoch count; leave out to train to convergence')
    training.add_argument('--tolerance', type=float, default=1e-3)
    training.add_argument('--patience', type=int, default=5)
    training.add_argument('--max_epochs', type=int, default=1000)
    training.add_argument('--batch_size', type=int, default=256)
    training.add_argument('--width', type=int, default=64)
    training.add_argument('--components', type=int, default=4)
    training.add_argument('--lr', type=float, default=1e-3)
    training.add_argument('--seed', type=int, default=23)
    training.add_argument('--test_pool', type=Path,
                          help='Independently gathered pool (different gather seed) for held-out operator scores')

    rendering = parser.add_argument_group('rendering')
    rendering.add_argument('--image_width', type=int, default=512)
    rendering.add_argument('--image_height', type=int, default=512)
    rendering.add_argument('--spp', type=int, default=32, help='Samples per pixel per seed')
    rendering.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103, 104])
    rendering.add_argument('--workers', type=int, default=4)
    rendering.add_argument('--camera', choices=('studio', 'spin'), default='studio')
    rendering.add_argument('--azimuth', type=float, default=0.)
    rendering.add_argument('--rotation_deg', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    rendering.add_argument('--scene_depth', type=int, default=8)
    rendering.add_argument('--rfilter', choices=('box', 'gaussian', 'tent'), default='box')
    rendering.add_argument('--stratify_wavelength', action='store_true')

    timing = parser.add_argument_group('inference benchmark')
    timing.add_argument('--benchmark_batches', type=int, nargs='+', default=[1, 64, 512])
    timing.add_argument('--benchmark_repeats', type=int, default=50)

    pairing = parser.add_argument_group('pairing with other runs')
    pairing.add_argument('--reference_run',
                         help='Reuse this run\'s analytic renders instead of rendering them again. The render '
                              'settings must match; the two runs are then compared on identical reference pixels.')
    pairing.add_argument('--compare_with', nargs='+', default=[],
                         help='Runs to compare against with a paired bootstrap (they must share the reference). '
                              'The reference run is always compared.')
    args = parser.parse_args()

    if len(args.seeds) < 2 or len(args.seeds) % 2:
        parser.error('Give an even number of seeds (at least two) so they split into equal halves')
    return args


# train a new operator through train_learned_interior, so the run uses exactly the shipped trainer
def train(args, folder, stages):
    output = REPO/'checkpoints'/args.name
    command_line = [REPO/'pipeline'/'train_learned_interior.py', '--data', args.pool, '--output', output,
                    '--head', args.head, '--batch_size', args.batch_size, '--width', args.width,
                    '--components', args.components, '--lr', args.lr, '--seed', args.seed]
    if args.epochs is not None:
        command_line += ['--epochs', args.epochs]
    else:
        command_line += ['--tolerance', args.tolerance, '--patience', args.patience,
                         '--max_epochs', args.max_epochs]
    began = time.perf_counter()
    run(command_line, folder/'logs'/'train.log')
    stages['train_seconds'] = time.perf_counter() - began

    (folder/'training').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output/'history.csv', folder/'training'/'history.csv')
    shutil.copyfile(output/'metrics.json', folder/'training'/'metrics.json')
    return output/'model.pt', [str(c) for c in command_line]


# how long the network takes per decision, outside the renderer: warm up, repeat, keep the median
def benchmark(model_path, pool, batches, repeats, threads=2):
    import torch
    from neural.learned_interior_model import encode_data, load_model
    torch.set_num_threads(threads)
    model, _ = load_model(model_path)
    model.eval()
    if pool is not None:
        from neural.learned_interior_data import load_learned_interior
        data, _ = load_learned_interior(pool)
        with np.load(pool, allow_pickle=False) as archive:
            vertices, faces = archive['vertices'], archive['faces']
        inputs = torch.from_numpy(encode_data(data, vertices, faces)[0])
        source = 'entry states from the training pool'
    else:
        inputs = torch.randn(4096, 10)
        source = 'random inputs (cost does not depend on the values)'
    generator = torch.Generator().manual_seed(0)
    rows = []
    for size in batches:
        batch = inputs[torch.randint(len(inputs), (size,), generator=generator)]
        for _ in range(5):
            model.sample(batch, generator)
        times = []
        for _ in range(repeats):
            began = time.perf_counter()
            model.sample(batch, generator)
            times.append(time.perf_counter() - began)
        median = statistics.median(times)
        rows.append(dict(batch_size=size, repeats=repeats, median_ms_per_batch=1e3*median,
                         p10_ms_per_batch=1e3*float(np.percentile(times, 10)),
                         p90_ms_per_batch=1e3*float(np.percentile(times, 90)),
                         median_us_per_query=1e6*median/size, torch_threads=threads))
    return rows, source


# the analytic renders of an earlier run, after checking they were made with this run's settings
def reference_inputs(args):
    renders = REPO/'renders'/args.reference_run
    seeds = [renders/'analytic'/('s%d' % s) for s in args.seeds]
    wanted = dict(width=args.image_width, height=args.image_height, spp=args.spp, azimuth=args.azimuth,
                  camera=args.camera, rotation_deg=args.rotation_deg, scene_depth=args.scene_depth,
                  rfilter=args.rfilter, stratify_wavelength=args.stratify_wavelength)
    for directory in seeds:
        report = directory/'render.json'
        if not report.exists():
            raise FileNotFoundError('%s has no analytic render for this seed: %s' % (args.reference_run, directory))
        found = json.loads(report.read_text())
        found.setdefault('camera', 'studio')
        different = {k: (found.get(k), v) for k, v in wanted.items() if found.get(k) != v}
        if different:
            raise ValueError('Reference run %s was rendered differently: %s' % (args.reference_run, different))
        if not (directory/'frames'/'component_operator.exr').exists():
            raise ValueError('Reference run %s has no interior split; it needs --split_components' % args.reference_run)
    return renders, seeds


# held-out operator scores on an independent pool: test losses and facet accuracy, then exit angles and branches
def test_scores(model_path, test_pool, folder, stages):
    began = time.perf_counter()
    losses, operator = folder/'metrics'/'test_losses.json', folder/'metrics'/'test_operator.json'
    run([REPO/'analysis'/'evaluate_learned_interior.py', '--model', model_path, '--data', test_pool,
         '--output', losses], folder/'logs'/'test_losses.log')
    run([REPO/'analysis'/'compare_operators.py', '--data', test_pool, '--models', 'run=%s' % model_path,
         '--output', operator], folder/'logs'/'test_operator.log')
    stages['test_seconds'] = time.perf_counter() - began
    test = json.loads(losses.read_text())['test']
    scores = json.loads(operator.read_text())
    mine = scores['operators']['run']
    return dict(pool=pool_record(test_pool), facet_accuracy=test['facet_accuracy'], joint_loss=test['joint_nll'],
                escape_bce=test['escape_bce'], facet_nll=test['facet_nll'],
                exit_angle_nearest_deg=mine['angle_nearest_deg'], exit_angle_matched_deg=mine['angle_matched_deg'],
                exit_position_nearest=mine['position_nearest'], facet_hit_rate=mine['facet_hit_rate'],
                distinct_facets_per_16=mine['distinct_facets_per_16'],
                within_branch_spread_deg=mine['within_branch_spread_deg'],
                real_distinct_facets_per_16=scores['reference']['distinct_facets_per_16'],
                entry_states=scores['entry_states'])


# render both modes over every seed (or only neural, against a reference run), merge and evaluate
def render(args, model_path, folder, stages):
    output = REPO/'renders'/args.name
    command_line = [REPO/'pipeline'/'render_paired.py', '--model', model_path, '--output_dir', output,
                    '--width', args.image_width, '--height', args.image_height, '--spp', args.spp,
                    '--seeds', *args.seeds, '--workers', args.workers, '--camera', args.camera,
                    '--azimuth', args.azimuth, '--scene_depth', args.scene_depth, '--rfilter', args.rfilter,
                    '--split_components']
    if args.rotation_deg:
        command_line += ['--rotation_deg', *args.rotation_deg]
    if args.stratify_wavelength:
        command_line.append('--stratify_wavelength')
    if args.reference_run:
        reference, analytic = reference_inputs(args)
        command_line += ['--modes', 'neural', '--skip_evaluation']
    else:
        reference, analytic = output, [output/'analytic'/('s%d' % s) for s in args.seeds]
    neural = [output/'neural'/('s%d' % s) for s in args.seeds]

    began = time.perf_counter()
    run(command_line, folder/'logs'/'render.log')
    commands = [[str(c) for c in command_line]]
    if args.reference_run:
        evaluate = [REPO/'analysis'/'evaluate_render.py', '--model', model_path,
                    '--output', output/'evaluation.json', '--analytic', *analytic, '--neural', *neural]
        run(evaluate, folder/'logs'/'evaluate.log')
        commands.append([str(c) for c in evaluate])
    stages['render_and_evaluate_seconds'] = time.perf_counter() - began
    inputs = dict(analytic=[str(p) for p in analytic], neural=[str(p) for p in neural],
                  analytic_image=str(reference/('analytic_az%03.0f.exr' % args.azimuth)),
                  neural_image=str(output/('neural_az%03.0f.exr' % args.azimuth)))
    return output, commands, inputs


# one row per seed and mode: how long the render took and how long its interior decisions took
def render_timing(renders):
    rows = []
    for report in sorted(renders.glob('*/s*/render.json')):
        r = json.loads(report.read_text())
        t = r.get('timing', {})
        rows.append(dict(mode=r['mode'], seed=r['seed'], width=r['width'], height=r['height'], spp=r['spp'],
                         render_seconds=t.get('render_seconds', r['seconds']),
                         interior_decision_seconds=t.get('interior_decision_seconds'),
                         interior_decisions=t.get('interior_decisions'),
                         microseconds_per_interior_decision=t.get('microseconds_per_interior_decision'),
                         interior_share_of_render=t.get('interior_share_of_render'),
                         mean_network_batch_size=t.get('mean_network_batch_size')))
    return rows


# a short per-mode summary of the timing rows
def timing_summary(rows):
    summary = {}
    for mode in sorted({r['mode'] for r in rows}):
        mine = [r for r in rows if r['mode'] == mode]
        decisions = sum(r['interior_decisions'] or 0 for r in mine)
        seconds = sum(r['interior_decision_seconds'] or 0. for r in mine)
        summary[mode] = dict(renders=len(mine), total_render_seconds=sum(r['render_seconds'] for r in mine),
                             mean_render_seconds=float(np.mean([r['render_seconds'] for r in mine])),
                             interior_decisions=decisions, interior_decision_seconds=seconds,
                             microseconds_per_interior_decision=1e6*seconds/decisions if decisions else None)
    return summary


# the headline metrics as a small table: one row per metric, whole stone and interior only
def headline_rows(evaluation):
    block = evaluation['headline']
    rows = []
    for key in HEADLINE_KEYS:
        whole, interior = block['whole_stone'], block['interior_only']
        rows.append(dict(label=LABELS[key], metric=key, whole_stone=whole.get(key),
                         interior_only=None if interior is None else interior.get(key)))
    for key in ('pixel_correlation_ci95', 'facet_correlation_ci95'):
        whole, interior = block['whole_stone'], block['interior_only']
        interval = lambda v: None if v is None else '[%.3f, %.3f]' % tuple(v)
        rows.append(dict(label=LABELS[key], metric=key, whole_stone=interval(whole[key]),
                         interior_only=None if interior is None else interval(interior[key])))
    return rows


# a readable summary of the run, written beside the data
def readme(record, headline, timing, convergence):
    def cell(value):
        if value is None:
            return 'n/a'
        if isinstance(value, float):
            return '%.4f' % value
        return str(value)
    model, mesh = record['model'], record['mesh']
    lines = [f"# {record['name']}", '']
    if record['note']:
        lines += [record['note'], '']
    lines += [f"Started {record['started']}, finished {record['finished']}. "
              f"Commit `{(record['git']['commit'] or 'unknown')[:10]}`"
              + (f" with {record['git']['uncommitted_files']} uncommitted files." if record['git']['uncommitted_files'] else '.'),
              '', '## Repeat this run', '', '```bash', record['command']['line'], '```', '',
              '## Conditions', '', '| | |', '|---|---|',
              f"| stone | {mesh['diamond_name']} ({mesh['triangles']} triangles, {mesh['geometric_facets']} facets) |",
              f"| operator | {model['head']}, width {model['width']}, {model['trainable_parameters']:,} trainable parameters |",
              f"| render | {record['render']['image_width']} x {record['render']['image_height']}, "
              f"{record['render']['spp']} spp x {len(record['render']['seeds'])} seeds, {record['render']['camera']} camera |"]
    if convergence:
        lines.append(f"| training | {convergence['stop_reason']} after {convergence['epochs_run']} epochs "
                     f"(best epoch {convergence['best_epoch']}) |")
    test = record.get('test')
    if test:
        lines += ['', '## Held-out operator scores', '',
                  f"On {test['entry_states']} entry states from an independently gathered pool "
                  f"(gather seed {test['pool']['gather']['seed']}):", '',
                  '| facet accuracy | exit angle, nearest branch | exit angle, matched branch | facet hit rate | '
                  'branches per 16 draws (real) | blur within a branch |', '|---|---|---|---|---|---|',
                  f"| {test['facet_accuracy']:.3f} | {test['exit_angle_nearest_deg']:.2f}° | "
                  f"{test['exit_angle_matched_deg']:.2f}° | {test['facet_hit_rate']:.3f} | "
                  f"{test['distinct_facets_per_16']:.2f} ({test['real_distinct_facets_per_16']:.2f}) | "
                  f"{cell(test['within_branch_spread_deg'])}° |"]
    lines += ['', '## Headline metrics', '', '| metric | whole stone | interior only |', '|---|---|---|']
    lines += [f"| {r['label']} | {cell(r['whole_stone'])} | {cell(r['interior_only'])} |" for r in headline]
    breakdown = record.get('breakdown')
    if breakdown:
        lines += ['', '## Where the operator goes wrong', '',
                  f"Held-out paths grouped by how many facets they hit inside the stone (exit included). "
                  f"Accuracy is how often the facet head's top choice is the recorded exit triangle; the exit error "
                  f"is decoded on the true triangle.", '',
                  '| facets hit | share of paths | exit facet accuracy | exit error, median | exit error, 90th percentile |',
                  '|---|---|---|---|---|']
        for r in breakdown['by_path_length']:
            lines.append(f"| {r['facet_hits']} | {100*r['share']:.1f}% ({r['paths']:,}) | {r['facet_accuracy']:.3f} | "
                         f"{r['exit_angle_median_deg']:.1f}° | {r['exit_angle_p90_deg']:.1f}° |")
        rank = breakdown['accuracy_vs_training_exits_rank_correlation']
        lines += ['', f"Per exit triangle, accuracy rises with the training paths that left through it "
                      f"(rank correlation {cell(rank)} over {breakdown['triangles_with_20_or_more_test_exits']} "
                      f"triangles with at least 20 held-out exits). See `operator/by_path_length.png` and "
                      f"`operator/by_facet.png`."]
    lines += ['', '## Time', '', '| mode | renders | mean render (s) | interior decisions | µs per decision |',
              '|---|---|---|---|---|']
    for mode, t in timing.items():
        lines.append(f"| {mode} | {t['renders']} | {t['mean_render_seconds']:.1f} | {t['interior_decisions']:,} | "
                     f"{cell(t['microseconds_per_interior_decision'])} |")
    for section, s in record.get('quality_vs_time', {}).items():
        if section == 'whole_stone':
            lines += ['', '## Quality against time', '',
                      f"The neural render cannot get closer than a relative RMSE of {s['neural_error_floor']:.3f} to "
                      f"the converged reference, however long it runs (its bias). In the time the neural render took "
                      f"({s['neural_seconds']:.0f} s) its error is {s['neural_error_at_its_time']:.3f}; the analytic "
                      f"render's is {s['analytic_error_at_equal_time']:.3f} in the same time. "
                      + (f"A single analytic seed render ({s['analytic_seconds_per_render']:.0f} s) is already "
                         f"closer to the converged reference than the neural render can ever get."
                         if s.get('analytic_below_neural_floor_after_one_render') else
                         f"The analytic render reaches the neural floor after about "
                         f"{s['analytic_seconds_to_reach_neural_floor']:.0f} s."
                         if s['analytic_seconds_to_reach_neural_floor'] is not None else ''),
                      '', 'See `render/quality_vs_time.png`.']
    for other, sections in record.get('comparisons', {}).items():
        block = sections.get('interior_only') or sections.get('whole_stone')
        which = 'interior light only' if sections.get('interior_only') else 'whole stone'
        lines += ['', f'## Paired against {other} ({which})', '',
                  '| metric | this run | ' + other + ' | difference | 95% interval | clear? |',
                  '|---|---|---|---|---|---|']
        for r in block['rows']:
            lines.append(f"| {r['metric']} | {r['run_a']:.4f} | {r['run_b']:.4f} | {r['difference']:+.4f} | "
                         f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {'yes' if r['interval_excludes_zero'] else 'no'} |")
    lines += ['', '## Files', '',
              '- `run.json`: every condition of the run (command, commit, software, machine, stone, data, model, '
              'settings, stage times, file hashes)',
              '- `training/`: per-epoch losses and times (`history.csv`), the trainer\'s report, `curves.png`',
              '- `render/`: per-render timing (`timing.csv`), the network benchmark (`inference_benchmark.csv`), '
              'quality against time (`quality_vs_time.*`), plots',
              '- `comparisons/`: paired bootstrap comparisons with other runs on the same reference',
              '- `metrics/`: the full evaluation (`evaluation.json`) and the headline table (`headline.csv`)',
              '- `operator/`: held-out errors by path length and by exit triangle, with training coverage per triangle',
              '- `images/`: the merged renders, the interior-only images and the comparison figure',
              '- `logs/`: the output of every stage', '']
    return '\n'.join(lines)


# record, train, benchmark, render, evaluate and draw, writing every result into figures/<name>/
def main():
    args = parse_arguments()
    folder = REPO/'figures'/args.name
    if folder.exists():
        raise FileExistsError('figures/%s already exists; pick a new --name' % args.name)
    (folder/'logs').mkdir(parents=True)
    began = time.perf_counter()
    stages = {}

    record = dict(schema_version=SCHEMA_VERSION, name=args.name, note=args.note, started=now(),
                  command=command(), git=git_state(), environment=environment(),
                  settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
    write_json(folder/'run.json', record)

    convergence, stage_commands = None, {}
    if args.pool is not None:
        record['data'] = pool_record(args.pool)
        model_path, stage_commands['train'] = train(args, folder, stages)
        convergence = json.loads((folder/'training'/'metrics.json').read_text()).get('convergence')
    else:
        model_path = args.model
    record['model'] = model_record(model_path)
    record['mesh'] = mesh_record(record['model']['diamond_name'])
    record['convergence'] = convergence

    (folder/'metrics').mkdir(exist_ok=True)
    record['test'] = test_scores(model_path, args.test_pool, folder, stages) if args.test_pool else None
    record['breakdown'] = None
    if args.test_pool and args.pool:
        began = time.perf_counter()
        run([REPO/'analysis'/'operator_breakdown.py', '--model', model_path, '--test_pool', args.test_pool,
             '--train_pool', args.pool, '--output_dir', folder/'operator', '--title', args.name],
            folder/'logs'/'operator_breakdown.log')
        stages['breakdown_seconds'] = time.perf_counter() - began
        breakdown = json.loads((folder/'operator'/'breakdown.json').read_text())
        record['breakdown'] = {k: v for k, v in breakdown.items() if k != 'by_facet'}
    write_json(folder/'run.json', record)

    t0 = time.perf_counter()
    rows, source = benchmark(model_path, args.pool, args.benchmark_batches, args.benchmark_repeats)
    stages['benchmark_seconds'] = time.perf_counter() - t0
    write_csv(folder/'render'/'inference_benchmark.csv', rows)
    record['inference_benchmark'] = dict(inputs=source, rows=rows)

    renders, stage_commands['render'], record['evaluation_inputs'] = render(args, model_path, folder, stages)
    record['reference_run'] = args.reference_run
    record['render'] = dict(image_width=args.image_width, image_height=args.image_height, spp=args.spp,
                            seeds=args.seeds, camera=args.camera, azimuth=args.azimuth,
                            rotation_deg=args.rotation_deg, scene_depth=args.scene_depth, rfilter=args.rfilter,
                            stratify_wavelength=args.stratify_wavelength, output=str(renders))
    timing_rows = render_timing(renders)
    write_csv(folder/'render'/'timing.csv', timing_rows)
    record['render_timing'] = timing_summary(timing_rows)

    (folder/'metrics').mkdir(exist_ok=True)
    shutil.copyfile(renders/'evaluation.json', folder/'metrics'/'evaluation.json')
    evaluation = json.loads((renders/'evaluation.json').read_text())
    headline = headline_rows(evaluation)
    write_csv(folder/'metrics'/'headline.csv', headline)
    record['headline'] = evaluation['headline']
    write_json(folder/'run.json', record)

    t0 = time.perf_counter()
    run([REPO/'analysis'/'quality_vs_time.py', '--run', args.name], folder/'logs'/'quality_vs_time.log')
    quality = json.loads((folder/'render'/'quality_vs_time.json').read_text())
    record['quality_vs_time'] = {s: b['summary'] for s, b in quality['sections'].items()}
    stages['quality_vs_time_seconds'] = time.perf_counter() - t0

    record['comparisons'] = {}
    others = list(dict.fromkeys(([args.reference_run] if args.reference_run else []) + args.compare_with))
    for other in others:
        t0 = time.perf_counter()
        run([REPO/'analysis'/'compare_runs.py', '--runs', args.name, other],
            folder/'logs'/('compare_%s.log' % other))
        stages['compare_%s_seconds' % other] = time.perf_counter() - t0
        result = json.loads((folder/'comparisons'/('%s_vs_%s.json' % (args.name, other))).read_text())
        record['comparisons'][other] = result['sections']

    t0 = time.perf_counter()
    run([REPO/'figures'/'plot_run.py', '--run', folder, '--renders', renders, '--model', model_path],
        folder/'logs'/'figures.log')
    run([REPO/'analysis'/'compare_figure.py', '--model', model_path,
         '--neural', record['evaluation_inputs']['neural_image'],
         '--analytic', record['evaluation_inputs']['analytic_image'],
         '--evaluation', renders/'evaluation.json', '--output', folder/'images'/'comparison.png',
         '--title', '%s: neural render against the analytic reference' % args.name,
         '--azimuth', args.azimuth, '--camera', args.camera,
         *(['--rotation_deg', *args.rotation_deg] if args.rotation_deg else [])],
        folder/'logs'/'comparison.log')
    stages['figures_seconds'] = time.perf_counter() - t0

    stages['total_seconds'] = time.perf_counter() - began
    record['stage_commands'] = stage_commands
    record['stages'] = stages
    record['finished'] = now()
    record['files'] = {str(p.relative_to(folder)).replace('\\', '/'): file_sha256(p)
                       for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'run.json'}
    write_json(folder/'run.json', record)
    (folder/'README.md').write_text(readme(record, headline, record['render_timing'], convergence), encoding='utf-8')

    index = REPO/'figures'/'runs.csv'
    whole = evaluation['headline']['whole_stone']
    interior = evaluation['headline']['interior_only'] or {}
    row = dict(name=args.name, finished=record['finished'], stone=record['mesh']['diamond_name'],
               head=record['model']['head'], commit=(record['git']['commit'] or '')[:10],
               pixel_correlation=whole['pixel_correlation'],
               disattenuated_correlation=whole['disattenuated_correlation'],
               facet_correlation=whole['facet_correlation'],
               interior_pixel_correlation=interior.get('pixel_correlation'),
               relative_energy_error=whole['relative_energy_error'],
               noise_free_contrast_neural=whole['noise_free_contrast_neural'],
               noise_free_contrast_analytic=whole['noise_free_contrast_analytic'],
               test_facet_accuracy=(record['test'] or {}).get('facet_accuracy'),
               test_exit_angle_matched_deg=(record['test'] or {}).get('exit_angle_matched_deg'),
               neural_error_floor=record['quality_vs_time'].get('whole_stone', {}).get('neural_error_floor'),
               seed=record['model']['seed'], reference_run=args.reference_run, note=args.note)
    new = not index.exists()
    if args.no_index:
        print('\n%s finished in %.0f s -> %s (not indexed)' % (args.name, stages['total_seconds'], folder))
        return
    with open(index, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)

    print('\n%s finished in %.0f s -> %s' % (args.name, stages['total_seconds'], folder))


if __name__ == '__main__':
    main()
