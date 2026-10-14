import csv
import json
import sys
from pathlib import Path

import numpy as np

# keys that hold wall-clock timings or dates; they change on every run and are left out of comparisons
TIMING_WORDS = ('second', 'time', 'elapsed', 'date', 'duration', 'began', 'per_frame', 'rate', 'created',
                'started', 'finished', 'ms_per', 'us_per', 'microsecond', 'timing', 'share_of_render')

# files that are about the machine or the clock rather than the result, so they are never identical between runs:
# the run record, timing tables and plots, the network benchmark, quality against time (it divides by render time),
# training curves (one panel is time per epoch), and seed summaries (they include timing rows)
CLOCK_FILES = ('run.json', 'timing.csv', 'timing.png', 'inference_benchmark.csv', 'quality_vs_time.json',
               'quality_vs_time.csv', 'quality_vs_time.png', 'curves.png', 'summary.json', 'summary.csv')


# drop timing keys and replace the run's own folder in strings, so two runs in different folders compare equal
def strip(value, root):
    if isinstance(value, dict):
        return {k: strip(v, root) for k, v in value.items() if not any(w in k.lower() for w in TIMING_WORDS)}
    if isinstance(value, list):
        return [strip(v, root) for v in value]
    if isinstance(value, str):
        for form in {str(root), str(root).replace('\\', '/'), root.as_posix()}:
            value = value.replace(form, '<ROOT>')
        return value
    return value


# a csv without its timing columns
def load_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return [{k: v for k, v in r.items() if not any(w in k.lower() for w in TIMING_WORDS)} for r in rows]


# the numeric content of a file, as a dict of arrays or values, by file type
def load(path):
    suffix = path.suffix.lower()
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if suffix == '.exr':
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        return {'pixels': np.array(mi.Bitmap(str(path)))}
    if suffix == '.png':
        from PIL import Image
        return {'pixels': np.asarray(Image.open(path))}
    if suffix == '.gif':
        from PIL import Image, ImageSequence
        return {'frame%d' % i: np.asarray(f.convert('RGB'))
                for i, f in enumerate(ImageSequence.Iterator(Image.open(path)))}
    if suffix in ('.pt', '.pth'):
        import torch
        flat = {}

        def walk(item, key):
            if isinstance(item, dict):
                for k, v in item.items():
                    walk(v, key + '/' + str(k))
            elif isinstance(item, (list, tuple)):
                for i, v in enumerate(item):
                    walk(v, key + '/' + str(i))
            elif hasattr(item, 'numpy'):
                flat[key] = item.detach().cpu().numpy()
            else:
                flat[key] = item
        walk(torch.load(path, map_location='cpu', weights_only=False), '')
        return flat
    return None


def same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        return x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind in 'fc')
    return x == y


# compare two output trees file by file; returns a list of human-readable differences (empty = identical)
def compare_trees(reference, candidate, skip_suffixes=('.log', '.txt', '.md', '.mp4')):
    problems = []
    files_a = {p.relative_to(reference) for p in reference.rglob('*') if p.is_file()}
    files_b = {p.relative_to(candidate) for p in candidate.rglob('*') if p.is_file()}
    for rel in sorted(files_a - files_b):
        if rel.suffix not in skip_suffixes:
            problems.append('missing in this run: %s' % rel)
    for rel in sorted(files_b - files_a):
        if rel.suffix not in skip_suffixes:
            problems.append('new file in this run: %s' % rel)
    for rel in sorted(files_a & files_b):
        a, b = reference/rel, candidate/rel
        if rel.suffix in skip_suffixes or rel.name in CLOCK_FILES:
            continue
        if rel.suffix == '.json':
            if strip(json.loads(a.read_text()), reference) != strip(json.loads(b.read_text()), candidate):
                problems.append('json differs: %s' % rel)
            continue
        if rel.suffix == '.csv':
            if load_csv(a) != load_csv(b):
                problems.append('csv differs: %s' % rel)
            continue
        da, db = load(a), load(b)
        if da is None:
            if a.read_bytes() != b.read_bytes():
                problems.append('bytes differ: %s' % rel)
            continue
        for data, root in ((da, reference), (db, candidate)):
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = strip(value, root)
        if rel.suffix == '.npz' and 'metadata_json' in da and 'metadata_json' in db:
            for data, root in ((da, reference), (db, candidate)):
                data['metadata_json'] = json.dumps(strip(json.loads(str(data['metadata_json'])), root), sort_keys=True)
        differing = [k for k in sorted(set(da) | set(db)) if k not in da or k not in db or not same(da[k], db[k])]
        if differing:
            problems.append('data differs: %s %s' % (rel, differing[:5]))
    return problems


# compare a run of run_all.sh with the reference run and say what, if anything, changed
def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python -B tests/regression/compare.py <reference folder> <new folder>')
    reference, candidate = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    problems = compare_trees(reference, candidate)
    checked = sum(1 for p in candidate.rglob('*') if p.is_file())
    for problem in problems:
        print(problem)
    print('checked %d files against %s: %d differences' % (checked, reference.name, len(problems)))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
