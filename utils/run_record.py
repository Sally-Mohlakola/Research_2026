import csv
import datetime
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


# the current time in UTC, to the second, for every timestamp a run records
def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


# sha256 of a file, read in chunks so large pools and checkpoints don't load into memory at once
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# the command exactly as typed, so the run can be repeated from its own record
def command():
    return dict(argv=sys.argv, line=' '.join(shlex.quote(a) for a in sys.argv),
                python=sys.executable, working_directory=os.getcwd())


# the commit the code ran at, and whether anything was uncommitted at the time
def git_state():
    def git(*args):
        done = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    changed = git('status', '--porcelain')
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                uncommitted_files=len(changed.splitlines()) if changed else 0)


# software versions and the machine, the parts a reviewer needs to judge a timing
def environment():
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
    machine = dict(os=platform.platform(), processor=platform.processor() or platform.machine(),
                   logical_cpus=os.cpu_count())
    try:
        import psutil
        machine['memory_gb'] = round(psutil.virtual_memory().total/2**30, 1)
    except ImportError:
        machine['memory_gb'] = None
    return dict(python=platform.python_version(), machine=machine,
                packages={name: version(name) for name in ('mitsuba', 'drjit', 'torch', 'numpy', 'matplotlib')})


# label each triangle with its geometric facet: coplanar triangles share one label
def geometric_facets(vertices, faces, tolerance=1e-5):
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    normal = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    offset = (normal*corners[:, 0]).sum(1)
    label = np.full(len(corners), -1, dtype=np.int64)
    planes = []
    for index in range(len(corners)):
        for facet, (n, d) in enumerate(planes):
            if normal[index] @ n > 1-tolerance and abs(offset[index]-d) < 1e-4:
                label[index] = facet
                break
        else:
            label[index] = len(planes)
            planes.append((normal[index], offset[index]))
    return label


# the stone itself: cut parameters, counts, size and the geometry hash checkpoints are checked against
def mesh_record(diamond_name):
    from config.parameters import get_diamond_parameters
    from ground_truth.cuts import make_diamond
    parameters = get_diamond_parameters(diamond_name)
    vertices, faces = make_diamond(parameters)
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    corners = vertices[faces].astype(np.float64)
    normal = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    up = normal[:, 2]/np.linalg.norm(normal, axis=1)
    region = np.where(up > .999, 'table', np.where(up > .05, 'crown', np.where(up > -.05, 'girdle', 'pavilion')))
    return dict(
        diamond_name=diamond_name,
        parameters={k: v for k, v in parameters.items() if isinstance(v, (int, float, str))},
        vertices=int(len(vertices)), triangles=int(len(faces)),
        geometric_facets=int(geometric_facets(vertices, faces).max()+1),
        triangles_by_region={r: int((region == r).sum()) for r in ('table', 'crown', 'girdle', 'pavilion')},
        bounding_box=dict(min=vertices.min(0).round(6).tolist(), max=vertices.max(0).round(6).tolist()),
        input_radius=float(np.linalg.norm(vertices, axis=1).max()),
        geometry_sha256=hashlib.sha256(vertices.astype(np.float32).tobytes()+faces.astype(np.uint32).tobytes()).hexdigest())


# what a gathered pool contains: its gather metadata, record counts and outcome shares
def pool_record(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        status = archive['status']
        entries = int(len(np.unique(archive['entry_id'])))
    return dict(path=str(path), sha256=file_sha256(path), records=int(len(status)), entries=entries,
                escaped=float((status == 0).mean()), truncated=float((status == 1).mean()),
                invalid=float((status == 2).mean()),
                gather={k: meta.get(k) for k in ('diamond_name', 'seed', 'workers', 'max_depth', 'dispersion',
                                                 'entry_sampling', 'azimuth_range', 'geometry_sha256')})


# what a checkpoint is: head, size, parameter count and how it was trained
def model_record(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    buffers = {'triangles', 'tangent', 'bitangent', 'normal', 'shift_scale'}
    parameters = int(sum(v.numel() for k, v in state.items() if k not in buffers))
    return dict(path=str(path), sha256=file_sha256(path), head=checkpoint.get('head', 'mixture'),
                width=checkpoint.get('width'), components=checkpoint.get('components'),
                trainable_parameters=parameters, seed=checkpoint.get('seed'),
                best_epoch=checkpoint.get('best_epoch'), diamond_name=checkpoint['metadata'].get('diamond_name'),
                input_radius=float(checkpoint['input_radius']))


# wall-clock timing of a block, added to a dict under `name` in seconds
@contextmanager
def timed(record, name):
    began = time.perf_counter()
    try:
        yield
    finally:
        record[name] = record.get(name, 0.) + time.perf_counter() - began


# json written the same way everywhere: indented, keys in a stable order, numpy values made plain
def write_json(path, data):
    def plain(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError('Cannot write %r to json' % type(value))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=plain) + '\n', encoding='utf-8')


# rows of dicts as a csv with one header line, columns in first-seen order
def write_csv(path, rows):
    rows = list(rows)
    columns = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
