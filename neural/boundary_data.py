"""Validated boundary records and entry-grouped train/validation split."""
import json
import numpy as np


def load_boundary(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        if metadata.get('schema_version') != 1:
            raise ValueError('Unsupported boundary schema')
        names = ('entry_id', 'entry_position', 'entry_direction', 'entry_normal',
                 'entry_facet', 'entry_local_wi', 'wavelength_nm', 'throughput',
                 'exit_position', 'exit_direction', 'exit_normal', 'exit_facet', 'depth', 'status',
                 'entry_transmittance', 'wavelength_pdf', 'branch_log_probability', 'repeat_id')
        data = {name: archive[name].copy() for name in names}
    n = len(data['entry_id'])
    if not n or any(len(a) != n for a in data.values()):
        raise ValueError('Empty or inconsistent record arrays')
    if not np.isin(data['status'], [0, 1, 2]).all():
        raise ValueError('Unknown status code')
    escaped = data['status'] == 0
    for name in ('entry_position', 'entry_direction', 'entry_normal', 'entry_local_wi'):
        if data[name].shape != (n, 3) or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid {name}')
    for name in ('exit_position', 'exit_direction'):
        if data[name].shape != (n, 3) or not np.isfinite(data[name][escaped]).all():
            raise ValueError(f'Invalid escaped {name}')
    if not np.allclose(np.linalg.norm(data['entry_direction'], axis=1), 1, atol=1e-5):
        raise ValueError('Entry directions must have unit length')
    if not np.allclose(np.linalg.norm(data['exit_direction'][escaped], axis=1), 1, atol=1e-5):
        raise ValueError('Exit directions must have unit length')
    if not np.isfinite(data['throughput']).all() or (data['throughput'] < 0).any():
        raise ValueError('Invalid throughput')
    return data, metadata


def split_entries(entry_ids, validation_fraction=.2, seed=0):
    """Keep repeated outcomes from a single entry entirely within one split."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    unique = np.unique(entry_ids)
    if len(unique) < 2:
        raise ValueError('Need at least two distinct entries')
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique)-1, max(1, int(round(len(unique)*validation_fraction))))
    validation = np.isin(entry_ids, shuffled[:count])
    return ~validation, validation
