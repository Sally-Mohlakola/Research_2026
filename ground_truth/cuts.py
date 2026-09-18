"""Dispatch a variant's parameter dict to the right mesh generator.

A variant in config.parameters may carry a 'cut' key naming its generator.
Absent, it means 'round', so every existing variant and every checkpoint
gathered before pear cuts existed keeps working untouched.
"""
from ground_truth.brilliant_geometry import make_round_brilliant
from ground_truth.pear_geometry import make_pear_brilliant
from ground_truth.step_geometry import make_step_cut

GENERATORS = {'round': make_round_brilliant, 'pear': make_pear_brilliant,
              'step': make_step_cut}


def make_diamond(parameters):
    """Return (vertices, faces) for a variant dict, ignoring its IOR entries."""
    geometry = {k: v for k, v in parameters.items() if k not in ('int_ior', 'ext_ior')}
    cut = geometry.pop('cut', 'round')
    if cut not in GENERATORS:
        raise ValueError('Unknown cut %r; expected one of %s'
                         % (cut, ', '.join(sorted(GENERATORS))))
    return GENERATORS[cut](**geometry)
