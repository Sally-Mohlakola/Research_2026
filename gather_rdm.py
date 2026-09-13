from config.parameters import get_diamond_parameters
from utils.rdm import compute_rdm

import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Gather RDM for a diamond.")

parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs", required=True)
parser.add_argument("--diamond_name", help="Name of the diamond preset defined in config/parameters.py",
                     required=True)
parser.add_argument("--batch_size", type=int, help="Samples per batch for RDM collection", default=1024 * 8)
parser.add_argument("--num_batches", type=int, help="Number of batches for RDM collection", default=1024)
parser.add_argument("--theta_bins", type=int, help="Theta resolution of the RDM", default=18)
parser.add_argument("--phi_bins", type=int, help="Phi resolution of the RDM", default=36)
parser.add_argument("--max_depth", type=int, help="Max path bounces before Russian-roulette/truncation",
                     default=64)
parser.add_argument("--sampling_method", type=str, help="Sampling method: uniform, cos_theta, or stratified",
                     default="stratified", choices=["uniform", "cos_theta", "stratified"])
parser.add_argument("--no_aim_jitter", action="store_true",
                     help="Aim every gather ray at the stone's centre instead of sweeping its "
                          "projected disc. Reproduces pre-fix checkpoints (run_09, run_14); "
                          "leaves most (facet, incidence angle) bins unreachable.")

parser.add_argument("--max_samples", type=int, default=0,
                     help="Keep this many raw multi-scatter (wi, wo, value) samples "
                          "alongside the histogram, written to samples.npz. They let "
                          "the outgoing bin size be chosen at TRAINING time instead of "
                          "here, so changing angular resolution costs a retrain rather "
                          "than a fresh gather. 2000000 is about 70 MB.")

parser.add_argument("--dispersion", action="store_true",
                     help="Gather through the wavelength-dependent DispersiveDielectric "
                          "instead of Mitsuba's achromatic `dielectric`. Without this the "
                          "RDM's per-bin channel spread is exactly zero, so the neural "
                          "term can never contribute colour and the render comes out a "
                          "uniform grey-tan.")

parser.add_argument("--k", type=int, default=2,
                     help="Bounce-count boundary of the decomposition. rdm_m keeps only "
                          "paths with depth > k, so the explicit half rendered with "
                          "eval.py --max_depth k and this residual partition the transport "
                          "exactly. k=2 is the historical default; raising it moves work "
                          "from the learned term to the analytic one.")

args = parser.parse_args()

output_dir = os.path.join('checkpoints/', args.checkpoint_name)
os.makedirs(output_dir, exist_ok=True)

diamond_kwargs = get_diamond_parameters(args.diamond_name)

(rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa, samples,
 var_m, count_cell_m, mean_cell_m) = compute_rdm(
    theta_bins=args.theta_bins,
    phi_bins=args.phi_bins,
    max_depth=args.max_depth,
    num_batches=args.num_batches,
    batch_size=args.batch_size,
    diamond_kwargs=diamond_kwargs,
    sampling_method=args.sampling_method,
    aim_jitter=not args.no_aim_jitter,
    max_samples=args.max_samples,
    dispersion=args.dispersion,
    k=args.k,
)

(rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa,
 var_m, count_cell_m, mean_cell_m) = [
    arr.numpy() for arr in [rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa,
                            var_m, count_cell_m, mean_cell_m]
]

print("Saving to", output_dir)

np.savez(
    os.path.join(output_dir, "rdm.npz"),
    rdm_t=rdm_t, rdm_r=rdm_r, rdm_m=rdm_m,
    count_t=count_t, count_r=count_r, count_m=count_m,
    # Within-cell spread of the multi-scatter component, and the per-cell path
    # count it was computed from. rdm_m holds the mean, which is the smoothest
    # summary the paths in a cell admit; var_m says how much detail that mean
    # threw away, so a renderer can draw a value with the right spread instead
    # of always returning the mean. Threshold on count_cell_m -- a cell with
    # one path reports zero variance, which is an absence of evidence rather
    # than evidence of a smooth cell.
    var_m=var_m, count_cell_m=count_cell_m, mean_cell_m=mean_cell_m,
    x=x, sa=sa,
    diamond_name=args.diamond_name,
    theta_bins=args.theta_bins,
    phi_bins=args.phi_bins,
    aim_jitter=not args.no_aim_jitter,
    dispersion=args.dispersion,
    k=args.k,
)



if samples is not None:
    sample_path = os.path.join(output_dir, "samples.npz")
    np.savez_compressed(sample_path, **samples, diamond_name=args.diamond_name,
                        count_i=count_t)
    print("Saved %d raw samples to %s" % (len(samples["wi"]), sample_path))

print("Done")