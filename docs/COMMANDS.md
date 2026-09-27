# Commands

Every method and ablation in this project, as the commands that run it. Flags
are taken from each script's own argument parser; defaults are omitted unless
they matter.

**Conventions**

- Run everything from the repository root, by path: `python -B pipeline/...`.
- `<name>` is a new output folder; most scripts refuse to overwrite one.
- Trained data lives in `checkpoints/` and renders in `renders/`. On
  27 September 2026 their contents were moved to `../checkpoint_history/` and
  `../renders_history/`; point input paths there, or move the folders back,
  before reusing an existing model or pool.
- The legacy scripts (`legacy/`) need Mitsuba's LLVM backend. It is missing on
  this laptop, so they run only where `legacy/eval.py` imports cleanly (for
  example the cluster the `slurm/` jobs target).

**Contents**

1. The learned boundary operator (the method)
2. Ablations of the learned operator
3. The RDM as an importance sampler for the operator
4. The RDM alone (the legacy Soh-and-Montazeri-style pipeline)
5. The LEAN-inspired sparkle
6. Learned dispersion and fire
7. Figures, measurement and tests
8. Results whose commands are not in the repo

---

## 1. The learned boundary operator

The operator replaces the whole journey of light inside the stone: given where
and how a ray enters, it predicts whether it escapes, through which facet,
where on it and in which direction. The first surface is traced exactly.

### Gather

```bash
# Round cut, camera-matched pool, held in memory (65,536 entries x 16 paths)
python -B pipeline/gather_boundary.py --output checkpoints/camera_pool/train.npz --entries 65536 \
  --paths_per_entry 16 --max_depth 128 --seed 401 --workers 8 --entry_sampling camera --azimuth_range 0 360
python -B pipeline/gather_boundary.py --output checkpoints/camera_pool/test.npz --entries 16384 \
  --paths_per_entry 16 --max_depth 128 --seed 402 --workers 8 --entry_sampling camera --azimuth_range 0 360

# Pear, large pools written as resumable shards (rerun the same command to continue)
python -B pipeline/gather_sharded.py --output_dir checkpoints/pear_pool_50m --records 50000000 \
  --workers 8 --max_seconds 480
python -B pipeline/gather_sharded.py --output_dir checkpoints/pear_pool_uniform_25m --records 25000000 \
  --entry_sampling uniform --workers 8 --max_seconds 480
```

`--diamond_name` picks the stone (`round_diamond_gia`, `pear_brilliant`,
`step_cut`, `step_cut_matched`, ...; see `config/parameters.py`).
`gather_sharded.py` defaults to `pear_brilliant`, `gather_boundary.py` to the
round cut.

### Train

```bash
# In memory: the mixture operator (default head)
python -B pipeline/train_boundary.py --data checkpoints/camera_pool/train.npz \
  --output checkpoints/camera_model --epochs 80

# Streaming over shards; the mixed model warm-starts from the 50M one
python -B pipeline/train_streaming.py --shards checkpoints/pear_pool_50m \
  --output checkpoints/pear_deep_50m --epochs 2 --max_seconds 540
python -B pipeline/train_streaming.py --shards checkpoints/pear_pool_50m checkpoints/pear_pool_uniform_25m \
  --init checkpoints/pear_deep_50m/model.pt --output checkpoints/pear_deep_mixed --max_seconds 540
```

`train_streaming.py` defaults to the `clone_deep` head; `--max_seconds` makes
it stop and save, and rerunning resumes.

### Render and evaluate

```bash
# Neural and analytic renders over 8 seeds (8 x 32 = 256 spp), merged and scored
python -B pipeline/render_paired.py --model checkpoints/camera_model/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 105 106 107 108 --workers 4

# A tumbled pose instead of the front view (pose 9 of the 60-frame tumble)
python -B pipeline/render_paired.py --model checkpoints/pear_deep_mixed/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 --workers 4 --rotation_deg 108 81 27

# Tumble animation, 60 frames (resumable)
python -B pipeline/animate_boundary.py --model checkpoints/pear_deep_mixed/model.pt --output_dir renders/<name> \
  --motion tumble --frames 60 --width 384 --height 384 --spp 32 --workers 4 --fps 30 [--resume]

# Side-by-side figure with the difference map
python -B analysis/compare_figure.py --model <model.pt> --neural renders/<name>/neural_az000.exr \
  --analytic renders/<name>/analytic_az000.exr --evaluation renders/<name>/evaluation.json --output <png>

# Score neural seed renders against an existing analytic reference (same seeds and size)
python -B analysis/evaluate_render.py --model <model.pt> \
  --analytic renders/<ref>/analytic/s101 ... renders/<ref>/analytic/s108 \
  --neural renders/<name>/neural/s101 ... renders/<name>/neural/s108 --output renders/<name>/evaluation.json
```

`render_paired.py --modes neural --skip_evaluation` renders the neural half
only, for reuse of an existing analytic reference.

---

## 2. Ablations of the learned operator

Each subsection is one experiment in `docs/FINDINGS.md`.

### Data ladder: is the model data-limited?

The ladder uses its own uniform-entry pools, gathered before camera matching.

```bash
# Uniform pools (training seed 101, held-out seed 202)
python -B pipeline/gather_boundary.py --output checkpoints/boundary_scale_v1/train_pool.npz --entries 65536 \
  --paths_per_entry 16 --max_depth 128 --seed 101 --workers 16
python -B pipeline/gather_boundary.py --output checkpoints/boundary_scale_v1/test_pool.npz --entries 16384 \
  --paths_per_entry 16 --max_depth 128 --seed 202 --workers 16

# Nested entry-grouped subsets; the largest rung (65,536) is the full pool
for n in 256 1024 4096 16384; do
  python -B analysis/make_boundary_subset.py --pool checkpoints/boundary_scale_v1/train_pool.npz \
    --output checkpoints/boundary_scale_v1/sub_$(printf %05d $n).npz --entries $n
done

# One model per rung, scored on the held-out pool
python -B pipeline/train_boundary.py --data checkpoints/boundary_scale_v1/sub_00256.npz \
  --output checkpoints/boundary_scale_v1/model_00256 --epochs <about 50k steps>
python -B analysis/evaluate_boundary.py --model checkpoints/boundary_scale_v1/model_00256/model.pt \
  --data checkpoints/boundary_scale_v1/test_pool.npz --output checkpoints/boundary_scale_v1/model_00256/test_metrics.json
```

Epochs per rung are chosen for roughly 50k gradient steps, so smaller rungs get
more epochs; each model's `metrics.json` records what it ran. Parallel gathers
reproduce only with the same `--workers` (16 here).

### Capacity

```bash
python -B pipeline/train_boundary.py --data checkpoints/boundary_scale_v1/train_pool.npz \
  --output checkpoints/boundary_scale_v1/final_w064c4 --width 64 --components 4 --epochs 80
python -B pipeline/train_boundary.py --data checkpoints/boundary_scale_v1/train_pool.npz \
  --output checkpoints/boundary_scale_v1/final_w256c8 --width 256 --components 8 --epochs 80
```

### Conditioning ablation: which inputs matter?

```bash
python -B analysis/ablate_conditioning.py --data checkpoints/camera_pool/train.npz \
  --test checkpoints/camera_pool/test.npz --output checkpoints/camera_model/conditioning_ablation.json --epochs 60
```

Six arms zero one input group each (position, direction, normal, wavelength).

### Behaviour cloning and the decoder sweep

```bash
# Deterministic heads instead of the mixture
python -B pipeline/train_boundary.py --data checkpoints/camera_pool/train.npz --output checkpoints/clone_model \
  --head clone --epochs 80                     # or --head clone_deep

# Decoder depth / separate trunk / loss weighting (resumable)
python -B analysis/improve_decoder.py --data checkpoints/camera_pool/train.npz \
  --test checkpoints/camera_pool/test.npz --output checkpoints/camera_model/decoder_ablation.json --epochs 30 --resume
```

### Oracles: how good could the operator be?

```bash
# Nearest-neighbour lookup on the same conditional (the 8.87 degree ceiling)
python -B analysis/oracle_boundary.py --train checkpoints/camera_pool/train.npz \
  --test checkpoints/camera_pool/test.npz --output checkpoints/camera_model/oracle.json

# Rendered: the true exit facet handed to the network (an upper bound, not a renderer)
python -B pipeline/render_paired.py --model checkpoints/camera_model/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 105 106 107 108 --workers 4 --oracle_facet
```

### Operator accuracy vs image accuracy

```bash
# Decode each exit at its component mean (no within-branch spread)
python -B pipeline/render_paired.py ... --deterministic_exit

# Exact split of every frame into the operator's contribution and the untouched rest
python -B pipeline/render_paired.py ... --split_components

# Exit-prediction error of several operators on the same data, without rendering
python -B analysis/compare_operators.py --data checkpoints/camera_pool/test.npz \
  --models mixture=checkpoints/camera_model/model.pt clone=checkpoints/clone_model/model.pt \
  --output <json> [--deterministic]
```

`...` is the paired-render command from section 1.

### Entry sampling: camera-matched, uniform and mixed

```bash
# The same gather with the other entry distribution
python -B pipeline/gather_boundary.py ... --entry_sampling uniform      # or camera
# Mixed training: see train_streaming.py with two --shards directories in section 1
```

### Scintillation

```bash
python -B analysis/flash_sweep.py --model checkpoints/camera_model/model.pt --output_dir renders/<name> \
  --azimuth_start 0 --azimuth_stop 45 --azimuth_step 1.5
```

### Cross-cut transfer: does an operator generalise to another stone?

```bash
# Same face count: round -> step cut, matching facets by normal (or --correspondence identity)
python -B pipeline/retarget_checkpoint.py --model checkpoints/camera_model/model.pt \
  --diamond_name step_cut_matched --output checkpoints/step_transfer/model.pt

# Different face count: pear (62) -> round (64)
python -B pipeline/retarget_checkpoint.py --model checkpoints/pear_deep_mixed/model.pt \
  --diamond_name round_diamond_gia --output checkpoints/gia_from_pear/model.pt --allow_resize
```

Then render the retargeted model with `render_paired.py`; the analytic half
traces the target stone for real.

### Sample count and wavelength sampling

```bash
# Same frame at 128 and 256 spp: add seeds 105-108 to double the samples
python -B pipeline/render_paired.py ... --seeds 101 102 103 104
python -B pipeline/render_paired.py ... --seeds 101 102 103 104 105 106 107 108

# Spread each pixel's hero wavelengths evenly over the spectrum
python -B pipeline/render_paired.py ... --stratify_wavelength
```

### Fourier input features (negative result, code removed)

The sweep and the Fourier heads were deleted after the result was recorded
(`docs/FINDINGS.md`, "Fourier features do not close the gap either"). Their
reports remain under `checkpoints/fourier_decoder/` and
`checkpoints/mixture_fourier/`.

---

## 3. The RDM as an importance sampler for the operator

Here the RDM is a directional histogram of the operator's own exits, used as a
defensive proposal: exits are drawn from `q = (1 - a) p + a p_RDM` and weighted
by `p / q`, so the image converges to the operator's answer and only its noise
changes. `a` is `--prior_fraction` (default 0.25).

```bash
# 1. Build the prior from the model's training entries (4 x 8 direction bins by default)
python -B analysis/build_boundary_prior.py --model checkpoints/<model>/model.pt \
  --data checkpoints/<pool>/train.npz --output checkpoints/<model>/rdm.npz

# 2. Render with and without it, two seeds each, same settings
python -B pipeline/render_boundary.py --model checkpoints/<model>/model.pt --output_dir renders/<name>_prior_s1 \
  --rdm_prior checkpoints/<model>/rdm.npz --prior_fraction 0.25 --width 160 --height 160 --spp 16 --seed 1
python -B pipeline/render_boundary.py --model checkpoints/<model>/model.pt --output_dir renders/<name>_base_s1 \
  --width 160 --height 160 --spp 16 --seed 1
#    ... and again with --seed 2 for both

# 3. Compare their image variance
python -B analysis/compare_boundary_sampling.py --baseline renders/<name>_base_s1 renders/<name>_base_s2 \
  --prior renders/<name>_prior_s1 renders/<name>_prior_s2 --output <json>
```

The prior must be built for the exact checkpoint it is used with; the renderer
checks the model's hash. `render_paired.py` deliberately does not expose
`--rdm_prior` (every paired result is RDM-free), so these renders use
`render_boundary.py` directly. The four-seed variance measurement quoted in
`docs/FINDINGS.md` came from `measure_prior_variance.py`, which is not in the
repo (section 8).

---

## 4. The RDM alone (legacy pipeline)

The project's starting point, after Soh and Montazeri's neural cloth model: an
RDM (a histogram of how light leaves the stone for each incoming direction) is
gathered, compressed into small networks, and used as a BSDF on the stone's
surface. `legacy/eval.py` always draws sampling directions from the RDM.

```bash
# 1. Gather the RDM (32 x 64 bins, paths split at k = 8 bounces, dispersive)
python -B legacy/gather_rdm.py --checkpoint_name <name> --diamond_name round_diamond_gia \
  --batch_size 1000000 --num_batches 30 --theta_bins 32 --phi_bins 64 --max_depth 64 --k 8 --dispersion

# 2. Train Model_M (multi-scatter) and Model_T
python -B legacy/train_models.py --checkpoint_name <name> --diamond_name round_diamond_gia \
  --width 128 --bands 6 --epochs_m <epochs> --epochs_t 2000 --batch_size 4096 --no_plot

# 3a. Render with the RDM alone: the aggregate BSDF, no explicit entry
python -B legacy/eval.py --checkpoint_name <name> --diamond_name round_diamond_gia --no_explicit_entry \
  --frames 1 --spp 32 --width 512 --height 512 --max_depth 64 --output_dir renders/<name>__agg --tile 128

# 3b. The hybrid ("Route A"): exact entry refraction, RDM for the rest
python -B legacy/eval.py --checkpoint_name <name> --diamond_name round_diamond_gia \
  --frames 1 --spp 32 --width 512 --height 512 --max_depth 64 --output_dir renders/<name>__routeA --tile 128

# Reference: analytic dielectric, no network
python -B legacy/eval.py --checkpoint_name <name> --diamond_name round_diamond_gia --no_neural \
  --frames 1 --spp 32 --width 512 --height 512 --max_depth 64 --output_dir renders/r18_ref
```

`--clamp_value 0` removes the output clamp (the `_off` variants);
`--no_dispersion` is the control for fire; `--tile 128` is needed for width
128 or Fourier-banded networks, which otherwise exceed memory. `--epochs_m` is
chosen to hold gradient steps near 160k (see `epochs_for` in
`legacy/experiments.py`).

### The legacy ablation matrix

`legacy/experiments.py` runs every legacy ablation as a two-stage SLURM job:
resolution (8x16 to 45x90), bounce split k (1 to 8), network width and Fourier
bands, gather size (12M to 100M rays), stone geometry, and dispersion on/off.
Each run renders the variants `routeA`, `routeA_off`, `agg`, `agg_off` and
`sparkle`.

```bash
python -B legacy/experiments.py --list                     # the manifest
python -B legacy/experiments.py --stage gather --index 0   # one gather
python -B legacy/experiments.py --stage run --index 0      # one training and its renders
python -B legacy/experiments.py --collect                  # results table
python -B legacy/experiments.py --emit-sbatch              # regenerate slurm/*.sbatch
bash slurm/submit.sh                                        # submit both stages
```

```bash
# Composite an explicit render with RDM aggregate on/off renders
python -B legacy/composite.py <explicit_dir> <aggregate_on_dir> <aggregate_off_dir> <out_dir>
```

---

## 5. The LEAN-inspired sparkle

The RDM stores only each bin's mean, which averages sparkle away. The sparkle
restores it from the gathered second moment: with probability
`p = 1 / (1 + r^2)` a shading point returns `value / p`, otherwise zero, decided
by a hash of its position, so the mean is unchanged and the variance comes back.
It needs a gather made after the second moment was recorded (any gather above).

```bash
python -B legacy/eval.py --checkpoint_name <name> --diamond_name round_diamond_gia \
  --no_explicit_entry --sparkle --sparkle_scale 0.02 \
  --frames 1 --spp 32 --width 512 --height 512 --max_depth 64 --output_dir renders/<name>__sparkle --tile 128
```

`--sparkle_scale` is the hashing lattice's edge length in scene units (girdle
radius 1); smaller gives finer grain. The `sparkle` variant of the ablation
matrix renders exactly this.

---

## 6. Learned dispersion and fire

```bash
# Paired-wavelength gather: every path at four wavelengths, colours replaying the hero's choices
python -B pipeline/gather_dispersion.py --output checkpoints/dispersion_pear/train.npz --entries 1500000 \
  --workers 4 --seed 41
python -B pipeline/gather_dispersion.py --output checkpoints/dispersion_pear/validation.npz --entries 200000 \
  --workers 4 --seed 43

# The go/no-go pilot (rotated lanes, as first run)
python -B pipeline/gather_dispersion.py --output checkpoints/dispersion_pilot/pear_100k.npz --entries 100000 \
  --workers 4 --seed 31 --wavelengths rotated

# Train the gate-and-shift head
python -B pipeline/train_dispersion.py --train checkpoints/dispersion_pear/train.npz \
  --validation checkpoints/dispersion_pear/validation.npz --output checkpoints/dispersion_pear/head --epochs 12

# Render with it (all four spectral lanes traced; neural mode only)
python -B pipeline/render_paired.py --model checkpoints/pear_deep_mixed/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 --workers 4 --modes neural --skip_evaluation \
  --rotation_deg 108 81 27 --dispersion checkpoints/dispersion_pear/head/model.pt

# Fire: colour that repeats between independent halves of a render
python -B analysis/measure_fire.py --model checkpoints/pear_deep_mixed/model.pt --rotation_deg 108 81 27 \
  --run analytic renders/<ref>/analytic --run neural renders/<ref>/neural \
  --run "neural + dispersion" renders/<name>/neural --output <json> --figure <png> --crop 176 208 112
```

---

## 7. Figures, measurement and tests

```bash
python -B figures/plot_results.py                  # ablation and accuracy figures (reads checkpoints/)
python -B figures/visualize_geometry.py            # facet outlines of the cuts
python -B figures/visualize_entry_sampling.py      # uniform vs camera entry sampling
python -B figures/denoise_chroma.py -h             # presentation denoising of chromatic speckle
python -B pipeline/merge_renders.py -h             # merge seed renders (render_paired does this)
python -B -m ground_truth.validate_meshes          # watertightness of all 9 meshes
python -B -m unittest discover tests               # the test suite
```

---

## 8. Results whose commands are not in the repo

Three analysis scripts produced numbers quoted in `docs/FINDINGS.md` but were
never committed (README section 6, item 1):

| script | result |
| --- | --- |
| `measure_within_facet.py` | within-facet decode error given the true facet |
| `compare_rdm_conditioning.py` | the RDM as a branch prior, by conditioning |
| `measure_prior_variance.py` | four-seed variance ratio of the RDM importance sampler |

Until they are recovered or rewritten, those numbers cannot be regenerated.

**Reproducibility.** A parallel gather reproduces only with the same seed and
worker count; `gather_sharded.py` reproduces regardless of worker count. Every
render records its settings in `render.json` (and `paired.json`), so any result
can be re-run from its own folder.
