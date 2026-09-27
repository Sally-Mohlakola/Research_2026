# Commands

The commands that produced the current models and results, grouped by stage.
Run from the repository root. `<name>` is a new output folder of your choosing;
most scripts refuse to overwrite an existing one.

Paths assume trained data lives in `checkpoints/` and renders in `renders/`.
On 27 September 2026 their contents were moved to
`C:\Users\sally\RENDERING\checkpoint_history\` and `renders_history\`; point
input paths there, or move the folders back, before reusing an existing model
or pool.

## Gather

```bash
# Round cut, camera-matched pool, held in memory
python -B gather_boundary.py --output checkpoints/camera_pool/train.npz --entries 65536 \
  --paths_per_entry 16 --max_depth 128 --seed 401 --workers 8 --entry_sampling camera --azimuth_range 0 360

# Pear, large sharded pools (resumable: rerun the same command to continue)
python -B gather_sharded.py --output_dir checkpoints/pear_pool_50m --records 50000000 --workers 8 --max_seconds 480
python -B gather_sharded.py --output_dir checkpoints/pear_pool_uniform_25m --records 25000000 \
  --entry_sampling uniform --workers 8 --max_seconds 480

# Dispersion: every path traced at four wavelengths
python -B gather_dispersion.py --output checkpoints/dispersion_pear/train.npz --entries 1500000 --workers 4 --seed 41
python -B gather_dispersion.py --output checkpoints/dispersion_pear/validation.npz --entries 200000 --workers 4 --seed 43
```

## Train

```bash
# Operator, in memory (mixture is the default head)
python -B train_boundary.py --data checkpoints/camera_pool/train.npz --output checkpoints/camera_model --epochs 80
#   other heads: --head clone | clone_deep

# Operator, streamed from shards; the mixed model warm-starts from the 50M one
python -B train_streaming.py --shards checkpoints/pear_pool_50m --output checkpoints/pear_deep_50m \
  --epochs 2 --max_seconds 540
python -B train_streaming.py --shards checkpoints/pear_pool_50m checkpoints/pear_pool_uniform_25m \
  --init checkpoints/pear_deep_50m/model.pt --output checkpoints/pear_deep_mixed --max_seconds 540

# Dispersion head
python -B train_dispersion.py --train checkpoints/dispersion_pear/train.npz \
  --validation checkpoints/dispersion_pear/validation.npz --output checkpoints/dispersion_pear/head --epochs 12

# Sweeps: each trains and scores several models on one protocol
python -B improve_decoder.py --data checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz \
  --output <json> --epochs 30 --resume
python -B ablate_conditioning.py --data checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz \
  --output <json> --epochs 60
python -B oracle_boundary.py --train checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz --output <json>
```

## Render

```bash
# Paired neural and analytic renders, evaluated against each other (8 x 32 = 256 spp)
python -B render_paired.py --model checkpoints/camera_model/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 105 106 107 108 --workers 4
#   --rotation_deg 108 81 27                                  a tumble pose instead of the front view
#   --modes neural --skip_evaluation                          neural render only
#   --dispersion checkpoints/dispersion_pear/head/model.pt    learned dispersion (neural only)
#   --stratify_wavelength                                     spread wavelengths evenly per pixel
#   --split_components                                        also write the operator/untouched split

# Tumble animation (resumable)
python -B animate_boundary.py --model checkpoints/pear_deep_mixed/model.pt --output_dir renders/<name> \
  --motion tumble --frames 60 --width 384 --height 384 --spp 32 --workers 4 --fps 30 [--resume]

# Put a trained operator on another cut (no retraining)
python -B retarget_checkpoint.py --model checkpoints/pear_deep_mixed/model.pt --diamond_name round_diamond_gia \
  --output checkpoints/gia_from_pear/model.pt --allow_resize
```

## Measure

```bash
# Evaluate neural seed renders against analytic ones (e.g. a reused reference)
python -B evaluate_render.py --model <model.pt> --analytic <analytic seed dirs...> \
  --neural <neural seed dirs...> --output <json>

# Side-by-side figure with the difference map
python -B compare_figure.py --model <model.pt> --neural renders/<name>/neural_az000.exr \
  --analytic renders/<name>/analytic_az000.exr --evaluation renders/<name>/evaluation.json --output <png>

# Fire: colour that repeats between independent halves of a render
python -B measure_fire.py --model <model.pt> --run analytic renders/<name>/analytic \
  --run neural renders/<name>/neural --output <json> [--rotation_deg X Y Z --figure <png> --crop X Y SIZE]

# Result figures
python -B plot_results.py
```

## Tests

```bash
python -B -m unittest discover tests
```

## Notes

- A parallel gather reproduces only with the same seed **and** worker count;
  `gather_sharded.py` reproduces regardless of worker count.
- Render settings are recorded in each render's `render.json` and `paired.json`,
  so any result can be re-run from its own folder.
- Section 9 of `README.md` lists the same commands for the original models.
