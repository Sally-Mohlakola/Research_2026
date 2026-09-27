# Step cut rendered with the learned operator

One front-view frame of the step (emerald-style) cut, rendered with a boundary
operator trained on the step cut itself and compared against the analytic
reference. Run on 27 September 2026.

The operator is trained natively rather than retargeted from another stone, so
nothing is borrowed from the round or pear cuts: the gather, the training and
the render all use the step cut's own geometry (`step_cut`, 58 vertices, 112
triangles; see `ground_truth/step_geometry.py`). The recipe is the one that
produced the round cut's `camera_model`, so the result is comparable with it.

Run every command from the repository root.

## 1. Gather

Trace true interior transport through the step cut for camera-matched entry
states: 65,536 entry states, 16 paths each, 1,048,576 records.

```bash
python -B pipeline/gather_boundary.py --output checkpoints/step_pool/train.npz --diamond_name step_cut \
  --entries 65536 --paths_per_entry 16 --max_depth 128 --seed 401 --workers 8 \
  --entry_sampling camera --azimuth_range 0 360
```

The gather reproduces exactly only with the same `--seed` and `--workers`.

## 2. Train

The mixture operator (the default head), width 64, four components, seed 23,
80 epochs, selected on validation joint NLL over an entry-grouped split.

```bash
python -B pipeline/train_boundary.py --data checkpoints/step_pool/train.npz \
  --output checkpoints/step_model --epochs 80
```

## 3. Render

Neural and analytic renders over four seeds of 32 samples per pixel (128
effective), 512 x 512, front view, merged per mode and scored against each
other with measured noise floors.

```bash
python -B pipeline/render_paired.py --model checkpoints/step_model/model.pt --output_dir renders/step_learned \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 --workers 4
```

## 4. Figure

```bash
python -B analysis/compare_figure.py --model checkpoints/step_model/model.pt \
  --neural renders/step_learned/neural_az000.exr --analytic renders/step_learned/analytic_az000.exr \
  --evaluation renders/step_learned/evaluation.json --output renders/step_learned_comparison.png \
  --title "Step cut: neural render against the analytic reference"
```

## Outputs

| what | where |
| --- | --- |
| training pool | `checkpoints/step_pool/train.npz` |
| operator | `checkpoints/step_model/model.pt`, training report `metrics.json` |
| renders | `renders/step_learned/` (`neural_az000.png`, `analytic_az000.png`, per-seed renders under `neural/` and `analytic/`) |
| evaluation | `renders/step_learned/evaluation.json` |
| figure | `renders/step_learned_comparison.png` |

## Results

Timings on this laptop: gather about 1 minute, training 20 minutes, the eight
renders 21 minutes (1,257 s).

**Operator.** Best at epoch 80 (still improving), validation joint NLL 7.456,
exit-facet top-1 0.305 on 838,852 training records. The step cut has 112
triangles to choose between against the round cut's 64, so these figures are not
comparable with the round cut's 6.407 and 0.371.

**Image**, 128 effective spp, 93,618 stone pixels; agreement is measured between
independent halves, with the analytic noise floor alongside:

| | learned operator | analytic reference | noise floor |
| --- | --- | --- | --- |
| correlation | 0.293 | -- | 0.942 |
| highlight overlap (top 1%) | 0.059 | -- | 0.650 |
| relative energy | +14.2% | -- | -0.3% |
| contrast | 0.84 | 2.37 | -- |
| peak / mean | 8.0 | 112.1 | -- |
| flash fraction | 0.0001 | 0.0149 | -- |

![Step cut, neural render against analytic](../renders/step_learned_comparison.png)

**Reading it.** The large-scale layout is right: a bright table, bright step
bands along the long sides and a darker centre, which is why correlation (0.29)
is higher than on the round cut (0.17). Everything finer is missing. The
reference's sharp step reflections and the crisp dark bars through the centre
become soft gradients, the brightest flashes reach a fourteenth of the
reference's peak, almost no pixels flash, and the image is 14% too bright. The
step cut's broad, flat facets suit a smooth operator better than the round
brilliant's many small ones, but the sharp detail is lost here too.
