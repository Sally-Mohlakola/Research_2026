# Neural Diamond Rendering — Current Status

*Last updated 26 September 2026. This file replaces the earlier `README.md`
(remaining work), `EXPERIMENTS.md` (experimental protocol) and `CODE_FREEZE.md`
(code freeze plan); all three are in git history.*

Honours research, Sally Mohlakola. About a month remains, from 25 September, for
the write-up and completion.

---

## 1. What this project is

A learned **boundary transport operator** for faceted gemstones: a small neural
network that replaces the entire journey of light inside a diamond with one
query — given where light enters, in which direction and at what wavelength, it
predicts whether and where it leaves, through which facet and in which
direction. It began as an adaptation of the radiance-distribution-map (RDM)
method of Soh and Montazeri for cloth, and was generalised after the RDM was
shown to discard the variable that matters for a faceted stone: position.

**Research question.** Do aggregate neural appearance models transfer from
fibrous media, where appearance is smooth at the aggregation scale, to resolved
specular media, where the fine structure *is* the appearance?

**Answer.** No. The operator learns *how much* light leaves the stone but not
*where* it goes, and the limit lies in the representation, not in data, capacity
or training distribution.

**The single most useful result.** Operator accuracy does not transfer to the
image: in four separate experiments, improving the operator made the rendered
image agree with the reference *less*. A sharp operator placing light wrongly is
further from the truth than a blurred one.

---

## 2. Where things stand

Details, tables and provenance are in `FINDINGS.md`.

| result | evidence |
| --- | --- |
| the RDM conditions on the wrong variable | entry direction predicts the exit facet at 6.8 %, barely above guessing (4.65 %); entry facet (a position proxy) 19.2 % |
| the operator learns | exit-facet accuracy 41.0 % (clone), beating every lookup including a nearest-neighbour oracle over a million paths (28.7 %) |
| the image does not follow | round cut, 256 spp: spatial correlation 0.174 against a noise floor of 0.976 |
| the harness is sound | light untouched by the network agrees with the reference at 0.96 in every run |
| the network carries the image | 75.4 % of image energy, 85.5 % of the stone's |
| not data | data ladder flattens above 16k entry states; 48× more data (1M → 50M records) raises facet accuracy 33.7 % → 37.2 % and correlation 0.054 → 0.095 |
| not capacity | width 256 gains 0.33 nats then overfits |
| not training distribution | camera-matched data fixed energy (+15.8 % → −1.1 %), not placement |
| not density sharpness or family | deterministic decode and behaviour cloning both render worse |
| not geometry | the pear cut reproduces the failure |
| position matters, wavelength does not | conditioning ablation: direction only 11.7 %, all inputs 40.4 %; removing wavelength costs 0.0004 |
| two-part failure | wrong exit facet ~60 % of the time; 24.7° exit error even given the correct facet (a lookup manages 8.9°) |
| the decoder saturates | a deeper head: 24.5° → 21.6°, only 18.6 % of the gap to the lookup |
| operator gains don't transfer | four instances; the one partial exception is more data within a fixed architecture |
| the RDM doesn't help | as a sampler 1.32× more variance (1.60× at equal time); as a branch prior +0.3 points at best |
| camera-only training has a blind spot | 0 % of training entries on the pavilion; a tumbling stone blacks out (escape 46 %); mixed training restores escape to 100 % and energy to within ~28 %, but not placement (correlation 0.018 vs 0.951) |
| it costs more | 1.46× slower per sample, 1.57× noisier per pixel |

---

## 3. Documents

| file | what it is |
| --- | --- |
| `README.md` | this status document |
| `FINDINGS.md` | every result with tables and sources |
| `PAPER_DRAFT.md` | full draft paper, abstract to conclusion |
| `METHODOLOGY_EVOLUTION.md` | RDM → LEAN-inspired sparkle → learned operator, with 33 references |
| `CORRECTIONS-2026-09-18.md` | dated record of claims revised when measurements contradicted them |
| `NEURAL_DIAMOND_NEXT.md` | the original implementation contract; historical, cited by the methodology |
| `DENOISING.md` | rules for the presentation-only chroma denoiser |

---

## 4. The code

The operator shares no code with the RDM system (checked by imports).

**The operator pipeline**

| script | role |
| --- | --- |
| `gather_boundary.py` | trace true interior transport; uniform (with aim offset) or camera-matched entry sampling; preallocated arrays, bit-identical to the original |
| `gather_sharded.py` | the same gather in resumable on-disk shards, for pools that don't fit in memory |
| `neural/boundary_model.py` | the operator: mixture, clone and deep-clone heads; data encoding |
| `neural/boundary_data.py` | loading and entry-grouped splitting |
| `train_boundary.py` | in-memory training |
| `train_streaming.py` | streaming training over shards; mixes several pools; warm start; resumable |
| `render_boundary.py` | spectral renderer calling the operator for interior transport; supports tumbled poses and stratified wavelengths |

**Evaluation and experiments**

| script | produces |
| --- | --- |
| `render_paired.py` | neural and analytic renders over several seeds, merged and evaluated; supports rotated poses |
| `merge_renders.py`, `evaluate_render.py`, `compare_figure.py` | seed merging, masked comparison with measured noise floors, side-by-side figures |
| `evaluate_boundary.py`, `compare_operators.py` | operator-level metrics on held-out transport |
| `oracle_boundary.py` | nearest-neighbour oracle |
| `ablate_conditioning.py` | conditioning ablation |
| `improve_decoder.py` | decoder sweep (resumable) |
| `retarget_checkpoint.py` | re-point a trained operator at another cut with the same face count |
| `animate_boundary.py` | orbit and tumble animations (resumable) |
| `flash_sweep.py` | scintillation across azimuths |
| `plot_results.py`, `visualize_geometry.py`, `visualize_entry_sampling.py` | figures |
| `make_boundary_subset.py`, `build_boundary_prior.py`, `denoise_chroma.py` | data ladder subsets, the boundary RDM prior, presentation denoising |

**Geometry** (`ground_truth/`): round, pear and step cut generators, dispatched
by `cuts.py`; `pear_validation.py` validates all three.

**Legacy RDM pipeline**, kept as history and baseline: `eval.py`, `gather_rdm.py`,
`train_models.py`, `experiments.py`, `slurm/`, `composite.py`,
`integrators/depth_aware.py`, `utils/rdm.py`, `bsdf/neural_bsdf.py` (including the
LEAN-inspired sparkle), `bsdf/rdm_sampler.py`, `neural/base_model.py`,
`neural/drjit_wrapper.py`. Shared with the operator: `bsdf/dispersion.py`,
`utils/studio_env.py`, `ground_truth/brilliant_geometry.py`.

**Key data and results.** Checkpoints: `camera_model` (mixture), `clone_model`,
`pear_model`, `pear_deep`, `pear_deep_50m`, `pear_deep_mixed`, `boundary_scale_v1`
(data ladder); pools `camera_pool`, `pear_pool`, `pear_pool_50m`,
`pear_pool_uniform_25m`. Renders (renamed since some documents were written):
`neural_gia_00` (round, mixture), `neural_gia_01` (deterministic decode),
`neural_gia_clone_01` (behaviour clone), `neural_gia_clone_00` (oracle facet),
`neural_pear_00`, `pear_deep`, `pear_deep_50m`, `pear_tumble`,
`pear_tumble_mixed`, `pear_mixed_pose09`, `figures/`.

---

## 5. Experiment status

**Done**

| experiment | result lives in |
| --- | --- |
| data ladder | `checkpoints/boundary_scale_v1/ladder_summary.json` |
| capacity probe, two widths | `FINDINGS.md` |
| camera-matched training | `checkpoints/camera_pool`, `camera_model` |
| paired render comparison, round and pear | `renders/neural_gia_00`, `renders/neural_pear_00` |
| component decomposition | every `evaluation.json` |
| deterministic decode | `renders/neural_gia_01` |
| behaviour cloning | `checkpoints/clone_model`, `renders/neural_gia_clone_01` |
| oracle facet | `renders/neural_gia_clone_00` |
| nearest-neighbour oracle | `checkpoints/camera_model/oracle.json` |
| conditioning ablation | `checkpoints/camera_model/conditioning_ablation.json` |
| decoder sweep | `checkpoints/camera_model/decoder_ablation.json` |
| 50M-record training | `checkpoints/pear_deep_50m`, `renders/pear_deep_50m` |
| RDM as sampler and as branch prior | `FINDINGS.md` (scripts not yet in repo — see §6) |
| cross-cut: pear | `checkpoints/pear_*` |
| tumble blind spot and its fix | `renders/pear_tumble*`, `renders/pear_mixed_pose09` |
| mode multiplicity | effectively answered: median 2 exit facets per entry state, zero spread within a facet |

**Partial**

- **Capacity:** only two points; the decoder sweep adds head-size points.
- **Multiple views:** one view at full resolution, fifteen at 96×96, plus the
  frame-9 tumble pose at 128 spp.
- **Coverage:** neighbour distances measured by the oracle; the histogram figure is
  not yet produced.
- **Step cut:** geometry built and validated, transfer operator built; the render
  never completed (lost to memory pressure).
- **Equal-time comparison:** measured for the RDM prior only.

**Open**

- **Analytic reference convergence** — never shown, and every result is measured
  against it.
- **Sample/PDF consistency** — validation gate 1, untested.
- **Held-out lighting** — needs a second rig in `utils/studio_env.py`.
- **Table operator** — named in the original contract, never run.
- **Mixture component sweep** — lower priority now: the deterministic clone also
  fails, so the claim no longer rests on the number of Gaussians.
- **Fourier input encoding** — the one untried improvement the evidence points to
  (`Model_M` improved 12× with it on this stone).

---

## 6. Remaining work, in order

**Needed for the thesis**

1. **Commit the three analysis scripts** (~1 h). `measure_within_facet.py`,
   `compare_rdm_conditioning.py` and `measure_prior_variance.py` produced numbers
   quoted in `FINDINGS.md` and `PAPER_DRAFT.md` but exist only in a temp
   directory, so those numbers cannot currently be reproduced.
2. **Analytic convergence check** (~2 h). One view at increasing spp; show masked
   error falling to the highest.
3. **Sample/PDF consistency test** (~half a day), or state it as a limitation.
4. **Bring `FINDINGS.md` up to date**: the render folders were renamed, and the
   50M, mixed-training and tumble results are not yet written in.
5. **Write** — see §10.

**Worth doing**

- Tag the commit that produced the thesis results (`git tag thesis-results`)
  before any refactor.
- Pass `--stratify_wavelength` through `render_paired.py` and
  `animate_boundary.py` (off by default) to reduce colour speckle for free.
- A from-scratch model on the mixed data, if the mixed-training numbers go in the
  thesis (the current one is fine-tuned).

**Optional, only if writing is on track by the end of week 2**

- Fourier input encoding, time-boxed to three days; either outcome is written up.
- Table operator; held-out lighting; step-cut transfer render.

---

## 7. Tests

Fourteen tests pass: `python -B -m unittest discover -s tests -p 'test_boundary*.py'`.

To add, ordered by the claim each protects:

1. **Sample/PDF consistency** — samples histogrammed against `exp(log_exit_pdf)`.
2. **PDF normalisation** — the density integrates to one; catches Jacobian errors
   the first test can miss.
3. **Importance weight has unit mean** under the prior mixture.
4. **Component split is exact** — operator + untouched equals the frame.
5. **Facet override is honoured** — `sample(x, facet=f)` returns facet `f`.
6. **Retarget preserves weights** and recomputes the geometry hash.
7. **Mesh properties** for randomised parameters, all three cuts.
8. **`encode_data` round-trip** recovers exit position and direction.
9. **Throughput is bounded.**
10. **Gather regression** — regenerate a stored pool and compare bit for bit (done
    by hand for the array change; not yet automated).

---

## 8. Code cleanup

**Delete**

- `renders_pear_tumble.log`, `renders_pear_tumble_mixed.log`,
  `renders_pear_mixed_pose09.log` (stray, committed by accident); ignore `*.log`.
- `ground_truth/geometry_leakage_validation.py` — reports false leaks on
  watertight meshes; `pear_validation.py` does the job correctly.
- `export_rdm_json.py` — feeds a viewer that isn't in the repo.
- `ground_truth/diamond.py` — an unused early prototype.
- `compare_boundary_sampling.py` — once `measure_prior_variance.py` replaces it.
- Probably `ground_truth/geometry_validation.py` — a misnamed interactive viewer,
  superseded.

**Tidy**

- Move the legacy RDM pipeline into `legacy/` and check `eval.py` still runs.
- Merge duplicated helpers: `stone_mask` (two files), `load_pool` (three).
- Rename `ground_truth/pear_validation.py` to `validate_meshes.py`.
- Decide whether `render_paired.py` exposes `--rdm_prior`, or state that it is
  deliberately RDM-free.
- Remove the unused constants in `oracle_boundary.py` and the unused field in
  `gather_sharded.py`.

**Do not delete** `checkpoints/` or `renders/` data: it is not in git, so deletion
is permanent, and "not cited" does not mean unused. If disk space is ever needed,
archive the legacy `run_*`, `ks_*` and `rA_*` checkpoints rather than deleting.

**Refactor rule:** tag first, change in small steps, and after any step that
touches computation reproduce a stored result exactly.

---

## 9. Reproducing the main results

```bash
# data (camera-matched, round cut)
python -B gather_boundary.py --output checkpoints/camera_pool/train.npz --entries 65536 \
  --paths_per_entry 16 --max_depth 128 --seed 401 --workers 8 --entry_sampling camera --azimuth_range 0 360

# operator
python -B train_boundary.py --data checkpoints/camera_pool/train.npz --output checkpoints/camera_model --epochs 80

# paired render and evaluation
python -B render_paired.py --model checkpoints/camera_model/model.pt --output_dir renders/<name> \
  --width 512 --height 512 --spp 32 --seeds 101 102 103 104 105 106 107 108 --workers 4 --split_components

# model-level analyses
python -B oracle_boundary.py --train checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz --output <json>
python -B ablate_conditioning.py --data checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz --output <json> --epochs 60
python -B improve_decoder.py --data checkpoints/camera_pool/train.npz --test checkpoints/camera_pool/test.npz --output <json> --epochs 30 --resume

# large pools
python -B gather_sharded.py --output_dir checkpoints/pear_pool_50m --records 50000000 --workers 8 --max_seconds 480
python -B train_streaming.py --shards checkpoints/pear_pool_50m --output checkpoints/pear_deep_50m --epochs 2 --max_seconds 540

# tumble (resumable)
python -B animate_boundary.py --model checkpoints/pear_deep_mixed/model.pt --output_dir renders/<name> \
  --motion tumble --frames 60 --width 384 --height 384 --spp 32 --workers 4 --fps 30 [--resume]

# figures
python -B plot_results.py
```

Parallel gathers are reproducible only with the same seed **and** worker count;
`gather_sharded.py` is reproducible regardless of worker count.

---

## 10. The write-up

**One-sentence claim.** *Aggregate neural appearance models transfer where
aggregation destroys the structure being modelled, and fail where fine structure
is the appearance — and this can be characterised precisely rather than
asserted.*

**Arc.**

1. The question, with diamond as the adversarial case.
2. The inherited RDM and its measured defect: direction 6.8 %, position 19.2 %.
3. The generalisation: continuous conditioning; the only discrete variable is the
   exit facet, set by the geometry — *stop binning the sphere; index the geometry*.
4. The elimination chain — data, capacity, distribution, sharpness, density family,
   geometry — each excluded by measurement.
5. The two-part diagnosis, with the oracle-facet render as the pivotal figure.
6. Operator accuracy does not transfer to the image — four instances, one
   qualified exception.
7. Where the limit is: the network beats interpolation; data and capacity are
   excluded; the representation is what remains.
8. The RDM retained and measured in both roles.
9. Training coverage: the tumble blind spot, and that fixing coverage fixes energy,
   not placement.
10. Conclusion: the boundary of the technique, not the failure of an
    implementation.

**Throughout.** Lead with measurement — the noise floors, the component split and
its 0.96 control are what make negative results credible. Include the
corrections record as an appendix. Say the speed result (1.46× slower) early and
plainly.

**Title.** Not "neural diamond rendering with radiance distribution maps": no
evaluated render uses the RDM. Something like *"A position-conditioned boundary
transport operator for faceted transparent solids"*.

**Figures.** Existing: paired comparisons (round, pear, pose 9, 50M, deep clone),
conditioning ablation and facet-accuracy summary (`renders/figures/`), entry
sampling diagram, tumble contact sheets (before and after), mesh views. Still
needed: data ladder curve, contrast against sample count, component decomposition
bar, operator-accuracy-versus-image scatter across the four instances, and — if
run — the convergence plot.

**Expected band.** Strong write-up of current results: 85–88. With the
convergence check and a Fourier experiment: 88–92. Marker variance on negative
results is real, which is a further reason an attempted fix is worth having.

---

## 11. Environment notes

- Windows Python 3.13, Mitsuba 3.9.1. Import `config` before Mitsuba in every
  entry point.
- Set `PYTHONIOENCODING=utf-8` when output is piped or redirected.
- **Memory is the binding constraint.** The machine has 15.3 GB and something
  outside this project holds ~11 GB. Background runs have repeatedly been killed
  under memory pressure. Use 4 workers rather than 8, prefer the resumable tools
  (`gather_sharded.py`, `train_streaming.py`, `improve_decoder.py --resume`,
  `animate_boundary.py --resume`), and run long jobs in bounded steps
  (`--max_seconds`).
- Each render process holds Mitsuba and PyTorch at ~250 MB.
- `eval.py` is the legacy path and needs `--diamond_name` explicitly; it cannot
  build pear or step cuts (its mesh builder is round-only).
- Render timings: a 512×512, 32-spp pear frame takes ~9 min with four running
  together; a 60-frame 384×384 tumble takes 75–105 min at 4 workers.
