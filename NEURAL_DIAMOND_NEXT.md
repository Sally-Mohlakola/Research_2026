# Neural diamond rendering: findings and implementation contract

Status: sampler fixes, spatial data collection, and a conditional neural escape prototype are implemented. No validated render-quality improvement yet.

## Verified implementation findings

- `bsdf/rdm_sampler.py` sampled theta uniformly while reporting mass divided by solid angle. Changed to uniform cosine-theta within each cell. Empty rows now normalize, and a 1% uniform-sphere mixture provides support outside measured cells.
- `bsdf/neural_bsdf.py` fallback sampling and PDF now both cover the full sphere, matching the RDM outgoing domain.
- `eval.py:create_scene` selects Mitsuba path; it does not attach DepthAwarePath. Gather k is not a render-time cutoff in this entry point.
- `utils/rdm.py:trace_path` expresses entry/exit directions in the first hit frame. The histogram drops entry and exit positions. An internal-hit query of this external-entry operator is not a matched conditional continuation model.
- Sparkle uses a position hash and Bernoulli-like sparse gain, not a Gaussian draw. First and second moments do not uniquely specify a distribution. Matching brightness statistics does not establish spatial fidelity, and independent random gains average away with increasing samples. A fixed spatial hash instead changes the represented appearance.
- Equal seeds do not guarantee identical paths or exact noise cancellation when algorithms consume random numbers differently. A disjoint depth partition alone does not remove approximation error in a learned operator.
- Neural values and estimator weights are clamped. The current estimator cannot be described as unbiased relative to physical ground truth.

## Recommended target

Build a position- and wavelength-conditioned boundary transport model. Keep analytic Fresnel reflection, refraction, total internal reflection, and geometry as the reference. Treat the existing directional RDM as a coarse proposal/prior; it is not sufficient by itself to locate facet detail.

For an entry state, retain object-space position, facet ID or barycentrics, incident direction, wavelength, and transport mode. Gather complete escape events: exit position, exit direction, throughput, path depth, and whether the depth cap was reached. Persist geometry/IOR/dispersion conventions and sampler PDFs. Partition training data by entry state, not random neighboring histogram cells.

A learned nonlocal operator must predict/sample both exit position and direction, then resume scene tracing from that exit. A direction-only ray spawned at an internal hit cannot represent the physical exit location or surrounding visibility. Perfect specular transport has singular branches: use a path/branch representation or explicitly defined finite-footprint filtering rather than assuming a smooth continuous density is exact.

If switching after k interactions, gather continuation targets conditioned on the actual state at that same switch, including position, direction, wavelength, and inside/outside status. Do not reuse the external-entry depth>k histogram as that target. The alternative is an entry-boundary decomposition into short paths and long paths, both evaluated from the original entry state.

A conservative first neural method is neural importance sampling over valid analytic path branches, retaining exact Fresnel/PDF correction and a physical proposal with nonzero support. This preserves the physical target while testing whether the network reduces equal-time error. The current RDM may inform proposals, but arbitrary RDM directions cannot replace delta reflection/refraction at a smooth interface.

## Validation gates

1. Sampler: normalization, empirical solid-angle frequencies, sample/PDF agreement, empty rows, and full support.
2. Ground truth: same geometry, camera, lights, spectrum and exposure; verify depth and sample convergence with independent seeds.
3. Representation ablation: analytic reference versus table operator versus neural operator. This separates information lost in the RDM from network fitting error.
4. Compare linear EXRs with diamond masks: relative energy error, RMSE, spatial correlation, highlight locations, and color error. Include multiple views and held-out lighting. Report independent-reference noise and equal-time results.
5. Disable synthetic sparkle for physical validation. Document all clamps and truncation. Do not choose k from one image's brightness histogram.

## Literature

- Sun et al., Rendering Diamonds: https://www2.cs.sfu.ca/~mark/ftp/Skigraph00/renderDiamonds.pdf — Fresnel reflection, absorption and dispersion in explicit gemstone transport.
- PBRT, Dielectric BSDF: https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF — delta transport, Fresnel sampling and radiance transport factors.
- Mueller et al., Neural Importance Sampling: https://jannovak.info/publications/NIS/index.html — learned sampling distributions with tractable PDFs and Monte Carlo weighting.
- Soh and Montazeri, Neural Appearance Model for Cloth Rendering: https://pure.manchester.ac.uk/ws/portalfiles/portal/328698468/neuralCloth_CGF24.pdf — the source architecture referenced in this project targets aggregated cloth appearance; transferring it to a resolved gemstone requires validating the spatial aggregation assumption.

## Current validation limitation

On this Windows installation, selecting llvm_ad_spectral succeeds, but executing the sampler test produces LLVM Failed to materialize symbols. No render-quality or speed claim is established by these edits. Resolve the runtime or use a validated backend before launching new expensive gathers/training.


## Implemented milestone: spatial boundary records

`gather_boundary.py` now traces repeated monochromatic paths from fixed entry states using scalar Mitsuba geometry. First-surface reflection remains analytic; entry transmission is forced. Internal branches are sampled with their Fresnel probabilities. Lossless escaped conditional throughput is one; multiply saved entry_transmittance once when composing with the entry reflection. This is not an RGB radiance dataset and branch_log_probability is not a continuous exit PDF.

Records include positions, directions, facet IDs, wavelength, entry transmittance, throughput, depth, status and discrete path log probability. The file embeds geometry, hash and conventions. Truncation and invalid geometry events remain explicit records, with undefined exits, rather than training the network to fabricate an exit. Split by entry_id with neural/boundary_data.py. Existing directional RDM checkpoints cannot substitute for this dataset.

Small reproducible gather:

```powershell
python -B gather_boundary.py --output checkpoints/boundary_smoke/transport.npz --entries 16 --paths_per_entry 4 --max_depth 64 --seed 17
python -B -m unittest discover -s tests -p test_boundary_transport.py
```

This CPU scalar gatherer is a correctness prototype, not a production-throughput collector. Wavelength-dependent IOR uses the same Sellmeier function as ground truth, without RGB conversion or spectral packet factors. Entry sampling is a specified hit-conditioned distribution; no absolute entry-area PDF is claimed. The next milestone is a conditional multimodal escape model and held-out transport evaluation, followed by a boundary-aware render adapter. No model is trained or installed in eval.py by this milestone.


## Implemented milestone: conditional neural escape prototype

`neural/boundary_model.py` provides BoundaryModel and load_model. The model conditions on normalized object-space entry position, incoming travel direction, normal, and wavelength. It predicts:

1. Probability of escape within the dataset depth cap.
2. A categorical exit triangle.
3. A triangle-conditioned Gaussian mixture jointly representing two barycentric logits and two outgoing tangent slopes.

This avoids averaging distinct exits into one point/direction. Decoding barycentrics keeps positions on the selected triangle; tangent-slope decoding gives outward unit directions. Sampling returns a finite-depth escape mask and unit conditional throughput for escaped paths. Callers MUST respect that mask and multiply entry_transmittance once, retaining analytic first-surface reflection. Exit values for nonescaped samples are placeholders, not physical events. Absorption/colored transport weights are not learned in this lossless prototype.

Mixture densities and likelihood metrics are in transformed coordinates, NOT surface-area/solid-angle measure. A future PDF-query adapter must include transformation Jacobians and the discrete facet probability. Do not plug this density into MIS as a physical BSDF PDF. Boundary-edge and grazing targets are clipped at 1e-5 in the coordinate transform and the count is reported. Smooth mixtures approximate singular specular paths; surface-valid samples need not correspond to physically realizable paths.

`train_boundary.py` uses entry-grouped validation, excludes invalid records, trains the escape head on truncated records, selects the best validation checkpoint, and saves geometry, hashes, data conventions, split IDs, and metrics. `evaluate_boundary.py` reloads that checkpoint and requires a separate gather seed and matching geometry/transport settings. The RDM prior is not yet incorporated into this spatial model; existing RDM sampler improvements remain separate.

Pilot commands (output paths must not already exist):

```powershell
python -B gather_boundary.py --output checkpoints/boundary_model_pilot_v2.npz --entries 256 --paths_per_entry 8 --max_depth 128 --seed 31
python -B train_boundary.py --data checkpoints/boundary_model_pilot_v2.npz --output checkpoints/boundary_model_pilot --epochs 100
python -B gather_boundary.py --output checkpoints/boundary_model_test.npz --entries 128 --paths_per_entry 8 --max_depth 128 --seed 47
python -B evaluate_boundary.py --model checkpoints/boundary_model_pilot/model.pt --data checkpoints/boundary_model_test.npz --output checkpoints/boundary_model_pilot/test_metrics.json
python -B -m unittest discover -s tests -p 'test_boundary*.py'
```

Observed pilot results: 2,048 gathered training/validation records, including 8 invalid records excluded from learning. Best validation epoch 15; joint transformed NLL improved from 12.603 to 10.315. Later epochs overfit. Independent test: 1,024 records, zero invalid, joint NLL 11.415; predicted escape 95.54% versus measured 99.90%; top-1 exit-triangle accuracy 9.38%. Triangle accuracy compares the most likely triangle with each stochastic reference outcome, so it is not a complete distribution-quality measure. These numbers establish pipeline operation and expose remaining modeling error, not realistic shading.

Nine tests cover transport physics/truncation, reproducibility, grouped splitting, constrained neural samples, missing exit targets, and learning two outcomes from identical inputs. The larger gather revealed accumulated ray-direction roundoff; directions are now normalized at each interface. The original pilot NPZ is retained for provenance but is superseded by pilot_v2.

Next: improve model/data coverage and evaluate joint spatial/angular distributions against an unconditional baseline; then implement the boundary render adapter and a properly measured RDM-prior combination. eval.py remains unchanged by this milestone.


## Implemented milestone: boundary render adapter

`render_boundary.py` renders a fixed checkpoint geometry under the shared studio rig, using scalar spectral Mitsuba intersections and batched PyTorch boundary inference. It supports neural and analytic modes with the same camera, geometry, lights, sensor and depth settings. It writes frames/frame_0000.exr, frames/frame_0000.png and render.json with model hash, timings and failure counters. It refuses to overwrite an existing output directory.

At an exterior stone hit it samples the Fresnel entry branch. Reflection remains analytic. On transmission it samples the neural operator conditional on transmission; it does NOT multiply by entry T again because branch sampling already accounts for T. A learned escaped event resumes tracing from its predicted exit surface. Scene occlusion and diffuse bounces remain explicit; subsequent exterior stone encounters can invoke the operator again. Exits that geometrically reenter the stone are rejected and counted. Finite-depth failures contribute zero, matching the dataset's finite-depth convention, not an infinite-depth unbiased claim.

The spectral sensor wavelength weight is retained and its first wavelength is traced with one packet-collapse factor. RGB display conversion happens only after transport. No RGB network output is substituted for monochromatic transport. The learned transformed-coordinate PDF is not used as a BSDF PDF or divided into throughput: samples already represent the learned conditional probability law for unit-throughput lossless paths. MIS/next-event estimation and the RDM sampling prior are not yet integrated.

Run the new model (use a fresh output directory):

```powershell
python -B render_boundary.py --model checkpoints/boundary_model_pilot/model.pt --output_dir renders/boundary_neural_02 --width 128 --height 128 --spp 32
python -B render_boundary.py --model checkpoints/boundary_model_pilot/model.pt --output_dir renders/boundary_analytic_02 --mode analytic --width 128 --height 128 --spp 32
```

`--azimuth` moves the camera around the fixed stone. `--scene_depth` controls surrounding-scene steps; the internal analytic cap comes from the checkpoint metadata (128 for the pilot). Output is a single frame; eval.py and its old checkpoints remain a separate rendering path. The box pixel filter differs from eval.py's Gaussian filter, so comparisons should use this adapter's paired modes.

Validation: tests/test_boundary_render.py runs scalar spectral checks in a separate process to avoid cross-test Mitsuba variant changes. Both the analytic path and a controlled unit-throughput neural operator reproduce a unit constant environment within Monte Carlo tolerance. The controlled operator makes thousands of real neural-adapter queries and checks no rejected exits. This is an energy-bookkeeping test, not model-quality validation.

Generated paired pilot previews: renders/boundary_neural_01 and renders/boundary_analytic_01, each 128x128 at 32 spp. Neural: 273,280 boundary queries, zero predicted reentries, 442 scene-cap terminations, about 90.7 seconds. Analytic: 263,451 successful exits, 545 internal truncations, 59 invalid internal paths, one unexpected inside hit, 404 scene-cap terminations, about 57.3 seconds. Runs were concurrent; these timings are not controlled speed benchmarks. Both images retain visible spectral sampling noise. Inspection shows the neural render is broad/smooth and misses the reference facet pattern. This confirms a working render path, not realistic model fidelity.

Next work: improve/validate the learned spatial-angular distribution, address flagged geometry events, incorporate an RDM prior with a compatible probability measure, and evaluate converged paired renders over held-out views and lights. There is no demonstrated quality or speed advantage yet.


## Implemented milestone: RDM proposal with compatible physical density

`build_boundary_prior.py` builds an object-space directional RDM from escaped TRAINING records only, bound by hashes to the exact checkpoint/source dataset. It conditions on entering and escaping the stone, unlike the legacy first-hit-local depth>k RDM. Do not pass old rdm.npz files to the boundary renderer. The pilot table uses 4x8 incoming and outgoing bins and 1,624 training escape records, with 5% uniform-sphere support.

`neural/boundary_prior.py` samples an outgoing direction uniformly in solid angle within the selected RDM cell. It then samples a triangle proportionally to its outward projected area A_f max(n_f dot d,0), followed by a uniform point on that triangle. Its joint area/solid-angle density is p_RDM(d|wi) max(n_f dot d,0) / sum_j A_j max(n_j dot d,0).

BoundaryModel.log_exit_pdf now converts the neural conditional density to that same measure. The position Jacobian is 2 A_f b0 b1 b2 for the two barycentric logits; the direction Jacobian is cos(theta)^3 for the two tangent slopes. The categorical triangle probability is included. These are conditional escape PDFs; escape probability is still sampled separately.

The proposal is q=(1-alpha)p_neural+alpha*p_RDM_joint. Conditional transport weight is p_neural/q, bounded above by 1/(1-alpha). This preserves the LEARNED target, including its spatial/spectral dependence, rather than correcting it toward ground truth. The renderer still rejects/counts stone-reentering exits. alpha=0 calls the original sampler unchanged. alpha must be below 1 so the neural proposal retains support. The prior does not alter model weights or recover missing facet structure.

Commands (prior output already exists for the pilot; build only for new checkpoints):

```powershell
python -B build_boundary_prior.py --model checkpoints/boundary_model_pilot/model.pt --data checkpoints/boundary_model_pilot_v2.npz --output checkpoints/boundary_model_pilot/rdm_prior.npz
python -B render_boundary.py --model checkpoints/boundary_model_pilot/model.pt --rdm_prior checkpoints/boundary_model_pilot/rdm_prior.npz --prior_fraction 0.25 --output_dir renders/boundary_prior_02 --width 128 --height 128 --spp 32
```

Validation: all 14 boundary tests pass. Added checks for analytic transformation Jacobians, solid-angle sampling, outward support, mixture mass preservation, weight bound, and alpha=0 equivalence. Paired 64x64/32-spp renders at seeds 71 and 72 were evaluated over 2,092 primary diamond pixels. compare_boundary_sampling.py saves the diagnostic to renders/boundary_sampling_comparison.json.

Observed result: prior/baseline paired luminance variance ratio 1.202 (about 20% MORE noise with the 25% RDM mixture). Mean diamond luminance was 0.004037 versus 0.003945; two seeds do not establish a systematic energy difference. Concurrent run times were about 32-34 s with the prior versus 22 s without; these are not controlled performance benchmarks. The coarse prior is therefore opt-in and not recommended as a demonstrated improvement. This experiment measures variance around the learned target, not error against physical reference.

Next priority is better spatial/angular model fidelity, with distribution-level comparisons against a simple baseline and held-out views/lighting. Any further prior tuning must show a measured benefit; its mere presence is not a rendering-quality contribution.


## Implemented milestone: data scaling, convergence, and a measured negative render result

This milestone answers the standing question of whether the boundary model was data limited, capacity limited, or representation limited. It is representation limited. The learned operator now fits its training distribution well and still produces an image that is spatially uncorrelated with analytic transport, at higher variance and higher cost.

`gather_boundary.py` accepts `--workers`. Single-worker output is bit-identical to the previous implementation, so existing dataset provenance is unaffected. Parallel chunks receive entry ranges through `numpy.random.SeedSequence.spawn`, and each chunk carries an `entry_offset` so emitted `entry_id` values stay globally distinct. This matters because `split_entries` groups on `entry_id`; colliding identifiers from separate workers would silently merge unrelated entry states into one split group. The stream partition depends on the worker count, so reproduction requires the same seed AND the same `--workers`, and metadata records both. Throughput on sixteen logical cores is 1,048,576 records in 28 seconds.

`encode_data` previously called `numpy.linalg.lstsq` once per escaped record. Only `len(faces)` distinct triangles exist, so one pseudo-inverse per triangle replaces the per-record solve exactly. The vectorized form is about 488 times faster on a 32,768-record dataset, agrees to float32 round-off, and returns identical facet, valid, escaped and clipped values. Encoding one million records now costs about 0.25 seconds rather than 131 seconds, on every training run and every evaluation.

`make_boundary_subset.py` writes nested entry-grouped subsets of a gathered pool. Subsets select whole entry states, never individual paths, so every rung of a data ladder stays compatible with the entry-grouped split.

`flash_sweep.py` renders paired neural and analytic frames over a camera orbit and measures scintillation. The camera orbits while studio lights stay fixed, so pixel (x, y) views different geometry in each frame and no per-pixel temporal variance is defined. Every statistic is therefore correspondence-free: within-frame highlight distribution shape, and the across-frame variation of those statistics. Each configuration renders under at least two independent seeds, and the seed-to-seed spread is reported as the Monte Carlo noise floor, because sparse-highlight statistics are precisely the ones path-tracing noise inflates.

`evaluate_render.py` performs the masked EXR comparison required by validation gate 4. Statistics are computed on linear values over primary-hit stone pixels only. Independent-reference noise is measured rather than assumed: the per-seed renders of each mode are split in half to give two independent estimates of the SAME image, and the neural comparison is repeated on matching half-splits so signal and noise are read at equal sample counts. The tool refuses unpaired comparisons that disagree on azimuth, resolution or spp, and refuses repeated seeds.

Data ladder. One pool of 65,536 entry states by 16 paths (1,048,576 records, seed 101) and an independent test pool of 16,384 entry states by 16 paths (262,144 records, seed 202). Nested subsets were trained with identical architecture and hyperparameters at approximately 50,000 gradient steps each, and evaluated on the single held-out test pool. Baselines were fit on exactly the records each model trained on: the marginal exit-facet distribution, and a per-facet diagonal Gaussian that knows the exit triangle but not the entry state.

| entry states | train records | facet NLL | baseline | mixture NLL | baseline | top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 3,259 | 3.960 | 4.031 | 7.251 | degenerate | 0.048 |
| 1,024 | 13,059 | 3.655 | 3.859 | 7.131 | 8.382 | 0.108 |
| 4,096 | 52,248 | 3.327 | 3.839 | 6.712 | 7.861 | 0.207 |
| 16,384 | 209,121 | 2.874 | 3.834 | 5.191 | 7.753 | 0.332 |
| 65,536 | 836,425 | 2.773 | 3.833 | 4.466 | 7.710 | 0.347 |

The 256-entry rung reproduces the earlier pilot behaviour and early-stopped at epoch 7 of 3,800, so it was never step starved; it exhausted distinct entry states. The 256-rung per-facet Gaussian baseline is degenerate, not informative: most triangles fall below the per-facet fitting threshold and collapse to a global Gaussian. Unconditional top-1 accuracy is 0.030.

Converged training on the full pool. Width 64 with 4 components reached best validation joint NLL 6.4767 at epoch 78 of 80, and independent test joint NLL 6.6047 with facet top-1 0.3829 and predicted escape 0.9975 against measured 0.9970. Width 256 with 8 components reached 6.1701 at epoch 40 and regressed to 6.2437 by epoch 60, so additional capacity buys about 0.40 nats and then overfits. The full held-out arc from the original pilot is joint NLL 11.415 to 6.605, and facet top-1 0.094 to 0.383 against a 0.030 unconditional baseline.

Render evaluation. Paired renders at 728x728, azimuth 0, eight seeds at 8 spp each for 64 effective spp per mode, converged width-64 checkpoint, 271,232 primary stone pixels.

| statistic | analytic | neural |
| --- | --- | --- |
| masked mean luminance | 0.0052453 | 0.0060982 |
| contrast, standard deviation over mean | 3.869 | 0.972 |
| peak over mean | 163.03 | 23.53 |
| flash fraction above 32x median | 0.00715 | 0.00264 |

| agreement | neural vs analytic | analytic vs itself |
| --- | --- | --- |
| spatial correlation | 0.0736 | 0.9121 |
| highlight overlap, brightest 1 percent | 0.1136 | 0.6932 |
| relative RMSE | 4.136 | 1.659 |
| relative energy error | +0.1619 | -0.0000079 |
| chromaticity error | 0.446 | 0.337 |

Observed result. Two independent analytic renders agree at correlation 0.912 despite both being noisy; the neural render agrees with the reference at 0.074. The model is therefore not a blurred version of the correct image, because blur preserves correlation. It places light in different locations. Highlight overlap states the same failure directly. The energy excess of 16.19 percent stands against a noise floor of 0.0008 percent and is unambiguous. Chromaticity error sits at 1.33 times its own noise floor and no colour conclusion is claimed from this single view.

Variance and cost. Measured across the eight independent seeds on stone pixels at equal 64 effective spp, median per-pixel relative standard error is 0.222 analytic and 0.348 neural, so the learned operator is about 1.57 times noisier per pixel. Wall time per 8-spp render was 424 seconds analytic and 620 seconds neural, about 1.46 times slower. Matching analytic noise would require roughly 2.5 times the samples at 1.46 times the cost each, approximately 3.5 times the analytic compute for equal quality, before accounting for the fact that the converged image is the wrong one. The learned operator is currently worse on both axes it was intended to improve.

Interpretation. The broadening that removes flashes and the broadening that raises variance are one defect observed in the mean and in the variance. Each camera sample entering the stone draws an exit triangle from a 64-way categorical, a component from the mixture, then Gaussian samples for exit position and outgoing direction, so two samples in one pixel may exit at opposite ends of the stone. Analytic internal transport is also stochastic, but its Fresnel branches are discrete and few and each branch gives an exact direction, so outcomes cluster. A smooth conditional density cannot represent singular specular transport, and improving its likelihood does not change that. Transformed-coordinate likelihood and image radiance are connected through the joint exit position and direction; the mixture scores well on the marginals while blurring the joint structure that determines where light lands.

Reproduction:

```powershell
python -B gather_boundary.py --output checkpoints/boundary_scale_v1/train_pool.npz --entries 65536 --paths_per_entry 16 --max_depth 128 --seed 101 --workers 16
python -B gather_boundary.py --output checkpoints/boundary_scale_v1/test_pool.npz --entries 16384 --paths_per_entry 16 --max_depth 128 --seed 202 --workers 16
python -B make_boundary_subset.py --pool checkpoints/boundary_scale_v1/train_pool.npz --output checkpoints/boundary_scale_v1/sub_04096.npz --entries 4096
python -B train_boundary.py --data checkpoints/boundary_scale_v1/train_pool.npz --output checkpoints/boundary_scale_v1/final_w064c4 --epochs 80
python -B evaluate_boundary.py --model checkpoints/boundary_scale_v1/final_w064c4/model.pt --data checkpoints/boundary_scale_v1/test_pool.npz --output checkpoints/boundary_scale_v1/final_w064c4/test_metrics.json
python -B flash_sweep.py --model checkpoints/boundary_scale_v1/final_w064c4/model.pt --output_dir renders/flash_sweep_converged --azimuth_start 0 --azimuth_stop 45 --azimuth_step 3 --width 96 --height 96 --spp 48 --seeds 71 72
python -B evaluate_render.py --model checkpoints/boundary_scale_v1/final_w064c4/model.pt --analytic renders/flash_hires_analytic/s101 renders/flash_hires_analytic/s102 --neural renders/flash_hires_neural/s101 renders/flash_hires_neural/s102 --output renders/flash_hires/evaluation.json
```

High-resolution renders were produced as eight independent 8-spp runs per mode and averaged, which is the same estimator as one 64-spp run when seeds differ, but parallelizable. Do not average renders that differ in spp or repeat a seed.

Validation limitation. Single view, single studio rig, one azimuth for the high-resolution comparison and fifteen azimuths at 96x96 for the sweep. No held-out lighting. The analytic mode is the reference transport path, not a converged physical ground truth, so these numbers bound agreement with that reference rather than with physical truth. Peak and peak-over-mean are extreme-value statistics and remain noisy at these sample counts.

Next priority. The evidence now supports the conservative method already identified in this document: neural importance sampling over valid analytic path branches, retaining exact Fresnel, delta refraction and total internal reflection, with the network restricted to choosing among physically realizable branches and the PDF correction applied exactly. That formulation cannot produce either failure measured here, because the singular structure is preserved by construction and the estimator remains anchored to a physical proposal with nonzero support. Further scaling of data or capacity for the smooth density model is not supported by this evidence; the ladder shows data is no longer the binding constraint and capacity yields 0.40 nats before overfitting.
