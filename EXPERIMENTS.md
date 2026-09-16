# Experimental protocol: results to compile

Every result needed to defend the thesis, with what it produces and which claim
it supports. Status is `HAVE` (measured, numbers in hand), `PARTIAL` (exists but
incomplete), or `NEED`.

The thesis claim being defended throughout:

> A smooth conditional density cannot represent singular specular transport.
> Improving its likelihood does not improve the image, because likelihood and
> image radiance are connected through the joint exit position and direction,
> and a smooth mixture blurs exactly that joint structure.

Two rival explanations must be excluded for that claim to stand: insufficient
**coverage** of the conditioning space, and insufficient **mode capacity** in the
mixture. Items A3, A7, C2 and C5 exist to exclude them. Without those the claim
is assertable but not defended.

---

## A. Model-space results

Distribution quality on held-out transport records, independent of rendering.

### A1. Data ladder — HAVE
Entry-state count against held-out NLL, with unconditional baselines fitted on
exactly the records each model trained on.

*Produces:* table and curve, 256 to 65,536 entry states.
*Supports:* not data limited above roughly 16k entry states.

### A2. Capacity probe — PARTIAL
Width 64 with 4 components reaches 6.4767; width 256 with 8 reaches 6.1701 then
overfits by epoch 60. Two points is a weak curve.

*Still needed:* a third and fourth capacity point so it reads as a curve rather
than an anecdote.
*Supports:* not capacity limited; roughly 0.40 nats then overfitting.

### A3. Mixture component sweep — NEED
K = 1, 4, 8, 16, 32, 64 at fixed data and fixed width. Report both held-out NLL
**and** render-space contrast, because the two may diverge and that divergence is
itself the finding.

*Produces:* contrast against K, NLL against K.
*Supports:* distinguishes "smooth densities cannot" from "four Gaussians cannot".
Without this the central claim narrows to the weaker form under questioning.

### A4. Conditioning ablation — NEED
Drop or permute each input group (entry position, incoming direction, surface
normal, wavelength) and measure the change in held-out NLL.

*Produces:* a table of nats lost per input group.
*Supports:* which conditioning variables carry signal; also detects encoding bugs
that would otherwise be mistaken for model failure.

### A5. Escape calibration — HAVE
Predicted 0.9975 against measured 0.9970 on held-out data.

*Supports:* the model is well calibrated on the quantity it can represent, which
strengthens rather than weakens the argument — the failure is not sloppiness.

### A6. Training curves and best-epoch selection — HAVE
Full history in each `metrics.json`.

### A7. Mode multiplicity in the gathered data — NEED
Cluster the 16 repeats per entry state in `train_pool.npz` and count distinct
exit modes per (entry, facet) pair. Compare the distribution against K.

*Produces:* histogram of modes per conditioning state, against mixture capacity.
*Supports:* converts the central claim from an observation into a mechanism.
Pairs directly with A3.

---

## B. Render-space results

All on linear EXRs over primary-hit stone pixels, with measured noise floors.

### B1. Paired analytic/neural comparison — HAVE
728x728, 128 effective spp, eight seeds per mode. Correlation 0.098 against a
0.955 noise floor; highlight overlap 18.5 against 78.0 percent; energy error
plus 15.8 percent against a floor of minus 0.017 percent.

*Produces:* the main results table and the side-by-side figure.

### B2. Component decomposition — HAVE
Learned operator carries 87.6 percent of stone radiance and correlates at 0.045
in isolation, while the identical-physics component in the same image correlates
at 0.922.

*Produces:* the strongest single table in the paper.
*Supports:* the failure is attributable to the learned component, and the control
proves the measurement apparatus is sound.

### B3. Scintillation sweep — PARTIAL
Fifteen azimuths at 96x96, paired, with seed-split noise floors.

*Still needed:* the docstring claim that a 45 degree sweep covers every distinct
pose is wrong. The lighting rig is deliberately asymmetric, so camera azimuth 0
and 45 are different views. Either extend to 360 degrees or restate the claim as
sampling a 45 degree arc.

### B4. Contrast against sample count — HAVE
Neural contrast falls 1.664, 0.972, 0.799, 0.683 at 16, 64, 128, 256 spp while
analytic holds near 3.8.

*Produces:* a two-line figure.
*Supports:* apparent facet detail in low-sample neural renders is speckle, not
structure. This is a reusable diagnostic and worth presenting as one.

### B5. Multi-view at high resolution — NEED
Several azimuths at full resolution rather than one.

*Supports:* the conclusion is not an artefact of a single camera position.

### B6. Held-out lighting — NEED
A second studio rig or environment. Requires extending `utils/studio_env.py`.

*Supports:* the conclusion is not an artefact of one lighting configuration.
Currently the most conspicuous gap in validation gate 4.

### B7. Analytic reference convergence — NEED
One very high sample-count analytic render; verify the 64 and 128 spp versions
approach it.

*Supports:* the reference is converged and the 0.955 noise floor reflects noise
rather than bias. Without this the control in B2 is assumed rather than shown.

---

## C. Ablations: the representation ladder

The core ablation. Each rung removes one source of error, so the gaps between
rungs attribute the failure.

| rung | what it is | status |
| --- | --- | --- |
| C1 analytic | exact transport, upper bound | HAVE |
| C2 nearest-neighbour oracle | real recorded exits, no smoothing, no network | NEED |
| C3 table operator | histogram over the same conditioning, no network | NEED |
| C4 neural operator | the trained model | HAVE |
| C5 coverage analysis | distance from render queries to the training manifold | NEED |
| C6 behavioural cloning | deterministic regression, negative control | NEED |
| C7 unconditional baselines | marginal facet, per-facet Gaussian | HAVE |

### C2. Nearest-neighbour oracle — NEED, highest priority
For each render query, retrieve a real escape event from the nearest training
entry state and use its actual exit position and direction. This is the best
achievable with this conditioning and this data, with zero smoothing.

Outcomes, all publishable and mutually exclusive:
- oracle flashes, network does not: representation failure, claim confirmed
- neither flashes: the conditioning does not determine the exit sharply enough
- oracle needs a denser pool: coverage failure, the ladder must extend

### C3. Table operator — NEED
Named in validation gate 3. Separates information lost in discretizing the
conditioning from network fitting error.

### C5. Coverage analysis — NEED, highest priority
Instrument `render_boundary.py` to record, per neural query, the distance to the
nearest training entry state in the model's own normalized encoding. Histogram
against nearest-neighbour distances within the training set.

*Supports:* excludes extrapolation as the explanation. The conditioning is at
least 5D and 65,536 entry states is roughly nine samples per dimension, so this
question will be asked.

### C6. Behavioural cloning — NEED
Deterministic regression from entry state to exit, trained on the same data.
Expected to fail by predicting a conditional mean — an exit point floating inside
the stone, corresponding to no physical path.

*Supports:* retroactively justifies the distributional design. Converts an
architectural choice into a demonstrated necessity.
*Note:* confirm with the supervisor whether their requested cloning test means
this, or cloning the sampler's branch decisions. The latter is the NIS pivot.

---

## D. Cost and performance

### D1. Per-sample cost — HAVE
620 s neural against 424 s analytic per 8-spp 728x728 render, 1.46x.

### D2. Noise per pixel — HAVE
Median per-pixel relative standard error 0.348 neural against 0.222 analytic at
matched sample counts, 1.57x.

### D3. Equal-quality cost — HAVE, needs a caveat
Roughly 3.5x analytic compute for equal noise.

*Caveat to state:* measured in a BSDF-only estimator with no next-event
estimation. Adding NEE would change both modes and not necessarily equally. Note
also that NEE is inapplicable to the analytic path's specular exit but *is*
applicable to the neural operator's continuous density — an asymmetry worth
discussing, since it is one genuine advantage of a smooth representation.

### D4. Equal-time comparison — NEED
Render both modes for a fixed wall-clock budget and compare error against the
reference. Distinct from the equal-sample comparison already held.

*Supports:* the honest practitioner's question, which is not "at equal samples"
but "at equal time".

---

## E. Correctness tests protecting the above

### E1. Sample and PDF consistency — NEED
Validation gate 1 requires it and it is absent. `log_exit_pdf` has analytic
Jacobian tests, but nothing verifies that `BoundaryModel.sample` draws from that
density. Load bearing: the RDM prior's transport weight is `p_neural/q`.

*Method:* many samples at fixed conditioning, binned over facet, barycentric cell
and direction cell, compared against the integrated density by chi-square.

### E2. Energy conservation — HAVE
`tests/test_boundary_render.py` reproduces a unit constant environment for both
the analytic path and a controlled unit-throughput operator.

### E3. Reproducibility — HAVE
Equal seeds reproduce equal datasets; parallel gather is deterministic given seed
and worker count.

### E4. Regression tests for recent changes — NEED
Nothing tests that `--workers` output is deterministic with globally unique
`entry_id` values, nor that vectorized `encode_data` matches the reference
implementation. Both were verified by hand and neither is locked down.

---

## Figure list

1. Analytic against neural, 728x728, identical camera, lights and sample count (B1)
2. Data ladder: gain over unconditional baseline against entry-state count (A1)
3. Flash sweep: contrast and flash fraction against azimuth, both modes, noise band (B3)
4. Component decomposition, operator against untouched (B2)
5. Mode multiplicity against mixture component count (A7)
6. Contrast against K (A3)
7. Contrast against sample count, both modes (B4)
8. Representation ladder: analytic, oracle, table, neural (C1 to C4)
9. Importance sampling variance against baseline branch sampling (the pivot)

## Priority if time is short

C5 and C2 first — they close the one hole in the main claim. Then A3 and A7,
which turn the observation into a defended mechanism. Then E1, which meets a gate
the thesis itself declares. Everything else is breadth.
