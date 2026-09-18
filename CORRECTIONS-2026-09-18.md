# Corrections — 18 September 2026

Claims in `FINDINGS.md`, `NEURAL_DIAMOND_NEXT.md` and the project framing that
today's measurements change. Each entry states what was claimed, what is now
measured, and what should replace it.

Numbers here were produced today on `checkpoints/camera_pool/test.npz`,
`checkpoints/camera_model`, `checkpoints/clone_model` and
`checkpoints/step_transfer`. Nothing below rests on a run that was interrupted.

---

## 1. Spread is not error. The within-facet decode was never measured.

**Claimed** — `FINDINGS.md` §"The operator is translucent where the transport is
transparent": real transport has **0.00 degrees** within-branch spread against
the model's **25.98 degrees**.

**Still true, but it has been read as something it does not say.** That table
compares the *spread* of exit directions about their own mean. It does not
measure how far the model's exits are from the *correct* exits. Those are
different quantities, and the second was never measured until today.

**Measured** — feeding each head the true exit facet, so only the within-facet
decode is its own, over all escaped records of the held-out pool:

| head | exit angle error given the true facet |
| --- | --- |
| clone | mean **24.66°**, median 18.23°, p90 54.48° |
| mixture | mean **32.51°**, median 22.14°, p90 79.36° |
| nearest-neighbour lookup | mean **8.87°** |

A deterministic head has 0.00° spread and is still 24.66° wrong. Zero spread is
not accuracy.

**Replace with**: the failure has two parts, not one — branch selection *and*
the within-facet decode. The nearest-neighbour figure shows the second part is
squarely a learning problem, since a lookup with no training beats the trained
head by 2.8x on the same conditional.

---

## 2. "One defect accounts for the whole appearance" is no longer supported.

**Claimed** — `FINDINGS.md` §"One defect accounts for the whole appearance":
the 25.98° blur is the single cause of absent fire, brilliance and
scintillation.

**Measured** — three experiments each removed or bypassed the blur and none
recovered the image:

| experiment | blur | correlation with reference |
| --- | --- | --- |
| mixture, as trained | 25.98° | 0.174 |
| deterministic decode | 8.09° | 0.153 |
| behaviour cloning | 0.00° | worse on the image despite better operator metrics |
| oracle facet (true branch, model coordinates) | — | **0.041** |

The oracle-facet render is decisive. Handed the correct branch, the render
reaches **contrast 3.734 against the reference's 3.744** and **peak/mean 114.7
against 121.1** — the sparkle returns in full — while correlation *falls* to
0.041 and the operator component to 0.017.

**Replace with**: blur suppresses the high-dynamic-range structure, and removing
it restores that structure, but the light is then placed sharply in the wrong
locations. Correlation rewards diffuse overlap, so a blurred render scores
higher partly *because* it is blurred. Sharpness and correctness are separate
axes and the project measured only the first.

---

## 3. Branch selection is not the whole gap either.

**Claimed** — repeatedly during analysis, including in this session: branch
prediction is *the* binding constraint.

**Measured** — with the analytic trace supplying the exit facet for free,
correlation is 0.041 and the operator component 0.017. If branch selection were
the whole gap, correlation should approach the 0.976 noise floor.

**Replace with**: branch selection (facet accuracy 0.4084) and the within-facet
decode (24.66°) are both unsolved, and neither alone explains the image.

---

## 4. No result in this project uses the RDM.

**Claimed** — the project title and framing, "neural diamond rendering with
radiance distribution maps".

**Measured** — all five evaluated renders record `rdm_prior: None,
prior_fraction: 0.0`: `camera_paired`, `det_exit`, `clone_paired`,
`pear_paired`, `oracle_facet`. `render_paired.py`, the orchestrator every one of
them ran through, exposes no `--rdm_prior` option at all, so the RDM was not
reachable from the path that produced the results.

**Replace with**: the work is a position-conditioned boundary transport
operator. The RDM is its ancestor and its baseline, not a component. The pivot
is legitimate and documented — `NEURAL_DIAMOND_NEXT.md` line 10 records the
diagnosis ("the histogram drops entry and exit positions") and line 17 the
response ("build a position- and wavelength-conditioned boundary transport
model") — but the title promises a method that was not built, and should change.

---

## 5. The RDM variance figure was measured on the wrong checkpoint.

**Claimed** — `NEURAL_DIAMOND_NEXT.md` line 137 and
`renders/boundary_sampling_comparison.json`: prior/baseline variance ratio
**1.202**.

**Why it was weak**: two seeds, 64x64, 2,092 diamond pixels, and the checkpoint
was `boundary_model_pilot` — not a model any reported result uses.

**Measured today** — four matched seeds per arm, current checkpoint
(`step_transfer`), step geometry, 160x160 at 16 spp, 25,600 lit pixels:

| | prior on | prior off |
| --- | --- | --- |
| mean luminance | 0.002596 | 0.002594 (+0.10%) |
| per-pixel variance | 4.105e-05 | 3.122e-05 |
| seconds | 29.6 | 24.3 |

**Variance ratio 1.315 at equal samples, 1.603 at equal time.** The means
agreeing to +0.10% is the importance weighting behaving correctly: the same
expected image, a worse estimator.

**Replace with**: the RDM as an importance-sampling proposal cannot change
accuracy by construction, and measurably costs 32% more variance at equal
samples and 60% at equal time.

---

## 6. The RDM cannot supply the missing branch information either.

**Claimed** — earlier today, by me: the RDM could correct the facet posterior as
a product of experts. This was wrong and is retracted.

**Measured** — clone facet accuracy baseline **0.4103**:

| RDM conditioning | cells | RDM alone | best blend with the model |
| --- | --- | --- | --- |
| entry direction only (the current RDM) | 32 | 0.0682 | 0.4094 |
| entry facet only | 64 | 0.1918 | 0.4103 |
| entry facet x direction (4x8) | 2,048 | 0.2272 | 0.4101 |
| entry facet x direction (8x16) | 8,192 | 0.2594 | **0.4135** |

The ceiling is **+0.0032**, and the RDM as currently conditioned makes the model
*worse*. The reason is structural: the RDM is a marginal of the same conditioning
the network already reads, so it cannot carry information the network lacks.

**Worth keeping as a finding**: entry *direction* alone predicts the exit facet
at 0.068, barely above the 0.0465 trivial baseline, while entry *facet* — a
proxy for position — reaches 0.192. Position matters roughly 3x more than
direction for branch selection. That is the quantitative justification for the
project's central design decision, and it did not exist before today.

---

## 7. "Coverage is not excluded" can now be partly closed.

**Claimed** — `FINDINGS.md` §"Not established": whether the limit is
representational or extrapolation beyond the training manifold is untested.

**Measured** — nearest-neighbour oracle over 1,046,359 training paths, 4,096
held-out queries:

| predictor | facet accuracy |
| --- | --- |
| most common facet, ignoring the input | 0.0465 |
| NN oracle, top-1 | 0.2869 |
| NN oracle, majority of 8 | 0.3428 |
| trained clone | **0.4103** |

The trained network **beats** the oracle by 42%. Local interpolation over a
million paths is not the ceiling, so the model is not simply failing to
interpolate a well-covered manifold.

Supporting detail: top-8 coverage is 0.386, so the true facet is absent from the
eight nearest neighbours 61% of the time; median neighbour distance 0.082,
about **4.7 degrees** of entry direction; eight near-identical entry states span
a median of **2 distinct exit facets** and all eight agree for only 18.6% of
queries.

**Replace with**: the branch boundary is high-frequency relative to achievable
sampling density, and the network already extracts more than local
interpolation can. This points at input encoding rather than at data volume or
capacity, both of which were already excluded.

---

## 8. Energy accounting figures.

**Claimed** — in analysis during this session: 87.6% of the render is neural.

**Measured** — on the component EXRs directly:

| | share of whole image | share of stone pixels |
| --- | --- | --- |
| mixture (`camera_paired`) | 75.4% | 85.5% |
| clone (`clone_paired`) | 80.2% | 88.6% |

The non-operator remainder is 12.1% background and ground plane, and 12.5%
stone pixels reached by light reflecting off the crown without entering.

**Caveat to carry**: the component split tags a path by whether it traversed the
interior, not by who decided the route. The oracle-facet render scores the same
80% while the analytic trace supplies two of its three decisions, so that figure
must not be quoted for it.

---

## 9. `geometry_leakage_validation.py` reports false leaks.

It runs on `make_flat_shaded` output, where every triangle owns its own three
vertices, so every edge appears unshared and it reports one boundary edge per
edge on a sound mesh. Use `ground_truth/pear_validation.py`, which runs on the
mesh the generator returns and also checks the Euler characteristic, normal
orientation and signed volume. All nine configured meshes pass every check.

---

## 10. A defect in today's oracle-facet ablation.

The run let the analytic trace supply **escape as well as the exit facet**. The
model's escape head fires 95.7% of the time in-render against the oracle's
99.8%, which keeps more paths alive and accounts for much of the **+28.4%**
energy error. The contrast and correlation findings are unaffected, but the
energy figure should not be quoted until the run is repeated with the model's
own escape retained. Estimated cost: one line, plus a rerun.

---

## Confirmed today, not changed

- The component split is sound. The untouched component correlates **0.9595 to
  0.9610** between neural and analytic renders — identical physics, as required
  — so the whole deficit is attributable to the operator.
- The clone samples its exit facet rather than taking the argmax
  (`multinomial`, `deterministic` defaults to False), so the branch
  multiplicity of real transport is preserved. There is no bug there.
- The mesh generators are watertight at every configured tessellation: Euler
  characteristic 2, no boundary or non-manifold edges, all normals outward,
  positive signed volume.
- The importance weighting is empirically consistent: the prior-on and
  prior-off arms agree in mean to +0.10%, against a standard error of roughly
  0.8%. This is a weak check, not a substitute for a sample/PDF unit test, but
  it is evidence the weights are not grossly wrong.

## Still open

- Analytic reference convergence has never been verified, yet every comparison
  is made against it.
- `log_exit_pdf` has zero unit-test coverage across all four test modules.
- The step-cut transfer render has not completed; three attempts were lost to
  memory pressure and one to a geometry-hash mismatch.
