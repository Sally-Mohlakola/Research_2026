# Neural Diamond Rendering — Remaining Work

A learned boundary transport operator for a round brilliant, adapted from the
neural cloth appearance model of Soh and Montazeri. The research question is
whether aggregate neural appearance models transfer from fibrous media, where
appearance genuinely is smooth at the aggregation scale, to resolved specular
media, where the fine structure *is* the appearance.

The answer so far is no, and the falsification is well isolated. See
`NEURAL_DIAMOND_NEXT.md` for the full findings and implementation contract.

## Where things stand

The model is trained, converged and evaluated. The failure is **representational**,
not a shortage of data or capacity.

Figures below are the converged paired comparison at 728x728 and 128 effective
spp (`renders/flash_128spp/evaluation.json`), eight seeds per mode.

| established | evidence |
| --- | --- |
| not data limited | data ladder, 256 to 65,536 entry states; gains flatten |
| not capacity limited | width 256 buys 0.40 nats then overfits by epoch 60 |
| the model genuinely learned | held-out joint NLL 11.415 to 6.605; facet top-1 0.094 to 0.383 against a 0.030 unconditional baseline |
| the image did not improve | spatial correlation 0.098 against a measured 0.955 noise floor |
| it is not merely blurred | blur preserves correlation; highlight overlap 18.5 percent against a 78.0 percent floor |
| energy is biased well beyond noise | plus 15.8 percent against a noise floor of minus 0.017 percent, a ratio of 925 |
| no scintillation | contrast 0.80 against 3.81; peak over mean 23.5 against 139.7; flash fraction 0.16 against 0.64 percent |
| it costs more, not less | 1.57x noisier per pixel and 1.46x slower, about 3.5x analytic compute for equal quality |
| the learned component carries the image | 87.6 percent of stone radiance, isolated correlation 0.045 |
| and the apparatus is sound | the first-surface component, identical physics in both renders, correlates at 0.922 in the same image under the same noise, a 20x gap |

The last row is the control that makes the rest defensible. It rose from 0.40 at
8 spp to 0.922 at 128 spp, as an identical signal must, while the learned
component stayed near zero.

A second diagnostic worth a figure: neural contrast *falls* with sampling
(1.664, 0.972, 0.799 at 16, 64, 128 spp) while analytic contrast holds
(3.87, 3.81). Apparent facet detail in low-sample neural renders is speckle,
not structure, and averages away.

Working tools: `gather_boundary.py`, `train_boundary.py`, `evaluate_boundary.py`,
`render_boundary.py`, `render_paired.py`, `merge_renders.py`, `evaluate_render.py`,
`flash_sweep.py`, `make_boundary_subset.py`, `build_boundary_prior.py`. Fourteen
tests pass under `python -B -m unittest discover -s tests -p 'test_boundary*.py'`.

---

## 1. Neural importance sampling over analytic branches

**Effort: about one week. This is the highest-value remaining item.**

Everything else below is polish on a negative result. This is the only task that
turns "we falsified X" into "we falsified X and showed what works instead", which
is what separates a strong pass from a distinction.

Keep exact Fresnel, delta refraction and total internal reflection. The network
only reweights the choice among physically valid branches, with the PDF
correction applied exactly. This formulation cannot reproduce either failure
measured so far: the singular structure is preserved by construction, and the
estimator stays anchored to a physical proposal with nonzero support.

It does not need to beat analytic transport. It needs to be measured on the
yardstick already built (`evaluate_render.py`, `flash_sweep.py`).

**Check with the supervisor first.** If the requested behavioural cloning test
means cloning the sampler's branch decisions, it is this task, and two work items
collapse into one.

**Produces:** a method section, a variance comparison against naive branch
sampling, and a positive result to set beside the falsification.

## 2. Mode multiplicity in the gathered data

**Effort: about one day.**

There is currently empirical evidence that smooth densities fail, but no
*argument* for why. This supplies the mechanism, and the data already exists.

A K-component mixture represents at most K modes. Paths average depth 9.1 with a
reflect/refract branch at every interface, so the number of distinct branch
sequences reaching a facet is large. Cluster the 16 repeats per entry state in
`checkpoints/boundary_scale_v1/train_pool.npz`, count distinct exit modes per
(entry, facet) pair, and plot that distribution against K = 4.

**Produces:** the central claim restated quantitatively — the target has N modes,
the model has 4, and here is the resulting contrast deficit. Pairs directly with
item 4.

## 3. Nearest-neighbour oracle and query coverage

**Effort: one to two days.**

This closes the one hole in the main claim. The conditioning is at least 5D
(surface position, direction, wavelength) and 65,536 entry states is roughly nine
samples per dimension. A sharp examiner can ask whether the failure is
representational or simply sparse coverage, and that question cannot currently be
answered.

- **Coverage:** instrument `render_boundary.py` to record, per neural query, the
  distance to the nearest training entry state in the model's own normalized
  encoding. Histogram against nearest-neighbour distances within the training set.
- **Oracle:** replace the network with a lookup that retrieves a real recorded
  escape event from the nearest training entry state and uses its actual exit
  position and direction. This is the best achievable with this conditioning and
  this data, with zero smoothing.

Outcomes are mutually exclusive and all publishable: the oracle flashes and the
network does not, confirming representation failure; neither flashes, meaning the
conditioning variables do not determine the exit sharply enough; or the oracle
needs a much denser pool, meaning coverage failure and the ladder must extend.

## 4. Mixture component sweep

**Effort: about one day, mostly compute.**

Train at K = 1, 4, 8, 16, 32, 64 on the full pool with everything else fixed, and
measure contrast and flash fraction in render space, not just likelihood.

This defends the claim against the obvious objection that the result shows only
that *four* Gaussians are insufficient rather than that smooth densities are. If
sharpness keeps climbing with K, the thesis must be restated more narrowly, and
it is far better to discover that before a marker does.

## 5. Sample and PDF consistency test

**Effort: half a day.**

Validation gate 1 requires sample/PDF agreement and it is absent. `log_exit_pdf`
has analytic Jacobian tests, but nothing verifies that `BoundaryModel.sample`
actually draws from that density.

This is load bearing: the RDM prior's transport weight is `p_neural/q`, so if
sampling and density disagree, that entire milestone is unsound.

Implement as a histogram or chi-square test — many samples at fixed conditioning,
binned over facet, barycentric cell and direction cell, compared against the
integrated density.

## 6. Analytic reference convergence

**Effort: half a day, mostly compute.**

The analytic render is used as the reference and its seed-split self-agreement as
the noise floor, but it has never been shown to be converged. Render one very high
sample-count analytic frame and verify that the 64 and 128 spp versions approach
it, and that the 0.912 correlation floor reflects noise rather than bias.

## 7. Table operator

**Effort: about one day.**

Named in validation gate 3 and still missing. A histogram over the same
conditioning, with no network, separates information lost in discretizing the
conditioning from network fitting error.

With item 3 this completes the ladder: analytic, oracle, table, neural.

## 8. Multi-view and held-out lighting

**Effort: about one day, mostly compute.**

Currently one studio rig, one high-resolution azimuth, fifteen azimuths at 96x96.
Closes validation gate 4 and answers whether the conclusion generalizes.

```bash
python -B render_paired.py --model checkpoints/boundary_scale_v1/final_w064c4/model.pt \
  --output_dir renders/view_<azimuth> --azimuth <azimuth> --width 728 --height 728 \
  --spp 16 --seeds 101 102 103 104 105 106 107 108 --workers 8 --split_components
```

A second lighting rig requires extending `utils/studio_env.py`.

## 9. Write the paper around the question, not the artefact

**Effort: four to five days. Worth more than any single experiment above.**

The same evidence framed as *"Do aggregate neural appearance models transfer to
resolved specular media?"* rather than *"A neural diamond renderer"* is worth
roughly fifteen points. Write the question first and let every number answer it.

Present the validation methodology as a contribution in its own right: measured
rather than assumed noise floors, correspondence-free scintillation statistics,
and an exact neural/analytic component split. Most work in this area reports
whole-image metrics on hybrid pipelines without ever stating what fraction is
neural.

State plainly that camera, geometry, Fresnel coefficients, first-surface
reflection and scene illumination remain analytic, and that the network replaces
interior transport conditional on entry transmission. That isolation is what
licenses attributing every measured discrepancy to the learned component.

Address the late baseline discovery directly rather than hoping it passes
unnoticed. "The conditional model was only 0.14 nats better than the marginal,
which we established late and which reframed the project" is a stronger sentence
than silence.

### Figures the paper needs

1. Analytic against neural at 728x728, identical camera, lights and sample count
2. Data ladder: gain over the unconditional baseline against entry-state count
3. Flash sweep: contrast and flash fraction against azimuth, both modes, with the noise band
4. Component decomposition: operator against untouched
5. Mode multiplicity against mixture component count (item 2)
6. Contrast against K (item 4)
7. Representation ladder: analytic, oracle, table, neural (items 3 and 7)
8. Importance sampling variance against the baseline (item 1)

---

## Expected banding

| scope | band |
| --- | --- |
| items 2 to 9 without the constructive alternative | 85 to 88 |
| plus item 1, even with modest results | 88 to 92 |
| plus item 2 argued rather than only measured | top of that band |

## If only two weeks remain

Do items 1, 3 and 4, then write. Skip items 7 and 8 and say so explicitly under
limitations. A thesis that names its own gaps reads as controlled; one that
quietly has them reads as incomplete.

## Environment notes

- Runs under both Windows Python and the WSL conda environment `jojo`. Checkpoints
  and datasets live on the Windows filesystem; mixing toolchains mid-experiment
  risks results produced by an unvalidated stack.
- Set `PYTHONIOENCODING=utf-8` whenever stdout is piped or redirected, or
  `eval.py` fails on a Unicode character before rendering starts.
- `eval.py` is the legacy path and needs `--diamond_name` passed explicitly; its
  default geometry does not match the checkpoints gathered on `round_diamond_gia`,
  and nothing checks this.
- Render processes hold Mitsuba and PyTorch at roughly 250 MB each. Eight
  concurrent renders alongside training will exhaust memory.
