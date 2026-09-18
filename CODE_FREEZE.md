# Code freeze plan — 18 September 2026

Submission: **30 September 2026**, twelve days out.

The code is roughly 93% of what the thesis needs. The document is not. Every
hour past the freeze line below is an hour taken from the thing being graded.
This file exists so that decision is made once, deliberately, rather than
re-argued every time an experiment looks tempting.

**Freeze line: items 1–5. Everything after item 5 happens only if the writing
is ahead of schedule.**

---

## Part 1 — Ordered work remaining

### Before the freeze

**1. Commit today's analysis scripts. (1 h)**
Three of the most important numbers in the thesis — the 24.66° within-facet
error, the RDM conditioning table, the 1.315x variance ratio — were produced by
throwaway scripts in a scratch directory. They are not in the repository and
cannot currently be reproduced. Promote them to:
- `measure_within_facet.py` — decode error given the true facet, per head
- `compare_rdm_conditioning.py` — facet accuracy by RDM conditioning and blend weight
- `measure_prior_variance.py` — paired variance ratio, N seeds, equal-sample and equal-time

Do this first. An unreproducible number in a thesis is a liability, and these
are load-bearing.

**2. Analytic reference convergence check. (2 h)**
Gate B7, never run. Every comparison in the project is made against the
analytic render, and nothing establishes that it has converged. Render one view
at increasing spp, plot masked RMSE against the highest, show the knee. Cheapest
insurance in the project.

**3. Sample/PDF consistency test. (half a day)**
`log_exit_pdf` has zero coverage across all four test modules. See Part 2.

**4. Fix the oracle-facet ablation and rerun. (1 h + compute)**
It currently takes escape *and* facet from the analytic trace, so its +28.4%
energy error is partly an artefact of the escape override. Keep the model's own
escape; override only the facet.

**5. Rewrite `FINDINGS.md`. (1 day)**
Apply `CORRECTIONS-2026-09-18.md`. The document still leads with the 26° blur as
a single cause, and states a 0.00° figure that has been read as though the
within-facet decode were solved. Restructure around the two-part diagnosis.

### After the freeze, only if writing is ahead

**6. Conditioning ablation.** See Part 3 — the most valuable missing experiment.
**7. Table-operator ablation.** Specified in your own contract doc, never run.
**8. Step-cut transfer render.** Three attempts lost to memory pressure.
**9. Input-encoding experiment (Fourier features).** A whole chapter's work.

### Code cleanup — do during writing breaks, not before

- **`ground_truth/geometry_leakage_validation.py` reports false leaks.** It runs
  on `make_flat_shaded` output, so every edge appears unshared. Either fix it to
  run on the generator's mesh or add a header saying it is superseded by
  `pear_validation.py`. Leaving a script that reports 192 boundary edges on a
  watertight mesh is a trap for anyone reading the repo.
- **Rename `ground_truth/pear_validation.py`.** It validates round, pear and
  step cuts. `validate_meshes.py` is what it is.
- **Document the two RDM systems.** `utils/rdm.py`, `bsdf/rdm_sampler.py`,
  `gather_rdm.py` (legacy, first-hit-local, used by `eval.py`) versus
  `neural/boundary_prior.py` (boundary-record). A reader cannot currently tell
  which is which without reading both. One README paragraph.
- **Decide about `render_paired.py` and `--rdm_prior`.** It does not expose the
  flag, which is why no evaluated render used the RDM. Either add it (1 h) or
  state in the README that the orchestrator is deliberately RDM-free. Silence is
  the one option that looks like an oversight.
- **Add `.gitattributes`.** Every commit emits CRLF warnings.
- **Index or prune `renders/` and `checkpoints/`.** Roughly 40 render
  directories and 45 checkpoints, most of them stale pilots. At minimum, a
  `renders/README.md` naming the ones the thesis cites, so provenance is
  unambiguous.
- **Remove the unused `FEATURE_NAMES` / `FEATURE_SLICES` constants** in
  `oracle_boundary.py`.

---

## Part 2 — Tests to write

Existing coverage is 14 test functions across four modules, concentrated on
transport physics and the prior's Jacobians. The gaps below are ordered by how
much they protect a claim you intend to make.

### Correctness tests

**T1. Sample/PDF consistency.** *(validation gate 1, still unmet)*
Draw many samples from `BoundaryModel.sample` for a fixed input; histogram them
over exit facet and over a coarse binning of the transformed coordinates;
compare against `exp(log_exit_pdf)` evaluated on the same bins. They must agree
within sampling error. This is load-bearing: `sample_mixture` forms its weight
as `p_neural / q`, so if these disagree, every prior-mixed render is wrong and
the variance result is meaningless.

**T2. PDF normalisation.** Monte-Carlo integrate `exp(log_exit_pdf)` over the
exit domain for several inputs; it must integrate to 1 within error. Catches a
missing or doubled Jacobian term, which T1 alone can miss if both sampler and
density share the same error.

**T3. Importance weight has unit mean.** Under the mixture proposal, the mean of
`p_neural / q` must be 1. A cheap end-to-end check on the whole prior path.

**T4. Component split is exact.** The docstring claims
`component_operator + component_untouched` equals the frame. Assert it
pixel-exact on a small render. This underwrites the 0.96 control that attributes
the entire deficit to the operator.

**T5. Facet override is honoured.** `sample(x, facet=f)` must return
`exit_facet == f` for every element, and must not change escape or coordinates
relative to a run where the categorical happened to pick `f`.

**T6. Retarget preserves weights.** `retarget_checkpoint.py` must produce a
permutation that is a bijection, leave all non-facet weights bit-identical, and
recompute `geometry_sha256` so `render_boundary` accepts it. The hash bug
already bit once.

**T7. Mesh property test.** For randomised parameters within valid ranges, every
generator returns a mesh with Euler characteristic 2, no boundary or
non-manifold edges, all normals outward, positive signed volume. Currently
checked only at four hand-picked tessellations.

**T8. `encode_data` round-trip.** Encoding an exit and decoding it back must
recover the original position and direction to float tolerance. Guards the
barycentric-logit and tangent-slope transforms.

**T9. Throughput is bounded.** No path may gain energy: `throughput <= 1` in the
unmixed sampler, and `<= 1/(1-alpha)` with the prior, which is the bound your
contract doc already states.

### Statistics-collection scripts

These are not unit tests; they produce numbers the thesis quotes, and each
should write a JSON report so a figure can be regenerated without a rerun.

**S1. `measure_within_facet.py`** — decode error given the true facet, per head,
with the nearest-neighbour value as the reference point. *Produces the 24.66 /
32.51 / 8.87 table.*

**S2. `compare_rdm_conditioning.py`** — facet accuracy for each RDM conditioning
and blend weight. *Produces the +0.003 ceiling and the 0.068-vs-0.192 result.*

**S3. `measure_prior_variance.py`** — paired variance ratio over N seeds, at
equal samples and equal time. *Supersedes the two-seed pilot figure.*

**S4. `oracle_boundary.py`** — already committed. Extend it to emit the coverage
histogram (neighbour distance distribution) as a figure, which closes the
"coverage is not excluded" item in `FINDINGS.md`.

**S5. Facet-accuracy summary** — one script that reports facet accuracy for
every head and every baseline on one scale: trivial marginal 0.0465, NN oracle
0.2869, oracle majority 0.3428, mixture, clone 0.4103. One table, one source.

---

## Part 3 — Ablations

### Already done — inventory these, do not repeat them

Data ladder (256 to 65,536 entry states); capacity (width 64 vs 256);
training distribution (uniform vs camera-matched); density sharpness
(deterministic decode); density family (mixture vs behaviour cloning);
geometry (round vs pear); branch oracle (oracle facet); identifiability
(nearest-neighbour oracle); RDM as sampler; RDM as branch prior.

That is a strong set. The gaps below are the ones that would change what you can
claim.

### A. Conditioning ablation — **the most valuable missing experiment**

Retrain the clone with inputs removed one at a time: no position, no wavelength,
no normal, direction only. Report facet accuracy and within-facet error for
each.

This directly tests the thesis's central design decision. Your whole pivot rests
on "the RDM discards position and position is what matters". You have indirect
evidence (a position-proxy table beats a direction table 0.192 to 0.068), but
the direct experiment — take position away from the network itself and watch the
accuracy fall — has never been run. If direction-only collapses toward the RDM's
0.068, the pivot is vindicated in one figure.

Cost: four short training runs. Cheapest high-value experiment left.

### B. Table operator — completes the representation ablation

`NEURAL_DIAMOND_NEXT.md` line 31 specifies *"analytic reference versus table
operator versus neural operator"*, and the middle term has never been run. Render
using `BoundaryPrior.sample` alone, with no network, by allowing `alpha = 1`
behind an explicit flag. You have shown the RDM cannot *help* the operator; this
shows whether it can *be* one. It makes the pivot airtight.

### C. Separate the two oracle overrides

Three runs: oracle escape only, oracle facet only, both. You currently have only
"both", which is why the energy figure is contaminated. Cheap and it turns one
compromised data point into a clean three-row table.

### D. Tessellation ablation — if time

Same cut at 48, 64 and 112 faces. Does facet accuracy degrade as branches
multiply? Speaks to whether the approach could ever scale to a real stone with
57+ facets.

---

## Part 4 — The story the write-up should tell

### The one-sentence claim

*Aggregate neural appearance models transfer to media where aggregation destroys
the structure being modelled, and fail where the fine structure is the
appearance — and this can be characterised precisely rather than asserted.*

Everything in the thesis should be arranged to support, qualify or test that.

### The arc

**1. The question.** Neural appearance models compress multiple scattering for
fibrous media. Diamond is the adversarial case: transport is specular, discrete
and perfectly coherent, and the facet structure *is* the appearance. Does the
technique transfer?

**2. The inherited method and its diagnosed defect.** The source method uses a
radiance distribution map conditioned on angle alone. For yarn that is
reasonable — light enters and leaves within a small cross-section. For a
brilliant cut it is not: light enters the crown and exits the pavilion, so
*which facet* carries the appearance. **Evidence: entry direction predicts the
exit facet at 0.068, barely above the 0.0465 trivial baseline, while entry facet
reaches 0.192.** This is the motivation, and it is measured, not argued.

**3. The generalisation.** A position-, direction- and wavelength-conditioned
boundary operator. Continuous inputs, no angular binning; the only discrete
variable is the exit facet, dictated by the geometry rather than chosen as a
resolution. Contrast this explicitly with the RDM — *stop binning the sphere,
start indexing the geometry and be continuous within it.*

**4. It does not work. Here is the elimination.** Present the chain as a table:
data volume, capacity, training distribution, density sharpness, density family,
geometry — each excluded by measurement. This is the spine of the thesis and the
part that distinguishes it from a project that merely reports failure.

**5. The two-part diagnosis.** Branch selection at 0.4084 facet accuracy, and a
within-facet decode error of 24.66° even when handed the correct branch. The
oracle-facet render is the pivotal figure: given the true branch it reproduces
the reference's contrast almost exactly, 3.734 against 3.744, while correlation
*falls* to 0.041. Sharpness and correctness are separate axes. Be explicit that
this corrects an earlier single-cause reading — that is a strength, not an
admission.

**6. Where the limit actually is.** The nearest-neighbour oracle over a million
paths scores 0.2869 while the trained network scores 0.4103. Local interpolation
is not the ceiling; the network already exceeds it. Combined with the exclusion
of data and capacity, this points at representation and input encoding. Eight
near-identical entry states, 4.7° apart, span a median of two distinct exit
facets — the branch boundary is high-frequency relative to any achievable
sampling density.

**7. The baseline, retained and measured.** The RDM tested in both available
roles: as an importance-sampling proposal it cannot affect accuracy by
construction and costs 1.315x variance at equal samples, 1.603x at equal time;
as a branch prior its ceiling is +0.003 facet accuracy. One explanation covers
both — it marginalises away the conditioning the operator depends on.

**8. Conclusion.** State the boundary of the technique, not the failure of the
implementation. Aggregate models work where the aggregation scale is far below
the pixel and the represented appearance is genuinely smooth. They fail where
fine structure survives to the image, and the failure is not a matter of network
size or training data — it is that the exit is a high-frequency discrete
function of the entry state, and smooth conditional density models are the wrong
class of approximator for it.

### Three things to do throughout

**Lead with measurement, not narrative.** Your instrumentation is the strongest
part of this project — noise floors measured by seed-splitting rather than
assumed, exact component splits with an identical-physics control at 0.96,
operator metrics separate from image metrics. Make that visible early; it is
what makes the negative results credible.

**Show the corrections.** `CORRECTIONS-2026-09-18.md` documents claims you
revised when measurements contradicted them. Include that as an appendix. A
project that visibly checks itself reads as more trustworthy, not less.

**Fix the title.** *"Neural diamond rendering with radiance distribution maps"*
describes a method not built; no evaluated render uses the RDM. Something like
*"A position-conditioned boundary transport operator for faceted transparent
solids"* describes the work, keeps the lineage visible, and claims only what is
defensible. This is the single largest unforced risk in the project and it costs
nothing to fix.

### What to say about speed

Say it plainly and early: **1.46x slower**, and the premise was weak before any
accuracy question arose, because mean path depth is 9.1 and a network query
costs more than a ray-triangle intersection. Stating this yourself is far
stronger than having it drawn out of you.
