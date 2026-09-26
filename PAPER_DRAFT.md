# When Operator Accuracy Does Not Transfer to the Image: A Position-Conditioned Boundary Transport Operator for Faceted Transparent Solids

**Sally Mohlakola** — Honours research, draft of 25 September 2026

> Draft status. Every number in this document is taken from a report file in the
> repository; the provenance table in Appendix A names each source. Citations in
> the references section should be checked against the originals before
> submission.

---

## Abstract

Neural appearance models compress multiple scattering into a learned function
and have been applied successfully to fibrous media such as cloth, where the
scattering structure sits well below the scale of a pixel. We ask whether the
approach transfers to the opposite case: a brilliant-cut diamond, where light
transport is specular, discrete and perfectly coherent, and where the fine
structure *is* the appearance.

We first show that the radiance distribution map (RDM) used by the source
method conditions on the wrong variable for this geometry. On held-out
transport, entry direction predicts which facet light will exit through at
0.068 top-1 accuracy against a trivial baseline of 0.047, whereas the entry
facet reaches 0.192. We therefore generalise the representation to a
*boundary transport operator*: a small neural network that takes the entry
position, direction, surface normal and wavelength of a light path and
replaces the entire interior journey with a single query, returning an escape
decision, an exit facet, and a continuous exit point and direction.

The operator learns: it beats every non-learned baseline, including a
nearest-neighbour oracle over a million training paths (0.410 against 0.287),
and a conditioning ablation confirms that spatial inputs are worth a factor of
3.5 in branch selection. It does not, however, reproduce the image. Rendered
against an analytic reference, spatial correlation is 0.17 where independent
renders of the reference agree at 0.98. An elimination chain excludes training
data volume, model capacity, training distribution, density sharpness, density
family and stone geometry as the cause, and isolates two failures: the
operator selects the wrong exit facet about 60 percent of the time, and even
when given the correct facet it places the exit 24.7 degrees from the truth.

The central finding is that **operator accuracy does not transfer to the
image**. Across four independent experiments, every measurable improvement to
the operator made the rendered image agree with the reference *less*, while
pushing its contrast toward, and in one case past, that of the real stone. A
sharp operator that places light incorrectly is further from the truth than a
blurred one, because blur buys accidental overlap with the correct regions.
The two error sources are therefore not additive, and partial fixes move the
image away from the reference. We argue that the remaining limit lies in the
representation rather than in data or capacity, and identify input encoding as
the most strongly indicated next step.

---

## 1. Introduction

### 1.1 Motivation

Rendering a diamond by path tracing is expensive in a specific way. A light
path that enters the stone undergoes, on average, around nine internal
interactions — refraction in, a sequence of total internal reflections, and
refraction out — before it escapes, and each interaction depends
wavelength-dispersively on the refractive index. The appearance of the stone,
its brilliance, fire and scintillation, is produced entirely by that interior
journey.

Neural appearance models offer a way to amortise such journeys. Soh and
Montazeri [1] compress the multiple scattering inside a yarn into a small
network conditioned on incoming and outgoing directions, trained from a
radiance distribution map gathered by tracing the true transport. For cloth
this is well motivated: a yarn contains hundreds of fibres, their scattering is
aggregated far below the scale of a pixel, and the represented appearance is
genuinely smooth.

A diamond inverts every one of those conditions. It has a few dozen facets,
each resolved across thousands of pixels. Its transport is specular rather
than diffusive: one entry state produces a small number of exact, coherent
beams rather than a lobe. And its appearance consists precisely of the fine
angular structure that aggregation discards. A diamond is therefore an
adversarial test of whether the aggregate approach transfers, and the answer
tells us where its boundary lies.

### 1.2 Research question

> **Do aggregate neural appearance models transfer from fibrous media, where
> the represented appearance is smooth at the aggregation scale, to resolved
> specular media, where the fine structure is the appearance?**

### 1.3 Contributions

1. **A diagnosis of the RDM for faceted geometry**, measured rather than
   argued. The RDM conditions on direction and discards position; for a
   brilliant cut, position carries roughly three times as much information
   about the exit facet.
2. **A position-conditioned boundary transport operator** that generalises the
   RDM: continuous conditioning on position, direction, normal and wavelength,
   with the only discrete variable, the exit facet, dictated by the geometry
   rather than by a chosen bin resolution.
3. **An evaluation apparatus** designed to make a negative result credible:
   noise floors measured by seed-splitting rather than assumed, an exact
   per-path split of the image into learned and untouched components with an
   identical-physics control, operator-level metrics separated from image
   metrics, and a nearest-neighbour oracle that tests identifiability
   independently of any architecture.
4. **An elimination chain** excluding data volume, capacity, training
   distribution, density sharpness, density family and geometry as causes.
5. **A two-part diagnosis** — branch selection and within-facet decode — and
   the finding that improving either in isolation makes the image worse.
6. **A measured evaluation of the RDM in both roles the architecture
   permits**, as an importance-sampling proposal and as a branch prior, with a
   single explanation for why neither helps.

---

## 2. Background and related work

**Neural appearance models for aggregate media.** Soh and Montazeri [1] fit
small networks to the multiple-scattering component of yarn appearance,
training from a radiance distribution map gathered by tracing ground-truth
transport and conditioning on incoming and outgoing directions. Our starting
point was an adaptation of that method to diamond; this repository retains it
as the legacy pipeline (`neural/base_model.py`, `utils/rdm.py`, `eval.py`).

**Learned transport through a bounded region.** Predicting where light leaves
a bounded region given where it enters is established for diffusive media;
shape-adaptive learned subsurface scattering [2] is close prior art. The
regime studied here differs materially: the exit is a discrete, high-frequency,
deterministic function of the entry state rather than a smooth diffusive
distribution.

**Mixture density networks.** The operator's primary head is a conditional
mixture density model [3]: it outputs a probability distribution over exits
from which the renderer draws, rather than a single regression target.

**Spectral bias.** Multilayer perceptrons fed raw coordinates are biased
toward low-frequency functions, and Fourier-feature input encodings are the
standard remedy [4]. This bears directly on our conclusion (Section 7.3).

**Rendering infrastructure.** All rendering uses Mitsuba 3 [5] in its scalar
spectral variant with hero-wavelength sampling [6].

---

## 3. Research questions

The top-level question decomposes into five sub-questions, each answered by a
measurement in Section 6.

| | question | answered in |
| --- | --- | --- |
| RQ1 | Is the source method's representation suited to faceted specular transport? | 6.1, 6.4 |
| RQ2 | Can a learned operator reproduce the appearance of a diamond? | 6.2 |
| RQ3 | If not, what limits it? | 6.3, 6.5, 6.6, 6.7 |
| RQ4 | Does improving the operator improve the image? | 6.8 |
| RQ5 | Does the RDM earn a place alongside the operator? | 6.9 |

---

## 4. Method

### 4.1 Scene and reference renderer

A single studio scene is used throughout: a perspective camera with a
30-degree field of view, a grey diffuse ground plane, and a structured lighting
rig (`utils/studio_env.py`) containing two small, intense sparks. The rig is
deliberately non-uniform. Under a constant environment every escape direction
returns the same radiance and a transparent object renders as featureless grey,
so structured lighting is a precondition for fire and scintillation to be
visible at all.

Rendering is spectral with one hero wavelength per camera sample, traced
through a four-lane spectral packet collapsed to a single wavelength. Diamond's
dispersion is modelled explicitly, with refractive index running from 2.4641 at
400 nm to 2.4062 at 700 nm.

The **analytic reference** traces the true interior journey: at every internal
interaction, reflection or refraction is chosen by the Fresnel coefficient,
until the path escapes or reaches a depth cap of 128. The estimator is
BSDF-only, with no next-event estimation or multiple importance sampling.

### 4.2 Geometry

Three cut families are generated procedurally (`ground_truth/`):

| cut | vertices | triangles | notes |
| --- | --- | --- | --- |
| round brilliant (`round_diamond_gia`) | 34 | 64 | crown 34.5°, pavilion 40.75°, table 0.56, small culet |
| pear brilliant | 33 | 62 | teardrop outline, same crown and pavilion angles |
| step (emerald) cut | 58 / 34 | 112 / 64 | octagonal outline, concentric terraces, flat culet |

All three share the same girdle radius and nominal proportions, so a cross-cut
comparison changes facet structure and nothing else. Every mesh is validated
for zero boundary edges, zero non-manifold edges, Euler characteristic 2,
outward normals, positive signed volume, no degenerate faces and no duplicate
vertices; all nine configured tessellations pass.

### 4.3 The boundary transport operator

**Definition.** When a camera path refracts into the stone, four quantities are
known: the entry position, the incident direction, the surface normal, and the
wavelength. The operator maps these ten numbers to the outcome of the entire
interior journey:

1. whether the path escapes at all;
2. which of the stone's triangles it exits through;
3. where on that triangle, and in what direction.

It is a direct generalisation of the RDM along the axis the RDM discards.
Where an RDM answers *light arriving from this direction leaves in these
directions*, the operator answers *light entering at this point, in this
direction, at this wavelength, leaves at this point on this facet, in this
direction*.

**Factorisation.**

```
p(exit | entry) = p(escape | x) · p(facet | x) · p(coordinates | x, facet)
```

The first factor is a Bernoulli, the second a categorical over the mesh's
triangles, and the third a four-component Gaussian mixture over four
continuous coordinates.

**Coordinates.** The exit point is expressed as two barycentric *logits* on
the chosen triangle and the exit direction as two slopes in that triangle's
tangent plane. Both are unbounded, which allows a Gaussian to be placed on
what is physically a bounded domain; the density is corrected by a
log-Jacobian so that it remains a valid density over surface area and solid
angle.

**Architecture.**

```
input        10 numbers: position / radius (3), direction (3), normal (3),
             (wavelength - 595) / 235 (1)
encoder      Linear(10→64) → SiLU → Linear(64→64) → SiLU
  escape     Linear(64→1)
  facet      Linear(64→n_facets)
  embedding  Embedding(n_facets, 16)
  mixture    Linear(80→64) → SiLU → Linear(64→4×9)
```

Roughly 18,800 parameters. Two further heads share the encoder, escape head and
facet head:

- **Behaviour clone** (15,557 parameters): the mixture is replaced by a
  deterministic smooth-L1 regression of the coordinates, conditioned on the
  exit facet. This isolates *distributional versus deterministic* as the single
  variable.
- **Deep clone** (28,579 parameters): the clone with a four-layer coordinate
  regressor reading its own trunk, selected by the decoder sweep in
  Section 6.7.

The network definition itself is about fifteen lines. The operator stack as a
whole — model, data conventions, gathering, training and render integration —
is about 1,300 lines; the evaluation and experiment apparatus around it is
about 2,400.

### 4.4 Data gathering

Training data are generated by tracing the analytic reference honestly
(`gather_boundary.py`). Rays are fired at the stone from camera-like entry
states, refracted in, and traced through the full interior journey. Each record
stores the entry state, the exit facet, position and direction, the path depth,
and a status flag distinguishing escaped, truncated and invalid paths, so the
network is never trained to fabricate an exit for a path that has none.

Entry states are sampled to match the distribution a camera actually queries
over a full azimuth range, rather than uniformly over the surface; Section 6.3
shows why this matters. The training pool contains 1,048,576 records from
65,536 entry states, and a separately seeded held-out pool contains 262,144
records from 16,384 entry states. Gathering runs in parallel, producing a
million records in about 28 seconds.

### 4.5 Training

Adam with learning rate 10⁻³, batch 256. The loss is the escape binary
cross-entropy plus the escape-weighted sum of the facet cross-entropy and the
coordinate term. The train/validation split is **grouped by entry state**: all
repeated outcomes of one entry state fall on the same side, since otherwise the
model would see the answer at validation time. The best-validation checkpoint is
saved together with the geometry, its hash, the split identifiers and the data
conventions, and the renderer refuses to run on a checkpoint whose geometry hash
does not match.

### 4.6 Render integration

The operator replaces only the interior journey. A camera path hits the stone;
the Fresnel coefficient decides between reflection and transmission, sampled
analytically; on transmission, the entry state is added to a batch of 512
queries evaluated together. Each returned exit spawns a fresh ray into the
scene. A predicted exit that would immediately re-enter the stone is discarded
rather than corrected. Because entry transmission has already been sampled,
the neural branch carries unit throughput; multiplying by the transmittance a
second time would double-count it.

Everything outside the stone — camera rays, the first surface hit, the crown
reflection, the ground plane, the lighting and all transport after exit —
remains ordinary path tracing, identical in both modes.

### 4.7 The RDM baseline

The RDM baseline (`neural/boundary_prior.py`) is a directional table of the
probability of exit-direction bin given entry-direction bin, built from the
same training records, 4 × 8 bins with a 5 percent uniform floor. It can be
used as an importance-sampling proposal mixed with the operator:

```
q(x) = (1 - α) · p_operator(x) + α · p_RDM(x),     weight = p_operator(x) / q(x)
```

with α = 0.25. By construction, this estimator converges to the operator's
distribution regardless of α.

---

## 5. Evaluation methodology

The methodology is designed around one concern: a negative result is only as
credible as the apparatus that produced it.

**Paired renders.** Neural and analytic renders are produced at identical
settings — 512 × 512, azimuth 0, box filter — each as eight independent seeds of
32 samples per pixel, averaged to 256 effective samples per pixel. Averaging
independent seeds is the same estimator as one render at the combined sample
count.

**Stone-pixel masking.** Agreement statistics are computed only over pixels
whose primary ray hits the stone (134,158 pixels for the round cut), so the
background cannot dilute an error.

**Measured noise floors.** Splitting the eight seeds into two independent
halves gives two estimates of the same image; their disagreement is pure Monte
Carlo noise. The neural-versus-analytic comparison is repeated on matching
half-splits, so signal and noise are compared at equal sample counts. The
analytic reference agrees with itself at spatial correlation 0.976.

**Image metrics.** Spatial correlation; overlap of the brightest 1 percent of
pixels; relative energy error; relative RMSE; chromaticity error; contrast
(standard deviation over mean); peak over mean; and flash fraction.

**Exact component split.** Every camera path is tagged by whether it passed
through the learned operator, splitting the frame exactly into an *operator*
component and an *untouched* component. The untouched component is identical
physics in both modes and serves as a control; it agrees between modes at
0.958 to 0.961 in every run reported here, confirming that the harness is sound
and that any deficit is attributable to the operator. The operator carries 75.4
percent of the rendered image's energy and 85.5 percent of the stone's.

**Operator-level metrics.** Measured directly on held-out transport with no
rendering involved: exit-facet top-1 accuracy (*branch selection*), and the
angle between predicted and true exit direction when the model is handed the
correct facet (*within-facet decode*).

**Nearest-neighbour oracle.** For each held-out entry state, the exits of the
most similar training entries are used as a prediction. This involves no
training and no capacity limit, and so tests whether the exit is predictable
from the conditioning at all, independently of any architecture.

**Ablation protocol.** Every ablation arm shares one budget, one seed and one
entry-grouped split, and is compared against a control arm trained under the
same conditions rather than against results from longer runs.

---

## 6. Results

### 6.1 The RDM conditions on the wrong variable (RQ1)

Lookup tables built from real transport, each asked to predict the exit facet
on held-out data:

| conditioning | cells | exit-facet top-1 |
| --- | --- | --- |
| trivial: always the most common facet | — | 0.0465 |
| **entry direction only, as the RDM** | 32 | **0.0682** |
| entry facet, a proxy for position | 64 | 0.1918 |

Direction alone is barely above the trivial baseline, and increasing the table
from 32 to 2,048 cells raises it only to 0.098. Resolution is not the
limitation; the conditioning variable is. For yarn this is the right choice,
since light enters and leaves within a sub-pixel cross-section. In a brilliant
cut light enters the crown and exits the pavilion, and which facet it exits
through carries the appearance.

### 6.2 The operator learns, but the image does not follow (RQ2)

The operator is well above every non-learned baseline:

| predictor | exit-facet top-1 |
| --- | --- |
| trivial baseline | 0.0465 |
| RDM table, direction | 0.0682 |
| RDM table, entry facet | 0.1918 |
| nearest-neighbour oracle, top-1 | 0.2869 |
| nearest-neighbour oracle, majority of 8 | 0.3428 |
| operator, mixture head | 0.3795 |
| operator, behaviour clone | **0.4103** |

It also predicts escape essentially perfectly: 0.9984 predicted against 0.9975
measured.

Rendered, it does not reproduce the stone. Round cut, mixture head, 256
effective samples per pixel:

| | operator | analytic noise floor |
| --- | --- | --- |
| spatial correlation | **0.174** | 0.976 |
| highlight overlap, top 1% | 0.235 | 0.847 |
| relative energy error | −1.1% | −0.1% |
| chromaticity error | 0.272 | 0.180 |
| contrast | 0.847 | 3.744 (reference) |
| peak over mean | 20.2 | 121.1 (reference) |
| operator component correlation | 0.125 | — |
| untouched component correlation | 0.961 | — |

Energy is correct, but light is not where it should be, and the stone has less
than a quarter of the reference's contrast. Visually it reads as frosted glass:
it transmits light but loses the direction information.

### 6.3 Elimination chain (RQ3)

Each plausible explanation was tested directly.

**Not data.** Identical architecture on training pools from 256 to 65,536 entry
states, evaluated on one held-out pool:

| entry states | train records | facet NLL | top-1 |
| --- | --- | --- | --- |
| 256 | 3,259 | 3.960 | 0.048 |
| 1,024 | 13,059 | 3.655 | 0.108 |
| 4,096 | 52,248 | 3.327 | 0.207 |
| 16,384 | 209,121 | 2.874 | 0.332 |
| 65,536 | 836,425 | 2.773 | 0.347 |

Gains flatten above roughly 16,000 entry states.

**Not capacity.** Width 256 with eight components improves validation NLL by
0.33 nats over width 64, then overfits, drifting 0.198 nats past its minimum by
epoch 80. Capacity between these two sizes is unexplored (Section 8).

**Not training distribution.** Uniform surface sampling queries the operator
off the distribution a camera produces. Camera-matched gathering improved
held-out likelihood by 2.12 nats and corrected a +15.8 percent energy bias to
−1.1 percent, but left spatial agreement far below the noise floor.

**Not density sharpness.** The mixture spreads exits within a facet by a median
of 25.98 degrees, where real transport spreads by 0.00 degrees — each beam is
perfectly coherent. Decoding at the component mean reduces this to 8.09
degrees. Correlation fell from 0.174 to 0.153.

**Not density family.** Replacing the mixture with deterministic behaviour
cloning removes within-facet spread entirely and improves branch selection,
0.3795 to 0.4103. Correlation fell to 0.049.

**Not geometry.** The pear cut reproduces the failure, correlation 0.164
against a noise floor of 0.950. The operator's output is also nearly
cut-invariant: the reference's contrast changes by 61 percent between the round
and pear cuts (3.744 to 1.460), while the operator's changes by 7 percent
(0.847 to 0.784).

### 6.4 The operator needs position, and does not need wavelength

Inputs removed one group at a time by zeroing, with the architecture held
identical so that differences reflect information rather than capacity. Sixty
epochs per arm on 838,736 records; held-out evaluation on 261,447.

| arm | exit-facet top-1 | vs full | within-facet error |
| --- | --- | --- | --- |
| full | **0.4042** | — | 24.68° |
| no wavelength | 0.4039 | −0.0004 | 24.76° |
| no normal | 0.3662 | −0.0380 | 25.88° |
| no position | 0.2732 | −0.1311 | 25.54° |
| no direction | 0.2010 | −0.2032 | 29.15° |
| **direction only** | **0.1167** | **−0.2875** | 30.41° |

Given only what the RDM sees, the network falls to 0.117, close to the RDM
table's own 0.068 — though it still extracts 1.7 times more from the same
inputs than a table does. Spatial conditioning is worth a factor of 3.5.

Position and normal are partially redundant, since each identifies the entry
facet. Removing either alone costs 0.131 and 0.038, but removing both, together
with wavelength, costs 0.288 — far more than their sum. The single-drop arms
therefore understate the value of spatial information, and only the
direction-only arm measures it.

Wavelength contributes nothing measurable. The model does learn a wavelength
dependence — querying one entry state across the visible spectrum shifts its
mean exit by a median of 6.91 degrees — but that dependence is worth 0.08
degrees against a 24.68-degree error.

The within-facet error barely responds to any input, remaining between 24.7
and 25.9 degrees for every arm that keeps a spatial input. This is independent
evidence that it is a separate problem from branch selection.

### 6.5 The failure has two parts

| component | measured | reference point |
| --- | --- | --- |
| branch selection | exit-facet top-1 0.41 | 1.0 |
| within-facet decode | 24.66° error given the true facet (clone); 32.51° (mixture) | 8.87° for a nearest-neighbour lookup |

The distinction between **spread** and **error** is essential here and was
initially missed. Real transport has zero within-facet spread, and the
behaviour clone also has zero spread by construction — yet the clone's exits
are still 24.66 degrees from the truth. Zero spread is not accuracy.

The oracle-facet experiment separates the two parts in the image. Supplying the
true exit facet from the analytic trace and leaving the network only the
within-facet coordinates:

| | clone, oracle facet | reference |
| --- | --- | --- |
| contrast | **3.734** | 3.744 |
| peak over mean | 114.7 | 121.1 |
| chromaticity error | **0.224** | floor 0.180 |
| spatial correlation | **0.041** | floor 0.976 |
| operator component correlation | 0.017 | — |

Given the correct branches, the render recovers the reference's contrast almost
exactly and achieves the best chromaticity of any run, since correct facets
produce correct dispersion geometry. Spatial agreement nonetheless falls to
0.041: the sparkle returns, in the wrong places.

### 6.6 Local interpolation is not the ceiling

The nearest-neighbour oracle over 1,046,359 training paths reaches 0.287 top-1,
or 0.343 by majority vote over eight neighbours. **The trained network beats it
by 42 percent.** The model is therefore not merely failing to interpolate a
well-covered manifold.

The neighbourhoods show why interpolation fails. The true facet is absent from
the eight nearest neighbours 61 percent of the time; the median neighbour
distance corresponds to about 4.7 degrees of entry direction; and eight
near-identical entry states span a median of two distinct exit facets, all
eight agreeing for only 18.6 percent of queries. The branch boundary is
high-frequency relative to any achievable sampling density. When the oracle
does identify the correct facet, however, its exit direction is within 8.87
degrees: within a branch, the map is smooth.

### 6.7 The decode improves with depth, then saturates

The within-facet decode was attacked directly. Two hypotheses: the two-layer,
5,444-parameter coordinate head is too shallow; and the trunk it shares with
the facet head is dominated by the 64-way cross-entropy. Thirty epochs per arm,
facet accuracy reported as a control:

| arm | within-facet error | vs baseline | median | facet top-1 |
| --- | --- | --- | --- | --- |
| separate trunk + depth 4 | **21.61°** | −2.91 | 14.22° | 0.3941 |
| depth 4 | 22.04° | −2.48 | 14.94° | 0.3887 |
| separate trunk | 24.09° | −0.43 | 17.57° | 0.3955 |
| coordinate loss × 5 | 24.25° | −0.27 | 17.74° | 0.3869 |
| baseline | 24.52° | — | 18.19° | 0.3935 |

Depth is the active ingredient. The shared-trunk hypothesis was largely wrong:
a separate trunk alone contributes 0.43 degrees and loss weighting nothing.
More importantly, the best arm closes 2.91 of the 15.65 degrees separating the
shipped head from the parameterless lookup — 18.6 percent — despite 85 percent
more parameters. The limit is not how much the head can compute.

### 6.8 Operator accuracy does not transfer to the image (RQ4)

This is the principal result. Across four independent experiments, each
measurable improvement to the operator made the rendered image agree with the
reference less.

| experiment | operator-level change | contrast | image correlation |
| --- | --- | --- | --- |
| mixture (baseline) | facet 0.380, within-facet 32.5° | 0.847 | **0.174** |
| deterministic decode | within-facet spread 26.0° → 8.1° | 1.018 | 0.153 |
| behaviour clone | spread 0°, facet 0.410, within-facet 24.7° | 2.320 | 0.049 |
| clone with oracle facet | branch error removed entirely | 3.734 | 0.041 |
| *reference* | — | *3.744* | *floor 0.976* |

Round cut, 256 effective samples per pixel, half-split correlation.

The fourth instance is on the pear cut, replacing the mixture with the deep
clone of Section 6.7, evaluated against the same analytic reference at the same
four seeds:

| | mixture | deep clone | reference |
| --- | --- | --- | --- |
| within-facet error | 37.16° | **26.25°** | — |
| exit-facet top-1 | 0.309 | **0.337** | — |
| spatial correlation | 0.136 | **0.054** | floor 0.899 |
| relative energy error | +17.3% | +47.8% | — |
| contrast | 0.886 | **2.315** | 1.476 |
| peak over mean | 20.0 | 82.3 | 64.4 |

**The improved operator overshoots.** Its contrast passes the reference and
keeps going. This is genuine structure rather than Monte Carlo speckle, which
matters because neural contrast is otherwise known to be noise-inflated at low
sample counts:

| contrast | 32 spp | 64 spp | 128 spp | change |
| --- | --- | --- | --- | --- |
| deep clone | 2.585 | 2.390 | 2.315 | −10% |
| mixture | 1.393 | 1.069 | 0.886 | −36% |
| analytic reference | 1.564 | 1.507 | 1.476 | −6% |

The mixture's contrast was largely an artefact of noise; the deep clone's
behaves like the analytic reference's and is converging on real structure. The
improved operator therefore recovers — and exceeds — the right *amount* of
structure without recovering the right *image*.

The difference image makes the two error modes visible: missing light appears
in facet-shaped regions, which is branch selection failing, and added light
appears in diffuse blobs, which is the within-facet decode placing exits where
no facet directs them.

### 6.9 The RDM in both available roles (RQ5)

**As an importance-sampling proposal**, the RDM cannot change the image's
expected value, because the weights target the operator by construction; it can
only change variance. Four matched seeds per arm, 160 × 160 at 16 samples per
pixel:

| | proposal on | proposal off |
| --- | --- | --- |
| mean luminance | 0.002596 | 0.002594 |
| per-pixel variance | 4.105 × 10⁻⁵ | 3.122 × 10⁻⁵ |
| time | 29.6 s | 24.3 s |

Variance increases by a factor of 1.315 at equal samples and 1.603 at equal
time. The means agree to 0.1 percent, confirming the weighting behaves as
designed: the same expected image from a worse estimator.

**As a branch prior**, combining the RDM's implied facet distribution with the
operator's posterior as a product of experts:

| RDM conditioning | cells | RDM alone | best blend |
| --- | --- | --- | --- |
| entry direction, as built | 32 | 0.0682 | 0.4094 |
| entry facet | 64 | 0.1918 | 0.4103 |
| entry facet × direction | 8,192 | 0.2594 | **0.4135** |

against an operator baseline of 0.4103. The ceiling is +0.003.

One explanation covers both roles. The RDM is a marginal of the conditioning
the operator already receives, so it cannot carry information the operator
lacks. No result in Section 6.2 onward uses it.

### 6.10 Cost

| | analytic | operator | ratio |
| --- | --- | --- | --- |
| seconds per 8-spp render at 728 × 728 | 424 | 620 | 1.46 |
| median per-pixel relative standard error | 0.222 | 0.348 | 1.57 |

Matching analytic noise would require about 2.5 times the samples at 1.46 times
the cost each — roughly 3.5 times the analytic compute, for a less accurate
image. Partitioning each render's variation into between-pixel structure and
within-pixel noise gives an image signal-to-noise ratio of 22.3 for the
analytic reference against 3.0 for the operator: 4.8 times less signal and 1.55
times more noise. With mean path depth around nine, there is roughly an order of
magnitude less transport to amortise than in fibrous media, and a network query
costs more than a ray–triangle intersection, so the speed premise was weak for
this domain before accuracy was considered.

---

## 7. Discussion

### 7.1 Why partial improvements make the image worse

Correlation between two images partly rewards diffuse overlap. A blurred
operator distributes each facet's light over a wide cone; wherever the true
light should land, some of the blurred light lands too, and the two images
correlate modestly even though no highlight is in the right place. A sharp
operator concentrates the same energy into narrow beams. If the beams are
aimed correctly, agreement rises sharply; if they are aimed wrongly, the
accidental overlap disappears and agreement falls.

Accuracy only helps the image once *both* branch selection and within-facet
placement are good enough that sharp light lands in the right place. Below
that threshold, every step toward sharpness is a step away from agreement.
This is why four separate improvements all moved the image in the wrong
direction, and it predicts that the path from the current operator to a
faithful image does not pass through a sequence of monotone improvements in
rendered quality.

### 7.2 Why the operator is blurred in the first place

Each camera sample draws one exit from the operator's distribution. Under a
broad exit cone, samples leaving one facet fan across many parts of the
lighting rig, and their average approaches the mean radiance of the rig — the
single condition under which a transparent object looks featureless. The
learned operator thereby converts a structured environment into an effectively
constant one, which accounts for the milkiness, the absent fire and the absent
scintillation from one cause. This mechanism explains the *blur*; it does not
explain the *misplacement*, which survives once the blur is removed.

### 7.3 Where the limit lies

Data volume, capacity and training distribution are excluded; the network
already exceeds local interpolation; and a substantially larger coordinate head
closes under a fifth of the gap to a parameterless lookup. Together these point
to the representation. The branch boundary is high-frequency — entry states 4.7
degrees apart routinely leave through different facets — and a small MLP
reading raw coordinates is spectrally biased toward smooth functions [4].

The project contains its own evidence for this. The legacy multiple-scattering
network, `Model_M`, uses a Fourier input encoding, and its documentation records
that encoding plus width reduced its median relative error on this stone from
0.555 to 0.047. The boundary operator reads raw inputs. The reason `Model_M`
was restricted to a small, unencoded configuration at render time — a Dr.Jit
wrapper that unrolls every weight per lane — does not apply to the operator,
which is evaluated as batched PyTorch. Input encoding is therefore the most
strongly indicated next step, supported from three independent directions.

### 7.4 What the smooth representation does offer

A continuous exit density admits next-event estimation, connecting the exit
directly to a light source; a delta refraction does not. This is the one
capability the learned operator has that analytic transport lacks, and it
bears directly on the noise half of the signal-to-noise deficit. It was not
exploited here.

### 7.5 The pivot from the RDM

The RDM was not abandoned but diagnosed, generalised and then measured as a
baseline. The diagnosis — that it drops position — was recorded before the
pivot; the generalisation replaces angular binning with continuous conditioning
and uses the stone's own facets as the only discrete variable; and the
conditioning ablation confirms on the network itself that the discarded
variable is the more important one. The RDM remains the right model where its
assumption holds: media whose aggregation scale lies below the pixel.

---

## 8. Limitations

**Reference convergence.** The analytic reference is used as ground truth and
its seed-split self-agreement as the noise floor, but its convergence with
sample count has not been demonstrated.

**Sample and density consistency.** No test verifies that the operator's
sampler and its density function describe the same distribution. The RDM
variance result depends on that agreement; the matching means in Section 6.9
are consistent with it but only to about 1 percent sensitivity.

**One view and one lighting rig.** Primary comparisons use a single azimuth at
512 × 512 and a single studio rig. Behaviour under held-out viewpoints and
illumination is untested at full resolution.

**Stone pixels only.** All agreement metrics are masked to primary stone hits.
Caustics — light passing through the stone onto the surroundings — fall outside
the mask and were not evaluated.

**Estimator.** Rendering is BSDF-only, with no next-event estimation or multiple
importance sampling, and transport is lossless: no absorption or body colour is
modelled.

**Simplified geometry.** The meshes have 62 to 112 triangles. Real brilliant
cuts carry 57 or more facets with finer structure; the operator's categorical
head is sized to the mesh and would have to scale with it.

**Single training seed.** Every ablation arm uses seed 23. Effects reported are
large relative to plausible seed variance, but no training-seed error bars were
measured.

**Two-variable comparisons.** The pear comparison in Section 6.8 changes both
the head type and the sample count (128 against 256 effective). The effect is
far larger than either would plausibly produce, but a plain-clone pear arm would
isolate it.

**Capacity range.** Only widths 64 and 256 were compared; intermediate
capacities are unexplored.

**Encoding untested.** The most strongly indicated improvement, Fourier input
encoding, was not attempted.

**Cross-cut transfer.** A round-trained operator re-pointed at step-cut
geometry was built and verified but never rendered to completion, because
repeated runs were terminated by memory pressure. Transfer in the pear
direction is blocked by the facet head's size (62 against 64 triangles).

**Cost measurements.** Timings were taken under concurrent load and are
indicative rather than controlled benchmarks.

---

## 9. Future work

1. **Fourier or hash input encoding** of the operator's inputs, motivated by
   Section 7.3, evaluated with the same operator- and image-level metrics. Either
   outcome is informative: success supplies a fix, and failure strengthens the
   case that smooth conditional density models are the wrong class of
   approximator for this transport.
2. **Next-event estimation** through the continuous exit density.
3. **Rendering-aligned training**, weighting the loss by path throughput so that
   the training objective matches the quantity the image integrates.
4. **A table-operator ablation**, rendering from the RDM alone, to complete the
   representation comparison between analytic, tabulated and neural transport.
5. **Held-out viewpoints and lighting**, and evaluation of caustics outside the
   stone mask.

---

## 10. Conclusion

Aggregate neural appearance models do not transfer to resolved specular media,
and the reason can be characterised precisely rather than asserted. The source
method's RDM conditions on direction where a faceted stone's appearance depends
on position; generalising it to a position-conditioned boundary operator yields
a network that learns substantially more than any lookup, yet cannot reproduce
the image. Data, capacity, training distribution, density sharpness, density
family and geometry are each excluded. What remains is a two-part failure —
branch selection and within-facet placement — whose parts interact: a sharper
operator placing light wrongly is further from the truth than a blurred one.
Operator accuracy, therefore, does not transfer to the image until it is good
enough everywhere at once. For media whose fine structure is the appearance,
the question is not whether a network can learn the transport approximately,
but whether it can learn it precisely enough for that precision to be visible.

---

## References

*To be verified against the originals before submission.*

1. G. Soh and Z. Montazeri. Neural Appearance Model for Cloth Rendering.
   *Computer Graphics Forum* 43(4), 2024. arXiv:2311.04061.
2. D. Vicini, V. Koltun and W. Jakob. A Learned Shape-Adaptive Subsurface
   Scattering Model. *ACM Transactions on Graphics* 38(4), 2019.
3. C. M. Bishop. Mixture Density Networks. Technical report, Aston University,
   1994.
4. M. Tancik et al. Fourier Features Let Networks Learn High Frequency Functions
   in Low Dimensional Domains. *NeurIPS*, 2020.
5. W. Jakob et al. Mitsuba 3 renderer, 2022. https://mitsuba-renderer.org
6. A. Wilkie et al. Hero Wavelength Spectral Sampling. *Computer Graphics Forum*
   33(4), 2014.

---

## Appendix A. Provenance

| result | source |
| --- | --- |
| data ladder | `checkpoints/boundary_scale_v1/ladder_summary.json` |
| mixture render, round | `renders/neural_gia_00/evaluation.json` |
| deterministic decode | `renders/neural_gia_01/evaluation.json` |
| behaviour clone render | `renders/neural_gia_clone_01/evaluation.json` |
| oracle facet render | `renders/neural_gia_clone_00/evaluation.json` |
| pear mixture render | `renders/neural_pear_00/evaluation.json` |
| pear deep clone render | `renders/pear_deep/evaluation.json` |
| operator branch accuracy | `checkpoints/camera_model/test_metrics.json`, `checkpoints/clone_model/metrics.json` |
| nearest-neighbour oracle | `checkpoints/camera_model/oracle.json` via `oracle_boundary.py` |
| conditioning ablation | `checkpoints/camera_model/conditioning_ablation.json` via `ablate_conditioning.py` |
| decoder sweep | `checkpoints/camera_model/decoder_ablation.json` via `improve_decoder.py` |
| within-facet error, RDM tables, RDM variance | analysis scripts not yet committed to the repository |
| cost and signal-to-noise | `renders/flash_hires_*`, `renders/flash_128spp` (earlier runs) |
| figures | `plot_results.py` → `renders/figures/` |

Training pool: 1,048,576 records from 65,536 entry states. Held-out pool:
262,144 records from 16,384 entry states. Round geometry `round_diamond_gia`,
64 triangles, internal depth cap 128.

## Appendix B. Corrections

Claims revised during the project are recorded with their replacements in
`CORRECTIONS-2026-09-18.md`. The most consequential was the separation of
*spread* from *error* (Section 6.5): an earlier account treated the operator's
blur as the single cause of the failure, which three experiments subsequently
showed to be incomplete.
