# From Radiance Distribution Maps to a Learned Boundary Operator

*How the method in this project evolved, why each step was taken, and what it
drew on.*

Sally Mohlakola — honours research, methodology note, 26 September 2026

This document explains, in plain language, the three stages the method went
through:

1. **Radiance distribution maps (RDMs)**, adapted from a neural cloth-rendering
   paper;
2. **a LEAN-inspired "sparkle" extension** that tried to recover detail the RDM
   averaged away;
3. **a learned boundary operator**, which replaced the RDM once measurements
   showed what it was missing.

Every transition is backed by something in the repository: a measurement, a
code comment written at the time, or a design document. Section 9 lists every
paper, book and technique the codebase draws on, and where each one is used.

---

## 1. The problem in one paragraph

When light enters a diamond, it rarely comes straight back out. It refracts
(bends) at the surface, then bounces around inside — mostly by *total internal
reflection*, where light hitting a surface at a shallow enough angle cannot
escape and reflects perfectly instead — and eventually leaves through one of
the facets. In this project's stones that journey averages about nine
bounces. Everything that makes a diamond look like a diamond comes from this
internal journey: **brilliance** (bright white return), **fire** (flashes of
colour, because different wavelengths bend by different amounts) and
**scintillation** (sparkle as the stone or viewer moves). Simulating the
journey exactly is accurate but expensive, so the question behind the whole
project is: *can a learned model replace that internal journey?*

---

## 2. Stage 1 — Radiance distribution maps

### 2.1 The idea, borrowed from cloth rendering

Soh and Montazeri [1] rendered cloth by learning how light scatters inside a
yarn. A yarn contains hundreds of tiny fibres, and simulating light through all
of them is expensive, so they measured it once and stored the result in a
**radiance distribution map (RDM)**.

An RDM is essentially a **lookup table**. You fire a very large number of
virtual rays at the object, and for each one you record the direction it
arrived from and the direction it left in. The table then answers one question:

> *For light arriving from this direction, how much leaves in each possible
> direction?*

Directions are grouped into bins (like the cells of a spreadsheet), so the
table is a *histogram* over pairs of directions. Small neural networks are then
trained to reproduce the table smoothly, and those networks are used at render
time instead of simulating the fibres.

This works well for cloth because a yarn is far smaller than a pixel: the
light enters and leaves within a tiny region, so *where* it enters barely
matters — only the directions do.

### 2.2 How it was built here

The adaptation to diamond lives in `utils/rdm.py`, `gather_rdm.py`,
`neural/base_model.py`, `bsdf/neural_bsdf.py` and `eval.py`.

- **Gathering.** `utils/rdm.py` fires rays at the stone from outside, traces
  them through the real internal journey, and bins the outcomes into a
  four-dimensional table indexed by incoming and outgoing direction (polar and
  azimuthal angle for each). `gather_rdm.py` defaults to 18 × 36 bins per
  direction.
- **Three components**, following the source paper's decomposition:
  - **R** — light reflected straight off the surface;
  - **T** — the fraction of light that transmits (enters) rather than reflects;
  - **M** — the *multiply scattered* light: paths that bounced inside more than
    *k* times (with *k* = 2 by default, so depth three and beyond).
- **Networks**, as in the source paper:
  - `Model_T`, a tiny 3-7-7-1 network predicting the transmitted fraction from
    the incoming direction (the paper's §5.2 architecture);
  - `Model_M`, a small network (by default 6-21-21-21-3, with PReLU
    activations and an exponential output) mapping an incoming and outgoing
    direction to an RGB value for the multiply scattered light.
- **The R component was handled analytically, not learned.** Measured on one
  gather, 100 % of R's energy fell into a single outgoing bin at the mirror
  direction. A polished facet is a perfect mirror for that part of the light,
  so fitting a histogram or network to it would approximate a spike with a
  coarse box. The source paper reached the same conclusion for its R
  component (its §4.2).
- **A hybrid material**, `NeuralDiamond` in `bsdf/neural_bsdf.py`, combined an
  exact analytic surface (Fresnel reflection and refraction) with the learned
  `Model_M` lobe, and used the RDM itself to decide which directions to sample.

### 2.3 What the RDM stage taught

Several problems surfaced, and each one pointed toward the eventual redesign.

**The learned light never actually entered the stone.** An early version of
the material treated every camera ray as reflecting off the outside, so no ray
ever crossed into the diamond and there were no internal paths to render at all
(recorded in `bsdf/neural_bsdf.py`, "Route A"). The fix — explicitly refracting
rays into the stone and only then handing over to the learned term — reproduced
analytic short-path transport to within 0.02 % on energy and peak brightness
(`integrators/depth_aware.py`).

**The RDM is not really a material.** A material, in rendering terms, describes
what happens at *one* point on a surface. The RDM describes what happens to
light across the *whole stone*: it was measured by firing rays at the complete
object and recording where they escaped. The project's own integrator notes
this directly:

> *"the RDM is not a material BSDF at all — it is a boundary operator for the
> whole stone, measured by firing rays at it from outside and recording where
> they escape. Applying it at every internal interaction composes that aggregate
> with itself and queries it far outside the domain it was measured on."*
> — `integrators/depth_aware.py`

That sentence, written during the RDM stage, is the seed of Stage 3.

**Averaging erased the detail.** Each RDM bin stores the *mean* brightness of
the paths that landed in it. When the spread around that mean was measured, it
turned out to be enormous: the standard deviation exceeded the mean in 99.8 %
of well-populated bins, with a median ratio of 2.53 (`bsdf/neural_bsdf.py`). In
other words, each stored value was the centre of a distribution several times
wider than itself. Replacing that distribution by its average is exactly why
the learned light rendered as a flat, even fill instead of sharp flashes.

**The networks that could represent the RDM could not be rendered.** A larger
`Model_M` with a Fourier input encoding fitted the RDM far better — median
relative error 0.047 against 0.555 — but the rendering wrapper
(`neural/drjit_wrapper.py`) evaluates every network weight separately for every
ray in flight and runs out of memory above roughly 5,000 weights. The better
network had to be dropped (`neural/base_model.py`).

---

## 3. Stage 2 — LEAN-inspired sparkle

### 3.1 The idea

LEAN mapping [2] solved a similar averaging problem for bumpy, shiny surfaces.
When a detailed surface is viewed from far away, many tiny bumps fall inside one
pixel, and simply averaging their orientations makes the surface look too
smooth. LEAN's insight was to store **two** numbers instead of one: the
*average* and the *average of the squares* (the "second moment"). From those
two, you can recover how *spread out* the original values were — the
**variance** — and use it to restore the lost roughness.

The same idea applies to the RDM. If each bin stores both the mean and the
second moment of the brightness of its paths, the renderer knows not only how
bright a bin is on average but how *uneven* it is — and diamond sparkle is
exactly that unevenness.

### 3.2 What was built

- `utils/rdm.py` records the second moment (`var_m`) alongside the mean for the
  M component, and `gather_rdm.py` saves both, plus a per-bin path count.
- `eval.py` converts these into a *relative spread* *r* for each bin: the
  standard deviation divided by the mean. Bins with fewer than eight paths are
  given zero spread, because one path reports zero variance and that is an
  absence of evidence, not evidence of smoothness.
- `bsdf/neural_bsdf.py` applies a **sparse random gain**. At each shading point
  the light is switched *on* with probability *p* = 1 / (1 + *r*²), and scaled
  up by 1/*p* when it is on; otherwise it is off. On average the gain equals
  exactly 1, so the overall brightness is unchanged — only the *pattern* changes.
  At the median spread, 13.5 % of points light up at 7.4 times their average
  brightness: sparse bright points, which is what sparkle physically looks like.
- The on/off decision is tied to the **surface position**, through a spatial
  hash, rather than drawn fresh for every sample. If every sample made its own
  random choice, averaging many samples would smooth the pattern straight back
  to the flat mean. Tying it to position makes the pattern stay put, an idea
  borrowed from stochastic glint rendering [3].
- A floor on *p* caps the gain at 50×, because the largest measured spreads
  (*r* up to 15.6) would otherwise demand gains near 250 and create isolated
  "fireflies" out of sparsely sampled bins.

### 3.3 Why it was not enough

The sparkle extension did what it was designed to do: it restored the *amount*
of variation without changing average brightness. But it could not put the
sparkle in the right *places*. Where the bright points appear is decided by a
hash of surface position — a deterministic but essentially arbitrary pattern —
so it is by construction unrelated to where the real stone flashes.

The project's design document recorded the limitation at the time:

> *"Sparkle uses a position hash and Bernoulli-like sparse gain … First and
> second moments do not uniquely specify a distribution. Matching brightness
> statistics does not establish spatial fidelity."*
> — `NEURAL_DIAMOND_NEXT.md`

And the sparkle code itself names the missing ingredient:

> *"the shading becomes position-dependent, so this smuggles in a positional
> input — using the measured variance to decide how much per-position variation
> is justified, rather than learning it."* — `bsdf/neural_bsdf.py`

Both observations point at the same variable: **position**. The sparkle
extension added position artificially, through a hash; the next stage made it a
genuine input.

---

## 4. The diagnosis that motivated the pivot

The design document stated the central problem plainly:

> *"The histogram drops entry and exit positions."* … *"Treat the existing
> directional RDM as a coarse proposal/prior; it is not sufficient by itself to
> locate facet detail."* — `NEURAL_DIAMOND_NEXT.md`

**Why position matters for a diamond when it did not for cloth.** In a yarn,
light enters and leaves within a region far smaller than a pixel. In a
brilliant-cut diamond, light typically enters through the top (the *crown*) and
leaves through the bottom (the *pavilion*), and *which facet it leaves through*
is what determines the image. A table indexed only by direction cannot say which
facet that is.

This was later measured rather than argued. Lookup tables built from real
transport were asked to predict the exit facet of held-out paths:

| information used | exit facet predicted correctly |
| --- | --- |
| nothing (always guess the most common facet) | 4.65 % |
| **entry direction only — what an RDM uses** | **6.82 %** |
| which facet the light entered, a stand-in for position | 19.18 % |

Direction alone is barely better than guessing; position carries roughly three
times as much information. **A finer RDM does not help:** raising the table from
32 to 2,048 direction bins only moved accuracy from 6.8 % to 9.8 %. The problem
was never resolution — it was the choice of what to index by.

The same result appears on the final neural network (Section 5): given only the
incoming direction, as an RDM is, it scores 11.7 %; given position and surface
normal as well, 40.4 %.

---

## 5. Stage 3 — The learned boundary operator

### 5.1 The idea

Instead of a table of directions, train a small neural network to answer the
full question an RDM leaves out:

> *Light enters the stone **here**, travelling in **this direction**, at **this
> wavelength**. Does it come out — and if so, **through which facet**, **where
> on that facet**, and **in what direction**?*

It is called a **boundary operator** because it maps a state on the stone's
boundary (the entry) directly to another state on the boundary (the exit),
skipping everything in between. It is the RDM generalised along the axis the
RDM threw away: position. One network query replaces the entire internal
journey.

### 5.2 How it works

- **Inputs** — ten numbers: entry position (3), incoming direction (3), surface
  normal at the entry point (3), and wavelength (1).
- **Outputs** — three predictions made in sequence:
  1. *Does it escape at all?* (a yes/no probability);
  2. *Which triangle of the stone does it leave through?* (a probability for
     each of the stone's triangles, 64 for the round cut);
  3. *Where on that triangle, and in what direction?* (four continuous numbers).
- **No binning.** Unlike the RDM, nothing is chopped into direction bins. The
  only "category" is the exit facet, and that comes from the stone's actual
  geometry rather than from a chosen resolution. In short: *stop binning the
  sphere; index the geometry and be continuous inside it.*
- **Clever coordinates.** The exit point is described by where it sits on its
  triangle and the exit direction by how far it tilts from the triangle's
  surface normal, both written in a form that can take any real value. That
  lets a standard bell-curve (Gaussian) model them, with a mathematical
  correction (a Jacobian) so the probabilities stay valid.
- **The network** — a small multilayer perceptron: two hidden layers of 64
  units with SiLU activations, then three output "heads", about 18,800 weights
  in total. Two variants were added later: a **behaviour clone** that predicts a
  single best exit instead of a probability distribution, and a **deep clone**
  with a four-layer head for the exit coordinates.

### 5.3 Where the training data come from

`gather_boundary.py` generates training data by doing the expensive thing
honestly: it fires rays at the stone, traces the true internal journey with
exact Fresnel physics at every bounce, and records entry and exit. Each record
is labelled *escaped*, *truncated* (still bouncing after 128 interactions) or
*invalid* (a numerical inconsistency), so the network is never trained to
invent an exit for a path that had none.

An important refinement: entry rays are sampled **the way the rendering camera
sees the stone**, not uniformly from all directions. Under uniform sampling,
fewer than 3 % of training rays arrived from within the camera's view, so the
network was trained mostly on situations the renderer never asked about.
Matching the camera's distribution improved held-out accuracy and corrected a
+15.8 % brightness error to −1.1 %. This is the machine-learning problem of
*covariate shift* [4]: training and testing on different input distributions.

### 5.4 How it is used when rendering

`render_boundary.py` handles everything outside the stone with ordinary path
tracing. When a camera ray hits the diamond and refracts in, the renderer asks
the network where the light comes out, then continues tracing from that exit
point. Queries are grouped in batches of 512, and a predicted exit that would
immediately re-enter the stone is discarded rather than patched.

### 5.5 A clean break in the code

The boundary operator shares **no code** with the RDM system. Checked against
the actual imports, none of the operator's files — the model, data handling,
gathering, training and rendering scripts — imports anything from
`utils/rdm.py`, `gather_rdm.py`, `neural/base_model.py`, `bsdf/rdm_sampler.py`
or `bsdf/neural_bsdf.py`. The only connection is optional: `render_boundary.py`
can load `neural/boundary_prior.py`, a *new* directional table built from the
operator's own data and used purely to guide random sampling. None of the
reported results use it.

So the operator is **conceptually descended** from the RDM but **independently
implemented**. Its results cannot be affected by anything in the older code.

---

## 6. Evaluating the operator, and the RDM as a baseline

The full results are in `FINDINGS.md` and `PAPER_DRAFT.md`; this section
records only what matters for the methodology.

**The measuring apparatus is a large part of the method.** Roughly 1,300 lines
of code implement the operator, and roughly 2,400 exist to test it. Key
elements: comparisons against an exact (analytic) reference rendered under
identical conditions; noise levels *measured* by rendering the same image twice
with different random seeds, rather than assumed; an exact split of every image
into light that went through the network and light that did not, where the
second part acts as a built-in control; and measurements taken directly on the
network, separate from image comparisons.

**The RDM was kept as a baseline and tested in both roles it could still play:**

- *As a guide for random sampling (importance sampling [5, 6]).* Here it cannot
  change what the image converges to — only how noisy it is — and it made the
  image 32 % noisier at equal sample count (60 % at equal time).
- *As extra information about the exit facet.* Combined with the network's own
  prediction, the best it achieved was +0.3 percentage points.

One explanation covers both: the RDM is a summary of the *same* information the
network already receives, with position removed, so it cannot tell the network
anything new.

**A "nearest-neighbour oracle"** — predicting a held-out path's exit by looking
up the most similar training paths [7], with no learning at all — tested
whether the exit is predictable from the inputs at all. The trained network
beat it (41 % against 29 %), showing the network learns more than simple
look-up.

**A conditioning ablation** — retraining with individual inputs removed —
confirmed the pivot on the network itself: direction only, 11.7 %; all inputs,
40.4 %.

---

## 7. What carried forward, and what was dropped

| idea | RDM stage | sparkle stage | boundary operator |
| --- | --- | --- | --- |
| measure true transport by tracing rays, then learn from it | yes | yes | yes |
| exact analytic physics at the entry surface | added via "Route A" | kept | kept |
| condition on incoming direction | yes | yes | yes |
| condition on **position** | no | faked with a hash | **yes, as a real input** |
| condition on wavelength | no | no | yes (measured to contribute little) |
| directions chopped into bins | yes | yes | **no** |
| treat the stone as one "boundary operator" | recognised, not built | — | **built** |
| recover lost spread / sparkle | no — mean only | yes, from second moments | through a sharp, position-aware model |
| training data matched to the camera | no | no | yes |

---

## 8. Describing the methodology in the thesis

A suggested summary, in words that stay close to the evidence:

> *The project began by adapting the radiance-distribution-map method of Soh and
> Montazeri from cloth to diamond. Two problems emerged. The RDM averages the
> light in each direction bin, erasing the sparse flashes that define a
> diamond's appearance; a LEAN-inspired extension restored that variation from
> stored second moments, but could only place it pseudo-randomly. And the RDM
> discards position, which for a faceted stone determines which facet light
> leaves through: direction alone predicts the exit facet barely better than
> chance. Both problems point to the same missing variable. The method was
> therefore generalised into a learned boundary operator that conditions on
> entry position, direction, normal and wavelength and predicts the full exit
> state, replacing the stone's internal transport with a single network query.
> The RDM was retained as a measured baseline.*

Three points worth making explicitly:

- **The pivot was diagnosed, not improvised.** The key limitations were written
  down during the RDM stage (`NEURAL_DIAMOND_NEXT.md`,
  `integrators/depth_aware.py`, `bsdf/neural_bsdf.py`) and later confirmed by
  measurement.
- **Each stage preserved what worked.** Exact physics at the surface, learning
  from traced ground truth, and the idea that light's journey through the whole
  stone is one object were all carried forward.
- **The RDM is not wrong in general.** Its assumption — that position does not
  matter — holds for cloth and fails for diamond. The work identifies where that
  boundary lies.

---

## 9. References and inspirations

Entries marked **(cited in code)** are referenced in the repository's own source
or documents. Entries marked **(identified)** are the published origin of
something the code implements without naming it; consider adding these
citations to the relevant code comments. *Check all bibliographic details
against the original publications before submission.*

### 9.1 The source method and its direct influences

1. **G. Soh and Z. Montazeri.** *Neural Appearance Model for Cloth Rendering.*
   Computer Graphics Forum 43(4), 2024. arXiv:2311.04061. **(cited in code)**
   — The origin of the RDM approach; the T/R/M decomposition; `Model_T`'s
   3-7-7-1 architecture (§5.2); analytic treatment of the R component (§4.2).
   Used in `utils/rdm.py`, `gather_rdm.py`, `neural/base_model.py`,
   `bsdf/neural_bsdf.py`, `eval.py`.
2. **M. Olano and D. Baker.** *LEAN Mapping.* Proceedings of the ACM SIGGRAPH
   Symposium on Interactive 3D Graphics and Games (I3D), 2010. **(identified)**
   — Storing first and second moments to recover variation lost to averaging.
   The inspiration for the sparkle extension (`utils/rdm.py` `var_m`,
   `bsdf/neural_bsdf.py` `_sparkle_gain`). Not yet cited in the code.
3. **W. Jakob, M. Hašan, L.-Q. Yan, J. Lawrence, R. Ramamoorthi and
   S. Marschner.** *Discrete Stochastic Microfacet Models.* ACM Transactions on
   Graphics 33(4), 2014. **(identified)** — Representative of the "stochastic
   glint models" the sparkle code cites generically: sparkle patterns anchored to
   surface position so they stay stable as samples accumulate.

### 9.2 Gemstones, optics and dispersion

4. **Y. Sun, F. D. Fracchia and M. S. Drew.** *Rendering Diamonds.* Proceedings
   of the Western Computer Graphics Symposium (Skigraph), 2000.
   **(cited in code)** — Explicit gemstone transport with Fresnel reflection,
   absorption and dispersion. Listed in `NEURAL_DIAMOND_NEXT.md`.
5. **F. Peter.** *Über Brechungsindizes und Absorptionskonstanten des Diamanten
   zwischen 644 und 226 mμ.* Zeitschrift für Physik 15, 1923.
   **(cited in code)** — The two-term Sellmeier fit for diamond's refractive
   index used in `bsdf/dispersion.py` (verified there against published indices;
   dispersion 0.0444 against the gemmological 0.044).
6. **W. Sellmeier.** Work on the dispersion formula, Annalen der Physik und
   Chemie, 1871. **(identified)** — The functional form of the dispersion model
   in `bsdf/dispersion.py`.
7. **M. Tolkowsky.** *Diamond Design: A Study of the Reflection and Refraction
   of Light in a Diamond.* E. & F. N. Spon, London, 1919. **(identified)** — The
   crown angle 34.5° and pavilion angle 40.75° used for every cut in
   `config/parameters.py` are Tolkowsky's ideal-cut angles.
8. **E. Hecht.** *Optics* (any recent edition). **(identified)** — Standard
   reference for the Fresnel equations, Snell's law and total internal
   reflection used throughout (`gather_boundary.dielectric`,
   `render_boundary.analytic_exit`, `bsdf/dispersive_dielectric.py`).
9. **Gemmological trade diagrams** of the step (emerald) cut and pear cut
   supplied during the project **(project source)** — The reference geometry for
   `ground_truth/step_geometry.py` and `ground_truth/pear_geometry.py`.

### 9.3 Rendering theory and infrastructure

10. **M. Pharr, W. Jakob and G. Humphreys.** *Physically Based Rendering: From
    Theory to Implementation*, 4th edition. MIT Press, 2023. **(cited in code)**
    — Dielectric BSDFs, delta transport, Fresnel sampling and radiance transport
    factors; listed in `NEURAL_DIAMOND_NEXT.md`. Also the standard reference
    for path tracing and next-event estimation.
11. **W. Jakob, S. Speierer, N. Roussel, M. Nimier-David, D. Vicini, T. Zeltner,
    B. Nicolet, M. Crespo, V. Leroy and Z. Zhang.** *Mitsuba 3 renderer*, 2022.
    https://mitsuba-renderer.org **(used throughout)** — The renderer underlying
    every gather and render; version 3.9.1 in this project.
12. **W. Jakob, S. Speierer, N. Roussel and D. Vicini.** *Dr.Jit: A Just-In-Time
    Compiler for Differentiable Rendering.* ACM Transactions on Graphics 41(4),
    2022. **(used)** — The array framework under Mitsuba; used directly in
    `neural/drjit_wrapper.py` and the legacy BSDFs.
13. **A. Wilkie, S. Nawaz, M. Droske, A. Weidlich and J. Hanika.** *Hero
    Wavelength Spectral Sampling.* Computer Graphics Forum 33(4), 2014.
    **(identified)** — One wavelength traced per camera sample, the spectral
    strategy used in `render_boundary.py`.
14. **ITU-R Recommendation BT.709.** **(identified)** — The luminance weights
    0.2126, 0.7152, 0.0722 used in every image comparison (`evaluate_render.py`,
    `compare_figure.py`).

### 9.4 Monte Carlo sampling

15. **T. Müller, B. McWilliams, F. Rousselle, M. Gross and J. Novák.** *Neural
    Importance Sampling.* ACM Transactions on Graphics 38(5), 2019.
    **(cited in code)** — Learned sampling distributions with tractable
    probability densities; listed in `NEURAL_DIAMOND_NEXT.md` and the model for
    using a learned density inside a Monte Carlo estimator.
16. **E. Veach and L. J. Guibas.** *Optimally Combining Sampling Techniques for
    Monte Carlo Rendering.* Proceedings of SIGGRAPH, 1995. **(identified)** —
    Combining sampling strategies with weights; the basis of the mixture weight
    in `neural/boundary_prior.py`.
17. **T. Hesterberg.** *Weighted Average Importance Sampling and Defensive
    Mixture Distributions.* Technometrics 37(2), 1995. **(identified)** —
    Mixing a broad "defensive" distribution into a proposal so weights stay
    bounded; the reason `neural/boundary_prior.py` keeps a uniform floor and
    requires the mixing fraction below 1.
18. **D. J. Wheeler and R. M. Needham.** *TEA, a Tiny Encryption Algorithm.* Fast
    Software Encryption, 1994; and **F. Zafar, M. Olano and A. Curtis.** *GPU
    Random Numbers via the Tiny Encryption Algorithm.* High Performance Graphics,
    2010. **(identified)** — The hash-based random numbers
    (`mi.sample_tea_float32`) that key the sparkle pattern to position.
19. **M. Teschner, B. Heidelberger, M. Müller, D. Pomeranets and M. Gross.**
    *Optimized Spatial Hashing for Collision Detection of Deformable Objects.*
    Vision, Modeling and Visualization, 2003. **(identified)** — The spatial hash
    primes 73856093, 19349663 and 83492791 in the sparkle code come from this
    paper.

### 9.5 The neural network

20. **C. M. Bishop.** *Mixture Density Networks.* Technical report, Aston
    University, 1994. **(identified)** — The operator's main head predicts a
    probability distribution (escape, facet, Gaussian mixture) rather than a
    single value.
21. **M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil,
    N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron and R. Ng.** *Fourier
    Features Let Networks Learn High Frequency Functions in Low Dimensional
    Domains.* NeurIPS, 2020. **(cited in code)** — Cited in
    `neural/base_model.py` for `Model_M`'s input encoding; the leading
    candidate for improving the boundary operator.
22. **N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht,
    Y. Bengio and A. Courville.** *On the Spectral Bias of Neural Networks.* ICML,
    2019. **(identified)** — Why small networks on raw coordinates favour smooth
    functions, which bears directly on the operator's failure to resolve sharp
    facet boundaries.
23. **D. P. Kingma and J. Ba.** *Adam: A Method for Stochastic Optimization.*
    ICLR, 2015. **(identified)** — The optimiser used in every training script.
24. **D. Hendrycks and K. Gimpel.** *Gaussian Error Linear Units (GELUs)*,
    2016; **S. Elfwing, E. Uchibe and K. Doya.** *Sigmoid-Weighted Linear Units
    for Neural Network Function Approximation in Reinforcement Learning.* Neural
    Networks, 2018. **(identified)** — The SiLU activation in the boundary
    operator.
25. **K. He, X. Zhang, S. Ren and J. Sun.** *Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification.* ICCV, 2015.
    **(identified)** — The PReLU activation in the legacy `Model_M` and
    `Model_T`.
26. **R. Girshick.** *Fast R-CNN.* ICCV, 2015. **(identified)** — The smooth-L1
    loss used by the behaviour-cloning heads.
27. **D. A. Pomerleau.** *ALVINN: An Autonomous Land Vehicle in a Neural
    Network.* NeurIPS, 1989. **(identified)** — The origin of behavioural
    cloning: learning to imitate recorded outcomes directly, the basis of the
    clone baselines.

### 9.6 Related work on learned light transport

28. **D. Vicini, V. Koltun and W. Jakob.** *A Learned Shape-Adaptive Subsurface
    Scattering Model.* ACM Transactions on Graphics 38(4), 2019. **(related
    work)** — Learning where light exits a region given where it enters, for
    diffusive materials; the closest prior art to the boundary operator.
29. **S. Kallweit, T. Müller, B. McWilliams, M. Gross and J. Novák.** *Deep
    Scattering: Rendering Atmospheric Clouds with Radiance-Predicting Neural
    Networks.* ACM Transactions on Graphics 36(6), 2017. **(related work)** —
    Neural networks replacing expensive multiple scattering in a volume.

### 9.7 Machine learning and evaluation methodology

30. **T. Cover and P. Hart.** *Nearest Neighbor Pattern Classification.* IEEE
    Transactions on Information Theory 13(1), 1967. **(identified)** — The
    nearest-neighbour oracle in `oracle_boundary.py`.
31. **H. Shimodaira.** *Improving Predictive Inference under Covariate Shift by
    Weighting the Log-Likelihood Function.* Journal of Statistical Planning and
    Inference 90(2), 2000. **(identified)** — The training/testing distribution
    mismatch that motivated camera-matched gathering.
32. **A. Buades, B. Coll and J.-M. Morel.** *A Non-Local Algorithm for Image
    Denoising.* CVPR, 2005. **(identified)** — The algorithm behind OpenCV's
    `fastNlMeansDenoisingColored`, used for presentation-only images in
    `denoise_chroma.py`.
33. **Euler's polyhedron formula** (*V* − *E* + *F* = 2). **(identified)** — The
    closed-surface check in `ground_truth/pear_validation.py`.

### 9.8 Software

PyTorch (network training and inference), NumPy (data handling), Matplotlib
(figures), OpenCV (presentation denoising), FFmpeg (animation encoding), and
Mitsuba 3 / Dr.Jit (rendering).
