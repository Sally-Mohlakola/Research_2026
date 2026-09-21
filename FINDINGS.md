# Findings

## Question

Do aggregate neural appearance models transfer from fibrous media, where the
represented appearance is genuinely smooth at the aggregation scale, to resolved
specular media, where the fine structure *is* the appearance?

Answer: no. Three assumptions of the source method fail to transfer, and the
dominant one is measurable.

The failure is not a shortage of data, capacity or training coverage — each is
excluded below by measurement. It has two parts. The model picks the wrong exit
facet 59 percent of the time, and even when handed the correct facet it places
the exit 24.66 degrees from the truth. Supplying the branch for free restores
the stone's sparkle almost exactly, contrast 3.734 against a reference 3.744,
while spatial agreement *falls* to 0.041 — so sharpness and correctness are
separate axes, and this document previously conflated them.

A nearest-neighbour lookup over a million training paths scores 0.2869 on the
same branch problem against the trained model's 0.4103. The network already
exceeds what local interpolation can do, which locates the limit in the
representation rather than in the data.

The most useful single result is that **operator accuracy does not transfer to
the image**. Three separate experiments improved the operator and each made the
render agree with the reference less: a deterministic decode, an oracle that
supplies the true branch, and a genuinely better-trained decoder. The two error
sources are not additive, so partial fixes move the image away from the
reference rather than toward it.

## What does not transfer

Source: Soh and Montazeri, *Neural Appearance Model for Cloth Rendering*, CGF
43(4), 2024 (arXiv 2311.04061).

| assumption in the source | value there | value here |
| --- | --- | --- |
| aggregation scale | hundreds of fibers per yarn | 64 facets, each resolved across thousands of pixels |
| conditioning | 4D RDM in angle coordinates only | transport is nonlocal: entry on the crown, exit on the pavilion |
| compression ratio | hundreds of scattering events per query | mean path depth 9.1 |

The second was diagnosed and fixed by this project: the boundary model conditions
on object-space position, facet, direction and wavelength, and predicts an exit
*position* as well as a direction. The first and third were inherited untested.

That diagnosis is now measured rather than argued. Building lookup tables from
real transport and asking each to predict the exit facet on held-out data:

| conditioning | cells | exit-facet top-1 |
| --- | --- | --- |
| trivial, most common facet | — | 0.0465 |
| entry **direction** only, as the RDM does | 32 | 0.0682 |
| entry **facet**, a proxy for position | 64 | 0.1918 |

Direction alone is barely above the trivial baseline; position carries roughly
three times as much. The RDM does not condition on the variable that determines
where light leaves this geometry, which is the whole motivation for the
generalisation.

The third matters independently of whether the model works: with roughly an order
of magnitude less transport to compress, and a network query costing more than a
ray-triangle intersection, the speed premise was weak for this domain before any
accuracy question arose. The measured result is 1.46x *slower*.

## Established

### The model is not data limited

Data ladder, identical architecture and hyperparameters, ~50k gradient steps per
rung, evaluated on one held-out pool of 262,144 records (seed 202). Baselines are
fitted on exactly the records each model trained on.

| entry states | train records | facet NLL | marginal baseline | top-1 |
| --- | --- | --- | --- | --- |
| 256 | 3,259 | 3.960 | 4.031 | 0.048 |
| 1,024 | 13,059 | 3.655 | 3.859 | 0.108 |
| 4,096 | 52,248 | 3.327 | 3.839 | 0.207 |
| 16,384 | 209,121 | 2.874 | 3.834 | 0.332 |
| 65,536 | 836,425 | 2.773 | 3.833 | 0.347 |

Gains flatten above roughly 16k entry states. Unconditional top-1 is 0.030. The
256-entry rung early-stopped at epoch 7 of 3,800, so it was never step starved;
it exhausted distinct entry states.

### The model is not capacity limited

Width 256 with 8 components reaches validation joint NLL 6.1501 at epoch 51 and
regresses to 6.3482 by epoch 80, a drift of 0.198 nats past its minimum. That is
0.33 nats better than width 64, followed by clear overfitting at 4.8 training
records per parameter against 44.6 for width 64.

This comparison carries a caveat. Width 64 reached its best epoch at 78 of 80 and
its validation curve was still falling, so it was still *under*-trained when
compared against a width-256 model that had already turned. The honest statement
is that width 64 is under-trained, width 256 overfits, and the capacity between
18,789 and 175,881 parameters is unexplored.

### The model genuinely learned

Held-out test set, original pilot to converged model:

| metric | pilot | converged |
| --- | --- | --- |
| joint NLL | 11.415 | 6.605 |
| exit-facet top-1 | 0.094 | 0.383 |
| predicted escape rate | 0.955 | 0.9975 (measured 0.9970) |

Unconditional top-1 baseline is 0.030.

### The operator is translucent where the transport is transparent

Measured directly on the operator, with no rendering, lighting or exposure
involved. For 300 held-out entry states, the angular spread of exit directions
about their mean, computed within each individual exit facet:

| | median spread | branches under 2 degrees |
| --- | --- | --- |
| real transport | **0.00 degrees** | 99.3 percent (n=547) |
| neural model | **25.98 degrees** | 0.3 percent (n=2,125) |

Real transport splits an entry into a median of two discrete beams (mean 2.43
distinct exit facets across 16 repeats; only 5.5 percent of entry states use a
single facet), and **each beam is perfectly coherent**, exactly as delta
refraction requires. Measuring spread across all beams together gives 19.78
degrees for real transport against 44.17 for the model, but that total is
branching, not blur. The within-branch figure isolates blur, and there the real
value is zero.

This is the difference between window glass and frosted glass: the same light is
transmitted, the direction information is destroyed. The learned operator is
translucent; the transport it replaces is transparent.

**Spread is not error.** The table above compares the dispersion of exit
directions about their own mean. It does not measure how far the model's exits
are from the *correct* exits, and those are different quantities. Feeding each
head the true exit facet, so that only the within-facet decode is its own:

| head | exit-direction error given the true facet |
| --- | --- |
| behaviour clone | mean **24.66 degrees**, median 18.23, p90 54.48 |
| mixture | mean **32.51 degrees**, median 22.14, p90 79.36 |
| nearest-neighbour lookup, no training | mean **8.87 degrees** |

The clone has 0.00 degrees of within-facet spread and is still 24.66 degrees
wrong. Zero spread is not accuracy. This measurement was absent from earlier
versions of this document, and its absence supported a single-cause reading of
the failure that the next section retracts.

The architecture is not hard-limited here. `log_std` is clamped at -4, permitting
roughly one degree of spread. The model learned 26 degrees instead, because four
Gaussian components must cover a branch structure they cannot otherwise
represent. The width is not a tuning failure; it is what a smooth density does
when asked to be a sum of deltas.

### Dispersion is learned and then destroyed

Diamond's index runs 2.4641 at 400 nm to 2.4062 at 700 nm. Querying a fixed entry
state across that range shifts the model's mean exit direction by a median of
**6.91 degrees**, so the model did learn a wavelength dependence.

That signal is **3.8 times smaller than the model's own 25.98 degree blur**. Every
wavelength's cone overlaps every other wavelength's cone almost entirely, so at
any image point all wavelengths arrive together and recombine to grey. From
`utils/studio_env.py`:

> dispersion spreads a path's wavelengths across a small angular fan, so the
> colours only separate visibly when that fan straddles an edge between a bright
> source and a dark surround

A 26 degree cone does not straddle edges; it averages over many at once.

The information is therefore present in the model and discarded by the
representation, which is a stronger statement than a failure to learn dispersion.

The conditioning ablation below sharpens this further: removing the wavelength
input altogether costs 0.0004 of branch accuracy and 0.08 degrees of exit
angle. The learned dependence is real but worth almost nothing against the
model's own error, so the input is effectively free to discard.

### Blur accounts for the missing sparkle, but not for the wrong image

Every optical signature distinguishing a diamond from ordinary glass requires
angular coherence finer than the measured blur:

| effect | requirement | measured |
| --- | --- | --- |
| fire | wavelengths resolved into different places | 6.91 degree separation inside 25.98 degree blur |
| brilliance | concentrated white return | peak over mean 23.5 against 139.7 |
| scintillation | sharp flashes under motion | contrast 0.799 against 3.806 |

A brilliant cut is an engineering design that presupposes coherent transport.
Supplied with an operator that blurs by 26 degrees, the cut stops functioning and
the facets become decorative geometry rather than optical elements.

**An earlier version of this document claimed blur was the single cause. Three
experiments show it is not.** Each removed or bypassed the blur, and none
recovered the image:

| configuration | within-facet blur | contrast | correlation with reference |
| --- | --- | --- | --- |
| reference | — | 3.744 | floor 0.976 |
| mixture, as trained | 25.98 deg | 0.847 | **0.174** |
| deterministic decode | 8.09 deg | 1.018 | 0.153 |
| behaviour cloning | 0.00 deg | 0.847 | 0.174 at best on operator metrics, worse on the image |
| oracle facet: true branch, learned coordinates | — | **3.734** | **0.041** |

The last row is decisive. Handed the correct branch, the render reproduces the
reference's contrast almost exactly, 3.734 against 3.744, and its peak-over-mean
reaches 114.7 against 121.1. The sparkle returns in full. Correlation
nonetheless *falls* to 0.041, and the operator component to 0.017.

So blur and correctness are separate axes. Blur suppresses the high-dynamic-range
structure, and removing it restores that structure; but the light is then placed
sharply in the wrong locations. Correlation partly rewards diffuse overlap, which
is why the blurred render scores higher than the sharp one. The rendered stone
reads as frosted glass because of the blur, and reads as the *wrong* stone for a
different reason.

### The image did not follow

Paired renders, 728x728, azimuth 0, 128 effective spp, eight seeds per mode,
271,232 primary stone pixels. Noise floors are measured by splitting the seeds in
half, giving two independent estimates of the same image.

| | neural vs analytic | analytic vs itself | ratio |
| --- | --- | --- | --- |
| spatial correlation | 0.098 | 0.955 | — |
| highlight overlap, top 1 percent | 0.185 | 0.780 | 0.24 |
| relative RMSE | 3.922 | 1.156 | 3.39 |
| relative energy error | +0.1584 | -0.00017 | 925 |
| chromaticity error | 0.361 | 0.246 | 1.47 |

Chromaticity sits at 1.47 times its own noise floor; no colour conclusion is
claimed from a single view.

### It is not merely blurred

Blur preserves correlation. Highlight overlap of 0.185 against a 0.780 floor
means the brightest pixels are in different places, not smeared versions of the
same places. The learned operator relocates light rather than softening it.

### The failure is attributable to the learned component

Each camera path is tagged by whether it passed through the stone interior,
giving an exact split of the frame (residual under 1e-7).

| component | share of stone radiance | correlation | energy error |
| --- | --- | --- | --- |
| learned operator | 0.876 | 0.045 | +0.186 |
| first-surface reflection and scene | 0.124 | 0.922 | +0.002 |

The second row is identical physics in both renders and acts as a control: it
rose from 0.40 at 8 spp to 0.922 at 128 spp, as an identical signal must, while
the learned component stayed near zero. Same image, same noise, same pixels, a
20x gap.

Isolating the learned component makes the failure worse than the whole-image
figure suggests, because the whole-image figure is inflated by the 12.4 percent
both modes compute identically.

### No scintillation

| | analytic | neural |
| --- | --- | --- |
| contrast, standard deviation over mean | 3.806 | 0.799 |
| peak over mean | 139.7 | 23.5 |
| flash fraction above 32x median | 0.644 percent | 0.160 percent |

### Apparent detail is speckle, not structure

Neural contrast falls with sampling while analytic contrast holds:

| effective spp | neural | analytic |
| --- | --- | --- |
| 16 | 1.664 | — |
| 64 | 0.972 | 3.869 |
| 128 | 0.799 | 3.806 |
| 256 | 0.683 | — |

Noise-driven contrast washes out with sampling; structure-driven contrast
survives. Low-sample neural renders therefore look more detailed than the
converged result.

Rows at and below 256 spp were measured under a box reconstruction filter and
the 1024 spp row under a gaussian one. A control re-render of the 256 spp case at
512x512 with the gaussian filter gives contrast 0.6895 against 0.683 for the
original, so the filter changes this statistic by about one percent and the trend
is attributable to sample count rather than to reconstruction.

Noise fell by 1.50x between 256 and 1024 spp rather than the 2x that a fourfold
sample increase would give under ideal 1/sqrt(N) convergence. The shortfall is
consistent with a heavy-tailed estimator, where rare high-value samples from exit
directions catching the two intense sparks dominate the variance and average down
slowly. With eight seeds the ratio is uncertain, so this is an observation rather
than a measured convergence exponent.

### It costs more, not less

| | analytic | neural | ratio |
| --- | --- | --- | --- |
| seconds per 8-spp 728x728 render | 424 | 620 | 1.46 |
| median per-pixel relative standard error | 0.222 | 0.348 | 1.57 |

Matching analytic noise requires roughly 2.5x the samples at 1.46x the cost each:
about 3.5x analytic compute for equal quality, before accounting for the fact
that the converged image is the wrong one.

### Structure is relocated into noise

Partitioning the variation in each render at identical settings (728x728, 128
effective spp, eight seeds) into variation *between* pixels, which is image
structure, and variation *within* a pixel, which is Monte Carlo noise:

| | between-pixel (signal) | within-pixel (noise) | image SNR |
| --- | --- | --- | --- |
| analytic | 3.806 | 0.171 | 22.3 |
| neural | 0.799 | 0.264 | 3.0 |

The learned operator delivers 7.4x less usable image at matched sample cost,
decomposing as 4.8x less signal multiplied by 1.55x more noise.

This is the clearest statement of the *blur's* consequence. The operator does not
destroy the environment's angular structure; it relocates that structure out of
image contrast and into per-sample variance. Milkiness and noise are therefore
not two problems but one, which is why they have tracked together across every
sample count measured.

The lighting rig sharpens the effect. It contains two small, very intense
sparks by design. Under a narrow exit cone a facet either sees a spark or does
not, which is a per-pixel fact and produces contrast. Under a broad exit cone
every pixel has some probability of catching one on any sample, which is the
firefly regime: rare extreme samples in every pixel, each carrying a single
saturated wavelength. The sparks introduced to create fire are what the broad
density converts into speckle.

### The operator needs position, and does not need wavelength

Each input group removed from the network by zeroing its slice, so the
architecture, parameter count and optimisation are identical across arms and a
difference is a difference in information rather than capacity. One budget, one
seed, one entry-grouped split; 60 epochs on 838,736 training records from
52,422 entry states, evaluated on 261,447 held-out escaped records.

| arm | exit-facet top-1 | vs full | exit angle given the true facet |
| --- | --- | --- | --- |
| full | **0.4042** | — | 24.68 deg |
| no wavelength | 0.4039 | -0.0004 | 24.76 deg |
| no normal | 0.3662 | -0.0380 | 25.88 deg |
| no position | 0.2732 | -0.1311 | 25.54 deg |
| no direction | 0.2010 | -0.2032 | 29.15 deg |
| **direction only** | **0.1167** | **-0.2875** | 30.41 deg |

**The last row is the direct test of this project's central design decision.**
Given only what the RDM is conditioned on, the network scores 0.1167; given
position and normal as well, it scores 0.4042. Spatial conditioning is worth a
factor of 3.5 on branch selection, measured on the network itself rather than
inferred from lookup tables.

Position and normal are partially redundant, and the ablation quantifies it.
Dropping either alone costs 0.131 and 0.038, but dropping both together with
wavelength costs 0.288 -- far more than the sum. The normal identifies which of
64 facets was entered and position implies the same thing plus where on it, so
each covers for the other. Only the direction-only arm measures the true cost of
removing spatial information, and single-drop arms understate it.

Direction is not unimportant: removing it costs 0.203. The ordering is what
matters. Spatial conditioning costs more than direction, which is the claim the
RDM's angle-only formulation gets backwards for this geometry.

**Wavelength contributes nothing measurable**: -0.0004 on branch selection and
+0.08 degrees on the exit angle. This sharpens the dispersion result above. The
model does learn a wavelength dependence, and that dependence is worth 0.08
degrees against its own 24.68 degree error, so removing the input entirely is
free. Dispersion is not merely destroyed downstream by the blur; it never
contributes to predictive accuracy in the first place.

The exit-angle column moves far less than the facet column, staying between
24.68 and 25.88 degrees for every arm that keeps a spatial input and reaching
only 30.41 degrees for the most crippled one. The within-facet decode is broken
at roughly 25 degrees almost regardless of conditioning, which is independent
support for treating it as a separate problem from branch selection.

Consistency check: the direction-only network reaches 0.1167 against the
direction-only lookup table's 0.0682. The network extracts 1.7 times more from
the same inputs than a table can, matching the pattern in the oracle result.

### The failure has two parts

| component | measured | reference |
| --- | --- | --- |
| branch selection | exit-facet top-1 **0.4084** | 1.0 |
| within-facet decode | **24.66 degrees** error given the true facet | 8.87 for a nearest-neighbour lookup |

Neither alone explains the image. Supplying the branch for free leaves
correlation at 0.041; driving within-facet spread to zero by behaviour cloning
leaves the image no better. Every earlier single-cause account in this document
should be read as superseded by this table.

### The decode improves with depth, and then saturates

The within-facet half had never been attacked. Two architectural suspects: the
regressor is two layers and 5,444 parameters, and it reads a trunk trained
jointly for escape, facet and coordinates, where the 64-way cross-entropy
dominates the gradient. One budget, one seed, 30 epochs, held-out pool, exit
angle given the true facet.

| arm | exit angle | vs baseline | median | facet top-1 | parameters |
| --- | --- | --- | --- | --- | --- |
| separate trunk + depth 4 | **21.61 deg** | **-2.91** | 14.22 | 0.3941 | 28,741 |
| depth 4 | 22.04 deg | -2.48 | 14.94 | 0.3887 | 23,877 |
| separate trunk | 24.09 deg | -0.43 | 17.57 | 0.3955 | 20,421 |
| coordinate loss x5 | 24.25 deg | -0.27 | 17.74 | 0.3869 | 15,557 |
| baseline, as shipped | 24.52 deg | — | 18.19 | 0.3935 | 15,557 |

Facet accuracy is unchanged across every arm, so nothing was bought by damaging
branch selection. The median improves more than the mean -- 18.19 to 14.22, a
22 percent cut -- while the p90 barely moves, 54.51 to 51.40, so the typical
case improves and the tail does not.

**Depth is the active ingredient and the second hypothesis was wrong.** Depth
alone captures 2.48 of the 2.91; a separate trunk alone manages 0.43, and
weighting the coordinate loss up does essentially nothing. The shared trunk was
not being starved by the facet cross-entropy. The head was simply too shallow.

**And it saturates well short.** The gap from the shipped head to a
nearest-neighbour lookup is 15.65 degrees; doubling the parameters and adding a
separate trunk closes 2.91 of it, 18.6 percent. A method with no parameters
still beats the best architecture tried here by 2.4 times. If the head were
merely too small, capacity would have closed more. This is independent support
for the encoding argument: the limit is not how much the head can compute but
what it can see.

### Operator accuracy does not transfer to the image

The clearest result in this project, and it took three experiments to see.

| experiment | operator-level change | image correlation |
| --- | --- | --- |
| deterministic decode | blur 25.98 to 8.09 deg | 0.174 -> 0.153 |
| oracle facet | branch error removed entirely | -> 0.041 |
| improved decoder, pear cut | exit angle 37.16 to 26.25 deg, facet 0.309 to 0.337 | 0.136 -> **0.054** |

Every time the operator is measurably improved, the render agrees with the
reference *less*.

The third case is the sharpest, because nothing about it is artificial: a
genuinely better-trained operator, evaluated against the same analytic
reference at the same four seeds. Pear cut, 128 effective spp, 90,607 stone
pixels.

| | shipped mixture | improved clone_deep | reference |
| --- | --- | --- | --- |
| correlation | 0.1359 | **0.0541** | floor 0.8987 |
| operator component | 0.1225 | 0.0355 | — |
| highlight overlap | 0.0066 | 0.0022 | floor 0.5949 |
| relative RMSE | 1.834 | 3.795 | floor 0.684 |
| relative energy error | +17.3% | **+47.8%** | — |
| contrast | 0.886 | **2.315** | 1.476 |
| peak over mean | 20.0 | 82.3 | 64.4 |
| flash fraction | 0.00065 | 0.0077 | 0.0015 |

**The improved model overshoots.** Earlier models were too flat; this one is
too sparkly. Contrast passes the reference and keeps going, 2.315 against
1.476, peak over mean 82.3 against 64.4, flash fraction five times too high.
Sharpness is therefore not converging on correctness -- it is a free parameter
that happens to be uncorrelated with accuracy.

That overshoot is real structure, not speckle, and the check matters because
this document establishes elsewhere that neural contrast is noise-driven and
washes out with sampling. Contrast against sample count, same stone, same
pixels:

| | 32 spp | 64 spp | 128 spp | change |
| --- | --- | --- | --- | --- |
| improved clone_deep | 2.585 | 2.390 | 2.315 | **-10%** |
| shipped mixture | 1.393 | 1.069 | 0.886 | **-36%** |
| analytic reference | 1.564 | 1.507 | 1.476 | -6% |

The shipped model's contrast was largely an artefact: it falls by more than a
third as samples accumulate. The improved model's falls by a tenth, close to
the analytic's own six percent, so it is converging on genuine structure.

This refines the "structure is relocated into noise" result above. The improved
decoder relocates it **back** -- apparent contrast that was noise becomes
contrast that is structure -- and the image still gets worse, because placement
rather than structure is what is wrong. Recovering the right amount of
structure and recovering the right image are separate achievements, and only
the first has happened.

The difference image shows the two error modes as visually distinct. Missing
light appears in **facet-shaped** regions, which is branch selection failing;
added light appears in **diffuse blobs**, which is the within-facet decode
placing exits where no facet directs them.

The energy result states the problem most starkly. The improved head's escape
prediction is *more* physically correct: measured escape on the pear is 1.0 and
it predicts 1.0, where the mixture under-escaped. Being right about escape made
the image worse, because the extra paths land in the wrong places. The
untouched control sits at 0.913 and 0.916 across the two runs, so the harness
is sound and the whole difference is the operator's.

**What this adds to the two-part diagnosis.** The two errors are not additive
and cannot be fixed independently. A sharp operator placing light incorrectly
is further from the truth than a blurred one, because blur buys accidental
overlap with the right regions. Any partial improvement therefore moves the
image away from the reference until branch selection and the within-facet
decode are *both* close enough for the placement to be right. That is a
stronger claim than "the model needs a better decoder", and three independent
experiments support it.

One caveat on the pear comparison: it is a mixture head against a clone_deep
head at 128 spp versus the original 256, so two variables move at once. The
effect is far too large to be either, but a plain clone arm on the pear would
isolate it.

### Local interpolation is not the ceiling

A nearest-neighbour oracle over the training pool answers a question no
architecture experiment can: is the exit predictable from this conditioning at
all? For each held-out entry state, take the exits of the most similar training
entries. No training, no capacity limit, no extrapolation.

| predictor | exit-facet top-1 |
| --- | --- |
| trivial, most common facet | 0.0465 |
| nearest-neighbour oracle, 1,046,359 training paths | 0.2869 |
| oracle, majority vote over 8 neighbours | 0.3428 |
| trained clone | **0.4103** |

**The trained network beats the oracle by 42 percent.** Local interpolation over
a million paths is not the ceiling, so the model is not merely failing to
interpolate a well-covered manifold. Combined with the exclusion of data volume
and capacity, this points at representation and input encoding.

Why interpolation fails is visible in the neighbourhoods. The true facet is
absent from the eight nearest neighbours 61 percent of the time; the median
nearest-neighbour distance corresponds to about **4.7 degrees** of entry
direction; and eight near-identical entry states span a median of **two distinct
exit facets**, with all eight agreeing for only 18.6 percent of queries. The
branch boundary is high-frequency relative to any achievable sampling density,
and closing that gap by brute force would scale as the tenth power of the
resolution.

One conditional is encouraging: when the oracle gets the facet right, its exit
direction is within **8.87 degrees**. Within a branch the map is smooth and
nearly free. Across branches it is not.

### The RDM does not help in either available role

The source method's radiance distribution map was tested in both roles the
architecture permits.

**As an importance-sampling proposal.** The mixture weight is `p_neural / q`, so
the estimator targets the learned model by construction and the RDM cannot
change accuracy, only variance. Four matched seeds per arm, current checkpoint,
160x160 at 16 spp, 25,600 lit pixels:

| | prior on | prior off |
| --- | --- | --- |
| mean luminance | 0.002596 | 0.002594 |
| per-pixel variance | 4.105e-05 | 3.122e-05 |
| seconds | 29.6 | 24.3 |

**Variance ratio 1.315 at equal samples, 1.603 at equal time.** The means agree
to +0.10 percent, which is the weighting behaving correctly: the same expected
image, a worse estimator. This supersedes an earlier two-seed 64x64 figure of
1.202, measured on a pilot checkpoint that no reported result uses.

**As a branch prior.** Combining the RDM's facet distribution with the model's
posterior, against a clone baseline of 0.4103:

| RDM conditioning | cells | RDM alone | best blend |
| --- | --- | --- | --- |
| entry direction only, as built | 32 | 0.0682 | 0.4094 |
| entry facet | 64 | 0.1918 | 0.4103 |
| entry facet x direction, 8x16 | 8,192 | 0.2594 | **0.4135** |

The ceiling is **+0.003**, and the RDM as currently conditioned makes the model
worse. One explanation covers both roles: the RDM is a marginal of the same
conditioning the network already reads, so it cannot carry information the
network lacks.

**No result in this document uses the RDM.** All evaluated renders record
`rdm_prior: None`, and `render_paired.py` does not expose the option. The RDM is
this project's ancestor and its baseline, not a component of its method.

## Mechanism

The broadening that erases flashes and the broadening that raises variance are
one defect, seen in the mean and in the variance. This section explains that
defect. It is one of the two identified above, and it is not the one that puts
the light in the wrong place.

Each camera sample entering the stone draws an exit triangle from a 64-way
categorical, a component from a Gaussian mixture, then Gaussian samples for exit
position and outgoing direction. Two samples in one pixel may exit at opposite
ends of the stone. Analytic internal transport is also stochastic, but its
Fresnel branches are discrete and few and each branch gives an exact direction,
so outcomes cluster.

Stated in the project's own terms, from `utils/studio_env.py`:

> A constant environment is the one environment in which a transparent object is
> guaranteed to look featureless. Under a constant environment every escape
> direction carries the same value, so all of those paths return the same number
> and the facets render as flat unshaded grey.

Because the learned exit density is broad, samples leaving a single facet fan out
across a wide cone and land on many parts of the lighting rig. Averaged, that
facet returns the mean radiance over the cone. The operator therefore converts a
deliberately structured environment into an effectively constant one, which is
the single condition under which a transparent object must look featureless. That
accounts for the milkiness, the absent fire (dispersion is only visible where a
fan straddles a bright/dark edge, and averaging removes the edges) and the absent
scintillation, from one cause.

Seen in image space, this is the relocation of structure into noise measured
above: a broad exit cone lowers between-pixel contrast and raises within-pixel
variance by the same act.

Likelihood does not detect this. Transformed-coordinate likelihood and image
radiance are connected through the *joint* exit position and direction; the
mixture scores well on the marginals while blurring the joint structure that
determines where light lands.

**This mechanism is necessary but not sufficient.** It explains the milkiness,
the absent fire and the absent scintillation, all of which follow from a broad
exit cone averaging a structured environment into a constant one. It does not
explain why the image remains wrong once the cone is narrowed. The oracle-facet
render has a narrow cone and correct branches and still correlates at 0.041,
so a second mechanism is at work: the learned exit lands 24.66 degrees from the
truth even within the correct facet, which relocates light without broadening
it. Milkiness and misplacement are distinct failures with distinct causes, and
only the first is explained here.

## Not established

- **Coverage is now partly excluded.** The nearest-neighbour oracle answers the
  interpolation half: with 1,046,359 training paths it reaches 0.2869 against the
  trained model's 0.4103, so the model is not failing to interpolate a covered
  manifold, and more data of this kind would not close the gap. What remains
  untested is whether a *different* conditioning, rather than more of this one,
  would make the exit predictable.
- **The claim's scope is undefended.** Without a mixture-component sweep,
  "smooth densities cannot" narrows to "four Gaussians cannot".
- **One view, one lighting rig.** One azimuth at high resolution, fifteen at
  96x96, a single studio rig, no held-out illumination.
- **Sample and PDF agreement is untested.** Required by the project's own
  validation gate 1, and load bearing for the RDM prior's `p_neural/q` weight.
- **The analytic reference is not shown converged.** It is used as reference and
  its seed-split self-agreement as the noise floor.
- **Cost figures assume a BSDF-only estimator.** No next-event estimation. NEE is
  inapplicable to the analytic path's delta exit but *is* applicable to the
  neural operator's continuous density, which is one genuine advantage of a smooth
  representation and should be discussed rather than omitted.
- **Transport is lossless and monochromatic.** No absorption or colour is learned;
  each sample carries one wavelength.

## Provenance

| result | source |
| --- | --- |
| data ladder | `checkpoints/boundary_scale_v1/ladder_summary.json` |
| converged model | `checkpoints/boundary_scale_v1/final_w064c4/` |
| paired render comparison | `renders/flash_128spp/evaluation.json` |
| operator blur, dispersion | measured on `checkpoints/camera_pool/test.npz` with `checkpoints/camera_model` |
| signal/noise partition | `renders/flash_128spp/{analytic,neural}/s*/frames/` |
| component split | same, `components` block |
| scintillation sweep | `renders/flash_scaled_01/flash_sweep.json` |
| cost and noise | `renders/flash_hires_*/s*/render.json` |
| deterministic decode | `renders/det_exit/evaluation.json` |
| oracle facet | `renders/oracle_facet/evaluation.json` |
| within-facet decode error | `oracle_boundary.py` conditioning path, `checkpoints/camera_pool/test.npz` |
| nearest-neighbour oracle | `checkpoints/camera_model/oracle.json` |
| RDM as branch prior | tables built from `checkpoints/camera_pool/train.npz`, evaluated on `test.npz` |
| RDM variance ratio | four matched seeds per arm, `checkpoints/step_transfer`, 160x160 at 16 spp |
| cross-cut ablation | `renders/pear_comparison.png`, `checkpoints/pear_model/test_metrics.json` |
| conditioning ablation | `checkpoints/camera_model/conditioning_ablation.json`, via `ablate_conditioning.py` |
| decoder sweep | `checkpoints/camera_model/decoder_ablation.json`, via `improve_decoder.py` |
| improved pear render | `renders/pear_deep/evaluation.json`, model `checkpoints/pear_deep` |
| shipped pear render | `renders/neural_pear_00/`, re-evaluated at four matched seeds |

Claims revised on 18 September 2026 are listed with their replacements in
`CORRECTIONS-2026-09-18.md`. Three results above were produced by analysis
scripts not yet committed; see `CODE_FREEZE.md` item 1.

Training pool: 1,048,576 records from 65,536 entry states (seed 101). Test pool:
262,144 records from 16,384 entry states (seed 202). Geometry
`round_diamond_gia`, 64 triangles, internal depth cap 128.
