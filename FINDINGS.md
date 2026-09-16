# Findings

## Question

Do aggregate neural appearance models transfer from fibrous media, where the
represented appearance is genuinely smooth at the aggregation scale, to resolved
specular media, where the fine structure *is* the appearance?

Answer: no. Three assumptions of the source method fail to transfer, and the
dominant one is measurable.

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

Width 256 with 8 components reaches validation joint NLL 6.1701 at epoch 40 and
regresses to 6.2437 by epoch 60. Roughly 0.40 nats over width 64, then
overfitting.

### The model genuinely learned

Held-out test set, original pilot to converged model:

| metric | pilot | converged |
| --- | --- | --- |
| joint NLL | 11.415 | 6.605 |
| exit-facet top-1 | 0.094 | 0.383 |
| predicted escape rate | 0.955 | 0.9975 (measured 0.9970) |

Unconditional top-1 baseline is 0.030.

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

This is the single clearest statement of the result. The operator does not
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

## Mechanism

The broadening that erases flashes and the broadening that raises variance are
one defect, seen in the mean and in the variance.

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

## Not established

- **Coverage is not excluded.** The conditioning is at least 5D and 65,536 entry
  states is roughly nine samples per dimension. Whether this is a representational
  limit or extrapolation beyond the training manifold is untested.
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
| signal/noise partition | `renders/flash_128spp/{analytic,neural}/s*/frames/` |
| component split | same, `components` block |
| scintillation sweep | `renders/flash_scaled_01/flash_sweep.json` |
| cost and noise | `renders/flash_hires_*/s*/render.json` |

Training pool: 1,048,576 records from 65,536 entry states (seed 101). Test pool:
262,144 records from 16,384 entry states (seed 202). Geometry
`round_diamond_gia`, 64 triangles, internal depth cap 128.
