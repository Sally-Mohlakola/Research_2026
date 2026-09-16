# Denoising: when to use it and when not to

`denoise_chroma.py` removes coloured speckle from display images. It is a
**presentation tool**. Using it in the wrong place would bias a result in favour
of the neural method, which is the worst direction for a finding you want
believed.

## The rule

| use | allowed |
| --- | --- |
| standalone neural render, for a slide or a look | yes, say so in the caption |
| rotation animation frames | yes, say so |
| **paired neural/analytic comparison figure** | **no** |
| **anything measured: correlation, contrast, peak, flash fraction, energy** | **no** |
| linear EXRs | refused by the tool |

## Why it is safe on the neural render

Each camera sample carries one hero wavelength, which converts to a saturated
colour, so speckle is coloured rather than grey. In the neural render the chroma
channels are almost entirely that speckle, because **the model produces no
dispersion** — the absence of fire is one of the project findings. There is
nothing in colour worth preserving, so chroma tolerates heavy smoothing while
luminance, which carries the facet shading, is treated gently.

Measured on the 1024 spp frame at `luma 3, chroma 18`:

| | chroma spread | luminance contrast | luminance mean |
| --- | --- | --- | --- |
| raw | 0.04499 | 0.4969 | 96.23 |
| denoised | 0.00839 | 0.4996 | 95.86 |

Colour noise down 5.4x; luminance contrast and mean moved under one percent.

## Why it is unsafe on the analytic reference

The analytic render contains **real dispersion**. From `utils/studio_env.py`:

> dispersion spreads a path's wavelengths across a small angular fan, so the
> colours only separate visibly when that fan straddles an edge between a bright
> source and a dark surround

Those coloured flashes are signal, not noise, and chroma denoising cannot tell
them apart from speckle. It would erase genuine fire.

## Why that asymmetry forbids the comparison figure

Applying the same denoiser to both modes does **not** make the comparison fair.
It removes noise from the neural render, which loses nothing, and removes fire
from the analytic render, which loses its defining feature. The gap between them
would shrink for reasons that have nothing to do with the model, flattering the
neural method.

A denoised pair is therefore worse than a noisy pair, even though it looks
better.

## Built-in protections

The tool refuses linear HDR input (`.exr`, `.hdr`, `.pfm`), so the measurement
record cannot be altered by accident — every number in `FINDINGS.md` comes from
EXRs, which this tool cannot touch. It refuses to overwrite its source, and it
writes a `.denoise.json` sidecar beside every output recording the source,
settings and this warning, so a denoised image can always be identified later.

None of that stops you from denoising a PNG and then putting it in a comparison
figure. That part is on you, which is why this file exists.

## Usage

Single image:

```bash
python -B denoise_chroma.py \
  --input renders/frame_1024spp/neural_az000.png \
  --output renders/frame_1024spp/neural_az000_denoised.png
```

A directory of animation frames:

```bash
python -B denoise_chroma.py \
  --input renders/rotation_neural/frames \
  --output renders/rotation_neural/frames_denoised
```

Defaults are `--luma 3 --chroma 18`. Raise `--chroma` freely on the neural
render; raise `--luma` only with care, since it softens facet shading.

## The honest footnote

This works as well as it does because **the colour in the neural render is
entirely noise**. A correct diamond render's chroma is signal. That the tool is
so effective here is itself evidence for the finding, and the caption of any
denoised figure should not pretend otherwise.
