#!/usr/bin/env python3
"""
composite.py - combine the two halves of the bounce-count decomposition.

    L  =  L_explicit( depth <= k, traced analytically )
       +  L_rdm     ( depth >  k, learned )

Exact at k = 2 with no double counting, because utils/rdm.py classifies
select_m = escaped & (depth >= 3), so Model_M is trained only on paths longer
than the explicit half traces.

The learned half is isolated by differencing the aggregate render against the
same render with --clamp_value 0, which zeroes Model_M and leaves every
analytic component untouched. Seeds are deterministic (eval.py:523), so both
trace identical paths and the Monte Carlo noise cancels in the difference.

This is an image sum, not a renderer: it cannot handle occlusion or the stone
interreflecting with its surroundings. It is an ablation figure.

usage:
    python composite.py <explicit_dir> <aggregate_on_dir> <aggregate_off_dir> <out_dir>
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `config` first: it resolves DRJIT_LIBLLVM_PATH, and Dr.Jit caches its LLVM
# lookup on first use. Only scalar_rgb is needed here, which does not touch
# LLVM, but keeping the order uniform across entry points means this file
# cannot become the odd one out later.
import config  # noqa: F401
import mitsuba as mi  # noqa: E402

mi.set_variant('scalar_rgb')
from utils.studio_env import display_exposure


def frame(d):
    return np.array(mi.Bitmap(os.path.join(d, 'frames', 'frame_0000.exr')),
                    dtype=np.float32)


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    exp_d, on_d, off_d, out_d = sys.argv[1:5]

    explicit = frame(exp_d)
    residual = np.maximum(frame(on_d) - frame(off_d), 0.0)
    comp = explicit + residual

    os.makedirs(os.path.join(out_d, 'frames'), exist_ok=True)
    exr = os.path.join(out_d, 'frames', 'frame_0000.exr')
    mi.Bitmap(comp).write(exr)

    png = os.path.join(out_d, 'frames', 'frame_0000.png')
    b = mi.Bitmap((comp * display_exposure()).astype(np.float32))
    b.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write(png)

    lum = lambda a: 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    print("explicit  sum %10.2f" % lum(explicit).sum())
    print("residual  sum %10.2f" % lum(residual).sum())
    print("composite sum %10.2f  -> %s" % (lum(comp).sum(), exr))


if __name__ == '__main__':
    main()
