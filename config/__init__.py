MITSUBA_PATH = './dependencies/mitsuba3/build/python'
# Spectral, so that wavelength-dependent refraction (dispersion) can produce
# a diamond's fire. Note there is no `llvm_spectral`: the LLVM spectral
# variant Mitsuba ships is `llvm_ad_spectral`. Run
#   python -c "import mitsuba as mi; print(mi.variants())"
# to see what this build actually offers.
VARIANT = 'llvm_ad_spectral'
#VARIANT = 'llvm_ad_rgb'      # achromatic; no fire is possible in RGB
#VARIANT = 'cuda_ad_spectral'
DEVICE = 'cuda' if VARIANT.startswith('cuda') else 'cpu'

# Aliases
variant = VARIANT
device = DEVICE

import sys
sys.path.append(MITSUBA_PATH)

import glob
import os

# Dr.Jit's LLVM backend needs to find the LLVM shared library, and on some
# installations it cannot do so unaided. This used to be a single hardcoded
# Linux path ('/usr/lib/llvm-14/lib/libLLVM.so'), which meant that on Windows
# every script here died with
#     ImportError: the LLVM backend is inactive because the LLVM shared
#     library ("LLVM-C.dll") could not be found!
# and had to be launched through a wrapper that imported Mitsuba first.
#
# Three rules, in order of precedence:
#   1. An explicit DRJIT_LIBLLVM_PATH already in the environment always wins,
#      so a user can point at a specific build without editing this file.
#   2. Otherwise try the platform's usual locations and take the first that
#      exists on disk.
#   3. If none exist, leave the variable *unset*. Setting it to a path that
#      does not exist is strictly worse than leaving it alone, because Dr.Jit
#      then stops searching and reports a missing file instead of falling back
#      to its own lookup.
_LLVM_CANDIDATES = {
    'win32': [
        r'C:\Program Files\LLVM\bin\LLVM-C.dll',
        r'C:\Program Files (x86)\LLVM\bin\LLVM-C.dll',
    ],
    # llvm-14 first, deliberately. It is the version this project was
    # originally pinned to and the one every working run under WSL has used,
    # and Dr.Jit's LLVM backend is version-sensitive in ways that only show up
    # for particular kernels: the Windows LLVM on this machine loads and
    # renders fine, then fails to JIT a larger histogram kernel with
    # "Failed to materialize symbols". Picking the newest available version
    # would silently move a working setup onto an untested one, so a known-good
    # version wins and anything else is only a fallback.
    'linux': (['/usr/lib/llvm-14/lib/libLLVM.so'] +
              sorted(glob.glob('/usr/lib/llvm-*/lib/libLLVM.so'), reverse=True) +
              ['/usr/lib/x86_64-linux-gnu/libLLVM.so']),
    'darwin': [
        '/opt/homebrew/opt/llvm/lib/libLLVM.dylib',
        '/usr/local/opt/llvm/lib/libLLVM.dylib',
    ],
}

DRJIT_LIBLLVM_PATH = os.environ.get('DRJIT_LIBLLVM_PATH') or next(
    (p for p in _LLVM_CANDIDATES.get(sys.platform, []) if os.path.exists(p)),
    None)

if DRJIT_LIBLLVM_PATH:
    os.environ['DRJIT_LIBLLVM_PATH'] = DRJIT_LIBLLVM_PATH