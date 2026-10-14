MITSUBA_PATH = './dependencies/mitsuba3/build/python'
import os as _os
# MI_VARIANT picks the Mitsuba variant per job, e.g. cuda_ad_spectral on a GPU node; the default stays on the CPU
VARIANT = _os.environ.get('MI_VARIANT', 'llvm_ad_spectral')
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


_LLVM_CANDIDATES = {
    'win32': [
        r'C:\Program Files\LLVM\bin\LLVM-C.dll',
        r'C:\Program Files (x86)\LLVM\bin\LLVM-C.dll',
    ],
    
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