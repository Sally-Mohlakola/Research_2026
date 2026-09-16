"""Chroma-targeted denoising for display images. PRESENTATION ONLY.

Read DENOISING.md before using this. The short version: it is safe on the
neural render because that render has no real colour structure, and unsafe on
the analytic reference because that one does. Never use it on a paired
comparison figure, and never on anything you measure.

Each camera sample carries one hero wavelength, which converts to a saturated
colour, so low-sample renders show coloured speckle rather than grey grain. In
the neural render the chroma channels are almost entirely that speckle: the
model produces no dispersion, so there is nothing in colour worth preserving.
Luminance, which carries all the facet shading, is therefore treated gently
while chroma is smoothed hard.

Two safety properties are built in rather than left to memory:

1. Only 8-bit display images are accepted. Linear EXRs are refused outright, so
   the measurement record cannot be altered by this tool even by mistake.
2. Every output is written beside a `.denoise.json` sidecar recording the source,
   the settings and a warning, so a denoised image can always be identified as
   such later.
"""
import argparse
import json
from pathlib import Path

import cv2

DISPLAY_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
LINEAR_SUFFIXES = {'.exr', '.hdr', '.pfm'}

WARNING = (
    'PRESENTATION ONLY. Do not use on a paired comparison figure: the analytic '
    'reference contains real dispersion and would be damaged, while the neural '
    'render would not, biasing the comparison in favour of the neural method. '
    'Do not use on anything measured. See DENOISING.md.'
)


def denoise(path, destination, luma, chroma, template, search):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Could not read %s as an 8-bit image' % path)
    result = cv2.fastNlMeansDenoisingColored(image, None, luma, chroma,
                                             template, search)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), result):
        raise RuntimeError('Could not write %s' % destination)
    destination.with_suffix('.denoise.json').write_text(json.dumps(dict(
        source=str(path), luma_strength=luma, chroma_strength=chroma,
        template_window=template, search_window=search,
        method='cv2.fastNlMeansDenoisingColored', warning=WARNING),
        indent=2), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True,
                        help='Display image, or a directory of them')
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination file, or directory when input is one')
    parser.add_argument('--luma', type=int, default=3,
                        help='Luminance denoising strength. Keep low: luminance '
                             'carries the facet shading you want to preserve.')
    parser.add_argument('--chroma', type=int, default=18,
                        help='Chroma denoising strength. Can be high for the '
                             'neural render, whose colour is almost entirely noise.')
    parser.add_argument('--template_window', type=int, default=7)
    parser.add_argument('--search_window', type=int, default=21)
    args = parser.parse_args()

    if args.luma < 0 or args.chroma < 0:
        parser.error('Denoising strengths must not be negative')
    if args.template_window % 2 == 0 or args.search_window % 2 == 0:
        parser.error('Window sizes must be odd')
    if args.input.resolve() == args.output.resolve():
        parser.error('Refusing to overwrite the source; denoising is not reversible')

    if args.input.is_dir():
        sources = sorted(p for p in args.input.iterdir()
                         if p.suffix.lower() in DISPLAY_SUFFIXES)
        if not sources:
            parser.error('No display images found in %s' % args.input)
        if args.output.is_file():
            parser.error('--output must be a directory when --input is one')
        targets = [args.output/p.name for p in sources]
    else:
        if args.input.suffix.lower() in LINEAR_SUFFIXES:
            parser.error('Refusing linear HDR input (%s). This tool is for display '
                         'images only; the EXR measurement record must stay '
                         'untouched. See DENOISING.md.' % args.input.suffix)
        if args.input.suffix.lower() not in DISPLAY_SUFFIXES:
            parser.error('Unsupported input type %s' % args.input.suffix)
        sources, targets = [args.input], [args.output]

    print(WARNING + '\n')
    for source, target in zip(sources, targets):
        print('  %s -> %s' % (source, denoise(source, target, args.luma,
                                              args.chroma, args.template_window,
                                              args.search_window)))
    print('\n%d image(s) denoised (luma %d, chroma %d)'
          % (len(sources), args.luma, args.chroma))


if __name__ == '__main__':
    main()
