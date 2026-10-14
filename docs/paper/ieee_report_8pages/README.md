# IEEE research report

`main.tex` is the editable IEEE conference-style, two-column report. `main.pdf`
is its compiled output. The bibliography is embedded in the source.

## Compile

Upload `main.tex` and the complete `figures/` folder to Overleaf, select
`main.tex` as the main document, and compile with XeLaTeX. A local TeX Live or
MiKTeX installation can also run `xelatex main.tex` twice. Tectonic can build it
with `tectonic main.tex` and automatically resolves references.

The report uses the standard IEEEtran class and standard LaTeX packages. It is
formatted as a research report, not as an accepted IEEE publication. Author
affiliation and contact information are omitted because they were not supplied.

## Evidence and scope

- No new model training or rendering experiment was run for this rewrite.
- Validation curves were plotted from existing checkpoint histories. The saved
  logs contain validation measurements, not paired training-loss curves.
- The main image table uses the stored `half_split_comparison` results: 64 spp
  per compared image. Displayed images and facet means use all seeds: 128 spp.
- Geometric facets are coplanar triangle groups; operator labels are triangles.
- The facet-scale step-cut result qualifies the original broad negative claim.
- Render timings are medians of saved per-seed durations, not a new controlled
  benchmark. Hardware and concurrent-load uncertainty are stated in the paper.
- Additional convergence, multi-seed training, lighting, integration and NMAE
  tests are explicitly proposed work, not completed results.
- The original report and experimental files are not modified.

`reported_results.json`, `recorded_timings.json`, and `validation_histories.json`
contain the records used for the new tables and plots. `source_manifest.json`
records local input paths and SHA-256 hashes for traceability. Historical
diagnostics also draw on `docs/PAPER_DRAFT.md`, `docs/FINDINGS.md` and the
previous paper source; they are distinguished from the matched render set.

The source package includes all figure assets referenced by `main.tex`.
