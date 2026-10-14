# Paper: IEEE two-column, 8 pages

`main.tex` is a self-contained IEEE conference paper (`IEEEtran`, bibliography
inline). The figures it uses are in `figures/`, downscaled copies of the
repository's renders and plots.

## Build

- **Overleaf:** upload this folder (or zip it), keep `main.tex` as the main
  document, compile with pdfLaTeX. `IEEEtran`, `tikz` and `booktabs` are built in.
- **Locally:** `pdflatex main.tex` twice (needs a TeX distribution such as MiKTeX
  or TeX Live; none is installed on this laptop).

## Before submitting

1. Replace the author placeholders: `[Department]`, `[University]`,
   `[City, Country]`, `[email]`.
2. Check the page count after compiling. If it runs over 8 pages, the easiest
   cuts are the dispersion paragraph (Sec. VI-F) and the entry-sampling half of
   Fig. 6; if under, the convergence plot (`figures/plots/convergence_operators.png`)
   fits naturally in Sec. VI-C.
3. Verify every reference against the original publication (volume, pages,
   venue); several were identified from what the code implements.
4. Correlations in the tables are split-half values, comparable with the noise
   floors; the labels inside the figures show full-frame values, as their
   captions say. Keep that distinction if you edit numbers.

## Where the numbers come from

| claim | source |
| --- | --- |
| RDM tables, baselines, oracle, ablation, decoder sweep, elimination | `docs/PAPER_DRAFT.md` and its provenance appendix |
| three-cut decoder comparison (Table II) | `renders/{gia_mixture,gia_clone,gia_clone_deep,pear_learned,pear_clone_deep,step_learned,step_clone_deep}/evaluation.json` |
| validation facet accuracy | each model's `metrics.json` |
| Fourier features | `checkpoint_history/fourier_decoder/fourier_decoder.json`, `renders_history/gia_mixture_fourier/evaluation.json` |
| coverage blind spot | `figures/plot_aim_offset.py` output; `README.md` results table |
| transfer and network-only split | `renders/gia_varied_from_pear*/COMMANDS.md` |
| dispersion | `checkpoint_history/dispersion_pear/`, `renders/pear_dispersion_pose09/fire.json` |
| parameter counts | counted from the round-cut checkpoints |
