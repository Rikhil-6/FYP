#!/bin/bash
# Rebuild every paper figure from the summaries in data/ (and the TikZ diagrams if pdflatex is installed).
set -e
cd "$(dirname "$0")"
python stats.py              # data/v2_stats.json from ../../../results/fmow_v2/run_outputs
python fig_results.py        # fig_drift, fig_accuracy, fig_efficiency_*, fig_cloud
python fig_reliance.py       # fig_shift, fig_maps, fig_learning, fig_router
python method_figures.py     # fig_gate_curves, ../equations/*.png
python fig_explanations.py   # fig_explain_goodbad, fig_explain_update, ../saliency_examples/*.png
python fig_cloud_examples.py data/cloud_examples.npz ../figures   # Appendix C.5 cloud versions of Figures 10 and 12
if command -v pdflatex >/dev/null; then
  for t in overview pipelines; do
    (cd ../tikz && pdflatex -interaction=batchmode $t.tex >/dev/null && pdftoppm -png -r 300 -singlefile $t.pdf ../figures/fig_$t)
    python -c "from style import FIGURES, TITLES, add_title; add_title(FIGURES / 'fig_$t.png', TITLES['fig_$t'])"
  done
fi
