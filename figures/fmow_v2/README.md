# Figures: saliency-guided attention on a frozen ViT (fMoW v2)

| Folder | Contents |
|---|---|
| `figures/` | Every figure (PNG, 300 dpi) |
| `saliency_examples/` | One explanation panel per example image (Section 6.6) |
| `equations/` | Equation images embedded in the paper |
| `plots/` | Code that draws every figure, and the summary data it reads (`plots/data/`) |
| `tikz/` | LaTeX sources of the two schematic diagrams |

Rebuild all figures: `bash plots/make_all.sh` (needs matplotlib, seaborn, scipy; pdflatex for the diagrams).

| Figure | Script | Function |
|---|---|---|
| fig_overview, fig_pipelines | `tikz/overview.tex`, `tikz/pipelines.tex` | – |
| fig_gate_curves | `plots/method_figures.py` | `gate_curves` |
| fig_drift, fig_accuracy, fig_efficiency, fig_cloud | `plots/fig_results.py` | `fig_*` |
| fig_shift, fig_maps, fig_learning, fig_router | `plots/fig_reliance.py` | `fig_*` |
| fig_explain_goodbad, fig_explain_update | `plots/fig_explanations.py` | `figure` |
