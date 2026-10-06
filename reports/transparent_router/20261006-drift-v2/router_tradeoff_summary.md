# Router trade-off summary

This note accompanies the revised Figure 3 in `output/docx/updated_ViT_router_mock_paper_2_router_plot.docx`. The figure foregrounds the project's deployment question: whether selective routing improves on continued use of the same frozen historical ViT as drift accumulates. The unconditional updated model remains reported below as a secondary performance ceiling.

## What the evidence shows

| Comparison | Result | Interpretation |
|---|---:|---|
| Saliency-routed vs historical ViT | Better in all 11 conditions | Routing improves on keeping the historical model alone. The largest gains are **+3.45 percentage points** for motion blur and **+2.03 points** for the realistic mixture. |
| Saliency-routed vs always using the updated ViT | Lower by 0.06–0.51 points in all 11 conditions | This secondary comparator bounds the cost of selective deployment. The router retains 87.1% of the updated model's gain under motion blur and 91.9% under the realistic mixture. |
| Saliency-routed vs matched router without auxiliary saliency losses | **+0.90 points** for motion blur and **+0.84 points** for the realistic mixture | The auxiliary saliency objectives help most under the two difficult conditions. Effects on clean and mild shifts are small or negative, so the gain is source dependent. |

The routing audit also exposes the cost of selective updating. For motion blur, viewpoint shift, and the realistic mixture, the router records 357/90/47, 88/56/17, and 261/95/30 helpful, harmful, and missed switches respectively. It therefore makes many useful corrections, but its gate still permits harmful switches and rejects some changes that would have helped.

The supported conclusion is that the router improves the intended frozen-model deployment baseline while making every switch inspectable. It does not outperform unconditional use of the updated ViT on aggregate macro F1, so that result remains disclosed in the text and table rather than being used as the main visual comparison. Its practical value is selective deployment, auditability, and the option to constrain update coverage; it should not be described as the accuracy winner against every comparator.

## Document and figure integrity

- Revised document: `output/docx/updated_ViT_router_mock_paper_2_router_plot.docx`
- Revised figure: `reports/transparent_router/20261006-drift-v2/drift_v2_router_tradeoff.png`
- The supplied 23-page layout was preserved.
- Only the Figure 3 image, its caption, and its adjacent interpretation paragraph were replaced in the supplied document package.
- All other embedded media retained their original hashes and dimensions.

## Reproduce the figure and document patch

Run through uv from the repository root:

```bash
tools/uv run --with seaborn --with pandas --with pillow scripts/remap_router_tradeoff_plot.py
tools/uv run --with pillow scripts/update_supplied_docx_router_plot.py \
  --input path/to/updated_ViT_router_mock_paper_2.docx \
  --plot reports/transparent_router/20261006-drift-v2/drift_v2_router_tradeoff.png \
  --output output/docx/updated_ViT_router_mock_paper_2_router_plot.docx
```

The plotted values come from `drift_v2_2014_metrics.csv`; the scripts do not recompute model results.
