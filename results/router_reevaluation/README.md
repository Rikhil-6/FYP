# Evidence-ranked mixture re-evaluation

The saved historical and updated checkpoints and the saved router policies were
evaluated without retraining. The severity-three condition applies the three
corruptions selected on the 2011 calibration partition: motion blur, sensor
noise, and JPEG compression.

## 2014 macro F1

| Variant | Historical | Updated | Routed | Routed gain |
| --- | ---: | ---: | ---: | ---: |
| Saliency supervised | 79.33 | 82.75 | 82.36 | +3.03 |
| No auxiliary saliency loss | 79.33 | 82.78 | 81.00 | +1.67 |

The updated branches differ by only 0.03 points, while saliency-supervised
routing is 1.36 points higher. This supports a routing effect rather than an
advantage caused by a stronger updated classifier. The saliency-supervised
router records 375 helpful and 117 harmful switches; the matched control records
251 helpful and 95 harmful switches.

Each subdirectory contains the complete JSON result and per-image routing CSV.
The JSON also records 2012 and 2013 results, class metrics, confusion matrices,
timing, and update-trigger diagnostics.
