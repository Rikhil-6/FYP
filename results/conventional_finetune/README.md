# Conventional temporal fine-tuning control

This folder records the classification-only control run used in the paper. The
model starts from the historical uniform-pooling DINOv2-small checkpoint,
updates the final three transformer blocks and classifier on the 2010
adaptation stream, selects its checkpoint on the held-out half of 2011, and is
then frozen for evaluation on 2012 to 2014.

The control uses the same chronological windows, realistic adaptation profile,
optimiser rates, five-epoch budget, batch size, and seed as the saliency-guided
experiment. It has no saliency gate, auxiliary saliency loss, or router.

Epoch 2 was selected at 0.9390 macro F1. Later epochs reduced the selection
score even as training loss continued to fall. On the evidence-ranked mixture,
the selected model obtained macro F1 scores of 0.9471, 0.9084, and 0.8155 in
2012, 2013, and 2014. Its 2014 score is 1.20 percentage points below the
always-used saliency-guided model and 0.81 points below the saliency-guided
router from the matched evaluation.

The files are:

- `config.json`: exact run configuration and software versions;
- `best.json`: selected epoch and 2011 selection score;
- `events.jsonl`: batch progress, epoch timing, and selection metrics;
- `future_drift.json`: clean and evidence-ranked mixture metrics for 2012 to
  2014.
- `historical_drift.json`: the same frozen evaluation for the historical
  uniform-pooling checkpoint.

These are single-seed results. They support a controlled comparison within this
prototype and do not establish a dataset-independent advantage.
