# Saliency-guided adaptation under temporal drift

Final-year project on keeping a vision transformer accurate when the images it
sees change over time. It asks whether *saliency* (which image patches carry the
evidence for a prediction) helps a pretrained model adapt to new acquisition
conditions, beyond what ordinary fine-tuning already does.

The current study uses satellite images from fMoW-RGB and a pretrained
DINOv2-large backbone. An earlier study on CLEAR-100 with DINOv2-small and a
transparent router lives on the `Rik/transparent-router` branch.

## Status

This is research code under active development, and the results are
preliminary: three seeds, one dataset subset, and no held-out replication yet.

What the evidence supports so far:

- Natural fMoW drift is borderline detectable (MMD permutation p = 0.05,
  domain-classifier AUC 0.61) and does not hurt accuracy. The future period
  is easier than the historical one, so the study adds realistic cloud cover
  as the harmful nuisance.
- Plain LoRA is the strongest adaptation baseline: none of the anchored arms
  beat it.
- Anchoring LoRA to saliency maps does **not** beat anchoring to uniform or
  shuffled maps.
- Saliency-driven attention heads do ground their attention on the labelled
  object, and the shuffled-map controls show that this comes from the saliency
  content.

## Research questions

1. Is there temporal drift in fMoW that actually harms a frozen model?
2. If not, can a realistic, calibrated nuisance (cloud cover) stand in for it?
3. Does saliency-guided adaptation help more than matched controls: a uniform
   map, a shuffled map, or a plain LoRA of the same size?
4. Are saliency-guided heads faithful to the evidence they use, and do they
   avoid the clouds?
5. Can an auditable router decide when to trust the updated model?

## Data and splits

The study uses a 7,000-image, country-matched subset of fMoW-RGB. Within each
period, train, validation and test splits are made by location group (country
and UTM zone), so no location is shared between those splits.

| Period | Years | Use |
| --- | --- | --- |
| Historical | 2010-13 | train and validation |
| Adaptation | 2014-15 | train and validation |
| Future | 2016-17 | test only |

Every choice (head checkpoint, cloud strength, anchor weight) is made on a
validation split. Future-period labels are read only to score finished models.

The cloud nuisance is a lobed cumulus layer with sun-consistent shadows. Its
strength is calibrated on historical validation data so that the frozen model
loses 5 to 10 points of macro-F1.

## Arms and studies

- **LoRA arms** adapt the last blocks of the backbone with no anchor (plain)
  or with a feature anchor weighted by a uniform, shuffled, or saliency map.
  The saliency map keeps the top 40% of true-class evidence.
- **Attention heads** put saliency inside the classifier in four ways (guided,
  prior, gated, input), each with a shuffled-map twin.
- **Faithfulness and grounding tests** delete patches by rank, and measure
  attention mass on the object box and the pointing-game score.
- **Parameter-matched LoRA** at ranks 1, 2 and 8 gives the fair comparison for
  the heads.
- **Router** chooses between the historical and updated models and writes an
  audit log for every decision.
- **Screens** include test-time token dropping and frozen-probe checks across
  several backbones and datasets.

## Results so far

Macro-F1 (%) on the 2016-17 test split, mean of three seeds. "Cloud" adds
the calibrated cloud layer to the test images.

| Method | Clean | Cloud |
| --- | --- | --- |
| LoRA rank 8, no anchor | 40.8 | 38.6 |
| LoRA rank 8, uniform anchor | 40.2 | 37.1 |
| LoRA rank 8, shuffled anchor | 40.1 | 37.1 |
| LoRA rank 8, saliency anchor | 40.1 | 37.0 |
| LoRA rank 2 | 38.3 | 35.7 |
| LoRA rank 1 | 36.8 | 34.0 |
| Learned attention head | 40.7 | 37.7 |
| Guided head | 41.2 | 38.8 |
| Input head | 42.2 | 38.6 |
| Gated head | 38.6 | 35.1 |
| Prior head | 37.2 | 34.7 |

Gated and prior heads put roughly ten times the chance level of attention
mass on the object box. They point at it in 0.57 to 0.62 of images, against
0.14 for a plain learned head and 0.30 for the gated head with shuffled maps.
Deleting the top 25% of patches by attention costs 11 to 18 points of
accuracy; deleting a random or bottom 25% costs about 0 to 2.

Treat these as exploratory. Differences of a point or less between arms are
within seed noise.

## Repository layout

```text
src/fyp_drift/
  data/          manifests, ingestion, corruptions, fMoW metadata
  models/        ViT wrapper, saliency routing, siamese pair
  artifacts/     run store, model snapshots, router reports
  system/        hardware report and synthetic smoke check
  training/      classifier, fine-tuning, router, LoRA pilots
  evaluation/    studies, metrics, drift tests, saliency heads, router
  visualization/ drift, router and saliency figures
  workflows/     end-to-end experiment runner
deploy/          Ubuntu setup scripts and logged experiment launchers
docs/            protocols, architecture and server notes
tests/           deterministic protocol and entry-point tests
```

The main study is `src/fyp_drift/evaluation/fmow_dinov2l_study.py`. The heads,
router and rank sweep are `saliency_attention_study.py`,
`saliency_router_study.py` and `lora_rank_sweep.py` in the same folder.

## Getting started

Python 3.11 or 3.12 is required. The project uses [uv][uv] with the locked
environment in `uv.lock`.

On an Ubuntu GPU host:

```bash
git clone https://github.com/Rikhil-6/FYP.git
cd FYP
git switch docs/readme-intro
bash deploy/server.sh setup
bash deploy/server.sh status
```

On Windows, use `.\bootstrap.ps1 -Action Setup` instead.

A CUDA GPU is needed for training. The study was run on a single 15 GB
Tesla T4.

### Running the fMoW study

The defaults point at paths on the original server, so pass `--manifest` and
`--cache` for your own copy of the data.

```bash
export PYTHONPATH=src
python -m fyp_drift.evaluation.fmow_dinov2l_study \
  --output runs/fmow \
  --manifest path/to/manifest_downloaded.csv \
  --cache path/to/fmow_u8_224.npy
```

Stages are `gate`, `calib`, `arms`, `summary` and `unseen_cloud`. Each stage
writes its result file and is skipped if that file exists, so an interrupted
run resumes where it stopped. Add `--limit 48 --epochs 1 --seeds 42` for a
quick execution check.

Follow-up studies read the finished run directory:

```bash
python -m fyp_drift.evaluation.lora_rank_sweep \
  --run runs/fmow --output runs/fmow_rank_sweep --ranks 1 2
```

### Tests

The suite is plain pytest. It is not in the locked dependencies, so install
it first:

```bash
pip install pytest
PYTHONPATH=src python -m pytest -q
```

## Documentation

- `docs/saliency_architecture_followup.md`: the saliency question and the
  literature behind the design choices.
- `docs/fmow_saliency_layer_pilot.md`: the earlier layer-selection pilot.
- `docs/experiment_protocol.md` and `docs/architecture.md`: protocol and
  code structure.
- `docs/server_setup.md`: server provisioning.

## Earlier work

The CLEAR-100 study, with its saliency-guided ViT, paired historical and
updated models and transparent router, is on the `Rik/transparent-router`
branch with its own README and training commands.

## Licence

No licence has been chosen yet. All rights reserved until one is added.

[uv]: https://docs.astral.sh/uv/
