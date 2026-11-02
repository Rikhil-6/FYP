# Image datasets for temporal drift correction

The study needs a chronological train → adaptation → future-test split **and** a measured performance loss. A significant MMD alone is insufficient. Each candidate below needs a clean-period reference, a null split-half control, MMD permutation test, domain-classifier AUC, and frozen historical-model accuracy before a correction claim is made. Patient/sequence/site IDs must stay disjoint where applicable; class composition and acquisition metadata need reporting by period.

| Candidate | Time and acquisition axes | Access and current project status | Main limitation / next test |
| --- | --- | --- | --- |
| [iWildCam-WILDS](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/iwildcam_dataset.py) | Individual image datetime **and** camera location are exposed in WILDS metadata. | Public WILDS download, 11.96 GB compressed; server folder currently empty. | Best time × camera candidate. Custom chronological split within locations, plus held-out cameras, needed because official split is by camera. Check enough species support in each window before training. |
| [FMoW-WILDS](https://wilds.stanford.edu/datasets/) | Satellite imagery has time period and region metadata. | Public WILDS loader; server folder currently empty. | Strong real chronological imaging case, but large archive and geography/time confounding. Verify year and class overlap and download feasibility. |
| [PadChest](https://arxiv.org/abs/1901.07441) | Chest radiographs from 2009–2017, with acquisition information. | Public dataset described by authors; not downloaded. | Medical time × acquisition setting, but only one hospital. Use patient-disjoint chronology; verify dates, device metadata and access before promising a run. Report label-quality issues because many labels are automatic. |
| [Wild-Time Yearbook](https://wild-time.github.io/) | Image year 1930–2013. | Already present on server and trialled on GPU. | Fast chronology check, but low-resolution binary portrait task and no hospital/camera axis. Our current sampled split has distribution change but no natural performance harm. |

[Camelyon17-WILDS](https://wilds.stanford.edu/datasets/) is a useful hospital acquisition control, but its public WILDS setup exposes hospitals rather than a defensible chronological split. It must not be presented as temporal drift.

## Recommended next dataset

iWildCam is the most direct test of the requested joint **time × camera** question because its loader explicitly parses datetime and location. Before model fitting, inspect counts by year, camera and species; freeze a split and keep all images from each burst/sequence together. Then run frozen-model MMD, AUC and performance loss. If the chronological loss is absent, move to a different window under a predeclared selection rule rather than optimizing the test window for a saliency gain.
