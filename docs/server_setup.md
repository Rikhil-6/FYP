# Ubuntu server setup

The deployment scripts assume an Ubuntu checkout and a mounted data volume. By
default the repository stays under the user's home directory while large files
are kept under `/mnt/sdd/FYP`.

## Initial setup

```bash
git clone https://github.com/Rikhil-6/FYP.git
cd FYP
git switch Rik/transparent-router
bash deploy/remote_setup.sh
```

The setup script installs uv inside the storage directory, creates the locked
environment, and links the following paths:

```text
data   -> /mnt/sdd/FYP/data
runs   -> /mnt/sdd/FYP/runs
tools  -> /mnt/sdd/FYP/tools
.venv  -> /mnt/sdd/FYP/venv
cache  -> /mnt/sdd/FYP/cache
```

Existing directories are not silently replaced. Set `FYP_STORAGE_DIR` and the
individual `FYP_*_DIR` variables when a second checkout needs separate runs or
an environment while sharing the dataset and package cache.

## Checks

```bash
bash deploy/server.sh status
bash deploy/server.sh smoke
```

`status` prints the Git revision, PyTorch build, CUDA visibility, GPU model, CPU
count, and resolved storage links. `smoke` performs a small DINOv2 forward and
backward pass without starting a training run.

Full training requires a visible CUDA GPU unless `FYP_ALLOW_CPU_TRAIN=1` is set
explicitly. The launcher also checks that the installed PyTorch build agrees
with the lock file.

## Logged runs

Data ingestion and training remain separate commands:

```bash
bash deploy/server.sh ingest
bash deploy/launch.sh --detach train --temporal-repeats 1 --epochs 5 --batch-size 8
```

The detached launcher creates a tmux session and writes its terminal log under
`runs/pipelines/<timestamp>/`. Model-specific configurations, epoch events,
metrics, predictions, and checkpoints stay within their own run directories.

Useful tmux commands are:

```bash
tmux ls
tmux attach -t <session-name>
```

Detach with `Ctrl+B`, followed by `D`.

## Updating a private server checkout

The server does not require a GitHub credential. A local machine can create a
Git bundle, transfer it over SSH, and fetch the named branch from that bundle.
Private keys must remain outside the repository and are excluded by `.gitignore`.
