# Mini-LeWM

Minimal JEPA-style latent world model on a toy pixel environment called PointWorld.

## Purpose

This repository trains a small latent dynamics model from offline trajectories:

- encoder: pixels -> latent
- predictor: `(z_t, a_t) -> z_hat_{t+1}`
- loss: prediction MSE + SIGReg anti-collapse regularizer

The initial version intentionally excludes EMA, reconstruction loss, and pretrained encoders.

## Setup

```bash
make install
```

## Data Generation

```bash
make data-small
make data-medium
```

Generated datasets are written under `data/raw/` and their manifests under `data/manifests/`.

## Training

```bash
make train
```

You can also override config values directly:

```bash
uv run python scripts/train.py \
  --config config/base.yaml \
  --set loss.lambda_sigreg=0.0 \
  --set experiment.name=pointworld_cnn_mlp_lambda0
```

By default the config uses `experiment.device=auto`, which selects `mps` on Apple Silicon when available and otherwise falls back to `cpu`.

## Evaluation

```bash
RUN_DIR=runs/<timestamped_run_name> make eval-probe
RUN_DIR=runs/<timestamped_run_name> make eval-rollout
RUN_DIR=runs/<timestamped_run_name> make eval-collapse
```

## Run Directories

Training creates timestamped folders under `runs/`:

```text
YYYYMMDD_HHMMSS_<experiment_name>
```

Each run stores resolved config, logs, metrics, checkpoints, figures, and evaluation outputs.

## Tests

```bash
make test
```

## Roadmap

- Add more PointWorld variants and walls/obstacles
- Compare `cnn_mlp` and `cnn_gru` predictors
- Extend evaluation probes and rollout diagnostics
- Add richer ablations over `lambda_sigreg`
