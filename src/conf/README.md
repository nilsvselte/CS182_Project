# Configuration Guide

This directory contains every YAML file that `train.py` can load with `--config`.  
Each file ultimately resolves to the schema in `src/schema.py`, which defines four
sections: `model`, `training`, `wandb`, and `out_dir` (plus the optional flag
`test_run`). Files often compose these sections via `inherit` chains handled by
Quinine (`train.py` lines 167‑186).

## Layout

- `models/`: Parametric snippets that only define the `model` block (family,
  depth, hidden size, etc.). Example: `models/standard.yaml` sets a GPT‑2
  backbone with 256‑wide embeddings and 12 layers.
- `wandb.yaml`: Shared logging defaults (entity, project, logging cadence).
- `base.yaml`: A long‑running Gaussian training recipe with a curriculum that
  grows dimensionality and sequence length.
- Task configs (e.g., `linear_regression.yaml`, `decision_tree.yaml`) inherit
  from `base.yaml` or the `models/*.yaml` files, then override task‑specific
  fields such as `training.task`, `training.task_kwargs`, or `curriculum`.
- `toy.yaml`: Minimal example used in the README for a fast sanity check run.

## Inheritance Mechanics

```yaml
inherit:
  - models/standard.yaml
  - wandb.yaml
```

The list is evaluated top‑to‑bottom. Later files override earlier ones, and the
current file overrides everything it inherits. Each referenced path is relative
to `src/conf/`.

## Common Fields

- `model`: Must match `model_schema` (family, `n_dims`, `n_positions`, `n_embd`,
  `n_layer`, `n_head`). These values are passed directly into
  `models.build_model`.
- `training`: Controls the data sampler, task sampler, optimizer hyper-parameters,
  checkpoint cadence, and curriculum schedule. See `src/schema.py` lines 37-58
  for allowed tasks and defaults.
- `training.task_labeling` *(optional)*: Enables a lightweight channel that tags
  every context point with the task family (e.g., linear vs. quadratic). Set
  `enabled: true`, choose a `dimension` (defaults to 1), and optionally provide
  `manual_map` entries such as `linear: [0.0]`, `quadratic: [1.0]`. When enabled,
  the model input dimension automatically increases by `dimension`, and the extra
  coordinates are populated with the task label.
- `wandb`: Mirrors `wandb.init` arguments in `src/train.py` lines 147-155,
  allowing per-run names or notes.
- `out_dir`: Base directory where checkpoints live; `train.py` appends a UUID to
  keep runs isolated.

## Creating a New Config

1. Decide which model backbone you need. Either inherit one of the existing
   `models/*.yaml` files or create a new fragment there.
2. Include `wandb.yaml` (or your own logging settings) near the top of the
   `inherit` list.
3. Override `model.n_dims`/`n_positions` if your task dictates a particular input
  size.
4. Fill out the `training` block: choose `task`, optionally add `task_kwargs`,
   and configure the curriculum schedule.
5. Point `out_dir` to a descriptive folder inside `../models/`.

When you run `python src/train.py --config src/conf/<file>.yaml`, Quinine expands
the inheritance tree, validates the result against the schema, and writes the
fully resolved configuration to `<out_dir>/<run_uuid>/config.yaml` so evaluation
can reproduce the setup.
