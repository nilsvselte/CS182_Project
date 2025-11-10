# Repository Guide

This document supplements `README.md` with a deeper tour of the codebase and
practical tips for extending it.

## Workflow Overview

1. **Configure** — Choose or author a YAML file under `src/conf/`. Quinine merges
   inheritance chains, validates against `src/schema.py`, and exposes the result
   to `train.py`.
2. **Train** — `python src/train.py --config <yaml>` instantiates the model
   (`src/models.py`), sampler (`src/samplers.py`), and task (`src/tasks.py`),
   then runs the curriculum defined in your config while logging to W&B.
3. **Checkpoint & Evaluate** — Each run produces `<out_dir>/<uuid>/state.pt`
   plus periodic `model_<step>.pt` snapshots. `src/eval.py` reloads them to
   compute metrics or run custom evaluations.

## Architecture at a Glance

### Config Flow

```
+----------------------+     inherit      +-----------------------+
| src/conf/<run>.yaml  | <-------------+  | Fragments (models/*.yaml,
| (experiment recipe)  |               |  | wandb.yaml, base.yaml) |
+----------+-----------+               |  +-----------+-----------+
           | Quinfig parse             |              ^
           v                           |              |
+----------v-----------+   validate    |   schema     |
| argparse + quinine   |--------------->| src/schema.py|---+
+----------+-----------+                +--------------+   |
           |                                            resolved
           v                                             config
+----------v-----------+                                    |
| Namespace args        |-----------------------------------+
| (model/training/etc.) |
+-----------------------+
```

Quinine loads `inherit`ed fragments first, applies the current file’s overrides,
ensures the merged object satisfies `src/schema.py`, and hands the structured
`args` object to `train.py`.

### Training & Evaluation Flow

```
                       +--------------------+
                       | src/train.py       |
                       | (entry point)      |
                       +----------+---------+
                                  |
                                  v
        +-------------------------+---------------------------+
        | Input pipeline                                    |
        |                                                   |
        |  +--------------+    +--------------------+       |
        |  | Curriculum   |--->| samplers.py        |       |
        |  | (n_points,   |    | sample_xs()        |       |
        |  | n_dims)      |    +--------------------+       |
        |  +------+-------+             |                    |
        |         |                     v                    |
        |         |             +--------------+             |
        |         +------------>| tasks.py     |-------------+
        |                       | get_task_*() |
        |                       +------+-------+
        |                              |
        +------------------------------+---------------------+
                                       v
                            +----------+----------+
                            | models.py           |
                            | build_model()       |
                            +----------+----------+
                                       |
                                       v
                          +------------+-------------+
                          | Optimizer / train_step() |
                          +------------+-------------+
                                       |
                           +-----------v------------+
                           | Checkpoints & W&B logs |
                           +-----------+------------+
                                       |
                                       v
                           +-----------+------------+
                           | eval.py / plot_utils   |
                           | (reloads checkpoints)  |
                           +------------------------+
```

Key relationships:

- `curriculum.py` controls how many context points and active dimensions each
  step uses; those values are passed into the samplers and tasks.
- `samplers.py` produces synthetic inputs (`xs`) while `tasks.py` creates the
  corresponding target functions and metrics (`ys`, loss/accuracy hooks).
- `models.py` consumes the merged `model` block and returns a GPT‑2 stack
  compatible with the training loop.
- `train.py` logs to W&B (`wandb.log`) and saves checkpoints that `eval.py` can
  later reload for comparisons or plotting.

## Repository Map

- `environment.yml` — Conda environment spec used in the paper; install this
  first to match library versions.
- `models/` — Optional directory where downloaded checkpoints live (see
  `README.md`).
- `src/`
  - `conf/` — All experiment configurations. See `src/conf/README.md` for
    inheritance details.
  - `models.py` — Model factory and baselines. Currently supports GPT‑2 style
    transformers (the primary in-context learner) plus classical baselines for
    evaluation.
  - `base_models.py` — Small feed-forward networks used by baseline training
    routines.
  - `curriculum.py` — Implements the curriculum scheduler that adjusts the
    number of points and dimensions during training.
  - `samplers.py` — Generates synthetic inputs (`sample_xs`) based on the data
    distribution named in the config (Gaussian by default).
  - `tasks.py` — Defines task families (linear regression, sparse regression,
    etc.), their data generation logic, and loss/metric helpers.
  - `train.py` — Orchestrates parsing, model creation, optimization, logging,
    and checkpointing.
  - `eval.py` — Loads saved runs, regenerates tasks, and reports performance
    curves.
  - `plot_utils.py` / `eval.ipynb` — Visualization helpers for the paper.

## Making Changes

- **New experiment config**
  1. Copy an existing file in `src/conf/`.
  2. Update the `inherit` list to pull in the desired model fragment and W&B
     defaults.
  3. Adjust `model`/`training` sections (tasks, curriculum, batch size, etc.).
  4. Point `out_dir` to a new folder inside `../models/` to avoid clobbering
     older runs.

- **New transformer size**
  1. Add `src/conf/models/<name>.yaml` with the `model:` block (family, layer
     counts, heads, embedding width).
  2. Reference it from your experiment’s `inherit` list.

- **New task type**
  1. Implement a sampler in `src/tasks.py`, returning an object with
     `evaluate()`, `get_training_metric()`, and `get_metric()`.
  2. Register it inside `get_task_sampler` and `TASK_LIST` in `src/schema.py` so
     configs can select it.
  3. Update `src/models.py::get_relevant_baselines` if you want baseline plots.

- **New data distribution**
  1. Add a function inside `src/samplers.py` that returns an object with
     `sample_xs`.
  2. Register it in `get_data_sampler`.
  3. Allow the distribution name in `schema.py`’s `training.data` field.

- **Task-aware channels**
  1. Flip on `training.task_labeling.enabled` for configs that mix task families
     (e.g., linear + quadratic). The runner will increase `model.n_dims` by the
     requested `dimension` and automatically overwrite the new coordinates with a
     deterministic encoding of the task label coming from `tasks.py`.
  2. Optionally pin exact values via `training.task_labeling.manual_map` (each map
     entry should provide a float or list matching `dimension`).
  3. Evaluation uses the saved config to recreate the same label injection, so no
     extra work is needed when running `eval.py`.

- **Alternative model families**
  1. Extend `build_model` in `src/models.py` and add the option to
     `model_schema['family']`.
  2. Ensure the new model exposes the same `forward(xs, ys, inds=None)` signature
     so `train.py` and `eval.py` continue to work.

## Tips

- Every training run writes its resolved configuration to
  `<out_dir>/<uuid>/config.yaml`. Use this to reproduce experiments or seed
  evaluation scripts.
- For quick debugging, pass `--test_run True` (or set `test_run: true` in a
  config) to collapse the curriculum and limit training to 100 steps.
- `wandb.log_every_steps` in the config governs logging cadence; keep it high
  for long runs to save bandwidth.
- When iterating on code, prefer adding unit tests or smoke tests in the `tests/`
  directory (create if needed) so regressions are easy to catch before launching
  lengthy training jobs.

Refer back to this guide whenever you need to locate functionality, wire up new
experiments, or explain the pipeline to collaborators.
