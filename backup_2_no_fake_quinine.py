import os
import sys
import types
from types import SimpleNamespace

import torch

sys.modules["schema"] = types.SimpleNamespace(schema=None)

sys.modules["wandb"] = types.SimpleNamespace(
    init=lambda *args, **kwargs: None,
    log=lambda *args, **kwargs: None,
    finish=lambda *args, **kwargs: None,
)

sys.path.append("src")

from models import build_model
from train import train


def make_args_for_experiment(
    out_dir,
    task_name,
    task_kwargs,
    data_name,
    data_kwargs,
    model_config,
    training_configs,
):
    n_dims = training_configs["n_dims"]
    n_points = training_configs["n_points"]

    model_conf = SimpleNamespace(
        family="gpt2",
        n_dims=n_dims,
        n_positions=n_points,
        n_embd=model_config["n_embd"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        transformer_seed=None,
    )

    # ----- TASK CONFIG -----
    # Pull out weight_type so it's not passed twice to get_task_sampler
    weight_type = task_kwargs.get("weight_type", "standard")
    base_task_kwargs = {k: v for k, v in task_kwargs.items() if k != "weight_type"}

    task_dict = {
        "task_0": {
            "task": task_name,
            **base_task_kwargs,   # e.g. {"degree": 1}
        }
    }

    # ----- DATA CONFIG -----
    # Ensure there is always a "type" key (standard vs skewed)
    data_type = data_kwargs.get("type", "standard")
    base_data_kwargs = {k: v for k, v in data_kwargs.items() if k != "type"}

    data_dict = {
        "data_0": {
            "data": data_name,    # e.g. "gaussian"
            "type": data_type,    # "standard" by default
            **base_data_kwargs,
        }
    }

    training_conf = SimpleNamespace(
        learning_rate=training_configs["learning_rate"],
        batch_size=training_configs["batch_size"],
        n_points=n_points,
        train_steps=training_configs["train_steps"],

        weight_type=weight_type,      # only here, not in task_kwargs
        task_kwargs=task_dict,
        task_schedule="random",

        data_kwargs=data_dict,
        data_split="steps_based",
        data_schedule="random",

        prompt_type="standard",
        prompt_kwargs={},

        val_type="none",
        validation_kwargs={},
        num_tasks=1,

        save_every_steps=training_configs.get(
            "save_every_steps", training_configs["train_steps"] + 1
        ),
        keep_every_steps=-1,
    )

    wandb_conf = SimpleNamespace(
        log_every_steps=10_000,
        project="",
        entity="",
        notes="",
        name="toy-sweep",
    )

    args = SimpleNamespace(
        out_dir=out_dir,
        test_run=True,
        model=model_conf,
        training=training_conf,
        wandb=wandb_conf,
    )

    return args

# -------------------------------------------------------------------
# 3) Hyperparameter sweep using THEIR train()
# -------------------------------------------------------------------
def hyperparameter_sweep(
    tasks_to_test=None,
    data_distributions=None,
    model_configs=None,
    training_configs=None,
    save_dir="./sweep_results",
):
    """
    Run a hyperparameter sweep.
    """

    # ----- Defaults -----
    if tasks_to_test is None:
        tasks_to_test = [
            ("hermite_regression", {"degree": 1, "weight_type": "standard"}),
            ("hermite_regression", {"degree": 2, "weight_type": "standard"}),
        ]

    if data_distributions is None:
        data_distributions = [
            ("gaussian", {}),
        ]

    if model_configs is None:
        model_configs = [
            {"n_embd": 64, "n_layer": 2, "n_head": 2},
            {"n_embd": 128, "n_layer": 4, "n_head": 4},
        ]

    if training_configs is None:
        training_configs = {
            "n_points": 100,
            "batch_size": 64,
            "train_steps": 50,
            "learning_rate": 1e-4,
            "n_dims": 1,
        }

    os.makedirs(save_dir, exist_ok=True)
    results = []

    total_experiments = len(tasks_to_test) * len(data_distributions) * len(model_configs)
    experiment_num = 0

    print("Using CPU (their train() is CPU-based)")

    for task_name, task_kwargs in tasks_to_test:
        for data_name, data_kwargs in data_distributions:
            for model_config in model_configs:
                experiment_num += 1

                print(f"\n{'='*80}")
                print(f"Experiment {experiment_num}/{total_experiments}")
                print(f"Task: {task_name} {task_kwargs}")
                print(f"Data: {data_name} {data_kwargs}")
                print(f"Model: {model_config}")
                print(f"{'='*80}\n")

                # Per-experiment output dir
                exp_dir = os.path.join(
                    save_dir,
                    f"exp_{experiment_num}_{task_name}_deg{task_kwargs.get('degree', 'NA')}"
                    f"_data_{data_name}_L{model_config['n_layer']}_H{model_config['n_head']}",
                )
                os.makedirs(exp_dir, exist_ok=True)

                # Build args and model
                args = make_args_for_experiment(
                    out_dir=exp_dir,
                    task_name=task_name,
                    task_kwargs=task_kwargs,
                    data_name=data_name,
                    data_kwargs=data_kwargs,
                    model_config=model_config,
                    training_configs=training_configs,
                )

                model = build_model(args.model)
                model.cpu()
                model.train()

                mse_path = os.path.join(exp_dir, "mse_log.txt")

                try:
                    with open(mse_path, "w") as f:
                        train(model, args, f)

                    losses = []
                    with open(mse_path, "r") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) == 2:
                                _, loss_val = parts
                                try:
                                    losses.append(float(loss_val))
                                except ValueError:
                                    pass

                    if len(losses) == 0:
                        final_loss = float("nan")
                        avg_last_100 = float("nan")
                    else:
                        final_loss = losses[-1]
                        tail = losses[-100:] if len(losses) >= 100 else losses
                        avg_last_100 = sum(tail) / len(tail)

                    print(f"Final loss: {final_loss:.6f}")
                    print(f"Avg last 100: {avg_last_100:.6f}")

                    results.append(
                        {
                            "task_name": task_name,
                            "task_kwargs": task_kwargs,
                            "data_name": data_name,
                            "data_kwargs": data_kwargs,
                            "model_config": model_config,
                            "final_loss": final_loss,
                            "avg_last_100": avg_last_100,
                            "losses": losses,
                            "exp_dir": exp_dir,
                        }
                    )

                except Exception as e:
                    print(f"Error in experiment {experiment_num}: {e}")
                    import traceback
                    traceback.print_exc()

    results_path = os.path.join(save_dir, "all_results.pt")
    torch.save(results, results_path)
    print(f"\nAll results saved to {results_path}")
    return results


# -------------------------------------------------------------------
# 4) Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Define tasks to test
    tasks = [
        ("hermite_regression", {"degree": 1, "weight_type": "standard"}),  # Linear
        ("hermite_regression", {"degree": 2, "weight_type": "standard"}),  # Quadratic
    ]

    # Data distributions
    data_dists = [
        ("gaussian", {}),
    ]

    # Model configurations
    models = [
        {"n_embd": 64, "n_layer": 2, "n_head": 2},
        {"n_embd": 128, "n_layer": 4, "n_head": 4},
    ]

    # Training configuration
    train_config = {
        "n_points": 100,
        "batch_size": 64,
        "train_steps": 50,   # short runs for testing
        "learning_rate": 1e-4,
        "n_dims": 1,
    }

    results = hyperparameter_sweep(
        tasks_to_test=tasks,
        data_distributions=data_dists,
        model_configs=models,
        training_configs=train_config,
        save_dir="./sweep_results",
    )

    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\nExperiment {i}:")
        print(
            f"  Task: {result['task_name']} "
            f"(degree={result['task_kwargs'].get('degree', 'N/A')})"
        )
        print(f"  Data: {result['data_name']}")
        print(
            f"  Model: embd={result['model_config']['n_embd']}, "
            f"layer={result['model_config']['n_layer']}, "
            f"head={result['model_config']['n_head']}"
        )
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Avg Last 100: {result['avg_last_100']:.6f}")
        print(f"  Dir: {result['exp_dir']}")
