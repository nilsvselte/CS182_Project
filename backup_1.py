import os
# Import from src directory
import sys
from itertools import product

import torch
import wandb
from tqdm import tqdm

sys.path.append('src')

from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler


def train_model(
    model,
    task_sampler,
    data_sampler,
    n_points=100,
    batch_size=64,
    train_steps=10000,
    learning_rate=1e-4,
    log_interval=500,
    device='cpu'
):
    """
    Train a model on a specific task and data distribution.
    
    Args:
        model: The transformer model to train
        task_sampler: Function that returns a task instance
        data_sampler: Data sampler for generating xs
        n_points: Number of context points
        batch_size: Batch size for training
        train_steps: Total training steps
        learning_rate: Learning rate for optimizer
        log_interval: Steps between logging
        device: Device to train on
    
    Returns:
        loss_history: List of losses throughout training
        final_model: Trained model state dict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    
    model.to(device)
    model.train()
    
    pbar = tqdm(range(train_steps), desc="Training")
    
    for step in pbar:
        # Sample data
        xs = data_sampler.sample_xs(n_points, batch_size)
        
        # Sample task and generate labels
        task = task_sampler()
        ys = task.evaluate(xs)
        
        # Move to device
        xs = xs.to(device)
        ys = ys.to(device)
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass (using standard prompt settings)
        _, _, predictions = model(xs, ys, 'standard', 0, 0)
        
        # Compute loss
        loss_func = task.get_training_metric()
        loss = loss_func(predictions, ys)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if step % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss_val:.6f}'})
    
    return loss_history, model.state_dict()


def hyperparameter_sweep(
    tasks_to_test=None,
    data_distributions=None,
    model_configs=None,
    training_configs=None,
    use_wandb=False,
    wandb_project="in-context-learning-sweep",
    save_dir="./sweep_results"
):
    """
    Run a hyperparameter sweep over tasks, data distributions, and model configurations.
    
    Args:
        tasks_to_test: List of tuples (task_name, task_kwargs)
        data_distributions: List of tuples (data_name, data_kwargs)
        model_configs: List of dicts with model hyperparameters
        training_configs: Dict with training hyperparameters
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        save_dir: Directory to save results
    """
    
    # Default configurations
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
            {"n_embd": 256, "n_layer": 6, "n_head": 4},
        ]
    
    if training_configs is None:
        training_configs = {
            "n_points": 100,
            "batch_size": 64,
            "train_steps": 10000,
            "learning_rate": 1e-4,
            "log_interval": 500,
            "n_dims": 1,
        }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Results storage
    results = []
    
    # Iterate over all combinations
    total_experiments = len(tasks_to_test) * len(data_distributions) * len(model_configs)
    experiment_num = 0
    
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
                
                # Initialize wandb run if enabled
                if use_wandb:
                    run = wandb.init(
                        project=wandb_project,
                        config={
                            "task_name": task_name,
                            "task_kwargs": task_kwargs,
                            "data_name": data_name,
                            "data_kwargs": data_kwargs,
                            **model_config,
                            **training_configs,
                        },
                        reinit=True,
                    )
                
                # Create model
                class ModelConfig:
                    def __init__(self, **kwargs):
                        self.family = "gpt2"
                        self.n_dims = training_configs["n_dims"]
                        self.n_positions = training_configs["n_points"]
                        self.transformer_seed = None
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                conf = ModelConfig(**model_config)
                model = build_model(conf)
                
                # Create task sampler
                task_sampler = get_task_sampler(
                    task_name,
                    n_dims=training_configs["n_dims"],
                    batch_size=training_configs["batch_size"],
                    **task_kwargs
                )
                
                # Create data sampler
                data_sampler = get_data_sampler(
                    data_name,
                    n_dims=training_configs["n_dims"],
                    **data_kwargs
                )
                
                # Train model
                try:
                    loss_history, final_state = train_model(
                        model,
                        task_sampler,
                        data_sampler,
                        n_points=training_configs["n_points"],
                        batch_size=training_configs["batch_size"],
                        train_steps=training_configs["train_steps"],
                        learning_rate=training_configs["learning_rate"],
                        log_interval=training_configs["log_interval"],
                        device=device,
                    )
                    
                    # Calculate final metrics
                    final_loss = loss_history[-1]
                    avg_last_100 = sum(loss_history[-100:]) / min(100, len(loss_history))
                    
                    print(f"\nFinal Loss: {final_loss:.6f}")
                    print(f"Avg Last 100 Steps: {avg_last_100:.6f}")
                    
                    # Log to wandb
                    if use_wandb:
                        for step, loss in enumerate(loss_history):
                            wandb.log({"loss": loss}, step=step)
                        wandb.log({
                            "final_loss": final_loss,
                            "avg_last_100": avg_last_100,
                        })
                    
                    # Save results
                    result = {
                        "task_name": task_name,
                        "task_kwargs": task_kwargs,
                        "data_name": data_name,
                        "data_kwargs": data_kwargs,
                        "model_config": model_config,
                        "final_loss": final_loss,
                        "avg_last_100": avg_last_100,
                        "loss_history": loss_history,
                    }
                    results.append(result)
                    
                    # Save model checkpoint
                    save_path = os.path.join(
                        save_dir,
                        f"model_{experiment_num}_{task_name}_{data_name}.pt"
                    )
                    torch.save(final_state, save_path)
                    
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    if use_wandb:
                        wandb.finish()
    
    # Save all results
    results_path = os.path.join(save_dir, "all_results.pt")
    torch.save(results, results_path)
    print(f"\nAll results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Example usage - customize these for your experiments
    
    # Define tasks to test
    tasks = [
        ("hermite_regression", {"degree": 1, "weight_type": "standard"}),  # Linear
        ("hermite_regression", {"degree": 2, "weight_type": "standard"}),  # Quadratic
    ]
    
    # Define data distributions
    data_dists = [
        ("gaussian", {}),
    ]
    
    # Define model configurations
    models = [
        {"n_embd": 64, "n_layer": 2, "n_head": 2},
        {"n_embd": 128, "n_layer": 4, "n_head": 4},
    ]
    
    # Training configuration
    train_config = {
        "n_points": 100,
        "batch_size": 64,
        "train_steps": 50,  # Shorter for quick testing
        "learning_rate": 1e-4,
        "log_interval": 250,
        "n_dims": 1,
    }
    
    # Run sweep
    results = hyperparameter_sweep(
        tasks_to_test=tasks,
        data_distributions=data_dists,
        model_configs=models,
        training_configs=train_config,
        use_wandb=False,  # Set to True if you want to use wandb
        wandb_project="cs182-sweep",
        save_dir="./sweep_results"
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\nExperiment {i}:")
        print(f"  Task: {result['task_name']} (degree={result['task_kwargs'].get('degree', 'N/A')})")
        print(f"  Data: {result['data_name']}")
        print(f"  Model: embd={result['model_config']['n_embd']}, "
              f"layer={result['model_config']['n_layer']}, "
              f"head={result['model_config']['n_head']}")
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Avg Last 100: {result['avg_last_100']:.6f}")
