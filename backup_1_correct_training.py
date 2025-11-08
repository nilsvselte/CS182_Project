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
from train import train


def create_args_object(
    model,
    task_name,
    task_kwargs,
    data_name,
    data_kwargs,
    n_dims=1,
    n_points=100,
    batch_size=64,
    train_steps=10000,
    learning_rate=1e-4,
    log_interval=500,
    save_dir="./sweep_results",
    prompt_type='standard',
    weight_type='standard',
    task_schedule='random',
    data_schedule='random',
    data_split='steps_based'
):
    """
    Create an args object that mimics the Quinine config structure expected by train().
    """
    class Args:
        pass
    
    args = Args()
    
    # Model config
    args.model = Args()
    args.model.family = "gpt2"
    args.model.n_dims = n_dims
    args.model.n_positions = n_points
    
    # Training config
    args.training = Args()
    args.training.batch_size = batch_size
    args.training.n_points = n_points
    args.training.learning_rate = learning_rate
    args.training.train_steps = train_steps
    args.training.save_every_steps = train_steps + 1  # Don't save during sweep
    args.training.keep_every_steps = -1
    
    # Task configuration
    args.training.task_kwargs = {
        "task_0": {
            "task": task_name,
            **task_kwargs
        }
    }
    
    # Data configuration  
    args.training.data_kwargs = {
        "data_0": {
            "data": data_name,
            "type": "standard",  # or "skewed" if needed
            **data_kwargs
        }
    }
    
    # Prompt configuration
    args.training.prompt_kwargs = {}
    args.training.prompt_type = prompt_type
    args.training.weight_type = weight_type
    
    # Scheduling
    args.training.task_schedule = task_schedule
    args.training.data_schedule = data_schedule
    args.training.data_split = data_split
    args.training.val_type = 'standard'
    
    # Wandb config
    args.wandb = Args()
    args.wandb.log_every_steps = log_interval
    args.wandb.project = "sweep"
    args.wandb.entity = "entity"
    args.wandb.notes = ""
    args.wandb.name = None
    
    # Other
    args.out_dir = save_dir
    args.test_run = True  # Set to True to avoid wandb initialization
    
    return args


def train_model(
    model,
    task_name,
    task_kwargs,
    data_name, 
    data_kwargs,
    n_dims=1,
    n_points=100,
    batch_size=64,
    train_steps=10000,
    learning_rate=1e-4,
    log_interval=500,
    device='cpu',
    save_dir="./sweep_results",
    prompt_type='standard',
    weight_type='standard',
    task_schedule='random',
    data_schedule='random',
    data_split='steps_based'
):
    """
    Train a model using the train() function from src/train.py.
    
    Args:
        model: The transformer model to train
        task_name: Name of the task (e.g., 'hermite_regression')
        task_kwargs: Dict of task parameters (e.g., {'degree': 1})
        data_name: Name of data sampler (e.g., 'gaussian')
        data_kwargs: Dict of data sampler parameters
        n_dims: Number of dimensions
        n_points: Number of context points
        batch_size: Batch size for training
        train_steps: Total training steps
        learning_rate: Learning rate for optimizer
        log_interval: Steps between logging
        device: Device to train on
        save_dir: Directory to save results
        prompt_type: Type of prompt
        weight_type: Type of weight initialization
        task_schedule: How to schedule tasks ('random', 'sequential', 'mixed_sequential')
        data_schedule: How to schedule data
        data_split: How to split data
    
    Returns:
        loss_history: List of losses throughout training (extracted from file)
        final_model: Trained model state dict
    """
    # Create args object
    args = create_args_object(
        model=model,
        task_name=task_name,
        task_kwargs=task_kwargs,
        data_name=data_name,
        data_kwargs=data_kwargs,
        n_dims=n_dims,
        n_points=n_points,
        batch_size=batch_size,
        train_steps=train_steps,
        learning_rate=learning_rate,
        log_interval=log_interval,
        save_dir=save_dir,
        prompt_type=prompt_type,
        weight_type=weight_type,
        task_schedule=task_schedule,
        data_schedule=data_schedule,
        data_split=data_split
    )
    
    # Create a temporary file to capture MSE losses
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Move model to device
    model.to(device)
    model.train()
    
    # Open the MSE file and call train
    with open(temp_file_path, 'w') as file_mse:
        train(model, args, file_mse)
    
    # Read loss history from the file
    loss_history = []
    with open(temp_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                loss_history.append(float(parts[1]))
    
    # Clean up temp file
    import os as os_module
    os_module.remove(temp_file_path)
    
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
                
                # Train model - now directly using train() from src
                try:
                    loss_history, final_state = train_model(
                        model,
                        task_name=task_name,
                        task_kwargs=task_kwargs,
                        data_name=data_name,
                        data_kwargs=data_kwargs,
                        n_dims=training_configs["n_dims"],
                        n_points=training_configs["n_points"],
                        batch_size=training_configs["batch_size"],
                        train_steps=training_configs["train_steps"],
                        learning_rate=training_configs["learning_rate"],
                        log_interval=training_configs["log_interval"],
                        device=device,
                        save_dir=save_dir,
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
