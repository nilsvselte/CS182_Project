import os

from dual_eval_2 import run_dual_eval, run_dual_eval_switched

# Get the src directory
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
results_dir = os.path.join(src_dir, "results")

# Model paths
models = {
    "dual_sequential": "models/dual_sequential/42c05d4a-c27e-40b2-bb14-8cc4e00d6102",
    "dual_mixed": "models/dual_mixed/4c0cb825-67fb-4ef5-93e9-81f73946620e",
    "dual_random": "models/dual_random/561208dd-1ba4-4de1-a0b9-fa90166ca896",
}

# Evaluation parameters
A_EXAMPLES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
B_EXAMPLES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
TRIALS = 1000
SKIP_EXISTING_QUADRATIC_QUERY = True  # Skip rerunning linear-first/quadratic-query evals if files already exist

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Run evaluation for each model
for model_name, model_path in models.items():
    print(f"\n{'='*50}")
    print(f"Running quadratic-query evaluation for: {model_name}")
    print(f"{'='*50}")

    run_dir = os.path.join(project_root, model_path)

    # Check if model directory exists
    if not os.path.exists(run_dir):
        print(f"Warning: Model directory not found: {run_dir}")
        continue

    try:
        mean_df, std_df = run_dual_eval(
            run_dir=run_dir,
            a_examples=A_EXAMPLES,
            b_examples=B_EXAMPLES,
            trials=TRIALS,
            batch_size=None,
            step=-1,
            device="cuda",
        )

        # Save both dataframes with descriptive filenames
        mean_csv = os.path.join(results_dir, f"{model_name}_quadratic_query_mean.csv")
        std_csv = os.path.join(results_dir, f"{model_name}_quadratic_query_sem.csv")

        if SKIP_EXISTING_QUADRATIC_QUERY and os.path.exists(mean_csv) and os.path.exists(std_csv):
            print(f"• Quadratic-query CSVs already exist for {model_name}; skipping rerun.")
            continue

        mean_df.to_csv(mean_csv)
        std_df.to_csv(std_csv)

        print(f"✓ Quadratic-query mean saved to: {mean_csv}")
        print(f"✓ Quadratic-query SEM saved to: {std_csv}")

    except Exception as e:
        print(f"✗ Error processing quadratic-query eval for {model_name}: {e}")


# Rerun evaluation with quadratic contexts first and linear query
for model_name, model_path in models.items():
    print(f"\n{'='*50}")
    print(f"Running linear-query evaluation for: {model_name}")
    print(f"{'='*50}")

    run_dir = os.path.join(project_root, model_path)

    if not os.path.exists(run_dir):
        print(f"Warning: Model directory not found: {run_dir}")
        continue

    try:
        mean_df, std_df = run_dual_eval_switched(
            run_dir=run_dir,
            a_examples=A_EXAMPLES,
            b_examples=B_EXAMPLES,
            trials=TRIALS,
            batch_size=None,
            step=-1,
            device="cuda",
        )

        mean_csv = os.path.join(results_dir, f"{model_name}_linear_query_mean.csv")
        std_csv = os.path.join(results_dir, f"{model_name}_linear_query_sem.csv")

        mean_df.to_csv(mean_csv)
        std_df.to_csv(std_csv)

        print(f"✓ Linear-query mean saved to: {mean_csv}")
        print(f"✓ Linear-query SEM saved to: {std_csv}")

    except Exception as e:
        print(f"✗ Error processing linear-query eval for {model_name}: {e}")

print(f"\n{'='*50}")
print("Evaluation complete!")
print(f"{'='*50}")
