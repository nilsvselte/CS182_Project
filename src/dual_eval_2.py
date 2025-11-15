from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F

from eval import build_task_labeler, get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler


def run_dual_eval(run_dir, a_examples, b_examples, trials, batch_size, step, device): 
    # trials: how many times you resample for each (A, B) combination
    # batch_size: number of sequences within each trial

    model, conf = get_model_from_run(run_dir, step=step)
    task_labeler = build_task_labeler(conf)

    if batch_size is None:
        batch_size = conf.training.batch_size

    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.to(device)
    model.eval()

    total_dims = task_labeler.model_n_dims
    feature_dims = task_labeler.feature_dims
    data_sampler = get_data_sampler(conf.training.data, n_dims=total_dims)

    task_kwargs = getattr(conf.training, "task_kwargs", {})
    num_tasks = getattr(conf.training, "num_tasks", None)

    linear_sampler = get_task_sampler(
        "linear_regression",
        feature_dims,
        batch_size,
        num_tasks=num_tasks,
        **task_kwargs,
    )
    quadratic_sampler = get_task_sampler(
        "quadratic_regression",
        feature_dims,
        batch_size,
        num_tasks=num_tasks,
        **task_kwargs,
    )

    truncation = task_labeler.augmentation_truncation(
        conf.training.curriculum.dims.end
    )

    results = OrderedDict()
    for n_linear in a_examples:
        row = OrderedDict()
        for n_quadratic in b_examples:
            row[n_quadratic] = _average_quadratic_loss(
                model,
                data_sampler,
                linear_sampler,
                quadratic_sampler,
                n_linear,
                n_quadratic,
                batch_size,
                trials,
                truncation,
                device,
            )
        results[n_linear] = row

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "linear_examples"
    df.columns = [f"quadratic_{c}" for c in df.columns]
    return df


def _average_quadratic_loss(
    model,
    data_sampler,
    linear_sampler,
    quadratic_sampler,
    a_examples,
    b_examples,
    batch_size,
    trials,
    truncation,
    device,
):
    # include one additional quadratic point whose label is hidden from the model
    total_points = a_examples + b_examples + 1

    losses = []
    with torch.no_grad():
        for _ in range(trials):
            xs = data_sampler.sample_xs(total_points, batch_size, truncation)
            segments = []
            if a_examples:
                linear_task = linear_sampler()
                segments.append(linear_task.evaluate(xs[:, :a_examples, :]))

            quad_task = quadratic_sampler()
            if b_examples:
                context_slice = xs[:, a_examples : a_examples + b_examples, :]
                quad_context = quad_task.evaluate(context_slice)
                segments.append(quad_context)

            query_slice = xs[:, a_examples + b_examples :, :]
            quad_query = quad_task.evaluate(query_slice)  # true label for next B example
            placeholder = torch.zeros_like(quad_query)
            segments.append(placeholder)

            ys_input = torch.cat(segments, dim=1)
            preds = model(xs.to(device), ys_input.to(device)).cpu()
            pred_query = preds[:, -1]
            if pred_query.ndim > 1:
                pred_query = pred_query.squeeze(-1)
            target_query = quad_query[:, 0]
            losses.append(F.mse_loss(pred_query, target_query, reduction="mean").item())

    return sum(losses) / len(losses)
