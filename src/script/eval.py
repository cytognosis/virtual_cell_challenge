import math
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torchdiffeq
import tqdm
import wandb
from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
from cell_eval2.config import DEParams

from src.utils.utils import make_lognorm_poisson_noise


def dense_matrix(X) -> np.ndarray:
    return X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return dense_matrix(x)


def sample_noise(config, reference: torch.Tensor) -> torch.Tensor:
    if config.noise_type == "Gaussian":
        return torch.randn_like(reference)
    return make_lognorm_poisson_noise(
        target_log=reference, alpha=config.poisson_alpha, per_cell_L=config.poisson_target_sum,
    )


@torch.no_grad()
def generate(config, model, gene_ids, source, perturbation_ids):
    noise = sample_noise(config, source)

    def velocity_field(t, x):
        return model(gene_ids, x, t, source, perturbation_ids, mode="velocity")

    times = torch.linspace(0.0, 1.0, config.ode_steps, device=source.device)
    trajectory = torchdiffeq.odeint(velocity_field, noise, times, method=config.ode_method)
    return trajectory[-1].clamp_min(0.0)


def eval_settings(config) -> EvalConfig:
    return EvalConfig(
        metrics=config.eval_profile,
        pert_col="perturbation",
        control="control",
        input_type=config.eval_input_type,
        device=config.eval_device,
        de=DEParams(backend=config.eval_de_backend),
        cache_real=os.path.join(config.make_path(), "eval_cache", "real"),
    )


def select_score(values: dict, metric: str) -> float:
    if metric not in values:
        raise KeyError(f"eval_scheme '{metric}' is not among the computed metrics: {sorted(values)}")
    return float(values[metric])


def log_eval(iteration, score, values):
    record = {"iteration": iteration, "eval/score": score}
    record.update({f"eval/{name}": value for name, value in values.items()})
    wandb.log(record)


@torch.no_grad()
def evaluate(config, data_sampler, model, gene_ids, perturbation_lookup, device, output_dir):
    """Generate predictions for every held-out condition and score them with cell_eval2.

    Runs on a single process with an unwrapped model. Returns (selected score, all aggregate metrics).
    """
    was_training = model.training
    model.eval()

    var_names = data_sampler.adata.var_names
    control = torch.as_tensor(to_numpy(data_sampler.get_control_data()["src_cell_data"]))
    pred_blocks, real_blocks = [control.numpy()], [control.numpy()]
    pred_names = ["control"] * control.shape[0]
    real_names = ["control"] * control.shape[0]

    conditions = data_sampler._perturbation_covariates
    print(f"[eval] conditions={len(conditions)}", flush=True)
    for condition in tqdm.tqdm(conditions, desc="eval"):
        data = data_sampler.get_perturbation_data(condition)
        target = to_numpy(data["tgt_cell_data"])
        condition_row = torch.as_tensor(data["condition_id"])[0].cpu().long()
        perturbation = perturbation_lookup[condition_row].to(device)

        source = control[torch.randperm(control.shape[0])[:config.eval_cells]].to(device)
        predictions = []
        for start in range(0, source.shape[0], config.batch_size):
            chunk = source[start:start + config.batch_size]
            chunk_perturbation = perturbation.unsqueeze(0).expand(chunk.shape[0], -1)
            predictions.append(generate(config, model, gene_ids, chunk, chunk_perturbation).cpu())
        predictions = torch.cat(predictions, dim=0).numpy()

        pred_blocks.append(predictions)
        real_blocks.append(target)
        pred_names.extend([condition] * predictions.shape[0])
        real_names.extend([condition] * target.shape[0])

    model.train(was_training)

    predictions = np.concatenate(pred_blocks, axis=0)
    n_nan, n_inf = int(np.isnan(predictions).sum()), int(np.isinf(predictions).sum())
    print(f"[eval] pred non-finite: nan={n_nan} inf={n_inf} "
          f"min={np.nanmin(predictions):.4f} max={np.nanmax(predictions):.4f}", flush=True)
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.concatenate(real_blocks, axis=0)

    def annotate(matrix, names):
        obs = pd.DataFrame({"perturbation": names})
        obs.index = obs.index.astype(str)
        return ad.AnnData(X=matrix.astype(np.float32), obs=obs, var=pd.DataFrame(index=var_names.copy()))

    pred = annotate(predictions, pred_names)
    real = annotate(targets, real_names)

    results = compute_metrics(pred, real, config=eval_settings(config))
    aggregate = aggregate_metrics(results)
    results.write_csv(os.path.join(output_dir, "results.csv"))
    aggregate.write_csv(os.path.join(output_dir, "agg_results.csv"))
    pred.write_h5ad(os.path.join(output_dir, "pred.h5ad"))
    real.write_h5ad(os.path.join(output_dir, "real.h5ad"))

    values = {
        name: float(mean)
        for name, mean in zip(aggregate["metric"].to_list(), aggregate["mean"].to_list())
        if mean is not None and math.isfinite(mean)
    }
    score = select_score(values, config.eval_scheme)
    print(f"[eval] metrics={len(values)} {config.eval_scheme}={score:.4f}", flush=True)
    return score, values