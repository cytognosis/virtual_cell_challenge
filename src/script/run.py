import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import hashlib
import inspect
import json
import math
import random
import re
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import tyro
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader

import eval as eval_module
import losses as losses_module
import optimizers as optimizers_module
import src.models.origin.blocks as blocks_module
import src.models.origin.layers as layers_module
import src.models.origin.model as model_module
from config.config_flow import FlowConfig
from eval import dense_matrix, evaluate, log_eval, sample_noise
from losses import build_endpoint_loss, build_weights
from optimizers import build_optimizer, build_scheduler, optimizer_summary
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.instantiate_model import instantiate_model
from src.models.origin.sampling import (build_gene_column_lookup, build_neighbor_column_table,
                                        sample_gene_columns)
from src.utils.utils import save_checkpoint, load_checkpoint, process_vocab

probability_path = AffineProbPath(scheduler=CondOTScheduler())

OPTIONAL_MODEL_FIELDS = (
    "num_heads", "norm", "blocks", "refiner", "refine_injection", "expression_encoder", "input_fusion",
    "gene_reinjection", "perturbation_pooling", "injection", "decoder", "time_scale",
    "vocabulary_interaction", "use_graph", "symmetric_graph",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_source_fingerprint(label, obj):
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(f"[source] {label} path={source_path} sha256={digest}", flush=True)


def file_digest(path: str, length: int = 12) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]


def save_tensor_atomic(tensor: torch.Tensor, path: str):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def coexpression_mask_path(data_manager, config) -> str:
    suffix = "_negative_edge" if config.use_negative_edge else ""
    name = f"mask_fold_{config.fold}topk_{config.topk}{config.split_method}{suffix}.pt"
    return os.path.join(data_manager.data_path, data_manager.data_name, name)


def build_grn_adjacency(edge_pairs, name_to_token: dict, num_tokens: int) -> torch.Tensor:
    rows, cols = [], []
    for a, b in edge_pairs:
        ta, tb = name_to_token.get(a), name_to_token.get(b)
        if ta is None or tb is None or ta == tb:
            continue
        rows.append(ta)
        cols.append(tb)
    adjacency = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    if rows:
        r = torch.tensor(rows, dtype=torch.long)
        c = torch.tensor(cols, dtype=torch.long)
        adjacency[r, c] = True
        adjacency[c, r] = True
    return adjacency


def build_correlation(train_X, gene_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
    corr_hvg = np.corrcoef(dense_matrix(train_X).T).astype(np.float32)
    corr_hvg = np.nan_to_num(corr_hvg, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    idx = gene_ids.cpu().long()
    corr[idx.unsqueeze(1), idx.unsqueeze(0)] = torch.from_numpy(corr_hvg)
    return corr


def build_manifold_coords(train_X, gene_ids: torch.Tensor, num_tokens: int, dim: int, k: int) -> torch.Tensor:
    from src.data_process.precompute_wire import compute_laplacian_eigvecs
    coords_hvg = compute_laplacian_eigvecs(dense_matrix(train_X), dim, k=k)
    coords = torch.zeros((num_tokens, dim), dtype=torch.float32)
    coords[gene_ids.cpu().long()] = torch.as_tensor(np.asarray(coords_hvg), dtype=torch.float32)
    return coords


def build_perturbation_lookup(perturbation_dict: dict, vocab, perturbation_type: str):
    """Map dataset condition ids to model perturbation tokens.

    Returns (lookup tensor indexed by condition id, control token id, number of label embeddings).
    """
    names = {str(name): int(idx) for name, idx in perturbation_dict.items()}
    if "control" not in names:
        raise ValueError("perturbation_dict has no 'control' entry")
    size = max(names.values()) + 1

    if perturbation_type == "label":
        return torch.arange(size, dtype=torch.long), names["control"], size

    missing = sorted(name for name in names if name not in vocab.stoi)
    if missing:
        raise ValueError(f"{len(missing)} perturbation names are not in the gene vocabulary: {missing[:20]}")
    lookup = torch.full((size,), -1, dtype=torch.long)
    for name, idx in names.items():
        lookup[idx] = int(vocab.stoi[name])
    return lookup, int(vocab.stoi["control"]), None


def model_arguments(config, num_tokens, num_labels, control_token_id, mask_path, grn_path, corr_path,
                    gene_ids, manifold_path) -> dict:
    arguments = dict(
        ntoken=num_tokens,
        num_labels=num_labels,
        d_model=config.d_model,
        d_hid=config.d_hid,
        num_blocks=config.num_blocks,
        refine_steps=config.refine_steps,
        dropout=config.dropout,
        residual_scale=config.residual_scale,
        perturbation_type=config.perturbation_type,
        control_token_id=control_token_id,
        mask_path=mask_path,
        grn_path=grn_path,
        corr_path=corr_path,
        max_neighbors=config.max_neighbors,
        gene_token_ids=gene_ids,
        manifold_path=manifold_path,
        neighbor_gate=config.neighbor_gate,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    for name in OPTIONAL_MODEL_FIELDS:
        value = getattr(config, name)
        if value is not None:
            arguments[name] = value
    return arguments


def freeze_for_mode(model: torch.nn.Module, mode: str):
    head_only = mode == "perturbation"
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not head_only) or name.startswith("perturbation_head.")


def find_latest_checkpoint(save_path: str) -> str | None:
    if not os.path.isdir(save_path):
        return None
    candidates = []
    for name in os.listdir(save_path):
        match = re.match(r"iteration_(\d+)$", name)
        if match:
            ckpt_path = os.path.join(save_path, name, "checkpoint.pt")
            if os.path.isfile(ckpt_path):
                candidates.append((int(match.group(1)), ckpt_path))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def train_step(config, model, accelerator, gene_ids, source, target, perturbation_ids, condition_ids,
               iteration, control_token_id, gene_column_lookup, neighbor_columns, endpoint, weights):
    batch = source.shape[0]
    device = accelerator.device
    n_genes = source.shape[-1]

    if config.perturbation_type == "crispr":
        gene_columns, target_count, mandatory_count, coverage = sample_gene_columns(
            n_genes=n_genes,
            n_select=config.infer_top_gene,
            perturbation_ids=perturbation_ids[perturbation_ids != control_token_id],
            gene_column_lookup=gene_column_lookup,
            neighbor_columns=neighbor_columns,
            device=device,
        )
    else:
        gene_columns = torch.randperm(n_genes, device=device)[:min(config.infer_top_gene, n_genes)]
        target_count, mandatory_count, coverage = 0, 0, float("nan")

    source = source[:, gene_columns]
    target = target[:, gene_columns]
    genes = gene_ids[gene_columns]
    stats = {"target_count": target_count, "mandatory_count": mandatory_count, "target_coverage": coverage}

    if config.mode == "velocity":
        t = torch.rand(batch, device=device)
        noise = sample_noise(config, source)
        sample = probability_path.sample(t=t, x_0=noise, x_1=target)
        velocity = model(genes, sample.x_t, sample.t, source, perturbation_ids, mode="velocity")
        flow_loss = F.mse_loss(velocity, sample.dx_t.float())
        loss = flow_loss

        gamma_t = config.gamma * min(1.0, iteration / max(config.gamma_warmup_steps, 1))
        endpoint_value = torch.zeros((), device=device)
        if endpoint is not None and gamma_t > 0:
            batch_weights = weights(condition_ids, gene_columns, device) if weights is not None else None
            if endpoint.weight_kind is None or batch_weights is not None:
                x1_hat = sample.x_t + velocity * (1.0 - t).unsqueeze(-1)
                endpoint_term = endpoint(
                    x1_hat, target, weights=batch_weights, source=source,
                    is_main_process=accelerator.is_main_process,
                )
                loss = loss + gamma_t * endpoint_term
                endpoint_value = endpoint_term.detach()
        stats.update({"flow_loss": flow_loss.detach(), "endpoint_loss": endpoint_value, "gamma": gamma_t})
    else:
        t = torch.ones(batch, device=device)
        predicted = model(genes, target, t, source, perturbation_ids, mode="perturbation")
        with torch.no_grad():
            reference, _, _ = accelerator.unwrap_model(model).encode_perturbation(perturbation_ids, enabled=True)
            has_perturbation = perturbation_ids.ne(control_token_id).any(dim=1)
        if has_perturbation.any():
            similarity = F.cosine_similarity(
                predicted[has_perturbation], reference[has_perturbation].float(), dim=-1
            )
            loss = 1.0 - similarity.mean()
        else:
            loss = predicted.sum() * 0.0
        stats["cosine_loss"] = loss.detach()

    return loss, stats


def log_train(iteration, loss, stats, grad_norm, optimizer):
    record = {"iteration": iteration, "train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}
    if grad_norm is not None:
        record["train/grad_norm"] = float(grad_norm)
    for key, value in stats.items():
        value = float(value)
        if math.isfinite(value):
            record[f"train/{key}"] = value
    wandb.log(record)


def init_wandb(config, save_path, n_params, n_trainable):
    result_path = os.path.normpath(config.result_path)
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=config.wandb_name or os.path.basename(result_path),
        group=config.wandb_group or os.path.basename(os.path.dirname(result_path)),
        id=hashlib.sha256(os.path.abspath(save_path).encode()).hexdigest()[:16],
        resume="allow",
        dir=save_path,
        tags=[config.data_name, config.model_type, config.endpoint_loss, config.optimizer],
        config={
            **asdict(config),
            "architecture": config.architecture(),
            "run_tag": config.run_tag(),
            "save_path": save_path,
            "params": n_params,
            "trainable_params": n_trainable,
        },
    )
    wandb.define_metric("iteration")
    wandb.define_metric("train/*", step_metric="iteration")
    wandb.define_metric("eval/*", step_metric="iteration")


def main():
    config = tyro.cli(FlowConfig)
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(hours=4)),
        ],
    )
    is_main = accelerator.is_main_process
    device = accelerator.device
    set_seed(config.seed)

    save_path = config.make_path()
    if is_main:
        print(config)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "architecture.json"), "w") as handle:
            json.dump(config.architecture(), handle, indent=2, default=str)
        log_source_fingerprint("run", train_step)
        log_source_fingerprint("config", FlowConfig)
        log_source_fingerprint("instantiate_model", instantiate_model)
        log_source_fingerprint("model", model_module)
        log_source_fingerprint("blocks", blocks_module)
        log_source_fingerprint("layers", layers_module)
        log_source_fingerprint("eval", eval_module)
        log_source_fingerprint("losses", losses_module)
        log_source_fingerprint("optimizers", optimizers_module)
        log_source_fingerprint("sampling", sample_gene_columns)
        log_source_fingerprint("data", Data)
        log_source_fingerprint("dataset", PerturbationDataset)

    data_manager = Data("./data")
    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
    )
    train_sampler, valid_sampler, _ = data_manager.load_flow_data(batch_size=config.batch_size)

    dataset_parameters = inspect.signature(PerturbationDataset).parameters
    dataset_kwargs = {}
    if "cell_type_col" in dataset_parameters:
        dataset_kwargs["cell_type_col"] = "cell_type"
    if "plate_col" in dataset_parameters:
        dataset_kwargs["plate_col"] = config.plate_col or None
    train_dataset = PerturbationDataset(train_sampler, config.batch_size, **dataset_kwargs)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    vocab = process_vocab(data_manager, config)
    num_tokens = len(vocab)
    mask_path = coexpression_mask_path(data_manager, config)
    mask_rows = torch.load(mask_path, map_location="cpu").shape[0]
    if mask_rows != num_tokens:
        raise ValueError(f"Co-expression mask rows {mask_rows} != vocab size {num_tokens}; "
                         "mask row order must equal vocab token-id order")

    gene_ids = torch.tensor(vocab.encode(list(data_manager.adata.var_names)), dtype=torch.long)

    manifold_path = (mask_path.replace(".pt", f"_manifold_{config.manifold_dim}.pt")
                     if config.use_manifold else None)
    grn_path = (mask_path.replace(".pt", f"_grn_{file_digest(config.grn_path)}.pt")
                if config.grn_path else None)
    corr_path = mask_path.replace(".pt", "_corr.pt") if config.use_signed_edges else None

    if is_main:
        if manifold_path and not os.path.exists(manifold_path):
            save_tensor_atomic(
                build_manifold_coords(data_manager.adata_train.X, gene_ids, num_tokens,
                                      config.manifold_dim, config.topk),
                manifold_path,
            )
        if grn_path and not os.path.exists(grn_path):
            edges = pd.read_csv(config.grn_path).iloc[:, :2].astype(str).itertuples(index=False, name=None)
            name_to_token = {name: int(vocab.stoi[name]) for name in data_manager.adata.var_names
                             if name in vocab.stoi}
            save_tensor_atomic(build_grn_adjacency(edges, name_to_token, num_tokens), grn_path)
        if corr_path and not os.path.exists(corr_path):
            save_tensor_atomic(build_correlation(data_manager.adata_train.X, gene_ids, num_tokens), corr_path)
    accelerator.wait_for_everyone()

    perturbation_lookup, control_token_id, num_labels = build_perturbation_lookup(
        data_manager.perturbation_dict, vocab, config.perturbation_type,
    )

    model = instantiate_model(
        config.model_type,
        **model_arguments(config, num_tokens, num_labels, control_token_id, mask_path,
                          grn_path, corr_path, gene_ids, manifold_path),
    )
    freeze_for_mode(model, config.mode)

    gene_column_lookup = build_gene_column_lookup(gene_ids, num_tokens)
    gene_ids = gene_ids.to(device)
    set_seed(config.seed + accelerator.process_index)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] params={n_params} trainable={n_trainable} architecture={config.architecture()}",
              flush=True)
        init_wandb(config, save_path, n_params, n_trainable)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    if is_main:
        print(f"[optimizer] type={config.optimizer} | {optimizer_summary(optimizer)}", flush=True)

    checkpoint_path = config.checkpoint_path or find_latest_checkpoint(save_path) or ""
    start_iteration = 0
    if checkpoint_path:
        if is_main:
            print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
        load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        match = re.search(r"iteration_(\d+)", checkpoint_path)
        if match:
            start_iteration = int(match.group(1)) + 1
    if start_iteration >= config.steps:
        if is_main:
            print(f"Checkpoint at iteration {start_iteration - 1} already reached steps={config.steps}.")
        start_iteration = config.steps

    if model.gene_graph is not None:
        neighbor_columns = build_neighbor_column_table(model.gene_graph.neighbors, gene_column_lookup)
    else:
        neighbor_columns = torch.full((num_tokens, 0), -1, dtype=torch.long)
    if is_main:
        print(f"[sampler] genes={config.infer_top_gene} neighbor_width={neighbor_columns.shape[1]}", flush=True)

    if config.test_only:
        if is_main:
            model.to(device)
            output_dir = os.path.join(save_path, f"eval_refine_{config.refine_steps}")
            os.makedirs(output_dir, exist_ok=True)
            score, values = evaluate(config, valid_sampler, model, gene_ids, perturbation_lookup, device, output_dir)
            log_eval(max(start_iteration - 1, 0), score, values)
            wandb.finish()
        accelerator.wait_for_everyone()
        return

    endpoint = build_endpoint_loss(config)
    weights = build_weights(config, train_sampler, endpoint)
    if is_main:
        print(f"[endpoint] loss={config.endpoint_loss} weights={getattr(endpoint, 'weight_kind', None)}",
              flush=True)

    model = accelerator.prepare(model)
    if os.environ.get("SCDFM_COMPILE", "0") == "1":
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, mode="max-autotune")
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    progress = tqdm.tqdm(total=config.steps, initial=start_iteration, disable=not is_main)
    iteration = start_iteration
    while iteration < config.steps:
        for batch_data in dataloader:
            if iteration >= config.steps:
                break
            source = batch_data["src_cell_data"].squeeze(0).to(device)
            target = batch_data["tgt_cell_data"].squeeze(0).to(device)
            condition_ids = batch_data["condition_id"].squeeze(0).cpu().long()
            perturbation_ids = perturbation_lookup[condition_ids].to(device)

            loss, stats = train_step(
                config, model, accelerator, gene_ids, source, target, perturbation_ids, condition_ids,
                iteration=iteration,
                control_token_id=control_token_id,
                gene_column_lookup=gene_column_lookup,
                neighbor_columns=neighbor_columns,
                endpoint=endpoint,
                weights=weights,
            )

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            grad_norm = None
            if config.grad_clip > 0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            scheduler.step()

            if is_main and iteration % config.log_every == 0:
                log_train(iteration, loss, stats, grad_norm, optimizer)

            if iteration > 0 and iteration % config.print_every == 0:
                if is_main:
                    checkpoint_dir = os.path.join(save_path, f"iteration_{iteration}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    base_model = accelerator.unwrap_model(model)
                    save_checkpoint(model=base_model, optimizer=optimizer, scheduler=scheduler,
                                    iteration=iteration, eval_score=None, save_path=checkpoint_dir,
                                    is_best=False)
                    score, values = evaluate(config, valid_sampler, base_model, gene_ids,
                                             perturbation_lookup, device, checkpoint_dir)
                    log_eval(iteration, score, values)
                accelerator.wait_for_everyone()

            if is_main and (iteration == start_iteration or iteration % 1000 == 0):
                print(f"[sampler] iteration={iteration} target_coverage={stats['target_coverage']:.3f} "
                      f"targets={stats['target_count']} mandatory_genes={stats['mandatory_count']}",
                      flush=True)

            progress.update(1)
            progress.set_description(f"loss: {loss.item():.4f}, iteration: {iteration}")
            iteration += 1
    progress.close()
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()