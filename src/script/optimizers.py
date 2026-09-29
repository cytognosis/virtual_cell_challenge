import math

import torch
import torch.nn as nn


def split_parameters(model: nn.Module) -> tuple[list, list]:
    """Trainable parameters as (decay, no decay). Biases, norms, embeddings and 1-D parameters get no decay."""
    embedding_names = {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules() if isinstance(module, nn.Embedding)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and name not in embedding_names:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg):
    betas = (cfg.adam_beta1, cfg.adam_beta2)
    if cfg.optimizer == "adam":
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(parameters, lr=cfg.lr, betas=betas, eps=cfg.adam_eps)
    if cfg.optimizer == "adamw":
        decay, no_decay = split_parameters(model)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
        ]
        return torch.optim.AdamW(
            [group for group in groups if group["params"]], lr=cfg.lr, betas=betas, eps=cfg.adam_eps,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg):
    """Linear warmup then cosine decay to eta_min, as a multiplier of each group's base lr."""
    min_ratio = min(max(cfg.eta_min / max(cfg.lr, 1e-12), 0.0), 1.0)
    total_steps = max(int(cfg.steps), 1)
    warmup_steps = max(int(cfg.warmup_steps), 0)

    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimizer_summary(optimizer) -> str:
    details = []
    for index, group in enumerate(optimizer.param_groups):
        count = sum(param.numel() for param in group["params"])
        name = group.get("name", f"group{index}")
        details.append(f"{name}:params={count},lr={group['lr']:.3g},wd={group.get('weight_decay', 0.0):g}")
    return " | ".join(details)