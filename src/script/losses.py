from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch

from src.models.origin.layers import Registry

ENDPOINT_LOSSES = Registry("endpoint loss")


def pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Squared Euclidean distances in fp32, optionally with per-feature weights."""
    with torch.autocast(device_type=X.device.type, enabled=False):
        X = X.float()
        Y = Y.float()
        if w is not None:
            scale = w.float().clamp_min(0).sqrt()
            X = X * scale
            Y = Y * scale
        d2 = X.pow(2).sum(dim=1)[:, None] + Y.pow(2).sum(dim=1)[None, :] - 2.0 * (X @ Y.T)
        return d2.clamp_min(0.0)


@torch.no_grad()
def median_sigmas(Z: torch.Tensor, scales=(0.5, 1.0, 2.0, 4.0)) -> list[float]:
    d2 = pairwise_sq_dists(Z, Z)
    off_diagonal = d2[~torch.eye(d2.size(0), dtype=torch.bool, device=d2.device)]
    median = torch.median(off_diagonal).clamp_min(1e-12)
    return [float(math.sqrt(s * median.item())) for s in scales]


def mmd2_unbiased(X: torch.Tensor, Y: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    m, n = X.size(0), Y.size(0)
    dxx = pairwise_sq_dists(X, X)
    dyy = pairwise_sq_dists(Y, Y)
    dxy = pairwise_sq_dists(X, Y)
    values = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * sigma ** 2 + 1e-12)
        kxx = torch.exp(-beta * dxx)
        kyy = torch.exp(-beta * dyy)
        kxy = torch.exp(-beta * dxy)
        term_xx = (kxx.sum() - kxx.diagonal().sum()) / max(m * (m - 1), 1)
        term_yy = (kyy.sum() - kyy.diagonal().sum()) / max(n * (n - 1), 1)
        values.append(term_xx + term_yy - 2.0 * kxy.mean())
    return torch.stack(values).mean()


def mmd(X: torch.Tensor, Y: torch.Tensor, scales=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    sigmas = median_sigmas(torch.cat([X.float(), Y.float()], dim=0), scales=scales)
    return mmd2_unbiased(X, Y, sigmas)


def sinkhorn_cost(C: torch.Tensor, eps: float, n_iters: int) -> torch.Tensor:
    """Entropic OT cost between uniform measures in log-domain.

    Potentials are iterated without gradient; one final differentiable update gives the
    envelope-theorem gradient with respect to C without unrolling the iterations.
    """
    C = C.float()
    m, n = C.shape
    log_a = torch.full((m,), -math.log(m), device=C.device)
    log_b = torch.full((n,), -math.log(n), device=C.device)
    g = torch.zeros(n, device=C.device)
    with torch.no_grad():
        C_fixed = C.detach()
        f = torch.zeros(m, device=C.device)
        for _ in range(n_iters):
            f = -eps * torch.logsumexp(log_b[None, :] + (g[None, :] - C_fixed) / eps, dim=1)
            g = -eps * torch.logsumexp(log_a[:, None] + (f[:, None] - C_fixed) / eps, dim=0)
    f_out = -eps * torch.logsumexp(log_b[None, :] + (g[None, :] - C) / eps, dim=1)
    g_out = -eps * torch.logsumexp(log_a[:, None] + (f[:, None] - C) / eps, dim=0)
    return f_out @ log_a.exp() + g_out @ log_b.exp()


def sinkhorn_divergence(X, Y, eps, n_iters, w=None):
    return (sinkhorn_cost(pairwise_sq_dists(X, Y, w), eps, n_iters)
            - 0.5 * sinkhorn_cost(pairwise_sq_dists(X, X, w), eps, n_iters)
            - 0.5 * sinkhorn_cost(pairwise_sq_dists(Y, Y, w), eps, n_iters))


def sliced_wasserstein(X, Y, w=None, n_projections=128, top_frac=0.1):
    """Mean of the largest squared 1-D Wasserstein distances over random projections."""
    with torch.autocast(device_type=X.device.type, enabled=False):
        X = X.float()
        Y = Y.float()
        if w is not None:
            scale = w.float().clamp_min(0).sqrt()
            X = X * scale
            Y = Y * scale
        theta = torch.randn(X.shape[1], n_projections, device=X.device)
        theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
        x_proj = torch.sort(X @ theta, dim=0).values
        y_proj = torch.sort(Y @ theta, dim=0).values
        w2 = (x_proj - y_proj).pow(2).mean(dim=0)
        k = max(1, int(round(top_frac * n_projections)))
        return torch.topk(w2, k).values.mean()


def is_control_condition(names) -> np.ndarray:
    """A condition is control only when every '+'-separated component is 'control'."""
    return np.array([all(part.strip().lower() == "control" for part in str(name).split("+")) for name in names])


def as_float32(X):
    return X.astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def group_means(X, codes: np.ndarray, n_groups: int) -> np.ndarray:
    indicator = sp.csr_matrix(
        (np.ones(codes.size, dtype=np.float32), (codes, np.arange(codes.size))),
        shape=(n_groups, codes.size),
    )
    counts = np.asarray(indicator.sum(axis=1)).ravel()
    sums = indicator @ X
    sums = sums.toarray() if sp.issparse(sums) else np.asarray(sums)
    return sums / np.maximum(counts, 1.0)[:, None]


def rows_dense(X, mask: np.ndarray) -> np.ndarray:
    rows = X[mask]
    return rows.toarray() if sp.issparse(rows) else np.asarray(rows, dtype=np.float32)


class DegWeights:
    """Per-condition per-gene weights from |mean(condition) - mean(control)|.

    w_i = epsilon + (1 - epsilon) * |lfc_i| / max_j |lfc_j|, keyed by the sorted condition-id
    tuple. Control conditions map to None. Calling the object returns the weights of the
    batch condition restricted to the sampled gene columns, or None when there are none.
    """

    def __init__(self, train_sampler, epsilon: float = 0.1, strict: bool = True):
        adata = train_sampler.adata
        X = as_float32(adata.X)
        names = adata.obs["perturbation_covariates"].astype(str).values
        condition_ids = np.asarray(train_sampler.perturbation_covariates_id)

        control_cells = is_control_condition(names)
        if not control_cells.any():
            raise ValueError("No control cells found in the training split")
        control_mean = rows_dense(X, control_cells).mean(axis=0)

        unique_names, first_index, codes = np.unique(names, return_index=True, return_inverse=True)
        means = group_means(X, codes, unique_names.size)
        control_names = is_control_condition(unique_names)

        self.table = {}
        for j in range(unique_names.size):
            key = tuple(sorted(int(v) for v in np.ravel(condition_ids[first_index[j]])))
            if control_names[j]:
                self.table[key] = None
                continue
            lfc = np.abs(means[j] - control_mean)
            w = epsilon + (1.0 - epsilon) * lfc / max(float(lfc.max()), 1e-12)
            self.table[key] = torch.from_numpy(w.astype(np.float32))

        n_genes = adata.shape[1]
        lengths = {w.shape[0] for w in self.table.values() if w is not None}
        if lengths != {n_genes}:
            raise ValueError(f"DEG weight lengths {lengths} != n_genes {n_genes}")
        self.strict = bool(strict)
        self.misses = 0

    def __call__(self, condition_ids, gene_columns, device):
        row = condition_ids[0] if condition_ids.dim() > 1 else condition_ids
        key = tuple(sorted(int(v) for v in row.reshape(-1).tolist()))
        if key in self.table:
            w = self.table[key]
            return None if w is None else w[gene_columns.cpu()].to(device)
        self.misses += 1
        if self.strict:
            raise KeyError(f"Condition key {key} absent from DEG weights (miss {self.misses})")
        return None


class PrecisionWeights:
    """Per-gene weights proportional to the inverse control-cell variance."""

    def __init__(self, train_sampler, epsilon: float = 0.1):
        adata = train_sampler.adata
        names = adata.obs["perturbation_covariates"].astype(str).values
        control_cells = is_control_condition(names)
        if not control_cells.any():
            raise ValueError("No control cells found in the training split")
        control = rows_dense(as_float32(adata.X), control_cells)
        precision = 1.0 / np.maximum(control.var(axis=0), 1e-4)
        w = epsilon + (1.0 - epsilon) * precision / max(float(precision.max()), 1e-12)
        self.weights = torch.from_numpy(w.astype(np.float32))

    def __call__(self, condition_ids, gene_columns, device):
        return self.weights[gene_columns.cpu()].to(device)


class EndpointLoss:
    """Distribution-level term on the one-step endpoint estimate, added to the flow-matching loss.

    ``weight_kind`` is None, "deg" or "precision" and selects the per-gene weights the run
    supplies. Non-finite values are replaced by zero until ``nonfinite_limit`` consecutive
    occurrences, then the run is stopped.
    """

    weight_kind: str | None = None

    def __init__(self, sinkhorn_eps_scale=0.1, sinkhorn_iters=50, n_slices=128,
                 sliced_top_frac=0.1, sparse_lambda=0.1, nonfinite_limit=20):
        self.sinkhorn_eps_scale = sinkhorn_eps_scale
        self.sinkhorn_iters = sinkhorn_iters
        self.n_slices = n_slices
        self.sliced_top_frac = sliced_top_frac
        self.sparse_lambda = sparse_lambda
        self.nonfinite_limit = nonfinite_limit
        self.nonfinite_count = 0

    def compute(self, prediction, target, weights, source) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, prediction, target, weights=None, source=None, is_main_process=True):
        if self.weight_kind is not None and weights is None:
            raise ValueError(f"{type(self).__name__} requires per-gene weights; skip control batches in the caller")
        value = self.compute(prediction, target, weights, source)
        if not torch.isfinite(value):
            self.nonfinite_count += 1
            if is_main_process:
                print(f"[endpoint] non-finite {type(self).__name__}, consecutive occurrence {self.nonfinite_count}")
            if self.nonfinite_count > self.nonfinite_limit:
                raise RuntimeError(
                    f"{type(self).__name__} non-finite {self.nonfinite_count} consecutive times "
                    f"(limit {self.nonfinite_limit})")
            return torch.zeros((), device=prediction.device)
        self.nonfinite_count = 0
        return value


@ENDPOINT_LOSSES.register("mmd")
class MMDEndpoint(EndpointLoss):
    def compute(self, prediction, target, weights, source):
        return mmd(prediction, target)


@ENDPOINT_LOSSES.register("deg_mmd")
class DegMMDEndpoint(EndpointLoss):
    weight_kind = "deg"

    def compute(self, prediction, target, weights, source):
        scale = weights.clamp_min(0).sqrt()
        return mmd(prediction.float() * scale, target.float() * scale)


@ENDPOINT_LOSSES.register("deg_mse")
class DegMSEEndpoint(EndpointLoss):
    weight_kind = "deg"

    def compute(self, prediction, target, weights, source):
        weighted = (weights * (prediction.float() - target.float()).pow(2)).mean()
        return weighted + 0.1 * mmd(prediction, target, scales=(1.0, 2.0))


@ENDPOINT_LOSSES.register("sinkhorn")
class SinkhornEndpoint(EndpointLoss):
    def divergence(self, prediction, target, weights):
        with torch.no_grad():
            median = torch.median(pairwise_sq_dists(prediction, target, weights)).clamp_min(1e-6)
            eps = float((self.sinkhorn_eps_scale * median).clamp_min(1e-3))
        return sinkhorn_divergence(prediction, target, eps, self.sinkhorn_iters, w=weights)

    def compute(self, prediction, target, weights, source):
        return self.divergence(prediction, target, weights)


@ENDPOINT_LOSSES.register("deg_sinkhorn")
class DegSinkhornEndpoint(SinkhornEndpoint):
    weight_kind = "deg"


@ENDPOINT_LOSSES.register("precision_sinkhorn")
class PrecisionSinkhornEndpoint(SinkhornEndpoint):
    weight_kind = "precision"


@ENDPOINT_LOSSES.register("deg_sinkhorn_sparse")
class DegSinkhornSparseEndpoint(SinkhornEndpoint):
    weight_kind = "deg"

    def compute(self, prediction, target, weights, source):
        if source is None:
            raise ValueError("deg_sinkhorn_sparse requires the source batch")
        mean_shift = prediction.float().mean(dim=0) - source.float().mean(dim=0)
        penalty = ((1.0 - weights) * mean_shift.abs()).mean()
        return self.divergence(prediction, target, weights) + self.sparse_lambda * penalty


@ENDPOINT_LOSSES.register("max_sliced")
class MaxSlicedEndpoint(EndpointLoss):
    def compute(self, prediction, target, weights, source):
        return sliced_wasserstein(prediction, target, w=weights,
                                  n_projections=self.n_slices, top_frac=self.sliced_top_frac)


@ENDPOINT_LOSSES.register("deg_max_sliced")
class DegMaxSlicedEndpoint(MaxSlicedEndpoint):
    weight_kind = "deg"


def build_endpoint_loss(cfg) -> EndpointLoss | None:
    if cfg.endpoint_loss == "none":
        return None
    return ENDPOINT_LOSSES.build(
        cfg.endpoint_loss,
        sinkhorn_eps_scale=cfg.sinkhorn_eps,
        sinkhorn_iters=cfg.sinkhorn_iters,
        n_slices=cfg.n_slices,
        sliced_top_frac=cfg.sliced_top_frac,
        sparse_lambda=cfg.sparse_lambda,
        nonfinite_limit=cfg.sinkhorn_nonfinite_limit,
    )


def build_weights(cfg, train_sampler, endpoint: EndpointLoss | None):
    if endpoint is None or endpoint.weight_kind is None:
        return None
    if endpoint.weight_kind == "deg":
        return DegWeights(train_sampler, epsilon=cfg.deg_epsilon, strict=cfg.deg_strict)
    if endpoint.weight_kind == "precision":
        return PrecisionWeights(train_sampler, epsilon=cfg.precision_epsilon)
    raise ValueError(f"Unknown weight kind: {endpoint.weight_kind}")