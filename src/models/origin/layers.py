from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self.entries: dict[str, type] = {}

    def register(self, name: str) -> Callable:
        def decorator(cls):
            if name in self.entries:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self.entries[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        if name not in self.entries:
            raise ValueError(f"Unknown {self.kind} '{name}', available: {self.names()}")
        return self.entries[name]

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self.entries)


EXPRESSION_ENCODERS = Registry("expression encoder")
FUSIONS = Registry("input fusion")
DECODERS = Registry("decoder")

ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight if self.weight is not None else out


def make_norm(name: str, dim: int) -> nn.Module:
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "rms":
        return RMSNorm(dim)
    raise ValueError(f"norm must be 'layer' or 'rms', got {name}")


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def gated_residual(delta: Tensor, gate: Tensor, scale: float) -> Tensor:
    return scale * torch.tanh(gate).unsqueeze(1) * delta


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch, length, dim = x.shape
    return x.view(batch, length, num_heads, dim // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        if activation != "swiglu" and activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {[*ACTIVATIONS, 'swiglu']}, got {activation}")
        self.activation = activation
        self.up = nn.Linear(dim, 2 * hidden_dim if activation == "swiglu" else hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.up(x)
        if self.activation == "swiglu":
            gate, value = hidden.chunk(2, dim=-1)
            hidden = F.silu(gate) * value
        else:
            hidden = ACTIVATIONS[self.activation](hidden)
        return self.down(self.dropout(hidden))


class AdaptiveModulation(nn.Module):
    def __init__(self, dim: int, chunks: int):
        super().__init__()
        self.chunks = chunks
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, chunks * dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.net(condition).chunk(self.chunks, dim=-1))


class MultiheadAttention(nn.Module):
    """Self or cross attention. ``attend_mask`` is boolean, True where attention is allowed."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.dropout = dropout
        head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.query_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, query: Tensor, context: Tensor | None = None, attend_mask: Tensor | None = None) -> Tensor:
        context = query if context is None else context
        q = self.query_norm(split_heads(self.query(query), self.num_heads))
        k = self.key_norm(split_heads(self.key(context), self.num_heads))
        v = split_heads(self.value(context), self.num_heads)
        out = F.scaled_dot_product_attention(
            q.to(v.dtype), k.to(v.dtype), v,
            attn_mask=attend_mask, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(merge_heads(out))


class DifferentialAttention(nn.Module):
    """Difference of two softmax attention maps sharing values.

    The first map always comes from the stream. The second map comes from the stream
    (``second_source="stream"``) or from a control-cell context (``second_source="control"``).
    """

    def __init__(self, dim: int, num_heads: int, depth: int, dropout: float = 0.0, second_source: str = "control"):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if second_source not in ("stream", "control"):
            raise ValueError(f"second_source must be 'stream' or 'control', got {second_source}")
        self.num_heads = num_heads
        self.dropout = dropout
        self.second_source = second_source
        head_dim = dim // num_heads
        self.query_1 = nn.Linear(dim, dim, bias=False)
        self.key_1 = nn.Linear(dim, dim, bias=False)
        self.query_2 = nn.Linear(dim, dim, bias=False)
        self.key_2 = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.head_norm = RMSNorm(head_dim, eps=1e-5)

    def forward(self, x: Tensor, control: Tensor | None = None) -> Tensor:
        second = x if self.second_source == "stream" else control
        if second is None:
            raise ValueError("control context is required when second_source='control'")
        heads = self.num_heads
        v = split_heads(self.value(x), heads)
        p = self.dropout if self.training else 0.0
        out_1 = F.scaled_dot_product_attention(
            split_heads(self.query_1(x), heads), split_heads(self.key_1(x), heads), v, dropout_p=p
        )
        out_2 = F.scaled_dot_product_attention(
            split_heads(self.query_2(second), heads), split_heads(self.key_2(second), heads), v, dropout_p=p
        )
        lambda_full = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        ).to(out_1.dtype)
        out = self.head_norm(out_1 - lambda_full * out_2) * (1.0 - self.lambda_init)
        return self.output(merge_heads(out))


@EXPRESSION_ENCODERS.register("log_linear")
class LogLinearExpressionEncoder(nn.Module):
    """Signed log1p and linear-scaled value features, then an MLP."""

    def __init__(self, dim: int, dropout: float = 0.0, max_value: float = 512.0):
        super().__init__()
        self.max_value = float(max_value)
        self.net = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().clamp(-self.max_value, self.max_value)
        features = torch.stack([torch.sign(x) * torch.log1p(x.abs()), x / self.max_value], dim=-1)
        return self.net(features)


@EXPRESSION_ENCODERS.register("scalar")
class ScalarExpressionEncoder(nn.Module):
    """Raw scalar through Linear, ReLU, Linear, LayerNorm; clamps the upper end only."""

    def __init__(self, dim: int, dropout: float = 0.0, max_value: float = 512.0):
        super().__init__()
        self.max_value = float(max_value)
        self.linear_in = nn.Linear(1, dim)
        self.linear_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().clamp(max=self.max_value).unsqueeze(-1)
        return self.dropout(self.norm(self.linear_out(F.relu(self.linear_in(x)))))


@EXPRESSION_ENCODERS.register("fourier")
class FourierExpressionEncoder(nn.Module):
    """Sinusoidal features of the signed log1p value at geometrically spaced frequencies."""

    def __init__(self, dim: int, dropout: float = 0.0, max_value: float = 512.0, num_frequencies: int = 32):
        super().__init__()
        self.max_value = float(max_value)
        frequencies = 2.0 * math.pi * torch.exp(
            torch.linspace(math.log(0.1), math.log(10.0), num_frequencies)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(2 * num_frequencies + 1, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().clamp(-self.max_value, self.max_value)
        angles = (torch.sign(x) * torch.log1p(x.abs())).unsqueeze(-1) * self.frequencies
        features = torch.cat([torch.sin(angles), torch.cos(angles), (x / self.max_value).unsqueeze(-1)], dim=-1)
        return self.net(features)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of t, multiplied by ``time_scale`` before encoding."""

    def __init__(self, dim: int, frequency_dim: int = 256, time_scale: float = 1000.0):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.time_scale = float(time_scale)
        self.mlp = nn.Sequential(nn.Linear(self.frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    @staticmethod
    def sinusoidal(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t.float() * self.time_scale, self.frequency_dim))


class FusionBase(nn.Module):
    multiplier = 3

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.multiplier * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )

    def features(self, gene: Tensor, state: Tensor, control: Tensor) -> list[Tensor]:
        raise NotImplementedError

    def forward(self, gene: Tensor, state: Tensor, control: Tensor) -> Tensor:
        return self.net(torch.cat(self.features(gene, state, control), dim=-1))


@FUSIONS.register("concat")
class ConcatFusion(FusionBase):
    multiplier = 3

    def features(self, gene, state, control):
        return [gene, state, control]


@FUSIONS.register("gene_added")
class GeneAddedFusion(FusionBase):
    multiplier = 2

    def features(self, gene, state, control):
        return [state + gene, control + gene]


@FUSIONS.register("sum")
class SumFusion(FusionBase):
    multiplier = 1

    def features(self, gene, state, control):
        return [gene + state + control]


@FUSIONS.register("difference")
class DifferenceFusion(FusionBase):
    multiplier = 3

    def features(self, gene, state, control):
        return [gene, state, state - control]


class VocabularyInteraction(nn.Module):
    """Cross-attention from each token to the embeddings of its unblocked vocabulary entries.

    Each token is an independent query, so any id shape is supported. Tokens whose row is
    fully blocked, or that lie outside the mask, receive no interaction.
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int, dropout: float, mask_path: str):
        super().__init__()
        if not mask_path:
            raise ValueError("mask_path is required for vocabulary interaction")
        blocked = torch.load(mask_path, map_location="cpu").bool()
        if blocked.dim() != 2 or blocked.shape[0] != blocked.shape[1]:
            raise ValueError(f"Mask at {mask_path} must be square, got {tuple(blocked.shape)}")
        num_nodes = blocked.shape[0]
        if num_nodes > num_tokens:
            raise ValueError(f"Mask has {num_nodes} nodes but vocabulary has {num_tokens} tokens")
        padded = torch.ones((num_tokens, num_nodes), dtype=torch.bool)
        padded[:num_nodes] = blocked
        self.register_buffer("blocked", padded, persistent=False)
        self.num_nodes = num_nodes
        self.query_norm = nn.LayerNorm(dim)
        self.attention = MultiheadAttention(dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, 4 * dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding: Tensor, ids: Tensor, memory: Tensor) -> Tensor:
        dim = embedding.size(-1)
        queries = embedding.reshape(1, -1, dim)
        blocked = self.blocked[ids.reshape(-1)]
        empty = blocked.all(dim=-1)
        attend = ((~blocked) | empty.unsqueeze(-1))[None, None]
        out = self.attention(self.query_norm(queries), memory.unsqueeze(0), attend)
        out = out * (~empty).view(1, -1, 1).to(out.dtype)
        x = queries + self.dropout(out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x.reshape(embedding.shape)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        manifold_path: str | None = None,
        vocabulary_interaction: bool = False,
        mask_path: str | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.norm = nn.LayerNorm(dim)

        self.manifold_mlp = None
        if manifold_path:
            coords = torch.load(manifold_path, map_location="cpu", weights_only=True).float()
            if coords.shape[0] != num_tokens:
                raise ValueError(f"Manifold coordinates have {coords.shape[0]} rows, expected {num_tokens}")
            self.register_buffer("manifold_coords", coords, persistent=True)
            self.manifold_mlp = nn.Sequential(nn.Linear(coords.shape[1], dim), nn.SiLU(), nn.Linear(dim, dim))

        self.interaction = (
            VocabularyInteraction(num_tokens, dim, num_heads, dropout, mask_path)
            if vocabulary_interaction else None
        )

    def forward(self, ids: Tensor) -> Tensor:
        emb = self.norm(self.embedding(ids))
        if self.manifold_mlp is not None:
            emb = emb + self.manifold_mlp(self.manifold_coords[ids])
        if self.interaction is not None:
            memory = self.norm(self.embedding.weight[: self.interaction.num_nodes])
            emb = self.interaction(emb, ids, memory)
        return emb


class LabelEncoder(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: Tensor) -> Tensor:
        return self.norm(self.embedding(ids))


class SetEncoder(nn.Module):
    """Permutation-invariant encoder of a perturbation token set.

    sum       masked sum over non-control tokens
    mean      masked mean over non-control tokens
    mean_all  mean over every token including control and padding
    The action is exactly zero when no token is valid (sum, mean) or when disabled (mean_all).
    """

    POOLING = ("sum", "mean", "mean_all")

    def __init__(self, dim: int, pooling: str = "sum"):
        super().__init__()
        if pooling not in self.POOLING:
            raise ValueError(f"pooling must be one of {self.POOLING}, got {pooling}")
        self.pooling = pooling
        self.element_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.set_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, elements: Tensor, valid: Tensor, enabled: bool = True) -> Tensor:
        encoded = self.element_mlp(elements)
        if self.pooling == "mean_all":
            action = self.set_mlp(encoded.mean(dim=1))
            return action if enabled else action * 0.0
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1)
        if self.pooling == "mean":
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        action = self.set_mlp(pooled)
        return action * valid.any(dim=1, keepdim=True).to(action.dtype)


class GeneConditioner(nn.Module):
    """Per-gene modulation of the stream from the gene embedding, zero-initialized."""

    def __init__(self, dim: int, norm: str = "layer"):
        super().__init__()
        self.norm = make_norm(norm, dim)
        self.modulation = AdaptiveModulation(dim, 3)

    def forward(self, gene_emb: Tensor, x: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(gene_emb)
        return x + gate * (self.norm(x) * (1.0 + scale) + shift)


def load_adjacency(path: str) -> Tensor:
    raw = torch.load(path, map_location="cpu")
    if raw.dim() != 2:
        raise ValueError(f"Adjacency at {path} must be 2-D, got shape {tuple(raw.shape)}")
    if raw.dtype == torch.bool:
        if raw.shape[0] != raw.shape[1]:
            raise ValueError(f"Boolean adjacency at {path} must be square, got {tuple(raw.shape)}")
        return raw.clone()
    if raw.is_floating_point():
        raise ValueError(f"Adjacency at {path} must be a boolean matrix or an integer neighbor table")
    table = raw.long()
    num_nodes = table.shape[0]
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    rows = torch.arange(num_nodes).unsqueeze(1).expand_as(table)
    valid = table >= 0
    adjacency[rows[valid], table[valid]] = True
    return adjacency


def symmetrize_graph(neighbors: Tensor, weights: Tensor, max_degree: int) -> tuple[Tensor, Tensor]:
    """Add reverse edges so that j in neighbors[i] implies i in neighbors[j], up to ``max_degree``.

    Original edges keep their rank order and always come first; reverse-only edges follow,
    ordered by |weight|.
    """
    num_nodes, k = neighbors.shape
    valid = neighbors >= 0
    rows = torch.arange(num_nodes).unsqueeze(1).expand_as(neighbors)[valid]
    cols = neighbors[valid]
    rank = ((k - torch.arange(k, dtype=torch.float32)) / k).unsqueeze(0).expand_as(neighbors)[valid]
    values = weights[valid]

    own = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    own[rows, cols] = True
    weight = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    weight[rows, cols] = values
    score = torch.full((num_nodes, num_nodes), float("-inf"))
    score[rows, cols] = 2.0 + rank

    reverse = own.t() & ~own
    score = torch.where(reverse, weight.t().abs().clamp(max=1.0), score)
    weight = torch.where(own, weight, weight.t())
    del own, reverse

    top_score, top_index = score.topk(min(int(max_degree), num_nodes), dim=1)
    keep = torch.isfinite(top_score)
    new_neighbors = torch.where(keep, top_index, torch.full_like(top_index, -1))
    new_weights = torch.where(keep, weight.gather(1, top_index), torch.zeros_like(top_score))
    return new_neighbors, new_weights


def build_gene_graph(
    mask_path: str,
    num_tokens: int,
    max_neighbors: int,
    grn_path: str | None = None,
    corr_path: str | None = None,
    gene_token_ids: Tensor | None = None,
    symmetric: bool = True,
    chunk_size: int = 1024,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Build a token-level signed neighbor table.

    ``mask_path`` holds a square boolean matrix where True marks a blocked pair.
    ``grn_path`` optionally adds edges. Neighbors are ranked by |correlation| when
    ``corr_path`` is given, otherwise by the number of supporting sources with a seeded
    random tie-break. ``gene_token_ids`` restricts candidate neighbors to measured genes.
    With ``symmetric`` the table is closed under edge reversal (degree capped at twice
    ``max_neighbors``).

    Returns ``neighbors`` (num_tokens, width) with -1 padding and signed ``weights``.
    """
    blocked = torch.load(mask_path, map_location="cpu").bool()
    if blocked.dim() != 2 or blocked.shape[0] != blocked.shape[1]:
        raise ValueError(f"Mask at {mask_path} must be square, got {tuple(blocked.shape)}")
    num_nodes = blocked.shape[0]
    if num_nodes > num_tokens:
        raise ValueError(f"Mask has {num_nodes} nodes but vocabulary has {num_tokens} tokens")

    source_count = (~blocked).to(torch.uint8)
    del blocked
    if grn_path:
        grn = load_adjacency(grn_path)
        if grn.shape != source_count.shape:
            raise ValueError(f"GRN shape {tuple(grn.shape)} does not match mask shape {tuple(source_count.shape)}")
        source_count += grn.to(torch.uint8)
        del grn
    source_count.fill_diagonal_(0)

    if gene_token_ids is not None:
        measured = torch.zeros(num_nodes, dtype=torch.bool)
        measured[gene_token_ids.detach().cpu().long()] = True
        source_count[:, ~measured] = 0

    corr = None
    if corr_path:
        corr = torch.nan_to_num(torch.load(corr_path, map_location="cpu").float(), nan=0.0)
        if corr.shape != source_count.shape:
            raise ValueError(f"Correlation shape {tuple(corr.shape)} does not match mask shape {tuple(source_count.shape)}")

    k = min(int(max_neighbors), num_nodes)
    neighbors = torch.full((num_nodes, k), -1, dtype=torch.long)
    weights = torch.zeros((num_nodes, k), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        counts = source_count[start:end].float()
        if corr is not None:
            signed = corr[start:end]
            connected = (counts > 0) & (signed != 0)
            score = signed.abs()
        else:
            connected = counts > 0
            signed = connected.float()
            score = counts + 0.5 * torch.rand(counts.shape, generator=generator)
        score = score.masked_fill(~connected, float("-inf"))
        top_score, top_index = score.topk(k, dim=1)
        keep = torch.isfinite(top_score)
        neighbors[start:end] = torch.where(keep, top_index, torch.full_like(top_index, -1))
        weights[start:end] = torch.where(keep, signed.gather(1, top_index), torch.zeros_like(top_score))
    del source_count, corr

    if symmetric:
        neighbors, weights = symmetrize_graph(neighbors, weights, 2 * k)

    width = max(int((neighbors >= 0).sum(dim=1).max().item()), 1)
    table = torch.full((num_tokens, width), -1, dtype=torch.long)
    table_weights = torch.zeros((num_tokens, width), dtype=torch.float32)
    table[:num_nodes] = neighbors[:, :width]
    table_weights[:num_nodes] = weights[:, :width]
    return table.contiguous(), table_weights.contiguous()


def propagate_messages(operator: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Apply a sparse (2G, G) signed operator to x of shape (B, G, d)."""
    batch, genes, dim = x.shape
    flat = x.transpose(0, 1).reshape(genes, batch * dim)
    with torch.autocast(device_type=x.device.type, enabled=False):
        out = torch.sparse.mm(operator, flat.float())
    out = out.to(x.dtype).reshape(2, genes, batch, dim).permute(0, 2, 1, 3)
    return out[0], out[1]


class GeneGraph(nn.Module):
    """Token-level signed neighbor table stored in the checkpoint, plus the per-subset operator."""

    def __init__(
        self,
        num_tokens: int,
        mask_path: str,
        max_neighbors: int = 128,
        grn_path: str | None = None,
        corr_path: str | None = None,
        gene_token_ids: Tensor | None = None,
        symmetric: bool = True,
        cache_operator: bool = True,
    ):
        super().__init__()
        if not mask_path:
            raise ValueError("mask_path is required to build the gene graph")
        neighbors, weights = build_gene_graph(
            mask_path, num_tokens, max_neighbors,
            grn_path=grn_path, corr_path=corr_path, gene_token_ids=gene_token_ids, symmetric=symmetric,
        )
        self.register_buffer("neighbors", neighbors, persistent=True)
        self.register_buffer("edge_weights", weights, persistent=True)
        self.cache_operator = bool(cache_operator)
        self.cache: tuple[Tensor, Tensor] | None = None

    def _apply(self, fn):
        self.cache = None
        return super()._apply(fn)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("neighbors", "edge_weights"):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = state_dict[key].detach().clone().to(self._buffers[name].device)
        self.cache = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def operator(self, gene_ids: Tensor, positions: Tensor) -> Tensor:
        if self.cache_operator and self.cache is not None:
            cached_ids, cached = self.cache
            if cached_ids.shape == gene_ids.shape and torch.equal(cached_ids, gene_ids):
                return cached
        built = self.build_operator(gene_ids, positions)
        if self.cache_operator:
            self.cache = (gene_ids.detach().clone(), built)
        return built

    def build_operator(self, gene_ids: Tensor, positions: Tensor) -> Tensor:
        """Sparse (2G, G): rows [0, G) aggregate positive edges, rows [G, 2G) negative edges.

        Each row is divided by the number of neighbors present in the subset, giving
        degree-normalized weighted means that keep the correlation magnitude.
        """
        num_genes = gene_ids.numel()
        neighbors = self.neighbors[gene_ids]
        weights = self.edge_weights[gene_ids]
        local = positions[neighbors.clamp_min(0)]
        valid = (neighbors >= 0) & (local >= 0) & (weights != 0)
        degree = valid.sum(dim=1, keepdim=True).clamp_min(1).to(weights.dtype)
        normalized = weights * valid.to(weights.dtype) / degree

        rows = torch.arange(num_genes, device=gene_ids.device).unsqueeze(1).expand_as(neighbors)
        positive = normalized > 0
        negative = normalized < 0
        index = torch.stack([
            torch.cat([rows[positive], rows[negative] + num_genes]),
            torch.cat([local[positive], local[negative]]),
        ])
        values = torch.cat([normalized[positive], -normalized[negative]]).float()
        return torch.sparse_coo_tensor(index, values, (2 * num_genes, num_genes)).coalesce()


@DECODERS.register("mlp")
class ExpressionDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


@DECODERS.register("gated_mlp")
class GatedExpressionDecoder(ExpressionDecoder):
    """MLP output multiplied by a learned per-gene sigmoid gate initialized near 0.12."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * torch.sigmoid(self.gate(x)).squeeze(-1)