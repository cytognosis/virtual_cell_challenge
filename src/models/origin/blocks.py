from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

try:
    from .layers import (
        AdaptiveModulation, DifferentialAttention, FeedForward, MultiheadAttention, Registry,
        gated_residual, make_norm, modulate, propagate_messages,
    )
except ImportError:
    from layers import (
        AdaptiveModulation, DifferentialAttention, FeedForward, MultiheadAttention, Registry,
        gated_residual, make_norm, modulate, propagate_messages,
    )

BLOCKS = Registry("block")


class StreamContext(NamedTuple):
    condition: Tensor            # (B, d) time and perturbation conditioning
    control: Tensor              # (B, G, d) control-cell embedding
    operator: Tensor | None      # sparse (2G, G) signed message operator


class StreamBlock(nn.Module):
    """Common block interface.

    ``residuals`` returns (dx, dg), the updates to the gene stream (B, G, d) and the global
    state (B, d). Blocks without a global state return zeros for dg. ``prepare`` computes
    everything that does not depend on the stream so tied-weight loops evaluate it once.
    """

    requires_graph = False

    def prepare(self, context: StreamContext):
        return None

    def residuals(self, x: Tensor, g: Tensor, context: StreamContext, cache) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, g: Tensor, context: StreamContext, cache=None) -> tuple[Tensor, Tensor]:
        if cache is None:
            cache = self.prepare(context)
        dx, dg = self.residuals(x, g, context, cache)
        return x + dx, g + dg


class GraphCache(NamedTuple):
    modulation: tuple[Tensor, ...]
    source_term: Tensor | None
    source_pooled: Tensor | None


@BLOCKS.register("graph")
class GraphBlock(StreamBlock):
    """Per-gene mixing with signed graph messages, a global state, and control-cell input."""

    requires_graph = True

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        neighbor_gate: bool = True,
        use_global: bool = True,
        use_source: bool = True,
        signed_messages: bool = True,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.use_global = bool(use_global)
        self.use_source = bool(use_source)
        self.signed_messages = bool(signed_messages)

        self.gene_norm = make_norm(norm, dim)
        self.modulation = AdaptiveModulation(dim, 6 if use_global else 3)
        self.self_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.positive_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.negative_proj = nn.Linear(dim, hidden_dim, bias=False) if signed_messages else None
        self.output_proj = nn.Linear(hidden_dim, dim, bias=False)

        if use_source:
            self.source_norm = make_norm(norm, dim)
            self.source_proj = nn.Linear(dim, hidden_dim, bias=False)

        if use_global:
            self.global_norm = make_norm(norm, dim)
            self.pool_norm = make_norm(norm, dim)
            self.global_proj = nn.Linear(dim, hidden_dim, bias=False)
            self.global_mlp = nn.Sequential(
                nn.Linear((3 if use_source else 2) * dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

        self.neighbor_gate = (
            nn.Sequential(
                nn.Linear((2 if use_global else 1) * dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2 * dim),
                nn.Sigmoid(),
            )
            if neighbor_gate else None
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.initialize_parameters()

    def initialize_parameters(self):
        layers = [self.self_proj, self.positive_proj]
        if self.negative_proj is not None:
            layers.append(self.negative_proj)
        if self.use_global:
            layers.append(self.global_proj)
        if self.use_source:
            layers.append(self.source_proj)
        for layer in layers:
            nn.init.trunc_normal_(layer.weight, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.01)

    def prepare(self, context: StreamContext) -> GraphCache:
        modulation = self.modulation(context.condition)
        source_term = source_pooled = None
        if self.use_source:
            source = self.source_norm(context.control)
            source_term = self.source_proj(source)
            source_pooled = source.mean(dim=1)
        return GraphCache(modulation, source_term, source_pooled)

    def residuals(self, x: Tensor, g: Tensor, context: StreamContext, cache: GraphCache) -> tuple[Tensor, Tensor]:
        if self.use_global:
            shift_x, scale_x, gate_x, shift_g, scale_g, gate_g = cache.modulation
        else:
            shift_x, scale_x, gate_x = cache.modulation

        x_mod = modulate(self.gene_norm(x), shift_x, scale_x)
        hidden = self.self_proj(x_mod)
        gate_input = context.condition
        if self.use_global:
            g_mod = self.global_norm(g) * (1.0 + scale_g) + shift_g
            hidden = hidden + self.global_proj(g_mod).unsqueeze(1)
            gate_input = torch.cat([g_mod, context.condition], dim=-1)
        if cache.source_term is not None:
            hidden = hidden + cache.source_term

        if context.operator is not None:
            positive, negative = propagate_messages(context.operator, x_mod)
            if self.neighbor_gate is not None:
                positive_gate, negative_gate = self.neighbor_gate(gate_input).chunk(2, dim=-1)
                positive = positive * positive_gate.unsqueeze(1)
                negative = negative * negative_gate.unsqueeze(1)
            if self.signed_messages:
                hidden = hidden + self.positive_proj(positive) + self.negative_proj(negative)
            else:
                hidden = hidden + self.positive_proj(positive + negative)

        dx = gated_residual(self.output_proj(self.dropout(self.act(hidden))), gate_x, self.residual_scale)
        if not self.use_global:
            return dx, torch.zeros_like(g)

        pooled = self.pool_norm((x + dx).mean(dim=1))
        features = [g_mod, pooled]
        if self.use_source:
            features.append(cache.source_pooled)
        dg = self.global_mlp(torch.cat(features, dim=-1))
        dg = self.residual_scale * torch.tanh(gate_g) * dg
        return dx, dg


class AttentionCache(NamedTuple):
    modulation: tuple[Tensor, ...]
    source: Tensor | None


class AttentionBlock(StreamBlock):
    """Modulated attention followed by a modulated feed-forward, both with gated residuals."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float, residual_scale: float, norm: str, feed_forward: str):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.attention_norm = make_norm(norm, dim)
        self.feed_forward_norm = make_norm(norm, dim)
        self.modulation = AdaptiveModulation(dim, 6)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout, feed_forward)

    def prepare_source(self, context: StreamContext) -> Tensor | None:
        return None

    def attend(self, x: Tensor, source: Tensor | None) -> Tensor:
        raise NotImplementedError

    def prepare(self, context: StreamContext) -> AttentionCache:
        return AttentionCache(self.modulation(context.condition), self.prepare_source(context))

    def residuals(self, x: Tensor, g: Tensor, context: StreamContext, cache: AttentionCache) -> tuple[Tensor, Tensor]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = cache.modulation
        attended = self.attend(modulate(self.attention_norm(x), shift_a, scale_a), cache.source)
        dx_attention = gated_residual(attended, gate_a, self.residual_scale)
        mixed = x + dx_attention
        transformed = self.feed_forward(modulate(self.feed_forward_norm(mixed), shift_f, scale_f))
        dx_feed_forward = gated_residual(transformed, gate_f, self.residual_scale)
        return dx_attention + dx_feed_forward, torch.zeros_like(g)


@BLOCKS.register("transformer")
class TransformerBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        feed_forward: str = "gelu",
        qk_norm: bool = False,
    ):
        super().__init__(dim, hidden_dim, dropout, residual_scale, norm, feed_forward)
        self.attention = MultiheadAttention(dim, num_heads, dropout, qk_norm)

    def attend(self, x: Tensor, source: Tensor | None) -> Tensor:
        return self.attention(x)


@BLOCKS.register("differential_transformer")
class DifferentialTransformerBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        feed_forward: str = "gelu",
        second_source: str = "control",
    ):
        super().__init__(dim, hidden_dim, dropout, residual_scale, norm, feed_forward)
        self.attention = DifferentialAttention(dim, num_heads, depth, dropout, second_source)
        self.source_norm = make_norm(norm, dim) if second_source == "control" else None

    def prepare_source(self, context: StreamContext) -> Tensor | None:
        return self.source_norm(context.control) if self.source_norm is not None else None

    def attend(self, x: Tensor, source: Tensor | None) -> Tensor:
        return self.attention(x, source)


@BLOCKS.register("cross_attention")
class CrossAttentionBlock(AttentionBlock):
    """Stream queries attend to the control-cell embedding."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        feed_forward: str = "gelu",
    ):
        super().__init__(dim, hidden_dim, dropout, residual_scale, norm, feed_forward)
        self.attention = MultiheadAttention(dim, num_heads, dropout)
        self.source_norm = make_norm(norm, dim)

    def prepare_source(self, context: StreamContext) -> Tensor:
        return self.source_norm(context.control)

    def attend(self, x: Tensor, source: Tensor | None) -> Tensor:
        return self.attention(x, source)


@BLOCKS.register("latent")
class LatentBlock(StreamBlock):
    """Perceiver-style bottleneck: learned latents read the stream, mix, and write back."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        feed_forward: str = "gelu",
        num_latents: int = 64,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.condition_proj = nn.Linear(dim, dim)
        self.modulation = AdaptiveModulation(dim, 3)
        self.input_norm = make_norm(norm, dim)
        self.read_norm = make_norm(norm, dim)
        self.mix_norm = make_norm(norm, dim)
        self.feed_forward_norm = make_norm(norm, dim)
        self.write_norm = make_norm(norm, dim)
        self.read = MultiheadAttention(dim, num_heads, dropout)
        self.mix = MultiheadAttention(dim, num_heads, dropout)
        self.write = MultiheadAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout, feed_forward)

    def prepare(self, context: StreamContext):
        return self.modulation(context.condition), self.condition_proj(context.condition)

    def residuals(self, x: Tensor, g: Tensor, context: StreamContext, cache) -> tuple[Tensor, Tensor]:
        (shift, scale, gate), condition_latent = cache
        x_mod = modulate(self.input_norm(x), shift, scale)
        z = self.latents.unsqueeze(0) + condition_latent.unsqueeze(1)
        z = z + self.read(self.read_norm(z), x_mod)
        z = z + self.mix(self.mix_norm(z))
        z = z + self.feed_forward(self.feed_forward_norm(z))
        dx = gated_residual(self.write(x_mod, self.write_norm(z)), gate, self.residual_scale)
        return dx, torch.zeros_like(g)


@BLOCKS.register("mlp")
class MLPBlock(StreamBlock):
    """Per-gene feed-forward with no gene-gene interaction."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        depth: int = 0,
        norm: str = "layer",
        feed_forward: str = "gelu",
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.norm = make_norm(norm, dim)
        self.modulation = AdaptiveModulation(dim, 3)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout, feed_forward)

    def prepare(self, context: StreamContext):
        return self.modulation(context.condition)

    def residuals(self, x: Tensor, g: Tensor, context: StreamContext, cache) -> tuple[Tensor, Tensor]:
        shift, scale, gate = cache
        dx = gated_residual(self.feed_forward(modulate(self.norm(x), shift, scale)), gate, self.residual_scale)
        return dx, torch.zeros_like(g)


class Adapter(nn.Module):
    """Residual injection of the perturbation action into the stream, applied between blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, action: Tensor) -> Tensor:
        expanded = action.unsqueeze(1).expand(-1, x.size(1), -1)
        return x + self.net(torch.cat([x, expanded], dim=-1))


def run_checkpointed(fn, enabled: bool, *args):
    if enabled and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class Refiner(nn.Module):
    """One block applied ``steps`` times with tied weights, optionally re-injecting its input."""

    def __init__(self, block: StreamBlock, steps: int, input_injection: bool = True):
        super().__init__()
        self.block = block
        self.steps = int(steps)
        self.input_injection = bool(input_injection)

    def forward(self, x: Tensor, g: Tensor, context: StreamContext, checkpointing: bool = False) -> tuple[Tensor, Tensor]:
        cache = self.block.prepare(context)
        x_initial, g_initial = x, g
        for _ in range(self.steps):
            if self.input_injection:
                x_in, g_in = x + x_initial, g + g_initial
            else:
                x_in, g_in = x, g
            dx, dg = run_checkpointed(self.block.residuals, checkpointing, x_in, g_in, context, cache)
            x = x + dx
            g = g + dg
        return x, g


def parse_block_spec(spec) -> tuple[str, dict]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping) or hasattr(spec, "items"):
        options = dict(spec)
        if "type" not in options:
            raise ValueError(f"Block spec {spec} needs a 'type' key")
        return options.pop("type"), options
    raise ValueError(f"Block spec must be a name or a mapping with a 'type' key, got {spec!r}")


def build_block(
    spec,
    *,
    dim: int,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    residual_scale: float,
    norm: str,
    depth: int,
) -> StreamBlock:
    name, options = parse_block_spec(spec)
    params = dict(
        dim=dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
        residual_scale=residual_scale, norm=norm, depth=depth,
    )
    params.update(options)
    return BLOCKS.build(name, **params)