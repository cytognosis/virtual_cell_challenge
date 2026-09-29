from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .layers import (
        DECODERS, EXPRESSION_ENCODERS, FUSIONS, GeneConditioner, GeneEncoder, GeneGraph,
        LabelEncoder, SetEncoder, TimestepEmbedder, make_norm,
    )
    from .blocks import BLOCKS, Adapter, Refiner, StreamContext, build_block, parse_block_spec, run_checkpointed
except ImportError:
    from layers import (
        DECODERS, EXPRESSION_ENCODERS, FUSIONS, GeneConditioner, GeneEncoder, GeneGraph,
        LabelEncoder, SetEncoder, TimestepEmbedder, make_norm,
    )
    from blocks import BLOCKS, Adapter, Refiner, StreamContext, build_block, parse_block_spec, run_checkpointed

DEFAULT_BLOCKS = ("graph",) * 8


class PerturbationFlowModel(nn.Module):
    """Conditional velocity field over gene expression for perturbation response.

    Every architectural choice is a constructor argument:

      expression_encoder      "log_linear" | "scalar" | "fourier"
      input_fusion            "concat" | "gene_added" | "sum" | "difference"
      blocks                  sequence of block specs, a name or {"type": name, **options}
                              names: graph, transformer, differential_transformer,
                              cross_attention, latent, mlp
      refiner                 block spec applied refine_steps times with tied weights, or None
      gene_reinjection        per-block gene-identity modulation of the stream
      perturbation_pooling    "sum" | "mean" | "mean_all"
      injection               subset of {"input", "condition", "adapter", "decoder", "target_node"}
      decoder                 "mlp" | "gated_mlp"
      vocabulary_interaction  static cross-attention of gene embeddings over the vocabulary
      use_graph, symmetric_graph, manifold_path, norm, time_scale

    Output modes:
      "velocity"      (B, G) velocity conditioned on the perturbation.
      "perturbation"  (B, d) cell representation computed without perturbation conditioning.
    """

    MODES = ("velocity", "perturbation")
    PERTURBATION_TYPES = ("crispr", "label")
    INJECTION_SITES = ("input", "condition", "adapter", "decoder", "target_node")

    def __init__(
        self,
        *,
        control_token_id: int,
        num_tokens: int = 6000,
        num_labels: int | None = None,
        dim: int = 512,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
        norm: str = "layer",
        perturbation_type: str = "crispr",
        perturbation_pooling: str = "sum",
        injection: Sequence[str] = ("input", "condition", "decoder", "target_node"),
        expression_encoder: str = "log_linear",
        expression_max_value: float = 512.0,
        time_scale: float = 1000.0,
        input_fusion: str = "concat",
        gene_reinjection: bool = False,
        blocks: Sequence = DEFAULT_BLOCKS,
        refiner=None,
        refine_steps: int = 8,
        refine_injection: bool = True,
        decoder: str = "gated_mlp",
        decoder_hidden_dim: int | None = None,
        use_graph: bool = True,
        mask_path: str | None = None,
        grn_path: str | None = None,
        corr_path: str | None = None,
        max_neighbors: int = 128,
        symmetric_graph: bool = True,
        gene_token_ids: Tensor | None = None,
        manifold_path: str | None = None,
        vocabulary_interaction: bool = False,
        autocast: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if perturbation_type not in self.PERTURBATION_TYPES:
            raise ValueError(f"perturbation_type must be one of {self.PERTURBATION_TYPES}, got {perturbation_type}")
        if control_token_id is None:
            raise ValueError("control_token_id is required")
        injection = tuple(injection)
        unknown = set(injection) - set(self.INJECTION_SITES)
        if unknown:
            raise ValueError(f"Unknown injection sites {sorted(unknown)}, available: {self.INJECTION_SITES}")
        if not injection:
            raise ValueError("injection must contain at least one site")
        if "target_node" in injection and perturbation_type != "crispr":
            raise ValueError("target_node injection requires perturbation_type='crispr'")

        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.perturbation_type = perturbation_type
        self.control_token_id = int(control_token_id)
        self.injection = injection
        self.autocast = bool(autocast)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        block_specs = list(blocks)
        all_specs = block_specs + ([refiner] if refiner is not None else [])
        needs_graph = any(BLOCKS.get(parse_block_spec(spec)[0]).requires_graph for spec in all_specs)
        self.gene_graph = (
            GeneGraph(
                num_tokens, mask_path, max_neighbors,
                grn_path=grn_path, corr_path=corr_path, gene_token_ids=gene_token_ids,
                symmetric=symmetric_graph,
            )
            if use_graph else None
        )
        self.graph_operator_needed = bool(use_graph and needs_graph)

        self.gene_encoder = GeneEncoder(
            num_tokens, dim, num_heads=num_heads, dropout=dropout,
            manifold_path=manifold_path, vocabulary_interaction=vocabulary_interaction, mask_path=mask_path,
        )
        self.label_encoder = (LabelEncoder(num_labels or num_tokens, dim) if perturbation_type == "label" else None)
        self.set_encoder = SetEncoder(dim, perturbation_pooling)

        self.state_encoder = EXPRESSION_ENCODERS.build(expression_encoder, dim=dim, dropout=dropout, max_value=expression_max_value)
        self.control_encoder = EXPRESSION_ENCODERS.build(expression_encoder, dim=dim, dropout=dropout, max_value=expression_max_value)
        self.time_embedder = TimestepEmbedder(dim, time_scale=time_scale)
        self.condition_mlp = nn.Sequential(nn.Linear((2 if "condition" in injection else 1) * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.fusion = FUSIONS.build(input_fusion, dim=dim)
        self.action_projection = (nn.Linear(dim, dim, bias=False) if ("input" in injection or "target_node" in injection) else None)
        self.target_action = nn.Linear(dim, dim) if "target_node" in injection else None

        common = dict(
            dim=dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
            residual_scale=residual_scale, norm=norm,
        )
        self.blocks = nn.ModuleList([build_block(spec, depth=i, **common) for i, spec in enumerate(block_specs)])
        self.gene_conditioners = (nn.ModuleList([GeneConditioner(dim, norm) for _ in block_specs]) if gene_reinjection else None)
        self.adapters = nn.ModuleList([Adapter(dim) for _ in block_specs]) if "adapter" in injection else None
        self.refiner = (
            Refiner(build_block(refiner, depth=len(block_specs), **common), refine_steps, refine_injection)
            if refiner is not None else None
        )

        self.output_norm = make_norm(norm, dim)
        self.perturbation_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        decoder_input = dim * (2 if "decoder" in injection else 1)
        self.decoder = DECODERS.build(decoder, input_dim=decoder_input, hidden_dim=decoder_hidden_dim or dim)

        self.initialize_weights()

    def initialize_weights(self):
        def linear_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(linear_init)
        for module in self.modules():
            initializer = getattr(module, "initialize_parameters", None)
            if callable(initializer):
                initializer()
        if self.action_projection is not None:
            nn.init.trunc_normal_(self.action_projection.weight, std=1e-2)

    @staticmethod
    def shared_gene_ids(gene_ids: Tensor) -> Tensor:
        if gene_ids.dim() == 1:
            return gene_ids
        if gene_ids.dim() != 2:
            raise ValueError(f"gene_ids must be (G,) or (B, G), got {tuple(gene_ids.shape)}")
        row = gene_ids[0]
        if not torch.equal(gene_ids, row.unsqueeze(0).expand_as(gene_ids)):
            raise ValueError("All batch rows must share the same gene order")
        return row

    def position_lookup(self, gene_ids: Tensor) -> Tensor:
        lookup = torch.full((self.num_tokens,), -1, dtype=torch.long, device=gene_ids.device)
        lookup[gene_ids] = torch.arange(gene_ids.numel(), device=gene_ids.device)
        return lookup

    def encode_perturbation(self, perturbation_ids: Tensor, enabled: bool) -> tuple[Tensor, Tensor, Tensor]:
        if self.perturbation_type == "crispr":
            elements = self.gene_encoder(perturbation_ids)
        else:
            elements = self.label_encoder(perturbation_ids)
        valid = perturbation_ids.ne(self.control_token_id)
        if not enabled:
            valid = torch.zeros_like(valid)
        action = self.set_encoder(elements, valid, enabled)
        return action, elements, valid

    def target_action_field(
        self,
        perturbation_ids: Tensor,
        elements: Tensor,
        valid: Tensor,
        positions: Tensor,
        num_genes: int,
    ) -> Tensor:
        local = positions[perturbation_ids]
        active = valid & (local >= 0)
        values = self.target_action(elements).float() * active.unsqueeze(-1).float()
        field = torch.zeros(
            (perturbation_ids.size(0), num_genes, values.size(-1)), device=values.device, dtype=values.dtype
        )
        index = local.clamp_min(0).unsqueeze(-1).expand_as(values)
        return field.scatter_add_(1, index, values)

    def forward(
        self,
        gene_ids: Tensor,
        state: Tensor,
        t: Tensor,
        control: Tensor,
        perturbation_ids: Tensor,
        mode: str = "velocity",
    ) -> Tensor:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode}")
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            output = self._forward(gene_ids, state, t, control, perturbation_ids, mode)
        return output.float()

    def _forward(
        self,
        gene_ids: Tensor,
        state: Tensor,
        t: Tensor,
        control: Tensor,
        perturbation_ids: Tensor,
        mode: str,
    ) -> Tensor:
        batch, num_genes = state.shape
        t = t.expand(batch) if t.dim() == 0 else t.reshape(batch)
        gene_row = self.shared_gene_ids(gene_ids)
        positions = self.position_lookup(gene_row)

        gene_emb = self.gene_encoder(gene_row)
        gene_batch = gene_emb.unsqueeze(0).expand(batch, -1, -1)
        state_emb = self.state_encoder(state)
        control_emb = self.control_encoder(control)

        action, elements, valid = self.encode_perturbation(perturbation_ids, enabled=(mode == "velocity"))

        condition_input = self.time_embedder(t)
        if "condition" in self.injection:
            condition_input = torch.cat([condition_input, action], dim=-1)
        condition = self.condition_mlp(condition_input)

        x = self.fusion(gene_batch, state_emb, control_emb)
        injected = None
        if "input" in self.injection:
            injected = action.unsqueeze(1)
        if "target_node" in self.injection:
            field = self.target_action_field(perturbation_ids, elements, valid, positions, num_genes)
            injected = field if injected is None else injected + field
        if injected is not None:
            x = x + self.action_projection(injected)
        g = x.mean(dim=1)

        operator = self.gene_graph.operator(gene_row, positions) if self.graph_operator_needed else None
        context = StreamContext(condition, control_emb, operator)

        checkpointing = self.gradient_checkpointing and self.training
        for i, block in enumerate(self.blocks):
            if self.gene_conditioners is not None:
                x = self.gene_conditioners[i](gene_emb, x)
            if self.adapters is not None:
                x = self.adapters[i](x, action)
            x, g = run_checkpointed(block, checkpointing, x, g, context)

        if self.refiner is not None:
            x, g = self.refiner(x, g, context, checkpointing)

        x = self.output_norm(x)
        if mode == "perturbation":
            return self.perturbation_head(x.mean(dim=1))

        features = x
        if "decoder" in self.injection:
            features = torch.cat([x, action.unsqueeze(1).expand(-1, num_genes, -1)], dim=-1)
        return self.decoder(features)