import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

PERTURBATION_TYPES = ("crispr", "label")
MODES = ("velocity", "perturbation")
NOISE_TYPES = ("Gaussian", "Poisson")
ENDPOINT_TYPES = (
    "none", "mmd", "sinkhorn", "precision_sinkhorn", "max_sliced",
    "deg_mmd", "deg_mse", "deg_sinkhorn", "deg_max_sliced", "deg_sinkhorn_sparse",
)

ARCHITECTURE_FIELDS = (
    "model_type", "d_model", "d_hid", "num_heads", "num_blocks", "refine_steps", "dropout",
    "residual_scale", "norm", "blocks", "refiner", "refine_injection", "expression_encoder",
    "input_fusion", "gene_reinjection", "perturbation_pooling", "injection", "decoder", "time_scale",
    "vocabulary_interaction", "use_graph", "symmetric_graph", "max_neighbors", "neighbor_gate",
    "perturbation_type", "use_manifold", "manifold_dim", "use_signed_edges", "grn_path",
)

# Fields that do not change what a run computes or where its checkpoints belong.
NON_IDENTITY_FIELDS = (
    "result_path", "checkpoint_path", "test_only", "print_every", "eval_cells",
    "ode_steps", "ode_method", "eval_scheme", "eval_profile", "eval_input_type",
    "eval_de_backend", "eval_device", "steps", "seed",
    "wandb_project", "wandb_entity", "wandb_group", "wandb_name", "log_every",
)


@dataclass
class FlowConfig:
    """Run configuration.

    Architecture fields left at None use the preset selected by ``model_type``.
    ``blocks`` and ``refiner`` take spec strings such as ``graph*4,transformer:qk_norm=true*4``;
    ``refiner=none`` removes the refiner. ``injection`` is a comma-separated list of sites.
    """

    # Architecture
    model_type: str = "origin"
    d_model: int = 512
    d_hid: int = 2048
    num_heads: int | None = None
    num_blocks: int = 8
    refine_steps: int = 8
    dropout: float = 0.1
    residual_scale: float = 0.1
    norm: str | None = None
    blocks: str | None = None
    refiner: str | None = None
    refine_injection: bool | None = None
    expression_encoder: str | None = None
    input_fusion: str | None = None
    gene_reinjection: bool | None = None
    perturbation_pooling: str | None = None
    injection: str | None = None
    decoder: str | None = None
    time_scale: float | None = None
    vocabulary_interaction: bool | None = None
    use_graph: bool | None = None
    symmetric_graph: bool | None = None
    max_neighbors: int = 128
    neighbor_gate: bool = True
    gradient_checkpointing: bool = False

    # Optimization
    optimizer: Literal["adam", "adamw"] = "adam"
    batch_size: int = 32
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_clip: float = 1.0
    seed: int = 42

    # Data and perturbations
    data_name: str = "combosciplex"
    perturbation_type: str = "crispr"
    mode: str = "velocity"
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    split_method: str = "additive"
    plate_col: str = ""
    fold: int = 0

    # Gene graph
    use_negative_edge: bool = False
    topk: int = 15
    use_signed_edges: bool = False
    grn_path: str = ""
    use_manifold: bool = False
    manifold_dim: int = 32

    # Source noise
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    # Endpoint loss (active only when gamma > 0)
    endpoint_loss: str = "mmd"
    gamma: float = 0.0
    gamma_warmup_steps: int = 15000
    deg_epsilon: float = 0.1
    deg_strict: bool = False
    precision_epsilon: float = 0.1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    sinkhorn_nonfinite_limit: int = 20
    n_slices: int = 128
    sliced_top_frac: float = 0.1
    sparse_lambda: float = 0.1

    # Evaluation and checkpoints
    print_every: int = 5000
    eval_cells: int = 128
    eval_scheme: str = "expr_mse"
    eval_profile: str = "full"
    eval_input_type: str = "lognorm"
    eval_de_backend: str = "pdex"
    eval_device: str = "auto"
    ode_steps: int = 20
    ode_method: str = "rk4"
    result_path: str = "./result"
    checkpoint_path: str = ""
    test_only: bool = False


    # Logging
    wandb_project: str = "scdfm-ablation"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_name: str = ""
    log_every: int = 50
    
    def __post_init__(self):
        if self.data_name == "norman_umi_go_filtered":
            self.n_top_genes = 5054
        if self.data_name == "norman":
            self.n_top_genes = 5000
        if self.perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(f"perturbation_type must be one of {PERTURBATION_TYPES}, got {self.perturbation_type}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode}")
        if self.noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {self.noise_type}")
        if self.endpoint_loss not in ENDPOINT_TYPES:
            raise ValueError(f"endpoint_loss must be one of {ENDPOINT_TYPES}, got {self.endpoint_loss}")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(f"optimizer must be 'adam' or 'adamw', got {self.optimizer}")

    def architecture(self) -> dict:
        return {name: getattr(self, name) for name in ARCHITECTURE_FIELDS}

    def run_tag(self) -> str:
        """Hash of every field that changes what the run computes, so runs never share a directory."""
        identity = {k: v for k, v in asdict(self).items() if k not in NON_IDENTITY_FIELDS}
        payload = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    def make_path(self):
        name = "-".join([
            "flow",
            self.data_name,
            self.model_type,
            self.mode,
            self.perturbation_type,
            self.split_method,
            f"f{self.fold}",
            self.endpoint_loss,
            f"g{self.gamma:g}",
            self.noise_type,
            self.optimizer,
            f"lr{self.lr:g}",
            f"s{self.seed}",
            self.run_tag(),
        ])
        return os.path.join(self.result_path, name)