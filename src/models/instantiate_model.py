from __future__ import annotations

import inspect

from src.models.origin.blocks import parse_block_spec
from src.models.origin.model import PerturbationFlowModel

PRESETS = {
    "origin": {
        "blocks": "graph*8",
        "refiner": "graph",
    },
    "differential_transformer": {
        "expression_encoder": "scalar",
        "time_scale": 1.0,
        "input_fusion": "gene_added",
        "gene_reinjection": True,
        "perturbation_pooling": "mean_all",
        "injection": ("adapter", "decoder"),
        "vocabulary_interaction": True,
        "blocks": "differential_transformer:residual_scale=1.0:second_source=control*8",
        "refiner": "none",
        "decoder": "mlp",
    },
}

ALIASES = {"ntoken": "num_tokens", "d_model": "dim", "d_hid": "hidden_dim", "nhead": "num_heads"}


def parse_value(text: str):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_block_text(text: str) -> tuple[dict, int]:
    """Parse ``name[:key=value]*[*count]``, for example ``graph:use_global=false*4``."""
    body, count = text.strip(), 1
    if "*" in body:
        body, repeat = body.rsplit("*", 1)
        count = int(repeat)
    name, *pairs = body.split(":")
    spec = {"type": name.strip()}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(f"Block option '{pair}' in '{text}' must have the form key=value")
        spec[key.strip()] = parse_value(value.strip())
    return spec, count


def parse_block_specs(text: str) -> list[dict]:
    """Parse comma-separated block segments, for example ``graph*4,transformer:qk_norm=true*4``."""
    specs = []
    for segment in text.split(","):
        if segment.strip():
            spec, count = parse_block_text(segment)
            specs.extend(dict(spec) for _ in range(count))
    if not specs:
        raise ValueError(f"No blocks found in '{text}'")
    return specs


def normalize_spec(spec) -> dict:
    name, options = parse_block_spec(spec)
    return {"type": name, **options}


def resolve_blocks(value) -> list[dict]:
    if isinstance(value, str):
        value = [value]
    specs = []
    for item in value:
        if isinstance(item, str):
            specs.extend(parse_block_specs(item))
        else:
            specs.append(normalize_spec(item))
    return specs


def resolve_refiner(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "none":
            return None
        specs = parse_block_specs(value)
        if len(specs) != 1:
            raise ValueError(f"refiner must be a single block, got '{value}'")
        return specs[0]
    return normalize_spec(value)


def resize_blocks(blocks: list[dict], count: int) -> list[dict]:
    if len(blocks) == count:
        return blocks
    if any(spec != blocks[0] for spec in blocks):
        raise ValueError("num_blocks cannot resize a heterogeneous block list; pass blocks explicitly")
    return [dict(blocks[0]) for _ in range(count)]


def parse_list(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


def instantiate_model(model_type: str, **kwargs) -> PerturbationFlowModel:
    """Build a model from a preset plus keyword overrides.

    Keyword values of None are treated as not provided. ``blocks`` and ``refiner`` accept
    a spec string (see ``parse_block_specs``), a list of names or option mappings, and
    ``refiner="none"`` removes a preset refiner. ``num_blocks`` resizes a homogeneous preset
    and is ignored when ``blocks`` is given. ``neighbor_gate`` is the default for every
    graph block that does not set it in its own spec.
    """
    if model_type not in PRESETS:
        raise ValueError(f"Invalid model type: {model_type}, available: {sorted(PRESETS)}")

    explicit_blocks = kwargs.get("blocks") is not None
    explicit_injection = kwargs.get("injection") is not None
    num_blocks = kwargs.pop("num_blocks", None)
    neighbor_gate = kwargs.pop("neighbor_gate", None)

    config = dict(PRESETS[model_type])
    for key, value in kwargs.items():
        if value is not None:
            config[ALIASES.get(key, key)] = value

    blocks = resolve_blocks(config["blocks"])
    if num_blocks is not None and not explicit_blocks:
        blocks = resize_blocks(blocks, int(num_blocks))
    refiner = resolve_refiner(config.get("refiner"))
    if neighbor_gate is not None:
        for spec in blocks + ([refiner] if refiner is not None else []):
            if spec["type"] == "graph":
                spec.setdefault("neighbor_gate", bool(neighbor_gate))
    config["blocks"] = blocks
    config["refiner"] = refiner

    parameters = inspect.signature(PerturbationFlowModel.__init__).parameters
    injection = parse_list(config.get("injection", parameters["injection"].default))
    if config.get("perturbation_type") == "label" and not explicit_injection:
        injection = tuple(site for site in injection if site != "target_node")
    config["injection"] = injection

    unknown = sorted(set(config) - (set(parameters) - {"self"}))
    if unknown:
        raise ValueError(f"Unknown model arguments {unknown}, available: {sorted(set(parameters) - {'self'})}")
    return PerturbationFlowModel(**config)