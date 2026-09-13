"""Load frozen Phase 1 notebook code (model + preprocessing) from disk."""

from __future__ import annotations

import importlib.util
import io
import inspect
import json
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_PHASE1 = PROJECT_ROOT / "src" / "_extracted_phase1.py"
PHASE1_CONFIG = PROJECT_ROOT / "models" / "phase1" / "config" / "phase1_config.json"
CHECKPOINT_BEST = PROJECT_ROOT / "models" / "phase1" / "checkpoints" / "best_model.pt"
PREPROCESSING_STATE = (
    PROJECT_ROOT / "models" / "phase1" / "preprocessing" / "preprocessing_state.json"
)


@lru_cache(maxsize=1)
def load_phase1_module() -> Any:
    if not EXTRACTED_PHASE1.is_file():
        raise FileNotFoundError(f"Missing Phase 1 source: {EXTRACTED_PHASE1}")
    spec = importlib.util.spec_from_file_location("phase1_extracted", EXTRACTED_PHASE1)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create module spec for Phase 1 extract.")
    module = importlib.util.module_from_spec(spec)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        spec.loader.exec_module(module)
    return module


def load_phase1_config(module: Any | None = None):
    module = module or load_phase1_module()
    raw = json.loads(PHASE1_CONFIG.read_text(encoding="utf-8"))
    sig = inspect.signature(module.Phase1Config)
    filtered = {k: v for k, v in raw.items() if k in sig.parameters}
    config = module.Phase1Config(**filtered)
    config.validate()
    return config


def build_phase1_model(
    device: torch.device | None = None,
    module: Any | None = None,
) -> torch.nn.Module:
    module = module or load_phase1_module()
    config = load_phase1_config(module)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = module.Phase1QuantilePredictor(
        config=config,
        node_input_dim=len(module.MODEL_NODE_FEATURES),
        edge_dim=len(module.MODEL_EDGE_FEATURES),
        num_core_types=module.core_type_encoder["num_categories"],
        numerical_context_dim=len(module.MODEL_CONTEXT_FEATURES) - 1,
        core_embedding_dim=8,
    )
    if not CHECKPOINT_BEST.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_BEST}")

    checkpoint = torch.load(CHECKPOINT_BEST, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_preprocessing_state() -> dict:
    return json.loads(PREPROCESSING_STATE.read_text(encoding="utf-8"))
