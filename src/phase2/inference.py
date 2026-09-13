"""
Frozen Phase 1 inference for Phase 2.

Runs the frozen best_model.pt over any split (calibration / test_id / test_ood)
using the exact GraphScenarioDataset / preprocessing pipeline extracted from the
Phase 1 notebook (src/_extracted_phase1.py), and returns one row per runtime
record with:

    - identity columns (sample_id, graph_id, target_node)
    - hardware/DVFS/z_t context columns (evaluation only, not re-derived)
    - y_exec_us                         ground truth, microseconds
    - Q50_pred_us / Q90_pred_us / Q95_pred_us / Q99_pred_us   raw model output
    - criticality, deadline_us, period_us  (joined from tasks/task_metadata.csv)

This module never fits/refits any preprocessing and never updates model weights.
It is a pure frozen-inference pass, matching the Phase 2 Handoff Guide section 14.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from phase1.bootstrap import (
    PROJECT_ROOT,
    build_phase1_model,
    load_phase1_module,
    load_preprocessing_state,
)
from phase2.target_inverse import inverse_transform_target

TASK_METADATA_PATH = (
    PROJECT_ROOT / "data" / "dag_runtime_dataset_25k" / "tasks" / "task_metadata.csv"
)

_QUANTILE_COLUMNS = ["Q50_pred_us", "Q90_pred_us", "Q95_pred_us", "Q99_pred_us"]


def load_task_metadata() -> pd.DataFrame:
    """graph_id, split, criticality, deadline_us, period_us (one row per DAG)."""
    df = pd.read_csv(TASK_METADATA_PATH)
    df["graph_id"] = df["graph_id"].astype(str)
    return df


@torch.no_grad()
def _infer_one_graph(model, graph_item, preprocessing_state, device) -> pd.DataFrame:
    """Re-implementation of the notebook's infer_one_graph, importable as a function."""
    x = graph_item["x"].to(device)
    edge_index = graph_item["edge_index"].to(device)
    edge_attr = graph_item["edge_attr"].to(device)
    target_node_indices = graph_item["target_node_indices"].to(device)
    core_type_indices = graph_item["core_type_indices"].to(device)
    context_numeric = graph_item["context_numeric"].to(device)

    predictions_standardized = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_node_indices=target_node_indices,
        core_type_indices=core_type_indices,
        context_numeric=context_numeric,
    )

    num_contexts = int(target_node_indices.shape[0])
    if tuple(predictions_standardized.shape) != (num_contexts, 4):
        raise RuntimeError(
            f"Unexpected prediction shape {tuple(predictions_standardized.shape)} "
            f"for graph {graph_item['graph_id']}"
        )
    if not torch.isfinite(predictions_standardized).all():
        raise RuntimeError(f"NaN/Inf raw predictions for graph {graph_item['graph_id']}")

    predictions_us = (
        inverse_transform_target(predictions_standardized, preprocessing_state)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if not np.isfinite(predictions_us).all():
        raise RuntimeError(f"NaN/Inf inverse-transformed predictions for graph {graph_item['graph_id']}")

    metadata = graph_item["metadata"]
    result_df = metadata.copy().reset_index(drop=True)
    if len(result_df) != num_contexts:
        raise RuntimeError("metadata/context count mismatch")

    result_df["y_exec_us"] = graph_item["y_raw_us"].detach().cpu().numpy().astype(np.float64)
    for i, col in enumerate(_QUANTILE_COLUMNS):
        result_df[col] = predictions_us[:, i]

    crossing = (
        (result_df["Q50_pred_us"] > result_df["Q90_pred_us"])
        | (result_df["Q90_pred_us"] > result_df["Q95_pred_us"])
        | (result_df["Q95_pred_us"] > result_df["Q99_pred_us"])
    )
    if crossing.any():
        raise RuntimeError(f"Quantile crossing detected for graph {graph_item['graph_id']}")

    if "graph_id" not in result_df.columns:
        result_df["graph_id"] = graph_item["graph_id"]

    return result_df


def run_frozen_inference(
    split: str,
    device: torch.device | None = None,
    model: torch.nn.Module | None = None,
    limit_graphs: int | None = None,
) -> pd.DataFrame:
    """
    Run frozen Phase 1 inference over an entire split ("calibration", "test_id",
    or "test_ood"). Returns one DataFrame row per runtime record (42 * n_graphs).
    """
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model = build_phase1_model(device=device, module=module)
    model.eval()

    records_path = module.RECORDS_DIR / f"{split}.csv"
    records_df = pd.read_csv(records_path)

    dataset = module.GraphScenarioDataset(
        records_df=records_df,
        split=split,
        preprocessing_state=preprocessing_state,
        expected_records_per_graph=42,
        cache_static_graphs=False,
    )

    n_graphs = len(dataset) if limit_graphs is None else min(limit_graphs, len(dataset))
    frames = []
    for i in range(n_graphs):
        graph_item = dataset[i]
        frames.append(_infer_one_graph(model, graph_item, preprocessing_state, device))

    predictions_df = pd.concat(frames, ignore_index=True)

    task_meta = load_task_metadata()[["graph_id", "criticality", "deadline_us", "period_us"]]
    predictions_df["graph_id"] = predictions_df["graph_id"].astype(str)
    predictions_df = predictions_df.merge(task_meta, on="graph_id", how="left", validate="many_to_one")
    if predictions_df["criticality"].isna().any():
        missing = predictions_df.loc[predictions_df["criticality"].isna(), "graph_id"].unique()
        raise RuntimeError(f"Missing criticality join for graph_ids: {missing[:5]}")

    predictions_df["split"] = split
    return predictions_df


def save_predictions(df: pd.DataFrame, split: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (PROJECT_ROOT / "results" / "phase2" / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_raw_predictions.csv"
    df.to_csv(out_path, index=False)
    return out_path
