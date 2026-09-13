"""
Graph-structure utilities for Phase 2 structural-sensitivity analysis.

Reuses the already-computed per-graph structural summary
(data/dag_runtime_dataset_25k/metadata/graph_summary.csv), which already contains
num_nodes, num_edges, depth, max_width, actual_density, ccr, and the three
generator "shape" parameters requested by the spec (fat, density_parameter,
regular), each varied in [0.2, 0.8].

Also derives a per-node "on critical path" proxy from topo_level/reverse_topo_level:
a node v lies on (one of) the longest source-to-sink path(s) of its DAG iff
    topo_level(v) + reverse_topo_level(v) == depth(graph) - 1
This is a topological (hop-count) longest-path proxy, not a compute-cycle-weighted
longest path; it is used only for grouping/correlation, not for scheduling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phase1.bootstrap import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data" / "dag_runtime_dataset_25k"
GRAPH_SUMMARY_PATH = DATA_ROOT / "metadata" / "graph_summary.csv"
NODE_FEATURES_DIR = DATA_ROOT / "features" / "nodes"


def load_graph_summary() -> pd.DataFrame:
    df = pd.read_csv(GRAPH_SUMMARY_PATH)
    df["graph_id"] = df["graph_id"].astype(str)
    return df


def attach_graph_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join graph_summary.csv structural columns onto a predictions table."""
    summary = load_graph_summary()[
        [
            "graph_id",
            "num_nodes",
            "num_edges",
            "depth",
            "max_width",
            "actual_density",
            "ccr_target",
            "ccr_actual",
            "fat",
            "density_parameter",
            "regular",
            "jump",
        ]
    ]
    out = df.copy()
    out["graph_id"] = out["graph_id"].astype(str)
    out = out.merge(summary, on="graph_id", how="left", validate="many_to_one")
    return out


def critical_path_node_ids(split: str, graph_id: str) -> set[str]:
    """Return the set of node_ids lying on a longest source-to-sink path (proxy)."""
    node_path = NODE_FEATURES_DIR / split / f"{graph_id}.csv"
    node_df = pd.read_csv(node_path)
    depth_row = load_graph_summary()
    depth_row = depth_row.loc[depth_row["graph_id"] == graph_id, "depth"]
    if depth_row.empty:
        raise KeyError(f"graph_id '{graph_id}' not found in graph_summary.csv")
    depth = int(depth_row.iloc[0])

    on_path = node_df["topo_level"] + node_df["reverse_topo_level"] == depth - 1
    return set(node_df.loc[on_path, "node_id"].astype(str))


def attach_critical_path_flag(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Add an `is_critical_path_node` boolean column to a predictions table for one split.
    Groups by graph_id internally so each graph's node CSV is only read once.
    """
    out = df.copy()
    out["graph_id"] = out["graph_id"].astype(str)
    out["target_node"] = out["target_node"].astype(str)

    flags = pd.Series(False, index=out.index)
    for graph_id, sub in out.groupby("graph_id"):
        cp_nodes = critical_path_node_ids(split, graph_id)
        flags.loc[sub.index] = sub["target_node"].isin(cp_nodes)
    out["is_critical_path_node"] = flags
    return out


def attach_node_operation_type(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Join each row's target-node `operation_type` (compute_bound / balanced /
    memory_bound) from the static node CSVs. Used to test whether the model
    learned genuinely different, non-linear behavior on compute-bound vs.
    memory-bound nodes (spec section 3).
    """
    out = df.copy()
    out["graph_id"] = out["graph_id"].astype(str)
    out["target_node"] = out["target_node"].astype(str)

    op_type = pd.Series(index=out.index, dtype="object")
    for graph_id, sub in out.groupby("graph_id"):
        node_path = NODE_FEATURES_DIR / split / f"{graph_id}.csv"
        node_df = pd.read_csv(node_path)
        node_df["node_id"] = node_df["node_id"].astype(str)
        lookup = node_df.set_index("node_id")["operation_type"]
        op_type.loc[sub.index] = sub["target_node"].map(lookup)
    out["operation_type"] = op_type
    return out


def bin_structural_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bin fat/density_parameter/regular (each generated in [0.2, 0.8]) into terciles."""
    out = df.copy()
    edges = [0.15, 0.35, 0.55, 0.85]
    labels = ["low(0.2)", "mid(0.4-0.6)", "high(0.8)"]
    for col in ["fat", "density_parameter", "regular"]:
        out[f"{col}_bin"] = pd.cut(out[col], bins=edges, labels=labels, include_lowest=True)
    out["num_nodes_bin"] = pd.cut(
        out["num_nodes"],
        bins=[0, 100, 200, 300, 400, 500, 700, 1000],
        labels=["<=100", "101-200", "201-300", "301-400", "401-500", "501-700", "701-1000"],
        include_lowest=True,
    )
    return out
