"""
OOD generalization + graph-structure sensitivity analysis (spec sections 4 & 5).

Uses the frozen raw predictions already produced by run_phase2_inference.py and the
calibrated budgets from run_phase2_calibration.py. Does not re-run the model.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase2.graph_structure import (  # noqa: E402
    attach_critical_path_flag,
    attach_graph_structure,
    bin_structural_columns,
)
from phase2.metrics import grouped_metrics, summarize  # noqa: E402

CAL_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"
OOD_DIR = PROJECT_ROOT / "results" / "phase2" / "ood"
OOD_DIR.mkdir(parents=True, exist_ok=True)


_SPLIT_FILES = {
    "calibration": "calibration_predictions.csv",
    "test_id": "calibrated_test_id.csv",
    "test_ood": "calibrated_test_ood.csv",
}


def enrich(split: str) -> pd.DataFrame:
    df = pd.read_csv(CAL_DIR / _SPLIT_FILES[split])
    df = attach_graph_structure(df)
    df = attach_critical_path_flag(df, split)
    df = bin_structural_columns(df)
    return df


def main() -> None:
    test_id = enrich("test_id")
    test_ood = enrich("test_ood")

    # --------------------------------------------------------------
    # 1. Global test-ID vs test-OOD comparison
    # --------------------------------------------------------------
    global_compare = pd.DataFrame(
        [
            {"eval_split": "test_id", **summarize(test_id)},
            {"eval_split": "test_ood", **summarize(test_ood)},
        ]
    )
    global_compare.to_csv(OOD_DIR / "test_id_vs_test_ood_global_metrics.csv", index=False)
    print("=== Global test_id vs test_ood ===")
    print(global_compare.to_string(index=False))

    with open(OOD_DIR / "test_ood_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summarize(test_ood), f, indent=2)

    # --------------------------------------------------------------
    # 2. Grouped diagnostics (test_ood), several dimensions
    # --------------------------------------------------------------
    group_cols = [
        "core_type", "dvfs_level", "criticality", "target_role",
        "num_nodes_bin", "fat_bin", "density_parameter_bin", "regular_bin",
        "is_critical_path_node",
    ]
    grouped_frames = {}
    for col in group_cols:
        g = grouped_metrics(test_ood, col)
        g.to_csv(OOD_DIR / f"grouped_ood_metrics_by_{col}.csv", index=False)
        grouped_frames[col] = g
        print(f"\n=== test_ood grouped by {col} ===")
        print(g[[col, "n", "mean_pinball", "mae", "coverage_Q95", "coverage_Q99"]].to_string(index=False))

    # Same grouped diagnostics on test_id for a like-for-like baseline comparison.
    for col in group_cols:
        g = grouped_metrics(test_id, col)
        g.to_csv(OOD_DIR / f"grouped_test_id_metrics_by_{col}.csv", index=False)

    # --------------------------------------------------------------
    # 3. Critical-path vs non-critical-path accuracy (combined table)
    # --------------------------------------------------------------
    cp_compare = pd.concat(
        [
            grouped_metrics(test_id, "is_critical_path_node").assign(eval_split="test_id"),
            grouped_metrics(test_ood, "is_critical_path_node").assign(eval_split="test_ood"),
        ],
        ignore_index=True,
    )
    cp_compare.to_csv(OOD_DIR / "critical_path_accuracy_comparison.csv", index=False)

    # --------------------------------------------------------------
    # 4. Edge-feature / communication-volume influence: communication_heavy nodes
    #    vs the rest, plus correlation between input_size context volume proxy
    #    (target_role) and pinball/coverage.
    # --------------------------------------------------------------
    comm_compare = pd.concat(
        [
            grouped_metrics(test_id, "target_role").assign(eval_split="test_id"),
            grouped_metrics(test_ood, "target_role").assign(eval_split="test_ood"),
        ],
        ignore_index=True,
    )
    comm_compare.to_csv(OOD_DIR / "communication_role_accuracy.csv", index=False)

    # --------------------------------------------------------------
    # 5. Out-of-training-range standardized feature report (num_nodes, depth, width)
    # --------------------------------------------------------------
    train_summary = pd.read_csv(PROJECT_ROOT / "data" / "dag_runtime_dataset_25k" / "metadata" / "graph_summary.csv")
    train_summary = train_summary[train_summary["split"] == "train"]
    range_report = pd.DataFrame(
        {
            "feature": ["num_nodes", "depth", "max_width"],
            "train_min": [train_summary[c].min() for c in ["num_nodes", "depth", "max_width"]],
            "train_max": [train_summary[c].max() for c in ["num_nodes", "depth", "max_width"]],
            "test_ood_min": [test_ood[c].min() for c in ["num_nodes", "depth", "max_width"]],
            "test_ood_max": [test_ood[c].max() for c in ["num_nodes", "depth", "max_width"]],
            "fraction_test_ood_out_of_train_range": [
                float(np.mean((test_ood[c] < train_summary[c].min()) | (test_ood[c] > train_summary[c].max())))
                for c in ["num_nodes", "depth", "max_width"]
            ],
        }
    )
    range_report.to_csv(OOD_DIR / "structural_feature_range_report.csv", index=False)
    print("\n=== Structural OOD range report ===")
    print(range_report.to_string(index=False))

    # --------------------------------------------------------------
    # 6. Accuracy vs graph size (regression-style trend for report narrative)
    # --------------------------------------------------------------
    combined = pd.concat([test_id.assign(eval_split="test_id"), test_ood.assign(eval_split="test_ood")])
    size_trend = grouped_metrics(combined, "num_nodes_bin")
    size_trend.to_csv(OOD_DIR / "accuracy_vs_graph_size.csv", index=False)


if __name__ == "__main__":
    main()
