"""
Closes three specific gaps identified in a requirement-by-requirement audit
against the spec's section 3/4 bullets:

  (a) grouped accuracy by operation_type (compute_bound / balanced /
      memory_bound) -- "non-linear behavior in compute-bound vs memory-bound
      nodes" (spec section 3)
  (b) inference time vs. graph size, 50-1000 nodes (spec section 4, "training
      time, inference time, and accuracy" -- training time is N/A since the
      Phase 1 predictor is frozen and never retrained per graph size)
  (c) GNN-embedding-based correlation with prediction accuracy on
      critical-path vs. non-critical-path nodes (spec section 4, "correlation
      between GNN-extracted features and prediction accuracy"), going beyond
      the topological critical-path proxy used elsewhere
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from phase1.bootstrap import build_phase1_model, load_phase1_module, load_preprocessing_state  # noqa: E402
from phase2.embeddings import embedding_features_for_predictions  # noqa: E402
from phase2.graph_structure import attach_critical_path_flag, attach_graph_structure, attach_node_operation_type  # noqa: E402
from phase2.inference import _infer_one_graph  # noqa: E402
from phase2.metrics import grouped_metrics, pinball_loss  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "ood"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAL_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"


def part_a_operation_type():
    print("=== (a) Grouped accuracy by operation_type (compute/balanced/memory-bound) ===")
    for split in ["test_id", "test_ood"]:
        df = pd.read_csv(CAL_DIR / f"calibrated_{split}.csv")
        df = attach_node_operation_type(df, split)
        g = grouped_metrics(df, "operation_type")
        g.to_csv(RESULTS_DIR / f"grouped_{split}_metrics_by_operation_type.csv", index=False)
        print(f"\n-- {split} --")
        print(g[["operation_type", "n", "mean_pinball", "mae", "coverage_Q95"]].to_string(index=False))


def part_b_inference_time_vs_size():
    print("\n=== (b) Inference time vs. graph size (50-1000 nodes) ===")
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")
    model = build_phase1_model(device=device, module=module)

    graph_summary = pd.read_csv(PROJECT_ROOT / "data" / "dag_runtime_dataset_25k" / "metadata" / "graph_summary.csv")
    rows = []
    for split in ["test_id", "test_ood"]:
        split_records = pd.read_csv(module.RECORDS_DIR / f"{split}.csv")
        for graph_id in sorted(split_records["graph_id"].unique().astype(str)):
            graph_records = split_records[split_records["graph_id"].astype(str) == graph_id]
            static_graph = module.load_static_graph(graph_id=graph_id, split=split,
                                                      preprocessing_state=preprocessing_state)
            num_nodes = static_graph["x"].shape[0]

            # Build one dataset item directly (mirrors GraphScenarioDataset.__getitem__ cost)
            dataset = module.GraphScenarioDataset(
                records_df=graph_records, split=split, preprocessing_state=preprocessing_state,
                expected_records_per_graph=42, cache_static_graphs=False,
            )
            item = dataset[0]

            t0 = time.perf_counter()
            _infer_one_graph(model, item, preprocessing_state, device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            rows.append({"split": split, "graph_id": graph_id, "num_nodes": num_nodes,
                         "inference_time_ms": elapsed_ms})

    timing = pd.DataFrame(rows)
    timing = timing.merge(graph_summary[["graph_id", "num_edges"]], on="graph_id", how="left")
    timing.to_csv(RESULTS_DIR / "inference_time_vs_graph_size.csv", index=False)

    corr = timing["num_nodes"].corr(timing["inference_time_ms"])
    print(f"Pearson correlation(num_nodes, inference_time_ms) = {corr:.3f}")
    print(timing.groupby("split")["inference_time_ms"].describe().to_string())

    # Training time note: N/A here by design -- the Phase 1 predictor is frozen
    # and is never retrained per graph size in Phase 2 (see PHASE2_REPORT.md).


def part_c_embedding_correlation():
    print("\n=== (c) GNN-embedding correlation with prediction accuracy ===")
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")
    model = build_phase1_model(device=device, module=module)

    for split in ["test_id", "test_ood"]:
        df = pd.read_csv(CAL_DIR / f"calibrated_{split}.csv")
        df = attach_critical_path_flag(df, split)
        df = attach_graph_structure(df)

        df = embedding_features_for_predictions(model, module, preprocessing_state, device, df, split)

        y = df["y_exec_us"].to_numpy(dtype=np.float64)
        q95 = df["Q95_pred_us"].to_numpy(dtype=np.float64)
        df["pinball_q95"] = pinball_loss(y, q95, 0.95)
        df["abs_error_q50"] = np.abs(df["Q50_pred_us"] - df["y_exec_us"])

        corr_table = pd.DataFrame(
            {
                "metric": ["pinball_q95", "abs_error_q50"],
                "corr_with_embedding_norm": [
                    df["embedding_norm"].corr(df["pinball_q95"]),
                    df["embedding_norm"].corr(df["abs_error_q50"]),
                ],
                "corr_with_embedding_dist_from_graph_mean": [
                    df["embedding_dist_from_graph_mean"].corr(df["pinball_q95"]),
                    df["embedding_dist_from_graph_mean"].corr(df["abs_error_q50"]),
                ],
            }
        )
        corr_table.to_csv(RESULTS_DIR / f"embedding_error_correlation_{split}.csv", index=False)
        print(f"\n-- {split} --")
        print(corr_table.to_string(index=False))

        by_cp = df.groupby("is_critical_path_node")[["embedding_norm", "embedding_dist_from_graph_mean",
                                                       "pinball_q95", "abs_error_q50"]].mean()
        by_cp.to_csv(RESULTS_DIR / f"embedding_by_critical_path_{split}.csv")
        print(by_cp.to_string())

        df[["sample_id", "graph_id", "target_node", "is_critical_path_node", "embedding_norm",
            "embedding_dist_from_graph_mean", "pinball_q95", "abs_error_q50"]].to_csv(
            RESULTS_DIR / f"embedding_features_{split}.csv", index=False
        )


if __name__ == "__main__":
    part_a_operation_type()
    part_b_inference_time_vs_size()
    part_c_embedding_correlation()
