"""
Train MLP and GNN-Mean baselines (reduced budget, from-scratch) and compare
them against the frozen GNN-Quantile model (raw and calibrated) on test_id and
test_ood. Produces results/phase2/baselines/model_comparison.csv.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from phase1.bootstrap import load_phase1_module, load_preprocessing_state  # noqa: E402
from phase2.baselines import GNNMeanBaseline, MLPBaseline, train_point_baseline  # noqa: E402
from phase2.target_inverse import inverse_transform_target  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAL_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"

torch.manual_seed(20260807)
np.random.seed(20260807)


@torch.no_grad()
def evaluate_point_model(model, dataset, preprocessing_state, device, limit_graphs=None) -> pd.DataFrame:
    n = len(dataset) if limit_graphs is None else min(limit_graphs, len(dataset))
    frames = []
    for i in range(n):
        item = dataset[i]
        pred_standardized = model(
            x=item["x"].to(device), edge_index=item["edge_index"].to(device),
            edge_attr=item["edge_attr"].to(device),
            target_node_indices=item["target_node_indices"].to(device),
            core_type_indices=item["core_type_indices"].to(device),
            context_numeric=item["context_numeric"].to(device),
        )
        pred_us = inverse_transform_target(pred_standardized, preprocessing_state).cpu().numpy()
        meta = item["metadata"].copy().reset_index(drop=True)
        meta["graph_id"] = item["graph_id"]
        meta["y_exec_us"] = item["y_raw_us"].cpu().numpy()
        meta["pred_us"] = pred_us
        frames.append(meta)
    return pd.concat(frames, ignore_index=True)


def point_metrics(df: pd.DataFrame) -> dict:
    y = df["y_exec_us"].to_numpy(dtype=np.float64)
    p = df["pred_us"].to_numpy(dtype=np.float64)
    err = p - y
    return {
        "n": len(df),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.maximum(y, 1e-9)) * 100),
    }


def main() -> None:
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")

    node_input_dim = len(module.MODEL_NODE_FEATURES)
    edge_dim = len(module.MODEL_EDGE_FEATURES)
    num_core_types = module.core_type_encoder["num_categories"]
    numerical_context_dim = len(module.MODEL_CONTEXT_FEATURES) - 1

    print("=== Training MLP baseline (no graph structure) ===")
    mlp = MLPBaseline(node_input_dim, num_core_types, numerical_context_dim)
    t0 = time.time()
    mlp_history = train_point_baseline(
        mlp, module.train_dataset, module.validation_dataset,
        epochs=8, train_graph_subset=150, val_graph_subset=30, device=device, log_prefix="[MLP] ",
    )
    print(f"MLP training done in {time.time()-t0:.1f}s, best_val_mse={mlp_history['best_val_mse']:.4f}")
    pd.DataFrame(mlp_history["history"]).to_csv(RESULTS_DIR / "mlp_training_history.csv", index=False)

    print("\n=== Training GNN-Mean baseline ===")
    gnn_mean = GNNMeanBaseline(module, node_input_dim, edge_dim, num_core_types, numerical_context_dim)
    t0 = time.time()
    gnn_history = train_point_baseline(
        gnn_mean, module.train_dataset, module.validation_dataset,
        epochs=8, train_graph_subset=150, val_graph_subset=30, device=device, log_prefix="[GNN-Mean] ",
    )
    print(f"GNN-Mean training done in {time.time()-t0:.1f}s, best_val_mse={gnn_history['best_val_mse']:.4f}")
    pd.DataFrame(gnn_history["history"]).to_csv(RESULTS_DIR / "gnn_mean_training_history.csv", index=False)

    print("\n=== Training GNN-Mean-NoZt ablation (offline, z_t blinded) ===")
    gnn_mean_no_zt = GNNMeanBaseline(module, node_input_dim, edge_dim, num_core_types,
                                      numerical_context_dim, ablate_zt=True)
    t0 = time.time()
    noz_history = train_point_baseline(
        gnn_mean_no_zt, module.train_dataset, module.validation_dataset,
        epochs=8, train_graph_subset=150, val_graph_subset=30, device=device, log_prefix="[GNN-Mean-NoZt] ",
    )
    print(f"GNN-Mean-NoZt training done in {time.time()-t0:.1f}s, best_val_mse={noz_history['best_val_mse']:.4f}")
    pd.DataFrame(noz_history["history"]).to_csv(RESULTS_DIR / "gnn_mean_no_zt_training_history.csv", index=False)

    torch.save(mlp.state_dict(), RESULTS_DIR / "mlp_baseline.pt")
    torch.save(gnn_mean.state_dict(), RESULTS_DIR / "gnn_mean_baseline.pt")
    torch.save(gnn_mean_no_zt.state_dict(), RESULTS_DIR / "gnn_mean_no_zt_baseline.pt")

    rows = []
    for split_name, dataset in [("test_id", module.test_id_dataset)]:
        mlp_pred = evaluate_point_model(mlp, dataset, preprocessing_state, device)
        mlp_pred.to_csv(RESULTS_DIR / f"mlp_predictions_{split_name}.csv", index=False)
        rows.append({"model": "MLP (no graph)", "eval_split": split_name, **point_metrics(mlp_pred)})

        gnn_pred = evaluate_point_model(gnn_mean, dataset, preprocessing_state, device)
        gnn_pred.to_csv(RESULTS_DIR / f"gnn_mean_predictions_{split_name}.csv", index=False)
        rows.append({"model": "GNN-Mean", "eval_split": split_name, **point_metrics(gnn_pred)})

        noz_pred = evaluate_point_model(gnn_mean_no_zt, dataset, preprocessing_state, device)
        noz_pred.to_csv(RESULTS_DIR / f"gnn_mean_no_zt_predictions_{split_name}.csv", index=False)
        rows.append({"model": "GNN-Mean-NoZt (offline, z_t blinded)", "eval_split": split_name,
                     **point_metrics(noz_pred)})

    # Frozen GNN-Quantile (Q50 as its point estimate), raw and calibrated, for comparison.
    frozen_test_id = pd.read_csv(CAL_DIR / "calibrated_test_id.csv")
    frozen_test_id = frozen_test_id.rename(columns={"Q50_pred_us": "pred_us"})
    rows.append({"model": "GNN-Quantile (frozen, Q50, uncalibrated)", "eval_split": "test_id",
                 **point_metrics(frozen_test_id)})

    comparison = pd.DataFrame(rows)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n=== Model comparison (test_id) -> {comparison_path} ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
