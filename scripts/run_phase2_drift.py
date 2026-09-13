"""
Drift / overload / bursty-state robustness analysis.
See src/phase2/drift.py module docstring for the exact experimental design.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from phase1.bootstrap import build_phase1_model, load_phase1_module, load_preprocessing_state  # noqa: E402
from phase2.conformal import CriticalityCalibration, apply_conformal  # noqa: E402
from phase2.config import DEFAULT_CONFORMAL  # noqa: E402
from phase2.drift import flag_tail_stress, sliding_window_conformal, synthetic_overload_records  # noqa: E402
from phase2.inference import _infer_one_graph, load_task_metadata  # noqa: E402
from phase2.metrics import grouped_metrics  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "drift"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAL_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"
CAL_STATE_PATH = PROJECT_ROOT / "models" / "phase2" / "calibration" / "conformal_state.json"


def load_frozen_calibration_state() -> dict[str, CriticalityCalibration]:
    raw = json.loads(CAL_STATE_PATH.read_text(encoding="utf-8"))
    state = {}
    for h, d in raw.items():
        state[h] = CriticalityCalibration(
            criticality=d["criticality"],
            base_quantile=d["base_quantile"],
            delta=d["delta"],
            n_calibration=d["n_calibration"],
            k_index=d["k_index"],
            q_conf_us=d["q_conf_us"],
            fraction_zero_scores=d["fraction_zero_scores"],
            scores_us=None,
        )
    return state


def part_a_tail_stress() -> None:
    print("\n=== (a) In-range tail-load stress (real ground truth) ===")
    test_ood = pd.read_csv(CAL_DIR / "calibrated_test_ood.csv")
    # z_t raw values are not carried in GraphScenarioDataset metadata; join them back
    # from the original records file via sample_id.
    module = load_phase1_module()
    raw_records = pd.read_csv(module.RECORDS_DIR / "test_ood.csv")[
        ["sample_id", "memory_active_tasks", "bus_utilization", "thermal_pressure", "release_jitter_us"]
    ]
    test_ood = test_ood.merge(raw_records, on="sample_id", how="left", validate="one_to_one")
    test_ood = flag_tail_stress(test_ood, quantile=0.9)
    report = grouped_metrics(test_ood, "is_high_load_tail")
    report.to_csv(RESULTS_DIR / "tail_load_stress_accuracy.csv", index=False)
    print(report[["is_high_load_tail", "n", "mean_pinball", "mae", "coverage_Q95", "coverage_Q99"]]
          .to_string(index=False))


def part_b_synthetic_overload() -> None:
    print("\n=== (b) Synthetic extreme overload (descriptive, no ground truth) ===")
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")
    model = build_phase1_model(device=device, module=module)

    base_records_df = pd.read_csv(module.TEST_ID_RECORDS_PATH)
    base_records_df["graph_id"] = base_records_df["graph_id"].astype(str)

    # Train ranges (see preprocessing_state.json): memory_active_tasks in [0,4],
    # bus_utilization in [0.02,0.98], thermal_pressure in [0,0.98],
    # release_jitter_us in [0,450]. The overload scenario pushes clearly beyond all four.
    overrides = {
        "memory_active_tasks": 10,
        "bus_utilization": 1.4,
        "thermal_pressure": 1.3,
        "release_jitter_us": 900,
    }
    scenario_records = synthetic_overload_records(base_records_df, overrides)

    dataset = module.GraphScenarioDataset(
        records_df=scenario_records,
        split="test_id",
        preprocessing_state=preprocessing_state,
        expected_records_per_graph=42,
        cache_static_graphs=False,
    )
    frames = [
        _infer_one_graph(model, dataset[i], preprocessing_state, device)
        for i in range(len(dataset))
    ]
    result = pd.concat(frames, ignore_index=True)
    result["graph_id"] = result["graph_id"].astype(str)

    task_meta = load_task_metadata()[["graph_id", "criticality"]]
    result = result.merge(task_meta, on="graph_id", how="left", validate="many_to_one")
    result = result.drop(columns=["y_exec_us"])  # not valid ground truth for this scenario

    calibration_state = load_frozen_calibration_state()
    result = apply_conformal(result, calibration_state, DEFAULT_CONFORMAL)

    baseline = pd.read_csv(CAL_DIR / "calibrated_test_id.csv")

    comparison = pd.DataFrame(
        [
            {
                "scenario": "baseline_test_id",
                "mean_Q50_us": baseline["Q50_pred_us"].mean(),
                "mean_Q95_us": baseline["Q95_pred_us"].mean(),
                "mean_C_e_us": baseline["C_e_us"].mean(),
            },
            {
                "scenario": "synthetic_overload",
                "mean_Q50_us": result["Q50_pred_us"].mean(),
                "mean_Q95_us": result["Q95_pred_us"].mean(),
                "mean_C_e_us": result["C_e_us"].mean(),
            },
        ]
    )
    comparison["pct_change_vs_baseline_C_e"] = (
        (comparison["mean_C_e_us"] - comparison.loc[0, "mean_C_e_us"]) / comparison.loc[0, "mean_C_e_us"] * 100
    )
    comparison.to_csv(RESULTS_DIR / "synthetic_overload_comparison.csv", index=False)
    result.to_csv(RESULTS_DIR / "synthetic_overload_predictions.csv", index=False)
    print(comparison.to_string(index=False))


def part_d_sliding_vs_static() -> None:
    print("\n=== (d) Static vs sliding-window (causal, adaptive) calibration on test_ood ===")
    test_ood = pd.read_csv(CAL_DIR / "calibrated_test_ood.csv")
    test_ood = test_ood.sort_values("graph_id").reset_index(drop=True)

    rows = []
    for crit, delta in [("HI", DEFAULT_CONFORMAL.delta_hi), ("LO", DEFAULT_CONFORMAL.delta_lo)]:
        sub = test_ood[test_ood["criticality"] == crit].sort_values("graph_id").reset_index(drop=True)
        sliding = sliding_window_conformal(
            sub, base_quantile=DEFAULT_CONFORMAL.base_quantile, delta=delta, window_graphs=5,
        )
        sliding.to_csv(RESULTS_DIR / f"sliding_window_{crit}.csv", index=False)

        static_coverage = float(sub["calibrated_covered"].mean())
        sliding_coverage = float(sliding["sliding_covered"].mean())
        static_mean_budget = float(sub["C_e_us"].mean())
        sliding_mean_budget = float(sliding["sliding_C_e_us"].mean())
        rows.append(
            {
                "criticality": crit,
                "target_coverage": 1 - delta,
                "static_coverage": static_coverage,
                "sliding_window_coverage": sliding_coverage,
                "static_mean_budget_us": static_mean_budget,
                "sliding_window_mean_budget_us": sliding_mean_budget,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "static_vs_sliding_window_calibration.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    part_a_tail_stress()
    part_b_synthetic_overload()
    part_d_sliding_vs_static()
