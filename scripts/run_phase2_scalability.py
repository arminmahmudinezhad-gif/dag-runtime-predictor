"""
4 / 8 / 16 / 32-core scalability stress test.

For each configuration in CORE_SCALABILITY, override active_core_count on the
real test_id evaluation contexts (everything else held fixed), run frozen
inference, apply the already-fitted (frozen) conformal correction, and report
how the predicted distribution / calibrated budget / prediction "uncertainty"
(Q99 - Q50 spread) shifts as active_core_count is pushed further outside the
[1, 4] training range. See src/phase2/scalability.py module docstring for why
this is the right way to interpret >4-core "scalability" for this frozen model.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from phase1.bootstrap import build_phase1_model, load_phase1_module, load_preprocessing_state  # noqa: E402
from phase2.conformal import CriticalityCalibration, apply_conformal  # noqa: E402
from phase2.config import CORE_SCALABILITY, DEFAULT_CONFORMAL  # noqa: E402
from phase2.scalability import run_scalability_scenario  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "scalability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
            scores_us=None,  # not needed for apply_conformal
        )
    return state


def main() -> None:
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")
    model = build_phase1_model(device=device, module=module)

    base_records_df = pd.read_csv(module.TEST_ID_RECORDS_PATH)
    calibration_state = load_frozen_calibration_state()

    # Join criticality for conformal application.
    from phase2.inference import load_task_metadata
    task_meta = load_task_metadata()[["graph_id", "criticality"]]
    base_records_df["graph_id"] = base_records_df["graph_id"].astype(str)

    summary_rows = []
    for scenario in CORE_SCALABILITY:
        active = scenario["active_core_count"]
        print(f"Running scenario '{scenario['label']}' "
              f"(m_big={scenario['mbig']}, m_little={scenario['mlittle']}, "
              f"active_core_count={active}) ...")

        result = run_scalability_scenario(
            base_records_df=base_records_df,
            split="test_id",
            active_core_count=active,
            device=device,
            model=model,
            module=module,
            preprocessing_state=preprocessing_state,
        )
        result["graph_id"] = result["graph_id"].astype(str)
        result = result.merge(task_meta, on="graph_id", how="left", validate="many_to_one")

        # y_exec_us here is the ORIGINAL (active_core_count<=4) ground truth; it is not
        # valid ground truth for the overridden OOD active_core_count scenario, so it must
        # not be used for pinball/coverage/accuracy claims. Keep it only for audit under an
        # explicit name, and exclude it from apply_conformal's coverage bookkeeping.
        original_y = result.pop("y_exec_us")
        result = apply_conformal(result, calibration_state, DEFAULT_CONFORMAL)
        result["y_exec_us_original_context_NOT_GROUND_TRUTH_for_this_scenario"] = original_y
        result["uncertainty_spread_us"] = result["Q99_pred_us"] - result["Q50_pred_us"]

        out_path = RESULTS_DIR / f"{scenario['label']}_metrics.csv"
        result.to_csv(out_path, index=False)

        row = {
            "scenario": scenario["label"],
            "m_big": scenario["mbig"],
            "m_little": scenario["mlittle"],
            "active_core_count": active,
            "active_core_count_zscore": float(result["active_core_count_zscore"].iloc[0]),
            "n": len(result),
            "mean_Q50_us": float(result["Q50_pred_us"].mean()),
            "mean_Q95_us": float(result["Q95_pred_us"].mean()),
            "mean_Q99_us": float(result["Q99_pred_us"].mean()),
            "mean_C_e_us": float(result["C_e_us"].mean()),
            "mean_uncertainty_spread_us": float(result["uncertainty_spread_us"].mean()),
        }
        summary_rows.append(row)
        print(f"  -> mean_Q50={row['mean_Q50_us']:.1f}us  mean_C_e={row['mean_C_e_us']:.1f}us  "
              f"uncertainty_spread={row['mean_uncertainty_spread_us']:.1f}us  "
              f"z(active_core_count)={row['active_core_count_zscore']:.2f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "core_scalability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved scalability summary -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
