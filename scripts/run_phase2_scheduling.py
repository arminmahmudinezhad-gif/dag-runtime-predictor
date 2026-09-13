"""
Schedule every test_id DAG (in isolation, on the base 4-core big/little
platform) with HEFT and a deadline-aware list scheduler, using three
alternative node-cost estimators:

    - "Q50_pred_us"  : raw median prediction (optimistic point estimate)
    - "Q95_pred_us"  : raw upper quantile, no conformal calibration
    - "C_e_us"        : conformal-calibrated budget (the project's actual output)

and report DMR / tardiness / slack / response time / makespan for each
(scheduler, cost estimator) combination, per spec eq. (42)-(44) and the
Handoff Guide's scheduling-interface section.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from phase1.bootstrap import build_phase1_model, load_phase1_module, load_preprocessing_state  # noqa: E402
from phase2.conformal import CriticalityCalibration  # noqa: E402
from phase2.config import DEFAULT_CONFORMAL  # noqa: E402
from phase2.inference import load_task_metadata  # noqa: E402
from phase2.scheduling import (  # noqa: E402
    deadline_aware_schedule,
    heft_schedule,
    predict_full_dag_costs,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "scheduling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAL_STATE_PATH = PROJECT_ROOT / "models" / "phase2" / "calibration" / "conformal_state.json"

COST_ESTIMATORS = ["Q50_pred_us", "Q95_pred_us", "C_e_us"]


def load_frozen_calibration_state() -> dict[str, CriticalityCalibration]:
    raw = json.loads(CAL_STATE_PATH.read_text(encoding="utf-8"))
    state = {}
    for h, d in raw.items():
        state[h] = CriticalityCalibration(
            criticality=d["criticality"], base_quantile=d["base_quantile"], delta=d["delta"],
            n_calibration=d["n_calibration"], k_index=d["k_index"], q_conf_us=d["q_conf_us"],
            fraction_zero_scores=d["fraction_zero_scores"], scores_us=None,
        )
    return state


def main() -> None:
    module = load_phase1_module()
    preprocessing_state = load_preprocessing_state()
    device = torch.device("cpu")
    model = build_phase1_model(device=device, module=module)
    calibration_state = load_frozen_calibration_state()

    task_meta = load_task_metadata()
    task_meta = task_meta[task_meta["split"] == "test_id"].set_index("graph_id")

    graph_ids = sorted(task_meta.index.tolist())
    print(f"Scheduling {len(graph_ids)} test_id DAGs ...")

    rows = []
    t_start = time.time()
    for i, graph_id in enumerate(graph_ids):
        criticality = task_meta.loc[graph_id, "criticality"]
        deadline_us = float(task_meta.loc[graph_id, "deadline_us"])
        period_us = float(task_meta.loc[graph_id, "period_us"])

        cost_df = predict_full_dag_costs(
            graph_id=graph_id, split="test_id", module=module,
            preprocessing_state=preprocessing_state, model=model, device=device,
            calibration_state=calibration_state, conformal_config=DEFAULT_CONFORMAL,
            criticality=criticality,
        )

        for cost_col in COST_ESTIMATORS:
            for scheduler_name, scheduler_fn in [
                ("HEFT", lambda: heft_schedule(module, "test_id", graph_id, cost_df, cost_col)),
                ("DeadlineAwareList", lambda: deadline_aware_schedule(
                    module, "test_id", graph_id, cost_df, cost_col, deadline_us)),
            ]:
                result = scheduler_fn()
                r_end = result.makespan_us
                rows.append(
                    {
                        "graph_id": graph_id,
                        "criticality": criticality,
                        "scheduler": scheduler_name,
                        "cost_estimator": cost_col,
                        "deadline_us": deadline_us,
                        "period_us": period_us,
                        "R_end_us": r_end,
                        "deadline_missed": r_end > deadline_us,
                        "tardiness_us": max(0.0, r_end - deadline_us),
                        "slack_us": deadline_us - r_end,
                        "makespan_us": r_end,
                    }
                )

        if (i + 1) % 10 == 0 or i == len(graph_ids) - 1:
            print(f"  [{i+1}/{len(graph_ids)}] elapsed={time.time()-t_start:.1f}s")

    per_job = pd.DataFrame(rows)
    per_job.to_csv(RESULTS_DIR / "realtime_metrics.csv", index=False)

    per_job[per_job["scheduler"] == "HEFT"].to_csv(RESULTS_DIR / "heft_results.csv", index=False)
    per_job[per_job["scheduler"] == "DeadlineAwareList"].to_csv(
        RESULTS_DIR / "deadline_aware_results.csv", index=False
    )

    summary = (
        per_job.groupby(["scheduler", "cost_estimator"])
        .agg(
            n=("graph_id", "count"),
            DMR=("deadline_missed", "mean"),
            mean_tardiness_us=("tardiness_us", "mean"),
            mean_slack_us=("slack_us", "mean"),
            mean_R_end_us=("R_end_us", "mean"),
            mean_makespan_us=("makespan_us", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "scheduling_summary.csv", index=False)
    print("\n=== Scheduling summary (all test_id DAGs, isolated single-DAG execution) ===")
    print(summary.to_string(index=False))

    summary_by_crit = (
        per_job.groupby(["scheduler", "cost_estimator", "criticality"])
        .agg(n=("graph_id", "count"), DMR=("deadline_missed", "mean"),
             mean_tardiness_us=("tardiness_us", "mean"), mean_slack_us=("slack_us", "mean"))
        .reset_index()
    )
    summary_by_crit.to_csv(RESULTS_DIR / "scheduling_summary_by_criticality.csv", index=False)
    print("\n=== By criticality ===")
    print(summary_by_crit.to_string(index=False))


if __name__ == "__main__":
    main()
