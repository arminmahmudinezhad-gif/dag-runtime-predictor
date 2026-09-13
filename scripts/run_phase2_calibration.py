"""
Fit split-conformal calibration on the calibration split, apply it (frozen) to
test_id and test_ood, and save all required Phase 2 calibration artifacts.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from phase2.conformal import (  # noqa: E402
    apply_conformal,
    coverage_report,
    fit_conformal,
    save_calibration_state,
    sensitivity_sweep,
)
from phase2.config import DEFAULT_CONFORMAL, SENSITIVITY_BASE_QUANTILES, SENSITIVITY_DELTAS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"
MODELS_DIR = PROJECT_ROOT / "models" / "phase2" / "calibration"


def load_raw(split: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{split}_raw_predictions.csv")


def main() -> None:
    calibration_df = load_raw("calibration")
    test_id_df = load_raw("test_id")
    test_ood_df = load_raw("test_ood")

    print("Fitting split-conformal calibration on the calibration split "
          f"(base_quantile={DEFAULT_CONFORMAL.base_quantile}, "
          f"delta_hi={DEFAULT_CONFORMAL.delta_hi}, delta_lo={DEFAULT_CONFORMAL.delta_lo}) ...")
    state = fit_conformal(calibration_df, DEFAULT_CONFORMAL)
    for h, c in state.items():
        print(f"  {h}: N={c.n_calibration} k={c.k_index} q_conf={c.q_conf_us:.2f} us "
              f"(fraction_zero_scores={c.fraction_zero_scores:.3f})")

    save_calibration_state(state, DEFAULT_CONFORMAL, MODELS_DIR)

    # Apply frozen calibration correction to calibration (self, sanity check),
    # test_id, and test_ood. Keep the correction frozen throughout.
    calibrated_calibration = apply_conformal(calibration_df, state, DEFAULT_CONFORMAL)
    calibrated_test_id = apply_conformal(test_id_df, state, DEFAULT_CONFORMAL)
    calibrated_test_ood = apply_conformal(test_ood_df, state, DEFAULT_CONFORMAL)

    calibrated_calibration.to_csv(RESULTS_DIR / "calibration_predictions.csv", index=False)
    calibrated_test_id.to_csv(RESULTS_DIR / "calibrated_test_id.csv", index=False)
    calibrated_test_ood.to_csv(RESULTS_DIR / "calibrated_test_ood.csv", index=False)

    for name, df in [
        ("calibration (self)", calibrated_calibration),
        ("test_id", calibrated_test_id),
        ("test_ood", calibrated_test_ood),
    ]:
        report = coverage_report(df, DEFAULT_CONFORMAL)
        report.insert(0, "eval_split", name)
        out_path = RESULTS_DIR / f"coverage_report_{name.split()[0]}.csv"
        report.to_csv(out_path, index=False)
        print(f"\n[{name}] coverage report -> {out_path}")
        print(report.to_string(index=False))

    coverage_all = pd.concat(
        [
            coverage_report(calibrated_calibration, DEFAULT_CONFORMAL).assign(eval_split="calibration"),
            coverage_report(calibrated_test_id, DEFAULT_CONFORMAL).assign(eval_split="test_id"),
            coverage_report(calibrated_test_ood, DEFAULT_CONFORMAL).assign(eval_split="test_ood"),
        ],
        ignore_index=True,
    )
    coverage_all.to_csv(RESULTS_DIR / "coverage_by_criticality.csv", index=False)

    inflation_by_crit = (
        pd.concat(
            [calibrated_test_id.assign(eval_split="test_id"), calibrated_test_ood.assign(eval_split="test_ood")],
            ignore_index=True,
        )
        .groupby(["eval_split", "criticality"])["inflation_us"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    inflation_by_crit.to_csv(RESULTS_DIR / "inflation_by_criticality.csv", index=False)
    print(f"\nInflation by criticality -> {RESULTS_DIR / 'inflation_by_criticality.csv'}")
    print(inflation_by_crit.to_string(index=False))

    print("\nRunning alpha_h/delta_h sensitivity sweep "
          f"(base_quantile in {SENSITIVITY_BASE_QUANTILES}, delta in {SENSITIVITY_DELTAS}) ...")
    sweep = sensitivity_sweep(calibration_df, test_id_df, SENSITIVITY_BASE_QUANTILES, SENSITIVITY_DELTAS)
    sweep_path = RESULTS_DIR / "calibration_sensitivity_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"Sensitivity sweep -> {sweep_path}")
    print(sweep.to_string(index=False))

    # Save the calibration-split nonconformity scores for plotting.
    scores_rows = []
    for h, c in state.items():
        for s in c.scores_us:
            scores_rows.append({"criticality": h, "nonconformity_score_us": s})
    pd.DataFrame(scores_rows).to_csv(RESULTS_DIR / "calibration_scores.csv", index=False)


if __name__ == "__main__":
    main()
