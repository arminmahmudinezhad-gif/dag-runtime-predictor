"""
Split conformal calibration by task criticality (HI / LO), per project spec eq. (30)-(36)
and Phase 2 Handoff Guide section 15.

    s_j^(h)    = [y_j - Qhat_(1-alpha_h)(x_j)]_+                       (one-sided score)
    k_h        = min(N_h, ceil((N_h + 1) * (1 - delta_h)))             (rank index)
    q_conf^(h) = k_h-th smallest calibration score                     (correction)
    C_e^(h)(x) = Qhat_(1-alpha_h)(x) + q_conf^(h)                      (calibrated budget, us)

Target statistical statement: P(Y <= C_e^(h)(X) | H=h) >= 1 - delta_h under the
exchangeability assumption between calibration data and evaluation data within
criticality level h. This is a marginal coverage guarantee, not a deterministic WCET.

The Phase 1 model is never touched here: calibration only post-processes its frozen,
already-inverse-transformed raw quantile predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import (
    QUANTILE_COLUMNS,
    ConformalConfig,
    QuantileName,
)

CRITICALITY_LEVELS = ("HI", "LO")


@dataclass
class CriticalityCalibration:
    criticality: str
    base_quantile: QuantileName
    delta: float
    n_calibration: int
    k_index: int
    q_conf_us: float
    fraction_zero_scores: float
    scores_us: np.ndarray

    def to_json_dict(self) -> dict:
        return {
            "criticality": self.criticality,
            "base_quantile": self.base_quantile,
            "delta": self.delta,
            "n_calibration": self.n_calibration,
            "k_index": self.k_index,
            "q_conf_us": self.q_conf_us,
            "fraction_zero_scores": self.fraction_zero_scores,
            "score_mean_us": float(np.mean(self.scores_us)) if self.n_calibration else None,
            "score_median_us": float(np.median(self.scores_us)) if self.n_calibration else None,
            "score_max_us": float(np.max(self.scores_us)) if self.n_calibration else None,
        }


def nonconformity_scores(df: pd.DataFrame, base_quantile: QuantileName) -> np.ndarray:
    """s_j = max(0, y_j - Qhat_base(x_j)), in microseconds."""
    base_col = QUANTILE_COLUMNS[base_quantile]
    y = df["y_exec_us"].to_numpy(dtype=np.float64)
    qhat = df[base_col].to_numpy(dtype=np.float64)
    return np.maximum(0.0, y - qhat)


def _delta_for(criticality: str, config: ConformalConfig) -> float:
    if criticality == "HI":
        return config.delta_hi
    if criticality == "LO":
        return config.delta_lo
    raise ValueError(f"Unknown criticality '{criticality}'")


def fit_conformal(
    calibration_df: pd.DataFrame,
    config: ConformalConfig = None,
) -> dict[str, CriticalityCalibration]:
    """
    Fit one split-conformal correction per criticality level on the (untouched-until-now)
    calibration split. Returns {"HI": CriticalityCalibration, "LO": CriticalityCalibration}.
    """
    from phase2.config import DEFAULT_CONFORMAL

    config = config or DEFAULT_CONFORMAL
    state: dict[str, CriticalityCalibration] = {}

    for h in CRITICALITY_LEVELS:
        sub = calibration_df.loc[calibration_df["criticality"] == h]
        if sub.empty:
            raise ValueError(f"No calibration rows found for criticality '{h}'.")

        scores = np.sort(nonconformity_scores(sub, config.base_quantile))
        n_h = len(scores)
        delta_h = _delta_for(h, config)

        k_h = min(n_h, math.ceil((n_h + 1) * (1 - delta_h)))
        k_h = max(k_h, 1)  # guard against delta so large that ceil(...) rounds to 0
        q_conf = float(scores[k_h - 1])  # k_h-th smallest, 1-indexed

        state[h] = CriticalityCalibration(
            criticality=h,
            base_quantile=config.base_quantile,
            delta=delta_h,
            n_calibration=n_h,
            k_index=k_h,
            q_conf_us=q_conf,
            fraction_zero_scores=float(np.mean(scores == 0.0)),
            scores_us=scores,
        )

    if config.enforce_lo_le_hi:
        # Optional final monotonic step (spec section 5/eq. after 36): ensure the HI
        # correction is never smaller than the LO correction, since HI nodes must not
        # receive a *less* conservative budget purely because of a calibration-set
        # sampling artifact.
        if state["HI"].q_conf_us < state["LO"].q_conf_us:
            hi = state["HI"]
            state["HI"] = CriticalityCalibration(
                criticality="HI",
                base_quantile=hi.base_quantile,
                delta=hi.delta,
                n_calibration=hi.n_calibration,
                k_index=hi.k_index,
                q_conf_us=state["LO"].q_conf_us,
                fraction_zero_scores=hi.fraction_zero_scores,
                scores_us=hi.scores_us,
            )

    return state


def apply_conformal(
    df: pd.DataFrame,
    calibration_state: dict[str, CriticalityCalibration],
    config: ConformalConfig,
) -> pd.DataFrame:
    """Add Qhat_base_us, nonconformity_score_us (if y_exec_us present), and C_e_us."""
    out = df.copy()
    base_col = QUANTILE_COLUMNS[config.base_quantile]
    out["Qhat_base_us"] = out[base_col]

    q_conf_map = {h: s.q_conf_us for h, s in calibration_state.items()}
    out["q_conf_us"] = out["criticality"].map(q_conf_map)
    if out["q_conf_us"].isna().any():
        missing = out.loc[out["q_conf_us"].isna(), "criticality"].unique()
        raise ValueError(f"No calibration state for criticality values: {missing}")

    out["C_e_us"] = out["Qhat_base_us"] + out["q_conf_us"]

    if "y_exec_us" in out.columns:
        out["raw_covered"] = out["y_exec_us"] <= out["Qhat_base_us"]
        out["calibrated_covered"] = out["y_exec_us"] <= out["C_e_us"]
        out["inflation_us"] = out["C_e_us"] - out["Qhat_base_us"]

    return out


def coverage_report(applied_df: pd.DataFrame, config: ConformalConfig) -> pd.DataFrame:
    """
    Per-criticality (and overall) raw vs. calibrated empirical coverage/miscoverage
    and inflation summary. Required by spec section "تحلیل اثر کالیبراسیون هم‌ریخت".
    """
    rows = []
    groups = list(applied_df.groupby("criticality")) + [("ALL", applied_df)]
    for h, sub in groups:
        target_delta = (
            config.delta_hi if h == "HI" else config.delta_lo if h == "LO" else None
        )
        rows.append(
            {
                "criticality": h,
                "n": len(sub),
                "base_quantile": config.base_quantile,
                "target_coverage": (1 - target_delta) if target_delta is not None else None,
                "raw_coverage": float(sub["raw_covered"].mean()),
                "raw_miscoverage": float(1 - sub["raw_covered"].mean()),
                "calibrated_coverage": float(sub["calibrated_covered"].mean()),
                "calibrated_miscoverage": float(1 - sub["calibrated_covered"].mean()),
                "mean_inflation_us": float(sub["inflation_us"].mean()),
                "median_inflation_us": float(sub["inflation_us"].median()),
                "mean_Qhat_base_us": float(sub["Qhat_base_us"].mean()),
                "mean_C_e_us": float(sub["C_e_us"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_calibration_state(
    calibration_state: dict[str, CriticalityCalibration],
    config: ConformalConfig,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conformal_config.json").write_text(
        json.dumps(config.to_json_dict(), indent=2), encoding="utf-8"
    )
    state_json = {h: c.to_json_dict() for h, c in calibration_state.items()}
    (out_dir / "conformal_state.json").write_text(
        json.dumps(state_json, indent=2), encoding="utf-8"
    )


def sensitivity_sweep(
    calibration_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    base_quantiles: list[QuantileName],
    deltas: list[float],
) -> pd.DataFrame:
    """
    Full alpha_h/delta_h sensitivity grid (spec: "Do not invent alpha/delta silently").
    Same delta applied to both HI and LO for each grid cell (symmetric sweep); the
    asymmetric default (delta_hi=0.01, delta_lo=0.05) is reported separately as the
    chosen operating point.
    """
    rows = []
    for base_q in base_quantiles:
        for delta in deltas:
            cfg = ConformalConfig(
                base_quantile=base_q, delta_hi=delta, delta_lo=delta, enforce_lo_le_hi=False
            )
            state = fit_conformal(calibration_df, cfg)
            applied = apply_conformal(eval_df, state, cfg)
            report = coverage_report(applied, cfg)
            for _, r in report.iterrows():
                rows.append(
                    {
                        "base_quantile": base_q,
                        "delta": delta,
                        "criticality": r["criticality"],
                        "q_conf_us": state[r["criticality"]].q_conf_us
                        if r["criticality"] in state
                        else np.nan,
                        "raw_coverage": r["raw_coverage"],
                        "calibrated_coverage": r["calibrated_coverage"],
                        "mean_inflation_us": r["mean_inflation_us"],
                    }
                )
    return pd.DataFrame(rows)
