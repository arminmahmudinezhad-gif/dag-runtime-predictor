"""
Drift / overload / bursty-state robustness analysis (spec sections 5 & 7;
Handoff Guide 16.3).

Two genuinely different questions are kept separate, as the spec requires:

(a) Does the frozen predictor generalize?
    -> "In-range tail stress": group REAL evaluation records (real ground truth)
       by the upper decile of memory/bus/thermal/jitter pressure and compare
       accuracy against the rest of the distribution. This is a valid accuracy
       claim because the ground truth is real simulator output.

(b) Does conformal calibration maintain useful coverage under synthetic,
    genuinely out-of-range overload (z_t pushed beyond anything ever observed)?
    -> Descriptive only (predicted-distribution shift + calibrated-budget
       behavior), same "no fabricated ground truth" discipline as scalability.py.

A third piece compares STATIC conformal calibration (fit once on the calibration
split, frozen) against a SLIDING-WINDOW / online-adaptive variant that only ever
uses *already-revealed* residuals from earlier windows of the evaluation stream
-- never future ones -- to correct upcoming predictions. This is a legitimate
"adaptive calibration under drift" comparison (spec 5.3), not test-set tuning:
each window's correction is fit exclusively on strictly earlier windows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from phase2.config import QUANTILE_COLUMNS, QuantileName


def flag_tail_stress(df: pd.DataFrame, quantile: float = 0.9) -> pd.DataFrame:
    """Flag records in the upper decile (by default) of a composite load proxy."""
    out = df.copy()
    load_cols = ["memory_active_tasks", "bus_utilization", "thermal_pressure", "release_jitter_us"]
    z = out[load_cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    out["composite_load_proxy"] = z.mean(axis=1)
    threshold = out["composite_load_proxy"].quantile(quantile)
    out["is_high_load_tail"] = out["composite_load_proxy"] >= threshold
    return out


def synthetic_overload_records(base_records_df: pd.DataFrame, overrides: dict[str, float]) -> pd.DataFrame:
    """Override selected z_t columns with fixed, genuinely out-of-range values."""
    out = base_records_df.copy()
    for col, value in overrides.items():
        out[col] = value
    return out


def sliding_window_conformal(
    ordered_df: pd.DataFrame,
    base_quantile: QuantileName,
    delta: float,
    window_graphs: int = 5,
    records_per_graph: int = 42,
) -> pd.DataFrame:
    """
    Causal, strictly-past-only sliding-window conformal calibration.

    `ordered_df` must be sorted by an arrival order (here: graph_id, i.e. arrival
    order of DAG jobs). Records are grouped into sequential windows of
    `window_graphs` DAGs each. Window i's correction is fit on the concatenated
    nonconformity scores of ALL windows < i (an expanding causal window); window 0
    has no prior data and falls back to the raw (uncorrected) quantile, exactly
    like a real online system would behave before it has seen any residuals.
    """
    base_col = QUANTILE_COLUMNS[base_quantile]
    graph_order = ordered_df["graph_id"].drop_duplicates().tolist()
    windows = [graph_order[i:i + window_graphs] for i in range(0, len(graph_order), window_graphs)]

    out_frames = []
    seen_scores: list[float] = []
    for window_graph_ids in windows:
        window_df = ordered_df[ordered_df["graph_id"].isin(window_graph_ids)].copy()

        if seen_scores:
            scores_sorted = np.sort(np.array(seen_scores))
            n = len(scores_sorted)
            k = min(n, math.ceil((n + 1) * (1 - delta)))
            k = max(k, 1)
            q_conf = float(scores_sorted[k - 1])
        else:
            q_conf = 0.0  # no prior data yet: fall back to the raw quantile

        window_df["sliding_q_conf_us"] = q_conf
        window_df["sliding_C_e_us"] = window_df[base_col] + q_conf
        window_df["sliding_covered"] = window_df["y_exec_us"] <= window_df["sliding_C_e_us"]
        out_frames.append(window_df)

        # Reveal this window's true residuals for the NEXT window only (causal).
        new_scores = np.maximum(0.0, window_df["y_exec_us"] - window_df[base_col]).tolist()
        seen_scores.extend(new_scores)

    return pd.concat(out_frames, ignore_index=True)
