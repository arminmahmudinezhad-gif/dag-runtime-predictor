"""Generic learning-metric helpers shared across OOD / scalability / drift analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILE_LEVELS = {"Q50": 0.5, "Q90": 0.9, "Q95": 0.95, "Q99": 0.99}
QUANTILE_COLS = ["Q50_pred_us", "Q90_pred_us", "Q95_pred_us", "Q99_pred_us"]


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, quantile: float) -> np.ndarray:
    u = y - q_pred
    return np.maximum(quantile * u, (quantile - 1) * u)


def mean_pinball_by_quantile(df: pd.DataFrame) -> dict[str, float]:
    y = df["y_exec_us"].to_numpy(dtype=np.float64)
    out = {}
    for col, q in QUANTILE_LEVELS.items():
        out[f"pinball_{col}"] = float(
            np.mean(pinball_loss(y, df[f"{col}_pred_us"].to_numpy(dtype=np.float64), q))
        )
    out["mean_pinball"] = float(np.mean(list(out.values())))
    return out


def coverage_by_quantile(df: pd.DataFrame) -> dict[str, float]:
    y = df["y_exec_us"].to_numpy(dtype=np.float64)
    out = {}
    for col in QUANTILE_LEVELS:
        out[f"coverage_{col}"] = float(np.mean(y <= df[f"{col}_pred_us"].to_numpy(dtype=np.float64)))
    return out


def point_error_metrics(df: pd.DataFrame, pred_col: str = "Q50_pred_us") -> dict[str, float]:
    y = df["y_exec_us"].to_numpy(dtype=np.float64)
    p = df[pred_col].to_numpy(dtype=np.float64)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(y, 1e-9)) * 100),
        "bias": float(np.mean(err)),
    }


def quantile_crossing_rate(df: pd.DataFrame) -> float:
    q50, q90, q95, q99 = (df[c].to_numpy(dtype=np.float64) for c in QUANTILE_COLS)
    crossing = (q50 > q90) | (q90 > q95) | (q95 > q99)
    return float(np.mean(crossing))


def summarize(df: pd.DataFrame, extra: dict | None = None) -> dict:
    out = {"n": len(df)}
    out.update(mean_pinball_by_quantile(df))
    out.update(coverage_by_quantile(df))
    out.update(point_error_metrics(df))
    out["quantile_crossing_rate"] = quantile_crossing_rate(df)
    if extra:
        out.update(extra)
    return out


def grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col, dropna=False):
        if len(sub) < 5:
            continue
        row = {group_col: key}
        row.update(summarize(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def bin_column(series: pd.Series, edges: list[float], labels: list[str] | None = None) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)
