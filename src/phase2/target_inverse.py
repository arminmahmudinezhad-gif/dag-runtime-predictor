"""Inverse target scaling (microseconds) — copied from Phase 1 inference utilities."""

from __future__ import annotations

import numpy as np
import torch


def _to_scalar_float(value) -> float | None:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def get_target_standardization_stats(preprocessing_state: dict) -> tuple[float, float, str]:
    mean_keys = {"mean", "mean_", "train_mean", "mean_us", "mu", "center"}
    std_keys = {"std", "std_", "train_std", "std_us", "sigma", "scale", "scale_"}
    candidates: list[dict] = []

    def inspect_mapping(mapping: dict, path: tuple[str, ...]) -> None:
        found_mean = found_std = None
        for key, value in mapping.items():
            key_lower = str(key).lower()
            if key_lower in mean_keys and found_mean is None:
                scalar = _to_scalar_float(value)
                if scalar is not None:
                    found_mean = scalar
            if key_lower in std_keys and found_std is None:
                scalar = _to_scalar_float(value)
                if scalar is not None:
                    found_std = scalar
        if found_mean is None or found_std is None or found_std <= 0:
            return
        path_text = ".".join(path)
        path_lower = path_text.lower()
        score = 0
        if "y_exec_us" in path_lower:
            score += 100
        if "target" in path_lower:
            score += 50
        if "scaler" in path_lower:
            score += 10
        if "transform" in path_lower:
            score += 10
        if "stat" in path_lower:
            score += 10
        candidates.append(
            {"mean": found_mean, "std": found_std, "path": path_text, "score": score}
        )

    def walk(obj, path: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            inspect_mapping(obj, path)
            for key, value in obj.items():
                walk(value, path + (str(key),))
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                walk(value, path + (f"[{index}]",))

    walk(preprocessing_state)
    target_candidates = [c for c in candidates if c["score"] > 0]
    if not target_candidates:
        raise KeyError("Could not locate train-fitted target mean/std in preprocessing_state.")
    target_candidates.sort(key=lambda c: (c["score"], len(c["path"])), reverse=True)
    selected = target_candidates[0]
    return float(selected["mean"]), float(selected["std"]), selected["path"]


def inverse_transform_target(values_standardized, preprocessing_state: dict):
    mean_us, std_us, _ = get_target_standardization_stats(preprocessing_state)
    if torch.is_tensor(values_standardized):
        return values_standardized * std_us + mean_us
    arr = np.asarray(values_standardized, dtype=np.float64)
    return arr * std_us + mean_us
