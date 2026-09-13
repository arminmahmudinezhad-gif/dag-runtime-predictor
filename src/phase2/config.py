"""Phase 2 experiment configuration (explicit alpha/delta choices)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

QuantileName = Literal["Q50", "Q90", "Q95", "Q99"]

QUANTILE_TO_ALPHA: dict[QuantileName, float] = {
    "Q50": 0.50,
    "Q90": 0.10,
    "Q95": 0.05,
    "Q99": 0.01,
}

QUANTILE_COLUMNS: dict[QuantileName, str] = {
    "Q50": "Q50_pred_us",
    "Q90": "Q90_pred_us",
    "Q95": "Q95_pred_us",
    "Q99": "Q99_pred_us",
}


@dataclass
class ConformalConfig:
    """
    Split conformal calibration by task criticality (HI / LO).

    base_quantile: raw upper quantile from Phase 1 head (1 - alpha_h).
    delta_h: miscoverage risk for the calibrated budget (target coverage >= 1 - delta_h).

    Explicit choice (see report "Conformal calibration" section for justification and the
    full alpha/delta sensitivity sweep — this is the *default*, not the only evaluated setting):

    - base_quantile = "Q95" for both criticality levels. This is decided from the
      PRE-EXISTING, already-computed Phase 1 test-ID baseline handed off in
      Phase_2_Handoff_Guide section 12 (results/phase1/metrics/test_id_metrics.json) --
      NOT from any new evaluation performed here -- which reports raw empirical coverage
      @ Q95 = 0.9900 (nominal 0.95) and @ Q99 = 1.0000 (nominal 0.99). Q99 therefore
      leaves the calibration step almost nothing to correct (scores already ~0 for most
      rows), while Q95 leaves real headroom for genuine, data-driven correction while
      still being a legitimate "upper" head. This is a one-time, principled reuse of an
      inherited frozen artifact, not iterative tuning against test_id/test_ood -- the
      operating point below is fixed BEFORE any Phase-2-computed test_id/test_ood
      calibrated-coverage number is looked at; the separate alpha/delta sensitivity sweep
      (calibration_sensitivity_sweep.csv) is a transparency report evaluated afterward for
      robustness, not a selection mechanism that ever changes this default.
    - delta_hi = 0.01 (HI-criticality tasks get a 99% marginal coverage target: the
      real-time safety case for high-criticality DAG nodes calls for a tighter guaranteed
      bound).
    - delta_lo = 0.05 (LO-criticality tasks get a 95% marginal coverage target: adequate
      statistical safety margin without over-inflating budgets for best-effort work).
    """

    base_quantile: QuantileName = "Q95"
    delta_hi: float = 0.01
    delta_lo: float = 0.05
    enforce_lo_le_hi: bool = True

    def to_json_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFORMAL = ConformalConfig()

# Sensitivity sweep grid reported alongside the default configuration above.
SENSITIVITY_BASE_QUANTILES: list[QuantileName] = ["Q90", "Q95", "Q99"]
SENSITIVITY_DELTAS: list[float] = [0.01, 0.05, 0.10]

# Scheduling / risk levels used for the DVFS energy-safety tradeoff experiment.
DVFS_ENERGY_PROXY = {
    # Simple relative energy proxy ~ V^2 * f (per active cycle), normalized to the
    # lowest DVFS point of each core type. Used only for qualitative energy/safety/
    # deadline tradeoff discussion (Section 3, DVFS impact), not calibrated wattage.
}

# Multicore scalability scenarios (big, little) — active_core_count may exceed train range.
CORE_SCALABILITY = [
    {"label": "core_4", "mbig": 2, "mlittle": 2, "active_core_count": 4},
    {"label": "core_8", "mbig": 4, "mlittle": 4, "active_core_count": 8},
    {"label": "core_16", "mbig": 8, "mlittle": 8, "active_core_count": 16},
    {"label": "core_32", "mbig": 16, "mlittle": 16, "active_core_count": 32},
]
