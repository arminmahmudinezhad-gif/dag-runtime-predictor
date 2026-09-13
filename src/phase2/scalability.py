"""
4 / 8 / 16 / 32-core scalability stress test (spec section 3; Handoff Guide 16.2).

Because physical core_id was never a Phase 1 predictor input (only core_type
big/little), a literal "32-core system" cannot be re-simulated by this frozen
model -- it was never given a total-core-count feature. What *can* be queried
honestly is the frozen predictor's behavior as the one system-state feature that
does encode core-count pressure, `active_core_count`, is pushed toward and beyond
its training range (which was only ever sampled in [1, 4]).

This module overrides `active_core_count` on real evaluation contexts (topology,
core_type, DVFS, and every other z_t feature are left untouched) and reuses the
exact frozen preprocessing + frozen model, per the "no refit" rule. The resulting
comparison is a genuine uncertainty/robustness characterization of the frozen
model under an explicit, clearly labeled OOD input -- not a claim about physical
16- or 32-core hardware accuracy, since no such ground truth exists for this
synthetic dataset.
"""

from __future__ import annotations

import pandas as pd
import torch

from phase1.bootstrap import build_phase1_model, load_phase1_module, load_preprocessing_state
from phase2.inference import _infer_one_graph


def active_core_count_zscore(active_core_count: float, preprocessing_state: dict) -> float:
    stats = preprocessing_state["context"]["z_t_standardized"]["active_core_count"]
    return (active_core_count - stats["mean"]) / stats["std"]


def build_scenario_records(base_records_df: pd.DataFrame, active_core_count: int) -> pd.DataFrame:
    """
    Override active_core_count only; every other column (topology, core_type,
    frequency_ghz, voltage_v, cpu_utilization, ready_queue_length,
    memory_active_tasks, bus_utilization, thermal_pressure, release_jitter_us)
    is left exactly as sampled for the real evaluation record.
    """
    out = base_records_df.copy()
    out["active_core_count"] = active_core_count
    return out


def run_scalability_scenario(
    base_records_df: pd.DataFrame,
    split: str,
    active_core_count: int,
    device: torch.device | None = None,
    model: torch.nn.Module | None = None,
    module=None,
    preprocessing_state: dict | None = None,
) -> pd.DataFrame:
    module = module or load_phase1_module()
    preprocessing_state = preprocessing_state or load_preprocessing_state()
    device = device or torch.device("cpu")
    model = model or build_phase1_model(device=device, module=module)
    model.eval()

    scenario_records = build_scenario_records(base_records_df, active_core_count)

    dataset = module.GraphScenarioDataset(
        records_df=scenario_records,
        split=split,
        preprocessing_state=preprocessing_state,
        expected_records_per_graph=42,
        cache_static_graphs=False,
    )

    frames = []
    for i in range(len(dataset)):
        graph_item = dataset[i]
        frames.append(_infer_one_graph(model, graph_item, preprocessing_state, device))
    result = pd.concat(frames, ignore_index=True)
    result["scenario_active_core_count"] = active_core_count
    result["active_core_count_zscore"] = active_core_count_zscore(active_core_count, preprocessing_state)
    return result
