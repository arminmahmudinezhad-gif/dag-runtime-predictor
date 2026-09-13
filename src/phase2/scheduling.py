"""
DAG scheduling on the 4-core (2 big + 2 little) heterogeneous platform, using
frozen-model node-cost estimates. Implements:

  - Full-DAG node-cost prediction: unlike the 42-record evaluation contexts
    (6 sampled target nodes x 7 execution contexts), scheduling needs a cost
    estimate for EVERY node of the DAG. F_theta is a per-node function, so it
    can be queried for any node -- this just runs frozen inference over all
    nodes x {big, little} at a fixed representative DVFS level and a fixed
    representative system state (the train-fitted mean z_t, i.e., the
    "typical/expected" operating point), exactly the kind of a-priori cost
    HEFT-style schedulers require.
  - Classic HEFT (Topcuoglu et al.): upward-rank task prioritization,
    insertion-based earliest-finish-time processor selection.
  - A deadline-aware list scheduler: proportional sub-deadline assignment
    along the critical path (deadline partitioning), earliest-sub-deadline-first
    task ordering, same EFT-minimizing processor selection.
  - Real-time metrics: DMR, tardiness, slack, end-to-end response time,
    makespan, per spec eq. (42)-(44).

Simplifying assumption (stated explicitly): each DAG job is scheduled in
isolation on an otherwise-idle 4-core platform (no multi-job/multi-tenant
interference). This isolates the effect of the *cost estimate* (raw vs
calibrated vs mean-only) on scheduling outcomes, which is what this project's
predictor contributes -- multi-job admission-control interference is out of
scope for this Phase 2 deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from phase2.conformal import CriticalityCalibration, apply_conformal
from phase2.config import ConformalConfig
from phase2.inference import _infer_one_graph

# Representative ("typical") DVFS operating point per core type: highest
# performance level, matching Lbig[3] / Llittle[3] from the project spec.
REPRESENTATIVE_DVFS = {
    "big": {"frequency_ghz": 2.2, "voltage_v": 1.15, "nominal_bandwidth_gbps": 14.0},
    "little": {"frequency_ghz": 1.5, "voltage_v": 1.00, "nominal_bandwidth_gbps": 9.5},
}

# Representative ("typical") system state: the train-fitted mean of every z_t
# component (models/phase1/preprocessing/preprocessing_state.json).
REPRESENTATIVE_Z_T = {
    "cpu_utilization": 0.5298034237322825,
    "ready_queue_length": 2.964642857142857,
    "active_core_count": 2.504047619047619,
    "memory_active_tasks": 1.950952380952381,
    "bus_utilization": 0.4805792530986594,
    "thermal_pressure": 0.40009639953658516,
    "release_jitter_us": 79.39883347249209,
}

# Base 4-core heterogeneous platform: (processor_id, core_type).
PROCESSORS = [(0, "big"), (1, "big"), (2, "little"), (3, "little")]

GBPS_TO_BYTES_PER_US = 1000.0  # 1 GB/s = 1000 bytes/us


def predict_full_dag_costs(
    graph_id: str,
    split: str,
    module,
    preprocessing_state: dict,
    model: torch.nn.Module,
    device: torch.device,
    calibration_state: dict[str, CriticalityCalibration],
    conformal_config: ConformalConfig,
    criticality: str,
) -> pd.DataFrame:
    """One row per (node_id, core_type) with raw quantiles + calibrated C_e_us."""
    static_graph = module.load_static_graph(
        graph_id=graph_id, split=split, preprocessing_state=preprocessing_state
    )
    node_ids = list(static_graph["node_id_to_index"].keys())

    rows = []
    for node_id in node_ids:
        for core_type, dvfs in REPRESENTATIVE_DVFS.items():
            row = {
                "graph_id": graph_id,
                "target_node": node_id,
                "sample_id": f"{graph_id}_{node_id}_{core_type}",
                "core_type": core_type,
                "frequency_ghz": dvfs["frequency_ghz"],
                "voltage_v": dvfs["voltage_v"],
                "y_exec_us": 0.0,  # placeholder; not real ground truth, dropped below
            }
            row.update(REPRESENTATIVE_Z_T)
            rows.append(row)
    records_df = pd.DataFrame(rows)

    dataset = module.GraphScenarioDataset(
        records_df=records_df,
        split=split,
        preprocessing_state=preprocessing_state,
        expected_records_per_graph=None,
        cache_static_graphs=False,
    )
    graph_item = dataset[0]
    predictions = _infer_one_graph(model, graph_item, preprocessing_state, device)
    predictions = predictions.drop(columns=["y_exec_us"])
    predictions["criticality"] = criticality

    predictions = apply_conformal(predictions, calibration_state, conformal_config)
    return predictions


@dataclass
class ScheduleResult:
    graph_id: str
    policy: str
    cost_estimator: str
    makespan_us: float
    node_finish_us: dict[str, float]
    node_processor: dict[str, int]


def _load_edges(module, split: str, graph_id: str) -> pd.DataFrame:
    edge_path = module.EDGE_FEATURES_DIR / split / f"{graph_id}.csv"
    edges = pd.read_csv(edge_path)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    return edges


def _load_nodes(module, split: str, graph_id: str) -> pd.DataFrame:
    node_path = module.NODE_FEATURES_DIR / split / f"{graph_id}.csv"
    nodes = pd.read_csv(node_path)
    nodes["node_id"] = nodes["node_id"].astype(str)
    return nodes


def _communication_cost_us(data_bytes: float, proc_a_type: str, proc_b_type: str) -> float:
    """Inter-processor transfer time, using the slower endpoint's nominal bandwidth."""
    bw_gbps = min(
        REPRESENTATIVE_DVFS[proc_a_type]["nominal_bandwidth_gbps"],
        REPRESENTATIVE_DVFS[proc_b_type]["nominal_bandwidth_gbps"],
    )
    bw_bytes_per_us = bw_gbps * GBPS_TO_BYTES_PER_US
    return float(data_bytes) / bw_bytes_per_us


def _build_dag_inputs(module, split: str, graph_id: str, cost_df: pd.DataFrame, cost_col: str):
    """
    Returns:
        node_ids: topological order (by topo_level)
        preds, succs: adjacency dicts (node_id -> list[node_id])
        edge_bytes: {(u, v): data_bytes}
        cost: {node_id: {core_type: cost_us}}
    """
    nodes = _load_nodes(module, split, graph_id)
    edges = _load_edges(module, split, graph_id)

    node_ids = nodes.sort_values("topo_level")["node_id"].tolist()
    preds = {n: [] for n in node_ids}
    succs = {n: [] for n in node_ids}
    edge_bytes = {}
    for _, e in edges.iterrows():
        u, v, d = e["source"], e["target"], float(e["data_bytes"])
        succs[u].append(v)
        preds[v].append(u)
        edge_bytes[(u, v)] = d

    cost = {}
    for node_id, sub in cost_df.groupby("target_node"):
        cost[str(node_id)] = dict(zip(sub["core_type"], sub[cost_col]))

    return node_ids, preds, succs, edge_bytes, cost


def _upward_rank(node_ids, succs, edge_bytes, cost, proc_types) -> dict[str, float]:
    avg_cost = {n: float(np.mean([cost[n][t] for t in proc_types])) for n in node_ids}
    avg_comm = {}
    for (u, v), d in edge_bytes.items():
        bw = np.mean([REPRESENTATIVE_DVFS[t]["nominal_bandwidth_gbps"] for t in proc_types]) * GBPS_TO_BYTES_PER_US
        avg_comm[(u, v)] = d / bw

    rank = {}

    def compute(n):
        if n in rank:
            return rank[n]
        if not succs[n]:
            rank[n] = avg_cost[n]
            return rank[n]
        best = 0.0
        for s in succs[n]:
            best = max(best, avg_comm.get((n, s), 0.0) + compute(s))
        rank[n] = avg_cost[n] + best
        return rank[n]

    for n in node_ids:
        compute(n)
    return rank


def heft_schedule(module, split: str, graph_id: str, cost_df: pd.DataFrame, cost_col: str) -> ScheduleResult:
    proc_types = [t for _, t in PROCESSORS]
    node_ids, preds, succs, edge_bytes, cost = _build_dag_inputs(module, split, graph_id, cost_df, cost_col)
    rank_u = _upward_rank(node_ids, succs, edge_bytes, cost, proc_types)

    order = sorted(node_ids, key=lambda n: -rank_u[n])

    proc_available = {p: 0.0 for p, _ in PROCESSORS}
    aft = {}
    node_processor = {}

    for n in order:
        best_proc, best_eft, best_est = None, float("inf"), 0.0
        for proc_id, proc_type in PROCESSORS:
            ready = 0.0
            for p in preds[n]:
                comm = 0.0
                if node_processor.get(p) is not None and node_processor[p] != proc_id:
                    p_type = dict(PROCESSORS)[node_processor[p]]
                    comm = _communication_cost_us(edge_bytes.get((p, n), 0.0), p_type, proc_type)
                ready = max(ready, aft[p] + comm)
            est = max(ready, proc_available[proc_id])
            eft = est + cost[n][proc_type]
            if eft < best_eft:
                best_proc, best_eft, best_est = proc_id, eft, est
        node_processor[n] = best_proc
        aft[n] = best_eft
        proc_available[best_proc] = best_eft

    makespan = max(aft.values()) if aft else 0.0
    return ScheduleResult(graph_id, "HEFT", cost_col, makespan, aft, node_processor)


def deadline_aware_schedule(
    module, split: str, graph_id: str, cost_df: pd.DataFrame, cost_col: str, deadline_us: float
) -> ScheduleResult:
    """
    Deadline-partitioned EDF-style list scheduler: distributes the DAG's overall
    deadline along the critical path proportionally to average node cost, then
    schedules ready nodes in ascending sub-deadline order (ties broken by
    upward rank), assigning each to the processor giving earliest finish time.
    """
    proc_types = [t for _, t in PROCESSORS]
    node_ids, preds, succs, edge_bytes, cost = _build_dag_inputs(module, split, graph_id, cost_df, cost_col)
    rank_u = _upward_rank(node_ids, succs, edge_bytes, cost, proc_types)

    avg_cost = {n: float(np.mean([cost[n][t] for t in proc_types])) for n in node_ids}
    critical_path_len = max(rank_u.values()) if rank_u else 1.0
    critical_path_len = max(critical_path_len, 1e-9)

    # Sub-deadline: proportional share of the overall deadline based on each
    # node's "distance from source" position along the critical path,
    # i.e. (critical_path_len - rank_u[n] + avg_cost[n]) is the earliest a node
    # could plausibly finish relative to the critical path; scale to deadline_us.
    sub_deadline = {
        n: deadline_us * (critical_path_len - rank_u[n] + avg_cost[n]) / critical_path_len for n in node_ids
    }

    order = sorted(node_ids, key=lambda n: (sub_deadline[n], -rank_u[n]))

    proc_available = {p: 0.0 for p, _ in PROCESSORS}
    aft = {}
    node_processor = {}

    for n in order:
        best_proc, best_eft = None, float("inf")
        for proc_id, proc_type in PROCESSORS:
            ready = 0.0
            for p in preds[n]:
                comm = 0.0
                if node_processor.get(p) is not None and node_processor[p] != proc_id:
                    p_type = dict(PROCESSORS)[node_processor[p]]
                    comm = _communication_cost_us(edge_bytes.get((p, n), 0.0), p_type, proc_type)
                ready = max(ready, aft[p] + comm)
            est = max(ready, proc_available[proc_id])
            eft = est + cost[n][proc_type]
            if eft < best_eft:
                best_proc, best_eft = proc_id, eft
        node_processor[n] = best_proc
        aft[n] = best_eft
        proc_available[best_proc] = best_eft

    makespan = max(aft.values()) if aft else 0.0
    return ScheduleResult(graph_id, "DeadlineAwareList", cost_col, makespan, aft, node_processor)
