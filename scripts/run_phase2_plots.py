"""Generate every Phase 2 plot required by the spec's expected-output section."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CAL_DIR = PROJECT_ROOT / "results" / "phase2" / "calibration"
OOD_DIR = PROJECT_ROOT / "results" / "phase2" / "ood"
SCALE_DIR = PROJECT_ROOT / "results" / "phase2" / "scalability"
SCHED_DIR = PROJECT_ROOT / "results" / "phase2" / "scheduling"
DRIFT_DIR = PROJECT_ROOT / "results" / "phase2" / "drift"
BASE_DIR = PROJECT_ROOT / "results" / "phase2" / "baselines"
PLOTS_DIR = PROJECT_ROOT / "results" / "phase2" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def plot_violin_by_role():
    df = pd.read_csv(CAL_DIR / "calibrated_test_id.csv")
    roles = df["target_role"].unique().tolist()
    fig, axes = plt.subplots(1, len(roles), figsize=(3 * len(roles), 4), sharey=True)
    for ax, role in zip(axes, roles):
        sub = df[df["target_role"] == role]
        data = [sub[c] for c in ["Q50_pred_us", "Q90_pred_us", "Q95_pred_us", "Q99_pred_us"]]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q50", "Q90", "Q95", "Q99"], rotation=45)
        ax.set_title(role, fontsize=8)
    axes[0].set_ylabel("predicted execution time (us)")
    fig.suptitle("Predicted quantile distributions by target-node role (test_id)")
    savefig(fig, "violin_predicted_quantiles_by_role.png")


def plot_violin_by_hardware_scenario():
    df = pd.read_csv(CAL_DIR / "calibrated_test_id.csv")
    df["scenario"] = df["core_type"] + " / dvfs" + df["dvfs_level"].astype(str)
    scenarios = sorted(df["scenario"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [df[df["scenario"] == s]["Q95_pred_us"] for s in scenarios]
    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("predicted Q95 execution time (us)")
    ax.set_title("Predicted Q95 distribution by hardware scenario (core_type x DVFS level, test_id)")
    savefig(fig, "violin_predicted_by_hardware_scenario.png")


def plot_box_actual_vs_predicted():
    df = pd.read_csv(CAL_DIR / "calibrated_test_id.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df["y_exec_us"], df["Q50_pred_us"], df["Q95_pred_us"], df["C_e_us"]]
    ax.boxplot(data, labels=["actual y_exec", "pred Q50", "pred Q95", "calibrated C_e"], showfliers=False)
    ax.set_ylabel("microseconds")
    ax.set_title("Actual vs. predicted/calibrated execution time (test_id)")
    savefig(fig, "box_actual_vs_predicted.png")


def plot_reliability_diagram():
    frames = []
    for split in ["test_id", "test_ood"]:
        df = pd.read_csv(CAL_DIR / f"calibrated_{split}.csv")
        for crit in ["HI", "LO"]:
            sub = df[df["criticality"] == crit]
            for q_col, nominal in [("Q50_pred_us", 0.50), ("Q90_pred_us", 0.90),
                                    ("Q95_pred_us", 0.95), ("Q99_pred_us", 0.99)]:
                emp = float((sub["y_exec_us"] <= sub[q_col]).mean())
                frames.append({"split": split, "criticality": crit, "nominal": nominal, "empirical": emp})
    rel = pd.DataFrame(frames)
    rel.to_csv(PLOTS_DIR / "reliability_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, split in zip(axes, ["test_id", "test_ood"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for crit, marker in [("HI", "o"), ("LO", "s")]:
            sub = rel[(rel["split"] == split) & (rel["criticality"] == crit)].sort_values("nominal")
            ax.plot(sub["nominal"], sub["empirical"], marker=marker, label=f"{crit}")
        ax.set_title(f"Reliability diagram — {split}")
        ax.set_xlabel("nominal quantile level")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("empirical coverage")
    savefig(fig, "reliability_diagram.png")


def plot_coverage_before_after():
    cov = pd.read_csv(CAL_DIR / "coverage_by_criticality.csv")
    cov = cov[cov["criticality"].isin(["HI", "LO"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    splits = cov["eval_split"].unique()
    x = np.arange(len(splits))
    width = 0.18
    offsets = {"HI_raw": -1.5, "HI_cal": -0.5, "LO_raw": 0.5, "LO_cal": 1.5}
    for key, off in offsets.items():
        crit, kind = key.split("_")
        col = "raw_coverage" if kind == "raw" else "calibrated_coverage"
        vals = [cov[(cov["eval_split"] == s) & (cov["criticality"] == crit)][col].iloc[0] for s in splits]
        ax.bar(x + off * width, vals, width, label=f"{crit} {'raw' if kind=='raw' else 'calibrated'}")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.axhline(0.99, color="gray", ls=":", lw=1)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Empirical coverage before/after conformal calibration")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, "coverage_before_after_calibration.png")


def plot_inflation_hist():
    """
    NOTE on what this plot shows and why: the *applied* per-record inflation
    (C_e - Q_base) is a CONSTANT within each criticality level -- q_conf is one
    fixed number added to every row -- so a histogram of it is a zero-width
    spike (invisible, std=0). The informative, genuinely-distributed quantity
    is the *calibration-set nonconformity score* s_j = max(0, y_j - Q_base(x_j))
    BEFORE the k_h-th order statistic collapses it to the single q_conf value;
    this is also exactly what the spec/handoff guide explicitly asks for
    ("distribution of nonconformity scores, including the fraction of zero
    scores"). Left panel: score distribution (log-scale y-axis, since ~98-99%
    of scores are exactly zero) with the resulting q_conf threshold marked.
    Right panel: the resulting single inflation/q_conf value applied per
    criticality (a bar, not a histogram, since it truly is one number).
    """
    scores = pd.read_csv(CAL_DIR / "calibration_scores.csv")
    with open(PROJECT_ROOT / "models" / "phase2" / "calibration" / "conformal_state.json") as f:
        import json
        conformal_state = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    colors = {"HI": "#4C72B0", "LO": "#DD8452"}
    max_score = scores["nonconformity_score_us"].max()
    bins = np.linspace(0, max_score, 40)
    for crit in ["HI", "LO"]:
        sub = scores[scores["criticality"] == crit]["nonconformity_score_us"]
        frac_zero = float((sub == 0).mean())
        ax.hist(sub, bins=bins, alpha=0.55, label=f"{crit} (n={len(sub)}, {frac_zero:.1%} zero)",
                color=colors[crit])
        ax.axvline(conformal_state[crit]["q_conf_us"], color=colors[crit], ls="--", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("nonconformity score s_j = max(0, y_j - Q_base(x_j))  (us)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Calibration-set nonconformity score distribution\n(dashed = q_conf threshold used)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    rows = []
    for split in ["test_id", "test_ood"]:
        df = pd.read_csv(CAL_DIR / f"calibrated_{split}.csv")
        for crit in ["HI", "LO"]:
            val = df.loc[df["criticality"] == crit, "inflation_us"].iloc[0]
            rows.append({"split": split, "criticality": crit, "inflation_us": val})
    inf_df = pd.DataFrame(rows)
    splits = inf_df["split"].unique()
    x = np.arange(len(splits))
    width = 0.35
    for i, crit in enumerate(["HI", "LO"]):
        vals = [inf_df[(inf_df["split"] == s) & (inf_df["criticality"] == crit)]["inflation_us"].iloc[0]
                for s in splits]
        bars = ax2.bar(x + (i - 0.5) * width, vals, width, label=crit, color=colors[crit])
        ax2.bar_label(bars, fmt="%.0f", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.set_ylabel("applied inflation / q_conf (us)")
    ax2.set_title("Resulting calibration correction actually applied\n(constant per criticality, by construction)")
    ax2.legend()

    savefig(fig, "inflation_distribution.png")


def plot_scalability():
    df = pd.read_csv(SCALE_DIR / "core_scalability_summary.csv")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["active_core_count"], df["mean_Q50_us"], "o-", label="mean Q50 (raw)")
    ax1.plot(df["active_core_count"], df["mean_C_e_us"], "s-", label="mean C_e (calibrated)")
    ax1.set_xlabel("active_core_count (scenario)")
    ax1.set_ylabel("predicted execution time (us)")
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(df["active_core_count"], df["active_core_count_zscore"], "^--", color="firebrick",
              label="standardized z-score (OOD distance)")
    ax2.set_ylabel("active_core_count z-score (train-fitted)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    for _, r in df.iterrows():
        ax1.axvline(r["active_core_count"], color="gray", alpha=0.15, lw=8)
    ax1.set_title("4/8/16/32-core scalability stress: prediction shift vs. OOD distance")
    savefig(fig, "scalability_core_count.png")


def plot_ood_global_comparison():
    df = pd.read_csv(OOD_DIR / "test_id_vs_test_ood_global_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(df["eval_split"], df["mean_pinball"], color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Mean pinball loss")
    axes[0].set_ylabel("us")
    axes[1].bar(df["eval_split"], df["mae"], color=["#4C72B0", "#C44E52"])
    axes[1].set_title("Q50 MAE")
    axes[1].set_ylabel("us")
    fig.suptitle("test_id vs. test_ood (501-1000 node graphs) global accuracy")
    savefig(fig, "ood_global_comparison.png")


def plot_dvfs_sensitivity():
    df = pd.read_csv(OOD_DIR / "grouped_ood_metrics_by_dvfs_level.csv")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(df["dvfs_level"].astype(str), df["mean_pinball"], color="#4C72B0", alpha=0.8)
    ax1.set_xlabel("DVFS level (0=lowest freq, 3=highest freq)")
    ax1.set_ylabel("mean pinball loss (us)", color="#4C72B0")
    ax1.set_title("DVFS sensitivity: prediction accuracy vs. DVFS level (test_ood)")
    savefig(fig, "dvfs_sensitivity.png")


def plot_graph_size_sensitivity():
    df = pd.read_csv(OOD_DIR / "accuracy_vs_graph_size.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df["num_nodes_bin"].astype(str), df["mean_pinball"], "o-", label="mean pinball")
    ax2 = ax.twinx()
    ax2.plot(df["num_nodes_bin"].astype(str), df["mae"], "s--", color="firebrick", label="MAE")
    ax.set_xlabel("graph size bin (num_nodes)")
    ax.set_ylabel("mean pinball (us)")
    ax2.set_ylabel("MAE (us)", color="firebrick")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Accuracy vs. graph size (test_id + test_ood combined)")
    savefig(fig, "accuracy_vs_graph_size.png")


def plot_critical_path_coverage():
    df = pd.read_csv(OOD_DIR / "critical_path_accuracy_comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    splits = df["eval_split"].unique()
    x = np.arange(len(splits))
    for i, val in enumerate([False, True]):
        vals = [df[(df["eval_split"] == s) & (df["is_critical_path_node"] == val)]["coverage_Q95"].iloc[0]
                for s in splits]
        ax.bar(x + (i - 0.5) * width, vals, width, label="critical-path node" if val else "other node")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.set_ylabel("empirical coverage @ Q95")
    ax.set_title("Coverage: critical-path vs. non-critical-path nodes")
    ax.legend()
    savefig(fig, "critical_path_coverage.png")


def plot_scheduling_comparison():
    df = pd.read_csv(SCHED_DIR / "scheduling_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    estimators = df["cost_estimator"].unique()
    x = np.arange(len(estimators))
    width = 0.35
    for i, sched in enumerate(df["scheduler"].unique()):
        sub = df[df["scheduler"] == sched].set_index("cost_estimator").reindex(estimators)
        axes[0].bar(x + (i - 0.5) * width, sub["DMR"], width, label=sched)
        axes[1].bar(x + (i - 0.5) * width, sub["mean_tardiness_us"] / 1e6, width, label=sched)
    for ax, title, ylabel in [(axes[0], "Deadline Miss Ratio", "DMR"),
                               (axes[1], "Mean tardiness", "seconds")]:
        ax.set_xticks(x)
        ax.set_xticklabels(estimators, rotation=20)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Scheduling outcome by scheduler x cost estimator (test_id, isolated execution)")
    savefig(fig, "scheduling_comparison.png")


def plot_model_comparison():
    df = pd.read_csv(BASE_DIR / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["model"], df["mae"], color=["#DD8452", "#55A868", "#4C72B0"])
    ax.set_ylabel("MAE (us)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_title("Baseline comparison: MAE on test_id")
    for i, v in enumerate(df["mae"]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    savefig(fig, "model_comparison_mae.png")


def plot_static_vs_sliding():
    df = pd.read_csv(DRIFT_DIR / "static_vs_sliding_window_calibration.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["static_coverage"], width, label="static calibration")
    ax.bar(x + width / 2, df["sliding_window_coverage"], width, label="sliding-window (causal)")
    for i, r in df.iterrows():
        ax.axhline(r["target_coverage"], color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["criticality"])
    ax.set_ylabel("empirical coverage (test_ood)")
    ax.set_title("Static vs. sliding-window conformal calibration under drift")
    ax.legend()
    savefig(fig, "static_vs_sliding_window_calibration.png")


if __name__ == "__main__":
    plot_violin_by_role()
    plot_violin_by_hardware_scenario()
    plot_box_actual_vs_predicted()
    plot_reliability_diagram()
    plot_coverage_before_after()
    plot_inflation_hist()
    plot_scalability()
    plot_ood_global_comparison()
    plot_dvfs_sensitivity()
    plot_graph_size_sensitivity()
    plot_critical_path_coverage()
    plot_scheduling_comparison()
    plot_model_comparison()
    plot_static_vs_sliding()
    print("\nAll Phase 2 plots saved to", PLOTS_DIR)
