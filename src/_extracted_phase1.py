# --- from TODO 2.1 ---
# ============================================================
# TODO 2.1 — Imports and reproducibility
# ============================================================

import sys
import json
import random
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch_geometric


# ------------------------------------------------------------
# Global random seed
# ------------------------------------------------------------

SEED = 20260807

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# Reproducibility settings
# ------------------------------------------------------------

# These settings improve reproducibility when CUDA/cuDNN is used.
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------

print("=" * 60)
print("Phase 1 Environment")
print("=" * 60)

print(f"Python version        : {platform.python_version()}")
print(f"PyTorch version       : {torch.__version__}")
print(f"PyG version           : {torch_geometric.__version__}")
print(f"NumPy version         : {np.__version__}")
print(f"Pandas version        : {pd.__version__}")

print("-" * 60)

print(f"Random seed           : {SEED}")
print(f"Selected device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"CUDA available        : True")
    print(f"CUDA version          : {torch.version.cuda}")
    print(f"GPU                    : {torch.cuda.get_device_name(0)}")
    print(f"GPU count              : {torch.cuda.device_count()}")
else:
    print("CUDA available        : False")
    print("GPU                    : CPU only")

print("=" * 60)

# --- from TODO 2.2 ---
# ============================================================
# TODO 2.2 — Project paths and artifact directories
# ============================================================


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find the project root automatically.

    The project root is defined as the directory containing:
        data/dag_runtime_dataset_25k

    This makes the notebook work whether Jupyter is started from:
        PROJECT_ROOT/
    or:
        PROJECT_ROOT/notebooks/
    """

    if start_path is None:
        start_path = Path.cwd()

    start_path = start_path.resolve()

    candidates = [
        start_path,
        *start_path.parents,
    ]

    for candidate in candidates:
        expected_dataset = (
            candidate
            / "data"
            / "dag_runtime_dataset_25k"
        )

        if expected_dataset.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root.\n"
        "Expected to find:\n"
        "data/dag_runtime_dataset_25k\n"
        f"Starting search from: {start_path}"
    )


# ------------------------------------------------------------
# Main project paths
# ------------------------------------------------------------

PROJECT_ROOT = find_project_root()

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dag_runtime_dataset_25k"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "phase1"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase1"
)


# ------------------------------------------------------------
# Model artifact directories
# ------------------------------------------------------------

CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"

PREPROCESSING_DIR = MODEL_ROOT / "preprocessing"

CONFIG_DIR = MODEL_ROOT / "config"


# ------------------------------------------------------------
# Result directories
# ------------------------------------------------------------

HISTORY_DIR = RESULT_ROOT / "history"

METRICS_DIR = RESULT_ROOT / "metrics"

PREDICTION_DIR = RESULT_ROOT / "predictions"

PLOT_DIR = RESULT_ROOT / "plots"


# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

OUTPUT_DIRECTORIES = [
    CHECKPOINT_DIR,
    PREPROCESSING_DIR,
    CONFIG_DIR,
    HISTORY_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    PLOT_DIR,
]

for directory in OUTPUT_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# Important dataset paths
# ------------------------------------------------------------

RECORDS_DIR = DATA_ROOT / "records"

NODE_FEATURES_DIR = (
    DATA_ROOT
    / "features"
    / "nodes"
)

EDGE_FEATURES_DIR = (
    DATA_ROOT
    / "features"
    / "edges"
)

GRAPH_DIR = DATA_ROOT / "graphs"

SCHEMA_DIR = DATA_ROOT / "schema"

TASKS_DIR = DATA_ROOT / "tasks"

HARDWARE_DIR = DATA_ROOT / "hardware"

METADATA_DIR = DATA_ROOT / "metadata"


# ------------------------------------------------------------
# Official split files
# ------------------------------------------------------------

TRAIN_RECORDS_PATH = (
    RECORDS_DIR / "train.csv"
)

VALIDATION_RECORDS_PATH = (
    RECORDS_DIR / "validation.csv"
)

CALIBRATION_RECORDS_PATH = (
    RECORDS_DIR / "calibration.csv"
)

TEST_ID_RECORDS_PATH = (
    RECORDS_DIR / "test_id.csv"
)

TEST_OOD_RECORDS_PATH = (
    RECORDS_DIR / "test_ood.csv"
)


# ------------------------------------------------------------
# Dataset metadata files
# ------------------------------------------------------------

FEATURE_SCHEMA_PATH = (
    SCHEMA_DIR
    / "model_feature_schema.json"
)

GENERATION_CONFIG_PATH = (
    DATA_ROOT
    / "generation_config.json"
)


# ------------------------------------------------------------
# Validate required dataset files/directories
# ------------------------------------------------------------

REQUIRED_PATHS = {
    "Dataset root": DATA_ROOT,

    "Train records":
        TRAIN_RECORDS_PATH,

    "Validation records":
        VALIDATION_RECORDS_PATH,

    "Calibration records":
        CALIBRATION_RECORDS_PATH,

    "Test-ID records":
        TEST_ID_RECORDS_PATH,

    "Test-OOD records":
        TEST_OOD_RECORDS_PATH,

    "Node features":
        NODE_FEATURES_DIR,

    "Edge features":
        EDGE_FEATURES_DIR,

    "Graphs":
        GRAPH_DIR,

    "Feature schema":
        FEATURE_SCHEMA_PATH,

    "Generation config":
        GENERATION_CONFIG_PATH,
}


missing_paths = []

for name, path in REQUIRED_PATHS.items():

    if not path.exists():
        missing_paths.append(
            (name, path)
        )


if missing_paths:

    message = [
        "The following required dataset paths are missing:"
    ]

    for name, path in missing_paths:
        message.append(
            f"  - {name}: {path}"
        )

    raise FileNotFoundError(
        "\n".join(message)
    )


# ------------------------------------------------------------
# Print resolved project structure
# ------------------------------------------------------------

print("=" * 70)
print("Phase 1 Project Paths")
print("=" * 70)

print(f"Project root          : {PROJECT_ROOT}")
print(f"Dataset root          : {DATA_ROOT}")

print("-" * 70)

print(f"Model root            : {MODEL_ROOT}")
print(f"Checkpoint directory  : {CHECKPOINT_DIR}")
print(f"Preprocessing dir     : {PREPROCESSING_DIR}")
print(f"Config directory      : {CONFIG_DIR}")

print("-" * 70)

print(f"Results root          : {RESULT_ROOT}")
print(f"History directory     : {HISTORY_DIR}")
print(f"Metrics directory     : {METRICS_DIR}")
print(f"Prediction directory  : {PREDICTION_DIR}")
print(f"Plot directory        : {PLOT_DIR}")

print("-" * 70)

print("All required dataset paths were found successfully.")

print("=" * 70)

# --- from TODO 3.1 ---
# ============================================================
# TODO 3.1 — Load and verify the official feature contract
# ============================================================

# ------------------------------------------------------------
# Expected model feature contract
# ------------------------------------------------------------

EXPECTED_NODE_FEATURES = [
    "compute_cycles",
    "memory_bytes",
    "input_size_bytes",
    "in_degree",
    "out_degree",
    "operation_type",
    "criticality",
    "topo_level",
    "reverse_topo_level",
]

EXPECTED_EDGE_FEATURES = [
    "data_bytes",
]

EXPECTED_CORE_FEATURES = [
    "core_type",
]

EXPECTED_DVFS_FEATURES = [
    "frequency_ghz",
    "voltage_v",
]

EXPECTED_ZT_FEATURES = [
    "cpu_utilization",
    "ready_queue_length",
    "active_core_count",
    "memory_active_tasks",
    "bus_utilization",
    "thermal_pressure",
    "release_jitter_us",
]

EXPECTED_TARGET = "y_exec_us"


# These columns may exist in the dataset, but they must never
# become model inputs.
FORBIDDEN_MODEL_INPUTS = {
    "lambda_v",
    "period_us",
    "deadline_us",
    "core_id",
    "generator_load_score",
    "target_role",
    "target_index",
    "context_index",
    "context_id",
}


# ------------------------------------------------------------
# Load official schema
# ------------------------------------------------------------

with FEATURE_SCHEMA_PATH.open("r", encoding="utf-8") as f:
    feature_schema = json.load(f)


# ------------------------------------------------------------
# Verify required schema keys
# ------------------------------------------------------------

REQUIRED_SCHEMA_KEYS = [
    "gnn_node_features",
    "gnn_edge_features",
    "quantile_head_core_features",
    "quantile_head_dvfs_features",
    "z_t_features",
    "target",
    "not_model_inputs",
]

missing_schema_keys = [
    key
    for key in REQUIRED_SCHEMA_KEYS
    if key not in feature_schema
]

if missing_schema_keys:
    raise KeyError(
        "The feature schema is missing required keys:\n"
        + "\n".join(
            f"  - {key}"
            for key in missing_schema_keys
        )
    )


# ------------------------------------------------------------
# Verify exact feature lists
# ------------------------------------------------------------

def assert_exact_feature_list(actual, expected, section_name):
    """
    Require exact agreement in both feature names and order.
    """

    if actual != expected:
        raise ValueError(
            f"Feature contract mismatch in '{section_name}'.\n\n"
            f"Expected:\n{expected}\n\n"
            f"Found:\n{actual}"
        )


assert_exact_feature_list(
    feature_schema["gnn_node_features"],
    EXPECTED_NODE_FEATURES,
    "gnn_node_features",
)

assert_exact_feature_list(
    feature_schema["gnn_edge_features"],
    EXPECTED_EDGE_FEATURES,
    "gnn_edge_features",
)

assert_exact_feature_list(
    feature_schema["quantile_head_core_features"],
    EXPECTED_CORE_FEATURES,
    "quantile_head_core_features",
)

assert_exact_feature_list(
    feature_schema["quantile_head_dvfs_features"],
    EXPECTED_DVFS_FEATURES,
    "quantile_head_dvfs_features",
)

assert_exact_feature_list(
    feature_schema["z_t_features"],
    EXPECTED_ZT_FEATURES,
    "z_t_features",
)


# ------------------------------------------------------------
# Verify target
# ------------------------------------------------------------

if feature_schema["target"] != EXPECTED_TARGET:
    raise ValueError(
        "Target mismatch.\n"
        f"Expected: {EXPECTED_TARGET}\n"
        f"Found: {feature_schema['target']}"
    )


# ------------------------------------------------------------
# Verify non-model-input schema section
# ------------------------------------------------------------

schema_not_model_inputs = feature_schema["not_model_inputs"]

if not isinstance(schema_not_model_inputs, list):
    raise TypeError(
        "'not_model_inputs' must be a list in the feature schema."
    )


# ------------------------------------------------------------
# Build the explicit model-input whitelist
# ------------------------------------------------------------

MODEL_NODE_FEATURES = EXPECTED_NODE_FEATURES.copy()

MODEL_EDGE_FEATURES = EXPECTED_EDGE_FEATURES.copy()

MODEL_CONTEXT_FEATURES = (
    EXPECTED_CORE_FEATURES
    + EXPECTED_DVFS_FEATURES
    + EXPECTED_ZT_FEATURES
)

MODEL_TARGET = EXPECTED_TARGET


# ------------------------------------------------------------
# Leakage protection for the model whitelist
# ------------------------------------------------------------

all_model_inputs = (
    MODEL_NODE_FEATURES
    + MODEL_EDGE_FEATURES
    + MODEL_CONTEXT_FEATURES
)

accidentally_included_forbidden = [
    column
    for column in all_model_inputs
    if (
        column in FORBIDDEN_MODEL_INPUTS
        or column.startswith("audit_")
    )
]

if accidentally_included_forbidden:
    raise ValueError(
        "Forbidden simulator/metadata/audit features were found "
        "in the model input whitelist:\n"
        + "\n".join(
            f"  - {column}"
            for column in accidentally_included_forbidden
        )
    )


# ------------------------------------------------------------
# Find sample training files
# ------------------------------------------------------------

train_node_files = sorted(
    (NODE_FEATURES_DIR / "train").glob("*.csv")
)

train_edge_files = sorted(
    (EDGE_FEATURES_DIR / "train").glob("*.csv")
)

if not train_node_files:
    raise FileNotFoundError(
        "No training node feature CSV files were found."
    )

if not train_edge_files:
    raise FileNotFoundError(
        "No training edge feature CSV files were found."
    )


# ------------------------------------------------------------
# Read small samples from the actual dataset
# ------------------------------------------------------------

sample_node_df = pd.read_csv(
    train_node_files[0],
    nrows=5,
)

sample_edge_df = pd.read_csv(
    train_edge_files[0],
    nrows=5,
)

sample_records_df = pd.read_csv(
    TRAIN_RECORDS_PATH,
    nrows=5,
)


# ------------------------------------------------------------
# Check required node columns
# ------------------------------------------------------------

missing_node_columns = [
    column
    for column in MODEL_NODE_FEATURES
    if column not in sample_node_df.columns
]

if missing_node_columns:
    raise ValueError(
        "Missing required node feature columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_node_columns
        )
    )


# ------------------------------------------------------------
# Check required edge columns
# ------------------------------------------------------------

missing_edge_columns = [
    column
    for column in MODEL_EDGE_FEATURES
    if column not in sample_edge_df.columns
]

if missing_edge_columns:
    raise ValueError(
        "Missing required edge feature columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_edge_columns
        )
    )


# ------------------------------------------------------------
# Check required execution-context columns
# ------------------------------------------------------------

missing_context_columns = [
    column
    for column in MODEL_CONTEXT_FEATURES
    if column not in sample_records_df.columns
]

if missing_context_columns:
    raise ValueError(
        "Missing required execution-context columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_context_columns
        )
    )


# ------------------------------------------------------------
# Check target column
# ------------------------------------------------------------

if MODEL_TARGET not in sample_records_df.columns:
    raise ValueError(
        f"Missing target column: {MODEL_TARGET}"
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Feature Contract Verification")
print("=" * 70)

print("\nNode features:")
for feature in MODEL_NODE_FEATURES:
    print(f"  - {feature}")

print("\nEdge features:")
for feature in MODEL_EDGE_FEATURES:
    print(f"  - {feature}")

print("\nExecution-context features:")
for feature in MODEL_CONTEXT_FEATURES:
    print(f"  - {feature}")

print("\nTarget:")
print(f"  - {MODEL_TARGET}")

print("\nLeakage protection:")
print("  - Simulator, metadata, and audit variables are excluded.")
print("  - Model inputs are controlled by an explicit whitelist.")

print("\nSchema status:")
print("  - Official feature schema matches the expected Phase 1 contract.")
print("  - Required dataset columns were found successfully.")

print("=" * 70)

# --- from TODO 4.1 ---
# ============================================================
# TODO 4.1 — Load split record tables
# ============================================================

# ------------------------------------------------------------
# Official split paths and expected record counts
# ------------------------------------------------------------

SPLIT_PATHS = {
    "train": TRAIN_RECORDS_PATH,
    "validation": VALIDATION_RECORDS_PATH,
    "calibration": CALIBRATION_RECORDS_PATH,
    "test_id": TEST_ID_RECORDS_PATH,
    "test_ood": TEST_OOD_RECORDS_PATH,
}

EXPECTED_RECORD_COUNTS = {
    "train": 16800,
    "validation": 2100,
    "calibration": 2100,
    "test_id": 2100,
    "test_ood": 2100,
}


# ------------------------------------------------------------
# Verify that all official split files exist
# ------------------------------------------------------------

for split_name, split_path in SPLIT_PATHS.items():
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing record file for split '{split_name}':\n"
            f"{split_path}"
        )


# ------------------------------------------------------------
# Load Phase 1 record tables
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_RECORDS_PATH)
validation_df = pd.read_csv(VALIDATION_RECORDS_PATH)
test_id_df = pd.read_csv(TEST_ID_RECORDS_PATH)


# ------------------------------------------------------------
# Read only graph_id from reserved Phase 2 splits
# ------------------------------------------------------------

calibration_graph_df = pd.read_csv(
    CALIBRATION_RECORDS_PATH,
    usecols=["graph_id"],
)

test_ood_graph_df = pd.read_csv(
    TEST_OOD_RECORDS_PATH,
    usecols=["graph_id"],
)


# ------------------------------------------------------------
# Verify expected record counts
# ------------------------------------------------------------

actual_record_counts = {
    "train": len(train_df),
    "validation": len(validation_df),
    "calibration": len(calibration_graph_df),
    "test_id": len(test_id_df),
    "test_ood": len(test_ood_graph_df),
}

for split_name, expected_count in EXPECTED_RECORD_COUNTS.items():
    actual_count = actual_record_counts[split_name]

    if actual_count != expected_count:
        raise ValueError(
            f"Unexpected record count for split '{split_name}'.\n"
            f"Expected: {expected_count}\n"
            f"Found:    {actual_count}"
        )


# ------------------------------------------------------------
# Verify graph_id column exists in active Phase 1 tables
# ------------------------------------------------------------

ACTIVE_TABLES = {
    "train": train_df,
    "validation": validation_df,
    "test_id": test_id_df,
}

for split_name, df in ACTIVE_TABLES.items():
    if "graph_id" not in df.columns:
        raise KeyError(
            f"'graph_id' column is missing from split '{split_name}'."
        )


# ------------------------------------------------------------
# Build graph_id sets for all official splits
# ------------------------------------------------------------

graph_id_sets = {
    "train": set(train_df["graph_id"].unique()),
    "validation": set(validation_df["graph_id"].unique()),
    "calibration": set(calibration_graph_df["graph_id"].unique()),
    "test_id": set(test_id_df["graph_id"].unique()),
    "test_ood": set(test_ood_graph_df["graph_id"].unique()),
}


# ------------------------------------------------------------
# Verify that graph_id sets are pairwise disjoint
# ------------------------------------------------------------

split_names = list(graph_id_sets.keys())

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        split_a = split_names[i]
        split_b = split_names[j]

        overlap = graph_id_sets[split_a] & graph_id_sets[split_b]

        if overlap:
            overlap_preview = sorted(overlap)[:10]

            raise ValueError(
                f"Graph leakage detected between '{split_a}' "
                f"and '{split_b}'.\n"
                f"Number of overlapping graph_ids: {len(overlap)}\n"
                f"Example overlapping graph_ids: {overlap_preview}"
            )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Official Split Verification")
print("=" * 70)

for split_name in split_names:
    print(
        f"{split_name:<12} : "
        f"{actual_record_counts[split_name]:>5} records | "
        f"{len(graph_id_sets[split_name]):>3} unique graphs"
    )

print("\nSplit integrity:")
print("  - All official record files exist.")
print("  - All record counts match the expected values.")
print("  - graph_id sets are disjoint across all official splits.")

print("\nPhase 1 loaded tables:")
print("  - train_df")
print("  - validation_df")
print("  - test_id_df")

print("\nReserved splits:")
print("  - calibration.csv verified but not loaded as a Phase 1 table.")
print("  - test_ood.csv verified but not loaded as a Phase 1 table.")

print("=" * 70)

# --- from TODO 5.1 ---
# ============================================================
# TODO 5.1 — Fit preprocessing on TRAIN only
# ============================================================

# IMPORTANT:
# Every preprocessing statistic and categorical mapping in this
# cell is fitted using TRAIN data only.
#
# Validation, calibration, test-ID, and test-OOD must not
# influence preprocessing parameters.


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if "train_df" not in globals():
    raise RuntimeError(
        "train_df is not available. Run TODO 4.1 first."
    )

if "MODEL_NODE_FEATURES" not in globals():
    raise RuntimeError(
        "MODEL_NODE_FEATURES is not available. Run TODO 3.1 first."
    )

if "MODEL_EDGE_FEATURES" not in globals():
    raise RuntimeError(
        "MODEL_EDGE_FEATURES is not available. Run TODO 3.1 first."
    )

if "MODEL_CONTEXT_FEATURES" not in globals():
    raise RuntimeError(
        "MODEL_CONTEXT_FEATURES is not available. Run TODO 3.1 first."
    )

if "MODEL_TARGET" not in globals():
    raise RuntimeError(
        "MODEL_TARGET is not available. Run TODO 3.1 first."
    )


# ------------------------------------------------------------
# Define preprocessing groups
# ------------------------------------------------------------

# Large positive-valued node features.
# These are log-transformed first, then standardized.
NODE_LOG1P_FEATURES = [
    "compute_cycles",
    "memory_bytes",
    "input_size_bytes",
]

# Structural numerical node features.
# These are standardized directly.
NODE_STANDARD_FEATURES = [
    "in_degree",
    "out_degree",
    "topo_level",
    "reverse_topo_level",
]

# Categorical node features.
NODE_CATEGORICAL_FEATURES = [
    "operation_type",
    "criticality",
]

# Edge communication volume.
EDGE_LOG1P_FEATURES = [
    "data_bytes",
]

# Core category.
CORE_CATEGORICAL_FEATURES = [
    "core_type",
]

# DVFS numerical context.
DVFS_STANDARD_FEATURES = [
    "frequency_ghz",
    "voltage_v",
]

# Dynamic system-state features z_t.
ZT_STANDARD_FEATURES = [
    "cpu_utilization",
    "ready_queue_length",
    "active_core_count",
    "memory_active_tasks",
    "bus_utilization",
    "thermal_pressure",
    "release_jitter_us",
]


# ------------------------------------------------------------
# Verify preprocessing groups against the feature contract
# ------------------------------------------------------------

def verify_feature_partition(
    contract_features,
    preprocessing_groups,
    section_name,
):
    """
    Verify that preprocessing groups contain exactly the same
    features as the official feature contract.

    Grouping order is intentionally ignored because preprocessing
    groups may organize numerical and categorical features
    differently from the final model feature order.
    """

    grouped_features = []

    for group in preprocessing_groups:
        grouped_features.extend(group)

    # Detect duplicated features inside preprocessing groups.
    duplicated_features = sorted({
        feature
        for feature in grouped_features
        if grouped_features.count(feature) > 1
    })

    if duplicated_features:
        raise ValueError(
            f"Duplicate features found in preprocessing definition "
            f"for '{section_name}':\n"
            + "\n".join(
                f"  - {feature}"
                for feature in duplicated_features
            )
        )

    contract_set = set(contract_features)
    grouped_set = set(grouped_features)

    missing_features = sorted(
        contract_set - grouped_set
    )

    extra_features = sorted(
        grouped_set - contract_set
    )

    if missing_features or extra_features:

        message = (
            f"Preprocessing definition does not match the official "
            f"feature contract for '{section_name}'."
        )

        if missing_features:
            message += (
                "\n\nMissing features:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing_features
                )
            )

        if extra_features:
            message += (
                "\n\nUnexpected features:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in extra_features
                )
            )

        raise ValueError(message)


verify_feature_partition(
    contract_features=MODEL_NODE_FEATURES,
    preprocessing_groups=[
        NODE_LOG1P_FEATURES,
        NODE_STANDARD_FEATURES,
        NODE_CATEGORICAL_FEATURES,
    ],
    section_name="node features",
)

verify_feature_partition(
    contract_features=MODEL_EDGE_FEATURES,
    preprocessing_groups=[
        EDGE_LOG1P_FEATURES,
    ],
    section_name="edge features",
)

verify_feature_partition(
    contract_features=MODEL_CONTEXT_FEATURES,
    preprocessing_groups=[
        CORE_CATEGORICAL_FEATURES,
        DVFS_STANDARD_FEATURES,
        ZT_STANDARD_FEATURES,
    ],
    section_name="execution-context features",
)


# ------------------------------------------------------------
# Training graph IDs
# ------------------------------------------------------------

train_graph_ids = set(
    train_df["graph_id"].astype(str).unique()
)

if not train_graph_ids:
    raise ValueError(
        "No training graph IDs were found in train_df."
    )


# ------------------------------------------------------------
# Helper: load TRAIN node/edge CSV files
# ------------------------------------------------------------

def load_training_feature_files(
    directory,
    required_columns,
    expected_graph_ids,
    table_name,
):
    """
    Load graph-level CSV files from the training split only.

    The function also verifies that:
      1. required columns exist,
      2. files are not empty,
      3. no non-training graph appears,
      4. every expected training graph is present.
    """

    csv_files = sorted(
        directory.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found for {table_name}:\n"
            f"{directory}"
        )

    frames = []
    seen_graph_ids = set()

    required_with_id = [
        "graph_id",
        *required_columns,
    ]

    for path in csv_files:

        # Read only the header first for a cheap schema check.
        header = pd.read_csv(
            path,
            nrows=0,
        )

        missing_columns = [
            column
            for column in required_with_id
            if column not in header.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {path.name}:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing_columns
                )
            )

        # Load only columns relevant to preprocessing.
        df = pd.read_csv(
            path,
            usecols=required_with_id,
        )

        if df.empty:
            raise ValueError(
                f"Training {table_name} file is empty:\n"
                f"{path}"
            )

        file_graph_ids = set(
            df["graph_id"]
            .astype(str)
            .unique()
        )

        unexpected_graph_ids = (
            file_graph_ids
            - expected_graph_ids
        )

        if unexpected_graph_ids:
            raise ValueError(
                f"{table_name} file contains graph IDs that do "
                f"not belong to the training split:\n"
                f"{path}\n"
                f"Examples: "
                f"{sorted(unexpected_graph_ids)[:10]}"
            )

        seen_graph_ids.update(
            file_graph_ids
        )

        frames.append(df)

    missing_graph_ids = (
        expected_graph_ids
        - seen_graph_ids
    )

    if missing_graph_ids:
        raise ValueError(
            f"Some training graphs are missing from "
            f"{table_name} files.\n"
            f"Missing count: {len(missing_graph_ids)}\n"
            f"Examples: "
            f"{sorted(missing_graph_ids)[:10]}"
        )

    combined_df = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined_df


# ------------------------------------------------------------
# Load TRAIN node and edge feature tables
# ------------------------------------------------------------

train_nodes_df = load_training_feature_files(
    directory=NODE_FEATURES_DIR / "train",
    required_columns=MODEL_NODE_FEATURES,
    expected_graph_ids=train_graph_ids,
    table_name="node feature",
)

train_edges_df = load_training_feature_files(
    directory=EDGE_FEATURES_DIR / "train",
    required_columns=MODEL_EDGE_FEATURES,
    expected_graph_ids=train_graph_ids,
    table_name="edge feature",
)


# ------------------------------------------------------------
# Helper: validate numerical columns
# ------------------------------------------------------------

def get_clean_numeric_array(
    df,
    column,
    require_nonnegative=False,
):
    """
    Convert a numerical column to float64 and verify that
    all values are finite and valid for preprocessing.
    """

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if np.isnan(values).any():
        raise ValueError(
            f"Column '{column}' contains missing "
            "or non-numeric values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Column '{column}' contains non-finite values."
        )

    if (
        require_nonnegative
        and (values < 0).any()
    ):
        raise ValueError(
            f"Column '{column}' contains negative values, "
            "but log1p preprocessing requires "
            "non-negative values."
        )

    return values


# ------------------------------------------------------------
# Helper: fit numerical preprocessing statistics
# ------------------------------------------------------------

def fit_standardizer(
    df,
    columns,
    use_log1p=False,
):
    """
    Fit training-only mean and standard deviation.

    If use_log1p=True:

        x -> log(1 + x) -> standardization

    Otherwise:

        x -> standardization
    """

    fitted = {}

    for column in columns:

        values = get_clean_numeric_array(
            df=df,
            column=column,
            require_nonnegative=use_log1p,
        )

        if use_log1p:

            transformed_values = np.log1p(
                values
            )

            transform_name = (
                "log1p_standardize"
            )

        else:

            transformed_values = values

            transform_name = (
                "standardize"
            )

        mean = float(
            transformed_values.mean()
        )

        std = float(
            transformed_values.std(
                ddof=0
            )
        )

        if not np.isfinite(mean):
            raise ValueError(
                f"Invalid fitted mean for '{column}'."
            )

        if not np.isfinite(std):
            raise ValueError(
                f"Invalid fitted standard deviation "
                f"for '{column}'."
            )

        if std <= 1e-12:
            raise ValueError(
                f"Column '{column}' has near-zero variance "
                "in the training split."
            )

        fitted[column] = {
            "transform": transform_name,
            "mean": mean,
            "std": std,
        }

    return fitted


# ------------------------------------------------------------
# Helper: fit categorical encoder
# ------------------------------------------------------------

def fit_category_encoder(
    series,
    feature_name,
):
    """
    Fit a deterministic integer encoder using training data only.

    Categories are converted to strings and sorted to make the
    mapping reproducible across runs.
    """

    if series.isna().any():
        raise ValueError(
            f"Categorical feature '{feature_name}' "
            "contains missing values."
        )

    values = series.astype(str)

    categories = sorted(
        values.unique().tolist()
    )

    if not categories:
        raise ValueError(
            f"No categories were found for "
            f"'{feature_name}'."
        )

    mapping = {
        category: index
        for index, category
        in enumerate(categories)
    }

    return {
        "encoding": "integer_index",
        "categories": categories,
        "mapping": mapping,
        "num_categories": len(categories),
    }


# ------------------------------------------------------------
# Fit node numerical preprocessing
# ------------------------------------------------------------

node_log1p_stats = fit_standardizer(
    df=train_nodes_df,
    columns=NODE_LOG1P_FEATURES,
    use_log1p=True,
)

node_standard_stats = fit_standardizer(
    df=train_nodes_df,
    columns=NODE_STANDARD_FEATURES,
    use_log1p=False,
)


# ------------------------------------------------------------
# Fit node categorical encoders
# ------------------------------------------------------------

operation_type_encoder = (
    fit_category_encoder(
        series=train_nodes_df[
            "operation_type"
        ],
        feature_name="operation_type",
    )
)

criticality_encoder = (
    fit_category_encoder(
        series=train_nodes_df[
            "criticality"
        ],
        feature_name="criticality",
    )
)


# ------------------------------------------------------------
# Fit edge preprocessing
# ------------------------------------------------------------

edge_stats = fit_standardizer(
    df=train_edges_df,
    columns=EDGE_LOG1P_FEATURES,
    use_log1p=True,
)


# ------------------------------------------------------------
# Verify required TRAIN record columns
# ------------------------------------------------------------

required_train_record_columns = (
    CORE_CATEGORICAL_FEATURES
    + DVFS_STANDARD_FEATURES
    + ZT_STANDARD_FEATURES
    + [MODEL_TARGET]
)

missing_train_record_columns = [
    column
    for column
    in required_train_record_columns
    if column not in train_df.columns
]

if missing_train_record_columns:
    raise ValueError(
        "Training record table is missing "
        "required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column
            in missing_train_record_columns
        )
    )


# ------------------------------------------------------------
# Fit core-type encoder
# ------------------------------------------------------------

core_type_encoder = (
    fit_category_encoder(
        series=train_df[
            "core_type"
        ],
        feature_name="core_type",
    )
)


# ------------------------------------------------------------
# Fit DVFS preprocessing
# ------------------------------------------------------------

dvfs_stats = fit_standardizer(
    df=train_df,
    columns=DVFS_STANDARD_FEATURES,
    use_log1p=False,
)


# ------------------------------------------------------------
# Fit dynamic system-state preprocessing
# ------------------------------------------------------------

zt_stats = fit_standardizer(
    df=train_df,
    columns=ZT_STANDARD_FEATURES,
    use_log1p=False,
)


# ------------------------------------------------------------
# Fit target transformation
# ------------------------------------------------------------

target_values = get_clean_numeric_array(
    df=train_df,
    column=MODEL_TARGET,
    require_nonnegative=True,
)

target_mean = float(
    target_values.mean()
)

target_std = float(
    target_values.std(
        ddof=0
    )
)

if not np.isfinite(target_mean):
    raise ValueError(
        f"Invalid target mean for '{MODEL_TARGET}'."
    )

if not np.isfinite(target_std):
    raise ValueError(
        f"Invalid target standard deviation "
        f"for '{MODEL_TARGET}'."
    )

if target_std <= 1e-12:
    raise ValueError(
        f"Target '{MODEL_TARGET}' "
        "has near-zero variance."
    )

target_stats = {
    MODEL_TARGET: {
        "transform": "standardize",
        "mean": target_mean,
        "std": target_std,
    }
}


# ------------------------------------------------------------
# Build complete preprocessing specification
# ------------------------------------------------------------

preprocessing_spec = {

    "version": "phase1_preprocessing_v1",

    "fit_split": "train",

    "feature_contract": {

        # Keep the official model feature order here.
        "node_features": (
            MODEL_NODE_FEATURES
        ),

        "edge_features": (
            MODEL_EDGE_FEATURES
        ),

        "context_features": (
            MODEL_CONTEXT_FEATURES
        ),

        "target": (
            MODEL_TARGET
        ),
    },

    "node": {

        "log1p_standardized": (
            node_log1p_stats
        ),

        "standardized": (
            node_standard_stats
        ),

        "categorical": {

            "operation_type": (
                operation_type_encoder
            ),

            "criticality": (
                criticality_encoder
            ),
        },
    },

    "edge": {

        "log1p_standardized": (
            edge_stats
        ),
    },

    "context": {

        "categorical": {

            "core_type": (
                core_type_encoder
            ),
        },

        "dvfs_standardized": (
            dvfs_stats
        ),

        "z_t_standardized": (
            zt_stats
        ),
    },

    "target": (
        target_stats
    ),

    "training_statistics": {

        "num_training_records": int(
            len(train_df)
        ),

        "num_training_graphs": int(
            len(train_graph_ids)
        ),

        "num_training_nodes": int(
            len(train_nodes_df)
        ),

        "num_training_edges": int(
            len(train_edges_df)
        ),
    },
}


# ------------------------------------------------------------
# Save preprocessing specification
# ------------------------------------------------------------

PREPROCESSING_SPEC_PATH = (
    PREPROCESSING_DIR
    / "preprocessing_spec.json"
)

with PREPROCESSING_SPEC_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        preprocessing_spec,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("TRAIN-Only Preprocessing Fit")
print("=" * 70)

print("\nTraining data used:")

print(
    f"  Records : "
    f"{len(train_df):,}"
)

print(
    f"  Graphs  : "
    f"{len(train_graph_ids):,}"
)

print(
    f"  Nodes   : "
    f"{len(train_nodes_df):,}"
)

print(
    f"  Edges   : "
    f"{len(train_edges_df):,}"
)


print("\nNode numerical transformations:")

for feature in NODE_LOG1P_FEATURES:
    print(
        f"  - {feature:<20} : "
        "log1p + standardization"
    )

for feature in NODE_STANDARD_FEATURES:
    print(
        f"  - {feature:<20} : "
        "standardization"
    )


print("\nNode categorical encoders:")

print(
    "  - operation_type :",
    operation_type_encoder[
        "mapping"
    ],
)

print(
    "  - criticality    :",
    criticality_encoder[
        "mapping"
    ],
)


print("\nEdge transformation:")

print(
    "  - data_bytes           : "
    "log1p + standardization"
)


print("\nContext preprocessing:")

print(
    "  - core_type       :",
    core_type_encoder[
        "mapping"
    ],
)

print(
    "  - DVFS            : "
    "standardization"
)

print(
    "  - z_t             : "
    "standardization"
)


print("\nTarget preprocessing:")

print(
    f"  - {MODEL_TARGET:<19} : "
    "standardization"
)


print("\nLeakage protection:")

print(
    "  - All preprocessing parameters "
    "were fitted on TRAIN only."
)

print(
    "  - Validation, calibration, test-ID, "
    "and test-OOD were not used."
)


print("\nSaved preprocessing specification:")

print(
    f"  {PREPROCESSING_SPEC_PATH}"
)

print("=" * 70)

# --- from TODO 5.2 ---
# ============================================================
# TODO 5.2 — Save preprocessing state and feature contract
# ============================================================

# This cell does NOT fit anything new.
# It only serializes the preprocessing state fitted in TODO 5.1
# and creates a stable Phase 1 -> Phase 2 feature contract.


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

required_objects = [
    "MODEL_NODE_FEATURES",
    "MODEL_EDGE_FEATURES",
    "MODEL_CONTEXT_FEATURES",
    "MODEL_TARGET",
    "node_log1p_stats",
    "node_standard_stats",
    "edge_stats",
    "dvfs_stats",
    "zt_stats",
    "operation_type_encoder",
    "criticality_encoder",
    "core_type_encoder",
    "target_stats",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "Required preprocessing objects are missing. "
        "Run TODO 5.1 first:\n"
        + "\n".join(
            f"  - {name}"
            for name in missing_objects
        )
    )


# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------

PREPROCESSING_STATE_PATH = (
    PREPROCESSING_DIR / "preprocessing_state.json"
)

FEATURE_CONTRACT_PATH = (
    PREPROCESSING_DIR / "feature_contract.json"
)


# ------------------------------------------------------------
# Build fitted preprocessing state
# ------------------------------------------------------------

preprocessing_state = {
    "version": "phase1_preprocessing_v1",
    "fit_split": "train",
    "random_seed": int(SEED),

    "node": {
        "log1p_standardized": node_log1p_stats,
        "standardized": node_standard_stats,

        "categorical": {
            "operation_type": operation_type_encoder,
            "criticality": criticality_encoder,
        },
    },

    "edge": {
        "log1p_standardized": edge_stats,
    },

    "context": {
        "categorical": {
            "core_type": core_type_encoder,
        },

        "dvfs_standardized": dvfs_stats,

        "z_t_standardized": zt_stats,
    },

    "target": target_stats,
}


# ------------------------------------------------------------
# Build stable feature contract
# ------------------------------------------------------------

feature_contract = {
    "version": "phase1_feature_contract_v1",

    # Feature order matters when tensors are constructed later.
    "ordered_node_features": list(
        MODEL_NODE_FEATURES
    ),

    "ordered_edge_features": list(
        MODEL_EDGE_FEATURES
    ),

    "ordered_context_features": list(
        MODEL_CONTEXT_FEATURES
    ),

    "categorical_vocabularies": {
        "operation_type": {
            "categories": list(
                operation_type_encoder["categories"]
            ),
            "mapping": dict(
                operation_type_encoder["mapping"]
            ),
        },

        "criticality": {
            "categories": list(
                criticality_encoder["categories"]
            ),
            "mapping": dict(
                criticality_encoder["mapping"]
            ),
        },

        "core_type": {
            "categories": list(
                core_type_encoder["categories"]
            ),
            "mapping": dict(
                core_type_encoder["mapping"]
            ),
        },
    },

    "target": {
        "name": MODEL_TARGET,

        "transformation": "standardize",

        "forward_description": (
            "y_scaled = (y_exec_us - train_mean) / train_std"
        ),

        "inverse_description": (
            "y_exec_us = y_scaled * train_std + train_mean"
        ),

        "statistics_source": "training split only",
    },

    "preprocessing_state_file": (
        PREPROCESSING_STATE_PATH.name
    ),
}


# ------------------------------------------------------------
# Helper: save JSON
# ------------------------------------------------------------

def save_json(data, path):
    """
    Save a JSON file with stable formatting.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

save_json(
    preprocessing_state,
    PREPROCESSING_STATE_PATH,
)

save_json(
    feature_contract,
    FEATURE_CONTRACT_PATH,
)


# ------------------------------------------------------------
# Reload and verify saved files
# ------------------------------------------------------------

with PREPROCESSING_STATE_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    saved_preprocessing_state = json.load(f)

with FEATURE_CONTRACT_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    saved_feature_contract = json.load(f)


# ------------------------------------------------------------
# Verify feature ordering survived serialization
# ------------------------------------------------------------

if (
    saved_feature_contract["ordered_node_features"]
    != MODEL_NODE_FEATURES
):
    raise ValueError(
        "Saved node feature order does not match "
        "the active feature contract."
    )

if (
    saved_feature_contract["ordered_edge_features"]
    != MODEL_EDGE_FEATURES
):
    raise ValueError(
        "Saved edge feature order does not match "
        "the active feature contract."
    )

if (
    saved_feature_contract["ordered_context_features"]
    != MODEL_CONTEXT_FEATURES
):
    raise ValueError(
        "Saved context feature order does not match "
        "the active feature contract."
    )


# ------------------------------------------------------------
# Verify categorical vocabularies survived serialization
# ------------------------------------------------------------

expected_vocabularies = {
    "operation_type": operation_type_encoder["mapping"],
    "criticality": criticality_encoder["mapping"],
    "core_type": core_type_encoder["mapping"],
}

for feature_name, expected_mapping in expected_vocabularies.items():

    saved_mapping = (
        saved_feature_contract[
            "categorical_vocabularies"
        ][feature_name]["mapping"]
    )

    if saved_mapping != expected_mapping:
        raise ValueError(
            f"Saved categorical mapping for "
            f"'{feature_name}' does not match "
            "the fitted training mapping."
        )


# ------------------------------------------------------------
# Verify target preprocessing state
# ------------------------------------------------------------

saved_target_state = (
    saved_preprocessing_state["target"]
)

if MODEL_TARGET not in saved_target_state:
    raise ValueError(
        f"Target '{MODEL_TARGET}' is missing from "
        "the saved preprocessing state."
    )

if (
    saved_target_state[MODEL_TARGET]["transform"]
    != "standardize"
):
    raise ValueError(
        "Unexpected target transformation in the "
        "saved preprocessing state."
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Phase 1 Preprocessing Interface Saved")
print("=" * 70)

print("\nPreprocessing state:")
print(f"  {PREPROCESSING_STATE_PATH}")

print("\nFeature contract:")
print(f"  {FEATURE_CONTRACT_PATH}")

print("\nOrdered feature dimensions:")
print(
    f"  Node features    : "
    f"{len(MODEL_NODE_FEATURES)}"
)
print(
    f"  Edge features    : "
    f"{len(MODEL_EDGE_FEATURES)}"
)
print(
    f"  Context features : "
    f"{len(MODEL_CONTEXT_FEATURES)}"
)

print("\nCategorical vocabularies:")
print(
    "  operation_type :",
    operation_type_encoder["mapping"],
)
print(
    "  criticality    :",
    criticality_encoder["mapping"],
)
print(
    "  core_type      :",
    core_type_encoder["mapping"],
)

print("\nTarget:")
print(f"  Name            : {MODEL_TARGET}")
print("  Transformation  : standardization")
print("  Statistics      : fitted on TRAIN only")

print("\nVerification:")
print("  - Files were saved successfully.")
print("  - Saved files were reloaded successfully.")
print("  - Feature ordering is preserved.")
print("  - Categorical mappings are preserved.")
print("  - Target transformation is preserved.")

print("=" * 70)

# --- from TODO 6.1 ---
# ============================================================
# TODO 6.1 — Implement static graph loader
# ============================================================

# This loader converts one static DAG into tensors that can later
# be consumed by the GNN encoder.
#
# Input:
#   - graph_id
#   - split
#   - fitted preprocessing state
#
# Output:
#   - x
#   - edge_index
#   - edge_attr
#   - node_id_to_index


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if "MODEL_NODE_FEATURES" not in globals():
    raise RuntimeError(
        "MODEL_NODE_FEATURES is not available. Run TODO 3.1 first."
    )

if "MODEL_EDGE_FEATURES" not in globals():
    raise RuntimeError(
        "MODEL_EDGE_FEATURES is not available. Run TODO 3.1 first."
    )

if "preprocessing_state" not in globals():
    raise RuntimeError(
        "preprocessing_state is not available. Run TODO 5.2 first."
    )


VALID_SPLITS = {
    "train",
    "validation",
    "calibration",
    "test_id",
    "test_ood",
}


# ------------------------------------------------------------
# Internal file-index cache
# ------------------------------------------------------------

# We do not want to repeatedly scan hundreds of graph files every
# time one graph is loaded. Therefore, each split directory is
# indexed once and cached.

_GRAPH_FILE_INDEX_CACHE = {}


def _canonical_graph_id(value):
    """
    Convert graph IDs to a consistent string representation.
    """

    if pd.isna(value):
        raise ValueError("Encountered a missing graph_id.")

    return str(value)


def _canonical_node_id(value):
    """
    Convert node IDs to a stable representation for edge lookup.
    """

    if pd.isna(value):
        raise ValueError("Encountered a missing node_id.")

    # Handle integer-like numerical node IDs cleanly.
    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(
                f"Invalid non-finite node ID: {value}"
            )

        if float(value).is_integer():
            return str(int(value))

    return str(value)


# ------------------------------------------------------------
# Build graph_id -> CSV path index
# ------------------------------------------------------------

def _build_graph_file_index(directory, table_name):
    """
    Build a mapping:

        graph_id -> CSV path

    for one split directory.

    Each graph-level CSV is expected to contain exactly one graph.
    """

    directory = Path(directory)

    cache_key = str(directory.resolve())

    if cache_key in _GRAPH_FILE_INDEX_CACHE:
        return _GRAPH_FILE_INDEX_CACHE[cache_key]

    if not directory.exists():
        raise FileNotFoundError(
            f"{table_name} directory does not exist:\n"
            f"{directory}"
        )

    csv_files = sorted(
        directory.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {table_name} directory:\n"
            f"{directory}"
        )

    graph_file_index = {}

    for path in csv_files:

        try:
            graph_id_sample = pd.read_csv(
                path,
                usecols=["graph_id"],
                nrows=10,
            )
        except ValueError as exc:
            raise ValueError(
                f"'graph_id' column is missing from:\n{path}"
            ) from exc

        if graph_id_sample.empty:
            raise ValueError(
                f"Empty {table_name} file:\n{path}"
            )

        file_graph_ids = {
            _canonical_graph_id(value)
            for value
            in graph_id_sample["graph_id"].unique()
        }

        if len(file_graph_ids) != 1:
            raise ValueError(
                f"{table_name} file must represent exactly one graph:\n"
                f"{path}\n"
                f"Found graph IDs: {sorted(file_graph_ids)}"
            )

        graph_id = next(
            iter(file_graph_ids)
        )

        if graph_id in graph_file_index:
            raise ValueError(
                f"Duplicate {table_name} files found for "
                f"graph_id '{graph_id}':\n"
                f"  - {graph_file_index[graph_id]}\n"
                f"  - {path}"
            )

        graph_file_index[graph_id] = path

    _GRAPH_FILE_INDEX_CACHE[cache_key] = (
        graph_file_index
    )

    return graph_file_index


# ------------------------------------------------------------
# Resolve one graph file
# ------------------------------------------------------------

def _resolve_graph_file(
    directory,
    graph_id,
    table_name,
):
    """
    Resolve the CSV file corresponding to one graph_id.
    """

    graph_id = _canonical_graph_id(
        graph_id
    )

    graph_file_index = (
        _build_graph_file_index(
            directory=directory,
            table_name=table_name,
        )
    )

    if graph_id not in graph_file_index:
        raise FileNotFoundError(
            f"No {table_name} file found for "
            f"graph_id '{graph_id}' in:\n"
            f"{directory}"
        )

    return graph_file_index[graph_id]


# ------------------------------------------------------------
# Numerical transformation helper
# ------------------------------------------------------------

def _transform_numeric_column(
    series,
    transform_spec,
    feature_name,
):
    """
    Apply a previously fitted numerical transformation.

    Supported transformations:
      - standardize
      - log1p_standardize
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if np.isnan(values).any():
        raise ValueError(
            f"Feature '{feature_name}' contains "
            "missing or non-numeric values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Feature '{feature_name}' contains "
            "NaN or infinite values."
        )

    transform_name = (
        transform_spec["transform"]
    )

    mean = float(
        transform_spec["mean"]
    )

    std = float(
        transform_spec["std"]
    )

    if (
        not np.isfinite(mean)
        or not np.isfinite(std)
        or std <= 1e-12
    ):
        raise ValueError(
            f"Invalid fitted preprocessing statistics "
            f"for '{feature_name}'."
        )

    if transform_name == "log1p_standardize":

        if (values < 0).any():
            raise ValueError(
                f"Feature '{feature_name}' contains negative "
                "values but requires log1p preprocessing."
            )

        values = np.log1p(
            values
        )

    elif transform_name == "standardize":
        pass

    else:
        raise ValueError(
            f"Unsupported transformation "
            f"'{transform_name}' for '{feature_name}'."
        )

    transformed = (
        values - mean
    ) / std

    if not np.isfinite(transformed).all():
        raise ValueError(
            f"Transformation of '{feature_name}' "
            "produced NaN or infinite values."
        )

    return transformed.astype(
        np.float32
    )


# ------------------------------------------------------------
# Categorical transformation helper
# ------------------------------------------------------------

def _transform_categorical_column(
    series,
    encoder_spec,
    feature_name,
):
    """
    Convert categorical values to their TRAIN-fitted integer IDs.
    """

    if series.isna().any():
        raise ValueError(
            f"Categorical feature '{feature_name}' "
            "contains missing values."
        )

    mapping = (
        encoder_spec["mapping"]
    )

    values = (
        series.astype(str)
        .tolist()
    )

    unknown_categories = sorted({
        value
        for value in values
        if value not in mapping
    })

    if unknown_categories:
        raise ValueError(
            f"Unknown categories found in "
            f"'{feature_name}'.\n"
            f"These categories were not observed "
            f"during TRAIN preprocessing:\n"
            + "\n".join(
                f"  - {category}"
                for category
                in unknown_categories
            )
        )

    encoded = np.array(
        [
            mapping[value]
            for value in values
        ],
        dtype=np.float32,
    )

    return encoded


# ------------------------------------------------------------
# Static graph loader
# ------------------------------------------------------------

def load_static_graph(
    graph_id,
    split,
    preprocessing_state,
):
    """
    Load and preprocess one static DAG.

    Returns
    -------
    dict
        {
            "x":
                FloatTensor [num_nodes, num_node_features],

            "edge_index":
                LongTensor [2, num_edges],

            "edge_attr":
                FloatTensor [num_edges, num_edge_features],

            "node_id_to_index":
                dict mapping original node IDs to local
                contiguous tensor indices
        }
    """

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of: {sorted(VALID_SPLITS)}"
        )

    graph_id = _canonical_graph_id(
        graph_id
    )

    if not isinstance(
        preprocessing_state,
        dict,
    ):
        raise TypeError(
            "preprocessing_state must be a dictionary."
        )


    # --------------------------------------------------------
    # Resolve node and edge files
    # --------------------------------------------------------

    node_directory = (
        NODE_FEATURES_DIR / split
    )

    edge_directory = (
        EDGE_FEATURES_DIR / split
    )

    node_path = _resolve_graph_file(
        directory=node_directory,
        graph_id=graph_id,
        table_name="node feature",
    )

    edge_path = _resolve_graph_file(
        directory=edge_directory,
        graph_id=graph_id,
        table_name="edge feature",
    )


    # --------------------------------------------------------
    # Load raw node and edge tables
    # --------------------------------------------------------

    node_df = pd.read_csv(
        node_path
    )

    edge_df = pd.read_csv(
        edge_path
    )


    # --------------------------------------------------------
    # Verify required raw columns
    # --------------------------------------------------------

    required_node_columns = [
        "graph_id",
        "node_id",
        *MODEL_NODE_FEATURES,
    ]

    required_edge_columns = [
        "graph_id",
        "source",
        "target",
        *MODEL_EDGE_FEATURES,
    ]

    missing_node_columns = [
        column
        for column in required_node_columns
        if column not in node_df.columns
    ]

    missing_edge_columns = [
        column
        for column in required_edge_columns
        if column not in edge_df.columns
    ]

    if missing_node_columns:
        raise ValueError(
            f"Missing required node columns "
            f"for graph '{graph_id}':\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_node_columns
            )
        )

    if missing_edge_columns:
        raise ValueError(
            f"Missing required edge columns "
            f"for graph '{graph_id}':\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_edge_columns
            )
        )


    # --------------------------------------------------------
    # Validate graph identity
    # --------------------------------------------------------

    node_graph_ids = {
        _canonical_graph_id(value)
        for value
        in node_df["graph_id"].unique()
    }

    edge_graph_ids = {
        _canonical_graph_id(value)
        for value
        in edge_df["graph_id"].unique()
    }

    if node_graph_ids != {graph_id}:
        raise ValueError(
            f"Node file does not exclusively contain "
            f"graph_id '{graph_id}'.\n"
            f"Found: {sorted(node_graph_ids)}"
        )

    if edge_graph_ids != {graph_id}:
        raise ValueError(
            f"Edge file does not exclusively contain "
            f"graph_id '{graph_id}'.\n"
            f"Found: {sorted(edge_graph_ids)}"
        )


    # --------------------------------------------------------
    # Validate node IDs
    # --------------------------------------------------------

    if node_df.empty:
        raise ValueError(
            f"Graph '{graph_id}' has no nodes."
        )

    canonical_node_ids = [
        _canonical_node_id(value)
        for value
        in node_df["node_id"]
    ]

    if len(
        set(canonical_node_ids)
    ) != len(canonical_node_ids):
        raise ValueError(
            f"Duplicate node IDs detected "
            f"in graph '{graph_id}'."
        )


    # --------------------------------------------------------
    # Build node_id -> local tensor index mapping
    # --------------------------------------------------------

    # PyG tensors always use contiguous local indices:
    #
    # original node IDs:
    #     14, 20, 91
    #
    # local indices:
    #     0,  1,  2

    canonical_node_to_index = {
        node_id: index
        for index, node_id
        in enumerate(canonical_node_ids)
    }

    node_id_to_index = {
        node_df["node_id"].iloc[index]: index
        for index
        in range(len(node_df))
    }


    # --------------------------------------------------------
    # Transform node features
    # --------------------------------------------------------

    transformed_node_columns = {}


    # Large positive node features:
    # log1p + standardization
    for feature_name, transform_spec in (
        preprocessing_state["node"][
            "log1p_standardized"
        ].items()
    ):

        transformed_node_columns[
            feature_name
        ] = _transform_numeric_column(
            series=node_df[feature_name],
            transform_spec=transform_spec,
            feature_name=feature_name,
        )


    # Structural numerical features:
    # standardization
    for feature_name, transform_spec in (
        preprocessing_state["node"][
            "standardized"
        ].items()
    ):

        transformed_node_columns[
            feature_name
        ] = _transform_numeric_column(
            series=node_df[feature_name],
            transform_spec=transform_spec,
            feature_name=feature_name,
        )


    # operation_type categorical encoding
    transformed_node_columns[
        "operation_type"
    ] = _transform_categorical_column(
        series=node_df["operation_type"],
        encoder_spec=(
            preprocessing_state["node"][
                "categorical"
            ]["operation_type"]
        ),
        feature_name="operation_type",
    )


    # criticality categorical encoding
    transformed_node_columns[
        "criticality"
    ] = _transform_categorical_column(
        series=node_df["criticality"],
        encoder_spec=(
            preprocessing_state["node"][
                "categorical"
            ]["criticality"]
        ),
        feature_name="criticality",
    )


    # --------------------------------------------------------
    # Construct x in the OFFICIAL feature order
    # --------------------------------------------------------

    # This is important:
    #
    # preprocessing grouped numerical/categorical features
    # differently, but the final tensor must follow
    # MODEL_NODE_FEATURES exactly.

    node_matrix = np.column_stack(
        [
            transformed_node_columns[
                feature_name
            ]
            for feature_name
            in MODEL_NODE_FEATURES
        ]
    ).astype(
        np.float32
    )

    x = torch.tensor(
        node_matrix,
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # Convert graph edges to local node indices
    # --------------------------------------------------------

    source_indices = []
    target_indices = []

    missing_edge_endpoints = []

    for row in edge_df.itertuples(
        index=False
    ):

        source_id = _canonical_node_id(
            row.source
        )

        target_id = _canonical_node_id(
            row.target
        )

        if source_id not in canonical_node_to_index:
            missing_edge_endpoints.append(
                (
                    "source",
                    row.source,
                    row.target,
                )
            )

            continue

        if target_id not in canonical_node_to_index:
            missing_edge_endpoints.append(
                (
                    "target",
                    row.source,
                    row.target,
                )
            )

            continue

        source_indices.append(
            canonical_node_to_index[
                source_id
            ]
        )

        target_indices.append(
            canonical_node_to_index[
                target_id
            ]
        )


    # --------------------------------------------------------
    # Validate every edge endpoint exists
    # --------------------------------------------------------

    if missing_edge_endpoints:

        preview = (
            missing_edge_endpoints[:10]
        )

        raise ValueError(
            f"Graph '{graph_id}' contains edges whose "
            "endpoints do not exist in the node table.\n"
            f"Invalid endpoint count: "
            f"{len(missing_edge_endpoints)}\n"
            f"Examples: {preview}"
        )


    # --------------------------------------------------------
    # Construct edge_index
    # --------------------------------------------------------

    if len(edge_df) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    else:

        edge_index = torch.tensor(
            [
                source_indices,
                target_indices,
            ],
            dtype=torch.long,
        )


    # --------------------------------------------------------
    # Transform edge attributes
    # --------------------------------------------------------

    transformed_edge_columns = {}

    for feature_name, transform_spec in (
        preprocessing_state["edge"][
            "log1p_standardized"
        ].items()
    ):

        transformed_edge_columns[
            feature_name
        ] = _transform_numeric_column(
            series=edge_df[feature_name],
            transform_spec=transform_spec,
            feature_name=feature_name,
        )


    if len(edge_df) == 0:

        edge_matrix = np.empty(
            (
                0,
                len(MODEL_EDGE_FEATURES),
            ),
            dtype=np.float32,
        )

    else:

        edge_matrix = np.column_stack(
            [
                transformed_edge_columns[
                    feature_name
                ]
                for feature_name
                in MODEL_EDGE_FEATURES
            ]
        ).astype(
            np.float32
        )

    edge_attr = torch.tensor(
        edge_matrix,
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if x.ndim != 2:
        raise ValueError(
            f"x must be 2-dimensional. "
            f"Found shape: {tuple(x.shape)}"
        )

    if edge_index.ndim != 2:
        raise ValueError(
            f"edge_index must be 2-dimensional. "
            f"Found shape: {tuple(edge_index.shape)}"
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E]. "
            f"Found: {tuple(edge_index.shape)}"
        )

    if edge_attr.ndim != 2:
        raise ValueError(
            f"edge_attr must be 2-dimensional. "
            f"Found shape: {tuple(edge_attr.shape)}"
        )


    # --------------------------------------------------------
    # Validate edge_attr rows == edge_index columns
    # --------------------------------------------------------

    if (
        edge_attr.shape[0]
        != edge_index.shape[1]
    ):
        raise ValueError(
            f"Edge count mismatch for graph '{graph_id}'.\n"
            f"edge_index columns: "
            f"{edge_index.shape[1]}\n"
            f"edge_attr rows:     "
            f"{edge_attr.shape[0]}"
        )


    # --------------------------------------------------------
    # Validate expected feature dimensions
    # --------------------------------------------------------

    if (
        x.shape[1]
        != len(MODEL_NODE_FEATURES)
    ):
        raise ValueError(
            f"Unexpected node feature dimension.\n"
            f"Expected: {len(MODEL_NODE_FEATURES)}\n"
            f"Found:    {x.shape[1]}"
        )

    if (
        edge_attr.shape[1]
        != len(MODEL_EDGE_FEATURES)
    ):
        raise ValueError(
            f"Unexpected edge feature dimension.\n"
            f"Expected: {len(MODEL_EDGE_FEATURES)}\n"
            f"Found:    {edge_attr.shape[1]}"
        )


    # --------------------------------------------------------
    # Validate edge indices are in range
    # --------------------------------------------------------

    if edge_index.numel() > 0:

        min_index = int(
            edge_index.min().item()
        )

        max_index = int(
            edge_index.max().item()
        )

        if min_index < 0:
            raise ValueError(
                f"Negative node index detected "
                f"in graph '{graph_id}'."
            )

        if max_index >= x.shape[0]:
            raise ValueError(
                f"Out-of-range node index detected "
                f"in graph '{graph_id}'.\n"
                f"Maximum edge index: {max_index}\n"
                f"Number of nodes:    {x.shape[0]}"
            )


    # --------------------------------------------------------
    # Validate NaN / Inf values
    # --------------------------------------------------------

    if not torch.isfinite(x).all():
        raise ValueError(
            f"x contains NaN or Inf values "
            f"for graph '{graph_id}'."
        )

    if not torch.isfinite(edge_attr).all():
        raise ValueError(
            f"edge_attr contains NaN or Inf values "
            f"for graph '{graph_id}'."
        )


    # --------------------------------------------------------
    # Return static graph representation
    # --------------------------------------------------------

    return {
        "graph_id": graph_id,
        "split": split,

        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,

        "node_id_to_index": (
            node_id_to_index
        ),
    }


# ------------------------------------------------------------
# Smoke test on one TRAIN graph
# ------------------------------------------------------------

sample_graph_id = sorted(
    train_df["graph_id"]
    .astype(str)
    .unique()
)[0]

sample_static_graph = (
    load_static_graph(
        graph_id=sample_graph_id,
        split="train",
        preprocessing_state=preprocessing_state,
    )
)


print("=" * 70)
print("Static Graph Loader Smoke Test")
print("=" * 70)

print(
    f"\nGraph ID       : "
    f"{sample_static_graph['graph_id']}"
)

print(
    f"Split          : "
    f"{sample_static_graph['split']}"
)

print(
    f"Nodes          : "
    f"{sample_static_graph['x'].shape[0]:,}"
)

print(
    f"Edges          : "
    f"{sample_static_graph['edge_index'].shape[1]:,}"
)

print(
    f"x shape        : "
    f"{tuple(sample_static_graph['x'].shape)}"
)

print(
    f"edge_index     : "
    f"{tuple(sample_static_graph['edge_index'].shape)}"
)

print(
    f"edge_attr      : "
    f"{tuple(sample_static_graph['edge_attr'].shape)}"
)

print(
    f"Node mapping   : "
    f"{len(sample_static_graph['node_id_to_index']):,} entries"
)

print("\nValidation:")
print("  - Every edge endpoint exists.")
print("  - edge_attr rows match edge_index columns.")
print("  - Node indices are within valid range.")
print("  - No NaN/Inf values were found.")

print("=" * 70)

# --- from TODO 6.2 ---
# ============================================================
# TODO 6.2 — Implement graph-scenario dataset
# ============================================================

from torch.utils.data import Dataset


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

required_objects = [
    "train_df",
    "validation_df",
    "test_id_df",
    "preprocessing_state",
    "MODEL_CONTEXT_FEATURES",
    "MODEL_TARGET",
    "load_static_graph",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "Required objects are missing. Run previous TODO cells first:\n"
        + "\n".join(
            f"  - {name}"
            for name in missing_objects
        )
    )


# ------------------------------------------------------------
# Official runtime-record target-node column
# ------------------------------------------------------------

TARGET_NODE_COLUMN = "target_node"


# ------------------------------------------------------------
# Helper: transform core type
# ------------------------------------------------------------

def transform_core_type(
    series,
    preprocessing_state,
):
    """
    Convert core_type values to TRAIN-fitted categorical indices.
    """

    encoder_spec = (
        preprocessing_state["context"]
        ["categorical"]
        ["core_type"]
    )

    encoded = _transform_categorical_column(
        series=series,
        encoder_spec=encoder_spec,
        feature_name="core_type",
    )

    return encoded.astype(np.int64)


# ------------------------------------------------------------
# Helper: transform DVFS and z_t
# ------------------------------------------------------------

def transform_context_numeric(
    records_df,
    preprocessing_state,
):
    """
    Transform numerical execution-context features using only
    TRAIN-fitted preprocessing statistics.

    Output order:
        frequency_ghz
        voltage_v
        cpu_utilization
        ready_queue_length
        active_core_count
        memory_active_tasks
        bus_utilization
        thermal_pressure
        release_jitter_us
    """

    transformed_columns = {}


    # --------------------------------------------------------
    # DVFS features
    # --------------------------------------------------------

    dvfs_specs = (
        preprocessing_state["context"]
        ["dvfs_standardized"]
    )

    for feature_name, transform_spec in dvfs_specs.items():

        transformed_columns[feature_name] = (
            _transform_numeric_column(
                series=records_df[feature_name],
                transform_spec=transform_spec,
                feature_name=feature_name,
            )
        )


    # --------------------------------------------------------
    # Dynamic system state z_t
    # --------------------------------------------------------

    zt_specs = (
        preprocessing_state["context"]
        ["z_t_standardized"]
    )

    for feature_name, transform_spec in zt_specs.items():

        transformed_columns[feature_name] = (
            _transform_numeric_column(
                series=records_df[feature_name],
                transform_spec=transform_spec,
                feature_name=feature_name,
            )
        )


    # Keep the official context order, excluding core_type
    # because core_type is handled separately as a categorical
    # index for a future embedding layer.

    numerical_context_order = [
        feature_name
        for feature_name in MODEL_CONTEXT_FEATURES
        if feature_name != "core_type"
    ]


    context_numeric = np.column_stack(
        [
            transformed_columns[feature_name]
            for feature_name
            in numerical_context_order
        ]
    ).astype(np.float32)


    if not np.isfinite(context_numeric).all():
        raise ValueError(
            "Context transformation produced NaN or Inf values."
        )


    return (
        context_numeric,
        numerical_context_order,
    )


# ------------------------------------------------------------
# Helper: transform y_exec_us
# ------------------------------------------------------------

def transform_target(
    records_df,
    preprocessing_state,
):
    """
    Transform y_exec_us using the target scaler fitted on TRAIN.
    """

    target_spec = (
        preprocessing_state["target"]
        [MODEL_TARGET]
    )

    transformed_target = (
        _transform_numeric_column(
            series=records_df[MODEL_TARGET],
            transform_spec=target_spec,
            feature_name=MODEL_TARGET,
        )
    )

    return transformed_target


# ------------------------------------------------------------
# Graph-scenario dataset
# ------------------------------------------------------------

class GraphScenarioDataset(Dataset):
    """
    One dataset item = one DAG + all runtime scenarios for that DAG.

    Static graph:
        x
        edge_index
        edge_attr

    Runtime scenarios:
        target_node_indices
        core_type_indices
        context_numeric
        y

    The static graph is shared across all scenarios belonging
    to the same DAG.
    """

    def __init__(
        self,
        records_df,
        split,
        preprocessing_state,
        expected_records_per_graph=42,
        cache_static_graphs=True,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Validate split
        # ----------------------------------------------------

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of: {sorted(VALID_SPLITS)}"
            )


        # ----------------------------------------------------
        # Validate input table
        # ----------------------------------------------------

        if records_df.empty:
            raise ValueError(
                f"records_df for split '{split}' is empty."
            )

        required_columns = [
            "graph_id",
            TARGET_NODE_COLUMN,
            "core_type",
            MODEL_TARGET,
            *[
                feature
                for feature in MODEL_CONTEXT_FEATURES
                if feature != "core_type"
            ],
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in records_df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Required runtime-record columns are missing "
                f"from split '{split}':\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing_columns
                )
            )


        # ----------------------------------------------------
        # Store dataset state
        # ----------------------------------------------------

        self.records_df = (
            records_df.copy()
            .reset_index(drop=True)
        )

        self.records_df["graph_id"] = (
            self.records_df["graph_id"]
            .astype(str)
        )

        self.split = split

        self.preprocessing_state = (
            preprocessing_state
        )

        self.expected_records_per_graph = (
            expected_records_per_graph
        )

        self.cache_static_graphs = (
            cache_static_graphs
        )

        self._static_graph_cache = {}


        # ----------------------------------------------------
        # Build ordered graph list
        # ----------------------------------------------------

        self.graph_ids = sorted(
            self.records_df[
                "graph_id"
            ]
            .unique()
            .tolist()
        )

        if not self.graph_ids:
            raise ValueError(
                f"No graph IDs were found in split '{split}'."
            )


        # ----------------------------------------------------
        # Build graph_id -> row-index mapping
        # ----------------------------------------------------

        self.graph_record_indices = {}

        for graph_id in self.graph_ids:

            indices = (
                self.records_df.index[
                    self.records_df["graph_id"]
                    == graph_id
                ]
                .to_numpy()
            )

            num_records = len(indices)

            if num_records == 0:
                raise ValueError(
                    f"No runtime records found for "
                    f"graph '{graph_id}'."
                )

            if (
                self.expected_records_per_graph is not None
                and num_records
                != self.expected_records_per_graph
            ):
                raise ValueError(
                    f"Unexpected runtime-record count "
                    f"for graph '{graph_id}'.\n"
                    f"Expected: "
                    f"{self.expected_records_per_graph}\n"
                    f"Found:    {num_records}"
                )

            self.graph_record_indices[
                graph_id
            ] = indices


        # ----------------------------------------------------
        # Evaluation-only metadata columns
        # ----------------------------------------------------

        # These columns are kept for later diagnostics.
        # They are NOT automatically used as model inputs.

        preferred_metadata_columns = [
            "sample_id",
            "graph_id",
            "target_node",
            "target_role",
            "target_index",
            "context_index",
            "context_id",
            "core_type",
            "core_id",
            "dvfs_level",
            "frequency_ghz",
            "voltage_v",
        ]

        self.metadata_columns = [
            column
            for column in preferred_metadata_columns
            if column in self.records_df.columns
        ]


    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    def __len__(self):
        """
        Number of DAGs, not number of runtime records.
        """

        return len(self.graph_ids)


    # --------------------------------------------------------
    # Load static graph with optional memory cache
    # --------------------------------------------------------

    def _get_static_graph(
        self,
        graph_id,
    ):

        if (
            self.cache_static_graphs
            and graph_id
            in self._static_graph_cache
        ):
            return self._static_graph_cache[
                graph_id
            ]

        graph = load_static_graph(
            graph_id=graph_id,
            split=self.split,
            preprocessing_state=(
                self.preprocessing_state
            ),
        )

        if self.cache_static_graphs:
            self._static_graph_cache[
                graph_id
            ] = graph

        return graph


    # --------------------------------------------------------
    # Return one DAG + all scenarios
    # --------------------------------------------------------

    def __getitem__(
        self,
        index,
    ):

        graph_id = self.graph_ids[
            index
        ]

        record_indices = (
            self.graph_record_indices[
                graph_id
            ]
        )

        graph_records = (
            self.records_df
            .loc[record_indices]
            .reset_index(drop=True)
        )

        num_scenarios = len(
            graph_records
        )


        # ----------------------------------------------------
        # Load static graph
        # ----------------------------------------------------

        static_graph = (
            self._get_static_graph(
                graph_id
            )
        )

        x = static_graph["x"]

        edge_index = (
            static_graph["edge_index"]
        )

        edge_attr = (
            static_graph["edge_attr"]
        )

        node_id_to_index = (
            static_graph[
                "node_id_to_index"
            ]
        )


        # ----------------------------------------------------
        # Build canonical node-ID mapping
        # ----------------------------------------------------

        canonical_node_to_index = {
            _canonical_node_id(
                original_node_id
            ): local_index

            for (
                original_node_id,
                local_index
            )
            in node_id_to_index.items()
        }


        # ----------------------------------------------------
        # Runtime target nodes -> local PyTorch indices
        # ----------------------------------------------------

        target_node_indices_list = []

        missing_target_nodes = []

        for target_node in graph_records[
            TARGET_NODE_COLUMN
        ]:

            canonical_target_node = (
                _canonical_node_id(
                    target_node
                )
            )

            if (
                canonical_target_node
                not in canonical_node_to_index
            ):

                missing_target_nodes.append(
                    target_node
                )

                continue

            target_node_indices_list.append(
                canonical_node_to_index[
                    canonical_target_node
                ]
            )


        if missing_target_nodes:
            raise ValueError(
                f"Runtime records for graph '{graph_id}' "
                "reference target nodes that do not exist "
                "in the static graph.\n"
                f"Missing count: "
                f"{len(missing_target_nodes)}\n"
                f"Examples: "
                f"{missing_target_nodes[:10]}"
            )


        target_node_indices = (
            torch.tensor(
                target_node_indices_list,
                dtype=torch.long,
            )
        )


        # ----------------------------------------------------
        # Transform core_type
        # ----------------------------------------------------

        core_type_indices_np = (
            transform_core_type(
                series=graph_records[
                    "core_type"
                ],
                preprocessing_state=(
                    self.preprocessing_state
                ),
            )
        )

        core_type_indices = (
            torch.tensor(
                core_type_indices_np,
                dtype=torch.long,
            )
        )


        # ----------------------------------------------------
        # Transform DVFS + z_t
        # ----------------------------------------------------

        (
            context_numeric_np,
            context_numeric_order,
        ) = transform_context_numeric(
            records_df=graph_records,
            preprocessing_state=(
                self.preprocessing_state
            ),
        )

        context_numeric = (
            torch.tensor(
                context_numeric_np,
                dtype=torch.float32,
            )
        )


        # ----------------------------------------------------
        # Build complete aligned context matrix
        # ----------------------------------------------------

        # Column 0:
        #     core_type categorical index
        #
        # Remaining columns:
        #     standardized DVFS + z_t
        #
        # The model will still receive core_type_indices
        # separately so it can use an embedding.

        context = torch.cat(
            [
                core_type_indices
                .to(torch.float32)
                .unsqueeze(1),

                context_numeric,
            ],
            dim=1,
        )


        # ----------------------------------------------------
        # Transform y_exec_us
        # ----------------------------------------------------

        y_scaled_np = (
            transform_target(
                records_df=graph_records,
                preprocessing_state=(
                    self.preprocessing_state
                ),
            )
        )

        y = torch.tensor(
            y_scaled_np,
            dtype=torch.float32,
        )


        # ----------------------------------------------------
        # Keep raw target for evaluation
        # ----------------------------------------------------

        y_raw_us_np = pd.to_numeric(
            graph_records[
                MODEL_TARGET
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        if (
            np.isnan(y_raw_us_np).any()
            or not np.isfinite(
                y_raw_us_np
            ).all()
        ):
            raise ValueError(
                f"Raw target '{MODEL_TARGET}' contains "
                f"NaN/Inf values for graph '{graph_id}'."
            )

        y_raw_us = torch.tensor(
            y_raw_us_np,
            dtype=torch.float32,
        )


        # ----------------------------------------------------
        # Scenario alignment validation
        # ----------------------------------------------------

        scenario_tensors = {
            "target_node_indices": (
                target_node_indices
            ),
            "core_type_indices": (
                core_type_indices
            ),
            "context_numeric": (
                context_numeric
            ),
            "context": (
                context
            ),
            "y": (
                y
            ),
            "y_raw_us": (
                y_raw_us
            ),
        }

        for tensor_name, tensor in (
            scenario_tensors.items()
        ):

            if tensor.shape[0] != num_scenarios:
                raise ValueError(
                    f"Scenario alignment error for "
                    f"'{tensor_name}' in graph "
                    f"'{graph_id}'.\n"
                    f"Expected rows: {num_scenarios}\n"
                    f"Found rows:    {tensor.shape[0]}"
                )


        # ----------------------------------------------------
        # Validate expected context dimensions
        # ----------------------------------------------------

        expected_numeric_context_dim = (
            len(MODEL_CONTEXT_FEATURES)
            - 1
        )

        if (
            context_numeric.shape[1]
            != expected_numeric_context_dim
        ):
            raise ValueError(
                f"Unexpected numerical-context dimension "
                f"for graph '{graph_id}'.\n"
                f"Expected: "
                f"{expected_numeric_context_dim}\n"
                f"Found:    "
                f"{context_numeric.shape[1]}"
            )

        if (
            context.shape[1]
            != len(MODEL_CONTEXT_FEATURES)
        ):
            raise ValueError(
                f"Unexpected full-context dimension "
                f"for graph '{graph_id}'.\n"
                f"Expected: "
                f"{len(MODEL_CONTEXT_FEATURES)}\n"
                f"Found:    "
                f"{context.shape[1]}"
            )


        # ----------------------------------------------------
        # Validate target-node index range
        # ----------------------------------------------------

        if target_node_indices.numel() > 0:

            min_target_index = int(
                target_node_indices
                .min()
                .item()
            )

            max_target_index = int(
                target_node_indices
                .max()
                .item()
            )

            if min_target_index < 0:
                raise ValueError(
                    f"Negative target-node index found "
                    f"for graph '{graph_id}'."
                )

            if max_target_index >= x.shape[0]:
                raise ValueError(
                    f"Out-of-range target-node index "
                    f"for graph '{graph_id}'.\n"
                    f"Maximum target index: "
                    f"{max_target_index}\n"
                    f"Number of nodes: "
                    f"{x.shape[0]}"
                )


        # ----------------------------------------------------
        # Validate finite tensors
        # ----------------------------------------------------

        finite_tensors = {
            "x": x,
            "edge_attr": edge_attr,
            "context_numeric": (
                context_numeric
            ),
            "context": (
                context
            ),
            "y": y,
            "y_raw_us": (
                y_raw_us
            ),
        }

        for tensor_name, tensor in (
            finite_tensors.items()
        ):

            if not torch.isfinite(
                tensor
            ).all():

                raise ValueError(
                    f"Tensor '{tensor_name}' contains "
                    f"NaN or Inf values for "
                    f"graph '{graph_id}'."
                )


        # ----------------------------------------------------
        # Evaluation metadata
        # ----------------------------------------------------

        evaluation_metadata = (
            graph_records[
                self.metadata_columns
            ]
            .copy()
            .reset_index(drop=True)
        )

        evaluation_metadata[
            "local_target_node_index"
        ] = (
            target_node_indices
            .cpu()
            .numpy()
        )

        evaluation_metadata[
            "y_exec_us_raw"
        ] = (
            y_raw_us
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Final item
        # ----------------------------------------------------

        return {

            # Identity
            "graph_id": graph_id,
            "split": self.split,

            # Static DAG
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,

            # Original ID -> local tensor index
            "node_id_to_index": (
                node_id_to_index
            ),

            # Scenario target nodes
            "target_node_indices": (
                target_node_indices
            ),

            # Categorical context
            "core_type_indices": (
                core_type_indices
            ),

            # Numerical context:
            # DVFS + z_t
            "context_numeric": (
                context_numeric
            ),

            # Complete aligned context matrix
            "context": (
                context
            ),

            # Official order for numerical context
            "context_numeric_order": (
                context_numeric_order
            ),

            # Standardized target
            "y": y,

            # Raw target in microseconds
            "y_raw_us": (
                y_raw_us
            ),

            # Evaluation-only metadata
            "metadata": (
                evaluation_metadata
            ),
        }


# ------------------------------------------------------------
# Build Phase 1 datasets
# ------------------------------------------------------------

train_dataset = GraphScenarioDataset(
    records_df=train_df,
    split="train",
    preprocessing_state=preprocessing_state,
    expected_records_per_graph=42,
    cache_static_graphs=True,
)

validation_dataset = GraphScenarioDataset(
    records_df=validation_df,
    split="validation",
    preprocessing_state=preprocessing_state,
    expected_records_per_graph=42,
    cache_static_graphs=True,
)

test_id_dataset = GraphScenarioDataset(
    records_df=test_id_df,
    split="test_id",
    preprocessing_state=preprocessing_state,
    expected_records_per_graph=42,
    cache_static_graphs=True,
)


# ------------------------------------------------------------
# Explicit Phase 1 split protection
# ------------------------------------------------------------

PHASE1_DATASET_SPLITS = {
    train_dataset.split,
    validation_dataset.split,
    test_id_dataset.split,
}

EXPECTED_PHASE1_DATASET_SPLITS = {
    "train",
    "validation",
    "test_id",
}

if (
    PHASE1_DATASET_SPLITS
    != EXPECTED_PHASE1_DATASET_SPLITS
):
    raise RuntimeError(
        "Unexpected Phase 1 dataset split configuration."
    )


# ------------------------------------------------------------
# Verify expected number of DAG items
# ------------------------------------------------------------

EXPECTED_PHASE1_GRAPH_COUNTS = {
    "train": 400,
    "validation": 50,
    "test_id": 50,
}

actual_dataset_sizes = {
    "train": len(train_dataset),
    "validation": len(validation_dataset),
    "test_id": len(test_id_dataset),
}

for split_name, expected_count in (
    EXPECTED_PHASE1_GRAPH_COUNTS.items()
):

    actual_count = (
        actual_dataset_sizes[
            split_name
        ]
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Unexpected number of DAG items "
            f"for split '{split_name}'.\n"
            f"Expected: {expected_count}\n"
            f"Found:    {actual_count}"
        )


# ------------------------------------------------------------
# Smoke test on one TRAIN item
# ------------------------------------------------------------

sample_item = (
    train_dataset[0]
)

num_sample_scenarios = (
    sample_item[
        "target_node_indices"
    ].shape[0]
)


# ------------------------------------------------------------
# Smoke-test validation
# ------------------------------------------------------------

if num_sample_scenarios != 42:
    raise ValueError(
        "Smoke-test graph does not contain "
        "the expected 42 runtime scenarios."
    )

if (
    sample_item["y"].shape[0]
    != num_sample_scenarios
):
    raise ValueError(
        "Smoke-test target count mismatch."
    )

if (
    sample_item["context"].shape[0]
    != num_sample_scenarios
):
    raise ValueError(
        "Smoke-test context count mismatch."
    )

if (
    sample_item[
        "target_node_indices"
    ].shape[0]
    != num_sample_scenarios
):
    raise ValueError(
        "Smoke-test target-node count mismatch."
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Graph-Scenario Dataset Verification")
print("=" * 70)

print("\nDataset sizes:")

print(
    f"  Train       : "
    f"{len(train_dataset):>3} DAG items"
)

print(
    f"  Validation  : "
    f"{len(validation_dataset):>3} DAG items"
)

print(
    f"  Test-ID     : "
    f"{len(test_id_dataset):>3} DAG items"
)


print("\nSample item:")

print(
    f"  Graph ID             : "
    f"{sample_item['graph_id']}"
)

print(
    f"  Nodes                : "
    f"{sample_item['x'].shape[0]:,}"
)

print(
    f"  Edges                : "
    f"{sample_item['edge_index'].shape[1]:,}"
)

print(
    f"  Runtime scenarios    : "
    f"{num_sample_scenarios}"
)


print("\nTensor shapes:")

print(
    f"  x                     : "
    f"{tuple(sample_item['x'].shape)}"
)

print(
    f"  edge_index            : "
    f"{tuple(sample_item['edge_index'].shape)}"
)

print(
    f"  edge_attr             : "
    f"{tuple(sample_item['edge_attr'].shape)}"
)

print(
    f"  target_node_indices   : "
    f"{tuple(sample_item['target_node_indices'].shape)}"
)

print(
    f"  core_type_indices     : "
    f"{tuple(sample_item['core_type_indices'].shape)}"
)

print(
    f"  context_numeric       : "
    f"{tuple(sample_item['context_numeric'].shape)}"
)

print(
    f"  context               : "
    f"{tuple(sample_item['context'].shape)}"
)

print(
    f"  y                     : "
    f"{tuple(sample_item['y'].shape)}"
)

print(
    f"  y_raw_us              : "
    f"{tuple(sample_item['y_raw_us'].shape)}"
)


print("\nTarget-node column:")
print(
    f"  {TARGET_NODE_COLUMN}"
)


print("\nNumerical context order:")

for feature_name in (
    sample_item[
        "context_numeric_order"
    ]
):
    print(
        f"  - {feature_name}"
    )


print("\nValidation:")

print(
    "  - One dataset item represents one DAG."
)

print(
    "  - Each DAG contains all 42 runtime scenarios."
)

print(
    "  - target_node values were mapped to local tensor indices."
)

print(
    "  - Context transformations use TRAIN-fitted state."
)

print(
    "  - Target transformation uses TRAIN-fitted state."
)

print(
    "  - Static graph and runtime scenarios are aligned."
)

print(
    "  - No NaN/Inf values were found."
)

print(
    "  - Train contains 400 DAG items."
)

print(
    "  - Validation contains 50 DAG items."
)

print(
    "  - Test-ID contains 50 DAG items."
)

print(
    "  - Calibration and test-OOD datasets were NOT created."
)

print("=" * 70)

# --- from TODO 6.3 ---
# ============================================================
# TODO 6.3 — Dataset smoke test
# ============================================================

# This cell performs a strict sanity check on one training DAG
# before the graph data is passed to the GNN model.


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if "train_dataset" not in globals():
    raise RuntimeError(
        "train_dataset is not available. Run TODO 6.2 first."
    )

if len(train_dataset) == 0:
    raise RuntimeError(
        "train_dataset is empty."
    )


# ------------------------------------------------------------
# Load one training graph-scenario item
# ------------------------------------------------------------

sample_item = train_dataset[0]


# ------------------------------------------------------------
# Extract tensors
# ------------------------------------------------------------

graph_id = sample_item["graph_id"]

x = sample_item["x"]

edge_index = sample_item["edge_index"]

edge_attr = sample_item["edge_attr"]

target_node_indices = (
    sample_item["target_node_indices"]
)

context_features = (
    sample_item["context"]
)

y = sample_item["y"]


# ------------------------------------------------------------
# Basic graph dimensions
# ------------------------------------------------------------

num_nodes = int(
    x.shape[0]
)

num_edges = int(
    edge_index.shape[1]
)

num_contexts = int(
    context_features.shape[0]
)


# ------------------------------------------------------------
# Validate tensor ranks
# ------------------------------------------------------------

if x.ndim != 2:
    raise ValueError(
        f"x must have shape [N, F]. "
        f"Found: {tuple(x.shape)}"
    )

if edge_index.ndim != 2:
    raise ValueError(
        f"edge_index must be 2-dimensional. "
        f"Found: {tuple(edge_index.shape)}"
    )

if edge_index.shape[0] != 2:
    raise ValueError(
        f"edge_index must have shape [2, E]. "
        f"Found: {tuple(edge_index.shape)}"
    )

if edge_attr.ndim != 2:
    raise ValueError(
        f"edge_attr must have shape [E, F_e]. "
        f"Found: {tuple(edge_attr.shape)}"
    )

if context_features.ndim != 2:
    raise ValueError(
        f"context_features must have shape [S, F_c]. "
        f"Found: {tuple(context_features.shape)}"
    )

if target_node_indices.ndim != 1:
    raise ValueError(
        f"target_node_indices must have shape [S]. "
        f"Found: {tuple(target_node_indices.shape)}"
    )

if y.ndim != 1:
    raise ValueError(
        f"y must have shape [S]. "
        f"Found: {tuple(y.shape)}"
    )


# ------------------------------------------------------------
# Validate expected feature dimensions
# ------------------------------------------------------------

if x.shape[1] != len(MODEL_NODE_FEATURES):
    raise ValueError(
        "Node feature dimension mismatch.\n"
        f"Expected: {len(MODEL_NODE_FEATURES)}\n"
        f"Found:    {x.shape[1]}"
    )

if edge_attr.shape[1] != len(MODEL_EDGE_FEATURES):
    raise ValueError(
        "Edge feature dimension mismatch.\n"
        f"Expected: {len(MODEL_EDGE_FEATURES)}\n"
        f"Found:    {edge_attr.shape[1]}"
    )

if context_features.shape[1] != len(MODEL_CONTEXT_FEATURES):
    raise ValueError(
        "Context feature dimension mismatch.\n"
        f"Expected: {len(MODEL_CONTEXT_FEATURES)}\n"
        f"Found:    {context_features.shape[1]}"
    )


# ------------------------------------------------------------
# Validate edge alignment
# ------------------------------------------------------------

if edge_attr.shape[0] != num_edges:
    raise ValueError(
        "edge_attr row count does not match "
        "the number of edges.\n"
        f"Edges:          {num_edges}\n"
        f"edge_attr rows: {edge_attr.shape[0]}"
    )


# ------------------------------------------------------------
# Validate scenario alignment
# ------------------------------------------------------------

if target_node_indices.shape[0] != num_contexts:
    raise ValueError(
        "target_node_indices length does not match "
        "the number of runtime contexts.\n"
        f"Contexts:            {num_contexts}\n"
        f"Target-node indices: {target_node_indices.shape[0]}"
    )

if y.shape[0] != num_contexts:
    raise ValueError(
        "Target vector length does not match "
        "the number of runtime contexts.\n"
        f"Contexts: {num_contexts}\n"
        f"Targets:  {y.shape[0]}"
    )


# ------------------------------------------------------------
# Validate expected number of scenarios
# ------------------------------------------------------------

EXPECTED_CONTEXTS_PER_GRAPH = 42

if num_contexts != EXPECTED_CONTEXTS_PER_GRAPH:
    raise ValueError(
        f"Unexpected number of runtime scenarios.\n"
        f"Expected: {EXPECTED_CONTEXTS_PER_GRAPH}\n"
        f"Found:    {num_contexts}"
    )


# ------------------------------------------------------------
# Validate all target node indices
# ------------------------------------------------------------

if target_node_indices.numel() == 0:
    raise ValueError(
        "No target node indices were found."
    )

min_target_index = int(
    target_node_indices.min().item()
)

max_target_index = int(
    target_node_indices.max().item()
)

if min_target_index < 0:
    raise ValueError(
        f"Invalid negative target-node index: "
        f"{min_target_index}"
    )

if max_target_index >= num_nodes:
    raise ValueError(
        "Target-node index is outside the graph range.\n"
        f"Maximum target index: {max_target_index}\n"
        f"Number of nodes:      {num_nodes}"
    )


# ------------------------------------------------------------
# Validate all edge indices
# ------------------------------------------------------------

if edge_index.numel() > 0:

    min_edge_index = int(
        edge_index.min().item()
    )

    max_edge_index = int(
        edge_index.max().item()
    )

    if min_edge_index < 0:
        raise ValueError(
            f"Invalid negative edge node index: "
            f"{min_edge_index}"
        )

    if max_edge_index >= num_nodes:
        raise ValueError(
            "edge_index contains an out-of-range node index.\n"
            f"Maximum edge index: {max_edge_index}\n"
            f"Number of nodes:    {num_nodes}"
        )


# ------------------------------------------------------------
# Validate tensor dtypes
# ------------------------------------------------------------

if x.dtype != torch.float32:
    raise TypeError(
        f"x must be torch.float32. "
        f"Found: {x.dtype}"
    )

if edge_index.dtype != torch.long:
    raise TypeError(
        f"edge_index must be torch.long. "
        f"Found: {edge_index.dtype}"
    )

if edge_attr.dtype != torch.float32:
    raise TypeError(
        f"edge_attr must be torch.float32. "
        f"Found: {edge_attr.dtype}"
    )

if target_node_indices.dtype != torch.long:
    raise TypeError(
        f"target_node_indices must be torch.long. "
        f"Found: {target_node_indices.dtype}"
    )

if context_features.dtype != torch.float32:
    raise TypeError(
        f"context_features must be torch.float32. "
        f"Found: {context_features.dtype}"
    )

if y.dtype != torch.float32:
    raise TypeError(
        f"y must be torch.float32. "
        f"Found: {y.dtype}"
    )


# ------------------------------------------------------------
# Validate finite values
# ------------------------------------------------------------

finite_tensors = {
    "x": x,
    "edge_attr": edge_attr,
    "context_features": context_features,
    "y": y,
}

for tensor_name, tensor in finite_tensors.items():

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{tensor_name} contains NaN or Inf values."
        )


# ------------------------------------------------------------
# Final smoke-test report
# ------------------------------------------------------------

print("=" * 70)
print("Dataset Smoke Test")
print("=" * 70)

print(f"\nGraph ID                 : {graph_id}")

print(f"Number of nodes          : {num_nodes:,}")
print(f"Number of edges          : {num_edges:,}")
print(f"Number of contexts       : {num_contexts}")

print("\nTensor shapes:")

print(
    f"  x                      : "
    f"{tuple(x.shape)}"
)

print(
    f"  edge_index             : "
    f"{tuple(edge_index.shape)}"
)

print(
    f"  edge_attr              : "
    f"{tuple(edge_attr.shape)}"
)

print(
    f"  context_features       : "
    f"{tuple(context_features.shape)}"
)

print(
    f"  target_node_indices    : "
    f"{tuple(target_node_indices.shape)}"
)

print(
    f"  y                      : "
    f"{tuple(y.shape)}"
)

print("\nTarget-node index range:")

print(
    f"  Minimum index          : "
    f"{min_target_index}"
)

print(
    f"  Maximum index          : "
    f"{max_target_index}"
)

print(
    f"  Valid node range       : "
    f"[0, {num_nodes - 1}]"
)

print("\nValidation:")

print(
    "  - Graph tensor dimensions are valid."
)

print(
    "  - Edge tensors are aligned."
)

print(
    "  - All 42 runtime contexts are aligned."
)

print(
    "  - All target-node indices are valid."
)

print(
    "  - All edge node indices are valid."
)

print(
    "  - Tensor dtypes are correct."
)

print(
    "  - No NaN/Inf values were found."
)

print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.1 ---
# ============================================================
# TODO 7.1 — Define Phase1Config
# ============================================================

from dataclasses import dataclass, asdict, field
from typing import Optional


# ------------------------------------------------------------
# Phase 1 configuration
# ------------------------------------------------------------

@dataclass
class Phase1Config:
    """
    Central configuration for the Phase 1 GNN quantile predictor.

    This object defines the main model and optimization
    hyperparameters used during training.
    """

    # --------------------------------------------------------
    # Quantile prediction targets
    # --------------------------------------------------------

    quantiles: list[float] = field(
        default_factory=lambda: [
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )


    # --------------------------------------------------------
    # GNN architecture
    # --------------------------------------------------------

    hidden_dim: int = 128

    num_gnn_layers: int = 3

    attention_hidden_dim: int = 64


    # --------------------------------------------------------
    # Quantile head architecture
    # --------------------------------------------------------

    head_hidden_dim: int = 128


    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout: float = 0.10


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate: float = 1e-3

    weight_decay: float = 1e-5

    max_epochs: int = 200

    early_stopping_patience: int = 25


    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    gradient_clip_norm: Optional[float] = 1.0


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed: int = SEED


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self):
        """
        Validate configuration values before model construction
        or training begins.
        """

        # Quantiles
        if not self.quantiles:
            raise ValueError(
                "At least one quantile must be specified."
            )

        if any(
            q <= 0.0 or q >= 1.0
            for q in self.quantiles
        ):
            raise ValueError(
                "All quantiles must lie strictly between 0 and 1."
            )

        if self.quantiles != sorted(self.quantiles):
            raise ValueError(
                "Quantiles must be sorted in ascending order."
            )

        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError(
                "Quantile values must be unique."
            )


        # Model dimensions
        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.num_gnn_layers <= 0:
            raise ValueError(
                "num_gnn_layers must be positive."
            )

        if self.attention_hidden_dim <= 0:
            raise ValueError(
                "attention_hidden_dim must be positive."
            )

        if self.head_hidden_dim <= 0:
            raise ValueError(
                "head_hidden_dim must be positive."
            )


        # Dropout
        if not (
            0.0 <= self.dropout < 1.0
        ):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )


        # Optimizer
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )


        # Training
        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive."
            )

        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive."
            )

        if (
            self.early_stopping_patience
            >= self.max_epochs
        ):
            raise ValueError(
                "early_stopping_patience must be smaller "
                "than max_epochs."
            )


        # Gradient clipping
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive "
                "or None."
            )


        # Seed
        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )


# ------------------------------------------------------------
# Create finalized Phase 1 configuration
# ------------------------------------------------------------

phase1_config = Phase1Config(
    quantiles=[
        0.50,
        0.90,
        0.95,
        0.99,
    ],

    hidden_dim=128,

    num_gnn_layers=3,

    attention_hidden_dim=64,

    head_hidden_dim=128,

    dropout=0.10,

    learning_rate=1e-3,

    weight_decay=1e-5,

    max_epochs=200,

    early_stopping_patience=25,

    gradient_clip_norm=1.0,

    seed=SEED,
)


# ------------------------------------------------------------
# Validate finalized configuration
# ------------------------------------------------------------

phase1_config.validate()


# ------------------------------------------------------------
# Convert configuration to serializable dictionary
# ------------------------------------------------------------

phase1_config_dict = asdict(
    phase1_config
)


# ------------------------------------------------------------
# Add immutable interface information
# ------------------------------------------------------------

phase1_config_dict[
    "num_quantiles"
] = len(
    phase1_config.quantiles
)

phase1_config_dict[
    "node_input_dim"
] = len(
    MODEL_NODE_FEATURES
)

phase1_config_dict[
    "edge_input_dim"
] = len(
    MODEL_EDGE_FEATURES
)

phase1_config_dict[
    "context_input_dim"
] = len(
    MODEL_CONTEXT_FEATURES
)

phase1_config_dict[
    "numerical_context_dim"
] = (
    len(MODEL_CONTEXT_FEATURES) - 1
)

phase1_config_dict[
    "core_type_num_categories"
] = (
    core_type_encoder[
        "num_categories"
    ]
)

phase1_config_dict[
    "operation_type_num_categories"
] = (
    operation_type_encoder[
        "num_categories"
    ]
)

phase1_config_dict[
    "criticality_num_categories"
] = (
    criticality_encoder[
        "num_categories"
    ]
)


# ------------------------------------------------------------
# Configuration output path
# ------------------------------------------------------------

PHASE1_CONFIG_PATH = (
    CONFIG_DIR / "phase1_config.json"
)

PHASE1_CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Save configuration BEFORE training
# ------------------------------------------------------------

with PHASE1_CONFIG_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        phase1_config_dict,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# Reload saved configuration
# ------------------------------------------------------------

with PHASE1_CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    saved_phase1_config = (
        json.load(f)
    )


# ------------------------------------------------------------
# Verify serialization
# ------------------------------------------------------------

required_saved_keys = [
    "quantiles",
    "hidden_dim",
    "num_gnn_layers",
    "attention_hidden_dim",
    "head_hidden_dim",
    "dropout",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "early_stopping_patience",
    "gradient_clip_norm",
    "seed",
    "num_quantiles",
    "node_input_dim",
    "edge_input_dim",
    "context_input_dim",
]

missing_saved_keys = [
    key
    for key in required_saved_keys
    if key not in saved_phase1_config
]

if missing_saved_keys:
    raise ValueError(
        "Saved Phase 1 configuration is missing keys:\n"
        + "\n".join(
            f"  - {key}"
            for key in missing_saved_keys
        )
    )


if (
    saved_phase1_config["quantiles"]
    != phase1_config.quantiles
):
    raise ValueError(
        "Saved quantile configuration does not match "
        "the active Phase 1 configuration."
    )


if (
    saved_phase1_config["seed"]
    != phase1_config.seed
):
    raise ValueError(
        "Saved random seed does not match "
        "the active Phase 1 configuration."
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Phase 1 Configuration")
print("=" * 70)

print("\nQuantile targets:")
print(
    "  ",
    phase1_config.quantiles,
)

print("\nGNN architecture:")
print(
    f"  Hidden dimension       : "
    f"{phase1_config.hidden_dim}"
)
print(
    f"  GNN layers             : "
    f"{phase1_config.num_gnn_layers}"
)
print(
    f"  Attention hidden dim   : "
    f"{phase1_config.attention_hidden_dim}"
)

print("\nQuantile head:")
print(
    f"  Head hidden dimension  : "
    f"{phase1_config.head_hidden_dim}"
)

print("\nRegularization:")
print(
    f"  Dropout                : "
    f"{phase1_config.dropout}"
)

print("\nOptimization:")
print(
    f"  Learning rate          : "
    f"{phase1_config.learning_rate}"
)
print(
    f"  Weight decay           : "
    f"{phase1_config.weight_decay}"
)
print(
    f"  Maximum epochs         : "
    f"{phase1_config.max_epochs}"
)
print(
    f"  Early-stop patience    : "
    f"{phase1_config.early_stopping_patience}"
)
print(
    f"  Gradient clipping      : "
    f"{phase1_config.gradient_clip_norm}"
)

print("\nInput dimensions:")
print(
    f"  Node input dimension   : "
    f"{phase1_config_dict['node_input_dim']}"
)
print(
    f"  Edge input dimension   : "
    f"{phase1_config_dict['edge_input_dim']}"
)
print(
    f"  Context dimension      : "
    f"{phase1_config_dict['context_input_dim']}"
)

print("\nReproducibility:")
print(
    f"  Random seed            : "
    f"{phase1_config.seed}"
)

print("\nSaved configuration:")
print(
    f"  {PHASE1_CONFIG_PATH}"
)

print("\nSTATUS: CONFIG FINALIZED AND SAVED")

print("=" * 70)

# --- from TODO 7.2 ---
# ============================================================
# TODO 7.2 — Implement node encoder
# ============================================================

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Node encoder
# ------------------------------------------------------------

class NodeEncoder(nn.Module):
    """
    Encode the transformed static node feature vector into the
    hidden representation used by the GNN.

    Input
    -----
    x : FloatTensor [num_nodes, node_input_dim]

    Output
    ------
    h : FloatTensor [num_nodes, hidden_dim]

    IMPORTANT:
    This encoder only processes intrinsic/static node features.
    It must not receive core type, DVFS, or dynamic system state.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(

            # Initial projection:
            # node feature space -> GNN hidden space
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),

            nn.LayerNorm(
                self.hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            # Additional nonlinear transformation inside
            # the hidden representation space
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),

            nn.LayerNorm(
                self.hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )


    def forward(self, x):
        """
        Encode transformed node features.
        """

        # ----------------------------------------------------
        # Validate tensor structure
        # ----------------------------------------------------

        if not torch.is_tensor(x):
            raise TypeError(
                "x must be a PyTorch tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must have shape [num_nodes, node_input_dim].\n"
                f"Found: {tuple(x.shape)}"
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                "Node input dimension mismatch.\n"
                f"Expected: {self.input_dim}\n"
                f"Found:    {x.shape[1]}"
            )

        if not x.is_floating_point():
            raise TypeError(
                "x must be a floating-point tensor."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "x contains NaN or Inf values."
            )


        # ----------------------------------------------------
        # Encode nodes independently
        # ----------------------------------------------------

        h = self.encoder(x)


        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        expected_shape = (
            x.shape[0],
            self.hidden_dim,
        )

        if tuple(h.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected NodeEncoder output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found:    {tuple(h.shape)}"
            )

        if not torch.isfinite(h).all():
            raise RuntimeError(
                "NodeEncoder produced NaN or Inf values."
            )

        return h


# ------------------------------------------------------------
# Create a node encoder instance for verification
# ------------------------------------------------------------

node_input_dim = len(
    MODEL_NODE_FEATURES
)

node_encoder = NodeEncoder(
    input_dim=node_input_dim,
    hidden_dim=phase1_config.hidden_dim,
    dropout=phase1_config.dropout,
).to(DEVICE)


# ------------------------------------------------------------
# Smoke test using one real training DAG
# ------------------------------------------------------------

sample_x = (
    train_dataset[0]["x"]
    .to(DEVICE)
)


# Evaluation mode avoids dropout randomness during the test.
node_encoder.eval()

with torch.no_grad():

    sample_h = node_encoder(
        sample_x
    )


# ------------------------------------------------------------
# Strict output validation
# ------------------------------------------------------------

expected_output_shape = (
    sample_x.shape[0],
    phase1_config.hidden_dim,
)

if tuple(sample_h.shape) != expected_output_shape:
    raise ValueError(
        "NodeEncoder smoke-test shape mismatch.\n"
        f"Expected: {expected_output_shape}\n"
        f"Found:    {tuple(sample_h.shape)}"
    )

if not torch.isfinite(sample_h).all():
    raise ValueError(
        "NodeEncoder smoke-test output contains NaN or Inf."
    )


# ------------------------------------------------------------
# Verify complete context independence
# ------------------------------------------------------------

# The encoder input dimension must be exactly the static
# node-feature dimension. Runtime context features must not
# appear here.

if node_encoder.input_dim != len(MODEL_NODE_FEATURES):
    raise RuntimeError(
        "NodeEncoder input dimension does not match "
        "the static node feature contract."
    )

if node_encoder.input_dim == len(MODEL_CONTEXT_FEATURES):
    raise RuntimeError(
        "NodeEncoder appears to be using context dimensions. "
        "The static node encoder must remain context-independent."
    )


# ------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------

node_encoder_parameter_count = sum(
    parameter.numel()
    for parameter
    in node_encoder.parameters()
)

trainable_node_encoder_parameters = sum(
    parameter.numel()
    for parameter
    in node_encoder.parameters()
    if parameter.requires_grad
)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("Node Encoder Verification")
print("=" * 70)

print("\nArchitecture:")

print(
    f"  Input dimension        : "
    f"{node_encoder.input_dim}"
)

print(
    f"  Hidden dimension       : "
    f"{node_encoder.hidden_dim}"
)

print(
    f"  Dropout                : "
    f"{phase1_config.dropout}"
)


print("\nSmoke-test graph:")

print(
    f"  Graph ID               : "
    f"{train_dataset[0]['graph_id']}"
)

print(
    f"  Number of nodes        : "
    f"{sample_x.shape[0]:,}"
)


print("\nTensor shapes:")

print(
    f"  Input x                : "
    f"{tuple(sample_x.shape)}"
)

print(
    f"  Encoded h              : "
    f"{tuple(sample_h.shape)}"
)


print("\nParameters:")

print(
    f"  Total parameters       : "
    f"{node_encoder_parameter_count:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_node_encoder_parameters:,}"
)


print("\nContext independence:")

print(
    "  - Core type is NOT used."
)

print(
    "  - DVFS state is NOT used."
)

print(
    "  - Dynamic system state z_t is NOT used."
)


print("\nValidation:")

print(
    "  - Input feature dimension is correct."
)

print(
    "  - Output hidden dimension is correct."
)

print(
    "  - Node count is preserved."
)

print(
    "  - No NaN/Inf values were produced."
)

print(
    "  - Encoder is independent of runtime context."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.3 ---
# ============================================================
# TODO 7.3 — Implement one bidirectional edge-aware attention layer
# ============================================================

import torch
import torch.nn as nn

from torch_geometric.utils import softmax as pyg_softmax


# ------------------------------------------------------------
# Bidirectional edge-aware attention layer
# ------------------------------------------------------------

class BidirectionalEdgeAwareAttentionLayer(nn.Module):
    """
    One bidirectional edge-aware GNN layer.

    For every node v:

        self:
            W_s h_v

        predecessor aggregation:
            sum_{u in pred(v)}
                alpha_uv * (W_p h_u + U_p r_uv)

        successor aggregation:
            sum_{w in succ(v)}
                beta_vw * (W_q h_w + U_q r_vw)

    Predecessor attention:
        alpha_uv = softmax over incoming edges of v

    Successor attention:
        beta_vw = softmax over outgoing edges of v

    Both raw attention scores depend on:
        - source node representation
        - destination node representation
        - edge attributes

    Parameters
    ----------
    hidden_dim:
        Dimension of node hidden representations.

    edge_dim:
        Dimension of edge attributes.

    attention_hidden_dim:
        Hidden dimension of the attention scoring networks.

    dropout:
        Dropout probability applied to the layer output.

    use_residual:
        If True, add the original h to the updated representation.

    use_layer_norm:
        If True, apply LayerNorm to the final representation.
    """

    def __init__(
        self,
        hidden_dim,
        edge_dim,
        attention_hidden_dim,
        dropout=0.10,
        use_residual=False,
        use_layer_norm=True,
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validate constructor arguments
        # ----------------------------------------------------

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if edge_dim <= 0:
            raise ValueError(
                "edge_dim must be positive."
            )

        if attention_hidden_dim <= 0:
            raise ValueError(
                "attention_hidden_dim must be positive."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )


        self.hidden_dim = int(
            hidden_dim
        )

        self.edge_dim = int(
            edge_dim
        )

        self.attention_hidden_dim = int(
            attention_hidden_dim
        )

        self.use_residual = bool(
            use_residual
        )

        self.use_layer_norm = bool(
            use_layer_norm
        )


        # ----------------------------------------------------
        # SELF transformation
        #
        # W_s h_v
        # ----------------------------------------------------

        self.W_s = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=True,
        )


        # ----------------------------------------------------
        # PREDECESSOR message transformations
        #
        # W_p h_u + U_p r_uv
        # ----------------------------------------------------

        self.W_p = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )

        self.U_p = nn.Linear(
            self.edge_dim,
            self.hidden_dim,
            bias=False,
        )


        # ----------------------------------------------------
        # SUCCESSOR message transformations
        #
        # W_q h_w + U_q r_vw
        # ----------------------------------------------------

        self.W_q = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )

        self.U_q = nn.Linear(
            self.edge_dim,
            self.hidden_dim,
            bias=False,
        )


        # ----------------------------------------------------
        # Attention input dimension
        #
        # [h_source || h_destination || edge_attr]
        # ----------------------------------------------------

        attention_input_dim = (
            2 * self.hidden_dim
            + self.edge_dim
        )


        # ----------------------------------------------------
        # PREDECESSOR attention network
        #
        # raw_alpha_uv =
        #     f_alpha(h_u, h_v, r_uv)
        # ----------------------------------------------------

        self.predecessor_attention = nn.Sequential(

            nn.Linear(
                attention_input_dim,
                self.attention_hidden_dim,
            ),

            nn.LeakyReLU(
                negative_slope=0.2
            ),

            nn.Linear(
                self.attention_hidden_dim,
                1,
                bias=False,
            ),
        )


        # ----------------------------------------------------
        # SUCCESSOR attention network
        #
        # raw_beta_vw =
        #     f_beta(h_v, h_w, r_vw)
        # ----------------------------------------------------

        self.successor_attention = nn.Sequential(

            nn.Linear(
                attention_input_dim,
                self.attention_hidden_dim,
            ),

            nn.LeakyReLU(
                negative_slope=0.2
            ),

            nn.Linear(
                self.attention_hidden_dim,
                1,
                bias=False,
            ),
        )


        # ----------------------------------------------------
        # Activation / dropout / normalization
        # ----------------------------------------------------

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                self.hidden_dim
            )
        else:
            self.layer_norm = nn.Identity()


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        h,
        edge_index,
        edge_attr,
        return_attention=False,
    ):
        """
        Parameters
        ----------
        h:
            FloatTensor [num_nodes, hidden_dim]

        edge_index:
            LongTensor [2, num_edges]

            edge_index[0] = source
            edge_index[1] = destination

        edge_attr:
            FloatTensor [num_edges, edge_dim]

        return_attention:
            If True, also return alpha and beta coefficients.

        Returns
        -------
        h_next:
            FloatTensor [num_nodes, hidden_dim]

        optionally:
            attention dictionary
        """


        # ----------------------------------------------------
        # Validate node representation
        # ----------------------------------------------------

        if not torch.is_tensor(h):
            raise TypeError(
                "h must be a PyTorch tensor."
            )

        if h.ndim != 2:
            raise ValueError(
                "h must have shape [num_nodes, hidden_dim].\n"
                f"Found: {tuple(h.shape)}"
            )

        if h.shape[1] != self.hidden_dim:
            raise ValueError(
                "Hidden dimension mismatch.\n"
                f"Expected: {self.hidden_dim}\n"
                f"Found:    {h.shape[1]}"
            )

        if not h.is_floating_point():
            raise TypeError(
                "h must be floating point."
            )


        # ----------------------------------------------------
        # Validate edge_index
        # ----------------------------------------------------

        if not torch.is_tensor(
            edge_index
        ):
            raise TypeError(
                "edge_index must be a PyTorch tensor."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have exactly two rows."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "edge_index must have dtype torch.long."
            )


        # ----------------------------------------------------
        # Validate edge_attr
        # ----------------------------------------------------

        if not torch.is_tensor(
            edge_attr
        ):
            raise TypeError(
                "edge_attr must be a PyTorch tensor."
            )

        if edge_attr.ndim != 2:
            raise ValueError(
                "edge_attr must have shape "
                "[num_edges, edge_dim]."
            )

        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                "Edge feature dimension mismatch.\n"
                f"Expected: {self.edge_dim}\n"
                f"Found:    {edge_attr.shape[1]}"
            )

        if (
            edge_attr.shape[0]
            != edge_index.shape[1]
        ):
            raise ValueError(
                "edge_attr rows must match "
                "edge_index columns.\n"
                f"edge_attr rows:     "
                f"{edge_attr.shape[0]}\n"
                f"edge_index columns: "
                f"{edge_index.shape[1]}"
            )

        if not edge_attr.is_floating_point():
            raise TypeError(
                "edge_attr must be floating point."
            )


        # ----------------------------------------------------
        # Validate finite values
        # ----------------------------------------------------

        if not torch.isfinite(h).all():
            raise ValueError(
                "h contains NaN or Inf values."
            )

        if not torch.isfinite(
            edge_attr
        ).all():
            raise ValueError(
                "edge_attr contains NaN or Inf values."
            )


        num_nodes = h.shape[0]

        num_edges = (
            edge_index.shape[1]
        )


        # ----------------------------------------------------
        # SELF term
        #
        # W_s h_v
        # ----------------------------------------------------

        self_term = self.W_s(
            h
        )


        # ----------------------------------------------------
        # Handle graphs with zero edges
        # ----------------------------------------------------

        if num_edges == 0:

            predecessor_aggregate = (
                torch.zeros_like(h)
            )

            successor_aggregate = (
                torch.zeros_like(h)
            )

            alpha = torch.empty(
                0,
                dtype=h.dtype,
                device=h.device,
            )

            beta = torch.empty(
                0,
                dtype=h.dtype,
                device=h.device,
            )


        else:

            # ------------------------------------------------
            # Edge orientation
            #
            # For each original DAG edge:
            #
            #       source -----> destination
            #
            #       u      -----> v
            #
            # For predecessor aggregation:
            #       u contributes to v
            #
            # For successor aggregation:
            #       v receives information from its successor w
            #       by aggregating the edge back to its source.
            # ------------------------------------------------

            source = edge_index[0]

            destination = (
                edge_index[1]
            )


            # ------------------------------------------------
            # Gather endpoint node representations
            # ------------------------------------------------

            h_source = h[
                source
            ]

            h_destination = h[
                destination
            ]


            # =================================================
            # PREDECESSOR DIRECTION
            # =================================================


            # ------------------------------------------------
            # Predecessor messages
            #
            # m_uv^pred =
            #     W_p h_u + U_p r_uv
            # ------------------------------------------------

            predecessor_messages = (
                self.W_p(
                    h_source
                )
                +
                self.U_p(
                    edge_attr
                )
            )


            # ------------------------------------------------
            # Raw predecessor attention
            #
            # e_uv =
            # f_alpha(h_u, h_v, r_uv)
            # ------------------------------------------------

            predecessor_attention_input = (
                torch.cat(
                    [
                        h_source,
                        h_destination,
                        edge_attr,
                    ],
                    dim=-1,
                )
            )

            predecessor_logits = (
                self.predecessor_attention(
                    predecessor_attention_input
                )
                .squeeze(-1)
            )


            # ------------------------------------------------
            # alpha_uv:
            #
            # softmax over all predecessors u of each v
            #
            # Grouping index = destination
            # ------------------------------------------------

            alpha = pyg_softmax(
                predecessor_logits,
                index=destination,
                num_nodes=num_nodes,
            )


            # ------------------------------------------------
            # Weighted predecessor messages
            # ------------------------------------------------

            weighted_predecessor_messages = (
                alpha.unsqueeze(-1)
                * predecessor_messages
            )


            # ------------------------------------------------
            # Aggregate predecessor messages into v
            # ------------------------------------------------

            predecessor_aggregate = (
                torch.zeros_like(h)
            )

            predecessor_aggregate.index_add_(
                0,
                destination,
                weighted_predecessor_messages,
            )


            # =================================================
            # SUCCESSOR DIRECTION
            # =================================================


            # ------------------------------------------------
            # Successor messages
            #
            # For original edge:
            #
            #       v -> w
            #
            # destination representation is h_w.
            #
            # m_vw^succ =
            #     W_q h_w + U_q r_vw
            # ------------------------------------------------

            successor_messages = (
                self.W_q(
                    h_destination
                )
                +
                self.U_q(
                    edge_attr
                )
            )


            # ------------------------------------------------
            # Raw successor attention
            #
            # e_vw =
            # f_beta(h_v, h_w, r_vw)
            # ------------------------------------------------

            successor_attention_input = (
                torch.cat(
                    [
                        h_source,
                        h_destination,
                        edge_attr,
                    ],
                    dim=-1,
                )
            )

            successor_logits = (
                self.successor_attention(
                    successor_attention_input
                )
                .squeeze(-1)
            )


            # ------------------------------------------------
            # beta_vw:
            #
            # softmax over all successors w of each v
            #
            # Grouping index = source
            # ------------------------------------------------

            beta = pyg_softmax(
                successor_logits,
                index=source,
                num_nodes=num_nodes,
            )


            # ------------------------------------------------
            # Weighted successor messages
            # ------------------------------------------------

            weighted_successor_messages = (
                beta.unsqueeze(-1)
                * successor_messages
            )


            # ------------------------------------------------
            # Aggregate successor messages back into v
            #
            # The original source node receives information
            # from its outgoing successor.
            # ------------------------------------------------

            successor_aggregate = (
                torch.zeros_like(h)
            )

            successor_aggregate.index_add_(
                0,
                source,
                weighted_successor_messages,
            )


        # ----------------------------------------------------
        # Combine all required components
        #
        # W_s h_v
        # +
        # predecessor aggregation
        # +
        # successor aggregation
        # ----------------------------------------------------

        h_next = (
            self_term
            +
            predecessor_aggregate
            +
            successor_aggregate
        )


        # ----------------------------------------------------
        # Nonlinear activation
        # ----------------------------------------------------

        h_next = self.activation(
            h_next
        )


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        h_next = self.dropout(
            h_next
        )


        # ----------------------------------------------------
        # Optional residual connection
        # ----------------------------------------------------

        if self.use_residual:

            h_next = (
                h_next + h
            )


        # ----------------------------------------------------
        # Optional LayerNorm
        # ----------------------------------------------------

        h_next = self.layer_norm(
            h_next
        )


        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        expected_shape = (
            num_nodes,
            self.hidden_dim,
        )

        if tuple(
            h_next.shape
        ) != expected_shape:
            raise RuntimeError(
                "Unexpected layer output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found:    {tuple(h_next.shape)}"
            )

        if not torch.isfinite(
            h_next
        ).all():
            raise RuntimeError(
                "GNN layer produced NaN or Inf values."
            )


        # ----------------------------------------------------
        # Return output
        # ----------------------------------------------------

        if return_attention:

            return (
                h_next,
                {
                    "alpha_predecessor": alpha,
                    "beta_successor": beta,
                },
            )

        return h_next


# ============================================================
# Smoke test on one real training DAG
# ============================================================

sample_item = train_dataset[0]

sample_x = (
    sample_item["x"]
    .to(DEVICE)
)

sample_edge_index = (
    sample_item["edge_index"]
    .to(DEVICE)
)

sample_edge_attr = (
    sample_item["edge_attr"]
    .to(DEVICE)
)


# ------------------------------------------------------------
# Encode static node features first
# ------------------------------------------------------------

node_encoder.eval()

with torch.no_grad():

    sample_h0 = node_encoder(
        sample_x
    )


# ------------------------------------------------------------
# Create one bidirectional attention layer
# ------------------------------------------------------------

bidirectional_attention_layer = (
    BidirectionalEdgeAwareAttentionLayer(

        hidden_dim=(
            phase1_config.hidden_dim
        ),

        edge_dim=len(
            MODEL_EDGE_FEATURES
        ),

        attention_hidden_dim=(
            phase1_config.attention_hidden_dim
        ),

        dropout=(
            phase1_config.dropout
        ),

        # Disabled to stay close to the required
        # mathematical update equation.
        use_residual=False,

        use_layer_norm=True,
    )
    .to(DEVICE)
)


# ------------------------------------------------------------
# Run one forward pass
# ------------------------------------------------------------

bidirectional_attention_layer.eval()

with torch.no_grad():

    (
        sample_h1,
        sample_attention,
    ) = (
        bidirectional_attention_layer(
            h=sample_h0,
            edge_index=sample_edge_index,
            edge_attr=sample_edge_attr,
            return_attention=True,
        )
    )


alpha = (
    sample_attention[
        "alpha_predecessor"
    ]
)

beta = (
    sample_attention[
        "beta_successor"
    ]
)


# ============================================================
# Strict attention validation
# ============================================================

num_nodes = (
    sample_h0.shape[0]
)

num_edges = (
    sample_edge_index.shape[1]
)


# ------------------------------------------------------------
# Validate attention vector lengths
# ------------------------------------------------------------

if alpha.shape != (num_edges,):
    raise ValueError(
        "Predecessor attention shape mismatch.\n"
        f"Expected: {(num_edges,)}\n"
        f"Found:    {tuple(alpha.shape)}"
    )

if beta.shape != (num_edges,):
    raise ValueError(
        "Successor attention shape mismatch.\n"
        f"Expected: {(num_edges,)}\n"
        f"Found:    {tuple(beta.shape)}"
    )


# ------------------------------------------------------------
# Validate predecessor softmax normalization
# ------------------------------------------------------------

if num_edges > 0:

    source = (
        sample_edge_index[0]
    )

    destination = (
        sample_edge_index[1]
    )


    # Sum alpha over all incoming edges of each node.
    alpha_sums = torch.zeros(
        num_nodes,
        dtype=alpha.dtype,
        device=alpha.device,
    )

    alpha_sums.index_add_(
        0,
        destination,
        alpha,
    )


    incoming_counts = torch.zeros(
        num_nodes,
        dtype=torch.long,
        device=DEVICE,
    )

    incoming_counts.index_add_(
        0,
        destination,
        torch.ones(
            num_edges,
            dtype=torch.long,
            device=DEVICE,
        ),
    )


    nodes_with_predecessors = (
        incoming_counts > 0
    )

    if not torch.allclose(
        alpha_sums[
            nodes_with_predecessors
        ],
        torch.ones_like(
            alpha_sums[
                nodes_with_predecessors
            ]
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Predecessor attention weights "
            "do not sum to 1 over predecessors."
        )


    # --------------------------------------------------------
    # Validate successor softmax normalization
    # --------------------------------------------------------

    beta_sums = torch.zeros(
        num_nodes,
        dtype=beta.dtype,
        device=beta.device,
    )

    beta_sums.index_add_(
        0,
        source,
        beta,
    )


    outgoing_counts = torch.zeros(
        num_nodes,
        dtype=torch.long,
        device=DEVICE,
    )

    outgoing_counts.index_add_(
        0,
        source,
        torch.ones(
            num_edges,
            dtype=torch.long,
            device=DEVICE,
        ),
    )


    nodes_with_successors = (
        outgoing_counts > 0
    )

    if not torch.allclose(
        beta_sums[
            nodes_with_successors
        ],
        torch.ones_like(
            beta_sums[
                nodes_with_successors
            ]
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Successor attention weights "
            "do not sum to 1 over successors."
        )


# ------------------------------------------------------------
# Validate output
# ------------------------------------------------------------

expected_output_shape = (
    sample_h0.shape[0],
    phase1_config.hidden_dim,
)

if tuple(
    sample_h1.shape
) != expected_output_shape:
    raise ValueError(
        "Bidirectional attention layer "
        "output shape mismatch.\n"
        f"Expected: {expected_output_shape}\n"
        f"Found:    {tuple(sample_h1.shape)}"
    )

if not torch.isfinite(
    sample_h1
).all():
    raise ValueError(
        "Layer output contains NaN or Inf values."
    )


# ------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------

layer_parameter_count = sum(
    parameter.numel()
    for parameter
    in bidirectional_attention_layer.parameters()
)

trainable_layer_parameters = sum(
    parameter.numel()
    for parameter
    in bidirectional_attention_layer.parameters()
    if parameter.requires_grad
)


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Bidirectional Edge-Aware Attention Layer Verification")
print("=" * 70)

print("\nGraph:")

print(
    f"  Graph ID               : "
    f"{sample_item['graph_id']}"
)

print(
    f"  Nodes                  : "
    f"{num_nodes:,}"
)

print(
    f"  Edges                  : "
    f"{num_edges:,}"
)


print("\nDimensions:")

print(
    f"  Hidden dimension       : "
    f"{phase1_config.hidden_dim}"
)

print(
    f"  Edge dimension         : "
    f"{len(MODEL_EDGE_FEATURES)}"
)

print(
    f"  Attention hidden dim   : "
    f"{phase1_config.attention_hidden_dim}"
)


print("\nTensor shapes:")

print(
    f"  Input h                : "
    f"{tuple(sample_h0.shape)}"
)

print(
    f"  edge_index             : "
    f"{tuple(sample_edge_index.shape)}"
)

print(
    f"  edge_attr              : "
    f"{tuple(sample_edge_attr.shape)}"
)

print(
    f"  Output h_next          : "
    f"{tuple(sample_h1.shape)}"
)

print(
    f"  alpha                  : "
    f"{tuple(alpha.shape)}"
)

print(
    f"  beta                   : "
    f"{tuple(beta.shape)}"
)


print("\nParameters:")

print(
    f"  Total parameters       : "
    f"{layer_parameter_count:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_layer_parameters:,}"
)


print("\nRequired message components:")

print(
    "  - SELF: W_s h_v"
)

print(
    "  - PREDECESSOR: W_p h_u + U_p r_uv"
)

print(
    "  - SUCCESSOR: W_q h_w + U_q r_vw"
)


print("\nAttention:")

print(
    "  - alpha is normalized over predecessors."
)

print(
    "  - beta is normalized over successors."
)

print(
    "  - Raw alpha depends on source, destination, and edge_attr."
)

print(
    "  - Raw beta depends on source, destination, and edge_attr."
)


print("\nEdge-awareness:")

print(
    "  - edge_attr is used inside predecessor messages."
)

print(
    "  - edge_attr is used inside successor messages."
)

print(
    "  - edge_attr is also used inside both attention networks."
)


print("\nValidation:")

print(
    "  - Node count is preserved."
)

print(
    "  - Hidden dimension is preserved."
)

print(
    "  - Attention dimensions are correct."
)

print(
    "  - Predecessor attention sums to 1."
)

print(
    "  - Successor attention sums to 1."
)

print(
    "  - No NaN/Inf values were produced."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.4 ---
# ============================================================
# TODO 7.4 — Stack graph layers into DAG encoder
# ============================================================

import torch
import torch.nn as nn


# ------------------------------------------------------------
# DAG encoder
# ------------------------------------------------------------

class DAGEncoder(nn.Module):
    """
    Encode a complete DAG into contextualized node embeddings.

    Pipeline
    --------
    transformed x
        ->
    NodeEncoder
        ->
    BidirectionalEdgeAwareAttentionLayer x num_gnn_layers
        ->
    final node embeddings

    Input
    -----
    x:
        FloatTensor [num_nodes, node_input_dim]

    edge_index:
        LongTensor [2, num_edges]

    edge_attr:
        FloatTensor [num_edges, edge_dim]

    Output
    ------
    h:
        FloatTensor [num_nodes, hidden_dim]

    IMPORTANT:
    The encoder is completely independent of:
        - core_type
        - DVFS
        - z_t

    Therefore, the same DAG embedding can later be reused
    across all runtime scenarios of that DAG.
    """

    def __init__(
        self,
        node_input_dim,
        edge_dim,
        hidden_dim,
        num_gnn_layers,
        attention_hidden_dim,
        dropout,
        use_residual=False,
        use_layer_norm=True,
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if node_input_dim <= 0:
            raise ValueError(
                "node_input_dim must be positive."
            )

        if edge_dim <= 0:
            raise ValueError(
                "edge_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_gnn_layers <= 0:
            raise ValueError(
                "num_gnn_layers must be positive."
            )

        if attention_hidden_dim <= 0:
            raise ValueError(
                "attention_hidden_dim must be positive."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )


        # ----------------------------------------------------
        # Store dimensions
        # ----------------------------------------------------

        self.node_input_dim = int(
            node_input_dim
        )

        self.edge_dim = int(
            edge_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.num_gnn_layers = int(
            num_gnn_layers
        )

        self.attention_hidden_dim = int(
            attention_hidden_dim
        )


        # ----------------------------------------------------
        # Initial node encoder
        #
        # [N, node_input_dim]
        #       ->
        # [N, hidden_dim]
        # ----------------------------------------------------

        self.node_encoder = NodeEncoder(
            input_dim=self.node_input_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )


        # ----------------------------------------------------
        # Stack bidirectional edge-aware GNN layers
        # ----------------------------------------------------

        self.gnn_layers = nn.ModuleList(
            [
                BidirectionalEdgeAwareAttentionLayer(

                    hidden_dim=(
                        self.hidden_dim
                    ),

                    edge_dim=(
                        self.edge_dim
                    ),

                    attention_hidden_dim=(
                        self.attention_hidden_dim
                    ),

                    dropout=dropout,

                    # Keep the baseline close to the
                    # required project equation.
                    use_residual=use_residual,

                    use_layer_norm=(
                        use_layer_norm
                    ),
                )

                for _ in range(
                    self.num_gnn_layers
                )
            ]
        )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        return_attention=False,
    ):
        """
        Encode one variable-size DAG.

        The number of nodes and edges may differ between calls.

        Only feature dimensions are fixed.
        """


        # ----------------------------------------------------
        # Validate x
        # ----------------------------------------------------

        if not torch.is_tensor(x):
            raise TypeError(
                "x must be a PyTorch tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must have shape "
                "[num_nodes, node_input_dim].\n"
                f"Found: {tuple(x.shape)}"
            )

        if x.shape[1] != self.node_input_dim:
            raise ValueError(
                "Node feature dimension mismatch.\n"
                f"Expected: {self.node_input_dim}\n"
                f"Found:    {x.shape[1]}"
            )

        if not x.is_floating_point():
            raise TypeError(
                "x must be a floating-point tensor."
            )


        # ----------------------------------------------------
        # Validate edge_index
        # ----------------------------------------------------

        if not torch.is_tensor(
            edge_index
        ):
            raise TypeError(
                "edge_index must be a PyTorch tensor."
            )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                "edge_index must have shape [2, num_edges].\n"
                f"Found: {tuple(edge_index.shape)}"
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "edge_index must have dtype torch.long."
            )


        # ----------------------------------------------------
        # Validate edge_attr
        # ----------------------------------------------------

        if not torch.is_tensor(
            edge_attr
        ):
            raise TypeError(
                "edge_attr must be a PyTorch tensor."
            )

        if edge_attr.ndim != 2:
            raise ValueError(
                "edge_attr must have shape "
                "[num_edges, edge_dim].\n"
                f"Found: {tuple(edge_attr.shape)}"
            )

        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                "Edge feature dimension mismatch.\n"
                f"Expected: {self.edge_dim}\n"
                f"Found:    {edge_attr.shape[1]}"
            )

        if (
            edge_attr.shape[0]
            != edge_index.shape[1]
        ):
            raise ValueError(
                "Number of edge attributes must match "
                "the number of graph edges.\n"
                f"Edges:           "
                f"{edge_index.shape[1]}\n"
                f"edge_attr rows:  "
                f"{edge_attr.shape[0]}"
            )

        if not edge_attr.is_floating_point():
            raise TypeError(
                "edge_attr must be floating point."
            )


        # ----------------------------------------------------
        # Validate graph indices
        # ----------------------------------------------------

        num_nodes = x.shape[0]

        if num_nodes <= 0:
            raise ValueError(
                "The DAG must contain at least one node."
            )

        if edge_index.numel() > 0:

            min_index = int(
                edge_index.min().item()
            )

            max_index = int(
                edge_index.max().item()
            )

            if min_index < 0:
                raise ValueError(
                    "edge_index contains a negative node index."
                )

            if max_index >= num_nodes:
                raise ValueError(
                    "edge_index contains an out-of-range "
                    "node index.\n"
                    f"Maximum index: {max_index}\n"
                    f"Number of nodes: {num_nodes}"
                )


        # ----------------------------------------------------
        # Validate finite input values
        # ----------------------------------------------------

        if not torch.isfinite(x).all():
            raise ValueError(
                "x contains NaN or Inf values."
            )

        if not torch.isfinite(
            edge_attr
        ).all():
            raise ValueError(
                "edge_attr contains NaN or Inf values."
            )


        # ====================================================
        # Stage 1 — Encode static node features
        # ====================================================

        h = self.node_encoder(
            x
        )

        # Shape:
        #
        # [num_nodes, node_input_dim]
        #       ->
        # [num_nodes, hidden_dim]


        # ====================================================
        # Stage 2 — Graph message passing
        # ====================================================

        attention_history = []

        for layer_index, layer in enumerate(
            self.gnn_layers
        ):

            if return_attention:

                (
                    h,
                    layer_attention,
                ) = layer(
                    h=h,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_attention=True,
                )

                attention_history.append(
                    {
                        "layer_index": (
                            layer_index
                        ),

                        "alpha_predecessor": (
                            layer_attention[
                                "alpha_predecessor"
                            ]
                        ),

                        "beta_successor": (
                            layer_attention[
                                "beta_successor"
                            ]
                        ),
                    }
                )

            else:

                h = layer(
                    h=h,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_attention=False,
                )


        # ====================================================
        # Final validation
        # ====================================================

        expected_output_shape = (
            num_nodes,
            self.hidden_dim,
        )

        if tuple(
            h.shape
        ) != expected_output_shape:
            raise RuntimeError(
                "Unexpected DAGEncoder output shape.\n"
                f"Expected: {expected_output_shape}\n"
                f"Found:    {tuple(h.shape)}"
            )

        if not torch.isfinite(
            h
        ).all():
            raise RuntimeError(
                "DAGEncoder produced NaN or Inf values."
            )


        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if return_attention:

            return (
                h,
                attention_history,
            )

        return h


# ============================================================
# Build the Phase 1 DAG encoder
# ============================================================

dag_encoder = DAGEncoder(

    node_input_dim=len(
        MODEL_NODE_FEATURES
    ),

    edge_dim=len(
        MODEL_EDGE_FEATURES
    ),

    hidden_dim=(
        phase1_config.hidden_dim
    ),

    num_gnn_layers=(
        phase1_config.num_gnn_layers
    ),

    attention_hidden_dim=(
        phase1_config.attention_hidden_dim
    ),

    dropout=(
        phase1_config.dropout
    ),

    # Baseline follows the project update equation
    # without adding an extra residual term.
    use_residual=False,

    use_layer_norm=True,

).to(DEVICE)


# ============================================================
# Smoke test on one real DAG
# ============================================================

sample_item = (
    train_dataset[0]
)

sample_x = (
    sample_item["x"]
    .to(DEVICE)
)

sample_edge_index = (
    sample_item["edge_index"]
    .to(DEVICE)
)

sample_edge_attr = (
    sample_item["edge_attr"]
    .to(DEVICE)
)


dag_encoder.eval()

with torch.no_grad():

    (
        sample_final_h,
        sample_attention_history,
    ) = dag_encoder(

        x=sample_x,

        edge_index=(
            sample_edge_index
        ),

        edge_attr=(
            sample_edge_attr
        ),

        return_attention=True,
    )


# ------------------------------------------------------------
# Verify output shape
# ------------------------------------------------------------

expected_sample_shape = (
    sample_x.shape[0],
    phase1_config.hidden_dim,
)

if tuple(
    sample_final_h.shape
) != expected_sample_shape:
    raise ValueError(
        "DAGEncoder smoke-test output shape mismatch.\n"
        f"Expected: {expected_sample_shape}\n"
        f"Found:    {tuple(sample_final_h.shape)}"
    )


# ------------------------------------------------------------
# Verify number of executed GNN layers
# ------------------------------------------------------------

if (
    len(sample_attention_history)
    != phase1_config.num_gnn_layers
):
    raise ValueError(
        "Unexpected number of attention-history entries.\n"
        f"Expected: {phase1_config.num_gnn_layers}\n"
        f"Found:    {len(sample_attention_history)}"
    )


# ============================================================
# Variable graph-size verification
# ============================================================

# Find another training DAG with a different node count.

second_item = None

for dataset_index in range(
    1,
    len(train_dataset)
):

    candidate_item = (
        train_dataset[
            dataset_index
        ]
    )

    if (
        candidate_item["x"].shape[0]
        != sample_item["x"].shape[0]
    ):

        second_item = (
            candidate_item
        )

        break


if second_item is None:
    raise RuntimeError(
        "Could not find two training DAGs with "
        "different node counts for the variable-size test."
    )


second_x = (
    second_item["x"]
    .to(DEVICE)
)

second_edge_index = (
    second_item["edge_index"]
    .to(DEVICE)
)

second_edge_attr = (
    second_item["edge_attr"]
    .to(DEVICE)
)


dag_encoder.eval()

with torch.no_grad():

    second_final_h = (
        dag_encoder(

            x=second_x,

            edge_index=(
                second_edge_index
            ),

            edge_attr=(
                second_edge_attr
            ),

            return_attention=False,
        )
    )


# ------------------------------------------------------------
# Verify second graph output
# ------------------------------------------------------------

expected_second_shape = (
    second_x.shape[0],
    phase1_config.hidden_dim,
)

if tuple(
    second_final_h.shape
) != expected_second_shape:
    raise ValueError(
        "Variable-size DAG test failed.\n"
        f"Expected: {expected_second_shape}\n"
        f"Found:    {tuple(second_final_h.shape)}"
    )


# ------------------------------------------------------------
# Verify same model handled different N
# ------------------------------------------------------------

if (
    sample_x.shape[0]
    == second_x.shape[0]
):
    raise RuntimeError(
        "Variable-size test accidentally used "
        "two graphs with the same node count."
    )


# ------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter
    in dag_encoder.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter
    in dag_encoder.parameters()
    if parameter.requires_grad
)


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("DAG Encoder Verification")
print("=" * 70)


print("\nArchitecture:")

print(
    f"  Node input dimension   : "
    f"{dag_encoder.node_input_dim}"
)

print(
    f"  Edge input dimension   : "
    f"{dag_encoder.edge_dim}"
)

print(
    f"  Hidden dimension       : "
    f"{dag_encoder.hidden_dim}"
)

print(
    f"  GNN layers             : "
    f"{dag_encoder.num_gnn_layers}"
)


print("\nPipeline:")

print(
    "  transformed x"
)

print(
    "      -> NodeEncoder"
)

for layer_index in range(
    dag_encoder.num_gnn_layers
):

    print(
        f"      -> Bidirectional edge-aware layer "
        f"{layer_index + 1}"
    )

print(
    "      -> final node embeddings"
)


print("\nGraph 1:")

print(
    f"  Graph ID               : "
    f"{sample_item['graph_id']}"
)

print(
    f"  Nodes                  : "
    f"{sample_x.shape[0]:,}"
)

print(
    f"  Edges                  : "
    f"{sample_edge_index.shape[1]:,}"
)

print(
    f"  Input shape            : "
    f"{tuple(sample_x.shape)}"
)

print(
    f"  Output shape           : "
    f"{tuple(sample_final_h.shape)}"
)


print("\nGraph 2:")

print(
    f"  Graph ID               : "
    f"{second_item['graph_id']}"
)

print(
    f"  Nodes                  : "
    f"{second_x.shape[0]:,}"
)

print(
    f"  Edges                  : "
    f"{second_edge_index.shape[1]:,}"
)

print(
    f"  Input shape            : "
    f"{tuple(second_x.shape)}"
)

print(
    f"  Output shape           : "
    f"{tuple(second_final_h.shape)}"
)


print("\nParameters:")

print(
    f"  Total parameters       : "
    f"{total_parameters:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_parameters:,}"
)


print("\nContext independence:")

print(
    "  - Core type is NOT used by the DAG encoder."
)

print(
    "  - DVFS is NOT used by the DAG encoder."
)

print(
    "  - Dynamic state z_t is NOT used by the DAG encoder."
)


print("\nVariable-size support:")

print(
    "  - The same DAGEncoder processed two different graph sizes."
)

print(
    "  - No fixed number of nodes is assumed."
)

print(
    "  - No fixed number of edges is assumed."
)

print(
    "  - Only feature dimensions are fixed."
)


print("\nValidation:")

print(
    "  - Node count is preserved."
)

print(
    "  - Final hidden dimension is correct."
)

print(
    "  - All configured GNN layers were executed."
)

print(
    "  - edge_attr is propagated through every GNN layer."
)

print(
    "  - No NaN/Inf values were produced."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.5 ---
# ============================================================
# TODO 7.5 — Implement context encoder / fusion
# ============================================================

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Context fusion encoder
# ------------------------------------------------------------

class ContextFusionEncoder(nn.Module):
    """
    Fuse one target-node embedding with its execution context.

    Allowed inputs per execution record
    -----------------------------------
    1. target node embedding
    2. core_type categorical index
    3. frequency_ghz
    4. voltage_v
    5. seven z_t features

    Forbidden metadata/audit variables are never accepted.

    Input shapes
    ------------
    target_node_embeddings:
        [num_scenarios, node_hidden_dim]

    core_type_indices:
        [num_scenarios]

    context_numeric:
        [num_scenarios, 9]

        Ordered as:
            frequency_ghz
            voltage_v
            cpu_utilization
            ready_queue_length
            active_core_count
            memory_active_tasks
            bus_utilization
            thermal_pressure
            release_jitter_us

    Output
    ------
    fused:
        [num_scenarios, fused_dim]
    """

    def __init__(
        self,
        node_hidden_dim,
        num_core_types,
        numerical_context_dim,
        fused_dim,
        dropout,
        core_embedding_dim=8,
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validate constructor arguments
        # ----------------------------------------------------

        if node_hidden_dim <= 0:
            raise ValueError(
                "node_hidden_dim must be positive."
            )

        if num_core_types <= 0:
            raise ValueError(
                "num_core_types must be positive."
            )

        if numerical_context_dim <= 0:
            raise ValueError(
                "numerical_context_dim must be positive."
            )

        if fused_dim <= 0:
            raise ValueError(
                "fused_dim must be positive."
            )

        if core_embedding_dim <= 0:
            raise ValueError(
                "core_embedding_dim must be positive."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )


        # ----------------------------------------------------
        # Store dimensions
        # ----------------------------------------------------

        self.node_hidden_dim = int(
            node_hidden_dim
        )

        self.num_core_types = int(
            num_core_types
        )

        self.numerical_context_dim = int(
            numerical_context_dim
        )

        self.core_embedding_dim = int(
            core_embedding_dim
        )

        self.fused_dim = int(
            fused_dim
        )


        # ----------------------------------------------------
        # Core-type embedding
        #
        # big/little integer index
        #       ->
        # learned vector phi_c
        # ----------------------------------------------------

        self.core_embedding = nn.Embedding(
            num_embeddings=self.num_core_types,
            embedding_dim=self.core_embedding_dim,
        )


        # ----------------------------------------------------
        # Fusion input dimension
        #
        # h_v
        # +
        # phi_c
        # +
        # DVFS
        # +
        # z_t
        # ----------------------------------------------------

        self.fusion_input_dim = (
            self.node_hidden_dim
            + self.core_embedding_dim
            + self.numerical_context_dim
        )


        # ----------------------------------------------------
        # Fusion network
        # ----------------------------------------------------

        self.fusion_network = nn.Sequential(

            nn.Linear(
                self.fusion_input_dim,
                self.fused_dim,
            ),

            nn.LayerNorm(
                self.fused_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                self.fused_dim,
                self.fused_dim,
            ),

            nn.LayerNorm(
                self.fused_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        target_node_embeddings,
        core_type_indices,
        context_numeric,
    ):
        """
        Produce one fused representation for every
        execution scenario.
        """


        # ----------------------------------------------------
        # Validate target node embeddings
        # ----------------------------------------------------

        if not torch.is_tensor(
            target_node_embeddings
        ):
            raise TypeError(
                "target_node_embeddings must be a PyTorch tensor."
            )

        if target_node_embeddings.ndim != 2:
            raise ValueError(
                "target_node_embeddings must have shape "
                "[num_scenarios, node_hidden_dim].\n"
                f"Found: {tuple(target_node_embeddings.shape)}"
            )

        if (
            target_node_embeddings.shape[1]
            != self.node_hidden_dim
        ):
            raise ValueError(
                "Target-node embedding dimension mismatch.\n"
                f"Expected: {self.node_hidden_dim}\n"
                f"Found:    "
                f"{target_node_embeddings.shape[1]}"
            )

        if not target_node_embeddings.is_floating_point():
            raise TypeError(
                "target_node_embeddings must be floating point."
            )


        # ----------------------------------------------------
        # Validate core_type indices
        # ----------------------------------------------------

        if not torch.is_tensor(
            core_type_indices
        ):
            raise TypeError(
                "core_type_indices must be a PyTorch tensor."
            )

        if core_type_indices.ndim != 1:
            raise ValueError(
                "core_type_indices must have shape "
                "[num_scenarios].\n"
                f"Found: {tuple(core_type_indices.shape)}"
            )

        if core_type_indices.dtype != torch.long:
            raise TypeError(
                "core_type_indices must have dtype torch.long."
            )


        # ----------------------------------------------------
        # Validate numerical context
        # ----------------------------------------------------

        if not torch.is_tensor(
            context_numeric
        ):
            raise TypeError(
                "context_numeric must be a PyTorch tensor."
            )

        if context_numeric.ndim != 2:
            raise ValueError(
                "context_numeric must have shape "
                "[num_scenarios, numerical_context_dim].\n"
                f"Found: {tuple(context_numeric.shape)}"
            )

        if (
            context_numeric.shape[1]
            != self.numerical_context_dim
        ):
            raise ValueError(
                "Numerical context dimension mismatch.\n"
                f"Expected: {self.numerical_context_dim}\n"
                f"Found:    {context_numeric.shape[1]}"
            )

        if not context_numeric.is_floating_point():
            raise TypeError(
                "context_numeric must be floating point."
            )


        # ----------------------------------------------------
        # Verify scenario alignment
        # ----------------------------------------------------

        num_scenarios = (
            target_node_embeddings.shape[0]
        )

        if (
            core_type_indices.shape[0]
            != num_scenarios
        ):
            raise ValueError(
                "core_type_indices are not aligned with "
                "target node embeddings.\n"
                f"Target embeddings: {num_scenarios}\n"
                f"Core types:        "
                f"{core_type_indices.shape[0]}"
            )

        if (
            context_numeric.shape[0]
            != num_scenarios
        ):
            raise ValueError(
                "context_numeric is not aligned with "
                "target node embeddings.\n"
                f"Target embeddings: {num_scenarios}\n"
                f"Context rows:      "
                f"{context_numeric.shape[0]}"
            )


        # ----------------------------------------------------
        # Validate core index range
        # ----------------------------------------------------

        if core_type_indices.numel() > 0:

            min_core_index = int(
                core_type_indices.min().item()
            )

            max_core_index = int(
                core_type_indices.max().item()
            )

            if min_core_index < 0:
                raise ValueError(
                    "core_type_indices contains "
                    "a negative category index."
                )

            if (
                max_core_index
                >= self.num_core_types
            ):
                raise ValueError(
                    "core_type_indices contains an "
                    "out-of-range category index.\n"
                    f"Maximum index: {max_core_index}\n"
                    f"Number of core types: "
                    f"{self.num_core_types}"
                )


        # ----------------------------------------------------
        # Validate finite numerical inputs
        # ----------------------------------------------------

        if not torch.isfinite(
            target_node_embeddings
        ).all():
            raise ValueError(
                "target_node_embeddings contains NaN or Inf."
            )

        if not torch.isfinite(
            context_numeric
        ).all():
            raise ValueError(
                "context_numeric contains NaN or Inf."
            )


        # ====================================================
        # Stage 1 — Core type encoding
        # ====================================================

        core_embedding = (
            self.core_embedding(
                core_type_indices
            )
        )

        # Shape:
        #
        # [S]
        #   ->
        # [S, core_embedding_dim]


        # ====================================================
        # Stage 2 — Concatenate allowed model inputs
        # ====================================================

        fusion_input = torch.cat(
            [
                target_node_embeddings,
                core_embedding,
                context_numeric,
            ],
            dim=-1,
        )

        # Shape:
        #
        # [S, hidden_dim]
        # +
        # [S, core_embedding_dim]
        # +
        # [S, 9]
        #
        # ->
        # [S, fusion_input_dim]


        # ----------------------------------------------------
        # Verify concatenated dimension
        # ----------------------------------------------------

        if (
            fusion_input.shape[1]
            != self.fusion_input_dim
        ):
            raise RuntimeError(
                "Unexpected fusion input dimension.\n"
                f"Expected: {self.fusion_input_dim}\n"
                f"Found:    {fusion_input.shape[1]}"
            )


        # ====================================================
        # Stage 3 — Learn fused representation
        # ====================================================

        fused = self.fusion_network(
            fusion_input
        )


        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        expected_shape = (
            num_scenarios,
            self.fused_dim,
        )

        if tuple(
            fused.shape
        ) != expected_shape:
            raise RuntimeError(
                "Unexpected fused representation shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found:    {tuple(fused.shape)}"
            )

        if not torch.isfinite(
            fused
        ).all():
            raise RuntimeError(
                "ContextFusionEncoder produced NaN or Inf."
            )


        return fused


# ============================================================
# Build Phase 1 context-fusion encoder
# ============================================================

NUM_CORE_TYPES = (
    core_type_encoder[
        "num_categories"
    ]
)

NUMERICAL_CONTEXT_DIM = (
    len(MODEL_CONTEXT_FEATURES)
    - 1
)

# Two core categories are small enough that an
# 8-dimensional learned embedding is sufficient.
CORE_EMBEDDING_DIM = 8


context_fusion_encoder = (
    ContextFusionEncoder(

        node_hidden_dim=(
            phase1_config.hidden_dim
        ),

        num_core_types=(
            NUM_CORE_TYPES
        ),

        numerical_context_dim=(
            NUMERICAL_CONTEXT_DIM
        ),

        # The fused representation will be consumed
        # by the quantile prediction head.
        fused_dim=(
            phase1_config.head_hidden_dim
        ),

        dropout=(
            phase1_config.dropout
        ),

        core_embedding_dim=(
            CORE_EMBEDDING_DIM
        ),
    )
    .to(DEVICE)
)


# ============================================================
# Smoke test on one real training DAG
# ============================================================

sample_item = (
    train_dataset[0]
)


# ------------------------------------------------------------
# Move static graph to device
# ------------------------------------------------------------

sample_x = (
    sample_item["x"]
    .to(DEVICE)
)

sample_edge_index = (
    sample_item["edge_index"]
    .to(DEVICE)
)

sample_edge_attr = (
    sample_item["edge_attr"]
    .to(DEVICE)
)


# ------------------------------------------------------------
# Move scenario information to device
# ------------------------------------------------------------

sample_target_node_indices = (
    sample_item[
        "target_node_indices"
    ]
    .to(DEVICE)
)

sample_core_type_indices = (
    sample_item[
        "core_type_indices"
    ]
    .to(DEVICE)
)

sample_context_numeric = (
    sample_item[
        "context_numeric"
    ]
    .to(DEVICE)
)


# ============================================================
# Step 1 — Encode the DAG once
# ============================================================

dag_encoder.eval()
context_fusion_encoder.eval()

with torch.no_grad():

    sample_node_embeddings = (
        dag_encoder(
            x=sample_x,
            edge_index=sample_edge_index,
            edge_attr=sample_edge_attr,
        )
    )


    # --------------------------------------------------------
    # Step 2 — Gather target-node embeddings
    # --------------------------------------------------------

    sample_target_embeddings = (
        sample_node_embeddings[
            sample_target_node_indices
        ]
    )


    # --------------------------------------------------------
    # Step 3 — Fuse target embedding with runtime context
    # --------------------------------------------------------

    sample_fused = (
        context_fusion_encoder(

            target_node_embeddings=(
                sample_target_embeddings
            ),

            core_type_indices=(
                sample_core_type_indices
            ),

            context_numeric=(
                sample_context_numeric
            ),
        )
    )


# ============================================================
# Strict smoke-test validation
# ============================================================

num_scenarios = (
    sample_target_node_indices.shape[0]
)


# ------------------------------------------------------------
# Validate target gathering
# ------------------------------------------------------------

expected_target_embedding_shape = (
    num_scenarios,
    phase1_config.hidden_dim,
)

if tuple(
    sample_target_embeddings.shape
) != expected_target_embedding_shape:
    raise ValueError(
        "Target-node embedding shape mismatch.\n"
        f"Expected: {expected_target_embedding_shape}\n"
        f"Found:    "
        f"{tuple(sample_target_embeddings.shape)}"
    )


# ------------------------------------------------------------
# Validate fused representation
# ------------------------------------------------------------

expected_fused_shape = (
    num_scenarios,
    phase1_config.head_hidden_dim,
)

if tuple(
    sample_fused.shape
) != expected_fused_shape:
    raise ValueError(
        "Context-fusion smoke-test shape mismatch.\n"
        f"Expected: {expected_fused_shape}\n"
        f"Found:    {tuple(sample_fused.shape)}"
    )


# ------------------------------------------------------------
# Explicit allowed-input contract
# ------------------------------------------------------------

ALLOWED_CONTEXT_INPUTS = [
    "core_type",
    "frequency_ghz",
    "voltage_v",
    "cpu_utilization",
    "ready_queue_length",
    "active_core_count",
    "memory_active_tasks",
    "bus_utilization",
    "thermal_pressure",
    "release_jitter_us",
]

if (
    ALLOWED_CONTEXT_INPUTS
    != MODEL_CONTEXT_FEATURES
):
    raise ValueError(
        "Context fusion feature contract mismatch.\n"
        f"Expected:\n{MODEL_CONTEXT_FEATURES}\n\n"
        f"Defined:\n{ALLOWED_CONTEXT_INPUTS}"
    )


# ------------------------------------------------------------
# Explicit forbidden-input protection
# ------------------------------------------------------------

FORBIDDEN_FUSION_INPUTS = {
    "lambda_v",
    "period_us",
    "deadline_us",
    "core_id",
    "generator_load_score",
    "target_role",
    "target_index",
    "context_index",
    "context_id",
}

for feature_name in ALLOWED_CONTEXT_INPUTS:

    if (
        feature_name
        in FORBIDDEN_FUSION_INPUTS
        or feature_name.startswith("audit_")
    ):
        raise RuntimeError(
            "Forbidden feature entered context fusion:\n"
            f"  - {feature_name}"
        )


# ------------------------------------------------------------
# Validate finite outputs
# ------------------------------------------------------------

if not torch.isfinite(
    sample_target_embeddings
).all():
    raise ValueError(
        "Target embeddings contain NaN or Inf."
    )

if not torch.isfinite(
    sample_fused
).all():
    raise ValueError(
        "Fused representations contain NaN or Inf."
    )


# ------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------

context_fusion_parameter_count = sum(
    parameter.numel()
    for parameter
    in context_fusion_encoder.parameters()
)

trainable_context_fusion_parameters = sum(
    parameter.numel()
    for parameter
    in context_fusion_encoder.parameters()
    if parameter.requires_grad
)


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Context Encoder / Fusion Verification")
print("=" * 70)


print("\nScenario count:")

print(
    f"  Runtime scenarios      : "
    f"{num_scenarios}"
)


print("\nInput dimensions:")

print(
    f"  Target embedding       : "
    f"{phase1_config.hidden_dim}"
)

print(
    f"  Core embedding         : "
    f"{CORE_EMBEDDING_DIM}"
)

print(
    f"  Numerical context      : "
    f"{NUMERICAL_CONTEXT_DIM}"
)

print(
    f"  Fusion input           : "
    f"{context_fusion_encoder.fusion_input_dim}"
)


print("\nOutput dimension:")

print(
    f"  Fused representation   : "
    f"{phase1_config.head_hidden_dim}"
)


print("\nTensor shapes:")

print(
    f"  All node embeddings    : "
    f"{tuple(sample_node_embeddings.shape)}"
)

print(
    f"  Target embeddings      : "
    f"{tuple(sample_target_embeddings.shape)}"
)

print(
    f"  Core indices           : "
    f"{tuple(sample_core_type_indices.shape)}"
)

print(
    f"  Numerical context      : "
    f"{tuple(sample_context_numeric.shape)}"
)

print(
    f"  Fused representations  : "
    f"{tuple(sample_fused.shape)}"
)


print("\nAllowed execution context:")

for feature_name in (
    ALLOWED_CONTEXT_INPUTS
):
    print(
        f"  - {feature_name}"
    )


print("\nForbidden-input protection:")

print(
    "  - No task timing metadata is used."
)

print(
    "  - No core_id is used."
)

print(
    "  - No target/context index metadata is used."
)

print(
    "  - No generator_load_score is used."
)

print(
    "  - No audit_* simulator variables are used."
)


print("\nParameters:")

print(
    f"  Total parameters       : "
    f"{context_fusion_parameter_count:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_context_fusion_parameters:,}"
)


print("\nValidation:")

print(
    "  - One fused representation is produced per scenario."
)

print(
    "  - Target-node embeddings are aligned with contexts."
)

print(
    "  - Core type is represented by a learned embedding."
)

print(
    "  - DVFS and z_t use TRAIN-fitted transformed values."
)

print(
    "  - No forbidden metadata/audit variables are used."
)

print(
    "  - No NaN/Inf values were produced."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.6 ---
# ============================================================
# TODO 7.6 — Implement ordered quantile head
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Ordered quantile head
# ------------------------------------------------------------

class OrderedQuantileHead(nn.Module):
    """
    Predict ordered execution-time quantiles.

    Output order:
        [Q50, Q90, Q95, Q99]

    Ordering is guaranteed by construction:

        Q50 = base

        Q90 = Q50 + softplus(delta_90)

        Q95 = Q90 + softplus(delta_95)

        Q99 = Q95 + softplus(delta_99)

    Therefore:

        Q50 <= Q90 <= Q95 <= Q99

    Input
    -----
    fused:
        FloatTensor [num_contexts, input_dim]

    Output
    ------
    quantiles:
        FloatTensor [num_contexts, 4]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        quantiles,
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )


        expected_quantiles = [
            0.50,
            0.90,
            0.95,
            0.99,
        ]

        if list(quantiles) != expected_quantiles:
            raise ValueError(
                "OrderedQuantileHead expects quantiles:\n"
                f"{expected_quantiles}\n"
                f"Found:\n{list(quantiles)}"
            )


        self.input_dim = int(
            input_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.quantiles = list(
            quantiles
        )

        self.num_quantiles = len(
            self.quantiles
        )


        # ----------------------------------------------------
        # Shared prediction network
        # ----------------------------------------------------

        self.shared_network = nn.Sequential(

            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),

            nn.LayerNorm(
                self.hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),

            nn.LayerNorm(
                self.hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )


        # ----------------------------------------------------
        # Raw quantile parameters
        #
        # Output:
        #
        # raw[:, 0] = Q50 base
        # raw[:, 1] = raw increment Q50 -> Q90
        # raw[:, 2] = raw increment Q90 -> Q95
        # raw[:, 3] = raw increment Q95 -> Q99
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.num_quantiles,
        )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        fused,
        return_raw=False,
    ):
        """
        Produce ordered quantile predictions.
        """


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not torch.is_tensor(
            fused
        ):
            raise TypeError(
                "fused must be a PyTorch tensor."
            )

        if fused.ndim != 2:
            raise ValueError(
                "fused must have shape "
                "[num_contexts, input_dim].\n"
                f"Found: {tuple(fused.shape)}"
            )

        if (
            fused.shape[1]
            != self.input_dim
        ):
            raise ValueError(
                "Fused representation dimension mismatch.\n"
                f"Expected: {self.input_dim}\n"
                f"Found:    {fused.shape[1]}"
            )

        if not fused.is_floating_point():
            raise TypeError(
                "fused must be floating point."
            )

        if not torch.isfinite(
            fused
        ).all():
            raise ValueError(
                "fused contains NaN or Inf values."
            )


        # ====================================================
        # Shared nonlinear representation
        # ====================================================

        hidden = self.shared_network(
            fused
        )


        # ====================================================
        # Raw output parameters
        # ====================================================

        raw = self.output_layer(
            hidden
        )

        # Shape:
        #
        # [num_contexts, 4]


        # ====================================================
        # Ordered quantile construction
        # ====================================================

        # Q50 is unconstrained because the standardized target
        # space may contain negative values.
        q50 = raw[
            :,
            0
        ]


        # Positive increments guarantee non-crossing.
        delta_90 = F.softplus(
            raw[
                :,
                1
            ]
        )

        delta_95 = F.softplus(
            raw[
                :,
                2
            ]
        )

        delta_99 = F.softplus(
            raw[
                :,
                3
            ]
        )


        # ----------------------------------------------------
        # Construct ordered quantiles
        # ----------------------------------------------------

        q90 = (
            q50
            + delta_90
        )

        q95 = (
            q90
            + delta_95
        )

        q99 = (
            q95
            + delta_99
        )


        # ----------------------------------------------------
        # Stack into required output order
        # ----------------------------------------------------

        quantile_predictions = (
            torch.stack(
                [
                    q50,
                    q90,
                    q95,
                    q99,
                ],
                dim=-1,
            )
        )


        # ====================================================
        # Final validation
        # ====================================================

        expected_shape = (
            fused.shape[0],
            self.num_quantiles,
        )

        if tuple(
            quantile_predictions.shape
        ) != expected_shape:
            raise RuntimeError(
                "Unexpected quantile output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found:    "
                f"{tuple(quantile_predictions.shape)}"
            )


        if not torch.isfinite(
            quantile_predictions
        ).all():
            raise RuntimeError(
                "Quantile head produced NaN or Inf values."
            )


        # ----------------------------------------------------
        # Explicit non-crossing validation
        # ----------------------------------------------------

        q50_check = (
            quantile_predictions[
                :,
                0
            ]
        )

        q90_check = (
            quantile_predictions[
                :,
                1
            ]
        )

        q95_check = (
            quantile_predictions[
                :,
                2
            ]
        )

        q99_check = (
            quantile_predictions[
                :,
                3
            ]
        )


        if not torch.all(
            q50_check <= q90_check
        ):
            raise RuntimeError(
                "Q50 > Q90 detected."
            )

        if not torch.all(
            q90_check <= q95_check
        ):
            raise RuntimeError(
                "Q90 > Q95 detected."
            )

        if not torch.all(
            q95_check <= q99_check
        ):
            raise RuntimeError(
                "Q95 > Q99 detected."
            )


        # ----------------------------------------------------
        # Optional raw outputs for diagnostics
        # ----------------------------------------------------

        if return_raw:

            return (
                quantile_predictions,
                {
                    "raw": raw,

                    "delta_90": (
                        delta_90
                    ),

                    "delta_95": (
                        delta_95
                    ),

                    "delta_99": (
                        delta_99
                    ),
                },
            )


        return quantile_predictions


# ============================================================
# Build Phase 1 quantile head
# ============================================================

quantile_head = (
    OrderedQuantileHead(

        input_dim=(
            phase1_config.head_hidden_dim
        ),

        hidden_dim=(
            phase1_config.head_hidden_dim
        ),

        dropout=(
            phase1_config.dropout
        ),

        quantiles=(
            phase1_config.quantiles
        ),
    )
    .to(DEVICE)
)


# ============================================================
# Smoke test using fused representations from TODO 7.5
# ============================================================

quantile_head.eval()

with torch.no_grad():

    (
        sample_quantile_predictions,
        sample_quantile_details,
    ) = quantile_head(

        fused=sample_fused,

        return_raw=True,
    )


# ------------------------------------------------------------
# Extract predicted quantiles
# ------------------------------------------------------------

sample_q50 = (
    sample_quantile_predictions[
        :,
        0
    ]
)

sample_q90 = (
    sample_quantile_predictions[
        :,
        1
    ]
)

sample_q95 = (
    sample_quantile_predictions[
        :,
        2
    ]
)

sample_q99 = (
    sample_quantile_predictions[
        :,
        3
    ]
)


# ============================================================
# Strict smoke-test validation
# ============================================================

num_contexts = (
    sample_fused.shape[0]
)

expected_output_shape = (
    num_contexts,
    4,
)

if tuple(
    sample_quantile_predictions.shape
) != expected_output_shape:
    raise ValueError(
        "Quantile-head output shape mismatch.\n"
        f"Expected: {expected_output_shape}\n"
        f"Found:    "
        f"{tuple(sample_quantile_predictions.shape)}"
    )


# ------------------------------------------------------------
# Verify non-crossing
# ------------------------------------------------------------

if not torch.all(
    sample_q50 <= sample_q90
):
    raise ValueError(
        "Quantile crossing detected: Q50 > Q90."
    )

if not torch.all(
    sample_q90 <= sample_q95
):
    raise ValueError(
        "Quantile crossing detected: Q90 > Q95."
    )

if not torch.all(
    sample_q95 <= sample_q99
):
    raise ValueError(
        "Quantile crossing detected: Q95 > Q99."
    )


# ------------------------------------------------------------
# Calculate crossing count
# ------------------------------------------------------------

crossing_mask = (
    (sample_q50 > sample_q90)
    |
    (sample_q90 > sample_q95)
    |
    (sample_q95 > sample_q99)
)

num_crossings = int(
    crossing_mask
    .sum()
    .item()
)


# ------------------------------------------------------------
# Verify positive increments
# ------------------------------------------------------------

delta_90 = (
    sample_quantile_details[
        "delta_90"
    ]
)

delta_95 = (
    sample_quantile_details[
        "delta_95"
    ]
)

delta_99 = (
    sample_quantile_details[
        "delta_99"
    ]
)

if not torch.all(
    delta_90 >= 0
):
    raise ValueError(
        "Negative Q90 increment detected."
    )

if not torch.all(
    delta_95 >= 0
):
    raise ValueError(
        "Negative Q95 increment detected."
    )

if not torch.all(
    delta_99 >= 0
):
    raise ValueError(
        "Negative Q99 increment detected."
    )


# ------------------------------------------------------------
# Finite-value validation
# ------------------------------------------------------------

if not torch.isfinite(
    sample_quantile_predictions
).all():
    raise ValueError(
        "Quantile predictions contain NaN or Inf."
    )


# ------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------

quantile_head_parameter_count = sum(
    parameter.numel()
    for parameter
    in quantile_head.parameters()
)

trainable_quantile_head_parameters = sum(
    parameter.numel()
    for parameter
    in quantile_head.parameters()
    if parameter.requires_grad
)


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Ordered Quantile Head Verification")
print("=" * 70)


print("\nQuantile order:")

for index, quantile in enumerate(
    phase1_config.quantiles
):
    print(
        f"  Output column {index}       : "
        f"Q{int(quantile * 100)}"
    )


print("\nTensor shapes:")

print(
    f"  Fused input            : "
    f"{tuple(sample_fused.shape)}"
)

print(
    f"  Quantile output        : "
    f"{tuple(sample_quantile_predictions.shape)}"
)


print("\nOrdering mechanism:")

print(
    "  Q50 = unconstrained base prediction"
)

print(
    "  Q90 = Q50 + softplus(delta_90)"
)

print(
    "  Q95 = Q90 + softplus(delta_95)"
)

print(
    "  Q99 = Q95 + softplus(delta_99)"
)


print("\nNon-crossing verification:")

print(
    f"  Number of crossings    : "
    f"{num_crossings}"
)

print(
    "  Q50 <= Q90             : PASSED"
)

print(
    "  Q90 <= Q95             : PASSED"
)

print(
    "  Q95 <= Q99             : PASSED"
)


print("\nParameters:")

print(
    f"  Total parameters       : "
    f"{quantile_head_parameter_count:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_quantile_head_parameters:,}"
)


print("\nValidation:")

print(
    "  - One row is produced per runtime context."
)

print(
    "  - Exactly four quantiles are produced."
)

print(
    "  - Output order is [Q50, Q90, Q95, Q99]."
)

print(
    "  - Quantile crossing is impossible by construction."
)

print(
    "  - No NaN/Inf values were produced."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 7.7 ---
# ============================================================
# TODO 7.7 — Assemble the full predictor
# ============================================================

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Full Phase 1 quantile predictor
# ------------------------------------------------------------

class Phase1QuantilePredictor(nn.Module):
    """
    Full Phase 1 execution-time quantile predictor.

    Forward pipeline
    ----------------
    1. Encode the static DAG once.
    2. Gather embeddings for all target nodes.
    3. Fuse target embeddings with core/DVFS/z_t context.
    4. Predict ordered execution-time quantiles.

    Inputs
    ------
    x:
        [num_nodes, node_input_dim]

    edge_index:
        [2, num_edges]

    edge_attr:
        [num_edges, edge_dim]

    target_node_indices:
        [num_contexts]

    core_type_indices:
        [num_contexts]

    context_numeric:
        [num_contexts, 9]

    Output
    ------
    predictions:
        [num_contexts, 4]

        columns:
            0 -> Q50
            1 -> Q90
            2 -> Q95
            3 -> Q99
    """

    def __init__(
        self,
        config,
        node_input_dim,
        edge_dim,
        num_core_types,
        numerical_context_dim,
        core_embedding_dim=8,
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if node_input_dim <= 0:
            raise ValueError(
                "node_input_dim must be positive."
            )

        if edge_dim <= 0:
            raise ValueError(
                "edge_dim must be positive."
            )

        if num_core_types <= 0:
            raise ValueError(
                "num_core_types must be positive."
            )

        if numerical_context_dim <= 0:
            raise ValueError(
                "numerical_context_dim must be positive."
            )

        if core_embedding_dim <= 0:
            raise ValueError(
                "core_embedding_dim must be positive."
            )


        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.config = config

        self.node_input_dim = int(
            node_input_dim
        )

        self.edge_dim = int(
            edge_dim
        )

        self.num_core_types = int(
            num_core_types
        )

        self.numerical_context_dim = int(
            numerical_context_dim
        )

        self.core_embedding_dim = int(
            core_embedding_dim
        )


        # ====================================================
        # Part 1 — Static DAG encoder
        # ====================================================

        self.dag_encoder = DAGEncoder(

            node_input_dim=(
                self.node_input_dim
            ),

            edge_dim=(
                self.edge_dim
            ),

            hidden_dim=(
                self.config.hidden_dim
            ),

            num_gnn_layers=(
                self.config.num_gnn_layers
            ),

            attention_hidden_dim=(
                self.config.attention_hidden_dim
            ),

            dropout=(
                self.config.dropout
            ),

            # Keep the baseline close to the required
            # mathematical GNN update.
            use_residual=False,

            use_layer_norm=True,
        )


        # ====================================================
        # Part 2 — Runtime context fusion
        # ====================================================

        self.context_fusion = (
            ContextFusionEncoder(

                node_hidden_dim=(
                    self.config.hidden_dim
                ),

                num_core_types=(
                    self.num_core_types
                ),

                numerical_context_dim=(
                    self.numerical_context_dim
                ),

                fused_dim=(
                    self.config.head_hidden_dim
                ),

                dropout=(
                    self.config.dropout
                ),

                core_embedding_dim=(
                    self.core_embedding_dim
                ),
            )
        )


        # ====================================================
        # Part 3 — Ordered quantile head
        # ====================================================

        self.quantile_head = (
            OrderedQuantileHead(

                input_dim=(
                    self.config.head_hidden_dim
                ),

                hidden_dim=(
                    self.config.head_hidden_dim
                ),

                dropout=(
                    self.config.dropout
                ),

                quantiles=(
                    self.config.quantiles
                ),
            )
        )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        target_node_indices,
        core_type_indices,
        context_numeric,
        return_intermediates=False,
    ):
        """
        Predict execution-time quantiles for all runtime
        contexts associated with one DAG.
        """


        # ====================================================
        # Validate scenario tensors
        # ====================================================

        if not torch.is_tensor(
            target_node_indices
        ):
            raise TypeError(
                "target_node_indices must be a PyTorch tensor."
            )

        if target_node_indices.ndim != 1:
            raise ValueError(
                "target_node_indices must have shape "
                "[num_contexts].\n"
                f"Found: {tuple(target_node_indices.shape)}"
            )

        if target_node_indices.dtype != torch.long:
            raise TypeError(
                "target_node_indices must have dtype torch.long."
            )


        if not torch.is_tensor(
            core_type_indices
        ):
            raise TypeError(
                "core_type_indices must be a PyTorch tensor."
            )

        if core_type_indices.ndim != 1:
            raise ValueError(
                "core_type_indices must have shape "
                "[num_contexts].\n"
                f"Found: {tuple(core_type_indices.shape)}"
            )


        if not torch.is_tensor(
            context_numeric
        ):
            raise TypeError(
                "context_numeric must be a PyTorch tensor."
            )

        if context_numeric.ndim != 2:
            raise ValueError(
                "context_numeric must have shape "
                "[num_contexts, numerical_context_dim].\n"
                f"Found: {tuple(context_numeric.shape)}"
            )


        # ----------------------------------------------------
        # Validate scenario alignment
        # ----------------------------------------------------

        num_contexts = (
            target_node_indices.shape[0]
        )

        if (
            core_type_indices.shape[0]
            != num_contexts
        ):
            raise ValueError(
                "core_type_indices are not aligned with "
                "target_node_indices."
            )

        if (
            context_numeric.shape[0]
            != num_contexts
        ):
            raise ValueError(
                "context_numeric rows are not aligned with "
                "target_node_indices."
            )

        if (
            context_numeric.shape[1]
            != self.numerical_context_dim
        ):
            raise ValueError(
                "Numerical context dimension mismatch.\n"
                f"Expected: {self.numerical_context_dim}\n"
                f"Found:    {context_numeric.shape[1]}"
            )


        # ====================================================
        # STEP 1 — Encode DAG ONCE
        # ====================================================

        node_embeddings = (
            self.dag_encoder(

                x=x,

                edge_index=(
                    edge_index
                ),

                edge_attr=(
                    edge_attr
                ),
            )
        )

        # Shape:
        #
        # [num_nodes, hidden_dim]


        # ====================================================
        # STEP 2 — Gather target-node embeddings
        # ====================================================

        if target_node_indices.numel() == 0:
            raise ValueError(
                "No target-node indices were provided."
            )

        min_target_index = int(
            target_node_indices
            .min()
            .item()
        )

        max_target_index = int(
            target_node_indices
            .max()
            .item()
        )

        if min_target_index < 0:
            raise ValueError(
                "target_node_indices contains "
                "a negative index."
            )

        if (
            max_target_index
            >= node_embeddings.shape[0]
        ):
            raise ValueError(
                "target_node_indices contains an "
                "out-of-range node index.\n"
                f"Maximum target index: "
                f"{max_target_index}\n"
                f"Number of nodes: "
                f"{node_embeddings.shape[0]}"
            )


        target_node_embeddings = (
            node_embeddings[
                target_node_indices
            ]
        )

        # Shape:
        #
        # [num_contexts, hidden_dim]


        # ====================================================
        # STEP 3 — Fuse target embedding with execution context
        # ====================================================

        fused = (
            self.context_fusion(

                target_node_embeddings=(
                    target_node_embeddings
                ),

                core_type_indices=(
                    core_type_indices
                ),

                context_numeric=(
                    context_numeric
                ),
            )
        )

        # Shape:
        #
        # [num_contexts, head_hidden_dim]


        # ====================================================
        # STEP 4 — Predict four ordered quantiles
        # ====================================================

        predictions = (
            self.quantile_head(
                fused
            )
        )

        # Shape:
        #
        # [num_contexts, 4]
        #
        # columns:
        # Q50, Q90, Q95, Q99


        # ====================================================
        # Final validation
        # ====================================================

        expected_prediction_shape = (
            num_contexts,
            len(
                self.config.quantiles
            ),
        )

        if tuple(
            predictions.shape
        ) != expected_prediction_shape:
            raise RuntimeError(
                "Unexpected predictor output shape.\n"
                f"Expected: "
                f"{expected_prediction_shape}\n"
                f"Found:    "
                f"{tuple(predictions.shape)}"
            )


        if not torch.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "Predictor produced NaN or Inf values."
            )


        # ----------------------------------------------------
        # Verify quantile ordering
        # ----------------------------------------------------

        if not torch.all(
            predictions[:, 0]
            <= predictions[:, 1]
        ):
            raise RuntimeError(
                "Q50 > Q90 detected."
            )

        if not torch.all(
            predictions[:, 1]
            <= predictions[:, 2]
        ):
            raise RuntimeError(
                "Q90 > Q95 detected."
            )

        if not torch.all(
            predictions[:, 2]
            <= predictions[:, 3]
        ):
            raise RuntimeError(
                "Q95 > Q99 detected."
            )


        # ----------------------------------------------------
        # Optional diagnostic outputs
        # ----------------------------------------------------

        if return_intermediates:

            return {

                "predictions": (
                    predictions
                ),

                "node_embeddings": (
                    node_embeddings
                ),

                "target_node_embeddings": (
                    target_node_embeddings
                ),

                "fused": (
                    fused
                ),
            }


        return predictions


# ============================================================
# Build the complete Phase 1 model
# ============================================================

phase1_model = (
    Phase1QuantilePredictor(

        config=(
            phase1_config
        ),

        node_input_dim=len(
            MODEL_NODE_FEATURES
        ),

        edge_dim=len(
            MODEL_EDGE_FEATURES
        ),

        num_core_types=(
            core_type_encoder[
                "num_categories"
            ]
        ),

        numerical_context_dim=(
            len(MODEL_CONTEXT_FEATURES)
            - 1
        ),

        core_embedding_dim=8,
    )
    .to(DEVICE)
)


# ============================================================
# Smoke test on one real training DAG
# ============================================================

sample_item = (
    train_dataset[0]
)


# ------------------------------------------------------------
# Move graph tensors to device
# ------------------------------------------------------------

sample_x = (
    sample_item["x"]
    .to(DEVICE)
)

sample_edge_index = (
    sample_item["edge_index"]
    .to(DEVICE)
)

sample_edge_attr = (
    sample_item["edge_attr"]
    .to(DEVICE)
)


# ------------------------------------------------------------
# Move scenario tensors to device
# ------------------------------------------------------------

sample_target_node_indices = (
    sample_item[
        "target_node_indices"
    ]
    .to(DEVICE)
)

sample_core_type_indices = (
    sample_item[
        "core_type_indices"
    ]
    .to(DEVICE)
)

sample_context_numeric = (
    sample_item[
        "context_numeric"
    ]
    .to(DEVICE)
)


# ------------------------------------------------------------
# Full forward pass
# ------------------------------------------------------------

phase1_model.eval()

with torch.no_grad():

    sample_outputs = (
        phase1_model(

            x=sample_x,

            edge_index=(
                sample_edge_index
            ),

            edge_attr=(
                sample_edge_attr
            ),

            target_node_indices=(
                sample_target_node_indices
            ),

            core_type_indices=(
                sample_core_type_indices
            ),

            context_numeric=(
                sample_context_numeric
            ),

            return_intermediates=True,
        )
    )


sample_predictions = (
    sample_outputs[
        "predictions"
    ]
)

sample_node_embeddings = (
    sample_outputs[
        "node_embeddings"
    ]
)

sample_target_embeddings = (
    sample_outputs[
        "target_node_embeddings"
    ]
)

sample_fused = (
    sample_outputs[
        "fused"
    ]
)


# ============================================================
# Strict smoke-test validation
# ============================================================

num_contexts = (
    sample_target_node_indices.shape[0]
)

expected_prediction_shape = (
    num_contexts,
    4,
)

if tuple(
    sample_predictions.shape
) != expected_prediction_shape:
    raise ValueError(
        "Full predictor output shape mismatch.\n"
        f"Expected: {expected_prediction_shape}\n"
        f"Found:    "
        f"{tuple(sample_predictions.shape)}"
    )


# ------------------------------------------------------------
# Verify intermediate dimensions
# ------------------------------------------------------------

expected_node_embedding_shape = (
    sample_x.shape[0],
    phase1_config.hidden_dim,
)

if tuple(
    sample_node_embeddings.shape
) != expected_node_embedding_shape:
    raise ValueError(
        "Unexpected DAG embedding shape."
    )


expected_target_embedding_shape = (
    num_contexts,
    phase1_config.hidden_dim,
)

if tuple(
    sample_target_embeddings.shape
) != expected_target_embedding_shape:
    raise ValueError(
        "Unexpected target-node embedding shape."
    )


expected_fused_shape = (
    num_contexts,
    phase1_config.head_hidden_dim,
)

if tuple(
    sample_fused.shape
) != expected_fused_shape:
    raise ValueError(
        "Unexpected fused representation shape."
    )


# ------------------------------------------------------------
# Verify ordered quantiles
# ------------------------------------------------------------

q50 = sample_predictions[:, 0]
q90 = sample_predictions[:, 1]
q95 = sample_predictions[:, 2]
q99 = sample_predictions[:, 3]

crossing_mask = (
    (q50 > q90)
    |
    (q90 > q95)
    |
    (q95 > q99)
)

num_crossings = int(
    crossing_mask
    .sum()
    .item()
)

if num_crossings != 0:
    raise ValueError(
        f"Quantile crossing detected in "
        f"{num_crossings} contexts."
    )


# ------------------------------------------------------------
# Verify finite values
# ------------------------------------------------------------

for tensor_name, tensor in {

    "node_embeddings":
        sample_node_embeddings,

    "target_embeddings":
        sample_target_embeddings,

    "fused":
        sample_fused,

    "predictions":
        sample_predictions,

}.items():

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            f"{tensor_name} contains NaN or Inf."
        )


# ------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter
    in phase1_model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter
    in phase1_model.parameters()
    if parameter.requires_grad
)


dag_encoder_parameters = sum(
    parameter.numel()
    for parameter
    in phase1_model.dag_encoder.parameters()
)

context_fusion_parameters = sum(
    parameter.numel()
    for parameter
    in phase1_model.context_fusion.parameters()
)

quantile_head_parameters = sum(
    parameter.numel()
    for parameter
    in phase1_model.quantile_head.parameters()
)


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Full Phase 1 Predictor Verification")
print("=" * 70)


print("\nGraph:")

print(
    f"  Graph ID               : "
    f"{sample_item['graph_id']}"
)

print(
    f"  Nodes                  : "
    f"{sample_x.shape[0]:,}"
)

print(
    f"  Edges                  : "
    f"{sample_edge_index.shape[1]:,}"
)

print(
    f"  Runtime contexts       : "
    f"{num_contexts}"
)


print("\nForward pipeline:")

print(
    "  1. DAGEncoder"
)

print(
    "  2. Gather target-node embeddings"
)

print(
    "  3. Context fusion"
)

print(
    "  4. Ordered quantile prediction"
)


print("\nTensor shapes:")

print(
    f"  x                      : "
    f"{tuple(sample_x.shape)}"
)

print(
    f"  Node embeddings        : "
    f"{tuple(sample_node_embeddings.shape)}"
)

print(
    f"  Target embeddings      : "
    f"{tuple(sample_target_embeddings.shape)}"
)

print(
    f"  Numerical context      : "
    f"{tuple(sample_context_numeric.shape)}"
)

print(
    f"  Fused representation   : "
    f"{tuple(sample_fused.shape)}"
)

print(
    f"  Final predictions      : "
    f"{tuple(sample_predictions.shape)}"
)


print("\nPrediction columns:")

print(
    "  Column 0               : Q50"
)

print(
    "  Column 1               : Q90"
)

print(
    "  Column 2               : Q95"
)

print(
    "  Column 3               : Q99"
)


print("\nQuantile ordering:")

print(
    f"  Crossing count         : "
    f"{num_crossings}"
)

print(
    "  Q50 <= Q90 <= Q95 <= Q99 : PASSED"
)


print("\nParameter counts:")

print(
    f"  DAG encoder            : "
    f"{dag_encoder_parameters:,}"
)

print(
    f"  Context fusion         : "
    f"{context_fusion_parameters:,}"
)

print(
    f"  Quantile head          : "
    f"{quantile_head_parameters:,}"
)

print(
    f"  Total                  : "
    f"{total_parameters:,}"
)

print(
    f"  Trainable              : "
    f"{trainable_parameters:,}"
)


print("\nInput protection:")

print(
    "  - DAG encoder receives only static node/edge features."
)

print(
    "  - Fusion receives only core_type, DVFS, and z_t."
)

print(
    "  - No forbidden metadata/audit variables are accepted."
)


print("\nValidation:")

print(
    "  - The DAG is encoded once per forward pass."
)

print(
    "  - One target embedding is gathered per runtime context."
)

print(
    "  - One fused representation is produced per runtime context."
)

print(
    "  - Exactly four ordered quantiles are predicted per context."
)

print(
    "  - No NaN/Inf values were produced."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 8.1 ---
# ============================================================
# TODO 8.1 — Implement vectorized pinball loss
# ============================================================

import torch


def pinball_loss(
    predictions,
    targets,
    quantiles,
    return_per_quantile=False,
):
    """
    Vectorized pinball loss for multi-quantile regression.

    Parameters
    ----------
    predictions:
        FloatTensor [N, Q]

        Example for this project:
            [N, 4]

        Columns:
            Q50, Q90, Q95, Q99

    targets:
        FloatTensor [N]

    quantiles:
        Sequence or tensor [Q]

        Example:
            [0.50, 0.90, 0.95, 0.99]

    return_per_quantile:
        If True, also return the mean loss for each quantile.

    Returns
    -------
    total_loss:
        Scalar tensor containing the mean pinball loss over
        all samples and all quantiles.

    per_quantile_loss:
        Optional FloatTensor [Q] containing the mean loss
        for each quantile.
    """


    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    if not torch.is_tensor(predictions):
        raise TypeError(
            "predictions must be a PyTorch tensor."
        )

    if predictions.ndim != 2:
        raise ValueError(
            "predictions must have shape [N, Q].\n"
            f"Found: {tuple(predictions.shape)}"
        )

    if not predictions.is_floating_point():
        raise TypeError(
            "predictions must be floating point."
        )


    # --------------------------------------------------------
    # Validate targets
    # --------------------------------------------------------

    if not torch.is_tensor(targets):
        raise TypeError(
            "targets must be a PyTorch tensor."
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [N].\n"
            f"Found: {tuple(targets.shape)}"
        )

    if not targets.is_floating_point():
        raise TypeError(
            "targets must be floating point."
        )


    # --------------------------------------------------------
    # Validate sample alignment
    # --------------------------------------------------------

    num_samples = predictions.shape[0]
    num_quantiles = predictions.shape[1]

    if num_samples == 0:
        raise ValueError(
            "pinball_loss requires at least one sample."
        )

    if targets.shape[0] != num_samples:
        raise ValueError(
            "predictions and targets are not aligned.\n"
            f"Prediction rows: {num_samples}\n"
            f"Targets:         {targets.shape[0]}"
        )


    # --------------------------------------------------------
    # Convert quantiles to tensor on the correct device
    # --------------------------------------------------------

    quantiles_tensor = torch.as_tensor(
        quantiles,
        dtype=predictions.dtype,
        device=predictions.device,
    )

    if quantiles_tensor.ndim != 1:
        raise ValueError(
            "quantiles must be one-dimensional."
        )

    if quantiles_tensor.shape[0] != num_quantiles:
        raise ValueError(
            "Number of quantiles must match prediction columns.\n"
            f"Prediction columns: {num_quantiles}\n"
            f"Quantiles:          {quantiles_tensor.shape[0]}"
        )

    if not torch.all(
        (quantiles_tensor > 0.0)
        &
        (quantiles_tensor < 1.0)
    ):
        raise ValueError(
            "All quantiles must lie strictly between 0 and 1."
        )


    # --------------------------------------------------------
    # Validate devices
    # --------------------------------------------------------

    if targets.device != predictions.device:
        raise ValueError(
            "predictions and targets must be on the same device."
        )


    # --------------------------------------------------------
    # Validate finite values
    # --------------------------------------------------------

    if not torch.isfinite(
        predictions
    ).all():
        raise ValueError(
            "predictions contains NaN or Inf."
        )

    if not torch.isfinite(
        targets
    ).all():
        raise ValueError(
            "targets contains NaN or Inf."
        )


    # ========================================================
    # Vectorized pinball loss
    # ========================================================

    # targets:
    #     [N]
    #
    # targets.unsqueeze(1):
    #     [N, 1]
    #
    # predictions:
    #     [N, Q]
    #
    # Broadcasting gives:
    #     errors [N, Q]

    errors = (
        targets.unsqueeze(1)
        - predictions
    )


    # --------------------------------------------------------
    # Pinball loss
    #
    # L_q(y, y_hat) =
    #
    #     q * (y - y_hat)       if y >= y_hat
    #
    #     (q - 1) * (y-y_hat)   if y < y_hat
    #
    # Vectorized equivalent:
    #
    # max(
    #     q * error,
    #     (q - 1) * error
    # )
    # --------------------------------------------------------

    losses = torch.maximum(
        quantiles_tensor * errors,
        (quantiles_tensor - 1.0) * errors,
    )

    # losses shape:
    #     [N, Q]


    # --------------------------------------------------------
    # Mean loss for each quantile
    # --------------------------------------------------------

    per_quantile_loss = (
        losses.mean(dim=0)
    )

    # Shape:
    #     [Q]


    # --------------------------------------------------------
    # Overall training objective
    # --------------------------------------------------------

    total_loss = (
        losses.mean()
    )

    # Scalar


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if total_loss.ndim != 0:
        raise RuntimeError(
            "Total pinball loss must be a scalar."
        )

    if (
        per_quantile_loss.shape
        != (num_quantiles,)
    ):
        raise RuntimeError(
            "Unexpected per-quantile loss shape."
        )

    if not torch.isfinite(
        total_loss
    ):
        raise RuntimeError(
            "Total pinball loss is NaN or Inf."
        )

    if not torch.isfinite(
        per_quantile_loss
    ).all():
        raise RuntimeError(
            "Per-quantile loss contains NaN or Inf."
        )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    if return_per_quantile:

        return (
            total_loss,
            per_quantile_loss,
        )

    return total_loss


# ============================================================
# Tiny hand-built unit test
# ============================================================

test_quantiles = [
    0.50,
    0.90,
    0.95,
    0.99,
]


# ------------------------------------------------------------
# Two artificial samples
#
# Sample 1:
# target = 1
# predictions = [0, 1, 2, 3]
#
# Sample 2:
# target = 3
# predictions = [4, 3, 2, 1]
# ------------------------------------------------------------

test_predictions = torch.tensor(
    [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
    ],
    dtype=torch.float32,
)


test_targets = torch.tensor(
    [
        1.0,
        3.0,
    ],
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Expected individual losses
#
# Sample 1:
#
# Q50:
#   error = 1 - 0 = +1
#   loss  = 0.50 * 1 = 0.50
#
# Q90:
#   error = 1 - 1 = 0
#   loss  = 0
#
# Q95:
#   error = 1 - 2 = -1
#   loss  = (0.95 - 1) * (-1) = 0.05
#
# Q99:
#   error = 1 - 3 = -2
#   loss  = (0.99 - 1) * (-2) = 0.02
#
#
# Sample 2:
#
# Q50:
#   error = 3 - 4 = -1
#   loss  = (0.50 - 1) * (-1) = 0.50
#
# Q90:
#   error = 3 - 3 = 0
#   loss  = 0
#
# Q95:
#   error = 3 - 2 = +1
#   loss  = 0.95
#
# Q99:
#   error = 3 - 1 = +2
#   loss  = 0.99 * 2 = 1.98
# ------------------------------------------------------------


# ------------------------------------------------------------
# Expected mean loss per quantile
# ------------------------------------------------------------

expected_per_quantile = torch.tensor(
    [
        0.50,
        0.00,
        0.50,
        1.00,
    ],
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Expected overall mean
#
# Mean of:
#
# [0.50, 0.00, 0.05, 0.02,
#  0.50, 0.00, 0.95, 1.98]
#
# = 4.00 / 8
# = 0.50
# ------------------------------------------------------------

expected_total_loss = torch.tensor(
    0.50,
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Run function
# ------------------------------------------------------------

(
    test_total_loss,
    test_per_quantile_loss,
) = pinball_loss(

    predictions=test_predictions,

    targets=test_targets,

    quantiles=test_quantiles,

    return_per_quantile=True,
)


# ------------------------------------------------------------
# Unit-test assertions
# ------------------------------------------------------------

torch.testing.assert_close(
    test_total_loss,
    expected_total_loss,
    atol=1e-7,
    rtol=1e-7,
)

torch.testing.assert_close(
    test_per_quantile_loss,
    expected_per_quantile,
    atol=1e-7,
    rtol=1e-7,
)


# ============================================================
# Gradient-flow unit test
# ============================================================

gradient_test_predictions = (
    test_predictions
    .clone()
    .detach()
    .requires_grad_(True)
)

gradient_test_loss = pinball_loss(

    predictions=(
        gradient_test_predictions
    ),

    targets=test_targets,

    quantiles=test_quantiles,
)

gradient_test_loss.backward()


if (
    gradient_test_predictions.grad
    is None
):
    raise RuntimeError(
        "Pinball loss did not produce gradients."
    )

if not torch.isfinite(
    gradient_test_predictions.grad
).all():
    raise RuntimeError(
        "Pinball loss produced invalid gradients."
    )


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Vectorized Pinball Loss Verification")
print("=" * 70)


print("\nTiny hand-built example:")

print(
    "  Predictions shape      : "
    f"{tuple(test_predictions.shape)}"
)

print(
    "  Targets shape          : "
    f"{tuple(test_targets.shape)}"
)

print(
    "  Quantiles              : "
    f"{test_quantiles}"
)


print("\nExpected per-quantile loss:")

print(
    f"  Q50                    : "
    f"{expected_per_quantile[0].item():.4f}"
)

print(
    f"  Q90                    : "
    f"{expected_per_quantile[1].item():.4f}"
)

print(
    f"  Q95                    : "
    f"{expected_per_quantile[2].item():.4f}"
)

print(
    f"  Q99                    : "
    f"{expected_per_quantile[3].item():.4f}"
)


print("\nComputed per-quantile loss:")

for quantile, loss_value in zip(
    test_quantiles,
    test_per_quantile_loss,
):

    print(
        f"  Q{int(quantile * 100):02d}"
        f"                    : "
        f"{loss_value.item():.4f}"
    )


print("\nOverall loss:")

print(
    f"  Expected                : "
    f"{expected_total_loss.item():.4f}"
)

print(
    f"  Computed                : "
    f"{test_total_loss.item():.4f}"
)


print("\nGradient test:")

print(
    "  - Backward pass succeeded."
)

print(
    "  - Gradients are finite."
)


print("\nValidation:")

print(
    "  - Loss is fully vectorized."
)

print(
    "  - No loop over samples is used."
)

print(
    "  - No loop over quantiles is required for computation."
)

print(
    "  - Per-quantile losses are correct."
)

print(
    "  - Mean total loss is correct."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 8.2 ---
# ============================================================
# TODO 8.2 — Implement quantile crossing metric
# ============================================================

import torch


def quantile_crossing_metric(
    predictions,
    tolerance=0.0,
):
    """
    Measure quantile-ordering violations.

    Expected prediction columns:
        [Q50, Q90, Q95, Q99]

    Violations:
        Q50 > Q90
        Q90 > Q95
        Q95 > Q99

    Parameters
    ----------
    predictions:
        FloatTensor [N, 4]

    tolerance:
        Optional numerical tolerance.

        A crossing is counted only when:

            lower_quantile > upper_quantile + tolerance

        For the architecturally ordered model, tolerance=0.0
        should already produce zero crossings.

    Returns
    -------
    metrics:
        Dictionary containing:
            num_records
            q50_q90_violations
            q90_q95_violations
            q95_q99_violations
            records_with_any_violation
            any_violation_fraction
    """


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not torch.is_tensor(
        predictions
    ):
        raise TypeError(
            "predictions must be a PyTorch tensor."
        )

    if predictions.ndim != 2:
        raise ValueError(
            "predictions must have shape [N, 4].\n"
            f"Found: {tuple(predictions.shape)}"
        )

    if predictions.shape[1] != 4:
        raise ValueError(
            "predictions must contain exactly four columns:\n"
            "[Q50, Q90, Q95, Q99]\n"
            f"Found shape: {tuple(predictions.shape)}"
        )

    if predictions.shape[0] == 0:
        raise ValueError(
            "At least one prediction record is required."
        )

    if not predictions.is_floating_point():
        raise TypeError(
            "predictions must be floating point."
        )

    if not torch.isfinite(
        predictions
    ).all():
        raise ValueError(
            "predictions contains NaN or Inf."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )


    # ========================================================
    # Extract quantiles
    # ========================================================

    q50 = predictions[:, 0]
    q90 = predictions[:, 1]
    q95 = predictions[:, 2]
    q99 = predictions[:, 3]


    # ========================================================
    # Pairwise crossing masks
    # ========================================================

    q50_q90_violation = (
        q50 > q90 + tolerance
    )

    q90_q95_violation = (
        q90 > q95 + tolerance
    )

    q95_q99_violation = (
        q95 > q99 + tolerance
    )


    # --------------------------------------------------------
    # Any violation within a record
    # --------------------------------------------------------

    any_violation = (
        q50_q90_violation
        |
        q90_q95_violation
        |
        q95_q99_violation
    )


    # ========================================================
    # Aggregate metrics
    # ========================================================

    num_records = int(
        predictions.shape[0]
    )

    q50_q90_count = int(
        q50_q90_violation
        .sum()
        .item()
    )

    q90_q95_count = int(
        q90_q95_violation
        .sum()
        .item()
    )

    q95_q99_count = int(
        q95_q99_violation
        .sum()
        .item()
    )

    records_with_any_violation = int(
        any_violation
        .sum()
        .item()
    )

    any_violation_fraction = (
        records_with_any_violation
        / num_records
    )


    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    metrics = {
        "num_records": (
            num_records
        ),

        "q50_q90_violations": (
            q50_q90_count
        ),

        "q90_q95_violations": (
            q90_q95_count
        ),

        "q95_q99_violations": (
            q95_q99_count
        ),

        "records_with_any_violation": (
            records_with_any_violation
        ),

        "any_violation_fraction": (
            float(
                any_violation_fraction
            )
        ),
    }

    return metrics


# ============================================================
# Unit test 1 — Hand-built crossing example
# ============================================================

crossing_test_predictions = torch.tensor(
    [
        # No violation
        [1.0, 2.0, 3.0, 4.0],

        # Q50 > Q90
        [2.0, 1.0, 3.0, 4.0],

        # Q90 > Q95
        [1.0, 3.0, 2.0, 4.0],

        # Q95 > Q99
        [1.0, 2.0, 4.0, 3.0],

        # Multiple violations
        [4.0, 3.0, 2.0, 1.0],
    ],
    dtype=torch.float32,
)


crossing_test_metrics = (
    quantile_crossing_metric(
        crossing_test_predictions
    )
)


# ------------------------------------------------------------
# Expected counts
# ------------------------------------------------------------

expected_q50_q90 = 2
expected_q90_q95 = 2
expected_q95_q99 = 2

expected_records_with_any = 4

expected_fraction = (
    4 / 5
)


# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------

assert (
    crossing_test_metrics[
        "q50_q90_violations"
    ]
    == expected_q50_q90
)

assert (
    crossing_test_metrics[
        "q90_q95_violations"
    ]
    == expected_q90_q95
)

assert (
    crossing_test_metrics[
        "q95_q99_violations"
    ]
    == expected_q95_q99
)

assert (
    crossing_test_metrics[
        "records_with_any_violation"
    ]
    == expected_records_with_any
)

assert abs(
    crossing_test_metrics[
        "any_violation_fraction"
    ]
    - expected_fraction
) < 1e-12


# ============================================================
# Unit test 2 — Ordered quantile-head predictions
# ============================================================

architectural_metrics = (
    quantile_crossing_metric(
        sample_predictions
    )
)


if (
    architectural_metrics[
        "records_with_any_violation"
    ]
    != 0
):
    raise RuntimeError(
        "Architecturally ordered quantile head "
        "produced quantile crossing."
    )


# ============================================================
# Optional numerical-tolerance check
# ============================================================

TOLERANCE = 1e-7

architectural_metrics_with_tolerance = (
    quantile_crossing_metric(
        sample_predictions,
        tolerance=TOLERANCE,
    )
)


if (
    architectural_metrics_with_tolerance[
        "records_with_any_violation"
    ]
    != 0
):
    raise RuntimeError(
        "Quantile crossing remained after "
        "numerical tolerance."
    )


# ============================================================
# Final report
# ============================================================

print("=" * 70)
print("Quantile Crossing Metric Verification")
print("=" * 70)


print("\nHand-built unit test:")

print(
    f"  Records                : "
    f"{crossing_test_metrics['num_records']}"
)

print(
    f"  Q50 > Q90 violations   : "
    f"{crossing_test_metrics['q50_q90_violations']}"
)

print(
    f"  Q90 > Q95 violations   : "
    f"{crossing_test_metrics['q90_q95_violations']}"
)

print(
    f"  Q95 > Q99 violations   : "
    f"{crossing_test_metrics['q95_q99_violations']}"
)

print(
    f"  Records with crossing  : "
    f"{crossing_test_metrics['records_with_any_violation']}"
)

print(
    f"  Crossing fraction      : "
    f"{crossing_test_metrics['any_violation_fraction']:.4f}"
)


print("\nArchitectural ordering test:")

print(
    f"  Records                : "
    f"{architectural_metrics['num_records']}"
)

print(
    f"  Q50 > Q90 violations   : "
    f"{architectural_metrics['q50_q90_violations']}"
)

print(
    f"  Q90 > Q95 violations   : "
    f"{architectural_metrics['q90_q95_violations']}"
)

print(
    f"  Q95 > Q99 violations   : "
    f"{architectural_metrics['q95_q99_violations']}"
)

print(
    f"  Records with crossing  : "
    f"{architectural_metrics['records_with_any_violation']}"
)

print(
    f"  Crossing fraction      : "
    f"{architectural_metrics['any_violation_fraction']:.8f}"
)


print("\nTolerance check:")

print(
    f"  Numerical tolerance    : "
    f"{TOLERANCE}"
)

print(
    f"  Crossing fraction      : "
    f"{architectural_metrics_with_tolerance['any_violation_fraction']:.8f}"
)


print("\nValidation:")

print(
    "  - All three adjacent quantile pairs are checked."
)

print(
    "  - A record is counted once if any crossing occurs."
)

print(
    "  - Pairwise violation counts are also reported."
)

print(
    "  - Architectural quantile ordering produced zero crossings."
)


print("\nSTATUS: PASSED")

print("=" * 70)

# --- from TODO 9.3 ---
# ============================================================
# TODO 9.3 — Implement validation epoch
# ============================================================

import torch


# ------------------------------------------------------------
# Validate for one complete epoch
# ------------------------------------------------------------

@torch.no_grad()
def validate_one_epoch(
    model,
    validation_dataset,
    device,
    quantiles,
):
    """
    Evaluate the model on the validation split.

    No gradients are computed.
    No optimizer updates are performed.

    For each DAG:
        1. Move tensors to the device.
        2. Predict all execution contexts.
        3. Compute vectorized pinball loss.
        4. Compute quantile crossing metrics.

    The primary model-selection metric is:

        validation mean pinball loss

    Returns
    -------
    metrics : dict
        {
            "loss",
            "model_selection_metric",
            "per_quantile_loss",
            "q50_loss",
            "q90_loss",
            "q95_loss",
            "q99_loss",
            "crossing_fraction",
            "records_with_crossing",
            "num_records",
            "num_graphs",
        }
    """


    # ========================================================
    # Validate inputs
    # ========================================================

    if len(validation_dataset) == 0:
        raise ValueError(
            "validation_dataset is empty."
        )

    if len(quantiles) != 4:
        raise ValueError(
            "This Phase 1 validation loop expects exactly "
            "four quantiles."
        )


    # ========================================================
    # Evaluation mode
    #
    # This disables training behavior such as dropout.
    # ========================================================

    model.eval()


    # ========================================================
    # Epoch accumulators
    # ========================================================

    num_graphs = len(
        validation_dataset
    )

    total_loss_sum = 0.0

    per_quantile_loss_sum = torch.zeros(
        4,
        dtype=torch.float64,
    )

    total_records = 0

    total_records_with_crossing = 0

    total_q50_q90_violations = 0
    total_q90_q95_violations = 0
    total_q95_q99_violations = 0


    # ========================================================
    # Iterate over validation DAGs
    # ========================================================

    for dataset_index in range(
        num_graphs
    ):

        item = validation_dataset[
            dataset_index
        ]


        # ====================================================
        # 1. Move tensors to device
        # ====================================================

        x = (
            item["x"]
            .to(
                device,
                non_blocking=True,
            )
        )

        edge_index = (
            item["edge_index"]
            .to(
                device,
                non_blocking=True,
            )
        )

        edge_attr = (
            item["edge_attr"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target_node_indices = (
            item[
                "target_node_indices"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        core_type_indices = (
            item[
                "core_type_indices"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        context_numeric = (
            item[
                "context_numeric"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        targets = (
            item["y"]
            .to(
                device,
                non_blocking=True,
            )
        )


        # ----------------------------------------------------
        # Number of execution records for this DAG
        # ----------------------------------------------------

        num_contexts = int(
            targets.shape[0]
        )

        if num_contexts <= 0:
            raise RuntimeError(
                f"Validation graph {item['graph_id']} "
                "has no execution contexts."
            )


        # ----------------------------------------------------
        # Verify scenario alignment
        # ----------------------------------------------------

        if (
            target_node_indices.shape[0]
            != num_contexts
        ):
            raise RuntimeError(
                "target_node_indices and targets are not aligned.\n"
                f"Graph: {item['graph_id']}"
            )

        if (
            core_type_indices.shape[0]
            != num_contexts
        ):
            raise RuntimeError(
                "core_type_indices and targets are not aligned.\n"
                f"Graph: {item['graph_id']}"
            )

        if (
            context_numeric.shape[0]
            != num_contexts
        ):
            raise RuntimeError(
                "context_numeric and targets are not aligned.\n"
                f"Graph: {item['graph_id']}"
            )


        # ====================================================
        # 2. Forward all contexts
        #
        # No optimizer.zero_grad()
        # No backward()
        # No optimizer.step()
        # ====================================================

        predictions = model(

            x=x,

            edge_index=edge_index,

            edge_attr=edge_attr,

            target_node_indices=(
                target_node_indices
            ),

            core_type_indices=(
                core_type_indices
            ),

            context_numeric=(
                context_numeric
            ),
        )


        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        expected_prediction_shape = (
            num_contexts,
            4,
        )

        if tuple(
            predictions.shape
        ) != expected_prediction_shape:

            raise RuntimeError(
                "Validation prediction shape mismatch.\n"
                f"Graph:    {item['graph_id']}\n"
                f"Expected: {expected_prediction_shape}\n"
                f"Found:    {tuple(predictions.shape)}"
            )

        if not torch.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "Non-finite validation predictions detected.\n"
                f"Graph: {item['graph_id']}"
            )


        # ====================================================
        # 3. Compute pinball loss
        # ====================================================

        (
            loss,
            per_quantile_loss,
        ) = pinball_loss(

            predictions=predictions,

            targets=targets,

            quantiles=quantiles,

            return_per_quantile=True,
        )


        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite validation loss detected.\n"
                f"Graph: {item['graph_id']}"
            )


        # ====================================================
        # 4. Quantile crossing metric
        # ====================================================

        crossing_metrics = (
            quantile_crossing_metric(
                predictions
            )
        )


        # ====================================================
        # Aggregate validation metrics
        # ====================================================

        # loss is the mean loss for this DAG.
        # Multiply by its number of records so that the final
        # result is the mean over all validation records.
        #
        # In our dataset every DAG has exactly 42 records,
        # so this is also equivalent to averaging DAG losses.

        total_loss_sum += (
            float(
                loss.item()
            )
            * num_contexts
        )


        per_quantile_loss_sum += (
            per_quantile_loss
            .cpu()
            .to(torch.float64)
            * num_contexts
        )


        total_records += (
            num_contexts
        )


        total_records_with_crossing += (
            crossing_metrics[
                "records_with_any_violation"
            ]
        )


        total_q50_q90_violations += (
            crossing_metrics[
                "q50_q90_violations"
            ]
        )

        total_q90_q95_violations += (
            crossing_metrics[
                "q90_q95_violations"
            ]
        )

        total_q95_q99_violations += (
            crossing_metrics[
                "q95_q99_violations"
            ]
        )


    # ========================================================
    # Compute validation epoch metrics
    # ========================================================

    if total_records <= 0:
        raise RuntimeError(
            "No validation records were processed."
        )


    # --------------------------------------------------------
    # PRIMARY MODEL-SELECTION METRIC
    # --------------------------------------------------------

    validation_mean_pinball_loss = (
        total_loss_sum
        / total_records
    )


    mean_per_quantile_loss = (
        per_quantile_loss_sum
        / total_records
    )


    crossing_fraction = (
        total_records_with_crossing
        / total_records
    )


    # ========================================================
    # Return metrics
    # ========================================================

    metrics = {

        # Primary validation objective
        "loss": float(
            validation_mean_pinball_loss
        ),

        "model_selection_metric": float(
            validation_mean_pinball_loss
        ),

        "model_selection_metric_name": (
            "validation_mean_pinball_loss"
        ),


        # Per-quantile pinball losses
        "per_quantile_loss": [
            float(value)
            for value
            in mean_per_quantile_loss.tolist()
        ],

        "q50_loss": float(
            mean_per_quantile_loss[0]
        ),

        "q90_loss": float(
            mean_per_quantile_loss[1]
        ),

        "q95_loss": float(
            mean_per_quantile_loss[2]
        ),

        "q99_loss": float(
            mean_per_quantile_loss[3]
        ),


        # Quantile crossing diagnostics
        "crossing_fraction": float(
            crossing_fraction
        ),

        "records_with_crossing": int(
            total_records_with_crossing
        ),

        "q50_q90_violations": int(
            total_q50_q90_violations
        ),

        "q90_q95_violations": int(
            total_q90_q95_violations
        ),

        "q95_q99_violations": int(
            total_q95_q99_violations
        ),


        # Dataset accounting
        "num_records": int(
            total_records
        ),

        "num_graphs": int(
            num_graphs
        ),
    }


    return metrics


# ============================================================
# Validate the expected validation dataset contract
# ============================================================

EXPECTED_CONTEXTS_PER_DAG = 42

expected_validation_records = (
    len(validation_dataset)
    * EXPECTED_CONTEXTS_PER_DAG
)


verification_item = (
    validation_dataset[0]
)

actual_contexts_in_sample = len(
    verification_item["y"]
)

if (
    actual_contexts_in_sample
    != EXPECTED_CONTEXTS_PER_DAG
):
    raise RuntimeError(
        "Unexpected number of validation contexts per DAG.\n"
        f"Expected: {EXPECTED_CONTEXTS_PER_DAG}\n"
        f"Found:    {actual_contexts_in_sample}"
    )


# ============================================================
# Final report
# ============================================================

print("=" * 80)
print("Validation Epoch Function")
print("=" * 80)


print("\nFunction created successfully:")

print(
    "  validate_one_epoch(...)"
)


print("\nValidation logic:")

print(
    "  1. model.eval()"
)

print(
    "  2. torch.no_grad()"
)

print(
    "  3. One validation DAG is loaded."
)

print(
    "  4. DAG/scenario tensors are moved to DEVICE."
)

print(
    "  5. All contexts are predicted in one forward pass."
)

print(
    "  6. Vectorized pinball loss is computed."
)

print(
    "  7. Quantile crossing metrics are computed."
)

print(
    "  8. No backward pass or optimizer update is performed."
)


print("\nPrimary model-selection metric:")

print(
    "  validation mean pinball loss"
)


print("\nValidation aggregation:")

print(
    "  - Mean pinball loss"
)

print(
    "  - Q50/Q90/Q95/Q99 pinball losses"
)

print(
    "  - Quantile crossing fraction"
)


print("\nExpected validation size:")

print(
    f"  DAGs                   : "
    f"{len(validation_dataset)}"
)

print(
    f"  Contexts per DAG       : "
    f"{EXPECTED_CONTEXTS_PER_DAG}"
)

print(
    f"  Records                : "
    f"{expected_validation_records:,}"
)


print("\nOptimizer behavior:")

print(
    "  - No gradients are computed."
)

print(
    "  - No optimizer step occurs."
)

print(
    "  - Model parameters are not modified."
)


print("\nSTATUS: VALIDATION EPOCH FUNCTION READY")

print("=" * 80)