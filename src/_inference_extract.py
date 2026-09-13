# ============================================================
# TODO 9.1 — Instantiate model and training objects
# ============================================================

import random
import numpy as np
import torch


# ============================================================
# 1. Reset random seeds before final model initialization
# ============================================================

TRAINING_SEED = phase1_config.seed

random.seed(TRAINING_SEED)
np.random.seed(TRAINING_SEED)
torch.manual_seed(TRAINING_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        TRAINING_SEED
    )


# ============================================================
# 2. Instantiate the FINAL Phase 1 model
# ============================================================

model = Phase1QuantilePredictor(

    config=phase1_config,

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

).to(DEVICE)


# ============================================================
# 3. Instantiate optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=(
        phase1_config.learning_rate
    ),

    weight_decay=(
        phase1_config.weight_decay
    ),
)


# ============================================================
# 4. Optional learning-rate scheduler
# ============================================================

USE_LR_SCHEDULER = True

LR_SCHEDULER_FACTOR = 0.5

LR_SCHEDULER_PATIENCE = 8

LR_SCHEDULER_MIN_LR = 1e-6


if USE_LR_SCHEDULER:

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=(
                LR_SCHEDULER_FACTOR
            ),

            patience=(
                LR_SCHEDULER_PATIENCE
            ),

            min_lr=(
                LR_SCHEDULER_MIN_LR
            ),
        )
    )

else:

    scheduler = None


# ============================================================
# 5. Early stopping state
# ============================================================

early_stopping_state = {

    "best_val_loss": float("inf"),

    "best_epoch": None,

    "epochs_without_improvement": 0,

    "patience": (
        phase1_config.early_stopping_patience
    ),

    "should_stop": False,
}


# ============================================================
# 6. Training history container
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "validation_loss": [],

    "train_q50_loss": [],
    "train_q90_loss": [],
    "train_q95_loss": [],
    "train_q99_loss": [],

    "validation_q50_loss": [],
    "validation_q90_loss": [],
    "validation_q95_loss": [],
    "validation_q99_loss": [],

    "train_crossing_fraction": [],

    "validation_crossing_fraction": [],

    "learning_rate": [],

    "gradient_norm": [],
}


# ============================================================
# 7. Checkpoint paths
# ============================================================

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "best_model.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "last_model.pt"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 8. Count model parameters
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
    if parameter.requires_grad
)

non_trainable_parameters = (
    total_parameters
    - trainable_parameters
)


# ------------------------------------------------------------
# Parameter count by major component
# ------------------------------------------------------------

dag_encoder_parameters = sum(
    parameter.numel()
    for parameter
    in model.dag_encoder.parameters()
    if parameter.requires_grad
)

context_fusion_parameters = sum(
    parameter.numel()
    for parameter
    in model.context_fusion.parameters()
    if parameter.requires_grad
)

quantile_head_parameters = sum(
    parameter.numel()
    for parameter
    in model.quantile_head.parameters()
    if parameter.requires_grad
)


# ============================================================
# 9. Basic training-object validation
# ============================================================

if trainable_parameters <= 0:
    raise RuntimeError(
        "The model has no trainable parameters."
    )


if (
    optimizer.param_groups[0]["lr"]
    != phase1_config.learning_rate
):
    raise RuntimeError(
        "Optimizer learning rate does not match "
        "Phase1Config."
    )


if (
    optimizer.param_groups[0]["weight_decay"]
    != phase1_config.weight_decay
):
    raise RuntimeError(
        "Optimizer weight decay does not match "
        "Phase1Config."
    )


if (
    early_stopping_state["patience"]
    != phase1_config.early_stopping_patience
):
    raise RuntimeError(
        "Early-stopping patience does not match "
        "Phase1Config."
    )


# ============================================================
# 10. Save scheduler/training-object settings before training
# ============================================================

training_object_config = {

    "optimizer": "AdamW",

    "learning_rate": (
        phase1_config.learning_rate
    ),

    "weight_decay": (
        phase1_config.weight_decay
    ),

    "use_lr_scheduler": (
        USE_LR_SCHEDULER
    ),

    "lr_scheduler": (
        "ReduceLROnPlateau"
        if USE_LR_SCHEDULER
        else None
    ),

    "lr_scheduler_factor": (
        LR_SCHEDULER_FACTOR
        if USE_LR_SCHEDULER
        else None
    ),

    "lr_scheduler_patience": (
        LR_SCHEDULER_PATIENCE
        if USE_LR_SCHEDULER
        else None
    ),

    "lr_scheduler_min_lr": (
        LR_SCHEDULER_MIN_LR
        if USE_LR_SCHEDULER
        else None
    ),

    "early_stopping_patience": (
        phase1_config.early_stopping_patience
    ),

    "gradient_clip_norm": (
        phase1_config.gradient_clip_norm
    ),

    "max_epochs": (
        phase1_config.max_epochs
    ),

    "seed": (
        TRAINING_SEED
    ),
}


TRAINING_OBJECT_CONFIG_PATH = (
    CONFIG_DIR
    / "training_objects.json"
)


with TRAINING_OBJECT_CONFIG_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        training_object_config,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 11. Final report
# ============================================================

print("=" * 80)
print("Phase 1 — Model and Training Objects")
print("=" * 80)


print("\nMODEL ARCHITECTURE")
print("-" * 80)

print(model)


print("\n" + "=" * 80)
print("PARAMETER COUNTS")
print("=" * 80)

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
    f"  Total parameters       : "
    f"{total_parameters:,}"
)

print(
    f"  Trainable parameters   : "
    f"{trainable_parameters:,}"
)

print(
    f"  Non-trainable          : "
    f"{non_trainable_parameters:,}"
)


print("\n" + "=" * 80)
print("MODEL HYPERPARAMETERS")
print("=" * 80)

print(
    f"  Quantiles              : "
    f"{phase1_config.quantiles}"
)

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

print(
    f"  Head hidden dim        : "
    f"{phase1_config.head_hidden_dim}"
)

print(
    f"  Dropout                : "
    f"{phase1_config.dropout}"
)


print("\n" + "=" * 80)
print("OPTIMIZATION")
print("=" * 80)

print(
    "  Optimizer              : AdamW"
)

print(
    f"  Learning rate          : "
    f"{phase1_config.learning_rate}"
)

print(
    f"  Weight decay           : "
    f"{phase1_config.weight_decay}"
)

print(
    f"  Max epochs             : "
    f"{phase1_config.max_epochs}"
)

print(
    f"  Gradient clip norm     : "
    f"{phase1_config.gradient_clip_norm}"
)


print("\n" + "=" * 80)
print("LR SCHEDULER")
print("=" * 80)

print(
    f"  Enabled                : "
    f"{USE_LR_SCHEDULER}"
)

if USE_LR_SCHEDULER:

    print(
        "  Scheduler              : ReduceLROnPlateau"
    )

    print(
        f"  Reduction factor       : "
        f"{LR_SCHEDULER_FACTOR}"
    )

    print(
        f"  Scheduler patience     : "
        f"{LR_SCHEDULER_PATIENCE}"
    )

    print(
        f"  Minimum LR             : "
        f"{LR_SCHEDULER_MIN_LR}"
    )


print("\n" + "=" * 80)
print("EARLY STOPPING")
print("=" * 80)

print(
    f"  Patience               : "
    f"{early_stopping_state['patience']}"
)

print(
    f"  Initial best val loss  : "
    f"{early_stopping_state['best_val_loss']}"
)


print("\n" + "=" * 80)
print("RUNTIME")
print("=" * 80)

print(
    f"  Device                 : "
    f"{DEVICE}"
)

print(
    f"  Seed                   : "
    f"{TRAINING_SEED}"
)

print(
    f"  Best checkpoint        : "
    f"{BEST_MODEL_PATH}"
)

print(
    f"  Last checkpoint        : "
    f"{LAST_MODEL_PATH}"
)

print(
    f"  Training-object config : "
    f"{TRAINING_OBJECT_CONFIG_PATH}"
)


print("\nSTATUS: TRAINING OBJECTS READY")

print("=" * 80)

# ============================================================
# TODO 9.2 — Implement one training epoch
# ============================================================

import math
import torch


# ------------------------------------------------------------
# Helper: compute global gradient norm when clipping is disabled
# ------------------------------------------------------------

def compute_global_gradient_norm(model):
    """
    Compute the global L2 norm of all available parameter gradients.
    """

    squared_norm_sum = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        grad_norm = (
            parameter.grad
            .detach()
            .norm(2)
        )

        squared_norm_sum += (
            grad_norm.item() ** 2
        )

    return math.sqrt(
        squared_norm_sum
    )


# ------------------------------------------------------------
# Train for one complete epoch
# ------------------------------------------------------------

def train_one_epoch(
    model,
    train_dataset,
    optimizer,
    device,
    quantiles,
    gradient_clip_norm=None,
    shuffle=True,
):
    """
    Train the model for one epoch.

    One optimizer update is performed per DAG.

    For each DAG:
        1. Move graph/scenario tensors to the device.
        2. Zero gradients.
        3. Encode the DAG once and predict all contexts.
        4. Compute vectorized pinball loss.
        5. Backpropagate.
        6. Optionally clip gradients.
        7. Update model parameters.

    Epoch metrics are aggregated over all execution records.

    Returns
    -------
    metrics : dict
        {
            "loss",
            "per_quantile_loss",
            "q50_loss",
            "q90_loss",
            "q95_loss",
            "q99_loss",
            "crossing_fraction",
            "records_with_crossing",
            "num_records",
            "num_graphs",
            "mean_gradient_norm",
            "max_gradient_norm",
        }
    """


    # ========================================================
    # Validate inputs
    # ========================================================

    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset is empty."
        )

    if len(quantiles) != 4:
        raise ValueError(
            "This Phase 1 training loop expects exactly "
            "four quantiles."
        )

    if (
        gradient_clip_norm is not None
        and gradient_clip_norm <= 0
    ):
        raise ValueError(
            "gradient_clip_norm must be positive or None."
        )


    # ========================================================
    # Training mode
    # ========================================================

    model.train()


    # ========================================================
    # Determine graph order
    # ========================================================

    num_graphs = len(
        train_dataset
    )

    if shuffle:

        graph_order = (
            torch.randperm(
                num_graphs
            )
            .tolist()
        )

    else:

        graph_order = list(
            range(num_graphs)
        )


    # ========================================================
    # Epoch accumulators
    # ========================================================

    total_loss_sum = 0.0

    per_quantile_loss_sum = torch.zeros(
        4,
        dtype=torch.float64,
    )

    total_records = 0

    total_records_with_crossing = 0

    gradient_norm_sum = 0.0

    maximum_gradient_norm = 0.0


    # ========================================================
    # Iterate over TRAIN DAGs
    # ========================================================

    for dataset_index in graph_order:

        item = train_dataset[
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
        # Scenario count for this DAG
        # ----------------------------------------------------

        num_contexts = int(
            targets.shape[0]
        )

        if num_contexts <= 0:
            raise RuntimeError(
                f"Graph {item['graph_id']} has no "
                "execution contexts."
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
        # 2. Zero gradients
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # 3. Forward all contexts
        #
        # The DAG encoder runs once.
        #
        # Output:
        #     [num_contexts, 4]
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
        # Validate prediction/target alignment
        # ----------------------------------------------------

        expected_prediction_shape = (
            num_contexts,
            4,
        )

        if tuple(
            predictions.shape
        ) != expected_prediction_shape:

            raise RuntimeError(
                "Training prediction shape mismatch.\n"
                f"Graph:    {item['graph_id']}\n"
                f"Expected: {expected_prediction_shape}\n"
                f"Found:    {tuple(predictions.shape)}"
            )


        if targets.ndim != 1:
            raise RuntimeError(
                "Training targets must have shape "
                "[num_contexts]."
            )


        # ====================================================
        # 4. Compute pinball loss
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


        # ----------------------------------------------------
        # Validate loss before backward
        # ----------------------------------------------------

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss detected.\n"
                f"Graph: {item['graph_id']}"
            )


        # ====================================================
        # Crossing metric before optimizer update
        # ====================================================

        crossing_metrics = (
            quantile_crossing_metric(
                predictions.detach()
            )
        )


        # ====================================================
        # 5. Backward
        # ====================================================

        loss.backward()


        # ====================================================
        # 6. Optional gradient clipping
        # ====================================================

        if gradient_clip_norm is not None:

            # clip_grad_norm_ returns the total gradient norm
            # before clipping.
            gradient_norm_tensor = (
                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=(
                        gradient_clip_norm
                    ),

                    error_if_nonfinite=True,
                )
            )

            gradient_norm = float(
                gradient_norm_tensor.item()
            )

        else:

            gradient_norm = (
                compute_global_gradient_norm(
                    model
                )
            )

            if not math.isfinite(
                gradient_norm
            ):
                raise RuntimeError(
                    "Non-finite gradient norm detected.\n"
                    f"Graph: {item['graph_id']}"
                )


        # ====================================================
        # 7. Optimizer step
        # ====================================================

        optimizer.step()


        # ====================================================
        # Aggregate epoch metrics
        # ====================================================

        # Weight metrics by the number of runtime records.
        # This remains correct even if context counts differ
        # between DAGs in a future dataset.

        total_loss_sum += (
            float(
                loss.detach().item()
            )
            * num_contexts
        )


        per_quantile_loss_sum += (
            per_quantile_loss
            .detach()
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


        gradient_norm_sum += (
            gradient_norm
        )


        maximum_gradient_norm = max(
            maximum_gradient_norm,
            gradient_norm,
        )


    # ========================================================
    # Epoch-level metrics
    # ========================================================

    if total_records <= 0:
        raise RuntimeError(
            "No training records were processed."
        )


    mean_total_loss = (
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


    mean_gradient_norm = (
        gradient_norm_sum
        / num_graphs
    )


    # ========================================================
    # Final epoch result
    # ========================================================

    metrics = {

        "loss": float(
            mean_total_loss
        ),

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

        "crossing_fraction": float(
            crossing_fraction
        ),

        "records_with_crossing": int(
            total_records_with_crossing
        ),

        "num_records": int(
            total_records
        ),

        "num_graphs": int(
            num_graphs
        ),

        "mean_gradient_norm": float(
            mean_gradient_norm
        ),

        "max_gradient_norm": float(
            maximum_gradient_norm
        ),
    }


    return metrics


# ============================================================
# Function verification
# ============================================================

EXPECTED_CONTEXTS_PER_DAG = 42

expected_records_per_epoch = (
    len(train_dataset)
    * EXPECTED_CONTEXTS_PER_DAG
)


# ------------------------------------------------------------
# Verify the dataset contract using one actual DAG
# ------------------------------------------------------------

verification_item = (
    train_dataset[0]
)

actual_contexts_in_sample = len(
    verification_item["y"]
)

if (
    actual_contexts_in_sample
    != EXPECTED_CONTEXTS_PER_DAG
):
    raise RuntimeError(
        "Unexpected number of runtime contexts per DAG.\n"
        f"Expected: {EXPECTED_CONTEXTS_PER_DAG}\n"
        f"Found:    {actual_contexts_in_sample}"
    )


# ============================================================
# Final report
# ============================================================

print("=" * 80)
print("Training Epoch Function")
print("=" * 80)


print(
    "\nFunction created successfully:"
)

print(
    "  train_one_epoch(...)"
)


print(
    "\nTraining logic:"
)

print(
    "  1. One DAG is loaded."
)

print(
    "  2. DAG/scenario tensors are moved to DEVICE."
)

print(
    "  3. Gradients are cleared."
)

print(
    "  4. All contexts are predicted in one forward pass."
)

print(
    "  5. Vectorized pinball loss is computed."
)

print(
    "  6. Backpropagation is performed."
)

print(
    "  7. Gradients are optionally clipped."
)

print(
    "  8. AdamW performs one optimizer step."
)


print(
    "\nEpoch aggregation:"
)

print(
    "  - Total pinball loss"
)

print(
    "  - Q50/Q90/Q95/Q99 pinball losses"
)

print(
    "  - Quantile crossing fraction"
)

print(
    "  - Mean/max gradient norm"
)


print(
    "\nGradient clipping:"
)

print(
    f"  Configured max norm    : "
    f"{phase1_config.gradient_clip_norm}"
)


print(
    "\nExpected training size:"
)

print(
    f"  DAGs per epoch         : "
    f"{len(train_dataset)}"
)

print(
    f"  Contexts per DAG       : "
    f"{EXPECTED_CONTEXTS_PER_DAG}"
)

print(
    f"  Records per epoch      : "
    f"{expected_records_per_epoch:,}"
)


print(
    "\nSTATUS: TRAINING EPOCH FUNCTION READY"
)

print("=" * 80)

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

# ============================================================
# TODO 11.1 — Implement inference for one graph
# ============================================================

import numpy as np
import pandas as pd
import torch
from IPython.display import display


# ============================================================
# 1. Small utility: convert a saved scalar to float
# ============================================================

def _to_scalar_float(value):
    """
    Convert a scalar or single-element container to float.
    """

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


# ============================================================
# 2. Find target mean/std inside preprocessing_state
# ============================================================

def _get_target_standardization_stats(
    preprocessing_state,
):
    """
    Locate the TRAIN-fitted target mean and standard deviation
    inside preprocessing_state.

    The saved state may contain nested structures such as:

        target -> stats -> mean/std

    or:

        target_transform -> scaler -> mean/scale

    Therefore we search recursively instead of assuming one
    exact nesting structure.

    Returns
    -------
    target_mean_us : float
    target_std_us : float
    source_path : str
        Location inside preprocessing_state where the
        statistics were found.
    """

    if not isinstance(
        preprocessing_state,
        dict,
    ):
        raise TypeError(
            "preprocessing_state must be a dictionary."
        )


    # --------------------------------------------------------
    # Possible names used for the center and scale
    # --------------------------------------------------------

    mean_keys = {
        "mean",
        "mean_",
        "train_mean",
        "mean_us",
        "mu",
        "center",
    }

    std_keys = {
        "std",
        "std_",
        "train_std",
        "std_us",
        "sigma",
        "scale",
        "scale_",
    }


    candidates = []


    # --------------------------------------------------------
    # Extract a mean/std pair from one dictionary
    # --------------------------------------------------------

    def inspect_mapping(
        mapping,
        path,
    ):
        found_mean = None
        found_std = None
        found_mean_key = None
        found_std_key = None


        for key, value in mapping.items():

            key_lower = str(
                key
            ).lower()

            if (
                key_lower in mean_keys
                and found_mean is None
            ):

                scalar_value = (
                    _to_scalar_float(
                        value
                    )
                )

                if scalar_value is not None:
                    found_mean = (
                        scalar_value
                    )
                    found_mean_key = (
                        str(key)
                    )


            if (
                key_lower in std_keys
                and found_std is None
            ):

                scalar_value = (
                    _to_scalar_float(
                        value
                    )
                )

                if scalar_value is not None:
                    found_std = (
                        scalar_value
                    )
                    found_std_key = (
                        str(key)
                    )


        if (
            found_mean is None
            or found_std is None
        ):
            return


        if found_std <= 0:
            return


        path_text = ".".join(
            path
        )

        path_lower = (
            path_text.lower()
        )


        # ----------------------------------------------------
        # Rank candidates.
        #
        # Target-related paths receive much higher scores than
        # node/edge/context preprocessing statistics.
        # ----------------------------------------------------

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
            {
                "mean": found_mean,
                "std": found_std,
                "path": path_text,
                "mean_key": found_mean_key,
                "std_key": found_std_key,
                "score": score,
            }
        )


    # --------------------------------------------------------
    # Recursive traversal
    # --------------------------------------------------------

    def walk(
        obj,
        path=(),
    ):

        if isinstance(
            obj,
            dict,
        ):

            inspect_mapping(
                obj,
                path,
            )

            for key, value in (
                obj.items()
            ):

                walk(
                    value,
                    path + (
                        str(key),
                    ),
                )


        elif isinstance(
            obj,
            (list, tuple),
        ):

            for index, value in enumerate(
                obj
            ):

                walk(
                    value,
                    path + (
                        f"[{index}]",
                    ),
                )


    walk(
        preprocessing_state
    )


    # --------------------------------------------------------
    # Keep only target-related candidates
    # --------------------------------------------------------

    target_candidates = [

        candidate

        for candidate
        in candidates

        if candidate[
            "score"
        ] > 0
    ]


    if not target_candidates:

        # Helpful diagnostic if the saved structure changes.
        available_top_level_keys = list(
            preprocessing_state.keys()
        )

        raise KeyError(
            "Could not locate TRAIN-fitted target mean/std "
            "inside preprocessing_state.\n"
            f"Top-level keys: {available_top_level_keys}"
        )


    # --------------------------------------------------------
    # Select strongest target-related candidate
    # --------------------------------------------------------

    target_candidates.sort(
        key=lambda candidate: (
            candidate["score"],
            len(candidate["path"]),
        ),
        reverse=True,
    )


    selected = (
        target_candidates[0]
    )


    return (
        float(
            selected["mean"]
        ),
        float(
            selected["std"]
        ),
        selected["path"],
    )


# ============================================================
# 3. Inverse-transform target values
# ============================================================

def inverse_transform_target(
    values_standardized,
    preprocessing_state,
):
    """
    Convert standardized predictions back to microseconds.

        y_std = (y_us - mean_train) / std_train

    Therefore:

        y_us = y_std * std_train + mean_train
    """

    (
        target_mean_us,
        target_std_us,
        _,
    ) = _get_target_standardization_stats(
        preprocessing_state
    )


    if torch.is_tensor(
        values_standardized
    ):

        return (
            values_standardized
            * target_std_us
            + target_mean_us
        )


    values_standardized = (
        np.asarray(
            values_standardized,
            dtype=np.float64,
        )
    )


    return (
        values_standardized
        * target_std_us
        + target_mean_us
    )


# ============================================================
# 4. Recover original target-node IDs
# ============================================================

def _recover_target_node_ids(
    graph_item,
):
    """
    Convert local tensor node indices back to original node IDs.
    """

    if (
        "node_id_to_index"
        not in graph_item
    ):
        raise KeyError(
            "graph_item does not contain node_id_to_index."
        )


    inverse_mapping = {

        int(local_index):
            original_node_id

        for (
            original_node_id,
            local_index
        ) in graph_item[
            "node_id_to_index"
        ].items()
    }


    local_target_indices = (

        graph_item[
            "target_node_indices"
        ]
        .detach()
        .cpu()
        .tolist()
    )


    target_node_ids = []


    for local_index in (
        local_target_indices
    ):

        local_index = int(
            local_index
        )

        if (
            local_index
            not in inverse_mapping
        ):
            raise RuntimeError(
                "Could not recover original target-node ID "
                f"for local index {local_index}."
            )


        target_node_ids.append(
            inverse_mapping[
                local_index
            ]
        )


    return target_node_ids


# ============================================================
# 5. Inference for ONE graph
# ============================================================

@torch.no_grad()
def infer_one_graph(
    model,
    graph_item,
    preprocessing_state,
    device,
):
    """
    Run the BEST Phase 1 predictor on one DAG.

    One row is returned for every execution context.

    Predictions are inverse-transformed from standardized
    target space back to microseconds.
    """

    model.eval()


    # ========================================================
    # Required graph fields
    # ========================================================

    required_keys = {
        "graph_id",
        "x",
        "edge_index",
        "edge_attr",
        "target_node_indices",
        "core_type_indices",
        "context_numeric",
        "y",
        "y_raw_us",
        "metadata",
        "node_id_to_index",
    }


    missing_keys = (
        required_keys
        - set(
            graph_item.keys()
        )
    )


    if missing_keys:
        raise KeyError(
            "Graph item is missing required fields:\n"
            f"{sorted(missing_keys)}"
        )


    # ========================================================
    # Read target scaling statistics
    # ========================================================

    (
        target_mean_us,
        target_std_us,
        scaler_source_path,
    ) = _get_target_standardization_stats(
        preprocessing_state
    )


    # ========================================================
    # IMPORTANT VALIDATION
    #
    # Check that the recovered mean/std reproduce the same
    # standardized targets already stored in the Dataset.
    #
    # This protects us against accidentally selecting scaler
    # statistics belonging to another feature.
    # ========================================================

    raw_targets_us = (

        graph_item[
            "y_raw_us"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    stored_standardized_targets = (

        graph_item[
            "y"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    reconstructed_standardized_targets = (

        (
            raw_targets_us
            - target_mean_us
        )
        / target_std_us
    )


    if not np.allclose(
        reconstructed_standardized_targets,
        stored_standardized_targets,
        rtol=1e-5,
        atol=1e-5,
    ):

        max_difference = float(
            np.max(
                np.abs(
                    reconstructed_standardized_targets
                    - stored_standardized_targets
                )
            )
        )

        raise RuntimeError(
            "The discovered target mean/std do not reproduce "
            "the Dataset target transformation.\n"
            f"Scaler path: {scaler_source_path}\n"
            f"Mean: {target_mean_us}\n"
            f"Std: {target_std_us}\n"
            f"Maximum standardized-target difference: "
            f"{max_difference}"
        )


    # ========================================================
    # Move model inputs to device
    # ========================================================

    x = (
        graph_item["x"]
        .to(
            device,
            non_blocking=True,
        )
    )

    edge_index = (
        graph_item["edge_index"]
        .to(
            device,
            non_blocking=True,
        )
    )

    edge_attr = (
        graph_item["edge_attr"]
        .to(
            device,
            non_blocking=True,
        )
    )

    target_node_indices = (

        graph_item[
            "target_node_indices"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    core_type_indices = (

        graph_item[
            "core_type_indices"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    context_numeric = (

        graph_item[
            "context_numeric"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )


    # ========================================================
    # Forward pass
    #
    # Output:
    #     [number_of_contexts, 4]
    #
    # Columns:
    #     Q50, Q90, Q95, Q99
    # ========================================================

    predictions_standardized = model(

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


    num_contexts = int(
        target_node_indices.shape[0]
    )


    expected_shape = (
        num_contexts,
        4,
    )


    if tuple(
        predictions_standardized.shape
    ) != expected_shape:

        raise RuntimeError(
            "Inference prediction shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Found: "
            f"{tuple(predictions_standardized.shape)}"
        )


    if not torch.isfinite(
        predictions_standardized
    ).all():

        raise RuntimeError(
            "Inference produced NaN/Inf predictions."
        )


    # ========================================================
    # Inverse-transform predictions into microseconds
    # ========================================================

    predictions_us = (

        inverse_transform_target(

            predictions_standardized,

            preprocessing_state,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    if not np.isfinite(
        predictions_us
    ).all():

        raise RuntimeError(
            "Inverse-transformed predictions contain "
            "NaN/Inf values."
        )


    # ========================================================
    # Prepare evaluation metadata
    # ========================================================

    metadata = (
        graph_item[
            "metadata"
        ]
    )


    if isinstance(
        metadata,
        pd.DataFrame,
    ):

        result_df = (
            metadata
            .copy()
            .reset_index(
                drop=True
            )
        )


    elif isinstance(
        metadata,
        dict,
    ):

        result_df = (
            pd.DataFrame(
                metadata
            )
            .reset_index(
                drop=True
            )
        )


    elif isinstance(
        metadata,
        list,
    ):

        result_df = (
            pd.DataFrame(
                metadata
            )
            .reset_index(
                drop=True
            )
        )


    else:

        raise TypeError(
            "graph_item['metadata'] must be a DataFrame, "
            "dictionary, or list."
        )


    if len(
        result_df
    ) != num_contexts:

        raise RuntimeError(
            "Metadata/context count mismatch.\n"
            f"Metadata rows: {len(result_df)}\n"
            f"Contexts:      {num_contexts}"
        )


    # ========================================================
    # Required identifiers
    # ========================================================

    graph_id = (
        graph_item[
            "graph_id"
        ]
    )


    if (
        "graph_id"
        not in result_df.columns
    ):

        result_df[
            "graph_id"
        ] = graph_id


    if (
        "target_node"
        not in result_df.columns
    ):

        result_df[
            "target_node"
        ] = (
            _recover_target_node_ids(
                graph_item
            )
        )


    if (
        "sample_id"
        not in result_df.columns
    ):

        raise KeyError(
            "Evaluation metadata does not contain sample_id."
        )


    # ========================================================
    # Actual target in MICROSECONDS
    # ========================================================

    result_df[
        "y_exec_us"
    ] = (
        raw_targets_us
    )


    # ========================================================
    # Predicted quantiles in MICROSECONDS
    # ========================================================

    result_df[
        "Q50_pred_us"
    ] = (
        predictions_us[
            :,
            0
        ]
    )

    result_df[
        "Q90_pred_us"
    ] = (
        predictions_us[
            :,
            1
        ]
    )

    result_df[
        "Q95_pred_us"
    ] = (
        predictions_us[
            :,
            2
        ]
    )

    result_df[
        "Q99_pred_us"
    ] = (
        predictions_us[
            :,
            3
        ]
    )


    # ========================================================
    # Verify quantile ordering after inverse transform
    # ========================================================

    crossing_mask = (

        (
            result_df[
                "Q50_pred_us"
            ]
            >
            result_df[
                "Q90_pred_us"
            ]
        )

        |

        (
            result_df[
                "Q90_pred_us"
            ]
            >
            result_df[
                "Q95_pred_us"
            ]
        )

        |

        (
            result_df[
                "Q95_pred_us"
            ]
            >
            result_df[
                "Q99_pred_us"
            ]
        )
    )


    if crossing_mask.any():

        raise RuntimeError(
            "Quantile crossing detected after "
            "inverse transformation."
        )


    # ========================================================
    # Put important columns first
    # ========================================================

    identifier_columns = [
        "sample_id",
        "graph_id",
        "target_node",
    ]


    prediction_columns = [
        "y_exec_us",
        "Q50_pred_us",
        "Q90_pred_us",
        "Q95_pred_us",
        "Q99_pred_us",
    ]


    metadata_columns = [

        column

        for column
        in result_df.columns

        if (
            column
            not in identifier_columns
            and column
            not in prediction_columns
        )
    ]


    final_columns = (

        identifier_columns

        + metadata_columns

        + prediction_columns
    )


    result_df = (
        result_df[
            final_columns
        ]
        .copy()
    )


    # ========================================================
    # Final numeric validation
    # ========================================================

    execution_time_columns = [
        "y_exec_us",
        "Q50_pred_us",
        "Q90_pred_us",
        "Q95_pred_us",
        "Q99_pred_us",
    ]


    if not np.isfinite(

        result_df[
            execution_time_columns
        ].to_numpy(
            dtype=np.float64
        )

    ).all():

        raise RuntimeError(
            "Final inference table contains NaN/Inf values."
        )


    # Store scaler diagnostics without adding them as
    # per-record model inputs.
    result_df.attrs[
        "target_mean_us"
    ] = (
        target_mean_us
    )

    result_df.attrs[
        "target_std_us"
    ] = (
        target_std_us
    )

    result_df.attrs[
        "target_scaler_path"
    ] = (
        scaler_source_path
    )


    return result_df


# ============================================================
# 6. Smoke test on ONE Test-ID DAG
# ============================================================

test_graph_item = (
    test_id_dataset[0]
)


one_graph_predictions_df = (
    infer_one_graph(

        model=(
            evaluation_model
        ),

        graph_item=(
            test_graph_item
        ),

        preprocessing_state=(
            preprocessing_state
        ),

        device=(
            DEVICE
        ),
    )
)


# ============================================================
# 7. One DAG must produce exactly 42 rows
# ============================================================

EXPECTED_CONTEXTS_PER_DAG = 42


if len(
    one_graph_predictions_df
) != EXPECTED_CONTEXTS_PER_DAG:

    raise RuntimeError(
        "Unexpected number of prediction rows.\n"
        f"Expected: {EXPECTED_CONTEXTS_PER_DAG}\n"
        f"Found:    {len(one_graph_predictions_df)}"
    )


# ============================================================
# 8. Final report
# ============================================================

target_mean_us = (
    one_graph_predictions_df.attrs[
        "target_mean_us"
    ]
)

target_std_us = (
    one_graph_predictions_df.attrs[
        "target_std_us"
    ]
)

target_scaler_path = (
    one_graph_predictions_df.attrs[
        "target_scaler_path"
    ]
)


print("=" * 90)
print("ONE-GRAPH INFERENCE VERIFICATION")
print("=" * 90)


print(
    f"\nGraph ID                 : "
    f"{test_graph_item['graph_id']}"
)

print(
    f"Execution contexts       : "
    f"{len(one_graph_predictions_df)}"
)

print(
    "Prediction model         : "
    "BEST validation checkpoint"
)


print(
    "\nRecovered TRAIN target scaler:"
)

print(
    f"  Source path            : "
    f"{target_scaler_path}"
)

print(
    f"  Mean                   : "
    f"{target_mean_us:.6f} us"
)

print(
    f"  Standard deviation     : "
    f"{target_std_us:.6f} us"
)


print(
    "\nInverse transformation:"
)

print(
    "  y_us = y_standardized * train_std + train_mean"
)


print(
    "\nOutput prediction columns:"
)

print(
    "  - y_exec_us"
)

print(
    "  - Q50_pred_us"
)

print(
    "  - Q90_pred_us"
)

print(
    "  - Q95_pred_us"
)

print(
    "  - Q99_pred_us"
)


print(
    "\nOutput shape:"
)

print(
    f"  {one_graph_predictions_df.shape}"
)


print(
    "\nFirst 5 prediction rows:"
)

display(
    one_graph_predictions_df.head()
)


print(
    "\nValidation:"
)

print(
    "  - TRAIN target scaler was recovered successfully."
)

print(
    "  - Recovered scaler reproduces stored standardized targets."
)

print(
    "  - Predictions were converted back to microseconds."
)

print(
    "  - One row was produced per execution context."
)

print(
    "  - Quantile crossing after inverse transform: 0"
)


print(
    "\nSTATUS: ONE-GRAPH INFERENCE PASSED"
)

print("=" * 90)