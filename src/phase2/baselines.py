"""
Comparison baselines requested by the spec's expected-output section:
MLP (no graph structure), GNN-Mean (same GNN backbone, single MSE-trained mean
head instead of the quantile head), and GNN-Quantile-without-calibration (simply
the frozen Phase 1 raw predictions already produced elsewhere -- no new model).

Both new baselines share the frozen model's exact forward() call signature
(x, edge_index, edge_attr, target_node_indices, core_type_indices,
context_numeric) so they can reuse the same GraphScenarioDataset pipeline, but
they are NEW, independently-trained models -- the frozen Phase 1 quantile
predictor itself is never touched, matching the "treat Phase 1 as frozen" rule.

Trained under a reduced, explicitly-stated compute budget (subset of graphs,
few epochs) for tractability on CPU; they are meant to establish a directional
baseline comparison, not to be exhaustively tuned competitors.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Point predictor using ONLY the target node's own static features + context
    (no message passing, no neighboring-node information) -- the "no graph
    structure" ablation requested by the spec."""

    def __init__(self, node_input_dim: int, num_core_types: int, numerical_context_dim: int,
                 core_embedding_dim: int = 4, hidden_dims=(64, 32)):
        super().__init__()
        self.core_embedding = nn.Embedding(num_core_types, core_embedding_dim)
        in_dim = node_input_dim + core_embedding_dim + numerical_context_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, target_node_indices, core_type_indices, context_numeric):
        node_feats = x[target_node_indices]
        core_emb = self.core_embedding(core_type_indices)
        fused_input = torch.cat([node_feats, core_emb, context_numeric], dim=1)
        return self.net(fused_input).squeeze(-1)


class GNNMeanBaseline(nn.Module):
    """Same DAGEncoder + ContextFusionEncoder backbone family as the frozen
    Phase 1 model, but a single linear mean head trained with MSE instead of
    the pinball-loss ordered-quantile head. Reduced hidden size for tractable
    from-scratch training on CPU."""

    def __init__(self, module, node_input_dim: int, edge_dim: int, num_core_types: int,
                 numerical_context_dim: int, hidden_dim: int = 64, num_gnn_layers: int = 2,
                 attention_hidden_dim: int = 32, core_embedding_dim: int = 8, dropout: float = 0.1,
                 ablate_zt: bool = False):
        """ablate_zt=True zeroes the 7 standardized z_t columns of context_numeric
        before context fusion (standardized mean == 0), simulating an "offline"
        predictor blind to current system state -- used for the offline-vs-online
        z_t-awareness ablation (spec section 7)."""
        super().__init__()
        self.ablate_zt = ablate_zt
        self.dag_encoder = module.DAGEncoder(
            node_input_dim=node_input_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers, attention_hidden_dim=attention_hidden_dim,
            dropout=dropout, use_residual=False, use_layer_norm=True,
        )
        self.context_fusion = module.ContextFusionEncoder(
            node_hidden_dim=hidden_dim, num_core_types=num_core_types,
            numerical_context_dim=numerical_context_dim, fused_dim=hidden_dim,
            dropout=dropout, core_embedding_dim=core_embedding_dim,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, target_node_indices, core_type_indices, context_numeric):
        if self.ablate_zt:
            context_numeric = context_numeric.clone()
            context_numeric[:, 2:] = 0.0  # columns 0-1 = DVFS, 2-8 = z_t (standardized mean = 0)
        node_embeddings = self.dag_encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
        target_embeddings = node_embeddings[target_node_indices]
        fused = self.context_fusion(
            target_node_embeddings=target_embeddings,
            core_type_indices=core_type_indices,
            context_numeric=context_numeric,
        )
        return self.head(fused).squeeze(-1)


def train_point_baseline(
    model: nn.Module,
    train_dataset,
    val_dataset,
    epochs: int = 8,
    lr: float = 1e-3,
    train_graph_subset: int | None = 150,
    val_graph_subset: int | None = 30,
    seed: int = 20260807,
    device: torch.device | None = None,
    log_prefix: str = "",
) -> dict:
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rng = random.Random(seed)
    train_indices = list(range(len(train_dataset)))
    rng.shuffle(train_indices)
    if train_graph_subset is not None:
        train_indices = train_indices[:train_graph_subset]

    val_indices = list(range(len(val_dataset)))
    if val_graph_subset is not None:
        val_indices = val_indices[:val_graph_subset]

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for idx in train_indices:
            item = train_dataset[idx]
            optimizer.zero_grad()
            pred = model(
                x=item["x"].to(device), edge_index=item["edge_index"].to(device),
                edge_attr=item["edge_attr"].to(device),
                target_node_indices=item["target_node_indices"].to(device),
                core_type_indices=item["core_type_indices"].to(device),
                context_numeric=item["context_numeric"].to(device),
            )
            loss = loss_fn(pred, item["y"].to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for idx in val_indices:
                item = val_dataset[idx]
                pred = model(
                    x=item["x"].to(device), edge_index=item["edge_index"].to(device),
                    edge_attr=item["edge_attr"].to(device),
                    target_node_indices=item["target_node_indices"].to(device),
                    core_type_indices=item["core_type_indices"].to(device),
                    context_numeric=item["context_numeric"].to(device),
                )
                val_losses.append(loss_fn(pred, item["y"].to(device)).item())

        mean_train, mean_val = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": mean_train, "val_mse": mean_val,
                         "elapsed_s": time.time() - t0})
        print(f"{log_prefix}epoch {epoch}: train_mse={mean_train:.4f} val_mse={mean_val:.4f} "
              f"({time.time()-t0:.1f}s)")

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_mse": best_val}
