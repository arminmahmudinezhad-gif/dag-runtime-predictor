"""
GNN node-embedding extraction, for correlating the frozen model's *learned
representations* (not just topological proxies) with prediction accuracy on
critical-path vs. non-critical-path nodes (spec section 4: "correlation
between GNN-extracted features and prediction accuracy on critical-path
nodes"). Read-only: calls model.dag_encoder(...) on the frozen model, which
performs no weight update.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract_node_embeddings(model, module, split: str, graph_id: str, preprocessing_state: dict,
                             device: torch.device) -> dict[str, np.ndarray]:
    """node_id -> hidden_dim embedding vector, from the frozen DAGEncoder."""
    model.eval()
    static_graph = module.load_static_graph(graph_id=graph_id, split=split,
                                             preprocessing_state=preprocessing_state)
    x = static_graph["x"].to(device)
    edge_index = static_graph["edge_index"].to(device)
    edge_attr = static_graph["edge_attr"].to(device)
    node_embeddings = model.dag_encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
    node_embeddings = node_embeddings.cpu().numpy()

    index_to_id = {v: k for k, v in static_graph["node_id_to_index"].items()}
    return {str(index_to_id[i]): node_embeddings[i] for i in range(node_embeddings.shape[0])}


def embedding_features_for_predictions(model, module, preprocessing_state: dict, device: torch.device,
                                        predictions_df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Add two embedding-derived columns to a predictions table:
      - embedding_norm: L2 norm of the target node's learned embedding
      - embedding_dist_from_graph_mean: L2 distance from that graph's mean
        node embedding (a proxy for "how structurally unusual" the node is
        within its own DAG)
    """
    out = predictions_df.copy()
    out["graph_id"] = out["graph_id"].astype(str)
    out["target_node"] = out["target_node"].astype(str)

    norms = pd.Series(index=out.index, dtype="float64")
    dists = pd.Series(index=out.index, dtype="float64")

    for graph_id, sub in out.groupby("graph_id"):
        emb = extract_node_embeddings(model, module, split, graph_id, preprocessing_state, device)
        all_vectors = np.stack(list(emb.values()))
        graph_mean = all_vectors.mean(axis=0)
        for idx, node_id in zip(sub.index, sub["target_node"]):
            vec = emb[node_id]
            norms.loc[idx] = float(np.linalg.norm(vec))
            dists.loc[idx] = float(np.linalg.norm(vec - graph_mean))

    out["embedding_norm"] = norms
    out["embedding_dist_from_graph_mean"] = dists
    return out
