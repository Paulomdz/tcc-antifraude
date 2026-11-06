"""GNN de risco por transação; IDs determinam somente as arestas do grafo."""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessamento.carregar_dataset import (
    GNN_FEATURE_NAMES,
    compute_if_features,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, GCNConv

    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False


if TORCH_GEO_AVAILABLE:
    class GNNFraudDetector(nn.Module):
        """Mantém camadas GAT/GAT/GCN, variando a profundidade permitida."""

        def __init__(
            self,
            in_channels: int = len(GNN_FEATURE_NAMES),
            hidden_channels: int = 32,
            dropout: float = 0.1,
            num_layers: int = 3,
            num_heads: int = 4,
        ):
            super().__init__()
            if num_layers not in {2, 3}:
                raise ValueError("num_layers deve ser 2 ou 3.")
            self.config = {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "dropout": dropout,
                "num_layers": num_layers,
                "num_heads": num_heads,
            }
            self.gat1 = GATConv(
                in_channels, hidden_channels, heads=num_heads, dropout=dropout
            )
            if num_layers == 3:
                self.gat2 = GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    dropout=dropout,
                )
                gcn_input = hidden_channels
            else:
                self.gat2 = None
                gcn_input = hidden_channels * num_heads
            self.gcn = GCNConv(gcn_input, hidden_channels)
            self.fc1 = nn.Linear(hidden_channels, 16)
            self.fc2 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            x = self.dropout(F.elu(self.gat1(x, edge_index)))
            if self.gat2 is not None:
                x = self.dropout(F.elu(self.gat2(x, edge_index)))
            x = F.elu(self.gcn(x, edge_index))
            x = self.dropout(F.relu(self.fc1(x)))
            return torch.sigmoid(self.fc2(x))
else:
    class GNNFraudDetector:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("PyTorch Geometric é necessário para instanciar a GNN.")


class TransactionMiniBatchDataset:
    """Gera subgrafos e features somente para os mini-batches solicitados."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
        context_df: pd.DataFrame | None = None,
    ):
        if not TORCH_GEO_AVAILABLE:
            raise RuntimeError("PyTorch Geometric é necessário para montar o grafo.")
        if not {"nameOrig", "nameDest", "step"}.issubset(df.columns):
            raise ValueError("O grafo requer IDs de conta e step para as arestas.")
        self.feature_names = feature_names or GNN_FEATURE_NAMES
        if not self.feature_names or any(
            name not in GNN_FEATURE_NAMES for name in self.feature_names
        ):
            raise ValueError("Features GNN inválidas.")

        context = context_df if context_df is not None else df
        if not context.index.is_unique or not df.index.is_unique:
            raise ValueError("Os índices do contexto e dos alvos devem ser únicos.")
        ordered_context = (
            context
            if context["step"].is_monotonic_increasing
            else context.sort_values("step", kind="stable")
        )
        ordered_targets = df.sort_values("step", kind="stable")
        target_positions = pd.Series(
            np.arange(len(ordered_context), dtype=np.int64),
            index=ordered_context.index,
        ).reindex(ordered_targets.index)
        if target_positions.isna().any():
            raise ValueError("Todas as transações-alvo devem existir no contexto.")

        self.context = ordered_context
        self.steps = ordered_context["step"].to_numpy(dtype=np.int64)
        self.target_nodes = target_positions.to_numpy(dtype=np.int64)
        self.y = (
            ordered_targets["isFraud"].to_numpy(dtype=np.float32)
            if "isFraud" in ordered_targets
            else None
        )
        self.num_nodes = len(ordered_targets)
        self._account_indices = [
            self._compact_account_index(
                ordered_context[column].astype("string").fillna("").astype(str).to_numpy()
            )
            for column in ("nameOrig", "nameDest")
        ]
        self._previous_nodes = [
            self._precompute_previous_nodes(offsets, members)
            for _codes, offsets, members in self._account_indices
        ]
        del ordered_targets, target_positions
        gc.collect()

    @staticmethod
    def _compact_account_index(account_ids: np.ndarray):
        codes, _ = pd.factorize(account_ids, sort=False)
        codes = codes.astype(np.int32, copy=False)
        missing = account_ids == ""
        if missing.any():
            codes[missing] = codes.max(initial=-1) + 1 + np.arange(
                int(missing.sum()), dtype=np.int32
            )
        counts = np.bincount(codes)
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        members = np.argsort(codes, kind="stable").astype(np.int32, copy=False)
        return codes, offsets, members

    def _precompute_previous_nodes(
        self, offsets: np.ndarray, members: np.ndarray
    ) -> np.ndarray:
        previous_nodes = np.full(len(self.steps), -1, dtype=np.int64)
        for group in range(len(offsets) - 1):
            account_nodes = members[offsets[group] : offsets[group + 1]]
            account_steps = self.steps[account_nodes]
            previous_positions = (
                np.searchsorted(account_steps, account_steps, side="left") - 1
            )
            has_previous = previous_positions >= 0
            previous_nodes[account_nodes[has_previous]] = account_nodes[
                previous_positions[has_previous]
            ]
        return previous_nodes

    def _neighbors(self, node: int):
        for previous_nodes in self._previous_nodes:
            previous_node = previous_nodes[node]
            if previous_node >= 0:
                yield int(previous_node)

    def _sample_nodes(
        self,
        seeds: np.ndarray,
        fanouts: tuple[int, ...],
    ) -> list[int]:
        nodes = list(map(int, seeds))
        seen = set(nodes)
        frontier = list(nodes)  # cópia: iterar sobre `nodes` expandia toda a componente
        for fanout in fanouts:
            next_frontier = []
            for node in frontier:
                candidates = list(dict.fromkeys(self._neighbors(node)))
                for neighbor in candidates[:fanout]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        nodes.append(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return nodes

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        fanouts: tuple[int, ...] = (10, 5),
    ):
        if batch_size < 1 or any(fanout < 1 for fanout in fanouts):
            raise ValueError("batch_size e fanouts devem ser positivos.")
        seed_order = np.arange(self.num_nodes)
        if shuffle:
            np.random.shuffle(seed_order)
        for start in range(0, self.num_nodes, batch_size):
            target_positions = seed_order[start : start + batch_size]
            seeds = self.target_nodes[target_positions]
            nodes = self._sample_nodes(seeds, fanouts)
            local_index = {node: index for index, node in enumerate(nodes)}
            edges = set()
            for node in nodes:
                for neighbor in self._neighbors(node):
                    source = local_index.get(neighbor)
                    if source is not None:
                        edges.add((source, local_index[node]))
            if edges:
                edge_index = torch.tensor(
                    sorted(edges), dtype=torch.long
                ).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            feature_frame = compute_if_features(self.context.iloc[nodes])
            features = feature_frame[self.feature_names].to_numpy(
                dtype=np.float32
            )
            yield (
                torch.from_numpy(features),
                edge_index,
                (
                    torch.from_numpy(self.y[target_positions].reshape(-1, 1))
                    if self.y is not None
                    else None
                ),
                len(target_positions),
            )


def treinar_gnn(df_treino: pd.DataFrame, epochs: int = 20, batch_size: int = 1):
    """Rotina de treino simples para compatibilidade e testes locais."""
    if not TORCH_GEO_AVAILABLE:
        raise RuntimeError("PyTorch Geometric é necessário para treinar a GNN.")
    dataset = TransactionMiniBatchDataset(df_treino)
    model = GNNFraudDetector().to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction="none")
    for _ in range(epochs):
        model.train()
        for x, edge_index, labels, seed_count in dataset.iter_batches(
            batch_size, shuffle=True
        ):
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities = model(x, edge_index)[:seed_count]
            loss = criterion(probabilities, labels).mean()
            loss.backward()
            optimizer.step()
    return model, dataset


def salvar_gnn(
    model: Any,
    dataset: TransactionMiniBatchDataset | None = None,
    filepath: str = "src/modelos/modelos_salvos/gnn_model.pt",
    metadata: dict[str, Any] | None = None,
) -> None:
    if not TORCH_GEO_AVAILABLE or model is None:
        raise RuntimeError("Não é possível salvar a GNN sem PyTorch Geometric.")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model_metadata = {
        "feature_names": (
            dataset.feature_names if dataset is not None else GNN_FEATURE_NAMES
        ),
        **(metadata or {}),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "metadata": model_metadata,
        },
        filepath,
    )


def carregar_gnn(filepath: str = "src/modelos/modelos_salvos/gnn_model.pt"):
    if not TORCH_GEO_AVAILABLE:
        raise RuntimeError("PyTorch Geometric não disponível para carregar a GNN.")
    model_path = Path(filepath)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo GNN não encontrado: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        model = GNNFraudDetector(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint.get("metadata", {})
    else:
        model = GNNFraudDetector()
        model.load_state_dict(checkpoint)
        metadata = {"feature_names": GNN_FEATURE_NAMES}
    model.eval()
    return model, metadata
