"""Arquitetura LSTM e sequências de transações por conta de origem."""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessamento.carregar_dataset import (
    LSTM_FEATURE_NAMES,
    compute_if_features,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LSTMFraudDetector(nn.Module):
        """LSTM bidirecional com atenção; loss e otimizador permanecem BCE/Adam."""

        def __init__(
            self,
            input_size: int = len(LSTM_FEATURE_NAMES),
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            if num_layers not in {2, 3}:
                raise ValueError("num_layers deve ser 2 ou 3.")
            self.config = {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            }
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.fc1 = nn.Linear(hidden_size * 2, 32)
            self.fc2 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_output, _ = self.lstm(x)
            attention_weights = self.sigmoid(self.attention(lstm_output))
            context = (lstm_output * attention_weights).sum(dim=1)
            hidden = self.dropout(self.relu(self.fc1(context)))
            return self.sigmoid(self.fc2(hidden))


    class TransactionSequenceDataset(Dataset):
        """Janelas causais por conta; IDs agrupam histórico, nunca são features."""

        def __init__(
            self,
            df: pd.DataFrame,
            seq_length: int = 5,
            feature_cols: list[str] | None = None,
            context_df: pd.DataFrame | None = None,
        ):
            if seq_length < 1:
                raise ValueError("seq_length deve ser positivo.")
            feature_cols = feature_cols or LSTM_FEATURE_NAMES
            missing = set(feature_cols) - set(LSTM_FEATURE_NAMES)
            if missing:
                raise ValueError(f"Features LSTM desconhecidas: {sorted(missing)}")

            context = context_df if context_df is not None else df
            if not context.index.is_unique or not df.index.is_unique:
                raise ValueError("Os índices do contexto e dos alvos devem ser únicos.")
            sorted_context = context.sort_values("step", kind="stable")
            sorted_targets = df.sort_values("step", kind="stable")
            target_positions = pd.Series(
                np.arange(len(sorted_context), dtype=np.int64),
                index=sorted_context.index,
            ).reindex(sorted_targets.index)
            if target_positions.isna().any():
                raise ValueError("Todas as transações-alvo devem existir no contexto.")

            all_features = compute_if_features(sorted_context)
            feature_indices = [
                all_features.columns.get_loc(name) for name in feature_cols
            ]
            self.features = all_features.iloc[:, feature_indices].to_numpy(
                dtype=np.float32
            )
            self.steps = sorted_context["step"].to_numpy(dtype=np.int64)
            self.target_positions = target_positions.to_numpy(dtype=np.int64)
            self.labels = sorted_targets["isFraud"].to_numpy(dtype=np.float32)
            self.seq_length = seq_length

            self.account_codes = np.full(len(sorted_context), -1, dtype=np.int32)
            self.account_offsets = np.zeros(1, dtype=np.int64)
            self.account_members = np.empty(0, dtype=np.int32)
            if "nameOrig" in sorted_context:
                account_ids = sorted_context["nameOrig"].astype("string")
                valid_positions = np.flatnonzero(
                    account_ids.fillna("").ne("").to_numpy()
                )
                valid_accounts = account_ids.iloc[valid_positions]
                valid_codes, _ = pd.factorize(valid_accounts, sort=False)
                valid_codes = valid_codes.astype(np.int32, copy=False)
                counts = np.bincount(valid_codes)
                self.account_offsets = np.empty(len(counts) + 1, dtype=np.int64)
                self.account_offsets[0] = 0
                np.cumsum(counts, out=self.account_offsets[1:])
                self.account_members = valid_positions[
                    np.argsort(valid_codes, kind="stable")
                ].astype(np.int32, copy=False)
                self.account_codes[valid_positions] = valid_codes
            del all_features, sorted_context, sorted_targets, target_positions
            gc.collect()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            target = self.target_positions[idx]
            group_id = self.account_codes[target]
            if group_id < 0:
                indices = np.empty(0, dtype=np.int32)
            else:
                start = self.account_offsets[group_id]
                end = self.account_offsets[group_id + 1]
                group = self.account_members[start:end]
                group_steps = self.steps[group]
                end = int(np.searchsorted(group_steps, self.steps[target], side="left"))
                indices = group[max(0, end - self.seq_length) : end]
            sequence = np.zeros(
                (self.seq_length, self.features.shape[1]), dtype=np.float32
            )
            sequence[self.seq_length - len(indices) :] = self.features[indices]
            return (
                torch.from_numpy(sequence),
                torch.tensor([self.labels[idx]], dtype=torch.float32),
            )
else:
    class LSTMFraudDetector:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("PyTorch é necessário para instanciar a LSTM.")


def treinar_lstm(
    df_treino: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """Rotina de treino simples para compatibilidade e testes locais."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch é necessário para treinar a LSTM.")
    from torch.utils.data import DataLoader

    dataset = TransactionSequenceDataset(df_treino)
    if not len(dataset):
        raise ValueError("Não há transações para treinar a LSTM.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMFraudDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(sequences), labels).backward()
            optimizer.step()
    return model


def salvar_lstm(
    model: Any,
    filepath: str = "src/modelos/modelos_salvos/lstm_model.pt",
    metadata: dict[str, Any] | None = None,
) -> None:
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("Não é possível salvar a LSTM sem PyTorch e modelo.")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.config,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, filepath)


def carregar_lstm(filepath: str = "src/modelos/modelos_salvos/lstm_model.pt"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch não disponível para carregar a LSTM.")
    model_path = Path(filepath)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo LSTM não encontrado: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        model = LSTMFraudDetector(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = LSTMFraudDetector()
        model.load_state_dict(checkpoint)
        checkpoint = {"metadata": {}}
    model.eval()
    return model, checkpoint.get("metadata", {})
