"""Busca temporal, retreino final e ablação dos três modelos antifraude."""

import argparse
import csv
import gc
import hashlib
import itertools
import json
import pickle
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.modelos.arquitetura_gnn import (
    TORCH_GEO_AVAILABLE,
    GNNFraudDetector,
    TransactionMiniBatchDataset,
    salvar_gnn,
)
from src.modelos.arquitetura_lstm import (
    TORCH_AVAILABLE,
    LSTMFraudDetector,
    TransactionSequenceDataset,
    salvar_lstm,
)
from src.ferramentas.inferencia_modelos import IF_SCORE_SCALE
from src.preprocessamento.carregar_dataset import (
    FEATURE_STATS_PATH,
    GNN_FEATURE_NAMES,
    IF_FEATURE_NAMES,
    LSTM_FEATURE_NAMES,
    PROCESSED_DATA_PATH,
    compute_if_features,
    generate_feature_stats,
    temporal_split,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    DataLoader = None

SEED = 42
MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos_salvos"
REPORT_PATH = Path(__file__).resolve().parents[3] / "reports" / "treinamento_etapa2.json"
GRID_CSV_PATH = REPORT_PATH.parent / "etapa2_grid.csv"
CHECKPOINT_DIR = REPORT_PATH.parent / "etapa2_checkpoints"
MODEL_TIME_LIMIT_SECONDS = 30 * 60
SEARCH_TIME_LIMIT_SECONDS = 15 * 60
MAX_EPOCHS = 10
PATIENCE = 2
LSTM_SEQUENCE_LENGTH = 5
LSTM_BATCH_SIZE = 512
GNN_EPOCHS = 10
GNN_BATCH_SIZE = 1024
GNN_FANOUTS = (10, 5)
DATASET_SIGNATURE = ""
FEATURE_POLICY_VERSION = "pre_execution_causal_v2"


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)


def _ram_snapshot() -> dict[str, float]:
    values = {}
    with Path("/proc/meminfo").open(encoding="ascii") as memory_info:
        for line in memory_info:
            if line.startswith(("MemFree:", "MemAvailable:")):
                name, value, _unit = line.split()
                values[name[:-1]] = int(value) / (1024**2)
    if "MemFree" not in values or "MemAvailable" not in values:
        raise RuntimeError("Não foi possível medir a RAM em /proc/meminfo.")
    return {
        "free_gib": values["MemFree"],
        "available_gib": values["MemAvailable"],
    }


def _current_rss_gib() -> float:
    with Path("/proc/self/status").open(encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024**2)
    raise RuntimeError("Não foi possível medir o RSS do processo.")


def _vram_snapshot() -> dict[str, float]:
    if torch is None or not torch.cuda.is_available():
        return {"free_gib": 0.0, "total_gib": 0.0}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_gib": free_bytes / (1024**3),
        "total_gib": total_bytes / (1024**3),
    }


def _reset_vram_peak() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gib() -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


class _PeakRSSMonitor:
    def __init__(self) -> None:
        self.peak_gib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _record(self) -> None:
        self.peak_gib = max(self.peak_gib, _current_rss_gib())

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(0.1)

    def __enter__(self):
        self._record()
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._stop.set()
        self._thread.join()
        self._record()


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _checkpoint_path(model_name: str, variant: str, config: dict[str, Any]) -> Path:
    key = "|".join(
        (FEATURE_POLICY_VERSION, DATASET_SIGNATURE, _config_key(config))
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    suffix = "json" if model_name == "isolation_forest" else "pt"
    return CHECKPOINT_DIR / f"{model_name}_{variant}_{digest}.{suffix}"


def _final_checkpoint_path(model_name: str, variant: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_{variant}_final.json"


def _load_final_checkpoint(model_name: str, variant: str):
    checkpoint_path = _final_checkpoint_path(model_name, variant)
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open(encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    if checkpoint.get("dataset_signature") != DATASET_SIGNATURE:
        return None
    if not Path(checkpoint["result"]["model_path"]).exists():
        return None
    return checkpoint["result"]


def _save_final_checkpoint(
    model_name: str, variant: str, result: dict[str, Any]
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _final_checkpoint_path(model_name, variant)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {"dataset_signature": DATASET_SIGNATURE, "result": result},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _grid_rows() -> list[dict[str, str]]:
    if not GRID_CSV_PATH.exists():
        return []
    with GRID_CSV_PATH.open(newline="", encoding="utf-8") as grid_file:
        rows = list(csv.DictReader(grid_file))
    if any(
        row.get("feature_policy") != FEATURE_POLICY_VERSION
        or row.get("dataset_signature") != DATASET_SIGNATURE
        for row in rows
    ):
        raise RuntimeError(
            "O grid contém resultados de outra política ou dataset; "
            "arquive-o antes de iniciar este treino."
        )
    return rows


def _append_grid_row(row: dict[str, Any]) -> None:
    GRID_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not GRID_CSV_PATH.exists() or GRID_CSV_PATH.stat().st_size == 0
    with GRID_CSV_PATH.open("a", newline="", encoding="utf-8") as grid_file:
        writer = csv.DictWriter(
            grid_file,
            fieldnames=(
                "model",
                "variant",
                "config",
                "validation_pr_auc",
                "validation_f1",
                "training_seconds",
                "peak_ram_gib",
                "ram_free_before_gib",
                "ram_available_before_gib",
                "feature_policy",
                "dataset_signature",
                "vram_free_before_gib",
                "peak_vram_gib",
                "status",
                "error",
            ),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        grid_file.flush()
        import os

        os.fsync(grid_file.fileno())


def _checkpoint_grid_row(
    model_name: str,
    variant: str,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "variant": variant,
        "config": _config_key(config),
        "validation_pr_auc": checkpoint.get("validation_metrics", {}).get(
            "pr_auc", ""
        ),
        "validation_f1": checkpoint.get("validation_metrics", {}).get(
            "f1", ""
        ),
        "training_seconds": checkpoint["training_seconds"],
        "peak_ram_gib": checkpoint["peak_ram_gib"],
        "ram_free_before_gib": checkpoint["ram_before"]["free_gib"],
        "ram_available_before_gib": checkpoint["ram_before"]["available_gib"],
        "feature_policy": FEATURE_POLICY_VERSION,
        "dataset_signature": DATASET_SIGNATURE,
        "vram_free_before_gib": checkpoint.get("vram_before", {}).get(
            "free_gib", 0.0
        ),
        "peak_vram_gib": checkpoint.get("peak_vram_gib", 0.0),
        "status": checkpoint.get("status", "completed"),
        "error": checkpoint.get("error", ""),
    }


def _record_memory_skip(
    model_name: str,
    variant: str,
    config: dict[str, Any],
    ram_before: dict[str, float],
    vram_before: dict[str, float],
    peak_ram_gib: float,
    elapsed_seconds: float,
    error: str,
    record_grid: bool = True,
) -> None:
    if not record_grid:
        return
    _append_grid_row(
        _checkpoint_grid_row(
            model_name,
            variant,
            config,
            {
                "validation_metrics": {},
                "training_seconds": elapsed_seconds,
                "peak_ram_gib": peak_ram_gib,
                "ram_before": ram_before,
                "vram_before": vram_before,
                "status": "skipped_memory",
                "error": error,
            },
        )
    )


def _metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def _if_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    decision = np.asarray(model.decision_function(features), dtype=np.float64)
    return 1.0 / (
        1.0 + np.exp(np.clip(decision * IF_SCORE_SCALE, -700, 700))
    )


def _candidate_configs(
    model_name: str, primeiro_passe: bool = False
) -> list[dict[str, Any]]:
    if model_name == "isolation_forest":
        initial_configs = [
            {
                "n_estimators": n_estimators,
                "max_samples": "auto",
                "contamination": contamination,
            }
            for n_estimators, contamination in itertools.product(
                (100, 200, 300), (0.03, 0.056, 0.10)
            )
        ]
        if primeiro_passe:
            return initial_configs
        expanded_configs = [
            {
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "contamination": contamination,
            }
            for n_estimators, max_samples, contamination in itertools.product(
                (100, 200, 300), (0.5, 0.8), (0.03, 0.056, 0.10)
            )
        ]
        return initial_configs + expanded_configs

    hidden_key = "hidden_size" if model_name == "lstm" else "hidden_dim"
    initial_configs = [
        {
            hidden_key: hidden,
            "dropout": dropout,
            "lr": lr,
            "num_layers": 2,
        }
        for hidden, dropout, lr in itertools.product(
            (32, 64), (0.1, 0.3), (1e-3, 5e-4)
        )
    ]
    if primeiro_passe:
        return initial_configs
    expanded_configs = [
        {
            hidden_key: hidden,
            "dropout": dropout,
            "lr": lr,
            "num_layers": 3,
        }
        for hidden, dropout, lr in itertools.product(
            (32, 64), (0.1, 0.3), (1e-3, 5e-4)
        )
    ]
    return initial_configs + expanded_configs


def _feature_matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    features = compute_if_features(df)
    if not isinstance(features, pd.DataFrame):
        raise TypeError("O treinamento requer features tabulares em DataFrame.")
    matrix = features[names].to_numpy(dtype=np.float32)
    del features
    gc.collect()
    return matrix


def _class_positive_weight(labels: np.ndarray) -> float:
    positive = int(np.count_nonzero(labels == 1))
    negative = int(np.count_nonzero(labels == 0))
    if positive == 0 or negative == 0:
        raise ValueError("O treino core precisa conter as duas classes.")
    return negative / positive


def _weighted_bce(criterion, probabilities, labels, positive_weight):
    weights = torch.where(labels > 0.5, positive_weight, 1.0)
    return (criterion(probabilities, labels) * weights).mean()


def _lstm_loaders(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    batch_size: int = LSTM_BATCH_SIZE,
):
    context = splits.get("context")
    datasets = {
        key: TransactionSequenceDataset(
            splits[key],
            seq_length=LSTM_SEQUENCE_LENGTH,
            feature_cols=feature_names,
            context_df=context,
        )
        for key in ("train", "val", "train_completo")
    }
    return {
        key: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=0,
            pin_memory=bool(torch.cuda.is_available()),
        )
        for key, dataset in datasets.items()
    }


def _evaluate_lstm(
    model,
    loader,
    device,
    positive_weight: float,
    criterion,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    labels_out = []
    scores_out = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = model(sequences)
            loss = _weighted_bce(criterion, probabilities, labels, positive_weight)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            labels_out.append(labels.detach().cpu().numpy().reshape(-1))
            scores_out.append(probabilities.detach().cpu().numpy().reshape(-1))
    return (
        total_loss / max(total_items, 1),
        np.concatenate(labels_out),
        np.concatenate(scores_out),
    )


def _train_lstm_candidate(
    loaders,
    feature_names: list[str],
    config: dict[str, Any],
    positive_weight: float,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed()
    model = LSTMFraudDetector(
        input_size=len(feature_names),
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.BCELoss(reduction="none")
    best_loss = float("inf")
    best_pr_auc = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_validation = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for sequences, labels in loaders["train"]:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            probabilities = model(sequences)
            loss = _weighted_bce(criterion, probabilities, labels, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        validation_loss, val_y, val_scores = _evaluate_lstm(
            model, loaders["val"], device, positive_weight, criterion
        )
        validation_metrics = _metrics(val_y, val_scores)
        scheduler.step(validation_loss)
        if validation_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = validation_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_validation = validation_metrics
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Treino da LSTM não produziu estado de validação.")
    return {
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "validation_metrics": best_validation,
    }


def _refit_lstm(
    loaders,
    feature_names: list[str],
    config: dict[str, Any],
    positive_weight: float,
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed()
    model = LSTMFraudDetector(
        input_size=len(feature_names),
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss(reduction="none")
    for _ in range(max(1, epochs)):
        model.train()
        for sequences, labels in loaders["train_completo"]:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = _weighted_bce(
                criterion, model(sequences), labels, positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model, device


def _if_candidate_search(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    deadline: float,
    variant: str,
    primeiro_passe: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], int, bool]:
    configs = _candidate_configs("isolation_forest", primeiro_passe)
    completed = 0
    reduced = False
    evaluated_configs = []
    rows = _grid_rows()
    existing = {
        (row["model"], row["variant"], row["config"]): row
        for row in rows
    }
    for config in configs:
        if time.perf_counter() >= deadline:
            reduced = True
            break
        checkpoint_path = _checkpoint_path("isolation_forest", variant, config)
        row_key = ("isolation_forest", variant, _config_key(config))
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("dataset_signature") != DATASET_SIGNATURE
                or checkpoint.get("feature_policy") != FEATURE_POLICY_VERSION
            ):
                raise RuntimeError(
                    f"Checkpoint IF incompatível: {checkpoint_path}"
                )
            if row_key not in existing:
                _append_grid_row(
                    _checkpoint_grid_row(
                        "isolation_forest", variant, config, checkpoint
                    )
                )
                existing[row_key] = _checkpoint_grid_row(
                    "isolation_forest", variant, config, checkpoint
                )
        elif row_key in existing:
            row = existing[row_key]
            if row.get("status") == "skipped_memory":
                reduced = True
                break
            checkpoint = {
                "validation_metrics": {
                    "pr_auc": float(row["validation_pr_auc"]),
                    "f1": float(row["validation_f1"]),
                    "threshold": 0.5,
                },
                "training_seconds": float(row["training_seconds"]),
                "peak_ram_gib": float(row["peak_ram_gib"]),
                "ram_before": {
                    "free_gib": float(row["ram_free_before_gib"]),
                    "available_gib": float(row["ram_available_before_gib"]),
                },
                "vram_before": {
                    "free_gib": float(row["vram_free_before_gib"] or 0.0),
                    "total_gib": 0.0,
                },
                "peak_vram_gib": float(row["peak_vram_gib"] or 0.0),
                "status": row.get("status", "completed"),
            }
        else:
            ram_before = _ram_snapshot()
            vram_before = _vram_snapshot()
            if ram_before["available_gib"] < 2:
                _record_memory_skip(
                    "isolation_forest",
                    variant,
                    config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    0.0,
                    "RAM disponível abaixo de 2 GiB antes do ajuste.",
                )
                reduced = True
                print("Grade IF reduzida: RAM disponível abaixo de 2 GiB.", flush=True)
                break
            started = time.perf_counter()
            _reset_vram_peak()
            model = IsolationForest(
                **config,
                random_state=SEED,
                n_jobs=4,
            )
            try:
                with _PeakRSSMonitor() as memory_monitor:
                    model.fit(x_train)
                    scores = _if_scores(model, x_val)
                    metrics = _metrics(y_val, scores)
            except MemoryError as error:
                _record_memory_skip(
                    "isolation_forest",
                    variant,
                    config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    time.perf_counter() - started,
                    f"{type(error).__name__}: {error}",
                )
                reduced = True
                break
            checkpoint = {
                "dataset_signature": DATASET_SIGNATURE,
                "feature_policy": FEATURE_POLICY_VERSION,
                "validation_metrics": metrics,
                "training_seconds": time.perf_counter() - started,
                "peak_ram_gib": memory_monitor.peak_gib,
                "ram_before": ram_before,
                "vram_before": vram_before,
                "peak_vram_gib": _peak_vram_gib(),
            }
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8"
            )
            temporary_path.replace(checkpoint_path)
            _append_grid_row(
                _checkpoint_grid_row(
                    "isolation_forest", variant, config, checkpoint
                )
            )
            existing[row_key] = _checkpoint_grid_row(
                "isolation_forest", variant, config, checkpoint
            )
            del model
        metrics = checkpoint["validation_metrics"]
        completed += 1
        evaluated_configs.append(
            {
                "config": config,
                "metrics": metrics,
            }
        )
        print(
            f"  IF {completed:02d}/{len(configs)}: {config} | "
            f"validação PR-AUC={metrics['pr_auc']:.4f} "
            f"| treino={checkpoint['training_seconds']:.1f}s "
            f"| RAM início={checkpoint['ram_before']['free_gib']:.2f} GiB "
            f"| RAM pico={checkpoint['peak_ram_gib']:.2f} GiB "
            f"| VRAM pico={checkpoint.get('peak_vram_gib', 0.0):.2f} GiB",
            flush=True,
        )
    if not evaluated_configs:
        raise TimeoutError("O orçamento de busca do Isolation Forest expirou.")
    best_contamination_by_base = {}
    for candidate in evaluated_configs:
        config = candidate["config"]
        base_config = {
            key: value
            for key, value in config.items()
            if key != "contamination"
        }
        base_key = _config_key(base_config)
        current = best_contamination_by_base.get(base_key)
        if current is None or candidate["metrics"]["f1"] > current["metrics"]["f1"]:
            best_contamination_by_base[base_key] = candidate
    best_result = max(
        best_contamination_by_base.values(),
        key=lambda candidate: (
            candidate["metrics"]["pr_auc"],
            candidate["metrics"]["f1"],
        ),
    )
    selected_config = best_result["config"]
    selected_ram_before = _ram_snapshot()
    selected_vram_before = _vram_snapshot()
    if selected_ram_before["available_gib"] < 2:
        _record_memory_skip(
            "isolation_forest",
            "selected_refit",
            selected_config,
            selected_ram_before,
            selected_vram_before,
            _current_rss_gib(),
            0.0,
            "RAM disponível abaixo de 2 GiB antes do refit selecionado.",
        )
        raise MemoryError("RAM insuficiente para refitar a configuração IF escolhida.")
    selected_started = time.perf_counter()
    _reset_vram_peak()
    best_model = IsolationForest(
        **selected_config,
        random_state=SEED,
        n_jobs=4,
    )
    with _PeakRSSMonitor() as selected_memory:
        best_model.fit(x_train)
    best_result = {
        "config": selected_config,
        "metrics": best_result["metrics"],
        "selection_refit": {
            "training_seconds": time.perf_counter() - selected_started,
            "peak_ram_gib": selected_memory.peak_gib,
            "ram_before": selected_ram_before,
            "vram_before": selected_vram_before,
            "peak_vram_gib": _peak_vram_gib(),
        },
    }
    return best_model, best_result, completed, reduced


def _prepare_gnn_graphs(
    splits: dict[str, pd.DataFrame], feature_names: list[str]
):
    context = splits.get("context")
    return {
        key: TransactionMiniBatchDataset(
            splits[key], feature_names, context_df=context
        )
        for key in ("train", "val", "train_completo")
    }


def _graph_scores(
    model,
    graph_data: TransactionMiniBatchDataset,
    device,
    positive_weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    labels_out = []
    scores_out = []
    criterion = nn.BCELoss(reduction="none")
    with torch.no_grad():
        for x, edge_index, labels, seed_count in graph_data.iter_batches(
            GNN_BATCH_SIZE, shuffle=False, fanouts=GNN_FANOUTS
        ):
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            probabilities = model(x, edge_index)[:seed_count]
            loss = _weighted_bce(
                criterion, probabilities, labels, positive_weight
            )
            total_loss += float(loss.item()) * seed_count
            total_items += seed_count
            labels_out.append(labels.detach().cpu().numpy().reshape(-1))
            scores_out.append(probabilities.detach().cpu().numpy().reshape(-1))
    return (
        total_loss / max(total_items, 1),
        np.concatenate(labels_out),
        np.concatenate(scores_out),
    )


def _train_gnn_candidate(
    graphs,
    feature_names: list[str],
    config: dict[str, Any],
    positive_weight: float,
    max_epochs: int = GNN_EPOCHS,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed()
    model = GNNFraudDetector(
        in_channels=len(feature_names),
        hidden_channels=config["hidden_dim"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.BCELoss(reduction="none")
    best_loss = float("inf")
    best_pr_auc = float("-inf")
    best_state = None
    best_epoch = 0
    best_validation = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, edge_index, labels, seed_count in graphs["train"].iter_batches(
            GNN_BATCH_SIZE, shuffle=True, fanouts=GNN_FANOUTS
        ):
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities = model(x, edge_index)[:seed_count]
            loss = _weighted_bce(criterion, probabilities, labels, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        validation_loss, val_y, val_scores = _graph_scores(
            model, graphs["val"], device, positive_weight
        )
        validation_metrics = _metrics(val_y, val_scores)
        scheduler.step(validation_loss)
        if validation_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = validation_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_validation = validation_metrics
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if best_state is None:
        raise RuntimeError("Treino da GNN não produziu estado de validação.")
    return {
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "validation_metrics": best_validation,
    }


def _refit_gnn(
    graphs,
    feature_names: list[str],
    config: dict[str, Any],
    positive_weight: float,
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed()
    model = GNNFraudDetector(
        in_channels=len(feature_names),
        hidden_channels=config["hidden_dim"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss(reduction="none")
    for _ in range(max(1, epochs)):
        model.train()
        for x, edge_index, labels, seed_count in graphs[
            "train_completo"
        ].iter_batches(GNN_BATCH_SIZE, shuffle=True, fanouts=GNN_FANOUTS):
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities = model(x, edge_index)[:seed_count]
            loss = _weighted_bce(criterion, probabilities, labels, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model, device, graphs["train_completo"]


def _nn_candidate_search(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    positive_weight: float,
    deadline: float,
    variant: str,
    primeiro_passe: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], int, bool, dict[str, Any]]:
    configs = _candidate_configs(model_name, primeiro_passe)
    prepared = (
        _lstm_loaders(splits, feature_names)
        if model_name == "lstm"
        else _prepare_gnn_graphs(splits, feature_names)
    )
    best_result = None
    best_candidate = None
    completed = 0
    reduced = False
    rows = _grid_rows()
    existing = {
        (row["model"], row["variant"], row["config"]): row
        for row in rows
    }
    for config in configs:
        if time.perf_counter() >= deadline:
            reduced = True
            break
        checkpoint_path = _checkpoint_path(model_name, variant, config)
        row_key = (model_name, variant, _config_key(config))
        if checkpoint_path.exists():
            candidate = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if row_key not in existing:
                _append_grid_row(
                    _checkpoint_grid_row(
                        model_name, variant, config, candidate
                    )
                )
                existing[row_key] = {}
        else:
            if row_key in existing:
                if existing[row_key].get("status") == "skipped_memory":
                    reduced = True
                    break
                raise RuntimeError(
                    f"Checkpoint ausente para configuração já registrada: {row_key}"
                )
            ram_before = _ram_snapshot()
            vram_before = _vram_snapshot()
            if ram_before["available_gib"] < 2:
                _record_memory_skip(
                    model_name,
                    variant,
                    config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    0.0,
                    "RAM disponível abaixo de 2 GiB antes do ajuste.",
                )
                reduced = True
                print(
                    f"Grade {model_name} reduzida: RAM disponível abaixo de 2 GiB.",
                    flush=True,
                )
                break
            started = time.perf_counter()
            _reset_vram_peak()
            try:
                with _PeakRSSMonitor() as memory_monitor:
                    candidate = (
                        _train_lstm_candidate(
                            prepared, feature_names, config, positive_weight
                        )
                        if model_name == "lstm"
                        else _train_gnn_candidate(
                            prepared, feature_names, config, positive_weight
                        )
                    )
            except (MemoryError, torch.cuda.OutOfMemoryError) as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _record_memory_skip(
                    model_name,
                    variant,
                    config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    time.perf_counter() - started,
                    f"{type(error).__name__}: {error}",
                )
                reduced = True
                print(
                    f"Grade {model_name} reduzida após {type(error).__name__}.",
                    flush=True,
                )
                break
            candidate["training_seconds"] = time.perf_counter() - started
            candidate["peak_ram_gib"] = memory_monitor.peak_gib
            candidate["ram_before"] = ram_before
            candidate["vram_before"] = vram_before
            candidate["peak_vram_gib"] = _peak_vram_gib()
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(candidate, temporary_path)
            temporary_path.replace(checkpoint_path)
            _append_grid_row(
                _checkpoint_grid_row(
                    model_name, variant, config, candidate
                )
            )
            existing[row_key] = {}
        metrics = candidate["validation_metrics"]
        completed += 1
        result = {
            "config": config,
            "metrics": metrics,
            "best_epoch": candidate["best_epoch"],
        }
        if best_result is None or (
            metrics["pr_auc"], metrics["f1"]
        ) > (
            best_result["metrics"]["pr_auc"],
            best_result["metrics"]["f1"],
        ):
            best_result = result
            best_candidate = candidate
        print(
            f"  {model_name.upper()} {completed:02d}/{len(configs)}: {config} | "
            f"val PR-AUC={metrics['pr_auc']:.4f} epoch={candidate['best_epoch']} "
            f"| treino={candidate['training_seconds']:.1f}s "
            f"| RAM início={candidate['ram_before']['free_gib']:.2f} GiB "
            f"| RAM pico={candidate['peak_ram_gib']:.2f} GiB "
            f"| VRAM pico={candidate.get('peak_vram_gib', 0.0):.2f} GiB",
            flush=True,
        )
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best_result is None or best_candidate is None:
        raise TimeoutError(f"O orçamento da busca de {model_name} expirou.")
    return best_candidate, best_result, completed, reduced, prepared


def _run_neural_model(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    deadline: float,
    fixed_config: dict[str, Any] | None = None,
    variant: str = "main",
) -> dict[str, Any]:
    started = time.perf_counter()
    cached_result = _load_final_checkpoint(model_name, variant)
    if cached_result is not None:
        print(
            f"Retomando resultado final de {model_name}/{variant}; "
            "treino e teste já concluídos.",
            flush=True,
        )
        return cached_result
    is_ablation = fixed_config is not None
    y_train = splits["train"]["isFraud"].to_numpy(dtype=np.int64)
    positive_weight = _class_positive_weight(y_train)
    if fixed_config is None:
        best, selection, candidates, reduced, prepared = _nn_candidate_search(
            model_name,
            splits,
            feature_names,
            positive_weight,
            min(deadline, time.perf_counter() + SEARCH_TIME_LIMIT_SECONDS),
            variant,
        )
    else:
        prepared = (
            _lstm_loaders(splits, feature_names)
            if model_name == "lstm"
            else _prepare_gnn_graphs(splits, feature_names)
        )
        ram_before = _ram_snapshot()
        vram_before = _vram_snapshot()
        checkpoint_path = _checkpoint_path(model_name, variant, fixed_config)
        row_key = (model_name, variant, _config_key(fixed_config))
        existing = {
            (row["model"], row["variant"], row["config"]): row
            for row in _grid_rows()
        }
        if checkpoint_path.exists():
            best = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if row_key not in existing:
                _append_grid_row(
                    _checkpoint_grid_row(
                        model_name, variant, fixed_config, best
                    )
                )
        else:
            if row_key in existing:
                raise RuntimeError(
                    f"Checkpoint ausente para configuração já registrada: {row_key}"
                )
            if ram_before["available_gib"] < 2:
                _record_memory_skip(
                    model_name,
                    variant,
                    fixed_config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    0.0,
                    "RAM disponível abaixo de 2 GiB antes do ajuste.",
                )
                raise MemoryError(
                    f"RAM disponível insuficiente para {model_name}/{variant}."
                )
            candidate_started = time.perf_counter()
            _reset_vram_peak()
            with _PeakRSSMonitor() as memory_monitor:
                best = (
                    _train_lstm_candidate(
                        prepared, feature_names, fixed_config, positive_weight
                    )
                    if model_name == "lstm"
                    else _train_gnn_candidate(
                        prepared, feature_names, fixed_config, positive_weight
                    )
                )
            best["training_seconds"] = time.perf_counter() - candidate_started
            best["peak_ram_gib"] = memory_monitor.peak_gib
            best["ram_before"] = ram_before
            best["vram_before"] = vram_before
            best["peak_vram_gib"] = _peak_vram_gib()
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(best, temporary_path)
            temporary_path.replace(checkpoint_path)
            _append_grid_row(
                _checkpoint_grid_row(
                    model_name, variant, fixed_config, best
                )
            )
        selection = {
            "config": fixed_config,
            "metrics": best["validation_metrics"],
            "best_epoch": best["best_epoch"],
        }
        candidates = 1
        reduced = False
    config = selection["config"]
    best_epoch = best["best_epoch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_fit_ram_before = _ram_snapshot()
    final_fit_vram_before = _vram_snapshot()
    final_fit_started = time.perf_counter()
    _reset_vram_peak()

    with _PeakRSSMonitor() as final_fit_memory:
        if model_name == "lstm":
            selection_model = LSTMFraudDetector(
                input_size=len(feature_names),
                hidden_size=config["hidden_size"],
                dropout=config["dropout"],
                num_layers=config["num_layers"],
            ).to(device)
            selection_model.load_state_dict(best["state_dict"])
            criterion = nn.BCELoss(reduction="none")
            _, y_train_pred, train_scores = _evaluate_lstm(
                selection_model,
                prepared["train"],
                device,
                positive_weight,
                criterion,
            )
            _, y_val, val_scores = _evaluate_lstm(
                selection_model,
                prepared["val"],
                device,
                positive_weight,
                criterion,
            )
            decision_threshold = _best_f1_threshold(y_val, val_scores)
            metrics = {
                "train": _metrics(
                    y_train_pred, train_scores, decision_threshold
                ),
                "validation": _metrics(y_val, val_scores, decision_threshold),
            }
            final_model, final_device = _refit_lstm(
                prepared, feature_names, config, positive_weight, best_epoch
            )
            test_context = pd.concat(
                (
                    splits["context"],
                    splits["test"].drop(columns=["isFraud"]),
                )
            )
            final_loader = DataLoader(
                TransactionSequenceDataset(
                    splits["test"],
                    seq_length=LSTM_SEQUENCE_LENGTH,
                    feature_cols=feature_names,
                    context_df=test_context,
                ),
                batch_size=LSTM_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            _, y_test, test_scores = _evaluate_lstm(
                final_model,
                final_loader,
                final_device,
                positive_weight,
                criterion,
            )
            del test_context, final_loader
            metrics["test"] = _metrics(
                y_test, test_scores, decision_threshold
            )
            metadata = {
                "feature_names": feature_names,
                "seq_length": LSTM_SEQUENCE_LENGTH,
                "seed": SEED,
                "training_step_max": 599,
                "selected_config": config,
                "best_epoch": best_epoch,
                "decision_threshold": decision_threshold,
                "threshold_selection": "validation_f1",
                "feature_policy_version": FEATURE_POLICY_VERSION,
            }
            model_path = MODEL_DIR / (
                "lstm_model_ablation.pt"
                if "amount_to_balance_ratio" not in feature_names
                else "lstm_model.pt"
            )
            salvar_lstm(final_model, str(model_path), metadata=metadata)
        else:
            selection_model = GNNFraudDetector(
                in_channels=len(feature_names),
                hidden_channels=config["hidden_dim"],
                dropout=config["dropout"],
                num_layers=config["num_layers"],
            ).to(device)
            selection_model.load_state_dict(best["state_dict"])
            _, y_train_pred, train_scores = _graph_scores(
                selection_model, prepared["train"], device, positive_weight
            )
            _, y_val, val_scores = _graph_scores(
                selection_model, prepared["val"], device, positive_weight
            )
            decision_threshold = _best_f1_threshold(y_val, val_scores)
            metrics = {
                "train": _metrics(
                    y_train_pred, train_scores, decision_threshold
                ),
                "validation": _metrics(y_val, val_scores, decision_threshold),
            }
            final_model, final_device, final_graph = _refit_gnn(
                prepared, feature_names, config, positive_weight, best_epoch
            )
            test_context = pd.concat(
                (
                    splits["context"],
                    splits["test"].drop(columns=["isFraud"]),
                )
            )
            test_dataset = TransactionMiniBatchDataset(
                splits["test"],
                feature_names,
                context_df=test_context,
            )
            _, y_test, test_scores = _graph_scores(
                final_model, test_dataset, final_device, positive_weight
            )
            del test_context
            metrics["test"] = _metrics(
                y_test, test_scores, decision_threshold
            )
            metadata = {
                "feature_names": feature_names,
                "seed": SEED,
                "training_step_max": 599,
                "selected_config": config,
                "best_epoch": best_epoch,
                "decision_threshold": decision_threshold,
                "threshold_selection": "validation_f1",
                "feature_policy_version": FEATURE_POLICY_VERSION,
            }
            model_path = MODEL_DIR / (
                "gnn_model_ablation.pt"
                if "amount_to_balance_ratio" not in feature_names
                else "gnn_model.pt"
            )
            salvar_gnn(final_model, None, str(model_path), metadata)
            del final_graph, test_dataset, final_model

    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    result = {
        "config": config,
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "candidate_count": candidates,
        "grid_total": 1 if is_ablation else 16,
        "initial_reduced_grid_total": 8,
        "expanded_grid_total": 16,
        "ablation_reuses_selected_main_config": is_ablation,
        "grid_reduced_due_to_time": reduced,
        "elapsed_seconds": elapsed,
        "final_fit_seconds": time.perf_counter() - final_fit_started,
        "ram_before_final_fit_gib": final_fit_ram_before["free_gib"],
        "ram_available_before_final_fit_gib": final_fit_ram_before[
            "available_gib"
        ],
        "vram_free_before_final_fit_gib": final_fit_vram_before["free_gib"],
        "peak_vram_gib": max(
            _peak_vram_gib(),
            max(
                (
                    float(row.get("peak_vram_gib") or 0.0)
                    for row in _grid_rows()
                    if row["model"] == model_name
                    and row["variant"] == variant
                ),
                default=0.0,
            ),
        ),
        "peak_ram_gib": max(
            final_fit_memory.peak_gib,
            max(
                (
                    float(row["peak_ram_gib"])
                    for row in _grid_rows()
                    if row["model"] == model_name
                    and row["variant"] == variant
                ),
                default=0.0,
            ),
        ),
        "final_fit_device": str(final_device),
        "decision_threshold": decision_threshold,
        "threshold_selection": "validation_f1",
        "within_30_min": elapsed <= MODEL_TIME_LIMIT_SECONDS,
        "model_path": str(model_path),
        "positive_class_weight_from_train_core": positive_weight,
    }
    _save_final_checkpoint(model_name, variant, result)
    return result


def _run_isolation_forest(
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    deadline: float,
    fixed_config: dict[str, Any] | None = None,
    variant: str = "main",
) -> dict[str, Any]:
    started = time.perf_counter()
    cached_result = _load_final_checkpoint("isolation_forest", variant)
    if cached_result is not None:
        print(
            f"Retomando resultado final de isolation_forest/{variant}; "
            "treino e teste já concluídos.",
            flush=True,
        )
        return cached_result
    is_ablation = fixed_config is not None
    matrices = {
        key: _feature_matrix(splits[key], feature_names)
        for key in ("train", "val", "train_completo")
    }
    labels = {
        key: splits[key]["isFraud"].to_numpy(dtype=np.int64)
        for key in ("train", "val", "train_completo")
    }
    if fixed_config is None:
        selection_model, selection, completed, reduced = _if_candidate_search(
            matrices["train"],
            matrices["val"],
            labels["val"],
            min(deadline, time.perf_counter() + SEARCH_TIME_LIMIT_SECONDS),
            variant,
        )
    else:
        ram_before = _ram_snapshot()
        vram_before = _vram_snapshot()
        checkpoint_path = _checkpoint_path(
            "isolation_forest", variant, fixed_config
        )
        row_key = ("isolation_forest", variant, _config_key(fixed_config))
        existing = {
            (row["model"], row["variant"], row["config"]): row
            for row in _grid_rows()
        }
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("dataset_signature") != DATASET_SIGNATURE
                or checkpoint.get("feature_policy") != FEATURE_POLICY_VERSION
            ):
                raise RuntimeError(
                    f"Checkpoint IF incompatível: {checkpoint_path}"
                )
            if row_key not in existing:
                _append_grid_row(
                    _checkpoint_grid_row(
                        "isolation_forest", variant, fixed_config, checkpoint
                    )
                )
        elif row_key in existing and existing[row_key].get("status") != "skipped_memory":
            row = existing[row_key]
            checkpoint = {
                "validation_metrics": {
                    "pr_auc": float(row["validation_pr_auc"]),
                    "f1": float(row["validation_f1"]),
                    "threshold": 0.5,
                },
                "training_seconds": float(row["training_seconds"]),
                "peak_ram_gib": float(row["peak_ram_gib"]),
                "ram_before": {
                    "free_gib": float(row["ram_free_before_gib"]),
                    "available_gib": float(row["ram_available_before_gib"]),
                },
                "vram_before": {
                    "free_gib": float(row["vram_free_before_gib"] or 0.0),
                    "total_gib": 0.0,
                },
                "peak_vram_gib": float(row["peak_vram_gib"] or 0.0),
            }
        else:
            if row_key in existing:
                raise MemoryError(
                    f"Configuração IF/{variant} já foi limitada por memória."
                )
            if ram_before["available_gib"] < 2:
                _record_memory_skip(
                    "isolation_forest",
                    variant,
                    fixed_config,
                    ram_before,
                    vram_before,
                    _current_rss_gib(),
                    0.0,
                    "RAM disponível abaixo de 2 GiB antes do ajuste.",
                )
                raise MemoryError(
                    f"RAM disponível insuficiente para isolation_forest/{variant}."
                )
            candidate_started = time.perf_counter()
            _reset_vram_peak()
            candidate_model = IsolationForest(
                **fixed_config, random_state=SEED, n_jobs=4
            )
            with _PeakRSSMonitor() as memory_monitor:
                candidate_model.fit(matrices["train"])
                validation = _metrics(
                    labels["val"],
                    _if_scores(candidate_model, matrices["val"]),
                )
            checkpoint = {
                "dataset_signature": DATASET_SIGNATURE,
                "feature_policy": FEATURE_POLICY_VERSION,
                "validation_metrics": validation,
                "training_seconds": time.perf_counter() - candidate_started,
                "peak_ram_gib": memory_monitor.peak_gib,
                "ram_before": ram_before,
                "vram_before": vram_before,
                "peak_vram_gib": _peak_vram_gib(),
            }
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8"
            )
            temporary_path.replace(checkpoint_path)
            _append_grid_row(
                _checkpoint_grid_row(
                    "isolation_forest", variant, fixed_config, checkpoint
                )
            )
            del candidate_model
        validation = checkpoint["validation_metrics"]
        selection = {"config": fixed_config, "metrics": validation}
        completed = 1
        reduced = False
        refit_ram_before = _ram_snapshot()
        refit_vram_before = _vram_snapshot()
        if refit_ram_before["available_gib"] < 2:
            _record_memory_skip(
                "isolation_forest",
                f"{variant}_selected_refit",
                fixed_config,
                refit_ram_before,
                refit_vram_before,
                _current_rss_gib(),
                0.0,
                "RAM disponível abaixo de 2 GiB antes do refit selecionado.",
            )
            raise MemoryError(
                f"RAM insuficiente para refitar isolation_forest/{variant}."
            )
        refit_started = time.perf_counter()
        _reset_vram_peak()
        selection_model = IsolationForest(
            **fixed_config, random_state=SEED, n_jobs=4
        )
        with _PeakRSSMonitor() as refit_memory:
            selection_model.fit(matrices["train"])
        selection["selection_refit"] = {
            "training_seconds": time.perf_counter() - refit_started,
            "peak_ram_gib": refit_memory.peak_gib,
            "ram_before": refit_ram_before,
            "vram_before": refit_vram_before,
            "peak_vram_gib": _peak_vram_gib(),
        }
    train_scores = _if_scores(selection_model, matrices["train"])
    val_scores = _if_scores(selection_model, matrices["val"])
    decision_threshold = _best_f1_threshold(labels["val"], val_scores)
    metrics = {
        "train": _metrics(labels["train"], train_scores, decision_threshold),
        "validation": _metrics(labels["val"], val_scores, decision_threshold),
    }

    final_fit_ram_before = _ram_snapshot()
    final_fit_vram_before = _vram_snapshot()
    final_fit_started = time.perf_counter()
    _reset_vram_peak()
    final_model = IsolationForest(
        **selection["config"],
        random_state=SEED,
        n_jobs=4,
    )
    with _PeakRSSMonitor() as final_fit_memory:
        final_model.fit(matrices["train_completo"])
    test_features = _feature_matrix(splits["test"], feature_names)
    test_labels = splits["test"]["isFraud"].to_numpy(dtype=np.int64)
    test_scores = _if_scores(final_model, test_features)
    metrics["test"] = _metrics(
        test_labels, test_scores, decision_threshold
    )
    final_model.training_step_max_ = 599
    final_model.training_seed_ = SEED
    final_model.decision_threshold_ = decision_threshold
    final_model.threshold_selection_ = "validation_f1"
    final_model.feature_policy_version_ = FEATURE_POLICY_VERSION
    model_path = MODEL_DIR / (
        "isolation_forest_ablation.pkl"
        if "amount_to_balance_ratio" not in feature_names
        else "isolation_forest.pkl"
    )
    with model_path.open("wb") as file:
        pickle.dump(final_model, file)
    elapsed = time.perf_counter() - started
    result = {
        "config": selection["config"],
        "feature_names": feature_names,
        "metrics": metrics,
        "candidate_count": completed,
        "grid_total": 1 if is_ablation else 27,
        "initial_reduced_grid_total": 9,
        "expanded_grid_total": 27,
        "ablation_reuses_selected_main_config": is_ablation,
        "grid_reduced_due_to_time": reduced,
        "elapsed_seconds": elapsed,
        "final_fit_seconds": time.perf_counter() - final_fit_started,
        "ram_before_final_fit_gib": final_fit_ram_before["free_gib"],
        "ram_available_before_final_fit_gib": final_fit_ram_before[
            "available_gib"
        ],
        "vram_free_before_final_fit_gib": final_fit_vram_before["free_gib"],
        "peak_vram_gib": max(
            _peak_vram_gib(),
            max(
                (
                    float(row.get("peak_vram_gib") or 0.0)
                    for row in _grid_rows()
                    if row["model"] == "isolation_forest"
                    and row["variant"] == variant
                ),
                default=0.0,
            ),
        ),
        "peak_ram_gib": max(
            final_fit_memory.peak_gib,
            max(
                (
                    float(row["peak_ram_gib"])
                    for row in _grid_rows()
                    if row["model"] == "isolation_forest"
                    and row["variant"] == variant
                ),
                default=0.0,
            ),
        ),
        "within_30_min": elapsed <= MODEL_TIME_LIMIT_SECONDS,
        "model_path": str(model_path),
        "selection_validation_pr_auc": selection["metrics"]["pr_auc"],
        "decision_threshold": decision_threshold,
        "threshold_selection": "validation_f1",
        "contamination_selection": "validation_f1",
        "selection_refit": selection.get("selection_refit"),
        "class_weights": "não aplicável: Isolation Forest é não supervisionado",
        "early_stopping": "não aplicável: estimador de árvores não incremental",
    }
    del matrices, labels, test_features, test_labels, test_scores, selection_model
    gc.collect()
    _save_final_checkpoint("isolation_forest", variant, result)
    return result


def _run_smoke_test(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    feature_names: list[str],
    persist_result: bool = True,
) -> dict[str, Any]:
    configs = {
        "isolation_forest": {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.056,
        },
        "lstm": {
            "hidden_size": 32,
            "dropout": 0.1,
            "lr": 1e-3,
            "num_layers": 2,
        },
        "gnn": {
            "hidden_dim": 32,
            "dropout": 0.1,
            "lr": 1e-3,
            "num_layers": 2,
        },
    }
    config = configs[model_name]
    checkpoint_path = _checkpoint_path(model_name, "smoke", config)
    row_key = (model_name, "smoke", _config_key(config))
    existing = {}
    if persist_result:
        existing = {
            (row["model"], row["variant"], row["config"]): row
            for row in _grid_rows()
        }

    if persist_result and checkpoint_path.exists():
        if model_name == "isolation_forest":
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        if row_key not in existing:
            _append_grid_row(
                _checkpoint_grid_row(
                    model_name, "smoke", config, checkpoint
                )
            )
        return {
            "config": config,
            "metrics": checkpoint["validation_metrics"],
            "training_seconds": checkpoint["training_seconds"],
            "peak_ram_gib": checkpoint["peak_ram_gib"],
            "peak_vram_gib": checkpoint.get("peak_vram_gib", 0.0),
            "resumed": True,
            "ram_before": checkpoint["ram_before"],
            "vram_before": checkpoint.get("vram_before", {}),
        }
    if row_key in existing:
        if existing[row_key].get("status") == "skipped_memory":
            return {"config": config, "status": "skipped_memory"}
        raise RuntimeError(
            f"Checkpoint smoke ausente para configuração registrada: {row_key}"
        )

    ram_before = _ram_snapshot()
    vram_before = _vram_snapshot()
    if ram_before["available_gib"] < 2:
        _record_memory_skip(
            model_name,
            "smoke",
            config,
            ram_before,
            vram_before,
            _current_rss_gib(),
            0.0,
            "RAM disponível abaixo de 2 GiB antes do smoke test.",
            record_grid=persist_result,
        )
        return {
            "config": config,
            "status": "skipped_memory",
            "error": "RAM disponível abaixo de 2 GiB antes do smoke test.",
            "ram_before": ram_before,
            "vram_before": vram_before,
        }

    started = time.perf_counter()
    _reset_vram_peak()
    try:
        if model_name == "isolation_forest":
            x_train = _feature_matrix(splits["train"], feature_names)
            x_val = _feature_matrix(splits["val"], feature_names)
            y_val = splits["val"]["isFraud"].to_numpy(dtype=np.int64)
            model = IsolationForest(**config, random_state=SEED, n_jobs=4)
            with _PeakRSSMonitor() as memory_monitor:
                model.fit(x_train)
                validation_metrics = _metrics(y_val, _if_scores(model, x_val))
            del x_train, x_val, model
            checkpoint = {}
        else:
            positive_weight = _class_positive_weight(
                splits["train"]["isFraud"].to_numpy(dtype=np.int64)
            )
            prepared = (
                _lstm_loaders(splits, feature_names)
                if model_name == "lstm"
                else _prepare_gnn_graphs(splits, feature_names)
            )
            with _PeakRSSMonitor() as memory_monitor:
                checkpoint = (
                    _train_lstm_candidate(
                        prepared,
                        feature_names,
                        config,
                        positive_weight,
                        max_epochs=1,
                    )
                    if model_name == "lstm"
                    else _train_gnn_candidate(
                        prepared,
                        feature_names,
                        config,
                        positive_weight,
                        max_epochs=1,
                    )
                )
            validation_metrics = checkpoint["validation_metrics"]
            del prepared
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        checkpoint.update(
            {
                "dataset_signature": DATASET_SIGNATURE,
                "feature_policy": FEATURE_POLICY_VERSION,
                "validation_metrics": validation_metrics,
                "training_seconds": time.perf_counter() - started,
                "peak_ram_gib": memory_monitor.peak_gib,
                "ram_before": ram_before,
                "vram_before": vram_before,
                "peak_vram_gib": _peak_vram_gib(),
            }
        )
    except (MemoryError, torch.cuda.OutOfMemoryError) as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _record_memory_skip(
            model_name,
            "smoke",
            config,
            ram_before,
            vram_before,
            _current_rss_gib(),
            time.perf_counter() - started,
            f"{type(error).__name__}: {error}",
            record_grid=persist_result,
        )
        return {
            "config": config,
            "status": "skipped_memory",
            "error": f"{type(error).__name__}: {error}",
            "ram_before": ram_before,
            "vram_before": vram_before,
        }

    if persist_result:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(".tmp")
        if model_name == "isolation_forest":
            temporary_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8"
            )
        else:
            torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
        _append_grid_row(
            _checkpoint_grid_row(model_name, "smoke", config, checkpoint)
        )
    return {
        "config": config,
        "metrics": validation_metrics,
        "training_seconds": checkpoint["training_seconds"],
        "peak_ram_gib": checkpoint["peak_ram_gib"],
        "peak_vram_gib": checkpoint["peak_vram_gib"],
        "resumed": False,
        "ram_before": ram_before,
        "vram_before": vram_before,
    }


def _archive_pre_policy_outputs() -> dict[str, Any]:
    archived = []
    diagnostic_report: dict[str, Any] = {
        "classification": "DIAGNÓSTICO — política pré-execução não aplicada",
        "models": {},
        "ablation": {},
    }
    archive_report = False

    if REPORT_PATH.exists():
        try:
            previous_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_report = {}
        if previous_report.get("feature_policy_version") != FEATURE_POLICY_VERSION:
            archive_report = True
            for name, result in previous_report.get("models", {}).items():
                diagnostic_report["models"][name] = {
                    "metrics": result.get("metrics", {}),
                    "feature_names": result.get("feature_names", []),
                }
            for name, result in previous_report.get("ablation", {}).items():
                diagnostic_report["ablation"][name] = {
                    "metrics": result.get("metrics", {}),
                    "feature_names": result.get("feature_names", []),
                }
    if archive_report:
        destination = REPORT_PATH.with_name(
            f"{REPORT_PATH.stem}_diagnostico_pre_politica{REPORT_PATH.suffix}"
        )
        if destination.exists():
            raise FileExistsError(
                f"Não vou sobrescrever o arquivo diagnóstico: {destination}"
            )
        REPORT_PATH.replace(destination)
        archived.append(destination)

    final_paths = {
        _final_checkpoint_path(model_name, variant): (model_name, variant)
        for model_name in ("isolation_forest", "lstm", "gnn")
        for variant in ("main", "ablation")
    }
    for path in [GRID_CSV_PATH, *final_paths]:
        if not path.exists():
            continue
        if path == GRID_CSV_PATH:
            rows = _grid_rows_unchecked(path)
            if rows and all(
                row.get("feature_policy") == FEATURE_POLICY_VERSION
                and row.get("dataset_signature") == DATASET_SIGNATURE
                for row in rows
            ):
                continue
        else:
            try:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                checkpoint = {}
            if checkpoint.get("dataset_signature") == DATASET_SIGNATURE:
                continue
            model_name, variant = final_paths[path]
            result = checkpoint.get("result", {})
            if result:
                bucket = (
                    "models" if variant == "main" else "ablation"
                )
                diagnostic_report[bucket][model_name] = {
                    "metrics": result.get("metrics", {}),
                    "feature_names": result.get("feature_names", []),
                    "classification": "DIAGNÓSTICO",
                }
        destination = path.with_name(
            f"{path.stem}_diagnostico_pre_politica{path.suffix}"
        )
        if destination.exists():
            raise FileExistsError(
                f"Não vou sobrescrever o arquivo diagnóstico: {destination}"
            )
        path.replace(destination)
        archived.append(destination)

    if CHECKPOINT_DIR.exists():
        for path in CHECKPOINT_DIR.iterdir():
            if (
                not path.is_file()
                or path.suffix not in {".json", ".pt"}
                or "_final." in path.name
                or "_diagnostico_pre_politica" in path.name
            ):
                continue
            checkpoint = {}
            try:
                if path.suffix == ".json":
                    checkpoint = json.loads(path.read_text(encoding="utf-8"))
                elif path.stat().st_size <= 64 * 1024 * 1024:
                    checkpoint = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                EOFError,
                pickle.UnpicklingError,
                RuntimeError,
                ValueError,
            ):
                checkpoint = {}
            if not isinstance(checkpoint, dict):
                checkpoint = {}
            if (
                checkpoint.get("dataset_signature") == DATASET_SIGNATURE
                and checkpoint.get("feature_policy") == FEATURE_POLICY_VERSION
            ):
                continue
            destination = path.with_name(
                f"{path.stem}_diagnostico_pre_politica{path.suffix}"
            )
            if destination.exists():
                raise FileExistsError(
                    f"Não vou sobrescrever o checkpoint diagnóstico: {destination}"
                )
            path.replace(destination)
            archived.append(destination)

    if archived:
        model_backup_dir = CHECKPOINT_DIR / "modelos_diagnostico_pre_politica"
        model_backups = []
        if MODEL_DIR.exists():
            for model_path in MODEL_DIR.iterdir():
                if model_path.suffix not in {".pkl", ".pt", ".pth"}:
                    continue
                model_backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = model_backup_dir / model_path.name
                if not backup_path.exists():
                    shutil.copy2(model_path, backup_path)
                model_backups.append(str(backup_path))
        diagnostic_report["archived_artifacts"] = [
            str(path) for path in archived
        ]
        diagnostic_report["model_artifact_backups"] = model_backups
        print(
            "Artefatos da política anterior preservados como DIAGNÓSTICO: "
            + ", ".join(str(path) for path in archived),
            flush=True,
        )
    return diagnostic_report


def _grid_rows_unchecked(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as grid_file:
        return list(csv.DictReader(grid_file))


def main() -> dict[str, Any]:
    """Seleciona por validação e faz uma única avaliação final no teste."""
    global DATASET_SIGNATURE
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dados processados não encontrados: {PROCESSED_DATA_PATH}. "
            "Execute python -m src.preprocessamento.carregar_dataset."
        )
    if not TORCH_AVAILABLE or not TORCH_GEO_AVAILABLE:
        raise RuntimeError("O retreino exige PyTorch e PyTorch Geometric.")

    _set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stat = PROCESSED_DATA_PATH.stat()
    feature_signature = ",".join(
        (*IF_FEATURE_NAMES, *LSTM_FEATURE_NAMES, *GNN_FEATURE_NAMES)
    )
    DATASET_SIGNATURE = (
        f"{dataset_stat.st_size}:{dataset_stat.st_mtime_ns}:seed-{SEED}:"
        f"{FEATURE_POLICY_VERSION}:"
        f"{hashlib.sha256(feature_signature.encode()).hexdigest()}"
    )
    dataframe = pd.read_parquet(PROCESSED_DATA_PATH)
    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "isFraud",
    ]
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Parquet processado sem colunas necessárias ao treino: "
            + ", ".join(sorted(missing_columns))
        )
    dataframe = dataframe[required_columns].copy()
    splits = temporal_split(dataframe)
    generate_feature_stats(splits["train_completo"], FEATURE_STATS_PATH)
    gc.collect()
    if not splits["train_completo"]["step"].lt(600).all():
        raise ValueError("Treino final contém step >= 600.")
    if not splits["train"]["step"].lt(550).all():
        raise ValueError("Class weights e seleção devem usar step < 550.")
    diagnostic_results = _archive_pre_policy_outputs()
    splits["context"] = dataframe.loc[
        dataframe["step"] < 600
    ].sort_values("step", kind="stable").drop(columns=["isFraud"])
    del dataframe
    gc.collect()

    print(
        "Split temporal D3: "
        f"treino={len(splits['train']):,}, validação={len(splits['val']):,}, "
        f"treino completo={len(splits['train_completo']):,}, "
        f"teste={len(splits['test']):,}".replace(",", "."),
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed={SEED}; dispositivo={device}; seleção exclusivamente na validação.")

    output: dict[str, Any] = {
        "seed": SEED,
        "split": {
            key: {
                "rows": int(len(frame)),
                "min_step": int(frame["step"].min()),
                "max_step": int(frame["step"].max()),
                **(
                    {"frauds": int(frame["isFraud"].sum())}
                    if key != "test"
                    else {}
                ),
            }
            for key, frame in splits.items()
            if key != "context"
        },
        "models": {},
        "ablation": {},
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "diagnostic_results": diagnostic_results,
        "training_policy": {
            "feature_policy_version": FEATURE_POLICY_VERSION,
            "prediction_moment": "before transaction execution",
            "forbidden_features": [
                "emptied_origin",
                "balance_mismatch_orig",
                "balance_mismatch_dest",
            ],
            "allowed_balance_ratio": "amount / (abs(oldbalanceOrg) + 1)",
            "lstm_history": "same account and step strictly less than target",
            "gnn_edges": "strictly past account events only",
            "class_weights_from": "step < 550",
            "validation": "step 550..599",
            "final_fit": "step < 600",
            "test_evaluated_once_after_final_fit": True,
            "configuration_selection": "validation PR-AUC",
            "threshold_selection": "validation F1",
            "isolation_forest_contamination_selection": "validation F1",
            "grid_time_limit_seconds": MODEL_TIME_LIMIT_SECONDS,
            "search_time_limit_seconds": SEARCH_TIME_LIMIT_SECONDS,
            "smoke_test_before_grid": True,
            "smoke_test_epochs": {"lstm": 1, "gnn": 1},
            "gemini_calls_during_training": False,
            "initial_reduced_grid": {
                "isolation_forest": {
                    "n_estimators": [100, 200, 300],
                    "max_samples": ["auto"],
                    "contamination": [0.03, 0.056, 0.10],
                },
                "lstm": {
                    "hidden_size": [32, 64],
                    "dropout": [0.1, 0.3],
                    "lr": [0.001, 0.0005],
                    "num_layers": [2],
                },
                "gnn": {
                    "hidden_dim": [32, 64],
                    "dropout": [0.1, 0.3],
                    "lr": [0.001, 0.0005],
                    "num_layers": [2],
                },
            },
            "lstm": {
                "loss": "BCELoss",
                "optimizer": "Adam",
                "batch_size": LSTM_BATCH_SIZE,
                "sequence_windows_generated_on_demand": True,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": PATIENCE,
                "sequence_length": LSTM_SEQUENCE_LENGTH,
                "lr_scheduler": "ReduceLROnPlateau(val_loss)",
            },
            "gnn": {
                "loss": "BCELoss",
                "optimizer": "Adam",
                "max_epochs": GNN_EPOCHS,
                "early_stopping_patience": PATIENCE,
                "lr_scheduler": "ReduceLROnPlateau(val_loss)",
                "graph_minibatches": True,
                "batch_size": GNN_BATCH_SIZE,
                "fanouts": list(GNN_FANOUTS),
                "full_graph_materialized": False,
            },
            "isolation_forest": {
                "early_stopping": "not_supported_by_estimator",
                "class_weights": "not_applicable_unsupervised",
                "n_jobs": 4,
                "contamination_selection": "validation_f1",
            },
        },
    }
    validation_features = compute_if_features(splits["val"])
    if not isinstance(validation_features, pd.DataFrame):
        raise TypeError("O baseline exige features tabulares de validação.")
    baseline_scores = validation_features[
        "amount_to_balance_ratio"
    ].to_numpy(dtype=np.float32)
    baseline_labels = splits["val"]["isFraud"].to_numpy(dtype=np.int64)
    baseline_threshold = _best_f1_threshold(baseline_labels, baseline_scores)
    output["baseline_validation"] = {
        "rule": "score = amount_to_balance_ratio",
        "forbidden_features_used": [],
        "metrics": _metrics(
            baseline_labels, baseline_scores, baseline_threshold
        ),
        "threshold_selection": "validation_f1",
        "note": (
            "emptied_origin não foi incluído por usar saldo pós-transação "
            "e estar proibido pela política atual."
        ),
    }
    del validation_features, baseline_scores, baseline_labels
    gc.collect()

    model_specs = (
        ("isolation_forest", IF_FEATURE_NAMES),
        ("lstm", LSTM_FEATURE_NAMES),
        ("gnn", GNN_FEATURE_NAMES),
    )
    smoke_elapsed: dict[str, float] = {}
    ram_before_models: dict[str, dict[str, float]] = {}
    smoke_results: dict[str, dict[str, Any]] = {}
    for name, feature_names in model_specs:
        smoke_started = time.perf_counter()
        ram_before_model = _ram_snapshot()
        vram_before_model = _vram_snapshot()
        ram_before_models[name] = ram_before_model
        print(
            f"\nSmoke test {name} com {len(feature_names)} features... "
            f"RAM livre={ram_before_model['free_gib']:.2f} GiB; "
            f"RAM disponível={ram_before_model['available_gib']:.2f} GiB; "
            f"VRAM livre={vram_before_model['free_gib']:.2f} GiB.",
            flush=True,
        )
        smoke = _run_smoke_test(name, splits, feature_names)
        if smoke.get("status") == "skipped_memory":
            raise MemoryError(
                f"Smoke test de {name} interrompido por limitação de memória; "
                "o motivo foi registrado no grid."
            )
        smoke_results[name] = smoke
        smoke_elapsed[name] = time.perf_counter() - smoke_started

    for name, feature_names in model_specs:
        full_training_started = time.perf_counter()
        deadline = full_training_started + max(
            0.0, MODEL_TIME_LIMIT_SECONDS - smoke_elapsed[name]
        )
        print(
            f"\nTreinando {name} com {len(feature_names)} features após smoke "
            f"{smoke_results[name]['config']}.",
            flush=True,
        )
        result = (
            _run_isolation_forest(splits, feature_names, deadline)
            if name == "isolation_forest"
            else _run_neural_model(name, splits, feature_names, deadline)
        )
        result["smoke_test"] = smoke_results[name]
        result["ram_before_model_gib"] = ram_before_models[name]["free_gib"]
        result["ram_available_before_model_gib"] = ram_before_models[name][
            "available_gib"
        ]
        output["models"][name] = result
        ablation_features = [
            feature for feature in feature_names
            if feature != "amount_to_balance_ratio"
        ]
        print(
            f"Ablação {name}: retreino sem amount_to_balance_ratio...",
            flush=True,
        )
        ablation = (
            _run_isolation_forest(
                splits,
                ablation_features,
                deadline,
                fixed_config=result["config"],
                variant="ablation",
            )
            if name == "isolation_forest"
            else _run_neural_model(
                name,
                splits,
                ablation_features,
                deadline,
                fixed_config=result["config"],
                variant="ablation",
            )
        )
        output["ablation"][name] = ablation
        total_elapsed = smoke_elapsed[name] + (
            time.perf_counter() - full_training_started
        )
        output["models"][name]["elapsed_seconds_including_smoke_and_ablation"] = (
            total_elapsed
        )
        output["models"][name]["peak_ram_gib_including_ablation"] = max(
            output["models"][name]["peak_ram_gib"],
            output["ablation"][name]["peak_ram_gib"],
            smoke_results[name]["peak_ram_gib"],
        )
        output["models"][name]["peak_vram_gib_including_ablation"] = max(
            output["models"][name]["peak_vram_gib"],
            output["ablation"][name]["peak_vram_gib"],
            smoke_results[name]["peak_vram_gib"],
        )
        output["models"][name]["within_30_min_including_ablation"] = (
            total_elapsed
            <= MODEL_TIME_LIMIT_SECONDS
        )
        if (
            output["models"][name]["grid_reduced_due_to_time"]
            or output["ablation"][name]["grid_reduced_due_to_time"]
        ):
            output["models"][name]["limitation"] = (
                "Grade reduzida pelo limite de 30 minutos por tipo de modelo; "
                "o número de configurações testadas está registrado."
            )
        elif not output["models"][name]["within_30_min_including_ablation"]:
            output["models"][name]["limitation"] = (
                "O ciclo completo ultrapassou 30 minutos mesmo com a busca "
                "limitada a 15 minutos; tempos e configurações concluídas "
                "estão registrados."
            )
        if "limitation" in output["models"][name]:
            print(f"Limitação {name}: {output['models'][name]['limitation']}")
        gc.collect()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nMétricas por modelo (threshold selecionado na validação):")
    for name in output["models"]:
        for variant, result in (
            ("principal", output["models"][name]),
            ("ablação", output["ablation"][name]),
        ):
            print(
                f"{name} / {variant}: config={result['config']}; "
                f"treino={result['metrics']['train']}; "
                f"validação={result['metrics']['validation']}; "
                f"teste={result['metrics']['test']}; "
                f"tempo={result['elapsed_seconds']:.1f}s"
            )
    print(f"Relatório salvo em {REPORT_PATH}")
    return output


def smoke_only() -> dict[str, dict[str, Any]]:
    """Executa uma configuração por modelo sem ler/escrever qualquer grid."""
    global DATASET_SIGNATURE
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dados processados não encontrados: {PROCESSED_DATA_PATH}."
        )
    if not TORCH_AVAILABLE or not TORCH_GEO_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError("O teste rápido exige PyTorch, PyG e CUDA disponíveis.")

    _set_seed()
    dataset_stat = PROCESSED_DATA_PATH.stat()
    feature_signature = ",".join(
        (*IF_FEATURE_NAMES, *LSTM_FEATURE_NAMES, *GNN_FEATURE_NAMES)
    )
    DATASET_SIGNATURE = (
        f"{dataset_stat.st_size}:{dataset_stat.st_mtime_ns}:seed-{SEED}:"
        f"{FEATURE_POLICY_VERSION}:"
        f"{hashlib.sha256(feature_signature.encode()).hexdigest()}"
    )
    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "isFraud",
    ]
    dataframe = pd.read_parquet(
        PROCESSED_DATA_PATH, columns=required_columns
    )
    splits = temporal_split(dataframe)
    splits["context"] = dataframe.loc[
        dataframe["step"] < 600
    ].sort_values("step", kind="stable").drop(columns=["isFraud"])
    del dataframe
    gc.collect()

    model_specs = (
        ("isolation_forest", IF_FEATURE_NAMES),
        ("lstm", LSTM_FEATURE_NAMES),
        ("gnn", GNN_FEATURE_NAMES),
    )
    results: dict[str, dict[str, Any]] = {}
    print(
        f"Smoke-only: seed={SEED}; device=cuda; "
        f"split train={len(splits['train'])}, val={len(splits['val'])}; "
        f"sem avaliação do teste; sem checkpoint ou grid.",
        flush=True,
    )
    print(
        f"GNN: mini-batch={GNN_BATCH_SIZE}; fanouts={GNN_FANOUTS}; "
        "num_layers=2.",
        flush=True,
    )
    for model_name, feature_names in model_specs:
        ram_before = _ram_snapshot()
        vram_before = _vram_snapshot()
        print(
            f"\nSmoke {model_name}: config única; "
            f"RAM livre={ram_before['free_gib']:.2f} GiB; "
            f"RAM disponível={ram_before['available_gib']:.2f} GiB; "
            f"VRAM livre={vram_before['free_gib']:.2f} GiB.",
            flush=True,
        )
        result = _run_smoke_test(
            model_name, splits, feature_names, persist_result=False
        )
        result["ram_before_outer"] = ram_before
        result["vram_before_outer"] = vram_before
        results[model_name] = result
        print(
            f"Smoke resultado {model_name}: "
            f"{json.dumps(result, ensure_ascii=False, default=str)}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def grade_if_lstm_only(
    grid_path: Path, primeiro_passe: bool = False
) -> dict[str, dict[str, Any]]:
    """Run only IF/LSTM validation searches; never load the test split or GNN."""
    global DATASET_SIGNATURE, GRID_CSV_PATH
    requested_grid = grid_path.resolve()
    expected_grid = Path("reports/etapa2_grid_v2.csv").resolve()
    if requested_grid != expected_grid:
        raise ValueError("A grade seletiva exige reports/etapa2_grid_v2.csv.")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dados processados não encontrados: {PROCESSED_DATA_PATH}."
        )
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("A grade IF/LSTM exige PyTorch.")
    if not torch.cuda.is_available():
        raise RuntimeError("A grade seletiva exige CUDA disponível para a LSTM.")

    GRID_CSV_PATH = requested_grid
    _set_seed()
    dataset_stat = PROCESSED_DATA_PATH.stat()
    feature_signature = ",".join(
        (*IF_FEATURE_NAMES, *LSTM_FEATURE_NAMES, *GNN_FEATURE_NAMES)
    )
    DATASET_SIGNATURE = (
        f"{dataset_stat.st_size}:{dataset_stat.st_mtime_ns}:seed-{SEED}:"
        f"{FEATURE_POLICY_VERSION}:"
        f"{hashlib.sha256(feature_signature.encode()).hexdigest()}"
    )
    columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "isFraud",
    ]
    dataframe = pd.read_parquet(
        PROCESSED_DATA_PATH,
        columns=columns,
        filters=[("step", "<", 600)],
    )
    splits = temporal_split(dataframe)
    if len(splits["test"]) != 0:
        raise RuntimeError("A carga seletiva incluiu linhas do conjunto de teste.")
    splits["context"] = dataframe.sort_values(
        "step", kind="stable"
    ).drop(columns=["isFraud"])
    generate_feature_stats(splits["train_completo"], FEATURE_STATS_PATH)
    del dataframe
    gc.collect()

    device = torch.device("cuda")
    print(
        f"Grade seletiva IF/LSTM; device={device}; "
        "GNN=disabled; test=not_loaded; Gemini=not_called; "
        f"grid={GRID_CSV_PATH}; checkpoints={CHECKPOINT_DIR}",
        flush=True,
    )
    results: dict[str, dict[str, Any]] = {}

    if _ram_snapshot()["available_gib"] < 2:
        raise MemoryError("RAM disponível abaixo de 2 GiB antes da grade IF.")
    if_started = time.perf_counter()
    if_matrices = {
        key: _feature_matrix(splits[key], IF_FEATURE_NAMES)
        for key in ("train", "val")
    }
    if_labels = splits["val"]["isFraud"].to_numpy(dtype=np.int64)
    if_best, if_selection, if_count, if_reduced = _if_candidate_search(
        if_matrices["train"],
        if_matrices["val"],
        if_labels,
        if_started + min(SEARCH_TIME_LIMIT_SECONDS, MODEL_TIME_LIMIT_SECONDS),
        "main",
        primeiro_passe=primeiro_passe,
    )
    results["isolation_forest"] = {
        "selected_config": if_selection["config"],
        "validation_metrics": if_selection["metrics"],
        "configurations_tested": if_count,
        "grid_reduced_due_to_time": if_reduced,
        "elapsed_seconds": time.perf_counter() - if_started,
    }
    print(
        f"IF selecionado={if_selection['config']}; "
        f"val_PR_AUC={if_selection['metrics']['pr_auc']:.6f}; "
        f"val_F1={if_selection['metrics']['f1']:.6f}; "
        f"configurations={if_count}; reduced={if_reduced}",
        flush=True,
    )
    del if_best, if_matrices, if_labels
    gc.collect()

    if _ram_snapshot()["available_gib"] < 2:
        raise MemoryError("RAM disponível abaixo de 2 GiB antes da grade LSTM.")
    lstm_started = time.perf_counter()
    positive_weight = _class_positive_weight(
        splits["train"]["isFraud"].to_numpy(dtype=np.int64)
    )
    lstm_best, lstm_selection, lstm_count, lstm_reduced, prepared = (
        _nn_candidate_search(
            "lstm",
            splits,
            LSTM_FEATURE_NAMES,
            positive_weight,
            lstm_started
            + min(SEARCH_TIME_LIMIT_SECONDS, MODEL_TIME_LIMIT_SECONDS),
            "main",
            primeiro_passe=primeiro_passe,
        )
    )
    results["lstm"] = {
        "selected_config": lstm_selection["config"],
        "validation_metrics": lstm_selection["metrics"],
        "best_epoch": lstm_selection["best_epoch"],
        "configurations_tested": lstm_count,
        "grid_reduced_due_to_time": lstm_reduced,
        "elapsed_seconds": time.perf_counter() - lstm_started,
    }
    print(
        f"LSTM selecionado={lstm_selection['config']}; "
        f"val_PR_AUC={lstm_selection['metrics']['pr_auc']:.6f}; "
        f"val_F1={lstm_selection['metrics']['f1']:.6f}; "
        f"configurations={lstm_count}; reduced={lstm_reduced}",
        flush=True,
    )
    del lstm_best, prepared, splits
    gc.collect()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Executa apenas um teste por modelo, sem grade, checkpoints ou teste final.",
    )
    parser.add_argument(
        "--grade-if-lstm",
        action="store_true",
        help="Executa somente as buscas de validação do IF e da LSTM; não carrega teste nem GNN.",
    )
    parser.add_argument(
        "--primeiro-passe",
        action="store_true",
        help="Limita as grades IF/LSTM às configurações iniciais com 2 camadas.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=Path("reports/etapa2_grid_v2.csv"),
        help="Caminho do grid seletivo (deve ser etapa2_grid_v2.csv).",
    )
    arguments = parser.parse_args()
    if arguments.primeiro_passe and not arguments.grade_if_lstm:
        parser.error("--primeiro-passe exige --grade-if-lstm.")
    if arguments.smoke_only and arguments.grade_if_lstm:
        parser.error("--smoke-only e --grade-if-lstm são mutuamente exclusivos.")
    if arguments.grade_if_lstm:
        grade_if_lstm_only(
            arguments.grid_path, primeiro_passe=arguments.primeiro_passe
        )
    elif arguments.smoke_only:
        smoke_only()
    elif arguments.grid_path != Path("reports/etapa2_grid_v2.csv"):
        parser.error("--grid-path exige --grade-if-lstm.")
    else:
        main()
