"""Retreino da GNN com o amostrador corrigido (dois saltos, fanouts 10 e 5).

Usa só step < 600: sementes de treino em step < 550 (o contexto de vizinhança
é todo o passado até 599) e seleção de época na validação (550..599). Não lê
step >= 600. Por padrão usa todo o treino core; --sementes limita a amostra.
Salva o modelo em gnn_model.pt e guarda o modelo anterior em
gnn_model_etapa2_v1.pt antes de sobrescrever.

Uso: python -m src.modelos.treinamento.retreino_gnn [--sementes N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.modelos.arquitetura_gnn import (
    TransactionMiniBatchDataset,
    salvar_gnn,
)
from src.modelos.treinamento import treinar_modelos as e2
from src.preprocessamento.carregar_dataset import (
    GNN_FEATURE_NAMES,
    PROCESSED_DATA_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = e2.MODEL_DIR
RELATORIO = PROJECT_ROOT / "reports" / "etapa5_retreino_gnn.json"
SEED = 42
COLUNAS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "isFraud",
]


def executar(sementes: int | None) -> dict[str, Any]:
    if not e2.torch.cuda.is_available():
        raise RuntimeError("O retreino exige CUDA.")
    inicio = time.perf_counter()
    e2._set_seed()
    dados = pd.read_parquet(
        PROCESSED_DATA_PATH, columns=COLUNAS, filters=[("step", "<", 600)]
    ).reset_index(drop=True)
    if not dados["step"].lt(600).all():
        raise ValueError("Retreino lê somente step < 600.")
    treino_core = dados[dados["step"] < 550]
    validacao = dados[(dados["step"] >= 550) & (dados["step"] < 600)]
    amostra = treino_core.sample(
        n=min(sementes or len(treino_core), len(treino_core)), random_state=SEED
    )
    contexto = dados.sort_values("step", kind="stable").drop(columns=["isFraud"])

    grafos = {
        "train": TransactionMiniBatchDataset(
            amostra, GNN_FEATURE_NAMES, context_df=contexto
        ),
        "val": TransactionMiniBatchDataset(
            validacao, GNN_FEATURE_NAMES, context_df=contexto
        ),
    }
    positivo = e2._class_positive_weight(amostra["isFraud"].to_numpy(dtype=np.int64))
    tempo_preparo = time.perf_counter() - inicio

    selecao = e2._train_final_gnn_candidate(
        grafos,
        GNN_FEATURE_NAMES,
        dict(e2.FINAL_GNN_CONFIG),
        positivo,
        max_epochs=e2.GNN_EPOCHS,
    )
    modelo = e2.GNNFraudDetector(
        in_channels=len(GNN_FEATURE_NAMES),
        hidden_channels=e2.FINAL_GNN_CONFIG["hidden_dim"],
        dropout=e2.FINAL_GNN_CONFIG["dropout"],
        num_layers=e2.FINAL_GNN_CONFIG["num_layers"],
    )
    modelo.load_state_dict(selecao["state_dict"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    antigo = MODEL_DIR / "gnn_model.pt"
    if antigo.exists() and not (MODEL_DIR / "gnn_model_etapa2_v1.pt").exists():
        shutil.copy2(antigo, MODEL_DIR / "gnn_model_etapa2_v1.pt")
    salvar_gnn(
        modelo,
        grafos["train"],
        filepath=str(antigo),
        metadata={
            "feature_names": GNN_FEATURE_NAMES,
            "fanouts": list(e2.FINAL_GNN_FANOUTS),
            "amostrador": "corrigido: 2 saltos, fanouts (10, 5)",
            "sementes_treino": int(len(amostra)),
            "passo_maximo_treino": 549,
            "passo_maximo_validacao": 599,
            "melhor_epoca": selecao["best_epoch"],
            "val_pr_auc": selecao["validation_metrics"]["pr_auc"],
            "semente": SEED,
        },
    )
    relatorio = {
        "status": "concluido",
        "modelo": "gnn_model.pt",
        "modelo_anterior": "gnn_model_etapa2_v1.pt",
        "leitura_teste": "nenhuma (apenas step < 600)",
        "sementes_treino": int(len(amostra)),
        "sementes_treino_total_core": int(len(treino_core)),
        "linhas_validacao": int(len(validacao)),
        "semente": SEED,
        "fanouts": list(e2.FINAL_GNN_FANOUTS),
        "config": dict(e2.FINAL_GNN_CONFIG),
        "melhor_epoca": selecao["best_epoch"],
        "epocas_executadas": selecao["epochs_completed"],
        "validacao": selecao["validation_metrics"],
        "tempo_preparo_s": tempo_preparo,
        "tempo_selecao_s": selecao["selection_training_seconds"],
        "tempo_total_s": time.perf_counter() - inicio,
    }
    RELATORIO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return relatorio


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sementes", type=int, default=None)
    resultado = executar(parser.parse_args(argv).sementes)
    print(json.dumps(resultado, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()
