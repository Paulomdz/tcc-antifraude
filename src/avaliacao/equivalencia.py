"""Prova de equivalência na validação: sistema por transação vs lote B.

Compara, nas mesmas transações, os scores que o sistema calcula uma
transação por vez (as mesmas funções que os agentes chamam) com os scores em
lote da Etapa 4 (``calcular_scores_b``). Lê só step < 600: o índice do
passado é montado com limite 600 e as transações vêm da validação (550..599).
Não chama o Gemini e não treina.

Uso: python -m src.avaliacao.equivalencia [--n 2000]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.avaliacao.avaliar_scores import (
    COLUNAS_TRANSACAO,
    SEED,
    _commit,
    _ram_pico_gib,
    calcular_scores_b,
    carregar_split,
)
from src.ferramentas.historico import grafo_do_passado
from src.ferramentas.inferencia_modelos import (
    gnn_identity_score,
    isolation_forest_score,
    lstm_sequence_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELATORIO = PROJECT_ROOT / "reports" / "etapa5_equivalencia_validacao.json"
TOLERANCIA = 1e-3
LIMITE_PASSO = 600


def _diferencas(sistema: np.ndarray, lote: np.ndarray) -> dict[str, Any]:
    diferenca = np.abs(sistema - lote)
    return {
        "diferenca_max": float(diferenca.max()),
        "diferenca_media": float(diferenca.mean()),
        "valores_distintos_sistema": int(len(np.unique(sistema.round(9)))),
        "valores_distintos_lote": int(len(np.unique(lote.round(9)))),
        "passou": bool(diferenca.max() < TOLERANCIA),
    }


def executar(n: int = 2000) -> dict[str, Any]:
    inicio = time.perf_counter()
    contexto, alvos = carregar_split("validacao")
    if not (contexto["step"].lt(LIMITE_PASSO).all() and alvos["step"].between(550, 599).all()):
        raise ValueError("A prova só pode ler step < 600.")
    escolhidas = np.sort(
        np.random.default_rng(SEED).choice(len(alvos), size=min(n, len(alvos)), replace=False)
    )
    amostra = alvos.iloc[escolhidas]

    marco = time.perf_counter()
    lote = calcular_scores_b(contexto, amostra)
    tempo_lote = time.perf_counter() - marco
    del contexto

    marco = time.perf_counter()
    grafo = grafo_do_passado(LIMITE_PASSO)
    tempo_indice = time.perf_counter() - marco

    transacoes = amostra[COLUNAS_TRANSACAO].to_dict("records")
    scores = {"if": [], "lstm": [], "gnn": []}
    marco = time.perf_counter()
    for transacao in transacoes:
        scores["if"].append(isolation_forest_score(transacao))
        scores["lstm"].append(
            lstm_sequence_score(None, current_transaction=transacao, passado=grafo)
        )
        scores["gnn"].append(gnn_identity_score(transacao, passado=grafo))
    tempo_sistema = time.perf_counter() - marco

    comparacoes = {
        "isolation_forest": _diferencas(np.asarray(scores["if"]), np.asarray(lote["comportamental"])),
        "lstm": _diferencas(np.asarray(scores["lstm"]), np.asarray(lote["temporal"])),
        "gnn": _diferencas(np.asarray(scores["gnn"]), np.asarray(lote["identidade"])),
    }
    relatorio = {
        "status": "passou" if all(c["passou"] for c in comparacoes.values()) else "falhou",
        "data": datetime.now(timezone.utc).isoformat(),
        "commit": _commit(),
        "split": "validacao",
        "leitura_teste": "nenhuma (contexto e índice com step < 600)",
        "transacoes": len(transacoes),
        "seed_amostra": SEED,
        "step_min": int(amostra["step"].min()),
        "step_max": int(amostra["step"].max()),
        "tolerancia": TOLERANCIA,
        "comparacoes": comparacoes,
        "dispositivo_lote": lote["dispositivo"],
        "tempo_segundos": {
            "lote": tempo_lote,
            "indice_do_passado": tempo_indice,
            "sistema_total": tempo_sistema,
            "sistema_por_transacao": tempo_sistema / max(len(transacoes), 1),
            "total": time.perf_counter() - inicio,
        },
        "ram_pico_gib": _ram_pico_gib(),
    }
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return relatorio


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    relatorio = executar(parser.parse_args(argv).n)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if relatorio["status"] != "passou":
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
