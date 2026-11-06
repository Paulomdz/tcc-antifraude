"""Histórico do passado para uma transação, a partir do Parquet processado.

Serve à GNN (subgrafo do passado), à LSTM (janela das 5 transações anteriores
da conta de origem, ``janela_lstm``) e à regra noturna (total da noite).

Replica, para uma transação isolada, o que ``TransactionMiniBatchDataset``
faz em lote: os vizinhos de um nó são as transações anteriores (step menor)
da mesma conta de origem e da mesma conta de destino, e a amostragem tem
dois saltos com fanouts (10, 5). Só o passado entra no grafo.

O índice do passado é montado uma vez por limite de step e fica em cache.
``limite_passo=None`` usa a base inteira; a validação usa 600 para não ler o
teste (step >= 600) por engano.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.modelos.arquitetura_gnn import TransactionMiniBatchDataset
from src.preprocessamento.carregar_dataset import (
    GNN_FEATURE_NAMES,
    IF_FEATURE_NAMES,
    PROCESSED_DATA_PATH,
    compute_if_features,
)

FANOUTS = (10, 5)
COLUNAS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
]
_POSICOES_GNN = [IF_FEATURE_NAMES.index(nome) for nome in GNN_FEATURE_NAMES]


class GrafoPassado:
    """Índice do passado: transações ordenadas por step e vizinhos prévios."""

    def __init__(self, contexto: pd.DataFrame):
        ordenado = contexto.sort_values("step", kind="stable").reset_index(drop=True)
        self.dataset = TransactionMiniBatchDataset(
            ordenado, GNN_FEATURE_NAMES, context_df=ordenado
        )
        self.contexto = self.dataset.context
        self.steps = self.contexto["step"].to_numpy(dtype=np.int64)
        self._contas = {
            coluna: {
                str(conta): np.asarray(posicoes, dtype=np.int64)
                for conta, posicoes in self.contexto.groupby(
                    coluna, sort=False
                ).indices.items()
            }
            for coluna in ("nameOrig", "nameDest")
        }

    def janela_lstm(
        self,
        transacao: dict[str, Any],
        feature_names: list[str],
        seq_length: int = 5,
    ) -> np.ndarray:
        """Últimas ``seq_length`` transações da mesma conta de origem com step menor.

        Mesma regra de ``TransactionSequenceDataset``: ordem por step, zeros à
        esquerda quando há menos transações e conta vazia sem histórico.
        """
        janela = np.zeros((seq_length, len(feature_names)), dtype=np.float32)
        conta = str(transacao.get("nameOrig", "") or "")
        posicoes = self._contas["nameOrig"].get(conta) if conta else None
        if posicoes is None:
            return janela
        passo = int(transacao["step"])
        fim = int(np.searchsorted(self.steps[posicoes], passo, side="left"))
        anteriores = posicoes[max(0, fim - seq_length) : fim]
        if len(anteriores):
            janela[seq_length - len(anteriores) :] = compute_if_features(
                self.contexto.iloc[anteriores]
            )[feature_names].to_numpy(dtype=np.float32)
        return janela

    def _anterior(self, coluna: str, conta: str, passo: int) -> int | None:
        posicoes = self._contas[coluna].get(conta)
        if posicoes is None:
            return None
        k = int(np.searchsorted(self.steps[posicoes], passo, side="left")) - 1
        return int(posicoes[k]) if k >= 0 else None

    def _vizinhos(self, no: int, vizinhos_do_novo: list[int]) -> list[int]:
        if no < 0:
            return vizinhos_do_novo
        return list(dict.fromkeys(self.dataset._neighbors(no)))

    def subgrafo(self, transacao: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Features (nó 0 é a transação) e arestas do passado, como no lote."""
        passo = int(transacao["step"])
        vizinhos_do_novo = list(
            dict.fromkeys(
                anterior
                for anterior in (
                    self._anterior(coluna, str(transacao[coluna]), passo)
                    for coluna in ("nameOrig", "nameDest")
                )
                if anterior is not None
            )
        )
        nos = [-1]
        vistos = {-1}
        fronteira = [-1]
        for fanout in FANOUTS:
            proxima: list[int] = []
            for no in fronteira:
                for vizinho in self._vizinhos(no, vizinhos_do_novo)[:fanout]:
                    if vizinho not in vistos:
                        vistos.add(vizinho)
                        nos.append(vizinho)
                        proxima.append(vizinho)
            fronteira = proxima
            if not fronteira:
                break

        local = {no: i for i, no in enumerate(nos)}
        arestas = set()
        for no in nos:
            for vizinho in self._vizinhos(no, vizinhos_do_novo):
                if vizinho in local:
                    arestas.add((local[vizinho], local[no]))

        x = np.zeros((len(nos), len(GNN_FEATURE_NAMES)), dtype=np.float32)
        x[0] = compute_if_features(transacao)[_POSICOES_GNN]
        globais = [no for no in nos if no >= 0]
        if globais:
            linhas = compute_if_features(self.contexto.iloc[globais])[
                GNN_FEATURE_NAMES
            ].to_numpy(dtype=np.float32)
            x[[local[no] for no in globais]] = linhas

        if arestas:
            edge_index = torch.tensor(sorted(arestas), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return torch.from_numpy(x), edge_index


@lru_cache(maxsize=2)
def grafo_do_passado(limite_passo: int | None = None) -> GrafoPassado:
    """Carrega o Parquet uma vez e monta o índice. ``None`` = base inteira."""
    filtros = None if limite_passo is None else [("step", "<", int(limite_passo))]
    contexto = pd.read_parquet(PROCESSED_DATA_PATH, columns=COLUNAS, filters=filtros)
    return GrafoPassado(contexto.reset_index(drop=True))
