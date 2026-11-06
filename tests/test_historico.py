"""Subgrafo do passado: igual ao lote e sem vazamento do futuro."""

import pandas as pd
import pytest
import torch

from src.ferramentas import historico
from src.modelos.arquitetura_gnn import TORCH_GEO_AVAILABLE, TransactionMiniBatchDataset
from src.preprocessamento.carregar_dataset import GNN_FEATURE_NAMES

pytestmark = pytest.mark.skipif(not TORCH_GEO_AVAILABLE, reason="PyG indisponível")


def _linha(step, origem, destino, valor=100.0):
    return {
        "step": step,
        "type": "PIX_P2P",
        "amount": valor,
        "nameOrig": origem,
        "nameDest": destino,
        "oldbalanceOrg": 500.0,
    }


def _lote_de_referencia(contexto: pd.DataFrame, novo: dict):
    """Mesma transação dentro do lote, como calcular_scores_b faz."""
    com_novo = pd.concat([contexto, pd.DataFrame([novo])], ignore_index=True)
    com_novo = com_novo.sort_values("step", kind="stable").reset_index(drop=True)
    alvo = com_novo.tail(1)
    ds = TransactionMiniBatchDataset(alvo, GNN_FEATURE_NAMES, context_df=com_novo)
    x, edge_index, _y, _n = next(ds.iter_batches(1, fanouts=(10, 5)))
    return x, edge_index


def test_subgrafo_igual_ao_lote_em_grafo_pequeno():
    contexto = pd.DataFrame(
        [
            _linha(1, "C1", "C2", 100.0),
            _linha(2, "C3", "C1", 50.0),
            _linha(3, "C2", "C4", 70.0),
            _linha(4, "C5", "C2", 20.0),
        ]
    )
    novo = _linha(5, "C1", "C2", 30.0)
    grafo = historico.GrafoPassado(contexto)

    x, edge_index = grafo.subgrafo(novo)
    x_lote, edge_lote = _lote_de_referencia(contexto, novo)

    assert torch.allclose(x, x_lote, atol=1e-6)
    assert torch.equal(edge_index, edge_lote)


def test_cadeia_da_mesma_conta_nao_expande_alem_de_dois_saltos():
    contexto = pd.DataFrame([_linha(passo, "C1", "C9") for passo in range(1, 11)])
    novo = _linha(11, "C1", "C8")
    grafo = historico.GrafoPassado(contexto)

    x, edge_index = grafo.subgrafo(novo)
    x_lote, edge_lote = _lote_de_referencia(contexto, novo)

    # semente, transação de step 10 e a de step 9: dois saltos com fanout 1 cada
    assert x.shape[0] == 3
    assert torch.allclose(x, x_lote, atol=1e-6)
    assert torch.equal(edge_index, edge_lote)


def test_conta_sem_passado_gera_um_no_sem_arestas():
    contexto = pd.DataFrame([_linha(1, "C1", "C2")])
    x, edge_index = historico.GrafoPassado(contexto).subgrafo(
        _linha(2, "C77", "C88")
    )

    assert x.shape == (1, len(GNN_FEATURE_NAMES))
    assert edge_index.shape == (2, 0)


def test_nao_usa_transacoes_do_mesmo_passo_nem_do_futuro():
    contexto = pd.DataFrame(
        [
            _linha(3, "C1", "C2"),
            _linha(5, "C1", "C2"),
            _linha(7, "C1", "C2"),
        ]
    )
    x, edge_index = historico.GrafoPassado(contexto).subgrafo(_linha(5, "C1", "C2"))

    # só o passo 3 existe antes do passo 5: uma aresta, da origem e do destino
    # para o mesmo vizinho, que deve aparecer uma única vez
    assert x.shape[0] == 2
    assert edge_index.shape == (2, 1)


def test_grafo_do_passado_respeita_limite_de_step(monkeypatch):
    chamadas = []

    def falso_parquet(_caminho, columns, filters):
        chamadas.append(filters)
        return pd.DataFrame([_linha(1, "C1", "C2"), _linha(2, "C1", "C2")])

    historico.grafo_do_passado.cache_clear()
    monkeypatch.setattr(historico.pd, "read_parquet", falso_parquet)
    try:
        historico.grafo_do_passado(600)
    finally:
        historico.grafo_do_passado.cache_clear()

    assert chamadas == [[("step", "<", 600)]]
