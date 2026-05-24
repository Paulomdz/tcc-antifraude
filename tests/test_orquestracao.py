"""Testes unitários para o módulo de orquestração."""

from unittest.mock import patch

import pytest


_MOCK_BEHAVIOR = {"score": 0.3, "label": "behavior", "details": {"explanation": "OK"}}
_MOCK_TEMPORAL = {"score": 0.4, "label": "temporal", "details": {"explanation": "OK"}}
_MOCK_IDENTITY = {"score": 0.2, "label": "identity", "details": {"explanation": "OK"}}
_MOCK_DECISION = {
    "decision": "Aprovada",
    "score": 0.3,
    "justification": "Risco baixo.",
    "specialist_outputs": {},
}

_PATCH_BEHAVIOR = "src.orquestração.fluxo_crewai.behavior_specialist_tool"
_PATCH_TEMPORAL = "src.orquestração.fluxo_crewai.temporal_specialist_tool"
_PATCH_IDENTITY = "src.orquestração.fluxo_crewai.identity_specialist_tool"
_PATCH_JUDGE = "src.orquestração.fluxo_crewai.judge_tool"


@pytest.fixture()
def mock_tools():
    """Mocka todas as ferramentas do fluxo de orquestração."""
    with (
        patch(_PATCH_BEHAVIOR, return_value=_MOCK_BEHAVIOR) as mb,
        patch(_PATCH_TEMPORAL, return_value=_MOCK_TEMPORAL) as mt,
        patch(_PATCH_IDENTITY, return_value=_MOCK_IDENTITY) as mi,
        patch(_PATCH_JUDGE, return_value=_MOCK_DECISION) as mj,
    ):
        yield {"behavior": mb, "temporal": mt, "identity": mi, "judge": mj}


class TestOrchestrateBatch:
    """Testa orchestrate_batch com ferramentas mockadas."""

    def test_retorna_lista(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_batch
        result = orchestrate_batch([sample_transaction])
        assert isinstance(result, list)

    def test_tamanho_igual_ao_input(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_batch
        result = orchestrate_batch([sample_transaction, sample_transaction])
        assert len(result) == 2

    def test_lista_vazia_retorna_lista_vazia(self, mock_tools):
        from src.orquestração.fluxo_crewai import orchestrate_batch
        result = orchestrate_batch([])
        assert result == []

    def test_cada_item_tem_chaves_esperadas(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_batch
        result = orchestrate_batch([sample_transaction])
        item = result[0]
        assert {"transaction", "behavior", "temporal", "identity", "decision"} <= set(item.keys())

    def test_chama_behavior_tool_uma_vez_por_transacao(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_batch
        orchestrate_batch([sample_transaction, sample_transaction])
        assert mock_tools["behavior"].call_count == 2


class TestOrchestrateSingle:
    """Testa orchestrate_transaction com ferramentas mockadas."""

    def test_retorna_dicionario(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_transaction
        result = orchestrate_transaction(sample_transaction)
        assert isinstance(result, dict)

    def test_chaves_corretas(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_transaction
        result = orchestrate_transaction(sample_transaction)
        assert set(result.keys()) == {"transaction", "behavior", "temporal", "identity", "decision"}

    def test_transaction_preservada(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_transaction
        result = orchestrate_transaction(sample_transaction)
        assert result["transaction"] == sample_transaction

    def test_behavior_score_correto(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_transaction
        result = orchestrate_transaction(sample_transaction)
        assert result["behavior"]["score"] == 0.3

    def test_judge_recebe_tres_scores(self, mock_tools, sample_transaction):
        from src.orquestração.fluxo_crewai import orchestrate_transaction
        orchestrate_transaction(sample_transaction)
        call_kwargs = mock_tools["judge"].call_args
        assert "behavior_score" in call_kwargs.kwargs
        assert "temporal_score" in call_kwargs.kwargs
        assert "identity_score" in call_kwargs.kwargs
