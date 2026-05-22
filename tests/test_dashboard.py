"""Testes unitários para o dashboard Streamlit — funções auxiliares puras."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mocking do Streamlit e Plotly ANTES de importar o módulo
# (evita execução de st.set_page_config e outros calls de UI)
_streamlit_mock = MagicMock()
_plotly_mock = MagicMock()
sys.modules.setdefault("streamlit", _streamlit_mock)
sys.modules.setdefault("plotly", _plotly_mock)
sys.modules.setdefault("plotly.express", _plotly_mock)

from src.dashboard.app import (  # noqa: E402 (import após mocking)
    _DECISION_MAP,
    _DECISION_ICON,
    build_result_row,
    decision_icon,
    map_decision,
)


class TestMapDecision:
    """Testa a conversão de decisões internas para labels do dashboard."""

    def test_aprovada_para_aprovado(self):
        assert map_decision("Aprovada") == "APROVADO"

    def test_recusada_para_bloqueado(self):
        assert map_decision("Recusada") == "BLOQUEADO"

    def test_revisao_humana_para_revisao(self):
        assert map_decision("Revisão Humana necessária") == "REVISÃO"

    def test_valor_desconhecido_retorna_original(self):
        assert map_decision("Desconhecido") == "Desconhecido"

    def test_string_vazia_retorna_vazia(self):
        assert map_decision("") == ""

    def test_mapa_cobre_todas_as_decisoes_possiveis(self):
        for key in _DECISION_MAP:
            assert map_decision(key) == _DECISION_MAP[key]


class TestDecisionIcon:
    """Testa emojis coloridos por decisão."""

    def test_aprovado_tem_verde(self):
        assert decision_icon("APROVADO") == "🟢"

    def test_revisao_tem_amarelo(self):
        assert decision_icon("REVISÃO") == "🟡"

    def test_bloqueado_tem_vermelho(self):
        assert decision_icon("BLOQUEADO") == "🔴"

    def test_desconhecido_retorna_branco(self):
        assert decision_icon("OUTRO") == "⚪"

    def test_icone_cobre_todos_os_estados(self):
        for key in _DECISION_ICON:
            assert len(decision_icon(key)) > 0


class TestBuildResultRow:
    """Testa a função que monta linha tabular de resultado."""

    def _make_resultado(self, decision_raw="Aprovada", score=0.3):
        return {
            "behavior": {"score": score},
            "temporal": {"score": score},
            "identity": {"score": score},
            "decision": {"decision": decision_raw, "score": score},
        }

    def test_retorna_dicionario(self, sample_transaction):
        resultado = self._make_resultado()
        row = build_result_row(sample_transaction, resultado)
        assert isinstance(row, dict)

    def test_contem_tipo(self, sample_transaction):
        resultado = self._make_resultado()
        row = build_result_row(sample_transaction, resultado)
        assert row["Tipo"] == "TRANSFER"

    def test_contem_valor(self, sample_transaction):
        resultado = self._make_resultado()
        row = build_result_row(sample_transaction, resultado)
        assert row["Valor (R$)"] == 5000.0

    def test_decisao_aprovado_no_campo(self, sample_transaction):
        resultado = self._make_resultado("Aprovada", 0.3)
        row = build_result_row(sample_transaction, resultado)
        assert "APROVADO" in row["Decisão"]

    def test_decisao_bloqueado_no_campo(self, sample_transaction):
        resultado = self._make_resultado("Recusada", 0.9)
        row = build_result_row(sample_transaction, resultado)
        assert "BLOQUEADO" in row["Decisão"]

    def test_decisao_revisao_no_campo(self, sample_transaction):
        resultado = self._make_resultado("Revisão Humana necessária", 0.6)
        row = build_result_row(sample_transaction, resultado)
        assert "REVISÃO" in row["Decisão"]

    def test_scores_sao_float_arredondados(self, sample_transaction):
        resultado = self._make_resultado("Aprovada", 0.123456)
        row = build_result_row(sample_transaction, resultado)
        assert row["Comportamental"] == 0.123
        assert row["Temporal"] == 0.123
        assert row["Identidade"] == 0.123

    def test_contem_origem_e_destino(self, sample_transaction):
        resultado = self._make_resultado()
        row = build_result_row(sample_transaction, resultado)
        assert row["Origem"] == "C1234567"
        assert row["Destino"] == "C7654321"

    def test_icone_incluido_na_decisao(self, sample_transaction):
        resultado = self._make_resultado("Aprovada")
        row = build_result_row(sample_transaction, resultado)
        assert "🟢" in row["Decisão"]


class TestRunAnalysis:
    """Testa run_analysis com módulo de orquestração mockado."""

    def test_chama_orchestrate_transaction(self, sample_transaction, mock_orchestrate_result):
        mock_module = MagicMock()
        mock_module.orchestrate_transaction.return_value = mock_orchestrate_result

        with patch("importlib.import_module", return_value=mock_module):
            from src.dashboard.app import run_analysis
            result = run_analysis(sample_transaction)

        assert result == mock_orchestrate_result
        mock_module.orchestrate_transaction.assert_called_once_with(sample_transaction)

    def test_propaga_excecao_do_modulo(self, sample_transaction):
        mock_module = MagicMock()
        mock_module.orchestrate_transaction.side_effect = RuntimeError("Falha no modelo")

        with patch("importlib.import_module", return_value=mock_module):
            from src.dashboard.app import run_analysis
            with pytest.raises(RuntimeError, match="Falha no modelo"):
                run_analysis(sample_transaction)
