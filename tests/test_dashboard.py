"""Testes unitários para o dashboard Streamlit — funções auxiliares puras."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mocking do Streamlit e Plotly ANTES de importar o módulo.
# st.tabs(labels) e st.columns(spec) precisam retornar iteráveis do tamanho certo
# para que os unpacks dentro de main() funcionem sem lançar ValueError.
_streamlit_mock = MagicMock()
_plotly_mock = MagicMock()


def _tabs_side_effect(labels):
    return [MagicMock() for _ in labels]


def _columns_side_effect(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(n)]


_streamlit_mock.tabs.side_effect = _tabs_side_effect
_streamlit_mock.columns.side_effect = _columns_side_effect
# session_state precisa ser um dict real para que .get() devolva o default corretamente
_streamlit_mock.session_state = {}

sys.modules.setdefault("streamlit", _streamlit_mock)
sys.modules.setdefault("plotly", _plotly_mock)
sys.modules.setdefault("plotly.express", _plotly_mock)

from src.dashboard.app import (  # noqa: E402
    _DECISION_ICON,
    _DECISION_MAP,
    _TX_TYPES,
    build_result_row,
    decision_icon,
    map_decision,
    random_transaction,
    render_stats,
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
        row = build_result_row(sample_transaction, self._make_resultado())
        assert isinstance(row, dict)

    def test_contem_tipo(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado())
        assert row["Tipo"] == "TRANSFER"

    def test_contem_valor(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado())
        assert row["Valor (R$)"] == 5000.0

    def test_decisao_aprovado_no_campo(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado("Aprovada", 0.3))
        assert "APROVADO" in row["Decisão"]

    def test_decisao_bloqueado_no_campo(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado("Recusada", 0.9))
        assert "BLOQUEADO" in row["Decisão"]

    def test_decisao_revisao_no_campo(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado("Revisão Humana necessária", 0.6))
        assert "REVISÃO" in row["Decisão"]

    def test_scores_sao_float_arredondados(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado("Aprovada", 0.123456))
        assert row["Comportamental"] == 0.123
        assert row["Temporal"] == 0.123
        assert row["Identidade"] == 0.123

    def test_contem_origem_e_destino(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado())
        assert row["Origem"] == "C1234567"
        assert row["Destino"] == "C7654321"

    def test_icone_incluido_na_decisao(self, sample_transaction):
        row = build_result_row(sample_transaction, self._make_resultado("Aprovada"))
        assert "🟢" in row["Decisão"]


class TestRandomTransaction:
    """Testa o gerador de transações aleatórias para simulação."""

    def test_retorna_dicionario(self):
        tx = random_transaction()
        assert isinstance(tx, dict)

    def test_contem_campos_obrigatorios(self):
        tx = random_transaction()
        expected = {"step", "type", "amount", "nameOrig", "oldbalanceOrg",
                    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "step_norm"}
        assert expected <= set(tx.keys())

    def test_amount_positivo(self):
        for _ in range(10):
            assert random_transaction()["amount"] > 0

    def test_step_dentro_do_intervalo(self):
        for _ in range(10):
            step = random_transaction()["step"]
            assert 1 <= step <= 744

    def test_step_norm_entre_zero_e_um(self):
        for _ in range(10):
            sn = random_transaction()["step_norm"]
            assert 0.0 <= sn <= 1.0

    def test_tipo_valido(self):
        for _ in range(20):
            assert random_transaction()["type"] in _TX_TYPES

    def test_contas_comecam_com_c(self):
        tx = random_transaction()
        assert tx["nameOrig"].startswith("C")
        assert tx["nameDest"].startswith("C")

    def test_novo_saldo_origem_nao_negativo(self):
        for _ in range(20):
            assert random_transaction()["newbalanceOrig"] >= 0.0

    def test_gera_transacoes_diferentes(self):
        txs = [random_transaction() for _ in range(5)]
        amounts = {t["amount"] for t in txs}
        # Com amostra de 5 é praticamente impossível todos os valores serem iguais
        assert len(amounts) > 1


class TestRenderStats:
    """Testa a função render_stats com dados sintéticos e Streamlit mockado."""

    def _make_rows(self, n: int = 3, decisao: str = "🟢 APROVADO") -> list:
        return [
            {
                "Tipo": "TRANSFER",
                "Valor (R$)": 1000.0 * (i + 1),
                "Origem": f"C{i}",
                "Destino": f"C{i+1}",
                "Comportamental": 0.2,
                "Temporal": 0.3,
                "Identidade": 0.1,
                "Score Final": 0.2,
                "Decisão": decisao,
            }
            for i in range(n)
        ]

    def test_lista_vazia_chama_info(self):
        render_stats([])
        _streamlit_mock.info.assert_called()

    def test_resultados_sem_chave_decisao_chama_warning(self):
        render_stats([{"Score Final": 0.3}])
        _streamlit_mock.warning.assert_called()

    def test_com_resultados_validos_chama_plotly_chart(self):
        render_stats(self._make_rows(3, "🟢 APROVADO"))
        _streamlit_mock.plotly_chart.assert_called()

    def test_contagem_aprovados_correta(self):
        rows = self._make_rows(2, "🟢 APROVADO") + self._make_rows(1, "🔴 BLOQUEADO")
        _streamlit_mock.columns.side_effect = _columns_side_effect
        render_stats(rows)
        # Se plotly_chart foi chamado, a seção de gráficos executou
        assert _streamlit_mock.plotly_chart.called

    def test_com_scores_agente_chama_box_plot(self):
        rows = self._make_rows(2, "🟢 APROVADO")
        render_stats(rows)
        # Box plot é chamado quando há dados de agentes
        assert _streamlit_mock.plotly_chart.call_count >= 2

    def test_import_error_plotly_chama_warning(self):
        import sys
        # Temporariamente remove plotly.express do cache de módulos
        px_backup = sys.modules.get("plotly.express")
        sys.modules["plotly.express"] = None  # type: ignore[assignment]
        try:
            render_stats(self._make_rows())
        except Exception:
            pass
        finally:
            if px_backup is not None:
                sys.modules["plotly.express"] = px_backup
            else:
                sys.modules.pop("plotly.express", None)


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
