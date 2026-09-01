"""Testes unitários para o módulo de orquestração."""

from unittest.mock import MagicMock, patch

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


class TestJudgeTool:
    """Testa a lógica de decisão consolidada dos especialistas."""

    def test_nao_bloqueia_quando_score_medio_esta_baixo(self):
        from src.ferramentas.ferramentas_crewai import judge_tool

        result = judge_tool(
            transaction={"amount": 100.0},
            behavior_score=0.10,
            temporal_score=0.20,
            identity_score=0.95,
            specialist_outputs={},
        )

        assert result["decision"] == "Aprovada" or result["decision"] == "Revisão Humana necessária"
        assert result["score"] < 0.8


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


class TestOrchestrateTransactionLLMJudge:
    """Testa o roteamento entre o Juiz-LLM (CrewAI + Gemini) e o fallback determinístico."""

    def test_usa_fallback_quando_llm_indisponivel(self, mock_tools, sample_transaction):
        from src.orquestração import fluxo_crewai

        with patch.object(fluxo_crewai, "_llm_judge_available", return_value=False):
            result = fluxo_crewai.orchestrate_transaction(sample_transaction)

        assert result["decision"] == _MOCK_DECISION
        mock_tools["judge"].assert_called_once()

    def test_usa_llm_quando_disponivel_e_bem_sucedido(self, mock_tools, sample_transaction):
        from src.orquestração import fluxo_crewai

        llm_result = {
            "decision": "Aprovada",
            "score": 0.1,
            "justification": "ok via LLM",
            "specialist_outputs": {},
        }
        with patch.object(fluxo_crewai, "_llm_judge_available", return_value=True), \
             patch.object(fluxo_crewai, "llm_judge_tool", return_value=llm_result) as mock_llm:
            result = fluxo_crewai.orchestrate_transaction(sample_transaction)

        assert result["decision"] == llm_result
        mock_llm.assert_called_once()
        mock_tools["judge"].assert_not_called()

    def test_cai_no_fallback_se_llm_falhar(self, mock_tools, sample_transaction):
        from src.orquestração import fluxo_crewai

        with patch.object(fluxo_crewai, "_llm_judge_available", return_value=True), \
             patch.object(fluxo_crewai, "llm_judge_tool", side_effect=RuntimeError("falha de rede")):
            result = fluxo_crewai.orchestrate_transaction(sample_transaction)

        assert result["decision"] == _MOCK_DECISION
        mock_tools["judge"].assert_called_once()

    def test_use_llm_judge_false_ignora_llm_mesmo_disponivel(self, mock_tools, sample_transaction):
        from src.orquestração import fluxo_crewai

        with patch.object(fluxo_crewai, "_llm_judge_available", return_value=True), \
             patch.object(fluxo_crewai, "llm_judge_tool") as mock_llm:
            fluxo_crewai.orchestrate_transaction(sample_transaction, use_llm_judge=False)

        mock_llm.assert_not_called()
        mock_tools["judge"].assert_called_once()

    def test_orchestrate_batch_usa_fallback_por_padrao(self, mock_tools, sample_transaction):
        from src.orquestração import fluxo_crewai

        with patch.object(fluxo_crewai, "_llm_judge_available", return_value=True), \
             patch.object(fluxo_crewai, "llm_judge_tool") as mock_llm:
            fluxo_crewai.orchestrate_batch([sample_transaction, sample_transaction])

        # orchestrate_batch usa use_llm_judge=False por padrão (evita 1
        # chamada de API por transação em lotes grandes).
        mock_llm.assert_not_called()
        assert mock_tools["judge"].call_count == 2


class TestHoraDoDia:
    """Testa a derivação da hora do dia a partir do campo `step`."""

    def test_dado_real_paysim_horas_desde_inicio(self):
        from src.ferramentas.ferramentas_crewai import _hora_do_dia

        # PaySim: step = horas desde o início da simulação (1-744).
        assert _hora_do_dia({"step": 100}) == 100 % 24
        assert _hora_do_dia({"step": 1}) == 1
        assert _hora_do_dia({"step": 744}) == 744 % 24

    def test_entrada_interativa_hora_direta(self):
        from src.ferramentas.ferramentas_crewai import _hora_do_dia

        # Entrada manual/dashboard: step já é a hora do dia (0-23).
        assert _hora_do_dia({"step": 3}) == 3
        assert _hora_do_dia({"step": 23}) == 23

    def test_ausente_retorna_zero(self):
        from src.ferramentas.ferramentas_crewai import _hora_do_dia

        assert _hora_do_dia({}) == 0

    def test_regra_madrugada_usa_hora_correta(self, mock_tools):
        """Regressão do bug em que step era tratado como minutos (step // 60),
        gerando horário incorreto para dados reais do PaySim (step em horas)."""
        from src.ferramentas.ferramentas_crewai import judge_tool

        # step=4 -> madrugada (04h); valor alto + TRANSFER -> bloqueio imediato.
        result = judge_tool(
            transaction={"step": 4, "amount": 15000.0, "type": "TRANSFER"},
            behavior_score=0.1,
            temporal_score=0.1,
            identity_score=0.1,
            specialist_outputs={},
        )
        assert result["decision"] == "Recusada"
        assert result["score"] == pytest.approx(0.95)


class TestParseJudgeOutput:
    """Testa o parsing da resposta em texto do Juiz-LLM."""

    def test_parse_formato_correto(self):
        from src.ferramentas.ferramentas_crewai import _parse_judge_output

        raw = "DECISAO: Aprovada\nSCORE: 0.234\nJUSTIFICATIVA: Risco baixo, tudo dentro do padrão."
        decision, score, justification = _parse_judge_output(raw)
        assert decision == "Aprovada"
        assert score == pytest.approx(0.234)
        assert "Risco baixo" in justification

    def test_parse_decisao_com_texto_extra_normaliza(self):
        from src.ferramentas.ferramentas_crewai import _parse_judge_output

        raw = "DECISAO: BLOQUEADO - risco altíssimo\nSCORE: 0.9\nJUSTIFICATIVA: Padrão suspeito."
        decision, _, _ = _parse_judge_output(raw)
        assert decision == "Recusada"

    def test_parse_score_e_limitado_a_um(self):
        from src.ferramentas.ferramentas_crewai import _parse_judge_output

        raw = "DECISAO: Recusada\nSCORE: 1.5\nJUSTIFICATIVA: x"
        _, score, _ = _parse_judge_output(raw)
        assert score == 1.0

    def test_parse_formato_invalido_levanta_erro(self):
        from src.ferramentas.ferramentas_crewai import _parse_judge_output

        with pytest.raises(ValueError):
            _parse_judge_output("resposta completamente fora do formato esperado")


class TestLlmJudgeTool:
    """Testa llm_judge_tool com o Crew do CrewAI mockado (sem chamar a API real)."""

    _SPECIALIST_OUTPUTS = {
        "behavior": {"details": {"explanation": "ok"}},
        "temporal": {"details": {"explanation": "ok"}},
        "identity": {"details": {"explanation": "ok"}},
    }

    def test_usa_resposta_bem_formatada_do_crew(self):
        from src.ferramentas import ferramentas_crewai as fc

        fake_crew_instance = MagicMock()
        fake_crew_instance.kickoff.return_value = (
            "DECISAO: Revisão Humana necessária\nSCORE: 0.6\nJUSTIFICATIVA: Padrão intermediário."
        )
        with patch.object(fc, "_get_judge_agent", return_value=MagicMock()), \
             patch("crewai.Crew", return_value=fake_crew_instance), \
             patch("crewai.Task", return_value=MagicMock()):
            result = fc.llm_judge_tool(
                transaction={"amount": 100.0, "type": "TRANSFER", "step": 10},
                behavior_score=0.3,
                temporal_score=0.4,
                identity_score=0.2,
                specialist_outputs=self._SPECIALIST_OUTPUTS,
            )

        assert result["decision"] == "Revisão Humana necessária"
        assert result["score"] == pytest.approx(0.6)
        assert "raw_llm_output" in result

    def test_resposta_fora_do_formato_levanta_erro(self):
        from src.ferramentas import ferramentas_crewai as fc

        fake_crew_instance = MagicMock()
        fake_crew_instance.kickoff.return_value = "resposta aleatória sem o formato pedido"
        with patch.object(fc, "_get_judge_agent", return_value=MagicMock()), \
             patch("crewai.Crew", return_value=fake_crew_instance), \
             patch("crewai.Task", return_value=MagicMock()):
            with pytest.raises(ValueError):
                fc.llm_judge_tool(
                    transaction={"amount": 100.0, "type": "TRANSFER", "step": 10},
                    behavior_score=0.3,
                    temporal_score=0.4,
                    identity_score=0.2,
                    specialist_outputs=self._SPECIALIST_OUTPUTS,
                )
