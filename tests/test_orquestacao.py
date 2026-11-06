import sys
from types import SimpleNamespace

import pytest

from src.ferramentas import ferramentas_agentes as tools
from src.ferramentas.regra_noturna import avaliar_regra_noturna
from src.orquestacao import fluxo_multiagente as flow


SPECIALIST_OUTPUTS = {
    name: {"score": score, "details": {"explanation": f"{name} explanation"}}
    for name, score in (("behavior", 0.2), ("temporal", 0.3), ("identity", 0.4))
}


def test_step_is_hour_index_in_pix_dataset():
    assert tools._hora_do_dia({"step": 100}) == 4
    assert tools._hora_do_dia({}) == 0


def _noturna(tipo="PIX_P2P", amount=1_200.0, step=22, origem="C1", destino="C2"):
    return {"step": step, "amount": amount, "type": tipo, "nameOrig": origem, "nameDest": destino}


def _juiz(tx, score, total_anterior=0.0):
    return tools.judge_tool(
        tx, score, score, score, {},
        regra_noturna=avaliar_regra_noturna(tx, total_anterior=total_anterior),
    )


@pytest.mark.parametrize("tx_type", ["PIX_P2P", "PIX_QRCODE", "PIX_AUTOMATICO", "TED"])
def test_regra_noturna_leva_aprovacao_para_revisao(tx_type):
    result = _juiz(_noturna(tx_type), 0.1)

    assert result["decision"] == "Revisão Humana necessária"
    assert result["score"] == pytest.approx(0.1)  # sem bônus
    assert result["regra_noturna"]["aplica"] is True
    assert "IN BCB 331/2022" in result["justification"]


def test_regra_noturna_nao_bloqueia_nem_rebaixa_recusa():
    assert _juiz(_noturna(amount=50_000.0), 0.1)["decision"] == "Revisão Humana necessária"
    assert _juiz(_noturna(), 0.9)["decision"] == "Recusada"


def test_regra_noturna_usa_o_total_acumulado_na_noite():
    # 900 já gastos + 200 agora passa de 1.000; 1.100 já gastos não estoura de novo
    assert _juiz(_noturna(amount=200.0), 0.1, total_anterior=900.0)["decision"] == (
        "Revisão Humana necessária"
    )
    assert _juiz(_noturna(amount=200.0), 0.1, total_anterior=0.0)["decision"] == "Aprovada"
    assert _juiz(_noturna(amount=50.0), 0.1, total_anterior=1_100.0)["decision"] == "Aprovada"


@pytest.mark.parametrize(
    "tx",
    [
        _noturna("TEF"),
        _noturna("BOLETO"),
        _noturna("PIX_SAQUE"),
        _noturna("PIX_TROCO"),
        _noturna(step=12),
        _noturna(destino="M2"),
        _noturna(origem="B1"),
        _noturna(destino="C1"),
    ],
)
def test_fora_do_escopo_a_regra_noturna_nao_se_aplica(tx):
    result = tools.judge_tool(tx, 0.1, 0.1, 0.1, {})

    assert result["decision"] == "Aprovada"
    assert result["regra_noturna"]["aplica"] is False


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.2, "Aprovada"),
        (0.6, "Revisão Humana necessária"),
        (0.9, "Recusada"),
    ],
)
def test_judge_applies_score_thresholds(score, expected):
    result = tools.judge_tool({"step": 12, "type": "BOLETO"}, score, score, score, {})

    assert result["decision"] == expected


def test_judge_output_parser_normalizes_and_clamps():
    parsed = tools._parse_judge_output(
        "DECISAO: bloqueada\nSCORE: 1.2\nJUSTIFICATIVA: Revisão efetuada."
    )

    assert parsed == ("Recusada", 1.0, "Revisão efetuada.")


@pytest.mark.parametrize(
    "output",
    [
        "resposta inválida",
        "DECISAO: incerta\nSCORE: 0.4\nJUSTIFICATIVA: texto",
    ],
)
def test_judge_output_parser_rejects_malformed_answers(output):
    with pytest.raises(ValueError):
        tools._parse_judge_output(output)


def test_specialist_tools_include_scores_and_explanations(monkeypatch):
    monkeypatch.setattr(tools, "isolation_forest_score", lambda _tx: 0.1)
    monkeypatch.setattr(
        tools, "lstm_sequence_score", lambda _seq, **_kwargs: 0.2
    )
    monkeypatch.setattr(tools, "gnn_identity_score", lambda _graph: 0.3)
    tx = {
        "step": 3,
        "amount": 12_000,
        "type": "PIX_P2P",
        "nameOrig": "C1",
        "nameDest": "C2",
    }

    behavior = tools.behavior_specialist_tool(tx)
    temporal = tools.temporal_specialist_tool(tx)
    identity = tools.identity_specialist_tool(tx)

    assert behavior["score"] == 0.1
    assert "03:00" in behavior["details"]["explanation"]
    assert "período noturno" in temporal["details"]["explanation"]
    assert "ALERTA CRÍTICO" not in temporal["details"]["explanation"]
    assert identity["score"] == 0.3


def test_llm_tool_passes_specialist_reports_to_crew(monkeypatch):
    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCrew:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def kickoff(self):
            return (
                "DECISAO: Aprovada\n"
                "SCORE: 0.2\n"
                "JUSTIFICATIVA: Pareceres consistentes."
            )

    monkeypatch.setitem(
        sys.modules, "crewai", SimpleNamespace(Crew=FakeCrew, Task=FakeTask)
    )
    monkeypatch.setattr(tools, "_get_judge_agent", lambda: object())

    result = tools.llm_judge_tool(
        {"step": 12, "type": "PIX_P2P", "amount": 100},
        0.1,
        0.2,
        0.3,
        SPECIALIST_OUTPUTS,
    )

    assert result["decision"] == "Aprovada"
    assert result["raw_llm_output"].startswith("DECISAO:")


def test_deep_tool_runs_four_sequential_agents(monkeypatch):
    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCrew:
        def __init__(self, **kwargs):
            assert len(kwargs["agents"]) == 4
            assert len(kwargs["tasks"]) == 4
            assert kwargs["process"] == "sequential"

        def kickoff(self):
            return (
                "DECISAO: Aprovada\n"
                "SCORE: 0.2\n"
                "JUSTIFICATIVA: Quatro pareceres considerados."
            )

    class FakeAgents:
        def especialista_comportamental(self):
            return "behavior"

        def especialista_temporal(self):
            return "temporal"

        def especialista_identidade(self):
            return "identity"

        def juiz_final(self):
            return "judge"

    monkeypatch.setitem(
        sys.modules,
        "crewai",
        SimpleNamespace(
            Crew=FakeCrew,
            Task=FakeTask,
            Process=SimpleNamespace(sequential="sequential"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.orquestacao.definicoes_agentes",
        SimpleNamespace(FraudSimulatorAgents=FakeAgents),
    )

    result = tools.deep_multiagent_tool(
        {"step": 12, "type": "PIX_P2P", "amount": 100},
        SPECIALIST_OUTPUTS,
    )

    assert result["decision"] == "Aprovada"


def test_fast_mode_uses_deterministic_aggregator(monkeypatch):
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["behavior"])
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["temporal"])
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["identity"])
    monkeypatch.setattr(flow, "judge_tool", lambda **kwargs: {"decision": "Aprovada"})
    monkeypatch.setattr(flow, "llm_judge_tool", lambda **_kwargs: pytest.fail("LLM não esperado"))

    result = flow.orchestrate_ml_only({"type": "PIX_P2P"})

    assert result["decision"]["decision"] == "Aprovada"
    assert result["modo"] == "fast"
    assert result["behavior"]["score"] == 0.2


def test_production_requires_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Gemini não configurado"):
        flow.orchestrate_transaction({"type": "PIX_P2P"}, modo="production")


def test_production_does_not_fall_back_when_llm_raises(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: {"score": 0.1})
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: {"score": 0.2})
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: {"score": 0.3})
    monkeypatch.setattr(flow, "llm_judge_tool", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(
        flow, "judge_tool", lambda **_kwargs: pytest.fail("não pode haver fallback")
    )

    with pytest.raises(RuntimeError, match="offline"):
        flow.orchestrate_transaction({"type": "PIX_P2P"}, modo="production")


def test_orchestrate_batch_keeps_input_order(monkeypatch):
    seen = []

    def fake_orchestrate(tx):
        seen.append(tx["id"])
        return {"transaction": tx}

    monkeypatch.setattr(flow, "orchestrate_ml_only", fake_orchestrate)
    results = flow.orchestrate_batch([{"id": 1}, {"id": 2}])

    assert [result["transaction"]["id"] for result in results] == [1, 2]
    assert seen == [1, 2]


def test_deep_mode_routes_to_four_agent_tool(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["behavior"])
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["temporal"])
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["identity"])
    monkeypatch.setattr(
        flow,
        "deep_multiagent_tool",
        lambda **_kwargs: {"decision": "Aprovada", "score": 0.2},
    )

    result = flow.orchestrate_transaction({"type": "PIX_P2P"}, modo="deep")

    assert result["modo"] == "deep"
    assert result["decision"]["decision"] == "Aprovada"


@pytest.mark.parametrize("modo", ["production", "deep"])
def test_modos_com_llm_recebem_a_regra_noturna_e_aplicam_o_piso(monkeypatch, modo):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: {"score": 0.1})
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: {"score": 0.1})
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: {"score": 0.1})
    regra = {"aplica": True, "total_com_atual": 1_500.0}
    monkeypatch.setattr(flow, "avaliar_regra_noturna", lambda _tx: regra)
    recebido = {}

    def llm_falso(**kwargs):
        recebido.update(kwargs)
        return {"decision": "Aprovada", "score": 0.1, "justification": "ok"}

    monkeypatch.setattr(flow, "llm_judge_tool", llm_falso)
    monkeypatch.setattr(flow, "deep_multiagent_tool", llm_falso)

    resultado = flow.orchestrate_transaction({"type": "PIX_P2P"}, modo=modo)

    assert recebido["regra_noturna"] is regra
    assert resultado["decision"]["decision"] == "Revisão Humana necessária"
    assert resultado["decision"]["score"] == 0.1


@pytest.mark.parametrize("modo", ["production", "deep"])
def test_modos_com_llm_mantem_a_decisao_quando_a_regra_nao_se_aplica(monkeypatch, modo):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    for nome in ("behavior_specialist_tool", "temporal_specialist_tool", "identity_specialist_tool"):
        monkeypatch.setattr(flow, nome, lambda _tx: {"score": 0.1})
    monkeypatch.setattr(flow, "avaliar_regra_noturna", lambda _tx: {"aplica": False})
    resposta = lambda **_kwargs: {"decision": "Aprovada", "score": 0.1, "justification": "ok"}
    monkeypatch.setattr(flow, "llm_judge_tool", resposta)
    monkeypatch.setattr(flow, "deep_multiagent_tool", resposta)

    assert flow.orchestrate_transaction({"type": "PIX_P2P"}, modo=modo)["decision"]["decision"] == "Aprovada"


def test_juiz_llm_recebe_o_texto_da_regra_noturna(monkeypatch):
    tarefas = []

    class FakeTask:
        def __init__(self, **kwargs):
            tarefas.append(kwargs)

    class FakeCrew:
        def __init__(self, **kwargs):
            pass

        def kickoff(self):
            return "DECISAO: Revisão Humana necessária\nSCORE: 0.4\nJUSTIFICATIVA: regra noturna."

    monkeypatch.setitem(sys.modules, "crewai", SimpleNamespace(Crew=FakeCrew, Task=FakeTask))
    monkeypatch.setattr(tools, "_get_judge_agent", lambda: "judge")

    tools.llm_judge_tool(
        {"step": 22, "type": "PIX_P2P", "amount": 1_200},
        0.1, 0.1, 0.1, SPECIALIST_OUTPUTS,
        regra_noturna={"aplica": True, "total_com_atual": 1_200.0},
    )

    assert "Regra noturna (IN BCB 331/2022" in tarefas[0]["description"]
    assert "APLICA" in tarefas[0]["description"]


def test_agente_temporal_sem_sequence_pede_o_historico_do_parquet(monkeypatch):
    chamadas = []

    def lstm_falsa(sequence, **kwargs):
        chamadas.append((sequence, kwargs["current_transaction"]))
        return 0.4

    monkeypatch.setattr(tools, "lstm_sequence_score", lstm_falsa)
    tx = {"step": 3, "amount": 100.0, "type": "PIX_P2P", "nameOrig": "C1"}

    assert tools.temporal_specialist_tool(tx)["score"] == 0.4
    assert chamadas == [(None, tx)]
