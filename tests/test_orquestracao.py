import sys
from types import SimpleNamespace

import pytest

from src.ferramentas import ferramentas_crewai as tools
from src.orquestacao import fluxo_crewai as flow


SPECIALIST_OUTPUTS = {
    name: {"score": score, "details": {"explanation": f"{name} explanation"}}
    for name, score in (("behavior", 0.2), ("temporal", 0.3), ("identity", 0.4))
}


def test_step_is_hour_index_in_pix_dataset():
    assert tools._hora_do_dia({"step": 100}) == 4
    assert tools._hora_do_dia({}) == 0


@pytest.mark.parametrize("tx_type", ["PIX_P2P", "PIX_AUTOMATICO", "TED", "TEF"])
def test_high_value_night_transfers_are_blocked(tx_type):
    result = tools.judge_tool(
        {"step": 4, "amount": 12_000, "type": tx_type},
        0.0,
        0.0,
        0.0,
        {},
    )

    assert result["decision"] == "Recusada"
    assert result["score"] == 0.95


def test_high_value_night_other_types_get_risk_increase():
    result = tools.judge_tool(
        {"step": 4, "amount": 12_000, "type": "BOLETO"},
        0.2,
        0.3,
        0.4,
        {},
    )

    assert result["score"] == pytest.approx(0.55)
    assert result["decision"] == "Revisão Humana necessária"


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
    monkeypatch.setattr(tools, "lstm_sequence_score", lambda _seq: 0.2)
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
    assert "ALERTA CRÍTICO" in temporal["details"]["explanation"]
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


def test_orchestrator_uses_deterministic_judge_when_llm_disabled(monkeypatch):
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["behavior"])
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["temporal"])
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: SPECIALIST_OUTPUTS["identity"])
    monkeypatch.setattr(flow, "judge_tool", lambda **kwargs: {"decision": "Aprovada"})
    monkeypatch.setattr(flow, "llm_judge_tool", lambda **_kwargs: pytest.fail("LLM não esperado"))

    result = flow.orchestrate_transaction({"type": "PIX_P2P"}, use_llm_judge=False)

    assert result["decision"]["decision"] == "Aprovada"
    assert set(result["behavior"]) == {"score", "details"}


def test_orchestrator_falls_back_when_llm_raises(monkeypatch, capsys):
    monkeypatch.setattr(flow, "behavior_specialist_tool", lambda _tx: {"score": 0.1})
    monkeypatch.setattr(flow, "temporal_specialist_tool", lambda _tx: {"score": 0.2})
    monkeypatch.setattr(flow, "identity_specialist_tool", lambda _tx: {"score": 0.3})
    monkeypatch.setattr(flow, "_llm_judge_available", lambda: True)
    monkeypatch.setattr(flow, "llm_judge_tool", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(flow, "judge_tool", lambda **_kwargs: {"decision": "Aprovada"})

    result = flow.orchestrate_transaction({"type": "PIX_P2P"})

    assert result["decision"]["decision"] == "Aprovada"
    assert "offline" in capsys.readouterr().out


def test_orchestrate_batch_keeps_input_order(monkeypatch):
    seen = []

    def fake_orchestrate(tx, use_llm_judge):
        seen.append((tx["id"], use_llm_judge))
        return {"transaction": tx}

    monkeypatch.setattr(flow, "orchestrate_transaction", fake_orchestrate)
    results = flow.orchestrate_batch([{"id": 1}, {"id": 2}])

    assert [result["transaction"]["id"] for result in results] == [1, 2]
    assert seen == [(1, False), (2, False)]
