from fastapi.testclient import TestClient

from src.api import main as api


client = TestClient(api.app)


def analysis_result(decision="Aprovada"):
    specialist = {
        "score": 0.2,
        "label": "especialista",
        "details": {"explanation": "Análise concluída."},
    }
    return {
        "decision": {
            "decision": decision,
            "score": 0.2,
            "justification": "Resultado de teste.",
        },
        "behavior": specialist,
        "temporal": specialist,
        "identity": specialist,
    }


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_single_transaction_endpoint(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction):
            assert transaction["type"] == "PIX_P2P"
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/v1/analise_transacao",
        json={"amount": 100, "type": "PIX_P2P"},
    )

    assert response.status_code == 200
    assert response.json()["decisao"] == "APROVADO"
    assert set(response.json()["scores"]) == {
        "comportamental",
        "temporal",
        "identidade",
    }


def test_single_transaction_reports_missing_model(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction):
            raise FileNotFoundError("isolation_forest.pkl")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/v1/analise_transacao", json={"amount": 10})

    assert response.status_code == 503
    assert "Modelo não encontrado" in response.json()["detail"]


def test_single_transaction_reports_runtime_error(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction):
            raise RuntimeError("erro controlado")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/v1/analise_transacao", json={"amount": 10})

    assert response.status_code == 500
    assert "erro controlado" in response.json()["detail"]


def test_batch_endpoint_counts_decisions_and_item_errors(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction, use_llm_judge):
            assert use_llm_judge is False
            if transaction["amount"] == 2:
                raise RuntimeError("item inválido")
            return analysis_result("Recusada")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1}, {"amount": 2}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2
    assert body["bloqueados"] == 1
    assert body["resultados"][1]["erro"] == "item inválido"


def test_batch_requires_at_least_one_transaction():
    response = client.post("/analise_lote", json={"transacoes": []})

    assert response.status_code == 422


def test_get_orchestrator_imports_the_actual_ascii_package():
    assert api._get_orchestrator().__name__ == "src.orquestacao.fluxo_crewai"
