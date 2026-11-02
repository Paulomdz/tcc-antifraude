import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.preprocessamento.carregar_dataset import (
    IF_FEATURE_NAMES,
    extract_if_features_row,
)


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def temporary_history_db(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "historico.db")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def analysis_result(decision="Aprovada"):
    specialist = {
        "score": 0.2,
        "label": "especialista",
        "details": {"explanation": "Análise concluída."},
    }
    return {
        "modo": "fast",
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
    assert response.json()["message"] == f"{api.SERVICE_NAME} API operacional"
    assert api.app.title == api.SERVICE_NAME
    assert api.SERVICE_NAME in api.app.description


@pytest.mark.parametrize("api_key", ["", "   "])
def test_production_requires_gemini_key_with_exact_503(monkeypatch, api_key):
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    response = client.post("/analise_transacao", json={"amount": 100})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Gemini não configurado. Defina GOOGLE_API_KEY no .env."
    }


def test_single_transaction_uses_production_and_persists(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction, modo):
            assert transaction["type"] == "PIX_P2P"
            assert modo == "production"
            result = analysis_result()
            result["modo"] = modo
            return result

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/analise_transacao", json={"amount": 100})

    assert response.status_code == 200
    assert response.json()["decisao"] == "APROVADO"
    assert response.json()["modo"] == "production"
    assert response.json()["explicabilidade"]["top3"] == []
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analises").fetchone()[0] == 1


def test_single_transaction_reports_missing_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction, modo):
            assert modo == "production"
            raise FileNotFoundError("isolation_forest.pkl")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/analise_transacao", json={"amount": 10})

    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo treinado não encontrado."


def test_single_transaction_reports_runtime_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction, modo):
            raise RuntimeError(f"erro controlado ({modo})")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/analise_transacao", json={"amount": 10})

    assert response.status_code == 502
    assert response.json()["detail"] == "Falha ao executar análise production."


def test_batch_uses_fast_and_persists_item_errors(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
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
    assert body["erros"] == 1
    assert (
        body["aprovados"] + body["revisoes"] + body["bloqueados"] + body["erros"]
        == body["total"]
    )
    assert body["resultados"][1]["erro"] == "item inválido"
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analises").fetchone()[0] == 2


def test_batch_of_ten_works_without_api_key_and_counts_every_transaction(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            decision = ["Aprovada", "Revisão Humana necessária", "Recusada"][
                int(transaction["amount"] - 1) % 3
            ]
            return analysis_result(decision)

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": amount} for amount in range(1, 11)]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 10
    assert (
        body["aprovados"] + body["revisoes"] + body["bloqueados"] + body["erros"]
        == body["total"]
    )
    assert body["erros"] == 0


def test_batch_counts_invalid_transaction_as_error(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            if transaction["amount"] == 9999:
                raise ValueError("transação inválida")
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1}, {"amount": 9999}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["erros"] == 1
    assert body["resultados"][1]["decisao"] == "ERRO"
    assert body["resultados"][1]["erro"] == "transação inválida"
    assert (
        body["aprovados"] + body["revisoes"] + body["bloqueados"] + body["erros"]
        == body["total"]
    )


def test_batch_continues_after_history_database_failure(monkeypatch, caplog):
    original_save = api.salvar_analise
    attempts = 0

    def fail_first_save(registro, db_path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("banco indisponível")
        return original_save(registro, db_path)

    class FakeOrchestrator:
        def orchestrate_ml_only(self, _transaction):
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "salvar_analise", fail_first_save)
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1}, {"amount": 2}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2
    assert body["aprovados"] == 2
    assert body["erros"] == 0
    assert attempts == 2
    assert "Falha ao gravar histórico da transação 0 do lote" in caplog.text
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analises").fetchone()[0] == 1


def test_unknown_decision_is_an_explicit_item_error(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, _transaction):
            return analysis_result("DECISAO_DESCONHECIDA")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post("/analise_lote", json={"transacoes": [{"amount": 1}]})

    body = response.json()
    assert response.status_code == 200
    assert body["erros"] == 1
    assert body["resultados"][0]["decisao"] == "ERRO"
    assert "Decisão desconhecida" in body["resultados"][0]["erro"]


def test_feature_hour_uses_dataset_step_modulo():
    hora_norm_index = IF_FEATURE_NAMES.index("hora_norm")

    features = extract_if_features_row({"step": 25})

    assert features[hora_norm_index] == pytest.approx(1 / 23)


def test_batch_requires_at_least_one_transaction():
    response = client.post("/analise_lote", json={"transacoes": []})
    assert response.status_code == 422


def test_history_and_statistics_work_without_gemini_key():
    recent = api._transaction_hash({"case": "history"})
    api.salvar_analise(
        {
            "transaction_hash": recent,
            "type": "PIX_P2P",
            "amount": 10,
            "score_final": 0.4,
            "decisao": "REVISÃO",
            "duracao_ms": 12,
            "llm_usado": "NAO",
        },
        db_path=api.DB_PATH,
    )

    history = client.get("/historico")
    stats = client.get("/historico/estatisticas")

    assert history.status_code == 200
    assert history.json()[0]["decisao"] == "REVISÃO"
    assert stats.status_code == 200
    assert stats.json()["total_transacoes"] == 1
    assert stats.json()["contagem_por_decisao"]["REVISÃO"] == 1
    filtered = client.get("/historico?limit=1&decisao=REVISÃO")
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
    with sqlite3.connect(api.DB_PATH) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(analises)")
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(analises)")
        }
    assert columns == {
        "id",
        "timestamp",
        "transaction_hash",
        "type",
        "amount",
        "score_final",
        "decisao",
        "score_behavior",
        "score_temporal",
        "score_identity",
        "duracao_ms",
        "llm_usado",
        "erro",
    }
    assert {"idx_analises_timestamp", "idx_analises_decisao"} <= indexes


def test_get_orchestrator_imports_the_actual_ascii_package():
    assert api._get_orchestrator().__name__ == "src.orquestacao.fluxo_multiagente"
