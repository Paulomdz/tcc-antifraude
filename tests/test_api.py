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


def lot_file_content(delimiter=",", row_count=3, include_label=False):
    columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    if include_label:
        columns.append("isFraud")
    rows = [delimiter.join(columns)]
    for index in range(row_count):
        values = [
            str(index + 1),
            "PIX_P2P",
            str(index + 1),
            f"C{index:07d}",
            "100",
            "90",
            f"C{index + 100:07d}",
            "0",
            "10",
        ]
        if include_label:
            values.append(str(index % 2))
        rows.append(delimiter.join(values))
    return "\n".join(rows).encode("utf-8")


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    expected_name = "Sistema Multiagente de IA para Detecção de Fraudes"
    assert response.json()["message"] == expected_name
    assert api.app.title == expected_name
    assert api.app.description == expected_name


@pytest.mark.parametrize("api_key", ["", "   "])
def test_production_requires_gemini_key_with_exact_503(monkeypatch, api_key):
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    response = client.post(
        "/analise_transacao", json={"amount": 100, "oldbalanceOrg": 1000}
    )

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
    response = client.post(
        "/analise_transacao", json={"amount": 100, "oldbalanceOrg": 1000}
    )

    assert response.status_code == 200
    assert response.json()["decisao"] == "APROVADO"
    assert response.json()["modo"] == "production"
    assert len(response.json()["explicabilidade"]["top3"]) == 3
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analises").fetchone()[0] == 1


def test_single_transaction_retorna_top3_preenchido(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction, modo):
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_transacao",
        json={"amount": 100, "oldbalanceOrg": 1000, "step": 12},
    )

    assert response.status_code == 200
    top3 = response.json()["explicabilidade"]["top3"]
    assert len(top3) == 3
    for item in top3:
        assert set(item) == {"nome", "valor", "z", "direcao", "desvio_zero"}
        assert item["nome"] != "log_amount"
    magnitudes = [abs(item["z"]) for item in top3]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top3_com_feature_stats_ausente_retorna_vazio_e_loga(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction, modo):
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "FEATURE_STATS_PATH", tmp_path / "ausente.json")
    api._carregar_feature_stats.cache_clear()
    try:
        response = client.post(
            "/analise_transacao", json={"amount": 100, "oldbalanceOrg": 1000}
        )
    finally:
        api._carregar_feature_stats.cache_clear()

    assert response.status_code == 200
    assert response.json()["explicabilidade"]["top3"] == []
    assert "Não foi possível calcular o top-3" in caplog.text
    assert "FileNotFoundError" in caplog.text


def test_lote_nao_calcula_top3(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            return analysis_result()

    def fail(*_args, **_kwargs):
        raise AssertionError("o lote não deve calcular top-3")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "top3_por_zscore", fail)
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 100, "oldbalanceOrg": 1000}]},
    )

    assert response.status_code == 200
    assert response.json()["erros"] == 0
    assert "explicabilidade" not in response.json()["resultados"][0]


def test_analise_transacao_sem_saldo_retorna_422(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    def fail():
        raise AssertionError("o orquestrador não deve ser chamado")

    monkeypatch.setattr(api, "_get_orchestrator", fail)
    response = client.post("/analise_transacao", json={"amount": 100})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "oldbalanceOrg"]


def test_lote_com_item_sem_saldo_retorna_422_para_o_lote(monkeypatch):
    def fail():
        raise AssertionError("o orquestrador não deve ser chamado")

    monkeypatch.setattr(api, "_get_orchestrator", fail)
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1, "oldbalanceOrg": 1000}, {"amount": 2}]},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "body",
        "transacoes",
        1,
        "oldbalanceOrg",
    ]


def test_single_transaction_reports_missing_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction, modo):
            assert modo == "production"
            raise FileNotFoundError("isolation_forest.pkl")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_transacao", json={"amount": 10, "oldbalanceOrg": 1000}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo treinado não encontrado."


def test_single_transaction_reports_runtime_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    class FakeOrchestrator:
        def orchestrate_transaction(self, _transaction, modo):
            raise RuntimeError(f"erro controlado ({modo})")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_transacao", json={"amount": 10, "oldbalanceOrg": 1000}
    )

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
        json={
            "transacoes": [
                {"amount": 1, "oldbalanceOrg": 1000},
                {"amount": 2, "oldbalanceOrg": 1000},
            ]
        },
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
        json={
            "transacoes": [
                {"amount": amount, "oldbalanceOrg": 1000}
                for amount in range(1, 11)
            ]
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 10
    assert (
        body["aprovados"] + body["revisoes"] + body["bloqueados"] + body["erros"]
        == body["total"]
    )
    assert body["erros"] == 0


def test_batch_rejects_more_than_the_single_configured_limit():
    response = client.post(
        "/analise_lote",
        json={
            "transacoes": [{"amount": 1, "oldbalanceOrg": 1000}]
            * (api.LIMITE_LOTE + 1)
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Lote máximo de 500 transações."}


def test_batch_accepts_exactly_the_configured_limit(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, _transaction):
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "_save_analysis", lambda *_args, **_kwargs: None)
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1, "oldbalanceOrg": 1000}] * api.LIMITE_LOTE},
    )

    assert response.status_code == 200
    assert response.json()["total"] == api.LIMITE_LOTE
    assert response.json()["llm_usado"] == "NAO"


def test_fast_batch_is_identical_with_and_without_api_key(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            return analysis_result(
                "Aprovada" if transaction["amount"] == 1 else "Recusada"
            )

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "_save_analysis", lambda *_args, **_kwargs: None)
    payload = {
        "transacoes": [
            {"amount": 1, "oldbalanceOrg": 1000},
            {"amount": 2, "oldbalanceOrg": 1000},
        ]
    }

    without_key = client.post("/analise_lote", json=payload)
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-test-value-not-a-credential")
    with_key = client.post("/analise_lote", json=payload)

    assert without_key.status_code == with_key.status_code == 200
    assert without_key.json() == with_key.json()
    assert with_key.json()["llm_usado"] == "NAO"


def test_config_reports_model_and_key_availability_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    unavailable = client.get("/config")
    assert unavailable.status_code == 200
    assert unavailable.json() == {
        "llm_disponivel": False,
        "modelo_llm": "gemini-3.5-flash-lite",
        "limite_lote": 500,
    }

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-test-value-not-a-credential")
    available = client.get("/config")
    assert available.status_code == 200
    assert available.json()["llm_disponivel"] is True
    assert available.json()["modelo_llm"] == "gemini-3.5-flash-lite"
    assert "GOOGLE_API_KEY" not in available.text
    assert "synthetic-test-value-not-a-credential" not in available.text


def test_csv_upload_uses_fixed_seed_and_reports_sample_size(monkeypatch):
    observed_amounts = []

    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            observed_amounts.append(transaction["amount"])
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "_save_analysis", lambda *_args, **_kwargs: None)
    file_content = lot_file_content(row_count=8, include_label=True)

    responses = [
        client.post(
            "/analise_lote/arquivo",
            params={"quantidade": 4},
            files={"arquivo": ("transacoes.csv", file_content, "text/csv")},
        )
        for _ in range(2)
    ]

    assert all(response.status_code == 200 for response in responses)
    first, second = [response.json() for response in responses]
    assert observed_amounts[:4] == observed_amounts[4:]
    assert first["quantidade_solicitada"] == 4
    assert first["quantidade_usada"] == 4
    assert first["total_arquivo"] == 8
    assert first["seed"] == 42
    assert first["arquivo_menor_que_quantidade"] is False
    assert first["llm_usado"] == "NAO"
    assert first == second


def test_txt_upload_uses_all_rows_when_file_is_smaller_than_requested(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, _transaction):
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    monkeypatch.setattr(api, "_save_analysis", lambda *_args, **_kwargs: None)
    response = client.post(
        "/analise_lote/arquivo",
        files={
            "arquivo": (
                "transacoes.txt",
                lot_file_content(";", row_count=2),
                "text/plain",
            )
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2
    assert body["quantidade_solicitada"] == 300
    assert body["quantidade_usada"] == 2
    assert body["total_arquivo"] == 2
    assert body["seed"] == 42
    assert body["arquivo_menor_que_quantidade"] is True


def test_upload_requires_documented_columns_and_separator():
    missing_columns = client.post(
        "/analise_lote/arquivo",
        files={"arquivo": ("invalido.csv", b"step,type\n1,PIX_P2P", "text/csv")},
    )
    wrong_separator = client.post(
        "/analise_lote/arquivo",
        files={
            "arquivo": (
                "invalido.csv",
                lot_file_content(";", row_count=1),
                "text/csv",
            )
        },
    )

    assert missing_columns.status_code == 422
    assert "Colunas obrigatórias ausentes" in missing_columns.json()["detail"]
    assert wrong_separator.status_code == 422
    assert "deve usar o separador" in wrong_separator.json()["detail"]


def test_batch_counts_invalid_transaction_as_error(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            if transaction["amount"] == 9999:
                raise ValueError("transação inválida")
            return analysis_result()

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_lote",
        json={
            "transacoes": [
                {"amount": 1, "oldbalanceOrg": 1000},
                {"amount": 9999, "oldbalanceOrg": 1000},
            ]
        },
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
        json={
            "transacoes": [
                {"amount": 1, "oldbalanceOrg": 1000},
                {"amount": 2, "oldbalanceOrg": 1000},
            ]
        },
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
    response = client.post(
        "/analise_lote",
        json={"transacoes": [{"amount": 1, "oldbalanceOrg": 1000}]},
    )

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


def test_analise_individual_fast_nao_exige_chave_e_nao_usa_gemini(monkeypatch):
    chamadas = []

    class FakeOrchestrator:
        def orchestrate_ml_only(self, transaction):
            chamadas.append(transaction["type"])
            return analysis_result()

        def orchestrate_transaction(self, *_args, **_kwargs):
            raise AssertionError("fast não deve chamar o juiz Gemini")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_transacao?modo=fast", json={"amount": 100, "oldbalanceOrg": 1000}
    )

    assert response.status_code == 200
    corpo = response.json()
    assert corpo["modo"] == "fast" and corpo["decisao"] == "APROVADO"
    assert len(corpo["explicabilidade"]["top3"]) == 3
    assert chamadas == ["PIX_P2P"]
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT llm_usado FROM analises").fetchone()[0] == "NAO"


def test_analise_individual_modo_invalido_retorna_422():
    response = client.post(
        "/analise_transacao?modo=turbo", json={"amount": 100, "oldbalanceOrg": 1000}
    )
    assert response.status_code == 422


def test_analise_individual_fast_falha_do_orquestrador_retorna_502(monkeypatch):
    class FakeOrchestrator:
        def orchestrate_ml_only(self, _transaction):
            raise RuntimeError("falhou")

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    response = client.post(
        "/analise_transacao?modo=fast", json={"amount": 100, "oldbalanceOrg": 1000}
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "Falha ao executar análise fast."


def test_analise_individual_deep_exige_chave_e_chama_deep(monkeypatch):
    resposta = client.post(
        "/analise_transacao?modo=deep", json={"amount": 100, "oldbalanceOrg": 1000}
    )
    assert resposta.status_code == 503  # sem chave, como production

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    pedidos = []

    class FakeOrchestrator:
        def orchestrate_transaction(self, transaction, modo):
            pedidos.append(modo)
            return {**analysis_result(), "modo": modo}

    monkeypatch.setattr(api, "_get_orchestrator", lambda: FakeOrchestrator())
    resposta = client.post(
        "/analise_transacao?modo=deep", json={"amount": 100, "oldbalanceOrg": 1000}
    )
    assert resposta.status_code == 200 and resposta.json()["modo"] == "deep"
    assert pedidos == ["deep"]
    with sqlite3.connect(api.DB_PATH) as connection:
        assert connection.execute("SELECT llm_usado FROM analises").fetchone()[0] == "SIM"
