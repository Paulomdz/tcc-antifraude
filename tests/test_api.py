"""Testes unitários para a API FastAPI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _DECISION_MAP

client = TestClient(app)

_VALID_PAYLOAD = {
    "step": 100,
    "type": "TRANSFER",
    "amount": 5000.0,
    "nameOrig": "C1234567",
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 5000.0,
    "nameDest": "C7654321",
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 6000.0,
    "step_norm": 0.134,
}


def _make_module(result: dict):
    """Cria módulo mock com orchestrate_transaction retornando result."""
    return type("M", (), {"orchestrate_transaction": staticmethod(lambda tx: result)})()


class TestHealthCheck:
    """Testa o endpoint GET /."""

    def test_retorna_200(self):
        assert client.get("/").status_code == 200

    def test_retorna_status_ok(self):
        assert client.get("/").json()["status"] == "ok"

    def test_retorna_mensagem(self):
        assert "message" in client.get("/").json()


class TestAnaliseTransacao:
    """Testa o endpoint POST /analise_transacao."""

    def test_aprovada_retorna_200(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.status_code == 200

    def test_decisao_aprovado_presente_na_resposta(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert data["decisao"] == "APROVADO"

    def test_resposta_tem_score_final(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert isinstance(data["score_final"], float)

    def test_resposta_tem_scores_tres_agentes(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert {"comportamental", "temporal", "identidade"} == set(data["scores"].keys())

    def test_resposta_tem_agentes_detalhados(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert set(data["agentes"].keys()) == {"comportamental", "temporal", "identidade"}

    def test_agente_tem_score_label_explicacao(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            agente = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()["agentes"]["comportamental"]
        assert {"score", "label", "explicacao"} <= set(agente.keys())

    def test_resposta_tem_justificativa(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert "justificativa" in data

    def test_decisao_bloqueado(self, sample_transaction):
        blocked = {
            "transaction": sample_transaction,
            "behavior": {"score": 0.9, "label": "behavior", "details": {"explanation": "Alto risco."}},
            "temporal": {"score": 0.85, "label": "temporal", "details": {"explanation": "Padrão anômalo."}},
            "identity": {"score": 0.7, "label": "identity", "details": {"explanation": "Conta suspeita."}},
            "decision": {"decision": "Recusada", "score": 0.9, "justification": "Recusada.", "specialist_outputs": {}},
        }
        with patch("importlib.import_module", return_value=_make_module(blocked)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert data["decisao"] == "BLOQUEADO"

    def test_decisao_revisao(self, sample_transaction):
        review = {
            "transaction": sample_transaction,
            "behavior": {"score": 0.6, "label": "behavior", "details": {"explanation": "Risco médio."}},
            "temporal": {"score": 0.55, "label": "temporal", "details": {"explanation": "Desvio leve."}},
            "identity": {"score": 0.5, "label": "identity", "details": {"explanation": "Incomum."}},
            "decision": {
                "decision": "Revisão Humana necessária",
                "score": 0.6,
                "justification": "Revisão.",
                "specialist_outputs": {},
            },
        }
        with patch("importlib.import_module", return_value=_make_module(review)):
            data = client.post("/analise_transacao", json=_VALID_PAYLOAD).json()
        assert data["decisao"] == "REVISÃO"

    def test_payload_invalido_sem_amount_retorna_422(self):
        payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "amount"}
        assert client.post("/analise_transacao", json=payload).status_code == 422

    def test_payload_minimo_apenas_amount(self):
        minimal_result = {
            "transaction": {"amount": 100.0},
            "behavior": {"score": 0.1, "label": "behavior", "details": {"explanation": "OK"}},
            "temporal": {"score": 0.1, "label": "temporal", "details": {"explanation": "OK"}},
            "identity": {"score": 0.1, "label": "identity", "details": {"explanation": "OK"}},
            "decision": {"decision": "Aprovada", "score": 0.1, "justification": "Baixo risco.", "specialist_outputs": {}},
        }
        with patch("importlib.import_module", return_value=_make_module(minimal_result)):
            assert client.post("/analise_transacao", json={"amount": 100.0}).status_code == 200

    def test_erro_na_analise_retorna_500(self):
        bad_module = type("M", (), {
            "orchestrate_transaction": staticmethod(
                lambda tx: (_ for _ in ()).throw(RuntimeError("falha interna"))
            )
        })()
        with patch("importlib.import_module", return_value=bad_module):
            assert client.post("/analise_transacao", json=_VALID_PAYLOAD).status_code == 500

    def test_modelo_nao_encontrado_retorna_503(self):
        bad_module = type("M", (), {
            "orchestrate_transaction": staticmethod(
                lambda tx: (_ for _ in ()).throw(FileNotFoundError("pkl ausente"))
            )
        })()
        with patch("importlib.import_module", return_value=bad_module):
            assert client.post("/analise_transacao", json=_VALID_PAYLOAD).status_code == 503


class TestAnaliseLote:
    """Testa o endpoint POST /analise_lote."""

    def _lote_payload(self, n: int = 2) -> dict:
        return {"transacoes": [_VALID_PAYLOAD] * n}

    def test_retorna_200_com_lote_valido(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            response = client.post("/analise_lote", json=self._lote_payload(2))
        assert response.status_code == 200

    def test_total_correto(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_lote", json=self._lote_payload(3)).json()
        assert data["total"] == 3

    def test_contagem_aprovados_correta(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_lote", json=self._lote_payload(2)).json()
        assert data["aprovados"] == 2

    def test_resultados_tem_mesmo_tamanho_que_total(self, mock_orchestrate_result):
        with patch("importlib.import_module", return_value=_make_module(mock_orchestrate_result)):
            data = client.post("/analise_lote", json=self._lote_payload(2)).json()
        assert len(data["resultados"]) == 2

    def test_lote_vazio_retorna_422(self):
        assert client.post("/analise_lote", json={"transacoes": []}).status_code == 422

    def test_erro_individual_nao_para_lote(self, mock_orchestrate_result):
        call_count = 0

        def side_effect(tx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("falha na primeira")
            return mock_orchestrate_result

        error_module = type("M", (), {"orchestrate_transaction": staticmethod(side_effect)})()
        with patch("importlib.import_module", return_value=error_module):
            data = client.post("/analise_lote", json=self._lote_payload(2)).json()
        assert len(data["resultados"]) == 2
        assert data["resultados"][0]["decisao"] == "ERRO"
        assert data["resultados"][1]["decisao"] == "APROVADO"


class TestImportError:
    """Testa o comportamento quando o módulo de orquestração não está disponível."""

    def test_importerror_analise_transacao_retorna_500(self):
        with patch("importlib.import_module", side_effect=ImportError("crewai ausente")):
            assert client.post("/analise_transacao", json=_VALID_PAYLOAD).status_code == 500

    def test_importerror_analise_lote_retorna_500(self):
        with patch("importlib.import_module", side_effect=ImportError("crewai ausente")):
            payload = {"transacoes": [_VALID_PAYLOAD]}
            assert client.post("/analise_lote", json=payload).status_code == 500


class TestDecisionMap:
    """Testa o mapeamento interno de decisões."""

    def test_aprovada_mapeia_para_aprovado(self):
        assert _DECISION_MAP["Aprovada"] == "APROVADO"

    def test_recusada_mapeia_para_bloqueado(self):
        assert _DECISION_MAP["Recusada"] == "BLOQUEADO"

    def test_revisao_mapeia_para_revisao(self):
        assert _DECISION_MAP["Revisão Humana necessária"] == "REVISÃO"
