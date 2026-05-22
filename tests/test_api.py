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


class TestHealthCheck:
    """Testa o endpoint GET /."""

    def test_retorna_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_retorna_status_ok(self):
        response = client.get("/")
        assert response.json()["status"] == "ok"

    def test_retorna_mensagem(self):
        response = client.get("/")
        assert "message" in response.json()


class TestAnaliseTransacao:
    """Testa o endpoint POST /analise_transacao."""

    def _mock_orchestrate(self, mock_result):
        """Cria patch do importlib para retornar módulo mockado."""
        mock_module = type("FakeModule", (), {"orchestrate_transaction": lambda tx: mock_result})()
        return patch("importlib.import_module", return_value=mock_module)

    def test_aprovada_retorna_200(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.status_code == 200

    def test_decisao_aprovado_presente_na_resposta(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        data = response.json()
        assert data["decisao"] == "APROVADO"

    def test_resposta_tem_score_final(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        data = response.json()
        assert "score_final" in data
        assert isinstance(data["score_final"], float)

    def test_resposta_tem_scores_tres_agentes(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        data = response.json()
        assert "scores" in data
        assert "comportamental" in data["scores"]
        assert "temporal" in data["scores"]
        assert "identidade" in data["scores"]

    def test_resposta_tem_agentes_detalhados(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        data = response.json()
        assert "agentes" in data
        assert set(data["agentes"].keys()) == {"comportamental", "temporal", "identidade"}

    def test_agente_tem_score_label_explicacao(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        agente = response.json()["agentes"]["comportamental"]
        assert "score" in agente
        assert "label" in agente
        assert "explicacao" in agente

    def test_resposta_tem_justificativa(self, mock_orchestrate_result):
        with self._mock_orchestrate(mock_orchestrate_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert "justificativa" in response.json()

    def test_decisao_bloqueado(self, sample_transaction):
        blocked_result = {
            "transaction": sample_transaction,
            "behavior": {"score": 0.9, "label": "behavior", "details": {"explanation": "Alto risco."}},
            "temporal": {"score": 0.85, "label": "temporal", "details": {"explanation": "Padrão anômalo."}},
            "identity": {"score": 0.7, "label": "identity", "details": {"explanation": "Conta suspeita."}},
            "decision": {
                "decision": "Recusada",
                "score": 0.9,
                "justification": "Decisão: Recusada.",
                "specialist_outputs": {},
            },
        }
        with self._mock_orchestrate(blocked_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.json()["decisao"] == "BLOQUEADO"

    def test_decisao_revisao(self, sample_transaction):
        review_result = {
            "transaction": sample_transaction,
            "behavior": {"score": 0.6, "label": "behavior", "details": {"explanation": "Risco médio."}},
            "temporal": {"score": 0.55, "label": "temporal", "details": {"explanation": "Desvio leve."}},
            "identity": {"score": 0.5, "label": "identity", "details": {"explanation": "Conexão incomum."}},
            "decision": {
                "decision": "Revisão Humana necessária",
                "score": 0.6,
                "justification": "Decisão: Revisão Humana necessária.",
                "specialist_outputs": {},
            },
        }
        with self._mock_orchestrate(review_result):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.json()["decisao"] == "REVISÃO"

    def test_payload_invalido_sem_amount_retorna_422(self):
        payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "amount"}
        response = client.post("/analise_transacao", json=payload)
        assert response.status_code == 422

    def test_payload_minimo_apenas_amount(self):
        mock_result = {
            "transaction": {"amount": 100.0},
            "behavior": {"score": 0.1, "label": "behavior", "details": {"explanation": "OK"}},
            "temporal": {"score": 0.1, "label": "temporal", "details": {"explanation": "OK"}},
            "identity": {"score": 0.1, "label": "identity", "details": {"explanation": "OK"}},
            "decision": {
                "decision": "Aprovada",
                "score": 0.1,
                "justification": "Baixo risco.",
                "specialist_outputs": {},
            },
        }
        mock_module = type("M", (), {"orchestrate_transaction": lambda tx: mock_result})()
        with patch("importlib.import_module", return_value=mock_module):
            response = client.post("/analise_transacao", json={"amount": 100.0})
        assert response.status_code == 200

    def test_erro_na_analise_retorna_500(self):
        mock_module = type("M", (), {
            "orchestrate_transaction": staticmethod(lambda tx: (_ for _ in ()).throw(RuntimeError("falha interna")))
        })()
        with patch("importlib.import_module", return_value=mock_module):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.status_code == 500

    def test_modelo_nao_encontrado_retorna_503(self):
        mock_module = type("M", (), {
            "orchestrate_transaction": staticmethod(lambda tx: (_ for _ in ()).throw(FileNotFoundError("pkl ausente")))
        })()
        with patch("importlib.import_module", return_value=mock_module):
            response = client.post("/analise_transacao", json=_VALID_PAYLOAD)
        assert response.status_code == 503


class TestDecisionMap:
    """Testa o mapeamento interno de decisões."""

    def test_aprovada_mapeia_para_aprovado(self):
        assert _DECISION_MAP["Aprovada"] == "APROVADO"

    def test_recusada_mapeia_para_bloqueado(self):
        assert _DECISION_MAP["Recusada"] == "BLOQUEADO"

    def test_revisao_mapeia_para_revisao(self):
        assert _DECISION_MAP["Revisão Humana necessária"] == "REVISÃO"
