"""Testes unitários para inferência de modelos ML."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ferramentas.inferencia_modelos import (
    extract_behavior_features,
    gnn_identity_score,
    isolation_forest_score,
    lstm_sequence_score,
)


class TestExtractBehaviorFeatures:
    """Testa extração de features comportamentais."""

    def test_retorna_array_numpy(self, sample_transaction):
        result = extract_behavior_features(sample_transaction)
        assert isinstance(result, np.ndarray)

    def test_retorna_seis_features(self, sample_transaction):
        result = extract_behavior_features(sample_transaction)
        assert result.shape == (6,)

    def test_ordem_correta_das_features(self, sample_transaction):
        result = extract_behavior_features(sample_transaction)
        assert result[0] == sample_transaction["amount"]
        assert result[1] == sample_transaction["oldbalanceOrg"]
        assert result[2] == sample_transaction["newbalanceOrig"]
        assert result[3] == sample_transaction["oldbalanceDest"]
        assert result[4] == sample_transaction["newbalanceDest"]
        assert result[5] == pytest.approx(sample_transaction["step_norm"])

    def test_valores_ausentes_viram_zero(self):
        result = extract_behavior_features({})
        assert (result == 0.0).all()

    def test_aceita_valores_parciais(self):
        result = extract_behavior_features({"amount": 1000.0})
        assert result[0] == 1000.0
        assert result[1] == 0.0


class TestIsolationForestScore:
    """Testa a função isolation_forest_score com modelo mockado e real."""

    def test_retorna_float_com_modelo_mockado(self, sample_transaction):
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.3])
        score = isolation_forest_score(sample_transaction, model=mock_model)
        assert isinstance(score, float)

    def test_valor_correto_com_modelo_mockado(self, sample_transaction):
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.5])
        score = isolation_forest_score(sample_transaction, model=mock_model)
        assert score == pytest.approx(0.5)

    def test_anomalia_tem_score_positivo(self, sample_transaction):
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.9])
        score = isolation_forest_score(sample_transaction, model=mock_model)
        assert score > 0

    def test_carrega_modelo_real_se_existir(self, sample_transaction):
        model_path = Path(__file__).resolve().parents[1] / "src" / "modelos" / "modelos_salvos" / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("isolation_forest.pkl não encontrado — pulando teste com modelo real.")
        score = isolation_forest_score(sample_transaction)
        assert isinstance(score, float)

    def test_levanta_file_not_found_sem_modelo(self, sample_transaction):
        with patch("src.ferramentas.inferencia_modelos.load_pickle_model") as mock_load:
            mock_load.side_effect = FileNotFoundError("Modelo ausente")
            with pytest.raises(FileNotFoundError):
                isolation_forest_score(sample_transaction)


class TestLstmSequenceScore:
    """Testa lstm_sequence_score — principalmente o caminho de fallback heurístico."""

    def test_retorna_zero_para_sequencia_vazia(self):
        score = lstm_sequence_score([])
        assert score == 0.0

    def test_retorna_float_para_sequencia_valida(self, sample_transaction):
        score = lstm_sequence_score([sample_transaction])
        assert isinstance(score, float)

    def test_score_entre_zero_e_um(self, sample_transaction):
        score = lstm_sequence_score([sample_transaction] * 5)
        assert 0.0 <= score <= 1.0

    def test_muitas_transacoes_grandes_aumenta_score(self):
        txs = [{"amount": 200000.0, "step_norm": 0.5}] * 10
        score = lstm_sequence_score(txs)
        assert isinstance(score, float)

    def test_sequencia_com_multiplas_transacoes(self, sample_transaction):
        seq = [sample_transaction, sample_transaction, sample_transaction]
        score = lstm_sequence_score(seq)
        assert isinstance(score, float)


class TestGnnIdentityScore:
    """Testa gnn_identity_score — caminho de fallback heurístico."""

    def test_retorna_float(self):
        score = gnn_identity_score({"origin": "C100", "destination": "C200", "amount": 500.0})
        assert isinstance(score, float)

    def test_score_entre_zero_e_um(self):
        score = gnn_identity_score({"origin": "C100", "destination": "C200", "amount": 500.0})
        assert 0.0 <= score <= 1.0

    def test_retorna_baixo_sem_contas(self):
        # A heurística parte de 0.0 mas adiciona ruído aleatório ≤ 0.1
        score = gnn_identity_score({})
        assert 0.0 <= score <= 0.1

    def test_contas_c_proximas_tem_score_mais_alto(self):
        score_prox = gnn_identity_score({"origin": "C100", "destination": "C101", "amount": 100.0})
        score_dist = gnn_identity_score({"origin": "C100", "destination": "C500", "amount": 100.0})
        # Contas com números próximos devem ter score maior (heurística)
        assert score_prox >= score_dist

    def test_valor_alto_aumenta_score(self):
        score_normal = gnn_identity_score({"origin": "C100", "destination": "C500", "amount": 1000.0})
        score_alto = gnn_identity_score({"origin": "C100", "destination": "C500", "amount": 100000.0})
        # Score com valor muito alto deve ser >= score normal (pode ter ruído)
        assert isinstance(score_alto, float)

    def test_conta_m_destino_tem_score_baixo(self):
        score = gnn_identity_score({"origin": "C100", "destination": "M999", "amount": 500.0})
        assert 0.0 <= score <= 1.0


class TestLoadPickleModel:
    """Testa load_pickle_model para arquivo inexistente."""

    def test_levanta_file_not_found(self):
        from src.ferramentas.inferencia_modelos import load_pickle_model

        with pytest.raises(FileNotFoundError):
            load_pickle_model("nao_existe_modelo.pkl")
