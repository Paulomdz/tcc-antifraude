"""Configuração global do pytest: ajusta sys.path e define fixtures compartilhadas."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Garante que a raiz do projeto (tcc-antifraude/) está em sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures de dados
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame sintético no formato PaySim para testes de preprocessamento."""
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
            "amount": [5000.0, 200.0, 100.0, 50.0, 3000.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
            "oldbalanceOrg": [10000.0, 500.0, 300.0, 100.0, 0.0],
            "newbalanceOrig": [5000.0, 300.0, 200.0, 50.0, 0.0],
            "nameDest": ["C6", "C7", "C8", "C9", "C1"],
            "oldbalanceDest": [1000.0, 0.0, 0.0, 0.0, 3000.0],
            "newbalanceDest": [6000.0, 200.0, 100.0, 50.0, 6000.0],
            "isFraud": [0, 1, 0, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 0],
        }
    )


@pytest.fixture()
def sample_transaction() -> dict:
    """Transação individual no formato esperado pelos modelos."""
    return {
        "step": 100,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C1234567",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C7654321",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 6000.0,
        "step_norm": 100.0 / 744.0,
    }


@pytest.fixture()
def mock_orchestrate_result(sample_transaction) -> dict:
    """Resultado simulado de orchestrate_transaction para mocking nos testes da API."""
    return {
        "transaction": sample_transaction,
        "behavior": {
            "score": 0.3,
            "label": "behavior",
            "details": {"explanation": "Valores dentro da normalidade."},
        },
        "temporal": {
            "score": 0.4,
            "label": "temporal",
            "details": {"explanation": "Padrão temporal regular."},
        },
        "identity": {
            "score": 0.2,
            "label": "identity",
            "details": {"explanation": "Conexão entre contas sem histórico suspeito."},
        },
        "decision": {
            "decision": "Aprovada",
            "score": 0.4,
            "justification": (
                "Decisão: Aprovada. O agente comportamental indicou 0.300, "
                "o temporal indicou 0.400 e o de identidade indicou 0.200."
            ),
            "specialist_outputs": {},
        },
    }
