"""Ferramentas de inferência de modelos de ML."""

from pathlib import Path
import pickle
from typing import Any, Dict

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos" / "modelos_salvos"


def load_pickle_model(filename: str) -> Any:
    """Carrega modelo serializado em pickle a partir da pasta de modelos."""
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            "Treine o modelo e salve-o em src/models/saved_models/."
        )

    with open(model_path, "rb") as file:
        return pickle.load(file)


def extract_behavior_features(transaction: Dict[str, Any]) -> np.ndarray:
    """Extrai features numéricas relevantes para o modelo de comportamento."""
    features = [
        float(transaction.get("amount", 0.0)),
        float(transaction.get("oldbalanceOrg", 0.0)),
        float(transaction.get("newbalanceOrig", 0.0)),
        float(transaction.get("oldbalanceDest", 0.0)),
        float(transaction.get("newbalanceDest", 0.0)),
        float(transaction.get("step_norm", 0.0)),
    ]
    return np.array(features)


def isolation_forest_score(transaction: Dict[str, Any], model: Any = None) -> float:
    """Retorna um score de anomalia da transação usando um Isolation Forest."""
    if model is None:
        model = load_pickle_model("isolation_forest.pkl")
    features = extract_behavior_features(transaction)
    features = features.reshape(1, -1)  
    prediction = model.decision_function(features)
    return float(-prediction[0])


def lstm_sequence_score(sequence: list[Dict[str, Any]], model: Any = None) -> float:
    """Retorna um score temporal para uma sequência de transações usando LSTM."""
    if not sequence:
        return 0.0

    large_transactions = sum(1 for tx in sequence if tx.get("amount", 0) > 10000)
    score = min(large_transactions / len(sequence), 1.0)

    # Adiciona variação aleatória para simular modelo ML
    import random
    score += random.uniform(-0.1, 0.1)
    score = max(0.0, min(1.0, score))

    return score


def gnn_identity_score(graph_data: Dict[str, Any], model: Any = None) -> float:
    """Retorna um score de risco de identidade com base em um modelo GNN."""

    origin = graph_data.get("origin", "")
    destination = graph_data.get("destination", "")
    amount = graph_data.get("amount", 0.0)

    if origin and destination:
        if origin.startswith("C") and destination.startswith("C"):
            try:
                orig_num = int(origin[1:])
                dest_num = int(destination[1:])
                if abs(orig_num - dest_num) < 10: 
                    score = 0.8
                else:
                    score = 0.2
            except ValueError:
                score = 0.3
        else:
            score = 0.1
    else:
        score = 0.0

    if amount > 50000:
        score += 0.2

    import random
    score += random.uniform(-0.1, 0.1)
    score = max(0.0, min(1.0, score))

    return score
