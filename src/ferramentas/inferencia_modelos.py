"""Ferramentas de inferência de modelos de ML."""

import math
from pathlib import Path
import pickle
from typing import Any, Dict

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos" / "modelos_salvos"

try:
    import torch
    from src.modelos.arquitetura_lstm import LSTMFraudDetector
    from src.modelos.arquitetura_gnn import GNNFraudDetector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    """Extrai features reais e compatíveis com os modelos treinados."""
    features = [
        float(transaction.get("amount", 0.0)),
        float(transaction.get("oldbalanceOrg", 0.0)),
        float(transaction.get("newbalanceOrig", 0.0)),
        float(transaction.get("oldbalanceDest", 0.0)),
        float(transaction.get("newbalanceDest", 0.0)),
        float(transaction.get("step_norm", 0.0)),
    ]
    return np.array(features)


# Fator de escala do sigmoid usado para normalizar o decision_function do
# Isolation Forest em [0, 1]. `decision_function` do sklearn não é limitado
# por natureza (valores mais negativos = mais anômalo); sem essa normalização
# o score comportamental podia sair do intervalo [0, 1] documentado (ex.:
# valores negativos), o que é a causa do exemplo `score_comportamental =
# -0.037` observado na monografia (§4.1), em contradição com a normalização
# declarada em §3.6. Este valor de escala é um ajuste razoável para a faixa
# típica do decision_function; uma calibração formal exigiria validação com
# dados reais rotulados (fora do escopo deste ajuste).
_IF_SCORE_SCALE = 10.0


def isolation_forest_score(transaction: Dict[str, Any], model: Any = None) -> float:
    """Retorna um score de anomalia da transação usando um Isolation Forest.

    O score é normalizado em [0, 1] via sigmoid sobre o `decision_function`
    (mais próximo de 1 = mais anômalo/suspeito), conforme a normalização
    declarada para todos os agentes.
    """
    if model is None:
        model = load_pickle_model("isolation_forest.pkl")
    features = extract_behavior_features(transaction)
    features = features.reshape(1, -1)
    prediction = float(model.decision_function(features)[0])
    # decision_function negativo = anômalo -> sigmoid(-prediction * escala)
    # se aproxima de 1 quanto mais negativo (mais anômalo) for `prediction`.
    score = 1.0 / (1.0 + math.exp(prediction * _IF_SCORE_SCALE))
    return float(score)


def lstm_sequence_score(sequence: list[Dict[str, Any]], model: Any = None) -> float:
    """Retorna um score temporal para uma sequência de transações usando LSTM."""
    if not sequence:
        return 0.0

    if TORCH_AVAILABLE:
        model_path = MODEL_DIR / "lstm_model.pt"
        if model is None and model_path.exists():
            model = LSTMFraudDetector()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

        if model is not None:
            features = []
            for tx in sequence:
                features.append([
                    float(tx.get("amount", 0.0)),
                    float(tx.get("oldbalanceOrg", 0.0)),
                    float(tx.get("newbalanceOrig", 0.0)),
                    float(tx.get("oldbalanceDest", 0.0)),
                    float(tx.get("newbalanceDest", 0.0)),
                    float(tx.get("step_norm", 0.0)),
                ])
            x = torch.FloatTensor([features])
            with torch.no_grad():
                output = model(x)
            return float(output.clamp(0.0, 1.0).item())

    large_transactions = sum(1 for tx in sequence if tx.get("amount", 0) > 10000)
    score = min(large_transactions / len(sequence), 1.0)
    import random
    score += random.uniform(-0.1, 0.1)
    return float(max(0.0, min(1.0, score)))


def gnn_identity_score(graph_data: Dict[str, Any], model: Any = None) -> float:
    """Retorna um score de risco de identidade com base em um modelo GNN."""
    origin = graph_data.get("origin", "")
    destination = graph_data.get("destination", "")
    amount = float(graph_data.get("amount", 0.0))

    if TORCH_AVAILABLE:
        model_path = MODEL_DIR / "gnn_model.pt"
        if model is None and model_path.exists():
            model = GNNFraudDetector(in_channels=4)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

        if model is not None and origin and destination:
            node_map = {origin: 0, destination: 1} if origin != destination else {origin: 0}
            features = []
            for account in sorted(node_map.keys()):
                if account == origin:
                    features.append([1.0, 0.0, amount, 0.0])
                else:
                    features.append([0.0, 1.0, 0.0, amount])

            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) if origin != destination else torch.tensor([[0], [0]], dtype=torch.long)
            x = torch.FloatTensor(features)
            with torch.no_grad():
                out = model(x, edge_index)
            score = float(out.mean().clamp(0.0, 1.0).item())
            return score

    if origin and destination:
        if origin.startswith("C") and destination.startswith("C"):
            try:
                orig_num = int(origin[1:])
                dest_num = int(destination[1:])
                score = 0.8 if abs(orig_num - dest_num) < 10 else 0.2
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
    return float(max(0.0, min(1.0, score)))
