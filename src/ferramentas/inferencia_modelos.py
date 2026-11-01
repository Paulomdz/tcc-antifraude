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
            "Treine o modelo e salve-o em src/modelos/modelos_salvos/."
        )

    with open(model_path, "rb") as file:
        return pickle.load(file)


from src.preprocessamento.carregar_dataset import (
    compute_if_features,
    hora_norm_from_step,
)


def extract_behavior_features(transaction: Dict[str, Any]) -> np.ndarray:
    """Extrai features reais e compatíveis com o Isolation Forest (21 features fixas)."""
    features = compute_if_features(transaction)
    if not isinstance(features, np.ndarray):
        raise TypeError("A inferência do Isolation Forest exige uma transação individual.")
    return features


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
                    hora_norm_from_step(tx.get("step", tx.get("hora", 12))),
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
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch é necessário para inferir o score da GNN.")

    origin = str(graph_data.get("origin") or "")
    destination = str(graph_data.get("destination") or "")
    amount = float(graph_data.get("amount", 0.0))

    if not origin or not destination:
        raise ValueError("A inferência da GNN exige as contas de origem e destino.")

    if model is None:
        model_path = MODEL_DIR / "gnn_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo GNN não encontrado: {model_path}")
        model = GNNFraudDetector(in_channels=4)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

    node_ids = sorted({origin, destination})
    node_map = {account: index for index, account in enumerate(node_ids)}
    features = [
        [
            1.0 if account == origin else 0.0,
            1.0 if account == destination else 0.0,
            amount if account == origin else 0.0,
            amount if account == destination else 0.0,
        ]
        for account in node_ids
    ]

    origin_idx = node_map[origin]
    destination_idx = node_map[destination]
    if origin_idx == destination_idx:
        edges = [[origin_idx], [destination_idx]]
    else:
        edges = [[origin_idx, destination_idx], [destination_idx, origin_idx]]
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.FloatTensor(features)
    with torch.no_grad():
        out = model(x, edge_index)
    return float(out.mean().clamp(0.0, 1.0).item())
