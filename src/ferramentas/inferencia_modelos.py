"""Ferramentas de inferência de modelos de ML."""

import math
from pathlib import Path
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos" / "modelos_salvos"

try:
    import torch
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
    GNN_FEATURE_NAMES,
    IF_FEATURE_NAMES,
    LSTM_FEATURE_NAMES,
    compute_if_features,
)


def extract_behavior_features(transaction: Dict[str, Any]) -> np.ndarray:
    """Extrai features reais compatíveis com o Isolation Forest."""
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
IF_SCORE_SCALE = 10.0


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
    score = 1.0 / (1.0 + math.exp(prediction * IF_SCORE_SCALE))
    return float(score)


def lstm_sequence_score(
    sequence: list[Dict[str, Any]],
    model: Any = None,
    current_transaction: Dict[str, Any] | None = None,
) -> float:
    """Pontua a transação atual usando apenas o histórico anterior da mesma conta."""
    if not sequence:
        return 0.0

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch é necessário para inferir o score da LSTM.")
    if model is None:
        from src.modelos.arquitetura_lstm import carregar_lstm

        model, metadata = carregar_lstm(str(MODEL_DIR / "lstm_model.pt"))
    else:
        metadata = {}

    feature_names = metadata.get("feature_names", LSTM_FEATURE_NAMES)
    seq_length = int(metadata.get("seq_length", 5))
    current = current_transaction or sequence[-1]
    current_step = int(current["step"])
    account_id = str(current.get("nameOrig", ""))
    history = [
        tx
        for tx in sequence
        if account_id
        and str(tx.get("nameOrig", "")) == account_id
        and int(tx["step"]) < current_step
    ]
    history.sort(key=lambda tx: int(tx["step"]))
    selected = history[-seq_length:]
    features = []
    feature_indices = [IF_FEATURE_NAMES.index(name) for name in feature_names]
    for tx in selected:
        row_features = compute_if_features(tx)
        if not isinstance(row_features, np.ndarray):
            raise TypeError("A inferência da LSTM exige transações individuais.")
        features.append(row_features[feature_indices].astype(np.float32))
    padded = np.zeros((seq_length, len(feature_indices)), dtype=np.float32)
    if features:
        padded[-len(features) :] = np.asarray(features, dtype=np.float32)
    x = torch.from_numpy(padded).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(x)
    return float(output.clamp(0.0, 1.0).item())


def _gnn_score_do_passado(
    transacao: Dict[str, Any], model: Any, passado: Any
) -> float:
    """Subgrafo do passado (dois saltos, fanouts 10 e 5), como no lote."""
    from src.ferramentas.historico import grafo_do_passado
    from src.modelos.arquitetura_gnn import carregar_gnn

    if model is None:
        model, _metadata = carregar_gnn(str(MODEL_DIR / "gnn_model.pt"))
    model.eval()
    grafo = passado if passado is not None else grafo_do_passado(None)
    features, edge_index = grafo.subgrafo(transacao)
    device = next(model.parameters(), torch.empty(0)).device
    with torch.no_grad():
        output = model(features.to(device), edge_index.to(device))
    return float(output[0].clamp(0.0, 1.0).item())


def gnn_identity_score(
    graph_data: Dict[str, Any], model: Any = None, passado: Any = None
) -> float:
    """Retorna um score de risco de identidade com base em um modelo GNN.

    Sem ``sequence`` explícita, usa o passado do Parquet (``passado`` permite
    injetar um índice, por exemplo com limite de step para a validação).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch é necessário para inferir o score da GNN.")
    from src.modelos.arquitetura_gnn import (
        TransactionMiniBatchDataset,
        carregar_gnn,
    )

    if "sequence" not in graph_data:
        return _gnn_score_do_passado(graph_data, model, passado)
    sequence = graph_data.get("sequence", [])
    if not isinstance(sequence, list):
        raise TypeError("O histórico da GNN deve ser uma lista de transações.")
    current = {
        key: value for key, value in graph_data.items() if key != "sequence"
    }
    current_step = int(current["step"])
    history = [
        transaction
        for transaction in sequence
        if int(transaction["step"]) < current_step
    ]
    frame = pd.DataFrame([*history, current])
    if model is None:
        model, metadata = carregar_gnn(str(MODEL_DIR / "gnn_model.pt"))
    else:
        metadata = {"feature_names": GNN_FEATURE_NAMES}
    feature_names = metadata.get("feature_names", GNN_FEATURE_NAMES)
    target = frame.tail(1)
    dataset = TransactionMiniBatchDataset(
        target,
        feature_names=feature_names,
        context_df=frame,
    )
    features, edge_index, _labels, seed_count = next(
        dataset.iter_batches(batch_size=1, fanouts=(10, 5))
    )
    device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    with torch.no_grad():
        output = model(
            features.to(device),
            edge_index.to(device),
        )[:seed_count]
    return float(output[0].clamp(0.0, 1.0).item())
