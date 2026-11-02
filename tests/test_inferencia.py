import numpy as np
import pytest
import torch

from src.ferramentas import inferencia_modelos as inference
from src.preprocessamento.carregar_dataset import LSTM_FEATURE_NAMES


class IsolationModel:
    def decision_function(self, features):
        assert features.shape == (1, 18)
        return np.array([-0.5])


class SequenceModel:
    def eval(self):
        return self

    def __call__(self, features):
        assert tuple(features.shape) == (1, 5, len(LSTM_FEATURE_NAMES))
        assert float(features[0, -1, 2]) == 1.0
        return torch.tensor([[0.65]])


class GraphModel(torch.nn.Module):
    def forward(self, features, edge_index):
        assert features.shape[1] == 4
        assert edge_index.shape[0] == 2
        return torch.full((len(features), 1), 0.4)


def transaction():
    return {
        "amount": 100.0,
        "oldbalanceOrg": 250.0,
        "newbalanceOrig": 150.0,
        "oldbalanceDest": 50.0,
        "newbalanceDest": 150.0,
        "step": 5,
        "type": "PIX_P2P",
        "nameOrig": "C1234567",
        "nameDest": "C7654321",
    }


def test_extract_behavior_features_uses_pre_execution_features():
    features = inference.extract_behavior_features(transaction())

    assert features.shape == (18,)
    assert features.dtype == np.float32
    assert features[0] == 100.0  # amount
    assert features[1] == 0.0    # is_incoming (sem B)


def test_isolation_forest_score_is_normalized():
    score = inference.isolation_forest_score(transaction(), model=IsolationModel())

    assert 0.0 < score < 1.0
    assert score == pytest.approx(1 / (1 + np.exp(-5)))


def test_load_pickle_model_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        inference.load_pickle_model("missing.pkl")


def test_lstm_empty_sequence_returns_zero():
    assert inference.lstm_sequence_score([]) == 0.0


def test_lstm_uses_injected_model(monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    history = [
        {**transaction(), "step": 4, "amount": 40.0},
        {**transaction(), "step": 5, "amount": 50.0},
        {**transaction(), "step": 3, "nameOrig": "OTHER"},
    ]

    assert inference.lstm_sequence_score(
        history,
        model=SequenceModel(),
        current_transaction=transaction(),
    ) == pytest.approx(0.65)


def test_gnn_inference_filters_same_and_future_steps():
    class CausalGraphModel(torch.nn.Module):
        def forward(self, features, edge_index):
            assert features.shape[0] == 3
            assert edge_index.shape == (2, 2)
            assert torch.equal(edge_index, torch.tensor([[1, 2], [0, 1]]))
            return torch.full((len(features), 1), 0.4)

    graph_data = {
        **transaction(),
        "sequence": [
            {**transaction(), "step": 3},
            {**transaction(), "step": 4},
            {**transaction(), "step": 5},
            {**transaction(), "step": 6},
        ],
    }

    assert inference.gnn_identity_score(
        graph_data, model=CausalGraphModel()
    ) == pytest.approx(0.4)


def test_lstm_does_not_use_a_heuristic_when_model_is_unavailable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="PyTorch é necessário"):
        inference.lstm_sequence_score([{"amount": 15_000.0}])


def test_gnn_uses_injected_model(monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    data = transaction()

    assert inference.gnn_identity_score(data, model=GraphModel()) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "data",
    [
        {"origin": "C100", "destination": "C101", "amount": 0},
        {"origin": "C999", "destination": "C001", "amount": 60_000},
        {"origin": "", "destination": "", "amount": 0},
    ],
)
def test_gnn_does_not_fall_back_to_identifier_scoring(monkeypatch, data):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="PyTorch é necessário"):
        inference.gnn_identity_score(data)


def test_gnn_reports_missing_model_instead_of_using_identifier_heuristic(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Modelo GNN não encontrado"):
        inference.gnn_identity_score(transaction())
