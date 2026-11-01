import random

import numpy as np
import pytest
import torch

from src.ferramentas import inferencia_modelos as inference


class IsolationModel:
    def decision_function(self, features):
        assert features.shape == (1, 21)
        return np.array([-0.5])


class SequenceModel:
    def __call__(self, features):
        assert tuple(features.shape) == (1, 1, 6)
        return torch.tensor([[0.65]])


class GraphModel:
    def __call__(self, features, edge_index):
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
        "step_norm": 0.25,
    }


def test_extract_behavior_features_uses_expected_twenty_one_features():
    features = inference.extract_behavior_features(transaction())

    assert features.shape == (21,)
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

    assert inference.lstm_sequence_score(
        [transaction()], model=SequenceModel()
    ) == pytest.approx(0.65)


def test_lstm_fallback_is_bounded_and_stable_with_zero_noise(monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(random, "uniform", lambda *_: 0.0)

    assert inference.lstm_sequence_score([{"amount": 15_000.0}]) == 1.0


def test_gnn_uses_injected_model(monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    data = {"origin": "C01", "destination": "C02", "amount": 100.0}

    assert inference.gnn_identity_score(data, model=GraphModel()) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"origin": "C100", "destination": "C101", "amount": 0}, 0.8),
        ({"origin": "A", "destination": "B", "amount": 60_000}, 0.3),
        ({"origin": "", "destination": "", "amount": 0}, 0.0),
    ],
)
def test_gnn_fallback_uses_identity_features(monkeypatch, data, expected):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(random, "uniform", lambda *_: 0.0)

    assert inference.gnn_identity_score(data) == pytest.approx(expected)


def test_gnn_fallback_clamps_large_amount(monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(random, "uniform", lambda *_: 0.1)

    assert inference.gnn_identity_score(
        {"origin": "C1", "destination": "C2", "amount": 80_000}
    ) == 1.0
