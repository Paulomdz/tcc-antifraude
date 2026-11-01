import pandas as pd
import pytest
import torch

from src.modelos import arquitetura_gnn as gnn
from src.modelos import arquitetura_lstm as lstm


@pytest.mark.skipif(not lstm.TORCH_AVAILABLE, reason="PyTorch não instalado")
def test_lstm_trains_and_round_trips_state_dict(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "step": step,
                "amount": float(step * 10),
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 90.0,
                "oldbalanceDest": 20.0,
                "newbalanceDest": 30.0,
                "step_norm": step / 743,
                "isFraud": step % 2,
            }
            for step in range(12)
        ]
    )

    model = lstm.treinar_lstm(sample, epochs=1, batch_size=2)
    model_path = tmp_path / "lstm.pt"
    lstm.salvar_lstm(model, str(model_path))
    restored = lstm.carregar_lstm(str(model_path))

    assert model is not None
    assert isinstance(restored, lstm.LSTMFraudDetector)
    assert restored(torch.zeros(1, 10, 6)).shape == (1, 1)


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_trains_and_round_trips_model_and_account_mapping(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "nameOrig": f"C{index}",
                "nameDest": f"C{index + 1}",
                "amount": 100.0 + index,
                "isFraud": index % 2,
            }
            for index in range(4)
        ]
    )

    model, graph_dataset = gnn.treinar_gnn(sample, epochs=1)
    model_path = tmp_path / "gnn.pt"
    gnn.salvar_gnn(model, graph_dataset, str(model_path))
    restored, mapping = gnn.carregar_gnn(str(model_path))

    assert model is not None
    assert isinstance(restored, gnn.GNNFraudDetector)
    assert mapping["account_to_idx"] == graph_dataset.account_to_idx
