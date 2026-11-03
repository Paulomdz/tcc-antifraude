import math

import pandas as pd
import pytest
import torch

from src.modelos import arquitetura_gnn as gnn
from src.modelos import arquitetura_lstm as lstm
from src.preprocessamento.carregar_dataset import (
    GNN_FEATURE_NAMES,
    LSTM_FEATURE_NAMES,
)


@pytest.mark.skipif(not lstm.TORCH_AVAILABLE, reason="PyTorch não instalado")
def test_lstm_trains_and_round_trips_state_dict(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "step": step,
                "amount": float(step * 10),
                "type": "PIX_P2P",
                "nameOrig": f"C{step:07d}",
                "nameDest": f"C{step + 1:07d}",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 90.0,
                "oldbalanceDest": 20.0,
                "newbalanceDest": 30.0,
                "isFraud": step % 2,
            }
            for step in range(12)
        ]
    )

    model = lstm.treinar_lstm(sample, epochs=1, batch_size=2)
    sample_dataset = lstm.TransactionSequenceDataset(sample)
    sequence, _label = sample_dataset[0]
    model_path = tmp_path / "lstm.pt"
    lstm.salvar_lstm(model, str(model_path))
    restored, metadata = lstm.carregar_lstm(str(model_path))

    assert model is not None
    assert isinstance(restored, lstm.LSTMFraudDetector)
    assert metadata == {}
    assert sequence.shape == (5, len(LSTM_FEATURE_NAMES))
    assert not hasattr(sample_dataset, "window_indices")
    assert restored(torch.zeros(1, 5, len(LSTM_FEATURE_NAMES))).shape == (1, 1)


@pytest.mark.skipif(not lstm.TORCH_AVAILABLE, reason="PyTorch não instalado")
def test_lstm_windows_only_include_strictly_prior_steps_from_same_account():
    sample = pd.DataFrame(
        [
            {"step": 1, "amount": 10.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "X", "oldbalanceOrg": 20.0, "isFraud": 0},
            {"step": 2, "amount": 20.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "Y", "oldbalanceOrg": 20.0, "isFraud": 0},
            {"step": 2, "amount": 30.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "Z", "oldbalanceOrg": 20.0, "isFraud": 0},
            {"step": 3, "amount": 40.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "X", "oldbalanceOrg": 20.0, "isFraud": 1},
            {"step": 3, "amount": 99.0, "type": "PIX_P2P", "nameOrig": "B",
             "nameDest": "X", "oldbalanceOrg": 20.0, "isFraud": 0},
        ]
    )
    dataset = lstm.TransactionSequenceDataset(
        sample, seq_length=5, feature_cols=["log_amount"]
    )

    first_account_step_2, _ = dataset[1]
    same_step_sibling, _ = dataset[2]
    other_account, _ = dataset[4]
    step_3, _ = dataset[3]

    assert torch.count_nonzero(dataset[0][0]) == 0
    assert torch.count_nonzero(first_account_step_2) == 1
    assert torch.count_nonzero(same_step_sibling) == 1
    assert torch.count_nonzero(other_account) == 0
    assert torch.count_nonzero(step_3) == 3


@pytest.mark.skipif(not lstm.TORCH_AVAILABLE, reason="PyTorch não instalado")
def test_lstm_janela_so_com_passado_estrito():
    steps = [1, 1, 2, 3, 3, 5]
    sample = pd.DataFrame(
        [
            {
                "step": step,
                "amount": math.expm1(line_index + 1),
                "type": "PIX_P2P",
                "nameOrig": "A",
                "nameDest": "X",
                "oldbalanceOrg": 20.0,
                "isFraud": 0,
            }
            for line_index, step in enumerate(steps)
        ]
    )
    dataset = lstm.TransactionSequenceDataset(
        sample, feature_cols=["log_amount"]
    )

    account_start = dataset.account_offsets[0]
    account_end = dataset.account_offsets[1]
    account_steps = dataset.steps[dataset.account_members[account_start:account_end]]
    assert account_steps.tolist() == sorted(steps)

    for target_index, target_step in enumerate(steps):
        window, _label = dataset[target_index]
        source_indices = [
            round(float(value)) - 1 for value in window[:, 0] if value > 0
        ]

        assert all(steps[source_index] < target_step for source_index in source_indices)
        assert target_index not in source_indices


@pytest.mark.skipif(not lstm.TORCH_AVAILABLE, reason="PyTorch não instalado")
def test_lstm_split_targets_can_use_only_their_earlier_context():
    sample = pd.DataFrame(
        [
            {"step": 1, "amount": 10.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "X", "oldbalanceOrg": 20.0, "isFraud": 0},
            {"step": 2, "amount": 20.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "Y", "oldbalanceOrg": 20.0, "isFraud": 0},
            {"step": 3, "amount": 30.0, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": "Z", "oldbalanceOrg": 20.0, "isFraud": 1},
        ],
        index=[10, 11, 12],
    )
    targets = sample.loc[[12]]
    context = sample.drop(columns=["isFraud"])
    dataset = lstm.TransactionSequenceDataset(
        targets, feature_cols=["log_amount"], context_df=context
    )
    sequence, label = dataset[0]

    assert torch.count_nonzero(sequence) == 2
    assert label.item() == 1


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_trains_and_round_trips_model_and_account_mapping(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "step": index,
                "type": "PIX_P2P",
                "nameOrig": f"C{index}",
                "nameDest": f"C{index + 1}",
                "amount": 100.0 + index,
                "oldbalanceOrg": 200.0,
                "newbalanceOrig": 100.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": index % 2,
            }
            for index in range(4)
        ]
    )

    model, graph_dataset = gnn.treinar_gnn(sample, epochs=1)
    model_path = tmp_path / "gnn.pt"
    gnn.salvar_gnn(model, graph_dataset, str(model_path))
    restored, metadata = gnn.carregar_gnn(str(model_path))

    assert model is not None
    assert isinstance(restored, gnn.GNNFraudDetector)
    assert metadata["feature_names"] == graph_dataset.feature_names
    assert graph_dataset.feature_names == GNN_FEATURE_NAMES
    assert not hasattr(graph_dataset, "x")
    batch_features, _edge_index, _labels, _seed_count = next(
        graph_dataset.iter_batches(batch_size=4, fanouts=(1,))
    )
    assert batch_features.shape == (4, len(GNN_FEATURE_NAMES))


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_minibatches_build_only_local_subgraphs():
    sample = pd.DataFrame(
        [
            {
                "step": index,
                "type": "PIX_P2P",
                "nameOrig": f"C{index // 2:07d}",
                "nameDest": f"C{(index + 1) // 2:07d}",
                "amount": 100.0 + index,
                "oldbalanceOrg": 200.0,
                "newbalanceOrig": 100.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": index % 2,
            }
            for index in range(8)
        ]
    )
    dataset = gnn.TransactionMiniBatchDataset(sample)
    assert not hasattr(dataset, "x")

    batches = list(dataset.iter_batches(batch_size=2, fanouts=(2, 2)))

    assert not hasattr(dataset, "edge_index")
    assert len(batches) == 4
    for features, edge_index, labels, seed_count in batches:
        assert features.dtype == torch.float32
        assert seed_count == 2
        assert labels.shape == (seed_count, 1)
        assert edge_index.shape[0] == 2
        if edge_index.numel():
            assert int(edge_index.max()) < features.shape[0]
            assert features.shape[0] <= dataset.num_nodes


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_neighbors_are_strictly_past_and_edges_point_forward():
    sample = pd.DataFrame(
        [
            {"step": step, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": f"D{index}", "amount": 10.0 + index,
             "oldbalanceOrg": 100.0, "isFraud": 0}
            for index, step in enumerate((1, 2, 2, 3))
        ]
    )
    dataset = gnn.TransactionMiniBatchDataset(
        sample.iloc[[3]], context_df=sample
    )
    features, edges, _labels, seed_count = next(
        dataset.iter_batches(batch_size=1, fanouts=(1, 1))
    )

    assert list(dataset._neighbors(1)) == [0]
    assert list(dataset._neighbors(2)) == [0]
    assert list(dataset._neighbors(3)) == [2]
    assert seed_count == 1
    assert features.shape[0] == 3
    assert torch.equal(edges, torch.tensor([[1, 2], [0, 1]]))
    assert dataset.steps[2] < dataset.steps[3]
    assert dataset.steps[0] < dataset.steps[2]


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_precomputed_neighbors_match_legacy_sampling_and_edges():
    import numpy as np

    sample = pd.DataFrame(
        [
            {"step": 1, "type": "PIX_P2P", "nameOrig": "A", "nameDest": "X",
             "amount": 10.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 2, "type": "PIX_P2P", "nameOrig": "A", "nameDest": "Y",
             "amount": 20.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 2, "type": "PIX_P2P", "nameOrig": "A", "nameDest": "X",
             "amount": 30.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 3, "type": "PIX_P2P", "nameOrig": "B", "nameDest": "X",
             "amount": 40.0, "oldbalanceOrg": 100.0, "isFraud": 1},
            {"step": 4, "type": "PIX_P2P", "nameOrig": "A", "nameDest": "Y",
             "amount": 50.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 4, "type": "PIX_P2P", "nameOrig": "B", "nameDest": "Z",
             "amount": 60.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 5, "type": "PIX_P2P", "nameOrig": "X", "nameDest": "A",
             "amount": 70.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"step": 6, "type": "PIX_P2P", "nameOrig": "B", "nameDest": "Y",
             "amount": 80.0, "oldbalanceOrg": 100.0, "isFraud": 1},
        ]
    )
    dataset = gnn.TransactionMiniBatchDataset(sample)

    def legacy_neighbors(node):
        for codes, offsets, members in dataset._account_indices:
            group = codes[node]
            account_nodes = members[offsets[group] : offsets[group + 1]]
            previous_count = int(
                np.searchsorted(
                    dataset.steps[account_nodes], dataset.steps[node], side="left"
                )
            )
            if previous_count:
                yield int(account_nodes[previous_count - 1])

    def legacy_sample_nodes(seeds, fanouts):
        nodes = list(map(int, seeds))
        seen = set(nodes)
        frontier = nodes
        for fanout in fanouts:
            next_frontier = []
            for node in frontier:
                candidates = list(dict.fromkeys(legacy_neighbors(node)))
                for neighbor in candidates[:fanout]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        nodes.append(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return nodes

    def legacy_edge_index(nodes):
        local_index = {node: index for index, node in enumerate(nodes)}
        edges = set()
        for node in nodes:
            for neighbor in legacy_neighbors(node):
                source = local_index.get(neighbor)
                if source is not None:
                    edges.add((source, local_index[node]))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

    fanout_options = ((1,), (2, 1), (3, 2))
    for seed in (0, 7, 42):
        for fanouts in fanout_options:
            seed_order = np.arange(dataset.num_nodes)
            np.random.RandomState(seed).shuffle(seed_order)
            np.random.seed(seed)
            batches = list(
                dataset.iter_batches(
                    batch_size=3, shuffle=True, fanouts=fanouts
                )
            )

            for batch_index, (features, edge_index, _labels, _count) in enumerate(
                batches
            ):
                target_positions = seed_order[
                    batch_index * 3 : (batch_index + 1) * 3
                ]
                seeds = dataset.target_nodes[target_positions]
                expected_nodes = legacy_sample_nodes(seeds, fanouts)
                assert dataset._sample_nodes(seeds, fanouts) == expected_nodes
                assert torch.equal(edge_index, legacy_edge_index(expected_nodes))
                expected_features = gnn.compute_if_features(
                    dataset.context.iloc[expected_nodes]
                )[dataset.feature_names].to_numpy(dtype=np.float32)
                np.testing.assert_array_equal(features.numpy(), expected_features)


@pytest.mark.skipif(not gnn.TORCH_GEO_AVAILABLE, reason="PyTorch Geometric não instalado")
def test_gnn_split_target_uses_context_without_context_labels():
    context = pd.DataFrame(
        [
            {"step": step, "type": "PIX_P2P", "nameOrig": "A",
             "nameDest": f"D{step}", "amount": float(step + 1),
             "oldbalanceOrg": 100.0}
            for step in (1, 2, 3)
        ],
        index=[10, 11, 12],
    )
    target = context.loc[[12]].copy()
    target["isFraud"] = 1
    dataset = gnn.TransactionMiniBatchDataset(
        target, context_df=context
    )

    assert dataset.num_nodes == 1
    assert dataset.target_nodes.tolist() == [2]
    assert list(dataset._neighbors(2)) == [1]
    batch = next(dataset.iter_batches(batch_size=1, fanouts=(1,)))
    assert batch[2].item() == 1
    assert batch[3] == 1
