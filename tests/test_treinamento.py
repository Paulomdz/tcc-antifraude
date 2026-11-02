import csv
import json

from src.modelos.treinamento import treinar_modelos as training


def test_primeiro_passe_if_lstm_grids_are_reduced_and_constrained():
    if_configs = training._candidate_configs("isolation_forest", primeiro_passe=True)
    lstm_configs = training._candidate_configs("lstm", primeiro_passe=True)

    assert len(if_configs) == 9
    assert {
        config["n_estimators"] for config in if_configs
    } == {100, 200, 300}
    assert {config["contamination"] for config in if_configs} == {
        0.03,
        0.056,
        0.10,
    }
    assert {config["max_samples"] for config in if_configs} == {"auto"}

    assert len(lstm_configs) == 8
    assert {config["hidden_size"] for config in lstm_configs} == {32, 64}
    assert {config["dropout"] for config in lstm_configs} == {0.1, 0.3}
    assert {config["lr"] for config in lstm_configs} == {1e-3, 5e-4}
    assert {config["num_layers"] for config in lstm_configs} == {2}
    assert all(config.get("num_layers") != 3 for config in if_configs + lstm_configs)
    assert all(
        config.get("max_samples", "auto") == "auto"
        for config in if_configs + lstm_configs
    )

    assert len(training._candidate_configs("isolation_forest")) == 27
    assert len(training._candidate_configs("lstm")) == 16


def test_grid_log_records_config_metrics_time_and_memory(tmp_path, monkeypatch):
    grid_path = tmp_path / "etapa2_grid.csv"
    monkeypatch.setattr(training, "GRID_CSV_PATH", grid_path)

    training._append_grid_row(
        {
            "model": "lstm",
            "variant": "main",
            "config": training._config_key(
                {"hidden_size": 32, "num_layers": 2}
            ),
            "validation_pr_auc": 0.9,
            "validation_f1": 0.75,
            "training_seconds": 12.5,
            "peak_ram_gib": 2.4,
            "ram_free_before_gib": 4.0,
            "ram_available_before_gib": 8.0,
            "feature_policy": training.FEATURE_POLICY_VERSION,
            "dataset_signature": training.DATASET_SIGNATURE,
            "vram_free_before_gib": 6.0,
            "peak_vram_gib": 1.2,
            "status": "completed",
            "error": "",
        }
    )

    with grid_path.open(newline="", encoding="utf-8") as grid_file:
        rows = list(csv.DictReader(grid_file))

    assert len(rows) == 1
    assert rows[0]["model"] == "lstm"
    assert rows[0]["config"] == json.dumps(
        {"hidden_size": 32, "num_layers": 2},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert rows[0]["validation_pr_auc"] == "0.9"
    assert rows[0]["validation_f1"] == "0.75"
    assert rows[0]["training_seconds"] == "12.5"
    assert rows[0]["peak_ram_gib"] == "2.4"
    assert rows[0]["ram_free_before_gib"] == "4.0"
    assert rows[0]["feature_policy"] == training.FEATURE_POLICY_VERSION
    assert rows[0]["dataset_signature"] == training.DATASET_SIGNATURE
    assert rows[0]["peak_vram_gib"] == "1.2"


def test_best_f1_threshold_uses_validation_labels():
    import numpy as np

    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.4, 0.8])

    threshold = training._best_f1_threshold(labels, scores)
    metrics = training._metrics(labels, scores, threshold)

    assert threshold == 0.4
    assert metrics["f1"] == 1.0
    assert metrics["threshold"] == threshold


def test_grid_refuses_rows_from_another_feature_policy(tmp_path, monkeypatch):
    grid_path = tmp_path / "legacy.csv"
    grid_path.write_text(
        "model,variant,config,validation_pr_auc\n"
        'lstm,main,"{}",0.95\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(training, "GRID_CSV_PATH", grid_path)

    import pytest

    with pytest.raises(RuntimeError, match="outra política ou dataset"):
        training._grid_rows()


def test_isolation_forest_smoke_checkpoint_resumes_without_refitting(
    tmp_path, monkeypatch
):
    import pandas as pd

    from src.preprocessamento.carregar_dataset import IF_FEATURE_NAMES

    rows = [
        {
            "step": step,
            "type": "PIX_P2P",
            "amount": float((step + 1) * 10),
            "nameOrig": f"C{step}",
            "nameDest": f"D{step}",
            "oldbalanceOrg": 100.0,
            "isFraud": step % 2,
        }
        for step in range(12)
    ]
    frame = pd.DataFrame(rows)
    splits = {
        "train": frame.iloc[:8],
        "val": frame.iloc[8:],
    }
    monkeypatch.setattr(training, "GRID_CSV_PATH", tmp_path / "grid.csv")
    monkeypatch.setattr(training, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(training, "DATASET_SIGNATURE", "test-dataset")

    first = training._run_smoke_test(
        "isolation_forest", splits, IF_FEATURE_NAMES
    )
    second = training._run_smoke_test(
        "isolation_forest", splits, IF_FEATURE_NAMES
    )

    assert first["metrics"]["pr_auc"] >= 0.0
    assert first["training_seconds"] >= 0.0
    assert second["resumed"] is True
    assert len(training._grid_rows()) == 1


def test_isolated_smoke_does_not_read_or_write_grid_or_checkpoint(
    tmp_path, monkeypatch
):
    import pandas as pd

    from src.preprocessamento.carregar_dataset import IF_FEATURE_NAMES

    rows = [
        {
            "step": step,
            "type": "PIX_P2P",
            "amount": float((step + 1) * 10),
            "nameOrig": f"C{step}",
            "nameDest": f"D{step}",
            "oldbalanceOrg": 100.0,
            "isFraud": step % 2,
        }
        for step in range(12)
    ]
    frame = pd.DataFrame(rows)
    splits = {"train": frame.iloc[:8], "val": frame.iloc[8:]}
    grid_path = tmp_path / "etapa2_grid.csv"
    grid_path.write_text("diagnostic-grid-must-remain-untouched\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(training, "GRID_CSV_PATH", grid_path)
    monkeypatch.setattr(training, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(training, "DATASET_SIGNATURE", "test-dataset")
    monkeypatch.setattr(
        training,
        "_ram_snapshot",
        lambda: {"free_gib": 3.0, "available_gib": 4.0},
    )
    monkeypatch.setattr(training, "_vram_snapshot", lambda: {"free_gib": 0.0})

    def fail_if_grid_accessed(*_args, **_kwargs):
        raise AssertionError("isolated smoke must not access the diagnostic grid")

    monkeypatch.setattr(training, "_grid_rows", fail_if_grid_accessed)
    monkeypatch.setattr(training, "_append_grid_row", fail_if_grid_accessed)

    result = training._run_smoke_test(
        "isolation_forest",
        splits,
        IF_FEATURE_NAMES,
        persist_result=False,
    )

    assert result["metrics"]["pr_auc"] >= 0.0
    assert result["ram_before"]["available_gib"] == 4.0
    assert grid_path.read_text(encoding="utf-8") == (
        "diagnostic-grid-must-remain-untouched\n"
    )
    assert not checkpoint_dir.exists()


def test_if_lstm_grade_mode_uses_v2_grid_and_never_uses_gnn_or_test(
    tmp_path, monkeypatch
):
    import pandas as pd

    processed_path = tmp_path / "processed.parquet"
    processed_path.touch()
    grid_path = tmp_path / "reports" / "etapa2_grid_v2.csv"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(training, "GRID_CSV_PATH", tmp_path / "old_grid.csv")
    monkeypatch.setattr(
        training,
        "CHECKPOINT_DIR",
        tmp_path / "reports" / "etapa2_checkpoints",
    )
    monkeypatch.setattr(
        training,
        "TORCH_AVAILABLE",
        True,
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "_set_seed", lambda: None)
    monkeypatch.setattr(
        training,
        "_ram_snapshot",
        lambda: {"free_gib": 4.0, "available_gib": 5.0},
    )
    monkeypatch.setattr(training, "generate_feature_stats", lambda *_args: None)

    source_frame = pd.DataFrame(
        {
            "step": [1, 2, 551],
            "type": ["PIX_P2P", "PIX_P2P", "PIX_P2P"],
            "amount": [10.0, 20.0, 30.0],
            "nameOrig": ["C0000001", "C0000002", "C0000003"],
            "nameDest": ["M0000001", "M0000002", "M0000003"],
            "oldbalanceOrg": [100.0, 200.0, 300.0],
            "isFraud": [0, 1, 1],
        }
    )
    train_frame = source_frame.iloc[:2]
    val_frame = source_frame.iloc[2:]
    split_frames = {
        "train": train_frame,
        "val": val_frame,
        "train_completo": source_frame,
        "test": source_frame.iloc[:0],
    }

    def fake_read_parquet(path, columns, filters):
        assert path == processed_path
        assert "isFraud" in columns
        assert filters == [("step", "<", 600)]
        return source_frame.copy()

    monkeypatch.setattr(training.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(training, "temporal_split", lambda _frame: split_frames)
    monkeypatch.setattr(
        training,
        "_feature_matrix",
        lambda frame, _features: frame[["amount"]].to_numpy(),
    )
    calls = []

    def fake_if_search(*args, **kwargs):
        calls.append(("isolation_forest", kwargs.get("primeiro_passe")))
        return object(), {
            "config": {"n_estimators": 100},
            "metrics": {"pr_auc": 0.6, "f1": 0.5},
        }, 1, False

    def fake_lstm_search(model_name, *_args, **kwargs):
        calls.append((model_name, kwargs.get("primeiro_passe")))
        return object(), {
            "config": {"hidden_size": 32, "num_layers": 2},
            "metrics": {"pr_auc": 0.4, "f1": 0.3},
            "best_epoch": 1,
        }, 1, False, object()

    monkeypatch.setattr(training, "_if_candidate_search", fake_if_search)
    monkeypatch.setattr(training, "_nn_candidate_search", fake_lstm_search)

    results = training.grade_if_lstm_only(grid_path, primeiro_passe=True)

    assert calls == [("isolation_forest", True), ("lstm", True)]
    assert set(results) == {"isolation_forest", "lstm"}
    assert training.GRID_CSV_PATH == grid_path.resolve()


def test_if_lstm_grade_rejects_non_v2_grid_path(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="etapa2_grid_v2.csv"):
        training.grade_if_lstm_only(tmp_path / "etapa2_grid.csv")


def test_archives_incompatible_candidate_checkpoints_and_preserves_current_ones(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    model_dir = tmp_path / "models"
    checkpoint_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setattr(training, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(training, "MODEL_DIR", model_dir)
    monkeypatch.setattr(training, "REPORT_PATH", tmp_path / "reports" / "report.json")
    monkeypatch.setattr(
        training, "GRID_CSV_PATH", tmp_path / "reports" / "etapa2_grid.csv"
    )
    monkeypatch.setattr(training, "DATASET_SIGNATURE", "current-dataset")

    stale_json = checkpoint_dir / "isolation_forest_main_old.json"
    stale_json.write_text(
        json.dumps(
            {
                "dataset_signature": "old-dataset",
                "feature_policy": "old-policy",
            }
        ),
        encoding="utf-8",
    )
    stale_torch = checkpoint_dir / "lstm_main_old.pt"
    training.torch.save(
        {
            "dataset_signature": "old-dataset",
            "feature_policy": "old-policy",
        },
        stale_torch,
    )
    current = checkpoint_dir / "lstm_main_current.pt"
    training.torch.save(
        {
            "dataset_signature": "current-dataset",
            "feature_policy": training.FEATURE_POLICY_VERSION,
        },
        current,
    )
    old_model = model_dir / "lstm_model.pt"
    old_model.write_bytes(b"diagnostic model")

    diagnostic_report = training._archive_pre_policy_outputs()

    assert not stale_json.exists()
    assert (checkpoint_dir / "isolation_forest_main_old_diagnostico_pre_politica.json").exists()
    assert not stale_torch.exists()
    assert (checkpoint_dir / "lstm_main_old_diagnostico_pre_politica.pt").exists()
    assert current.exists()
    assert (
        checkpoint_dir
        / "modelos_diagnostico_pre_politica"
        / old_model.name
    ).read_bytes() == b"diagnostic model"
    assert len(diagnostic_report["archived_artifacts"]) == 2
