import pandas as pd
import pytest

from src.preprocessamento.carregar_dataset import (
    RAW_DATA_PATH,
    TRANSACTION_TYPES,
    load_dataset,
    preprocess_dataset,
    process_dataset_in_chunks,
)


def test_loads_final_dataset_schema_and_categories():
    dataset = load_dataset(RAW_DATA_PATH)

    assert len(dataset) == 454_117
    assert set(dataset["type"].unique()) == set(TRANSACTION_TYPES)
    assert "isFlaggedFraud" not in dataset.columns


def test_preprocesses_all_categories_and_derives_features():
    raw = pd.read_csv(RAW_DATA_PATH, nrows=12)

    processed = preprocess_dataset(raw)

    assert "isFlaggedFraud" not in processed
    assert all(f"type_{tx_type}" in processed for tx_type in TRANSACTION_TYPES)
    assert processed["hora_norm"].between(0, 1).all()
    assert processed["amount_log"].notna().all()


def test_preprocess_rejects_unknown_transaction_types():
    raw = pd.read_csv(RAW_DATA_PATH, nrows=1)
    raw.loc[0, "type"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Tipos de transação"):
        preprocess_dataset(raw)


def test_preprocess_requires_dataset_columns():
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        preprocess_dataset(pd.DataFrame({"step": [0]}))


def test_chunk_processor_writes_all_rows_with_stable_categories(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "processed.parquet"
    raw = pd.read_csv(RAW_DATA_PATH, nrows=7)
    raw.to_csv(source, index=False)

    process_dataset_in_chunks(source, output, chunksize=2)
    processed = pd.read_parquet(output)

    assert len(processed) == 7
    assert all(f"type_{tx_type}" in processed for tx_type in TRANSACTION_TYPES)


def test_chunk_processor_reports_invalid_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset não encontrado"):
        process_dataset_in_chunks(tmp_path / "missing.csv")

    source = tmp_path / "input.csv"
    pd.read_csv(RAW_DATA_PATH, nrows=1).to_csv(source, index=False)
    with pytest.raises(ValueError, match="chunksize"):
        process_dataset_in_chunks(source, chunksize=0)


def test_temporal_split_no_intersection():
    from src.preprocessamento.carregar_dataset import temporal_split
    sample = pd.DataFrame({
        "step": [100, 549, 550, 599, 600, 743],
        "isFraud": [0, 1, 0, 1, 0, 1]
    })
    splits = temporal_split(sample)
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["train_completo"]) == 4
    assert len(splits["test"]) == 2
    assert set(splits["train"].index).isdisjoint(set(splits["val"].index))
    assert set(splits["train"].index).isdisjoint(set(splits["test"].index))
    assert set(splits["val"].index).isdisjoint(set(splits["test"].index))


def test_anti_leakage_ids_not_in_if_features():
    from src.preprocessamento.carregar_dataset import IF_FEATURE_NAMES
    for feat in IF_FEATURE_NAMES:
        assert "nameOrig" not in feat
        assert "nameDest" not in feat
    assert len(IF_FEATURE_NAMES) == 21


def test_if_features_equivalence_df_and_dict():
    import numpy as np
    from src.preprocessamento.carregar_dataset import (
        extract_if_features_df,
        extract_if_features_row,
    )
    raw = pd.read_csv(RAW_DATA_PATH, nrows=3)
    df_feats = extract_if_features_df(raw)
    for idx in range(len(raw)):
        row_dict = raw.iloc[idx].to_dict()
        row_feats = extract_if_features_row(row_dict)
        assert np.allclose(df_feats.iloc[idx].values, row_feats, atol=1e-9)

