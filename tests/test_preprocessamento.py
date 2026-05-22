"""Testes unitários para o módulo de preprocessamento PaySim."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessamento.carregar_paysim import preprocess_paysim, save_processed


class TestPreprocessPaysim:
    """Testa a função preprocess_paysim com dados sintéticos."""

    def test_retorna_dataframe(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_nao_modifica_original(self, sample_df):
        original_cols = list(sample_df.columns)
        preprocess_paysim(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_cria_coluna_amount_log(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "amount_log" in result.columns
        assert (result["amount_log"] >= 0).all()

    def test_amount_log_usa_log1p(self, sample_df):
        result = preprocess_paysim(sample_df)
        expected = np.log1p(sample_df["amount"])
        pd.testing.assert_series_equal(result["amount_log"], expected, check_names=False)

    def test_cria_balance_org_delta(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "balance_org_delta" in result.columns
        expected = sample_df["oldbalanceOrg"] - sample_df["newbalanceOrig"]
        pd.testing.assert_series_equal(
            result["balance_org_delta"], expected, check_names=False
        )

    def test_cria_balance_dest_delta(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "balance_dest_delta" in result.columns

    def test_cria_same_account(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "same_account" in result.columns
        assert result["same_account"].dtype in (int, "int32", "int64")

    def test_same_account_detecta_igual(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "type": ["TRANSFER"],
                "amount": [100.0],
                "nameOrig": ["C1"],
                "oldbalanceOrg": [1000.0],
                "newbalanceOrig": [900.0],
                "nameDest": ["C1"],
                "oldbalanceDest": [0.0],
                "newbalanceDest": [100.0],
                "isFraud": [0],
                "isFlaggedFraud": [0],
            }
        )
        result = preprocess_paysim(df)
        assert result["same_account"].iloc[0] == 1

    def test_cria_step_norm_entre_zero_e_um(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "step_norm" in result.columns
        assert (result["step_norm"] >= 0).all()
        assert (result["step_norm"] <= 1).all()

    def test_cria_is_incoming(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert "is_incoming" in result.columns

    def test_cria_colunas_one_hot_type(self, sample_df):
        result = preprocess_paysim(sample_df)
        type_cols = [c for c in result.columns if c.startswith("type_")]
        assert len(type_cols) > 0

    def test_is_fraud_convertido_para_int(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert result["isFraud"].dtype in (int, "int32", "int64")

    def test_amount_convertido_para_float(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert result["amount"].dtype == float

    def test_preserva_numero_de_linhas(self, sample_df):
        result = preprocess_paysim(sample_df)
        assert len(result) == len(sample_df)


class TestSaveProcessed:
    """Testa a função save_processed."""

    def test_salva_parquet(self, sample_df, tmp_path):
        output = tmp_path / "test_output.parquet"
        processed = preprocess_paysim(sample_df)
        save_processed(processed, output)
        assert output.exists()

    def test_parquet_legivel(self, sample_df, tmp_path):
        output = tmp_path / "test_output.parquet"
        processed = preprocess_paysim(sample_df)
        save_processed(processed, output)
        loaded = pd.read_parquet(output)
        assert len(loaded) == len(processed)
        assert list(loaded.columns) == list(processed.columns)

    def test_cria_diretorio_pai(self, sample_df, tmp_path):
        output = tmp_path / "subdir" / "output.parquet"
        processed = preprocess_paysim(sample_df)
        save_processed(processed, output)
        assert output.exists()


class TestLoadPaysim:
    """Testa o comportamento de load_paysim para arquivos ausentes."""

    def test_levanta_file_not_found_para_arquivo_inexistente(self):
        from src.preprocessamento.carregar_paysim import load_paysim

        with pytest.raises(FileNotFoundError):
            load_paysim(Path("nao_existe.csv"))
