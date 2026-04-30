"""Carrega e pré-processa o dataset PaySim para o pipeline de detecção de fraudes."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
PROCESSED_DATA_PATH = Path("data/paysim_dados_processados.parquet")


def load_paysim(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o arquivo CSV do PaySim e retorna um DataFrame bruto."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo PaySim não encontrado em: {csv_path}. "
            "Baixe o dataset e coloque-o na pasta data/."
        )

    df = pd.read_csv(csv_path)
    return df


def preprocess_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza básica e extração de features para detecção de fraude."""
    df = df.copy()

    df["type"] = df["type"].astype("category")
    df["isFraud"] = df["isFraud"].astype(int)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int)

    df["step"] = df["step"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["oldbalanceOrg"] = df["oldbalanceOrg"].astype(float)
    df["newbalanceOrig"] = df["newbalanceOrig"].astype(float)
    df["oldbalanceDest"] = df["oldbalanceDest"].astype(float)
    df["newbalanceDest"] = df["newbalanceDest"].astype(float)

    df["amount_log"] = np.log1p(df["amount"])
    df["balance_org_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["same_account"] = (df["nameOrig"] == df["nameDest"]).astype(int)
    df["is_incoming"] = (df["type"] == "PAYMENT") | (df["type"] == "TRANSFER")

    df["step_norm"] = df["step"] / df["step"].max()

    # One-hot encoding para tipos de transação
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    return df


def save_processed(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> None:
    """Salva o DataFrame processado em formato Parquet para uso posterior."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dados processados salvos em: {output_path}")


def main(raw_path: str | None = None, output_path: str | None = None) -> None:
    """Ponto de entrada para carregar, processar e salvar o dataset PaySim."""
    csv_path = Path(raw_path) if raw_path else RAW_DATA_PATH
    processed_path = Path(output_path) if output_path else PROCESSED_DATA_PATH

    raw_df = load_paysim(csv_path)
    processed_df = preprocess_paysim(raw_df)
    save_processed(processed_df, processed_path)


if __name__ == "__main__":
    main()
