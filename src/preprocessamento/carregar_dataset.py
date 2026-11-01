"""Carregamento e pré-processamento do dataset Pix/Banco final."""

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "dataset_pix_banco.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "dataset" / "dataset_pix_banco_final_processado.parquet"
)
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_PATH
STEP_MAX = 743
DEFAULT_CHUNKSIZE = 50_000

TRANSACTION_TYPES = [
    "PIX_P2P",
    "PIX_QRCODE",
    "PIX_SAQUE",
    "PIX_TROCO",
    "PIX_AUTOMATICO",
    "CARTAO_ECOMMERCE",
    "CARTAO_PRESENCIAL",
    "TEF",
    "BOLETO",
    "SAQUE",
    "TED",
    "EMPRESTIMO",
]

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
}


def load_dataset(csv_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV do dataset final e valida seu esquema básico."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa os dados e cria atributos usados pelo pipeline antifraude."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing}")

    df = df.copy()
    unknown_types = set(df["type"].dropna().astype(str)) - set(TRANSACTION_TYPES)
    if unknown_types:
        unknown = ", ".join(sorted(unknown_types))
        raise ValueError(f"Tipos de transação não reconhecidos: {unknown}")

    df["type"] = pd.Categorical(df["type"], categories=TRANSACTION_TYPES)
    df["isFraud"] = df["isFraud"].astype(int)

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["step"] = df["step"].astype(int)
    for column in numeric_columns[1:]:
        df[column] = df[column].astype(float)

    df["amount_log"] = np.log1p(df["amount"])
    df["balance_org_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    # same_account removed – anti‑vazamento (D2)

    # is_incoming based on origin account type B – regra D5
    df["is_incoming"] = df["nameOrig"].str.startswith("B").astype(int)
    # hora (0‑23) derivada de step, hora_norm e flag madrugada – regra D2
    df["hora"] = df["step"] % 24
    df["hora_norm"] = df["hora"] / 23.0
    df["is_madrugada"] = (df["hora"] <= 5).astype(int)



    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    return pd.concat([df, type_dummies], axis=1)


def save_processed(
    df: pd.DataFrame, output_path: str | Path = DEFAULT_PROCESSED_DATA_PATH
) -> None:
    """Salva os dados processados em formato Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dados processados salvos em: {output_path}")


def _iter_processed_chunks(
    csv_path: Path, chunksize: int
) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        yield preprocess_dataset(chunk)


def process_dataset_in_chunks(
    csv_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Pré-processa o CSV em blocos e grava Parquet sem carregar tudo em RAM."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {csv_path}")
    if chunksize < 1:
        raise ValueError("chunksize deve ser maior que zero.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    total_fraud = 0
    try:
        for processed_chunk in _iter_processed_chunks(csv_path, chunksize):
            table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += len(processed_chunk)
            total_fraud += int(processed_chunk["isFraud"].sum())
            print(f"Processadas {total_rows:,} linhas...".replace(",", "."))
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError(f"Dataset vazio: {csv_path}")

    print(f"Dados processados salvos em: {output_path}")
    print(f"Total de linhas: {total_rows:,}".replace(",", "."))
    print(f"Total de fraudes: {total_fraud:,}".replace(",", "."))


def main(raw_path: str | None = None, output_path: str | None = None) -> None:
    """Gera o Parquet usado pelo treinamento e pela simulação."""
    process_dataset_in_chunks(
        raw_path or RAW_DATA_PATH,
        output_path or DEFAULT_PROCESSED_DATA_PATH,
    )


if __name__ == "__main__":
    main()
