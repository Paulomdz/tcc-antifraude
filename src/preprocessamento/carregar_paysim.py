"""Carrega e pré-processa o dataset PaySim para o pipeline de detecção de fraudes."""

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
# Nome padrão em português, consistente com a monografia (§3.4): o
# pré-processamento persiste o dataset em data/paysim_dados_processados.parquet.
DEFAULT_PROCESSED_DATA_PATH = Path("data/paysim_dados_processados.parquet")
# Nome legado em inglês, mantido apenas como fallback de leitura para quem já
# tinha gerado o arquivo processado com o nome antigo.
LEGACY_PROCESSED_DATA_PATH = Path("data/paysim_processed.parquet")
PROCESSED_DATA_PATH = DEFAULT_PROCESSED_DATA_PATH

# Duração canônica da simulação PaySim: 31 dias * 24 horas = 744 steps
# (step 1 é a primeira hora simulada). Usado como divisor fixo para
# normalizar "step" em [0, 1], consistente com o restante do projeto
# (dashboard, testes e API já assumem esse mesmo valor de referência).
STEP_MAX = 744

# Tamanho de chunk usado ao processar o CSV bruto do PaySim (~6,3M linhas /
# ~470MB). Processar em chunks evita estourar a memória disponível ao ler o
# arquivo inteiro de uma vez.
DEFAULT_CHUNKSIZE = 300_000

# Tipos de transação canônicos do simulador PaySim. Fixar essa lista (em vez
# de inferir as categorias presentes em cada lote) garante que o one-hot
# encoding de "type" produza sempre as mesmas colunas, mesmo processando o
# CSV em blocos onde um bloco isolado pode não conter todos os tipos.
PAYSIM_TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def resolve_processed_data_path(path: str | Path | None = None) -> Path:
    """Resolve o caminho do parquet processado, priorizando o nome padrão."""
    candidate = Path(path) if path is not None else PROCESSED_DATA_PATH

    if candidate.exists():
        return candidate

    if DEFAULT_PROCESSED_DATA_PATH.exists():
        return DEFAULT_PROCESSED_DATA_PATH

    if LEGACY_PROCESSED_DATA_PATH.exists():
        return LEGACY_PROCESSED_DATA_PATH

    return candidate if path is not None else PROCESSED_DATA_PATH


def load_paysim(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o arquivo CSV do PaySim e retorna um DataFrame bruto.

    Atenção: para o dataset completo (~6,3M linhas), isto carrega tudo em
    memória de uma vez. Para processar o arquivo real sem estourar a
    memória disponível, prefira ``main()``/``process_paysim_in_chunks``,
    que processam o CSV em blocos.
    """
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

    df["type"] = pd.Categorical(df["type"], categories=PAYSIM_TRANSACTION_TYPES)
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

    # Normaliza pelo tamanho fixo da simulação (STEP_MAX = 744), não pelo
    # máximo observado no lote/DataFrame em mãos: isso mantém "step_norm"
    # consistente entre chunks e com o restante do projeto (que já assume
    # 744 como referência), em vez de variar conforme a amostra processada.
    df["step_norm"] = df["step"] / STEP_MAX

    # One-hot encoding para tipos de transação
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    return df


def save_processed(df: pd.DataFrame, output_path: Path = DEFAULT_PROCESSED_DATA_PATH) -> None:
    """Salva o DataFrame processado em formato Parquet para uso posterior."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dados processados salvos em: {output_path}")


def _iter_processed_chunks(
    csv_path: Path, chunksize: int
) -> Iterator[pd.DataFrame]:
    """Lê o CSV em blocos e aplica ``preprocess_paysim`` a cada bloco."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        yield preprocess_paysim(chunk)


def process_paysim_in_chunks(
    csv_path: Path = RAW_DATA_PATH,
    output_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Processa um CSV grande do PaySim em blocos, sem carregar tudo em memória.

    Cada bloco é lido, pré-processado (``preprocess_paysim``) e anexado
    incrementalmente ao arquivo Parquet de saída via ``pyarrow.ParquetWriter``.
    Isto permite processar o dataset real (~6,3M linhas / ~470MB) mesmo em
    ambientes com pouca memória disponível.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo PaySim não encontrado em: {csv_path}. "
            "Baixe o dataset e coloque-o na pasta data/."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
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
            print(f"  Processadas {total_rows:,} linhas...".replace(",", "."))
    finally:
        if writer is not None:
            writer.close()

    print(f"Dados processados salvos em: {output_path}")
    print(f"Total de linhas: {total_rows:,}".replace(",", "."))
    print(f"Total de fraudes: {total_fraud:,}".replace(",", "."))


def main(raw_path: str | None = None, output_path: str | None = None) -> None:
    """Ponto de entrada para carregar, processar e salvar o dataset PaySim.

    Processa o CSV em blocos (``process_paysim_in_chunks``) para lidar com
    o dataset completo do PaySim sem exigir que ele caiba inteiro em
    memória.
    """
    csv_path = Path(raw_path) if raw_path else RAW_DATA_PATH
    processed_path = Path(output_path) if output_path else DEFAULT_PROCESSED_DATA_PATH

    process_paysim_in_chunks(csv_path, processed_path)


if __name__ == "__main__":
    main()
