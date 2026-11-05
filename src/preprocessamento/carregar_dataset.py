"""Carregamento e pré-processamento do dataset Pix/Banco final para o TCC Antifraude.

Decisões de Engenharia / TCC:
- D2: hora = step % 24 e hora_norm = hora / 23.0 tanto no treino quanto na inferência.
- D3: Split temporal estrito: treino core (<550), validação (550..599), treino completo (<600), teste (>=600).
- D4: PIX_TROCO tratado como outgoing.
- D5: is_incoming = 1 se nameOrig começa com 'B' (banco/TED salário/empréstimo), senão 0.
- Regra Anti-Vazamento: IDs nameOrig/nameDest nunca usados como features de entrada.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "dataset_pix_banco.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "dataset_pix_banco_processado.parquet"
FEATURE_STATS_PATH = Path(__file__).resolve().parent / "feature_stats.json"

DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_PATH
DEFAULT_CHUNKSIZE = 50_000

# Lista oficial dos 12 tipos de transação do dataset
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

# TCC / anti-vazamento: a lista final tem 18 features; as flags de saldo zerado
# e divergência foram removidas porque usam saldos pós-transação. A razão
# amount_to_balance_ratio permanece porque usa somente oldbalanceOrg.
IF_FEATURE_NAMES = [
    "amount",
    "is_incoming",
    "type_PIX_P2P",
    "type_PIX_QRCODE",
    "type_PIX_SAQUE",
    "type_PIX_TROCO",
    "type_PIX_AUTOMATICO",
    "type_CARTAO_ECOMMERCE",
    "type_CARTAO_PRESENCIAL",
    "type_TEF",
    "type_BOLETO",
    "type_SAQUE",
    "type_TED",
    "type_EMPRESTIMO",
    "hora_norm",
    "amount_to_balance_ratio",
    "log_amount",
    "is_madrugada",
]

# Features sequenciais e de nó também são derivadas somente das transações.
# IDs servem exclusivamente para agrupar o histórico (LSTM) e formar arestas (GNN).
LSTM_FEATURE_NAMES = [
    "log_amount",
    "is_incoming",
    *[f"type_{tx_type}" for tx_type in TRANSACTION_TYPES],
    "hora_norm",
    "amount_to_balance_ratio",
    "is_madrugada",
]
GNN_FEATURE_NAMES = [
    "log_amount",
    "is_incoming",
    "hora_norm",
    "amount_to_balance_ratio",
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
    """Lê o CSV do dataset canônico e valida seu esquema básico."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing}")
    return df


def hora_norm_from_step(step: Any) -> float:
    """Aplica a mesma conversão horária do dataset à inferência interativa."""
    return (int(step) % 24) / 23.0


def extract_if_features_row(tx: Dict[str, Any]) -> np.ndarray:
    """Extrai as features pré-transação para uma transação individual.

    Garante idêntica ordenação e fórmulas às usadas no pré-processamento tabular do treino.
    """
    amount = float(tx.get("amount", 0.0))
    old_orig = float(tx.get("oldbalanceOrg", 0.0))

    # Regra D5: is_incoming baseado na origem B
    name_orig = str(tx.get("nameOrig", ""))
    is_incoming = 1.0 if name_orig.startswith("B") else 0.0

    # One-hot dos 12 tipos
    tx_type = str(tx.get("type", ""))
    one_hot = [1.0 if tx_type == t else 0.0 for t in TRANSACTION_TYPES]

    # Regra D2: hora = step % 24 e hora_norm = hora / 23.0
    step_val = tx.get("step")
    if step_val is None:
        hora_val = int(tx.get("hora", 12))
    else:
        hora_val = int(step_val) % 24
    hora_norm = hora_norm_from_step(hora_val)

    # TCC / anti-vazamento: não derivar emptied_origin ou balance_mismatch_*,
    # pois dependem de saldos pós-transação; a razão usa apenas saldo anterior.
    ratio = amount / (abs(old_orig) + 1.0)
    log_amt = math.log1p(max(0.0, amount))
    is_madrug = 1.0 if hora_val in {0, 1, 2, 3, 4, 5} else 0.0

    vec = [
        amount,
        is_incoming,
        *one_hot,
        hora_norm,
        ratio,
        log_amt,
        is_madrug,
    ]
    return np.array(vec, dtype=np.float32)


def extract_if_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai as colunas pré-transação na ordem fixa do Isolation Forest."""
    res = pd.DataFrame(index=df.index)

    amount = df["amount"].astype(float)
    old_orig = df["oldbalanceOrg"].astype(float)

    # Base
    res["amount"] = amount
    res["is_incoming"] = df["nameOrig"].astype(str).str.startswith("B").astype(float)

    for t in TRANSACTION_TYPES:
        col_name = f"type_{t}"
        if col_name in df.columns:
            res[col_name] = df[col_name].astype(float)
        else:
            res[col_name] = (df["type"] == t).astype(float)

    hora = (df["step"].astype(int) % 24)
    res["hora_norm"] = (hora / 23.0).astype(float)

    # TCC / anti-vazamento: não derivar emptied_origin ou balance_mismatch_*,
    # pois dependem de saldos pós-transação; a razão usa apenas saldo anterior.
    res["amount_to_balance_ratio"] = (amount / (old_orig.abs() + 1.0)).astype(float)
    res["log_amount"] = np.log1p(np.maximum(0.0, amount)).astype(float)
    res["is_madrugada"] = hora.isin([0, 1, 2, 3, 4, 5]).astype(float)

    return res[IF_FEATURE_NAMES].astype(np.float32)


def compute_if_features(data: Any) -> np.ndarray | pd.DataFrame:
    """Gera as features oficiais para treino (DataFrame) ou inferência (transação)."""
    if isinstance(data, pd.DataFrame):
        return extract_if_features_df(data)
    elif isinstance(data, dict):
        return extract_if_features_row(data)
    else:
        raise TypeError(f"Tipo não suportado para compute_if_features: {type(data)}")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa os dados brutos e anexa as features derivadas do pipeline antifraude."""
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

    # Regra D5: is_incoming
    df["is_incoming"] = df["nameOrig"].astype(str).str.startswith("B").astype(int)

    # Regra D2: hora e hora_norm
    hora = df["step"] % 24
    df["hora_norm"] = hora / 23.0
    df["is_madrugada"] = hora.isin([0, 1, 2, 3, 4, 5]).astype(int)

    # TCC / anti-vazamento: as flags emptied_origin e balance_mismatch_* usam
    # saldos pós-transação e são excluídas; esta razão usa saldo anterior.
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"].abs() + 1.0)
    df["log_amount"] = np.log1p(np.maximum(0.0, df["amount"]))

    # One-hot encoding fixo para os 12 tipos
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    # Garante que todas as 12 colunas type_ estejam presentes
    for t in TRANSACTION_TYPES:
        col = f"type_{t}"
        if col not in type_dummies.columns:
            type_dummies[col] = 0

    return pd.concat([df, type_dummies], axis=1)


def temporal_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Divide o dataset conforme a regra D3 de split temporal estrito.

    Retorna:
      - train: step < 550 (335.973 linhas)
      - val: step 550..599 (30.677 linhas)
      - train_completo: step < 600 (366.650 linhas)
      - test: step >= 600 (87.467 linhas)
    """
    train_completo = df[df["step"] < 600].copy()
    train = df[df["step"] < 550].copy()
    val = df[(df["step"] >= 550) & (df["step"] < 600)].copy()
    test = df[df["step"] >= 600].copy()

    # Verificação de não-interseção
    assert len(set(train.index).intersection(set(val.index))) == 0, "Interseção entre train e val!"
    assert len(set(train.index).intersection(set(test.index))) == 0, "Interseção entre train e test!"
    assert len(set(val.index).intersection(set(test.index))) == 0, "Interseção entre val e test!"
    assert len(train_completo) == len(train) + len(val), "train_completo deve ser união exata de train e val!"

    return {
        "train": train,
        "val": val,
        "train_completo": train_completo,
        "test": test,
    }


def generate_feature_stats(
    train_completo_df: pd.DataFrame, output_path: Path = FEATURE_STATS_PATH
) -> Dict[str, Any]:
    """Calcula estatísticas (mean, std, min, max, p05, p95) SOMENTE sobre o treino (step < 600).

    Gera o artefato feature_stats.json na ordem fixa das features permitidas.
    """
    if "step" not in train_completo_df.columns:
        raise ValueError("Não é possível gerar feature_stats sem a coluna step.")
    if train_completo_df.empty:
        raise ValueError("Não é possível gerar feature_stats com treino vazio.")
    steps = pd.to_numeric(train_completo_df["step"], errors="raise")
    if not steps.lt(600).all():
        raise ValueError("feature_stats só pode usar transações com step < 600.")

    if_df = compute_if_features(train_completo_df)
    stats: Dict[str, Any] = {}

    for col in IF_FEATURE_NAMES:
        series = if_df[col].astype(float)
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=0))
        min_val = float(series.min())
        max_val = float(series.max())
        p05_val = float(series.quantile(0.05))
        p95_val = float(series.quantile(0.95))

        stats[col] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "p05": p05_val,
            "p95": p95_val,
            "low_variance": bool(std_val < 1e-6),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"feature_stats.json salvo em: {output_path} ({len(stats)} features)")
    return stats


def process_dataset_in_chunks(
    csv_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Pré-processa o CSV canônico e grava o Parquet processado."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {csv_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    total_fraud = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        processed_chunk = preprocess_dataset(chunk)
        table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema)
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        total_rows += len(processed_chunk)
        total_fraud += int(processed_chunk["isFraud"].sum())
        print(f"Processadas {total_rows:,} linhas...".replace(",", "."))

    if writer is not None:
        writer.close()

    print(f"Parquet processado salvo em: {output_path}")
    print(f"Total de linhas: {total_rows:,} | Fraudes: {total_fraud:,}".replace(",", "."))


def main(raw_path: str | None = None, output_path: str | None = None) -> None:
    """Executa a rotina de pré-processamento e geração de feature_stats."""
    raw = raw_path or RAW_DATA_PATH
    out = output_path or DEFAULT_PROCESSED_DATA_PATH
    process_dataset_in_chunks(raw, out)

    print("Calculando feature_stats.json a partir do treino (step < 600)...")
    df = pd.read_parquet(out)
    splits = temporal_split(df)
    generate_feature_stats(splits["train_completo"])


if __name__ == "__main__":
    main()
