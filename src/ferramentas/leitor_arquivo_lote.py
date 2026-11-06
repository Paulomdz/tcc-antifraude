"""Leitura compartilhada de arquivos de transações para API e dashboard."""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd

COLUNAS_OBRIGATORIAS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
}


SEED_AMOSTRAGEM = 42


def amostrar_lote(
    dataframe: pd.DataFrame, quantidade: int, seed: int = SEED_AMOSTRAGEM
) -> pd.DataFrame:
    """Amostra usada pela API; o dashboard a repete para casar índice e linha."""
    if len(dataframe) > quantidade:
        return dataframe.sample(n=quantidade, random_state=seed)
    return dataframe


def ler_arquivo_lote(conteudo: bytes, nome_arquivo: str) -> pd.DataFrame:
    """Lê CSV/TXT, detectando vírgula ou ponto e vírgula pelo cabeçalho."""
    extensao = Path(nome_arquivo).suffix.lower()
    if extensao not in {".csv", ".txt"}:
        raise ValueError("Formato inválido: envie um arquivo CSV ou TXT.")
    if not conteudo:
        raise ValueError("O arquivo está vazio.")

    try:
        texto = conteudo.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("O arquivo deve estar codificado em UTF-8.") from exc

    cabecalho = next((linha for linha in texto.splitlines() if linha.strip()), "")
    if not cabecalho:
        raise ValueError("O arquivo não contém cabeçalho.")

    separadores = {",": cabecalho.count(","), ";": cabecalho.count(";")}
    if separadores[","] == separadores[";"]:
        raise ValueError("Não foi possível detectar o separador do cabeçalho.")
    separador = max(separadores, key=separadores.get)
    separador_esperado = "," if extensao == ".csv" else ";"
    if separador != separador_esperado:
        raise ValueError(
            f"Arquivo {extensao.upper()} deve usar o separador "
            f"{separador_esperado!r}."
        )

    colunas = [nome.strip() for nome in next(csv.reader([cabecalho], delimiter=separador))]
    ausentes = COLUNAS_OBRIGATORIAS - set(colunas)
    if ausentes:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(ausentes))
        )
    if len(colunas) != len(set(colunas)):
        raise ValueError("O cabeçalho contém nomes de coluna duplicados.")

    try:
        dataframe = pd.read_csv(
            StringIO(texto),
            sep=separador,
            dtype={"type": "string", "nameOrig": "string", "nameDest": "string"},
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError("O arquivo não contém transações.") from exc

    dataframe.columns = [str(coluna).strip() for coluna in dataframe.columns]
    if dataframe.empty:
        raise ValueError("O arquivo não contém transações.")
    return dataframe
