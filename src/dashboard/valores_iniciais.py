"""Valores iniciais do formulário, vindos dos dados e não de números fixos."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATASET = Path(__file__).resolve().parents[2] / "dataset" / "dataset_pix_banco.csv"
STEP_MAXIMO_TREINO = 600  # transações com step < 600 formam o treino
SALDO_PADRAO = 1000.0
VALOR_PADRAO = 100.0


@lru_cache(maxsize=1)
def medianas_legitimas(caminho: str = str(DATASET)) -> dict[str, dict[str, float]]:
    """Mediana de saldo e valor das transações legítimas por tipo (step < 600)."""
    try:
        df = pd.read_csv(
            caminho, usecols=["step", "type", "amount", "oldbalanceOrg", "isFraud"]
        )
    except (OSError, ValueError):
        return {}
    legitimas = df[(df["step"] < STEP_MAXIMO_TREINO) & (df["isFraud"] == 0)]
    agrupado = legitimas.groupby("type")[["oldbalanceOrg", "amount"]].median()
    return {
        tipo: {"saldo": float(linha["oldbalanceOrg"]), "valor": float(linha["amount"])}
        for tipo, linha in agrupado.iterrows()
    }


def valores_do_tipo(tipo: str, maximo: float) -> tuple[float, float, bool]:
    """(valor, saldo, veio_dos_dados), limitados a ``maximo`` e valor <= saldo."""
    medianas = medianas_legitimas().get(tipo)
    saldo = medianas["saldo"] if medianas else SALDO_PADRAO
    valor = medianas["valor"] if medianas else VALOR_PADRAO
    saldo = round(min(max(saldo, 0.0), maximo), 2)
    valor = round(min(max(valor, 0.0), saldo), 2)
    return valor, saldo, medianas is not None


CONTAS_POR_PREFIXO = 5  # 3 prefixos x 5 contas = 15 contas por lista
SEED_CONTAS = 42
CONTAS_PADRAO = {
    "origem": ["C1234567"],
    "destino": ["C7654321"],
}


@lru_cache(maxsize=1)
def contas_do_dataset(caminho: str = str(DATASET)) -> dict[str, list[str]]:
    """15 contas reais da base para origem e 15 para destino.

    Pega 5 contas de cada prefixo (letra inicial) mais comum da coluna, sorteadas
    com seed fixa, para a lista mostrar os tipos de conta que existem na base.
    """
    try:
        df = pd.read_csv(caminho, usecols=["nameOrig", "nameDest"], dtype=str)
    except (OSError, ValueError):
        return {chave: list(valor) for chave, valor in CONTAS_PADRAO.items()}
    listas = {}
    for chave, coluna in (("origem", "nameOrig"), ("destino", "nameDest")):
        contas = pd.Series(df[coluna].dropna().unique())
        prefixos = contas.str[0].value_counts().index[:3]
        escolhidas: list[str] = []
        for prefixo in prefixos:
            do_prefixo = contas[contas.str[0] == prefixo]
            amostra = do_prefixo.sample(
                n=min(CONTAS_POR_PREFIXO, len(do_prefixo)), random_state=SEED_CONTAS
            )
            escolhidas.extend(sorted(amostra))
        listas[chave] = escolhidas
    return listas
