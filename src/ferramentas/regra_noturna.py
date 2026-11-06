"""Regra noturna do Pix e da TED entre pessoas físicas.

Fonte: Instrução Normativa BCB nº 331/2022, alterada pelas IN BCB nº 512/2024
e nº 669/2025 (limite noturno de R$ 1.000,00 para Pix e TED entre pessoas
físicas, das 20h às 6h).

Como o projeto aplica a regra:
- Noite: hora >= 20 ou hora < 6, com hora = step % 24 (mesma fonte de
  is_madrugada). A noite que começa às 20h termina às 6h do dia seguinte.
- Pagador e recebedor pessoa física: ID começando com "C", e recebedor
  diferente do pagador. MEI não pode ser separado nos dados.
- Tipos (escopo B): PIX_P2P, PIX_QRCODE, PIX_AUTOMATICO e TED. Pix saque e
  troco ficam de fora, embora a norma também os limite à noite.
- O total é acumulado por conta pagadora e por noite, somando só as
  transações elegíveis com step menor que o da transação atual.
- A transação que faz o total passar de R$ 1.000 gera REVISÃO. Isso é escolha
  do projeto, não da norma: não bloqueia e não soma bônus ao score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LIMITE_NOTURNO = 1000.0
TIPOS_REGRA_NOTURNA = frozenset({"PIX_P2P", "PIX_QRCODE", "PIX_AUTOMATICO", "TED"})
FONTE = "IN BCB 331/2022, alterada pelas IN BCB 512/2024 e 669/2025"


def e_noite(step: Any) -> bool:
    hora = int(step) % 24
    return hora >= 20 or hora < 6


def inicio_da_noite(step: Any) -> int:
    """Step das 20h em que começou a noite da transação."""
    passo = int(step)
    hora = passo % 24
    return passo - (hora - 20) if hora >= 20 else passo - hora - 4


def _pessoa_fisica(conta: Any) -> bool:
    return str(conta or "").startswith("C")


def elegivel(transacao: dict[str, Any]) -> bool:
    """Pix (escopo B) ou TED, à noite, de pessoa física para outra pessoa física."""
    origem = str(transacao.get("nameOrig", "") or "")
    destino = str(transacao.get("nameDest", "") or "")
    return (
        "step" in transacao
        and e_noite(transacao["step"])
        and str(transacao.get("type", "")).upper() in TIPOS_REGRA_NOTURNA
        and _pessoa_fisica(origem)
        and _pessoa_fisica(destino)
        and destino != origem
    )


def total_noturno_anterior(transacao: dict[str, Any], grafo: Any = None) -> float:
    """Soma das transações elegíveis do pagador na mesma noite, antes deste step."""
    if grafo is None:
        from src.ferramentas.historico import grafo_do_passado

        grafo = grafo_do_passado(None)
    posicoes = grafo._contas["nameOrig"].get(str(transacao.get("nameOrig", "")))
    if posicoes is None:
        return 0.0
    passo = int(transacao["step"])
    inicio = inicio_da_noite(passo)
    passos = grafo.steps[posicoes]
    janela = posicoes[(passos >= inicio) & (passos < passo)]
    if not len(janela):
        return 0.0
    linhas = grafo.contexto.iloc[janela]
    mascara = marcar_elegiveis(linhas)
    return float(linhas.loc[mascara, "amount"].sum())


def avaliar_regra_noturna(
    transacao: dict[str, Any],
    total_anterior: float | None = None,
    grafo: Any = None,
) -> dict[str, Any]:
    """Diz se a transação estoura o limite noturno. Só consulta o histórico se for elegível."""
    resultado = {
        "aplica": False,
        "elegivel": False,
        "total_anterior": 0.0,
        "total_com_atual": 0.0,
        "limite": LIMITE_NOTURNO,
        "fonte": FONTE,
    }
    if not elegivel(transacao):
        return resultado
    if total_anterior is None:
        total_anterior = total_noturno_anterior(transacao, grafo)
    total = float(total_anterior) + float(transacao.get("amount", 0.0))
    resultado.update(
        elegivel=True,
        total_anterior=float(total_anterior),
        total_com_atual=total,
        aplica=float(total_anterior) <= LIMITE_NOTURNO < total,
    )
    return resultado


def marcar_elegiveis(df: pd.DataFrame) -> pd.Series:
    passo = df["step"].astype(int)
    hora = passo % 24
    origem = df["nameOrig"].astype(str)
    destino = df["nameDest"].astype(str)
    return (
        ((hora >= 20) | (hora < 6))
        & df["type"].astype(str).str.upper().isin(TIPOS_REGRA_NOTURNA)
        & origem.str.startswith("C")
        & destino.str.startswith("C")
        & destino.ne(origem)
    )


def marcar_lote(df: pd.DataFrame) -> pd.Series:
    """Versão em lote, com a mesma regra de uma transação por vez.

    O total anterior de cada transação soma só as elegíveis do mesmo pagador,
    da mesma noite e com step menor; transações do mesmo step não se somam.
    """
    saida = pd.Series(False, index=df.index)
    mascara = marcar_elegiveis(df)
    if not mascara.any():
        return saida
    e = df.loc[mascara, ["step", "nameOrig", "amount"]].copy()
    passo = e["step"].astype(int)
    hora = passo % 24
    e["noite"] = np.where(hora >= 20, passo - (hora - 20), passo - hora - 4)
    por_passo = (
        e.groupby(["nameOrig", "noite", "step"], sort=True)["amount"]
        .sum()
        .rename("soma_passo")
        .reset_index()
    )
    por_passo["anterior"] = (
        por_passo.groupby(["nameOrig", "noite"])["soma_passo"].cumsum()
        - por_passo["soma_passo"]
    )
    e = (
        e.reset_index()
        .merge(
            por_passo[["nameOrig", "noite", "step", "anterior"]],
            on=["nameOrig", "noite", "step"],
            how="left",
        )
        .set_index(e.index.name or "index")
    )
    total = e["anterior"] + e["amount"].astype(float)
    saida.loc[e.index] = (e["anterior"] <= LIMITE_NOTURNO) & (total > LIMITE_NOTURNO)
    return saida
