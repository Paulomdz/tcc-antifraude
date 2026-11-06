"""Une a resposta do lote da API às linhas amostradas e calcula métricas."""

import pandas as pd


def _linha(origem: pd.Series, indice: int, decisao: str, score: float, scores: dict, erro: str) -> dict:
    linha = {
        "indice": indice,
        "tipo": origem["type"],
        "valor": float(origem["amount"]),
        "origem": origem["nameOrig"],
        "destino": origem["nameDest"],
        "decisao": decisao,
        "score_final": float(score),
        "comportamental": scores.get("comportamental"),
        "temporal": scores.get("temporal"),
        "identidade": scores.get("identidade"),
        "erro": erro,
    }
    if "isFraud" in origem.index:
        linha["isFraud"] = origem["isFraud"]
    return linha


def _como_tabela(linhas: list[dict]) -> pd.DataFrame:
    resultado = pd.DataFrame(linhas)
    for coluna in ("comportamental", "temporal", "identidade"):
        resultado[coluna] = pd.to_numeric(resultado[coluna], errors="coerce")
    return resultado


def montar_resultados(amostra: pd.DataFrame, resposta: dict) -> pd.DataFrame:
    """Uma linha por transação do lote fast, na ordem do índice devolvido pela API."""
    amostra = amostra.reset_index(drop=True)
    itens = resposta.get("resultados", [])
    if len(itens) != len(amostra):
        raise ValueError(
            f"A API analisou {len(itens)} transações, mas a amostra local tem "
            f"{len(amostra)}. Confira se dashboard e API estão na mesma versão."
        )
    return _como_tabela(
        [
            _linha(
                amostra.iloc[int(item["indice"])], int(item["indice"]), item["decisao"],
                item["score_final"], item.get("scores") or {}, item.get("erro", ""),
            )
            for item in itens
        ]
    )


def montar_resultados_individuais(amostra: pd.DataFrame, respostas: list[dict]) -> pd.DataFrame:
    """Lote production: uma resposta de /analise_transacao (ou {"erro": ...}) por linha."""
    amostra = amostra.reset_index(drop=True)
    if len(respostas) != len(amostra):
        raise ValueError(
            f"Há {len(respostas)} respostas para {len(amostra)} transações da amostra."
        )
    linhas = []
    for indice, resposta in enumerate(respostas):
        origem = amostra.iloc[indice]
        if "decisao" in resposta:
            linha = _linha(origem, indice, resposta["decisao"], resposta["score_final"],
                           resposta.get("scores") or {}, "")
            linha["justificativa"] = resposta.get("justificativa", "")
        else:
            linha = _linha(origem, indice, "ERRO", 0.0, {}, str(resposta.get("erro", "")))
            linha["justificativa"] = ""
        linhas.append(linha)
    return _como_tabela(linhas)


def transacao_da_linha(linha: pd.Series) -> dict:
    """Linha do arquivo -> corpo de POST /analise_transacao.

    Saldo anterior vazio fica de fora de propósito: a API recusa (422) e a
    transação vira ERRO, em vez de ser analisada com um saldo inventado.
    """

    def numero(coluna: str, padrao: float = 0.0) -> float | None:
        valor = pd.to_numeric(linha.get(coluna), errors="coerce")
        return padrao if pd.isna(valor) else float(valor)

    transacao = {
        "step": int(numero("step", 12.0)),
        "type": str(linha["type"]),
        "amount": numero("amount"),
        "nameOrig": str(linha["nameOrig"]),
        "newbalanceOrig": numero("newbalanceOrig"),
        "nameDest": str(linha["nameDest"]),
        "oldbalanceDest": numero("oldbalanceDest"),
        "newbalanceDest": numero("newbalanceDest"),
    }
    if not pd.isna(pd.to_numeric(linha.get("oldbalanceOrg"), errors="coerce")):
        transacao["oldbalanceOrg"] = float(linha["oldbalanceOrg"])
    return transacao


def _razao(numerador: int, denominador: int) -> float | None:
    return numerador / denominador if denominador else None


def _prf(previsto: pd.Series, real: pd.Series) -> dict:
    tp = int((previsto & real).sum())
    fp = int((previsto & ~real).sum())
    fn = int((~previsto & real).sum())
    precisao = _razao(tp, tp + fp)
    recall = _razao(tp, tp + fn)
    f1 = None
    if precisao is not None and recall is not None and precisao + recall > 0:
        f1 = 2 * precisao * recall / (precisao + recall)
    return {"precisao": precisao, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calcular_metricas(df: pd.DataFrame) -> dict | None:
    """Precisão, recall e F1 em duas leituras; None se o arquivo não tem isFraud.

    Estrita: só BLOQUEADO conta como fraude prevista.
    Ampla: BLOQUEADO ou REVISÃO contam como fraude prevista.
    Linhas com ERRO ou sem rótulo válido ficam de fora.
    """
    if "isFraud" not in df.columns:
        return None
    rotulo = pd.to_numeric(df["isFraud"], errors="coerce")
    validas = df[(df["decisao"] != "ERRO") & rotulo.isin([0, 1])]
    if validas.empty:
        return None
    real = pd.to_numeric(validas["isFraud"]).astype(int).astype(bool)
    bloqueado = validas["decisao"] == "BLOQUEADO"
    revisao = validas["decisao"] == "REVISÃO"
    return {
        "avaliadas": len(validas),
        "fraudes": int(real.sum()),
        "excluidas": len(df) - len(validas),
        "estrita": _prf(bloqueado, real),
        "ampla": _prf(bloqueado | revisao, real),
        "taxa_revisao": float(revisao.mean()),
    }
