"""Rótulos em linguagem simples para tipos, colunas e características."""

import pandas as pd

# código do tipo -> (nome simples, descrição curta)
TIPOS: dict[str, tuple[str, str]] = {
    "PIX_P2P": ("Pix entre pessoas", "Transferência Pix de uma pessoa para outra."),
    "PIX_QRCODE": ("Pix por QR Code", "Pagamento Pix feito lendo um QR Code."),
    "PIX_SAQUE": ("Pix Saque", "Retirada de dinheiro em espécie no comércio, via Pix."),
    "PIX_TROCO": ("Pix Troco", "Troco em dinheiro dado numa compra paga com Pix."),
    "PIX_AUTOMATICO": (
        "Pix Automático",
        "Cobrança recorrente autorizada antes, como mensalidades.",
    ),
    "CARTAO_ECOMMERCE": ("Cartão em compra online", "Pagamento com cartão em loja virtual."),
    "CARTAO_PRESENCIAL": ("Cartão presencial", "Pagamento com cartão na maquininha."),
    "TEF": ("TEF", "Transferência eletrônica de fundos entre contas."),
    "BOLETO": ("Boleto", "Pagamento de boleto bancário."),
    "SAQUE": ("Saque", "Retirada de dinheiro em caixa eletrônico ou agência."),
    "TED": ("TED", "Transferência eletrônica entre bancos."),
    "EMPRESTIMO": ("Empréstimo", "Crédito liberado pelo banco ao cliente."),
}

# Frase usada no top-3 (característica "tipo da transação")
_FRASE_TIPO = {
    "CARTAO_ECOMMERCE": "compra online com cartão",
    "CARTAO_PRESENCIAL": "pagamento presencial com cartão",
    "TEF": "transferência TEF",
    "BOLETO": "pagamento de boleto",
    "SAQUE": "saque em dinheiro",
    "TED": "transferência TED",
    "EMPRESTIMO": "empréstimo",
}

COLUNAS = {
    "step": "hora",
    "type": "tipo",
    "amount": "valor",
    "nameOrig": "conta de origem",
    "oldbalanceOrg": "saldo antes (origem)",
    "newbalanceOrig": "saldo depois (origem)",
    "nameDest": "conta de destino",
    "oldbalanceDest": "saldo antes (destino)",
    "newbalanceDest": "saldo depois (destino)",
    "isFraud": "fraude real",
}

_FEATURES = {
    "amount": "valor da transação",
    "is_incoming": "origem é um banco (salário, empréstimo)",
    "hora_norm": "horário da transação",
    "amount_to_balance_ratio": "valor em relação ao saldo",
    "is_madrugada": "transação na madrugada (0h às 5h)",
}
_BINARIAS = {"is_incoming", "is_madrugada"}

_DIRECAO = {
    "acima": "acima do normal",
    "abaixo": "abaixo do normal",
    "na_media": "na média",
}

DECISOES = ["APROVADO", "REVISÃO", "BLOQUEADO", "ERRO"]
ICONES = {"APROVADO": "🟢", "REVISÃO": "🟡", "BLOQUEADO": "🔴", "ERRO": "⚪"}


def nome_tipo(codigo: str) -> str:
    return TIPOS.get(str(codigo), (str(codigo), ""))[0]


def descricao_tipo(codigo: str) -> str:
    return TIPOS.get(str(codigo), ("", ""))[1]


def numero_br(valor: float, casas: int = 2) -> str:
    texto = f"{float(valor):,.{casas}f}"
    return texto.replace(",", "~").replace(".", ",").replace("~", ".")


def moeda_br(valor: float) -> str:
    return f"R$ {numero_br(valor)}"


def rotulo_feature(nome: str) -> str:
    if nome.startswith("type_"):
        codigo = nome[len("type_"):]
        frase = _FRASE_TIPO.get(codigo) or nome_tipo(codigo)
    else:
        frase = _FEATURES.get(nome, nome.replace("_", " "))
    return frase[:1].upper() + frase[1:]


def valor_feature(nome: str, valor: float) -> str:
    """Valor da característica em texto simples."""
    valor = float(valor)
    if nome.startswith("type_") or nome in _BINARIAS:
        return "sim" if valor >= 0.5 else "não"
    if nome == "hora_norm":
        return f"{round(valor * 23):02d}h"
    if nome == "amount":
        return moeda_br(valor)
    if nome == "amount_to_balance_ratio":
        return f"{numero_br(valor * 100, 1)}% do saldo"
    return numero_br(valor, 3)


def direcao_feature(direcao: str, desvio_zero: bool = False) -> str:
    if desvio_zero:
        return "sem variação no treino"
    return _DIRECAO.get(direcao, "na média")


def traduzir_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas do arquivo e troca códigos de tipo por nomes simples."""
    saida = df.copy()
    if "type" in saida:
        saida["type"] = saida["type"].map(nome_tipo)
    return saida.rename(columns=COLUNAS)
