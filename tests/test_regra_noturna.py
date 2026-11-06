"""Regra noturna: janela, elegibilidade, total acumulado e lote igual ao individual."""

import pandas as pd
import pytest

from src.ferramentas import regra_noturna as rn
from src.ferramentas.historico import GrafoPassado


def _tx(step, amount, tipo="PIX_P2P", origem="C1", destino="C2"):
    return {
        "step": step,
        "type": tipo,
        "amount": amount,
        "nameOrig": origem,
        "nameDest": destino,
        "oldbalanceOrg": 5_000.0,
    }


@pytest.mark.parametrize(
    ("hora", "esperado"),
    [(19, False), (20, True), (23, True), (0, True), (5, True), (6, False), (12, False)],
)
def test_janela_noturna_vai_das_20h_as_6h(hora, esperado):
    assert rn.e_noite(24 * 3 + hora) is esperado


@pytest.mark.parametrize(
    ("step", "inicio"),
    [(20, 20), (23, 20), (24, 20), (29, 20), (44, 44), (48, 44)],
)
def test_inicio_da_noite(step, inicio):
    assert rn.inicio_da_noite(step) == inicio


@pytest.mark.parametrize(
    ("transacao", "esperado"),
    [
        (_tx(22, 1), True),
        (_tx(22, 1, "PIX_QRCODE"), True),
        (_tx(22, 1, "PIX_AUTOMATICO"), True),
        (_tx(22, 1, "TED"), True),
        (_tx(22, 1, "PIX_SAQUE"), False),
        (_tx(22, 1, "PIX_TROCO"), False),
        (_tx(22, 1, "TEF"), False),
        (_tx(12, 1), False),
        (_tx(22, 1, destino="M2"), False),
        (_tx(22, 1, origem="B1"), False),
        (_tx(22, 1, destino="C1"), False),
        ({"step": 22, "amount": 1, "type": "PIX_P2P"}, False),
    ],
)
def test_elegibilidade(transacao, esperado):
    assert rn.elegivel(transacao) is esperado


@pytest.mark.parametrize(
    ("anterior", "valor", "aplica"),
    [(900.0, 200.0, True), (0.0, 1_000.0, False), (0.0, 1_000.01, True), (1_100.0, 50.0, False)],
)
def test_so_a_transacao_que_passa_do_limite_gera_revisao(anterior, valor, aplica):
    resultado = rn.avaliar_regra_noturna(_tx(22, valor), total_anterior=anterior)

    assert resultado["aplica"] is aplica
    assert resultado["total_com_atual"] == pytest.approx(anterior + valor)


def test_nao_elegivel_nao_consulta_o_historico(monkeypatch):
    monkeypatch.setattr(
        rn, "total_noturno_anterior", lambda *_a, **_k: pytest.fail("não deveria consultar")
    )

    assert rn.avaliar_regra_noturna(_tx(12, 5_000.0))["aplica"] is False


def _contexto():
    return pd.DataFrame(
        [
            _tx(1, 700.0),  # noite anterior (começou no step -4)
            _tx(20, 300.0),  # mesma noite
            _tx(22, 400.0, "TED"),  # mesma noite
            _tx(22, 999.0, destino="M5"),  # recebedor não é pessoa física
            _tx(23, 999.0, "BOLETO"),  # tipo fora da regra
            _tx(23, 999.0, origem="C9"),  # outro pagador
            _tx(25, 200.0),  # mesmo step da transação atual: não soma
            _tx(27, 999.0),  # futuro
        ]
    )


def test_total_noturno_soma_so_o_passado_elegivel_da_mesma_noite():
    grafo = GrafoPassado(_contexto())

    assert rn.total_noturno_anterior(_tx(25, 100.0), grafo) == pytest.approx(700.0)
    assert rn.total_noturno_anterior(_tx(25, 100.0, origem="C77"), grafo) == 0.0


def test_lote_igual_a_uma_transacao_por_vez():
    contexto = _contexto()
    grafo = GrafoPassado(contexto)
    lote = rn.marcar_lote(contexto)

    individual = [
        rn.avaliar_regra_noturna(linha, grafo=grafo)["aplica"]
        for linha in contexto.to_dict("records")
    ]

    assert lote.tolist() == individual
    # 300 + 400 = 700 até o step 22; o passo 25 soma 200 e não passa; o 27 passa
    assert lote.tolist() == [False, False, False, False, False, False, False, True]
