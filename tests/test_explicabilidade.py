import copy
import json

from src.ferramentas.explicabilidade import CANDIDATAS_TOP3, top3_por_zscore
from src.preprocessamento.carregar_dataset import (
    FEATURE_STATS_PATH,
    IF_FEATURE_NAMES,
)


def _entrada():
    valores = {name: 0.0 for name in IF_FEATURE_NAMES}
    stats = {name: {"mean": 0.0, "std": 1.0} for name in IF_FEATURE_NAMES}
    return valores, stats


def test_top3_retorna_tres_itens_ordenados_por_abs_z():
    valores, stats = _entrada()
    valores.update(amount=1.0, hora_norm=3.0, is_madrugada=2.0)

    top3 = top3_por_zscore(valores, stats)

    assert [item["nome"] for item in top3] == [
        "hora_norm",
        "is_madrugada",
        "amount",
    ]
    assert set(top3[0]) == {"nome", "valor", "z", "direcao", "desvio_zero"}


def test_top3_nunca_inclui_log_amount():
    valores, stats = _entrada()
    valores["log_amount"] = 1000.0

    assert "log_amount" not in {
        item["nome"] for item in top3_por_zscore(valores, stats)
    }


def test_top3_considera_as_17_features_candidatas():
    assert len(CANDIDATAS_TOP3) == 17
    assert set(CANDIDATAS_TOP3) == set(IF_FEATURE_NAMES) - {"log_amount"}
    for nome in CANDIDATAS_TOP3:
        valores, stats = _entrada()
        valores[nome] = 5.0
        assert top3_por_zscore(valores, stats)[0]["nome"] == nome


def test_top3_calcula_z_com_media_e_desvio_do_stats():
    valores, stats = _entrada()
    valores["amount"] = 30.0
    stats["amount"] = {"mean": 10.0, "std": 4.0}

    item = top3_por_zscore(valores, stats)[0]

    assert item["nome"] == "amount"
    assert item["valor"] == 30.0
    assert item["z"] == 5.0


def test_top3_direcao_acima_e_abaixo_da_media():
    valores, stats = _entrada()
    valores.update(amount=3.0, hora_norm=-2.0)

    top3 = top3_por_zscore(valores, stats)

    assert top3[0]["direcao"] == "acima"
    assert top3[1]["direcao"] == "abaixo"
    assert top3[2]["z"] == 0
    assert top3[2]["direcao"] == "na_media"


def test_top3_z_negativo_grande_vence_z_positivo_menor():
    valores, stats = _entrada()
    valores.update(amount=2.0, hora_norm=-4.0)

    top3 = top3_por_zscore(valores, stats)

    assert top3[0]["nome"] == "hora_norm"
    assert top3[0]["z"] == -4.0


def test_top3_desvio_zero_retorna_z_zero_e_marca():
    valores, stats = _entrada()
    valores["amount"] = 50.0
    stats = {name: {"mean": 0.0, "std": 0.0} for name in IF_FEATURE_NAMES}

    item = top3_por_zscore(valores, stats)[0]

    assert item["nome"] == "amount"
    assert item["z"] == 0
    assert item["desvio_zero"] is True
    assert item["direcao"] == "na_media"


def test_top3_empate_mantem_ordem_estavel():
    valores, stats = _entrada()
    for nome in CANDIDATAS_TOP3:
        valores[nome] = 1.0

    top3 = top3_por_zscore(valores, stats)

    assert [item["nome"] for item in top3] == CANDIDATAS_TOP3[:3]


def test_top3_nao_altera_entradas():
    valores, stats = _entrada()
    valores["amount"] = 2.0
    valores_antes = copy.deepcopy(valores)
    stats_antes = copy.deepcopy(stats)

    top3_por_zscore(valores, stats)

    assert valores == valores_antes
    assert stats == stats_antes


def test_feature_stats_real_tem_todas_as_candidatas():
    stats = json.loads(FEATURE_STATS_PATH.read_text(encoding="utf-8"))

    for nome in CANDIDATAS_TOP3:
        assert {"mean", "std"} <= set(stats[nome])
