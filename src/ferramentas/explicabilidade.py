"""Explicabilidade por z-score das features de uma transação."""

from collections.abc import Mapping
from typing import Any

from src.preprocessamento.carregar_dataset import IF_FEATURE_NAMES

CANDIDATAS_TOP3 = [name for name in IF_FEATURE_NAMES if name != "log_amount"]


def _direcao(z: float) -> str:
    if z > 0:
        return "acima"
    if z < 0:
        return "abaixo"
    return "na_media"


def top3_por_zscore(
    valores: Mapping[str, float], stats: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    """Retorna as 3 features candidatas de maior |z|, com empates em ordem fixa."""
    itens = []
    for nome in CANDIDATAS_TOP3:
        valor = float(valores[nome])
        media = float(stats[nome]["mean"])
        desvio = float(stats[nome]["std"])
        desvio_zero = desvio == 0
        z = 0.0 if desvio_zero else (valor - media) / desvio
        itens.append(
            {
                "nome": nome,
                "valor": valor,
                "z": z,
                "direcao": _direcao(z),
                "desvio_zero": desvio_zero,
            }
        )
    return sorted(itens, key=lambda item: abs(item["z"]), reverse=True)[:3]
