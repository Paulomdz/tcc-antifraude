"""Persistência local das análises antifraude."""

from .database import (
    DB_PATH,
    init_db,
    obter_estatisticas_24h,
    obter_historico,
    salvar_analise,
)

__all__ = [
    "DB_PATH",
    "init_db",
    "obter_estatisticas_24h",
    "obter_historico",
    "salvar_analise",
]
