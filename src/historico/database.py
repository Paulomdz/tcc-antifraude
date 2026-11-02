"""Acesso centralizado ao histórico SQLite de análises antifraude."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "historico.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path = DB_PATH) -> None:
    connection = get_connection(db_path)
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analises (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    transaction_hash TEXT,
                    type TEXT,
                    amount REAL,
                    score_final REAL,
                    decisao TEXT,
                    score_behavior REAL,
                    score_temporal REAL,
                    score_identity REAL,
                    duracao_ms INTEGER,
                    llm_usado TEXT,
                    erro TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analises_timestamp "
                "ON analises(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analises_decisao "
                "ON analises(decisao)"
            )
    finally:
        connection.close()


def salvar_analise(registro: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    timestamp = registro.get("timestamp") or datetime.now(timezone.utc).isoformat()
    connection = get_connection(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO analises (
                    timestamp, transaction_hash, type, amount, score_final,
                    decisao, score_behavior, score_temporal, score_identity,
                    duracao_ms, llm_usado, erro
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    registro.get("transaction_hash"),
                    registro.get("type"),
                    float(registro.get("amount", 0.0)),
                    float(registro.get("score_final", 0.0)),
                    registro.get("decisao"),
                    float(registro.get("score_behavior", 0.0)),
                    float(registro.get("score_temporal", 0.0)),
                    float(registro.get("score_identity", 0.0)),
                    int(registro.get("duracao_ms", 0)),
                    registro.get("llm_usado"),
                    registro.get("erro"),
                ),
            )
            return int(cursor.lastrowid or 0)
    finally:
        connection.close()


def obter_historico(
    limit: int = 100,
    decisao: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    init_db(db_path)
    connection = get_connection(db_path)
    try:
        if decisao:
            cursor = connection.execute(
                "SELECT * FROM analises WHERE decisao = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (decisao, max(1, limit)),
            )
        else:
            cursor = connection.execute(
                "SELECT * FROM analises ORDER BY timestamp DESC LIMIT ?",
                (max(1, limit),),
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def obter_estatisticas_24h(db_path: Path = DB_PATH) -> dict[str, Any]:
    init_db(db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    connection = get_connection(db_path)
    try:
        counts = connection.execute(
            "SELECT decisao, COUNT(*) AS total FROM analises "
            "WHERE timestamp >= ? GROUP BY decisao",
            (cutoff,),
        )
        count_by_decision = {
            row["decisao"]: row["total"] for row in counts.fetchall()
        }
        metrics = connection.execute(
            """
            SELECT COUNT(*) AS total_transacoes,
                   AVG(score_final) AS score_medio,
                   AVG(duracao_ms) AS latencia_media_ms
            FROM analises
            WHERE timestamp >= ?
            """,
            (cutoff,),
        ).fetchone()
        return {
            "periodo": "ultimas_24h",
            "total_transacoes": int(metrics["total_transacoes"] or 0),
            "contagem_por_decisao": count_by_decision,
            "score_medio": round(float(metrics["score_medio"] or 0.0), 4),
            "latencia_media_ms": round(
                float(metrics["latencia_media_ms"] or 0.0), 2
            ),
        }
    finally:
        connection.close()
