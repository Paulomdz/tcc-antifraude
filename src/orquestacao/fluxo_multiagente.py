"""Orquestra os modelos e agentes nos modos fast, production e deep."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.ferramentas.ferramentas_agentes import (
    behavior_specialist_tool,
    deep_multiagent_tool,
    identity_specialist_tool,
    judge_tool,
    llm_judge_tool,
    temporal_specialist_tool,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GEMINI_NOT_CONFIGURED = "Gemini não configurado. Defina GOOGLE_API_KEY no .env."


def _require_gemini_key() -> None:
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise RuntimeError(GEMINI_NOT_CONFIGURED)


def _run_specialists(transaction: dict[str, Any]) -> dict[str, Any]:
    behavior = behavior_specialist_tool(transaction)
    temporal = temporal_specialist_tool(transaction)
    identity = identity_specialist_tool(transaction)
    return {
        "behavior": behavior,
        "temporal": temporal,
        "identity": identity,
    }


def orchestrate_ml_only(transaction: dict[str, Any]) -> dict[str, Any]:
    """Modo fast: três modelos e agregação determinística, sem chamadas LLM."""
    outputs = _run_specialists(transaction)
    decision = judge_tool(
        transaction=transaction,
        behavior_score=outputs["behavior"]["score"],
        temporal_score=outputs["temporal"]["score"],
        identity_score=outputs["identity"]["score"],
        specialist_outputs=outputs,
    )
    return {
        "transaction": transaction,
        **outputs,
        "decision": decision,
        "modo": "fast",
    }


def orchestrate_transaction(
    transaction: dict[str, Any], modo: str = "production"
) -> dict[str, Any]:
    """Executa production (Juiz LLM) ou deep (quatro agentes LLM).

    O modo fast fica explicitamente separado em ``orchestrate_ml_only``. A
    chave ausente ou uma falha no CrewAI/Gemini nunca troca silenciosamente o
    caminho solicitado por uma decisão determinística.
    """
    if modo not in {"production", "deep"}:
        raise ValueError("modo deve ser 'production' ou 'deep'.")
    _require_gemini_key()

    outputs = _run_specialists(transaction)
    scores = {
        "behavior_score": outputs["behavior"]["score"],
        "temporal_score": outputs["temporal"]["score"],
        "identity_score": outputs["identity"]["score"],
    }
    if modo == "production":
        decision = llm_judge_tool(
            transaction=transaction,
            **scores,
            specialist_outputs=outputs,
        )
    else:
        decision = deep_multiagent_tool(
            transaction=transaction,
            specialist_outputs=outputs,
        )
    return {
        "transaction": transaction,
        **outputs,
        "decision": decision,
        "modo": modo,
    }


def orchestrate_batch(
    transactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Batch usa fast para manter execução rápida e independente de chave."""
    return [orchestrate_ml_only(transaction) for transaction in transactions]
