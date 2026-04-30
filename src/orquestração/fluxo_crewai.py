"""Fluxo de orquestração do CrewAI para simulação de detecção de fraudes."""

from typing import Dict

from src.ferramentas.ferramentas_crewai import (
    behavior_specialist_tool,
    identity_specialist_tool,
    judge_tool,
    temporal_specialist_tool,
)


def orchestrate_transaction(transaction: Dict[str, object]) -> Dict[str, object]:
    """Executa o fluxo de avaliação completo para uma única transação."""
    # O Orquestrador coordena os especialistas e passa as pontuações para o Juiz.
    behavior_result = behavior_specialist_tool(transaction)
    temporal_result = temporal_specialist_tool(transaction)
    identity_result = identity_specialist_tool(transaction)

    decision_result = judge_tool(
        transaction=transaction,
        behavior_score=behavior_result["score"],
        temporal_score=temporal_result["score"],
        identity_score=identity_result["score"],
        specialist_outputs={
            "behavior": behavior_result,
            "temporal": temporal_result,
            "identity": identity_result,
        },
    )

    return {
        "transaction": transaction,
        "behavior": behavior_result,
        "temporal": temporal_result,
        "identity": identity_result,
        "decision": decision_result,
    }


def orchestrate_batch(transactions: list[Dict[str, object]]) -> list[Dict[str, object]]:
    """Orquestra uma lista de transações de forma sequencial."""
    results = []
    for transaction in transactions:
        results.append(orchestrate_transaction(transaction))
    return results
