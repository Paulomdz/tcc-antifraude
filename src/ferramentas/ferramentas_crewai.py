"""Ferramentas que conectam agentes CrewAI a Python."""

from typing import Any, Dict

from .inferencia_modelos import (
    gnn_identity_score,
    isolation_forest_score,
    lstm_sequence_score,
)


def behavior_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """tool agente comportamental que retorna um score de anomalia."""
    score = isolation_forest_score(transaction)
    return {
        "score": score,
        "label": "behavior",
        "details": {
            "explanation": (
                "O modelo de comportamento avalia a transação por irregularidade de valores "
                "e mudanças de saldo."
            ),
        },
    }


def temporal_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Tool do agente temporal que retorna um score de sequenciamento."""
    sequence = transaction.get("sequence", [transaction])
    score = lstm_sequence_score(sequence)
    return {
        "score": score,
        "label": "temporal",
        "details": {
            "explanation": (
                "O modelo temporal analisa o histórico de transações e identifica padrões "
                "de comportamento inesperados."
            ),
        },
    }


def identity_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Tool do agente de identidade que analisa relações suspeitas de grafo."""
    graph_data = {
        "origin": transaction.get("nameOrig"),
        "destination": transaction.get("nameDest"),
        "amount": transaction.get("amount"),
    }
    score = gnn_identity_score(graph_data)
    return {
        "score": score,
        "label": "identity",
        "details": {
            "explanation": (
                "O modelo de identidade avalia conexões entre contas e potenciais relação "
                "com redes de fraude."
            ),
        },
    }


def judge_tool(
    transaction: Dict[str, Any],
    behavior_score: float,
    temporal_score: float,
    identity_score: float,
    specialist_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Tool do agente juiz que decide a ação final e gera justificativa em NL."""
    max_score = max(behavior_score, temporal_score, identity_score)

    if max_score >= 0.8:
        decision = "Recusada"
    elif max_score >= 0.5:
        decision = "Revisão Humana necessária"
    else:
        decision = "Aprovada"

    justification = (
        f"Decisão: {decision}. "
        f"O agente comportamental indicou {behavior_score:.3f}, "
        f"o temporal indicou {temporal_score:.3f} e "
        f"o de identidade indicou {identity_score:.3f}."
    )

    return {
        "decision": decision,
        "score": max_score,
        "justification": justification,
        "specialist_outputs": specialist_outputs,
    }
