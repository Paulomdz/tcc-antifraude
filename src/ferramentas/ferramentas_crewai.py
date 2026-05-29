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
    
    # Converte minutos para formato 24h HH:MM
    step_minutes = int(transaction.get("step", 0))
    hours = step_minutes // 60
    minutes = step_minutes % 60
    time_24h = f"{hours:02d}:{minutes:02d}"
    
    return {
        "score": score,
        "label": "behavior",
        "details": {
            "explanation": (
                f"O modelo de comportamento avalia a transação por irregularidade de valores "
                f"e mudanças de saldo. Horário: {time_24h} (formato 24h). "
                f"Se a transação ocorre entre 00:00-05:59 com valor alto, é muito suspeita."
            ),
        },
    }


def temporal_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Tool do agente temporal que retorna um score de sequenciamento."""
    sequence = transaction.get("sequence", [transaction])
    score = lstm_sequence_score(sequence)
    
    # Converte minutos para formato 24h HH:MM
    step_minutes = int(transaction.get("step", 0))
    hours = step_minutes // 60
    minutes = step_minutes % 60
    time_24h = f"{hours:02d}:{minutes:02d}"
    
    # Verifica se está na madrugada (00:00-05:59)
    is_midnight = 0 <= hours < 6
    amount = float(transaction.get("amount", 0.0))
    is_high_value = amount >= 10000.0
    
    extra_warning = ""
    if is_midnight and is_high_value:
        extra_warning = (
            f" ⚠️ ALERTA CRÍTICO: Transação de R$ {amount:,.2f} na madrugada ({time_24h}). "
            "Padrão altamente suspeito de fraude."
        )
    
    return {
        "score": score,
        "label": "temporal",
        "details": {
            "explanation": (
                f"O modelo temporal analisa o histórico de transações e identifica padrões "
                f"de comportamento inesperados. Horário da transação: {time_24h} (24h). "
                f"Valor: R$ {amount:,.2f}."
                f"{extra_warning}"
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
    
    # Converte minutos para formato 24h HH:MM
    step_minutes = int(transaction.get("step", 0))
    hours = step_minutes // 60
    minutes = step_minutes % 60
    time_24h = f"{hours:02d}:{minutes:02d}"
    
    return {
        "score": score,
        "label": "identity",
        "details": {
            "explanation": (
                f"O modelo de identidade avalia conexões entre contas ({transaction.get('nameOrig')} → "
                f"{transaction.get('nameDest')}) e potenciais relações com redes de fraude. "
                f"Horário: {time_24h}. Valor: R$ {float(transaction.get('amount', 0.0)):,.2f}."
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
    
    # Extrai dados críticos da transação
    step_minutes = int(transaction.get("step", 0))
    hours = step_minutes // 60
    minutes = step_minutes % 60
    time_24h = f"{hours:02d}:{minutes:02d}"
    
    amount = float(transaction.get("amount", 0.0))
    tx_type = str(transaction.get("type", "")).upper()
    
    # REGRA RÍGIDA 1: Débito de alto valor na madrugada = BLOQUEIO IMEDIATO
    is_midnight = 0 <= hours < 6
    is_high_value = amount >= 10000.0
    is_debit_or_transfer = tx_type in {"DEBIT", "TRANSFER"}
    
    if is_midnight and is_high_value and is_debit_or_transfer:
        return {
            "decision": "Recusada",
            "score": float(0.95),
            "justification": (
                f"🚨 BLOQUEIO IMEDIATO - REGRA CRÍTICA DE RISCO:\n"
                f"Transação {tx_type} de R$ {amount:,.2f} às {time_24h} (madrugada).\n"
                f"Este padrão (alto valor + madrugada + débito/transferência) é indicador "
                f"crítico de fraude no sistema financeiro brasileiro.\n"
                f"Score: 0.950 (máximo risco)"
            ),
            "specialist_outputs": specialist_outputs,
        }
    
    # Média ponderada com limiares mais equilibrados
    score = 0.35 * behavior_score + 0.30 * temporal_score + 0.35 * identity_score
    
    # REGRA RÍGIDA 2: Se houver padrão noturno suspeito, elevar score
    if is_midnight and is_high_value:
        score = min(score + 0.25, 1.0)
    
    # Limiares de decisão
    if score >= 0.80:
        decision = "Recusada"
    elif score >= 0.50:
        decision = "Revisão Humana necessária"
    else:
        decision = "Aprovada"
    
    justification = (
        f"Decisão: {decision} | Horário: {time_24h} | Valor: R$ {amount:,.2f} | Tipo: {tx_type}\n"
        f"Análise dos agentes:\n"
        f"  • Comportamental: {behavior_score:.3f}\n"
        f"  • Temporal: {temporal_score:.3f}\n"
        f"  • Identidade: {identity_score:.3f}\n"
        f"Risco consolidado: {score:.3f}"
    )
    
    return {
        "decision": decision,
        "score": float(score),
        "justification": justification,
        "specialist_outputs": specialist_outputs,
    }
