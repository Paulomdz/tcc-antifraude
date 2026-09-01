"""Fluxo de orquestração do CrewAI para simulação de detecção de fraudes."""

import importlib.util
import os
from typing import Dict

from src.ferramentas.ferramentas_crewai import (
    behavior_specialist_tool,
    identity_specialist_tool,
    judge_tool,
    llm_judge_tool,
    temporal_specialist_tool,
)


def _llm_judge_available() -> bool:
    """Verifica rapidamente (sem chamar a API) se o Juiz-LLM pode ser tentado.

    Checa apenas a presença da variável de ambiente e a instalação dos
    pacotes — nenhuma chamada de rede é feita aqui. Isso garante que, em
    ambientes sem GOOGLE_API_KEY (como testes automatizados), o sistema cai
    direto no fallback determinístico sem tentar (e esperar por) uma chamada
    de rede que sabidamente vai falhar.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        return False
    return importlib.util.find_spec("crewai") is not None


def orchestrate_transaction(
    transaction: Dict[str, object], use_llm_judge: bool = True
) -> Dict[str, object]:
    """Executa o fluxo de avaliação completo para uma única transação.

    Args:
        transaction: dados da transação a analisar.
        use_llm_judge: quando True (padrão), tenta consolidar a decisão via
            Agente Juiz real (CrewAI + Google Gemini), caindo automaticamente
            para a regra determinística (``judge_tool``) se a chave de API
            não estiver configurada, as dependências não estiverem instaladas
            ou a chamada ao LLM falhar por qualquer motivo (rede, formato de
            resposta inesperado, etc.). Quando False, usa diretamente o
            caminho determinístico — usado por padrão na análise em lote,
            para evitar uma chamada de API por transação em lotes de
            centenas de itens.
    """
    # O Orquestrador coordena os especialistas e passa as pontuações para o Juiz.
    behavior_result = behavior_specialist_tool(transaction)
    temporal_result = temporal_specialist_tool(transaction)
    identity_result = identity_specialist_tool(transaction)

    specialist_outputs = {
        "behavior": behavior_result,
        "temporal": temporal_result,
        "identity": identity_result,
    }

    decision_result = None
    if use_llm_judge and _llm_judge_available():
        try:
            decision_result = llm_judge_tool(
                transaction=transaction,
                behavior_score=behavior_result["score"],
                temporal_score=temporal_result["score"],
                identity_score=identity_result["score"],
                specialist_outputs=specialist_outputs,
            )
        except Exception as exc:  # noqa: BLE001 - fallback deliberado e amplo
            print(
                "[AVISO] Juiz-LLM (CrewAI/Gemini) indisponível ou falhou "
                f"({exc}); usando decisão determinística como fallback."
            )
            decision_result = None

    if decision_result is None:
        decision_result = judge_tool(
            transaction=transaction,
            behavior_score=behavior_result["score"],
            temporal_score=temporal_result["score"],
            identity_score=identity_result["score"],
            specialist_outputs=specialist_outputs,
        )

    return {
        "transaction": transaction,
        "behavior": behavior_result,
        "temporal": temporal_result,
        "identity": identity_result,
        "decision": decision_result,
    }


def orchestrate_batch(
    transactions: list[Dict[str, object]], use_llm_judge: bool = False
) -> list[Dict[str, object]]:
    """Orquestra uma lista de transações de forma sequencial.

    Por padrão, usa o caminho determinístico (``use_llm_judge=False``) para
    lotes, já que uma chamada real ao LLM por transação em um lote de
    centenas de itens é lenta e tem custo de API proporcional ao tamanho do
    lote. Passe ``use_llm_judge=True`` explicitamente para forçar o Juiz-LLM
    também no lote.
    """
    results = []
    for transaction in transactions:
        results.append(orchestrate_transaction(transaction, use_llm_judge=use_llm_judge))
    return results
