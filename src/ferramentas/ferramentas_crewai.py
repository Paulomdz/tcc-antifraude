"""Ferramentas que conectam agentes CrewAI a Python."""

import re
from typing import Any, Dict, Tuple

from .inferencia_modelos import (
    gnn_identity_score,
    isolation_forest_score,
    lstm_sequence_score,
)


def _hora_do_dia(transaction: Dict[str, Any]) -> int:
    """Deriva a hora do dia (0-23) a partir do campo ``step``.

    O PaySim define ``step`` como o número de horas desde o início da
    simulação (1 a 744, cobrindo 31 dias) — ou seja, ``step`` já É uma
    contagem de horas, não de minutos. Para uso interativo (dashboard/API),
    ``step`` é preenchido diretamente com a hora do dia (0-23). Em ambos os
    casos, `step % 24` recupera corretamente a hora do dia:
      - dado real do PaySim, step=100 -> 100 % 24 = 4  (04h, madrugada)
      - entrada manual, step=4        -> 4   % 24 = 4  (04h, madrugada)

    Anteriormente o código tratava ``step`` como minutos desde a meia-noite
    (`step // 60`), o que produzia horários incorretos para qualquer
    transação vinda do dataset real do PaySim.
    """
    step_value = int(transaction.get("step", 0))
    return step_value % 24


def behavior_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """tool agente comportamental que retorna um score de anomalia."""
    score = isolation_forest_score(transaction)

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

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

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

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

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

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
    """Tool do agente juiz (determinística) que decide a ação final e gera justificativa em NL.

    Este é o caminho de decisão usado como fallback sempre que o Juiz-LLM
    (``llm_judge_tool``) não está disponível (sem GOOGLE_API_KEY, dependências
    ausentes ou erro na chamada ao Gemini) e também o caminho usado por
    padrão na análise em lote, por ser determinístico, rápido e sem custo de
    API.
    """
    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

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


# ---------------------------------------------------------------------------
# Juiz-LLM (CrewAI + Google Gemini)
# ---------------------------------------------------------------------------

_DECISOES_VALIDAS = {"Aprovada", "Revisão Humana necessária", "Recusada"}

# Agente Juiz é construído uma única vez e reutilizado entre chamadas
# (evita reconstruir o LLM a cada transação).
_judge_agent = None


def _get_judge_agent():
    """Cria (uma vez) e retorna o Agent CrewAI do Juiz. Lança se indisponível."""
    global _judge_agent
    if _judge_agent is None:
        from src.orquestração.definicoes_agentes import FraudSimulatorAgents

        _judge_agent = FraudSimulatorAgents().juiz_final()
    return _judge_agent


def _parse_judge_output(raw_output: str) -> Tuple[str, float, str]:
    """Extrai (decisao, score, justificativa) da resposta em texto do LLM."""
    decision_match = re.search(r"DECISAO:\s*(.+)", raw_output, re.IGNORECASE)
    score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", raw_output, re.IGNORECASE)
    justification_match = re.search(
        r"JUSTIFICATIVA:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL
    )

    if not (decision_match and score_match and justification_match):
        raise ValueError(f"Resposta do Juiz-LLM fora do formato esperado: {raw_output!r}")

    decision_raw = decision_match.group(1).strip().splitlines()[0].strip()
    if decision_raw in _DECISOES_VALIDAS:
        decision = decision_raw
    else:
        normalized = decision_raw.lower()
        if "recus" in normalized or "bloque" in normalized:
            decision = "Recusada"
        elif "revis" in normalized:
            decision = "Revisão Humana necessária"
        elif "aprov" in normalized:
            decision = "Aprovada"
        else:
            raise ValueError(f"Decisão do Juiz-LLM não reconhecida: {decision_raw!r}")

    score = max(0.0, min(1.0, float(score_match.group(1))))
    justification = justification_match.group(1).strip()

    return decision, score, justification


def llm_judge_tool(
    transaction: Dict[str, Any],
    behavior_score: float,
    temporal_score: float,
    identity_score: float,
    specialist_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Tool do agente juiz que consolida os pareceres via LLM (Gemini) orquestrado por CrewAI.

    Não reprocessa os dados brutos da transação: sintetiza as inferências
    numéricas e as explicações textuais já produzidas pelos três
    especialistas, pondera eventuais conflitos e produz uma decisão final
    acompanhada de justificativa em linguagem natural gerada pelo LLM —
    conforme descrito na monografia (§2.5 e §3.7).

    Lança exceção se o CrewAI/LLM não estiver disponível ou a chamada falhar;
    o chamador (``fluxo_crewai.orchestrate_transaction``) deve tratar isso e
    cair no fallback determinístico (``judge_tool``).
    """
    from crewai import Crew, Task

    agent = _get_judge_agent()

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"
    amount = float(transaction.get("amount", 0.0))
    tx_type = str(transaction.get("type", "")).upper()

    description = (
        "Avalie a transação financeira abaixo com base nos pareceres técnicos dos três "
        "especialistas e emita o veredito final, seguindo rigorosamente as regras de risco "
        "bancário brasileiro descritas no seu backstory (bloqueio imediato para alto valor "
        "na madrugada).\n\n"
        f"Transação: tipo={tx_type}, valor=R$ {amount:,.2f}, horário={time_24h} (24h), "
        f"origem={transaction.get('nameOrig', 'N/A')}, destino={transaction.get('nameDest', 'N/A')}.\n\n"
        f"Parecer do especialista Comportamental (Isolation Forest) — score {behavior_score:.3f}: "
        f"{specialist_outputs['behavior']['details']['explanation']}\n\n"
        f"Parecer do especialista Temporal (LSTM) — score {temporal_score:.3f}: "
        f"{specialist_outputs['temporal']['details']['explanation']}\n\n"
        f"Parecer do especialista de Identidade (GNN) — score {identity_score:.3f}: "
        f"{specialist_outputs['identity']['details']['explanation']}\n\n"
        "Responda ESTRITAMENTE no formato abaixo, em três linhas, sem nenhum texto adicional "
        "antes ou depois:\n"
        "DECISAO: <Aprovada|Revisão Humana necessária|Recusada>\n"
        "SCORE: <número decimal entre 0.0 e 1.0>\n"
        "JUSTIFICATIVA: <justificativa em linguagem natural, em português, citando os três "
        "pareceres e a regra de risco aplicada>"
    )

    task = Task(
        description=description,
        agent=agent,
        expected_output=(
            "Exatamente três linhas no formato DECISAO/SCORE/JUSTIFICATIVA, sem texto extra."
        ),
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    raw_output = str(crew.kickoff())

    decision, score, justification = _parse_judge_output(raw_output)

    return {
        "decision": decision,
        "score": score,
        "justification": justification,
        "specialist_outputs": specialist_outputs,
        "raw_llm_output": raw_output,
    }
