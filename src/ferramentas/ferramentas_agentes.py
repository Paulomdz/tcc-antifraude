"""Ferramentas que conectam agentes CrewAI a Python."""

import re
from typing import Any, Dict, Tuple

from .inferencia_modelos import (
    gnn_identity_score,
    isolation_forest_score,
    lstm_sequence_score,
)
from .regra_noturna import FONTE, LIMITE_NOTURNO, avaliar_regra_noturna, e_noite


def _hora_do_dia(transaction: Dict[str, Any]) -> int:
    """Deriva a hora do dia (0-23) a partir do campo ``step``.

    O dataset Pix/Banco define ``step`` como o número de horas desde o início
    da simulação (0 a 743, cobrindo 31 dias) — ou seja, ``step`` já É uma
    contagem de horas, não de minutos. Para uso interativo (dashboard/API),
    ``step`` é preenchido diretamente com a hora do dia (0-23). Em ambos os
    casos, `step % 24` recupera corretamente a hora do dia:
      - dado do dataset Pix/Banco, step=100 -> 100 % 24 = 4 (04h, madrugada)
      - entrada manual, step=4        -> 4   % 24 = 4  (04h, madrugada)

    Anteriormente o código tratava ``step`` como minutos desde a meia-noite
    (`step // 60`), o que produzia horários incorretos para qualquer
    transação vinda do dataset Pix/Banco.
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
                f"e saldo anterior à execução. Horário: {time_24h} (formato 24h)."
            ),
        },
    }


def temporal_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Tool do agente temporal que retorna um score de sequenciamento."""
    sequence = transaction.get("sequence", [transaction])
    score = lstm_sequence_score(sequence, current_transaction=transaction)

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

    amount = float(transaction.get("amount", 0.0))
    periodo = "noturno (20h às 6h)" if e_noite(transaction.get("step", 0)) else "diurno"

    return {
        "score": score,
        "label": "temporal",
        "details": {
            "explanation": (
                f"O modelo temporal analisa o histórico de transações e identifica padrões "
                f"de comportamento inesperados. Horário da transação: {time_24h} (24h), "
                f"período {periodo}. Valor: R$ {amount:,.2f}."
            ),
        },
    }


def identity_specialist_tool(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Tool do agente de identidade que analisa relações suspeitas de grafo."""
    score = gnn_identity_score(transaction)

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

    return {
        "score": score,
        "label": "identity",
        "details": {
            "explanation": (
                "O modelo de identidade avalia relações entre as contas da transação "
                "e potenciais conexões suspeitas sem usar identificadores como features. "
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
    aplicar_regra_madrugada: bool = True,
    regra_noturna: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Agregador determinístico usado somente pelo modo fast, sem chamada LLM.

    O score é a média ponderada dos três agentes. A regra noturna (limite de
    R$ 1.000 por noite em Pix e TED entre pessoas físicas) só eleva uma
    aprovação para revisão: não bloqueia e não muda o score.
    ``aplicar_regra_madrugada=False`` desliga a regra (usado na avaliação).
    ``regra_noturna`` permite passar o resultado já calculado.
    """
    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"

    amount = float(transaction.get("amount", 0.0))
    tx_type = str(transaction.get("type", "")).upper()

    score = 0.35 * behavior_score + 0.30 * temporal_score + 0.35 * identity_score

    if score >= 0.80:
        decision = "Recusada"
    elif score >= 0.50:
        decision = "Revisão Humana necessária"
    else:
        decision = "Aprovada"

    regra = None
    if aplicar_regra_madrugada:
        regra = regra_noturna if regra_noturna is not None else avaliar_regra_noturna(transaction)
        decision = aplicar_piso_noturno(decision, regra)

    justification = (
        f"Decisão: {decision} | Horário: {time_24h} | Valor: R$ {amount:,.2f} | Tipo: {tx_type}\n"
        f"Análise dos agentes:\n"
        f"  • Comportamental: {behavior_score:.3f}\n"
        f"  • Temporal: {temporal_score:.3f}\n"
        f"  • Identidade: {identity_score:.3f}\n"
        f"Risco consolidado: {score:.3f}"
        f"{_texto_regra_noturna(regra)}"
    )

    return {
        "decision": decision,
        "score": float(score),
        "justification": justification,
        "specialist_outputs": specialist_outputs,
        "regra_noturna": regra,
    }


def aplicar_piso_noturno(decision: str, regra: Dict[str, Any] | None) -> str:
    """Se a regra noturna se aplica, a decisão mínima é revisão humana."""
    if regra and regra.get("aplica") and decision == "Aprovada":
        return "Revisão Humana necessária"
    return decision


def _texto_regra_noturna(regra: Dict[str, Any] | None) -> str:
    if not regra or not regra.get("aplica"):
        return ""
    return (
        f"\nRegra noturna ({FONTE}): o total da conta pagadora nesta noite passa a "
        f"R$ {regra['total_com_atual']:,.2f}, acima do limite de R$ {LIMITE_NOTURNO:,.2f} "
        "para Pix e TED entre pessoas físicas. A transação vai, no mínimo, para revisão humana."
    )


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
        from src.orquestacao.definicoes_agentes import FraudSimulatorAgents

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
    regra_noturna: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Tool do agente juiz que consolida os pareceres via LLM (Gemini) orquestrado por CrewAI.

    Não reprocessa os dados brutos da transação: sintetiza as inferências
    numéricas e as explicações textuais já produzidas pelos três
    especialistas, pondera eventuais conflitos e produz uma decisão final
    acompanhada de justificativa em linguagem natural gerada pelo LLM —
    conforme descrito na monografia (§2.5 e §3.7).

    Lança exceção se o CrewAI/LLM não estiver disponível ou a chamada falhar.
    No modo production, não existe fallback determinístico.
    """
    from crewai import Crew, Task

    agent = _get_judge_agent()

    hours = _hora_do_dia(transaction)
    time_24h = f"{hours:02d}:00"
    amount = float(transaction.get("amount", 0.0))
    tx_type = str(transaction.get("type", "")).upper()
    if regra_noturna and regra_noturna.get("aplica"):
        texto_regra = (
            f"Regra noturna ({FONTE}): APLICA. O total da conta pagadora nesta noite passa a "
            f"R$ {regra_noturna['total_com_atual']:,.2f}, acima do limite de "
            f"R$ {LIMITE_NOTURNO:,.2f}. A decisão mínima é Revisão Humana necessária.\n\n"
        )
    else:
        texto_regra = f"Regra noturna ({FONTE}): não se aplica a esta transação.\n\n"

    description = (
        "Avalie a transação financeira abaixo com base nos pareceres técnicos dos três "
        "especialistas e emita o veredito final, seguindo as regras descritas no seu "
        "backstory.\n\n"
        f"Transação: tipo={tx_type}, valor=R$ {amount:,.2f}, horário={time_24h} (24h), "
        "identificadores de conta omitidos.\n\n"
        f"{texto_regra}"
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


def deep_multiagent_tool(
    transaction: Dict[str, Any],
    specialist_outputs: Dict[str, Any],
    regra_noturna: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Executa os três especialistas LLM e o juiz final no modo deep."""
    from crewai import Crew, Process, Task
    from src.orquestacao.definicoes_agentes import FraudSimulatorAgents

    agents = FraudSimulatorAgents()
    behavior_agent = agents.especialista_comportamental()
    temporal_agent = agents.especialista_temporal()
    identity_agent = agents.especialista_identidade()
    judge_agent = agents.juiz_final()
    agent_tasks = [
        Task(
            description=(
                "Interprete o parecer do modelo comportamental para a transação. "
                f"Score: {specialist_outputs['behavior']['score']:.4f}. "
                f"Dados técnicos: {specialist_outputs['behavior']['details']['explanation']}"
            ),
            agent=behavior_agent,
            expected_output="Síntese técnica do risco comportamental em português.",
        ),
        Task(
            description=(
                "Interprete o parecer do modelo temporal para a transação. "
                f"Score: {specialist_outputs['temporal']['score']:.4f}. "
                f"Dados técnicos: {specialist_outputs['temporal']['details']['explanation']}"
            ),
            agent=temporal_agent,
            expected_output="Síntese técnica do risco temporal em português.",
        ),
        Task(
            description=(
                "Interprete o parecer do modelo GNN de relações entre transações. "
                f"Score: {specialist_outputs['identity']['score']:.4f}. "
                f"Dados técnicos: {specialist_outputs['identity']['details']['explanation']}"
            ),
            agent=identity_agent,
            expected_output="Síntese técnica do risco de identidade em português.",
        ),
    ]
    hours = _hora_do_dia(transaction)
    judge_task = Task(
        description=(
            "Consolide os três pareceres anteriores e emita a decisão final. "
            f"Tipo={transaction.get('type', 'N/A')}; "
            f"valor=R$ {float(transaction.get('amount', 0.0)):,.2f}; "
            f"horário={hours:02d}:00. "
            + (
                f"Regra noturna ({FONTE}): APLICA; o total noturno da conta pagadora passa "
                f"de R$ {LIMITE_NOTURNO:,.2f}, então a decisão mínima é Revisão Humana necessária. "
                if regra_noturna and regra_noturna.get("aplica")
                else f"Regra noturna ({FONTE}): não se aplica. "
            )
            + "Responda estritamente com DECISAO, SCORE e JUSTIFICATIVA em linhas "
            "separadas. Decisões permitidas: Aprovada, Revisão Humana necessária "
            "ou Recusada; score entre 0 e 1."
        ),
        agent=judge_agent,
        context=agent_tasks,
        expected_output=(
            "DECISAO: <Aprovada|Revisão Humana necessária|Recusada>\n"
            "SCORE: <número entre 0 e 1>\nJUSTIFICATIVA: <texto>"
        ),
    )
    crew = Crew(
        agents=[behavior_agent, temporal_agent, identity_agent, judge_agent],
        tasks=[*agent_tasks, judge_task],
        process=Process.sequential,
        verbose=False,
    )
    raw_output = str(crew.kickoff())
    decision, score, justification = _parse_judge_output(raw_output)
    return {
        "decision": decision,
        "score": score,
        "justification": justification,
        "specialist_outputs": specialist_outputs,
        "raw_llm_output": raw_output,
    }
