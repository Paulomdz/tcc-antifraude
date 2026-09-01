"""Definições de agentes CrewAI usando Google Gemini.

Observação de compatibilidade: versões recentes do CrewAI (>=1.x) não aceitam mais
um objeto langchain (ex.: ChatGoogleGenerativeAI) diretamente no parâmetro ``llm``
de ``Agent`` — é necessário usar ``crewai.LLM``, que roteia nativamente para o
Gemini via string no formato ``"gemini/<modelo>"``. Por isso o LLM é construído
aqui com ``crewai.LLM`` em vez de ``langchain_google_genai``.
"""

import os

# Desativa telemetria do CrewAI (evita chamadas de rede extras a
# telemetry.crewai.com e a respectiva latência de retry quando a rede não
# permite o egress) — deve ser definido antes de importar crewai.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, LLM
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (chave API)
load_dotenv()

# Nome do modelo Gemini configurável via variável de ambiente, para facilitar a
# atualização quando a Google descontinuar/renomear versões do modelo sem
# precisar alterar o código.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def build_judge_llm() -> LLM:
    """Constrói o LLM (Gemini) usado pelo Agente Juiz.

    Lança ``ValueError`` se ``GOOGLE_API_KEY`` não estiver configurada — o
    chamador (``ferramentas_crewai.llm_judge_tool``) deve tratar essa exceção e
    cair no fallback determinístico (``judge_tool``).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não configurada no ambiente (.env).")

    return LLM(
        model=f"gemini/{GEMINI_MODEL}",
        api_key=api_key,
        temperature=0.1,
    )


class FraudSimulatorAgents:
    """Gerenciador de agentes para o simulador de fraude."""

    def __init__(self, llm: LLM | None = None):
        # O LLM só é construído sob demanda (lazy) para não falhar em ambientes
        # sem GOOGLE_API_KEY configurada apenas por importar este módulo.
        self._llm = llm

    def _get_llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_judge_llm()
        return self._llm

    def especialista_comportamental(self):
        return Agent(
            role="Analista Comportamental (Isolation Forest)",
            goal=(
                "Identificar anomalias estatísticas na transação atual, comparando com perfis normais de risco. "
                "Valores de transação muito acima do esperado, especialmente em horários incomuns, devem ser "
                "sinalizados imediatamente."
            ),
            backstory=(
                "Você é um especialista em Machine Learning com foco em detecção de fraudes financeiras. "
                "Sua função é interpretar o resultado do modelo Isolation Forest, que classifica transações "
                "como normais ou anômalas. Se o modelo indicar anomalia, explique por que as features da "
                "transação (montante, saldos, padrão temporal) parecem suspeitas. Você entende que o sistema "
                "financeiro brasileiro tem padrões específicos e que transações de altíssimo valor (R$ 10.000+) "
                "fora do horário comercial são indicadores críticos de risco."
            ),
            llm=self._get_llm(),
            verbose=True,
            allow_delegation=False,
        )

    def especialista_temporal(self):
        return Agent(
            role="Analista de Séries Temporais (LSTM) e Risco Temporal",
            goal=(
                "Detectar desvios no padrão sequencial de transações do usuário, com atenção especial "
                "a cenários de alto risco: transferências de alto valor (R$ 10.000+) ocorrendo em horários "
                "críticos como madrugada (00:00-05:59). Indicadores como débito noturno de valor elevado "
                "devem aumentar significativamente a suspeita de fraude."
            ),
            backstory=(
                "Você é um especialista sênior em análise temporal e comportamento de risco bancário. "
                "Sua especialidade é entender se uma transação faz sentido dado o histórico recente do cliente, "
                "interpretando o output de uma rede neural LSTM. Você conhece bem as regras de risco do sistema "
                "financeiro brasileiro: transferências muito grandes na madrugada são típicas de fraudes "
                "coordenadas. Quando detectar amount >= R$ 10.000 + horário entre 00:00-05:59, considere a "
                "transação ALTAMENTE SUSPEITA e inclinar para BLOQUEIO."
            ),
            llm=self._get_llm(),
            verbose=True,
            allow_delegation=False,
        )

    def especialista_identidade(self):
        return Agent(
            role="Analista de Grafos e Identidade (GNN)",
            goal=(
                "Verificar conexões suspeitas entre contas de origem e destino, com foco especial em padrões "
                "conhecidos de redes de fraude. Transferências para contas novas ou desconhecidas combinadas "
                "com alto valor e madrugada são EXTREMAMENTE SUSPEITAS."
            ),
            backstory=(
                "Você é especialista em análise de grafos e redes sociais financeiras. Sua função é verificar "
                "se a conta de destino já foi associada a fraudes, se a conexão entre as partes é atípica usando "
                "modelos de grafos neurais (GNN). Você conhece as padrões de risco: transferências para contas "
                "novas, em volume elevado (R$ 10.000+), durante a madrugada (00:00-05:59) são características "
                "de operações fraudulentas organizadas. Quando detectar estas combinações, recomende BLOQUEIO."
            ),
            llm=self._get_llm(),
            verbose=True,
            allow_delegation=False,
        )

    def juiz_final(self):
        return Agent(
            role="Juiz de Risco Financeiro (Autoridade Final)",
            goal=(
                "Consolidar os pareceres dos especialistas e dar o veredito final com critérios rígidos de risco "
                "bancário brasileiro. BLOQUEIE IMEDIATAMENTE transferências de alto valor (R$ 10.000+) durante "
                "madrugada (00:00-05:59), independentemente de outros scores."
            ),
            backstory=(
                "Você é a autoridade máxima do sistema de fraude. Recebe os relatórios técnico-comportamentais, "
                "temporais e de identidade dos especialistas. Sua decisão deve ser rigorosa e conservadora em "
                "relação a segurança financeira. Você conhece as regras críticas de risco do sistema financeiro "
                "brasileiro:\n\n"
                "REGRAS RÍGIDAS DE BLOQUEIO:\n"
                "1. Débito (DEBIT) de valor >= R$ 10.000 entre 00:00-05:59 (madrugada) = BLOQUEIO IMEDIATO\n"
                "2. TRANSFER de valor >= R$ 10.000 entre 00:00-05:59 = BLOQUEIO IMEDIATO\n"
                "3. Combinação: alto valor noturno + agentes apontando suspeita = BLOQUEIO\n"
                "4. Se incerto ou houver múltiplas flags, preferir REVISÃO MANUAL a APROVADO\n\n"
                "Responda sempre no formato solicitado pela tarefa: DECISAO, SCORE e JUSTIFICATIVA."
            ),
            llm=self._get_llm(),
            verbose=True,
            allow_delegation=False,
        )
