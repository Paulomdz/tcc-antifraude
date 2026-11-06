"""Definições de agentes CrewAI usando Google Gemini.

Observação de compatibilidade: versões recentes do CrewAI (>=1.x) não aceitam mais
um objeto langchain (ex.: ChatGoogleGenerativeAI) diretamente no parâmetro ``llm``
de ``Agent`` — é necessário usar ``crewai.LLM``, que roteia nativamente para o
Gemini via string no formato ``"gemini/<modelo>"``. Por isso o LLM é construído
aqui com ``crewai.LLM`` em vez de ``langchain_google_genai``.
"""

import os
from pathlib import Path

# Desativa telemetria do CrewAI (evita chamadas de rede extras a
# telemetry.crewai.com e a respectiva latência de retry quando a rede não
# permite o egress) — deve ser definido antes de importar crewai.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, LLM
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (chave API)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def build_judge_llm() -> LLM:
    """Constrói o Gemini; sem chave, interrompe production/deep explicitamente."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("GOOGLE_API_KEY não configurada no ambiente (.env).")

    model_name = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    model_name = model_name.removeprefix("models/")
    return LLM(
        model=f"gemini/{model_name}",
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
                "financeiro brasileiro tem padrões específicos: o Banco Central limita a R$ 1.000 o total "
                "noturno (20h às 6h) de Pix e TED entre pessoas físicas, e valores fora do perfil nesse "
                "período merecem atenção."
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
                "ao período noturno (20h às 6h), em que o total de Pix e TED entre pessoas físicas é "
                "limitado a R$ 1.000 por conta pagadora. Débitos noturnos fora do histórico do cliente "
                "devem aumentar a suspeita de fraude."
            ),
            backstory=(
                "Você é um especialista sênior em análise temporal e comportamento de risco bancário. "
                "Sua especialidade é entender se uma transação faz sentido dado o histórico recente do cliente, "
                "interpretando o output de uma rede neural LSTM. Você conhece bem as regras de risco do sistema "
                "financeiro brasileiro: o limite noturno de R$ 1.000 para Pix e TED entre pessoas físicas "
                "(IN BCB 331/2022, alterada pelas IN BCB 512/2024 e 669/2025). Quando a regra noturna "
                "indicar que o limite foi estourado, recomende no mínimo REVISÃO."
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
                "com valores altos no período noturno (20h às 6h) merecem atenção especial."
            ),
            backstory=(
                "Você é especialista em análise de grafos e redes sociais financeiras. Sua função é verificar "
                "se a conta de destino já foi associada a fraudes, se a conexão entre as partes é atípica usando "
                "modelos de grafos neurais (GNN). Você conhece as padrões de risco: transferências para contas "
                "novas, em volume elevado, durante o período noturno (20h às 6h) são características "
                "de operações fraudulentas organizadas. Quando detectar estas combinações, recomende REVISÃO."
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
                "bancário brasileiro. Quando a regra noturna se aplicar, a decisão mínima é REVISÃO MANUAL."
            ),
            backstory=(
                "Você é a autoridade máxima do sistema de fraude. Recebe os relatórios técnico-comportamentais, "
                "temporais e de identidade dos especialistas. Sua decisão deve ser rigorosa e conservadora em "
                "relação a segurança financeira. Você conhece as regras críticas de risco do sistema financeiro "
                "brasileiro:\n\n"
                "REGRAS DE DECISÃO:\n"
                "1. Regra noturna (IN BCB 331/2022, alterada pelas IN BCB 512/2024 e 669/2025): Pix entre "
                "pessoas, Pix QR Code, Pix Automático ou TED, de pessoa física para outra pessoa física, das "
                "20h às 6h, quando o total acumulado da conta pagadora na noite passa de R$ 1.000 = no mínimo "
                "REVISÃO MANUAL. Não é bloqueio automático e não aumenta o score.\n"
                "2. A tarefa informa se a regra noturna se aplica; não recalcule o total.\n"
                "3. Bloqueio só quando os pareceres dos especialistas indicarem risco alto.\n"
                "4. Se incerto ou houver múltiplas flags, preferir REVISÃO MANUAL a APROVADO\n\n"
                "Responda sempre no formato solicitado pela tarefa: DECISAO, SCORE e JUSTIFICATIVA."
            ),
            llm=self._get_llm(),
            verbose=True,
            allow_delegation=False,
        )
