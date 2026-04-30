"""Definições de agentes CrewAI usando Gemini 1.5 Flash."""

import os
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (chave API)
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.1,  
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class FraudSimulatorAgents:
    """Gerenciador de agentes para o simulador de fraude."""

    def especialista_comportamental(self):
        return Agent(
            role="Analista Comportamental (ML)",
            goal="Identificar anomalias estatísticas na transação atual.",
            backstory=(
                "Você é um especialista em Machine Learning. Sua função é interpretar "
                "o resultado do modelo Isolation Forest. Se o modelo indicar -1, você "
                "deve explicar por que as features da transação (montante, saldos) parecem suspeitas."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    def especialista_temporal(self):
        return Agent(
            role="Analista de Séries Temporais (LSTM)",
            goal="Detectar desvios no padrão sequencial de transações do usuário.",
            backstory=(
                "Você analisa o fluxo do tempo. Sua especialidade é entender se uma "
                "transação faz sentido dado o histórico recente do cliente, interpretando "
                "o output de uma rede neural LSTM."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    def especialista_identidade(self):
        return Agent(
            role="Analista de Grafos e Identidade (GNN)",
            goal="Verificar conexões suspeitas entre contas de origem e destino.",
            backstory=(
                "Você foca na rede de contatos. Analisa se a conta de destino já foi "
                "associada a fraudes ou se a conexão entre as partes é atípica usando modelos de grafos."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    def juiz_final(self):
        return Agent(
            role="Juiz de Risco Financeiro",
            goal="Consolidar os pareceres dos especialistas e dar o veredito final.",
            backstory=(
                "Você é a autoridade máxima. Você recebe os relatórios técnico-comportamentais, "
                "temporais e de identidade. Sua decisão deve equilibrar segurança e experiência do cliente. "
                "Você deve responder com: APROVADO, BLOQUEADO ou REVISÃO MANUAL, seguido de uma justificativa."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=True  # O Juiz pode fazer perguntas de volta aos especialistas
        )