"""Script principal para executar o simulador de detecção de fraudes com CrewAI."""

import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

from src.orquestração.fluxo_crewai import orchestrate_transaction
from src.ferramentas.ferramentas_crewai import (
    behavior_specialist_tool,
    identity_specialist_tool,
    temporal_specialist_tool,
)
from src.preprocessamento.carregar_paysim import PROCESSED_DATA_PATH

# Carrega variáveis de ambiente
load_dotenv()


def load_sample_transactions(n_samples: int = 10) -> List[Dict[str, Any]]:
    """Carrega transações de exemplo do dataset processado."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {PROCESSED_DATA_PATH}. "
            "Execute primeiro: python src/preprocessamento/carregar_paysim.py"
        )

    df = pd.read_parquet(PROCESSED_DATA_PATH)

    # Removido random_state para variar as amostras a cada execução
    sample_df = df.sample(n=n_samples)
    return sample_df.to_dict('records')


def run_crewai_simulation(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Executa a simulação completa com agentes CrewAI."""

    result = orchestrate_transaction(transaction)

    max_score = max(result['behavior']['score'], result['temporal']['score'], result['identity']['score'])

    if max_score >= 0.8:
        decision = "BLOQUEADO"
        justification = f"Transação bloqueada devido a alto risco detectado (score: {max_score:.3f})."
    elif max_score >= 0.5:
        decision = "REVISÃO MANUAL"
        justification = f"Transação requer revisão manual devido a risco moderado (score: {max_score:.3f})."
    else:
        decision = "APROVADO"
        justification = f"Transação aprovada - risco baixo detectado (score: {max_score:.3f})."

    result['decision']['decision'] = decision
    result['decision']['justification'] = justification

    return result


def main():
    """Ponto de entrada principal do simulador."""
    print(" Iniciando Simulador de Detecção de Fraudes com CrewAI + Gemini")
    print("=" * 60)

    try:

        transactions = load_sample_transactions(5)
        print(f"Carregadas {len(transactions)} transações para análise")

        for i, transaction in enumerate(transactions, 1):
            print(f"\nAnalisando Transação {i}/{len(transactions)}")
            print(f"   Tipo: {transaction.get('type')}, Valor: ${transaction.get('amount', 0):.2f}")

            result = run_crewai_simulation(transaction)

            print("   Resultado:")
            print(f"      Decisão: {result['decision']['decision']}")
            print(f"      Score Máximo: {result['decision']['score']:.3f}")
            print(f"      Justificativa: {result['decision']['justification'][:100]}...")

        print("\nSimulação concluída!")

    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("Verifique se:")
        print("- O dataset foi processado (execute load_paysim.py)")
        print("- As dependências estão instaladas")
        print("- A chave API do Gemini está configurada no .env")


if __name__ == "__main__":
    main()