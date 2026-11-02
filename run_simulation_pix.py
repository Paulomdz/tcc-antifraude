"""Script principal para executar o simulador de detecção de fraudes com CrewAI."""

from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.orquestacao.fluxo_multiagente import orchestrate_transaction
from src.preprocessamento.carregar_dataset import PROCESSED_DATA_PATH

# Carrega variáveis de ambiente
load_dotenv()


def load_sample_transactions(n_samples: int = 10) -> List[Dict[str, Any]]:
    """Carrega transações de exemplo do dataset processado."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {PROCESSED_DATA_PATH}. "
            "Execute primeiro: python -m src.preprocessamento.carregar_dataset"
        )

    df = pd.read_parquet(PROCESSED_DATA_PATH)

    # Removido random_state para variar as amostras a cada execução
    sample_df = df.sample(n=min(n_samples, len(df)))
    return sample_df.to_dict('records')


def run_crewai_simulation(
    transaction: Dict[str, Any], modo: str = "production"
) -> Dict[str, Any]:
    """Executa explicitamente um dos modos production/deep da orquestração."""
    return orchestrate_transaction(transaction, modo=modo)


def main():
    """Ponto de entrada principal do simulador."""
    print(" Iniciando Simulador de Detecção de Fraudes com CrewAI + Gemini")
    print("=" * 60)

    try:

        transactions = load_sample_transactions(5)
        print(f"Carregadas {len(transactions)} transações para análise")

        for i, transaction in enumerate(transactions, 1):
            print(f"\nAnalisando Transação {i}/{len(transactions)}")
            print(f"   Tipo: {transaction.get('type')}, Valor: R$ {transaction.get('amount', 0):.2f}")

            result = run_crewai_simulation(transaction)

            print("   Resultado:")
            print(f"      Decisão: {result['decision']['decision']}")
            print(f"      Score Máximo: {result['decision']['score']:.3f}")
            print(f"      Justificativa: {result['decision']['justification'][:100]}...")

        print("\nSimulação concluída!")

    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("Verifique se:")
        print("- O dataset foi processado (execute python -m src.preprocessamento.carregar_dataset)")
        print("- As dependências estão instaladas")
        print("- A chave GOOGLE_API_KEY está configurada no .env para production/deep")


if __name__ == "__main__":
    main()
