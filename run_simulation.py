"""Script principal para executar o simulador de detecção de fraudes com CrewAI."""

from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.orquestração.fluxo_crewai import orchestrate_transaction
from src.preprocessamento.carregar_paysim import PROCESSED_DATA_PATH

# Carrega variáveis de ambiente
load_dotenv()


def load_sample_transactions(n_samples: int = 10) -> List[Dict[str, Any]]:
    """Carrega transações de exemplo do dataset processado.

    O dataset real do PaySim processado pode ter milhões de linhas (~470MB de
    CSV bruto / ~400MB de Parquet), o que pode não caber confortavelmente em
    memória em todos os ambientes (``pd.read_parquet`` carregando o arquivo
    inteiro chegou a causar OOM em uma máquina com ~4GB de RAM). Como este
    script só precisa de algumas transações de exemplo, lemos apenas um
    row-group do Parquet (via ``pyarrow``) e amostramos a partir dele, em vez
    de carregar o arquivo inteiro.
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {PROCESSED_DATA_PATH}. "
            "Execute primeiro: python src/preprocessamento/carregar_paysim.py"
        )

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(PROCESSED_DATA_PATH))
    chunk = pf.read_row_group(0).to_pandas()

    # Removido random_state para variar as amostras a cada execução
    sample_df = chunk.sample(n=min(n_samples, len(chunk)))
    return sample_df.to_dict('records')


def run_crewai_simulation(transaction: Dict[str, Any], use_llm_judge: bool = True) -> Dict[str, Any]:
    """Executa a simulação completa com o Agente Juiz (CrewAI + Gemini) para uma transação.

    A decisão final e a justificativa retornadas são exatamente as produzidas
    por ``orchestrate_transaction`` — ou pelo Juiz-LLM real, ou (em fallback)
    pela regra determinística. Anteriormente esta função recalculava a
    decisão a partir do score com limiares diferentes (0.75/0.45) dos usados
    pelo Juiz (0.80/0.50) e descartava a justificativa original; isso foi
    corrigido para não haver mais duas fontes de verdade divergentes.
    """
    return orchestrate_transaction(transaction, use_llm_judge=use_llm_judge)


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
        print("- O dataset foi processado (execute src/preprocessamento/carregar_paysim.py)")
        print("- As dependências estão instaladas")
        print("- A chave API do Gemini está configurada no .env (opcional — sem ela, o sistema usa a regra determinística)")


if __name__ == "__main__":
    main()
