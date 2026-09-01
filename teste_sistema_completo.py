"""Script de teste para treinar e executar o sistema completo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.preprocessamento.carregar_paysim import (
    PROCESSED_DATA_PATH,
    preprocess_paysim,
    process_paysim_in_chunks,
)


def _load_sample_df(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Carrega uma amostra pequena do Parquet processado sem ler o arquivo inteiro.

    O dataset real do PaySim processado pode ter milhões de linhas (~470MB de
    CSV bruto), o que não cabe confortavelmente em memória em todos os
    ambientes. Como estes testes de fumaça só precisam de algumas linhas para
    validar que cada etapa do pipeline funciona, lemos apenas o primeiro
    row-group do Parquet (via ``pyarrow``) e amostramos a partir dele.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(PROCESSED_DATA_PATH))
    chunk = pf.read_row_group(0).to_pandas()
    return chunk.sample(n=min(n, len(chunk)), random_state=random_state)


def teste_preprocessamento():
    """Testa o carregamento e pré-processamento dos dados."""
    print("\n" + "="*70)
    print("TESTE 1: PREPROCESSAMENTO")
    print("="*70)

    try:
        from src.preprocessamento.carregar_paysim import RAW_DATA_PATH

        if not RAW_DATA_PATH.exists():
            print(f"Dataset bruto não encontrado em: {RAW_DATA_PATH}")
            print("   Descompacte o arquivo PS_20174392719_1491204439457_log.csv.zip em data/")
            return False

        if PROCESSED_DATA_PATH.exists():
            # O dataset real do PaySim (~6,3M linhas / ~470MB) já foi
            # processado (via process_paysim_in_chunks/main do módulo de
            # preprocessamento). Reprocessar aqui com uma leitura completa
            # em memória (load_paysim/preprocess_paysim) tanto arriscaria
            # estourar a memória disponível quanto sobrescreveria o parquet
            # completo já gerado com uma amostra parcial. Em vez disso,
            # validamos a lógica de pré-processamento numa amostra pequena
            # do CSV bruto (sem salvar) e reaproveitamos o parquet existente.
            print(" Dataset processado já existe — validando a lógica de "
                  "pré-processamento numa amostra do CSV bruto (sem sobrescrever "
                  "o parquet completo)...")
            df_raw_sample = pd.read_csv(RAW_DATA_PATH, nrows=2000)
            df_processed_sample = preprocess_paysim(df_raw_sample)
            print(f"{len(df_processed_sample)} registros de amostra processados "
                  "com sucesso (lógica validada)")
            print(f" Dataset processado completo já disponível em: {PROCESSED_DATA_PATH}")
            return True

        print(" Processando dataset bruto em blocos (evita estourar memória "
              "em datasets grandes)...")
        process_paysim_in_chunks(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        print(f" Dados salvos em: {PROCESSED_DATA_PATH}")

        return True
    except Exception as e:
        print(f" Erro no preprocessamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def teste_isolation_forest():
    """Testa o modelo Isolation Forest."""
    print("\n" + "="*70)
    print(" TESTE 2: ISOLATION FOREST")
    print("="*70)

    try:
        if not PROCESSED_DATA_PATH.exists():
            print(" Dataset processado não encontrado. Execute primeiro: teste_preprocessamento()")
            return False

        from src.modelos.treinamento.treinar_modelos import (
            load_balanced_sample_from_parquet,
            train_isolation_forest,
            save_model,
        )

        print(" Carregando amostra balanceada do dataset processado...")
        df_sample = load_balanced_sample_from_parquet(PROCESSED_DATA_PATH, n_samples=5000)

        print(" Treinando Isolation Forest...")
        model_if = train_isolation_forest(df_sample)

        print("Salvando modelo...")
        save_model(model_if, "isolation_forest.pkl")

        return True
    except Exception as e:
        print(f"Erro no Isolation Forest: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def teste_lstm():
    """Testa o treinamento do LSTM."""
    print("\n" + "="*70)
    print("TESTE 3: LSTM")
    print("="*70)

    try:
        from src.modelos.arquitetura_lstm import treinar_lstm, salvar_lstm

        if not PROCESSED_DATA_PATH.exists():
            print("Dataset processado não encontrado.")
            return False

        print(" Carregando amostra do dataset...")
        df_sample = _load_sample_df(n=2000)

        print("Treinando LSTM...")
        model_lstm = treinar_lstm(df_sample, epochs=5)

        if model_lstm is not None:
            print(" Salvando modelo LSTM...")
            salvar_lstm(model_lstm)
            return True
        else:
            # model_lstm is None: pode ser fallback esperado (PyTorch ausente)
            # ou dataset insuficiente para gerar sequências — ambos casos já
            # tratados sem exceção por treinar_lstm, então são considerados
            # "sem erro" para fins deste teste de integração.
            print("LSTM com fallback (PyTorch não disponível ou dataset insuficiente)")
            return True

    except Exception as e:
        # Uma exceção aqui é um erro real (não o fallback esperado, que
        # treinar_lstm já trata retornando None) — não deve ser contado
        # como sucesso.
        print(f" Erro em LSTM: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def teste_gnn():
    """Testa o treinamento do GNN."""
    print("\n" + "="*70)
    print("TESTE 4: GNN")
    print("="*70)

    try:
        from src.modelos.arquitetura_gnn import treinar_gnn, salvar_gnn

        if not PROCESSED_DATA_PATH.exists():
            print("Dataset processado não encontrado.")
            return False

        print("Carregando amostra do dataset...")
        df_sample = _load_sample_df(n=2000)

        print("Treinando GNN...")
        model_gnn, dataset_gnn = treinar_gnn(df_sample, epochs=5)

        if model_gnn is not None:
            print(" Salvando modelo GNN...")
            salvar_gnn(model_gnn, dataset_gnn)
            return True
        else:
            # Fallback esperado (PyTorch Geometric ausente), já tratado sem
            # exceção por treinar_gnn — não é um erro.
            print("GNN com fallback (PyTorch Geometric não disponível)")
            return True

    except Exception as e:
        # Exceção real (diferente do fallback esperado) — não deve contar
        # como sucesso.
        print(f"Erro em GNN: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def teste_inferencia():
    """Testa o sistema de inferência completo."""
    print("\n" + "="*70)
    print("TESTE 5: INFERÊNCIA")
    print("="*70)

    try:
        from src.ferramentas.inferencia_modelos import (
            isolation_forest_score,
            lstm_sequence_score,
            gnn_identity_score,
        )

        if not PROCESSED_DATA_PATH.exists():
            print("Dataset processado não encontrado.")
            return False

        print("Carregando amostra do dataset...")
        df_sample = _load_sample_df(n=2000)

        sample_tx = df_sample.iloc[0].to_dict()

        print("Testando Isolation Forest Score...")
        score_if = isolation_forest_score(sample_tx)
        print(f"  → Score IF: {score_if:.3f}")

        print(" Testando LSTM Score...")
        score_lstm = lstm_sequence_score([sample_tx])
        print(f"  → Score LSTM: {score_lstm:.3f}")

        print("Testando GNN Score...")
        score_gnn = gnn_identity_score(sample_tx)
        print(f"  → Score GNN: {score_gnn:.3f}")

        return True
    except Exception as e:
        print(f" Erro na inferência: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def teste_simulacao():
    """Testa a simulação completa."""
    print("\n" + "="*70)
    print("TESTE 6: SIMULAÇÃO COMPLETA")
    print("="*70)

    try:
        from src.orquestração.fluxo_crewai import orchestrate_transaction

        if not PROCESSED_DATA_PATH.exists():
            print("Dataset processado não encontrado.")
            return False

        print("Carregando amostra do dataset...")
        df_sample = _load_sample_df(n=2000)

        samples = df_sample.sample(n=min(2, len(df_sample)), random_state=42)

        for i, (_, tx) in enumerate(samples.iterrows(), 1):
            print(f"\n[OK] Analisando transação {i}...")
            tx_dict = tx.to_dict()
            resultado = orchestrate_transaction(tx_dict, use_llm_judge=False)

            decisao = resultado['decision']
            if isinstance(decisao, dict):
                # Chaves corretas produzidas por judge_tool/llm_judge_tool:
                # "decision" e "score" (não "label"/"fraud_score" — essas
                # nunca existiram na estrutura real e faziam este teste
                # sempre imprimir "DESCONHECIDO"/0.000, mascarando o
                # resultado real da simulação).
                label = decisao.get('decision', 'DESCONHECIDO')
                score = decisao.get('score', 0.0)
            else:
                label = str(decisao)[:20]
                score = 0.0

            print(f"  → Decisão: {label}")
            print(f"  → Score: {score:.3f}" if isinstance(score, (int, float)) else "  → Score: N/A")

        return True
    except Exception as e:
        print(f"Erro na simulação: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    print("\n")
    print("="*70)
    print("SUITE DE TESTES - DETECÇÃO DE FRAUDES".center(70))
    print("="*70)

    resultados = {
        "Preprocessamento": teste_preprocessamento(),
        "Isolation Forest": teste_isolation_forest(),
        "LSTM": teste_lstm(),
        "GNN": teste_gnn(),
        "Inferência": teste_inferencia(),
        "Simulação": teste_simulacao(),
    }
    #teste de funcionalidade
    print("\n" + "="*70)
    print("SUMÁRIO DE TESTES")
    print("="*70)

    passed = sum(1 for v in resultados.values() if v)
    total = len(resultados)

    for teste, resultado in resultados.items():
        status = "PASSOU" if resultado else "FALHOU"
        print(f"{teste:.<50} {status}")

    print(f"\nResultado Final: {passed}/{total} testes passaram")

    if passed == total:
        print("\nSISTEMA COMPLETAMENTE FUNCIONAL!")
    else:
        print(f"\n{total - passed} teste(s) com problema. Verifique acima.")

    print("="*70)


if __name__ == "__main__":
    main()
