"""Executa verificações integradas do pipeline Pix/Banco."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pyarrow.parquet as pq

from src.preprocessamento.carregar_dataset import (
    DEFAULT_CHUNKSIZE,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    process_dataset_in_chunks,
    preprocess_dataset,
)

_TEST_MODEL_DIR: Path | None = None


def _model_path(filename: str) -> Path:
    directory = _TEST_MODEL_DIR or (
        Path(__file__).resolve().parent / "src" / "modelos" / "modelos_salvos"
    )
    return directory / filename


def _load_sample_df(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Lê uma amostra do primeiro grupo do Parquet processado."""
    parquet_file = pq.ParquetFile(str(PROCESSED_DATA_PATH))
    if parquet_file.num_row_groups == 0:
        raise ValueError(f"Dataset processado vazio: {PROCESSED_DATA_PATH}")
    chunk = parquet_file.read_row_group(0).to_pandas()
    if chunk.empty:
        raise ValueError(f"Primeiro grupo do Parquet está vazio: {PROCESSED_DATA_PATH}")
    return chunk.sample(n=min(n, len(chunk)), random_state=random_state)


def teste_preprocessamento() -> bool:
    """Gera o Parquet, se necessário, e valida seu esquema e conteúdo."""
    print("\n" + "=" * 70)
    print("TESTE 1: PREPROCESSAMENTO PIX/BANCO")
    print("=" * 70)
    try:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset CSV não encontrado: {RAW_DATA_PATH}")

        if not PROCESSED_DATA_PATH.exists():
            print("Gerando o Parquet processado a partir do CSV...")
            process_dataset_in_chunks(
                RAW_DATA_PATH,
                PROCESSED_DATA_PATH,
                chunksize=DEFAULT_CHUNKSIZE,
            )

        parquet_file = pq.ParquetFile(str(PROCESSED_DATA_PATH))
        sample = parquet_file.read_row_group(0).to_pandas().head(100)
        if parquet_file.metadata.num_rows == 0:
            raise ValueError("O arquivo Parquet não contém registros.")
        if "isFlaggedFraud" in sample:
            raise AssertionError("A coluna isFlaggedFraud não deve ser criada.")
        if "type_EMPRESTIMO" not in sample:
            raise AssertionError("As categorias do dataset não foram codificadas.")

        raw_sample = pd.read_csv(RAW_DATA_PATH, nrows=100)
        processed_sample = preprocess_dataset(raw_sample)
        if len(processed_sample) != len(raw_sample):
            raise AssertionError("O pré-processamento alterou a quantidade de linhas.")

        print(
            f"Parquet validado: {parquet_file.metadata.num_rows:,} registros "
            f"em {PROCESSED_DATA_PATH}".replace(",", ".")
        )
        return True
    except Exception as exc:
        print(f"Erro no preprocessamento: {exc}")
        return False


def teste_isolation_forest() -> bool:
    """Treina e persiste o Isolation Forest sobre uma amostra balanceada."""
    print("\n" + "=" * 70)
    print("TESTE 2: ISOLATION FOREST")
    print("=" * 70)
    try:
        from src.modelos.treinamento.treinar_modelos import (
            load_balanced_sample_from_parquet,
            save_model,
            train_isolation_forest,
        )
        from src.modelos.treinamento import treinar_modelos

        treinar_modelos.MODEL_DIR = _model_path("isolation_forest.pkl").parent
        sample = load_balanced_sample_from_parquet(
            PROCESSED_DATA_PATH, n_samples=2000
        )
        if sample.empty:
            raise ValueError("Não foi possível montar uma amostra de treinamento.")
        model = train_isolation_forest(sample)
        save_model(model, "isolation_forest.pkl")
        return True
    except Exception as exc:
        print(f"Erro no Isolation Forest: {exc}")
        return False


def teste_lstm() -> bool:
    """Treina e persiste o modelo LSTM."""
    print("\n" + "=" * 70)
    print("TESTE 3: LSTM")
    print("=" * 70)
    try:
        from src.modelos.arquitetura_lstm import salvar_lstm, treinar_lstm

        model = treinar_lstm(_load_sample_df(n=512), epochs=1)
        if model is None:
            raise RuntimeError("O treinamento LSTM não produziu um modelo.")
        salvar_lstm(model, filepath=str(_model_path("lstm_model.pt")))
        return True
    except Exception as exc:
        print(f"Erro no LSTM: {exc}")
        return False


def teste_gnn() -> bool:
    """Treina e persiste o modelo GNN."""
    print("\n" + "=" * 70)
    print("TESTE 4: GNN")
    print("=" * 70)
    try:
        from src.modelos.arquitetura_gnn import salvar_gnn, treinar_gnn

        model, dataset = treinar_gnn(_load_sample_df(n=256), epochs=1)
        if model is None or dataset is None:
            raise RuntimeError("O treinamento GNN não produziu modelo e grafo.")
        salvar_gnn(model, dataset, filepath=str(_model_path("gnn_model.pt")))
        return True
    except Exception as exc:
        print(f"Erro no GNN: {exc}")
        return False


def teste_inferencia() -> bool:
    """Executa os três escores de inferência com dados processados."""
    print("\n" + "=" * 70)
    print("TESTE 5: INFERÊNCIA")
    print("=" * 70)
    try:
        from src.ferramentas.inferencia_modelos import (
            gnn_identity_score,
            isolation_forest_score,
            lstm_sequence_score,
        )

        transaction = _load_sample_df(n=1).iloc[0].to_dict()
        scores = (
            isolation_forest_score(transaction),
            lstm_sequence_score([transaction]),
            gnn_identity_score(
                {
                    "origin": transaction["nameOrig"],
                    "destination": transaction["nameDest"],
                    "amount": transaction["amount"],
                }
            ),
        )
        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise AssertionError(f"Scores fora do intervalo [0, 1]: {scores}")
        print(f"Scores comportamental/temporal/identidade: {scores}")
        return True
    except Exception as exc:
        print(f"Erro na inferência: {exc}")
        return False


def teste_simulacao() -> bool:
    """Executa a orquestração determinística sobre uma transação do CSV."""
    print("\n" + "=" * 70)
    print("TESTE 6: SIMULAÇÃO COMPLETA")
    print("=" * 70)
    try:
        from src.orquestacao.fluxo_crewai import orchestrate_transaction

        transaction = _load_sample_df(n=1).iloc[0].to_dict()
        result = orchestrate_transaction(transaction, use_llm_judge=False)
        decision = result["decision"]
        if decision["decision"] not in {
            "Aprovada",
            "Revisão Humana necessária",
            "Recusada",
        }:
            raise AssertionError(f"Decisão inesperada: {decision['decision']}")
        if not 0.0 <= float(decision["score"]) <= 1.0:
            raise AssertionError(f"Score final inválido: {decision['score']}")
        print(f"Decisão: {decision['decision']} — score {decision['score']:.3f}")
        return True
    except Exception as exc:
        print(f"Erro na simulação: {exc}")
        return False


def main() -> None:
    """Executa as seis etapas integradas e retorna código de erro em falhas."""
    global _TEST_MODEL_DIR

    print("\n" + "=" * 70)
    print("SUITE INTEGRADA - DETECÇÃO DE FRAUDES".center(70))
    print("=" * 70)

    tests = [
        ("Preprocessamento", teste_preprocessamento),
        ("Isolation Forest", teste_isolation_forest),
        ("LSTM", teste_lstm),
        ("GNN", teste_gnn),
        ("Inferência", teste_inferencia),
        ("Simulação", teste_simulacao),
    ]
    with TemporaryDirectory(prefix="antifraude-test-models-") as model_directory:
        _TEST_MODEL_DIR = Path(model_directory)
        from src.ferramentas import inferencia_modelos
        from src.modelos.treinamento import treinar_modelos

        original_model_dir = inferencia_modelos.MODEL_DIR
        original_training_dir = treinar_modelos.MODEL_DIR
        inferencia_modelos.MODEL_DIR = _TEST_MODEL_DIR
        treinar_modelos.MODEL_DIR = _TEST_MODEL_DIR
        try:
            results = [(name, test()) for name, test in tests]
        finally:
            inferencia_modelos.MODEL_DIR = original_model_dir
            treinar_modelos.MODEL_DIR = original_training_dir
            _TEST_MODEL_DIR = None
    passed = sum(result for _, result in results)

    print("\n" + "=" * 70)
    print("SUMÁRIO DE TESTES")
    print("=" * 70)
    for name, result in results:
        print(f"{name:.<50} {'PASSOU' if result else 'FALHOU'}")
    print(f"\nResultado Final: {passed}/{len(results)} testes passaram")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
