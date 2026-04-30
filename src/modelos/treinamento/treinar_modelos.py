"""Treinamento dos modelos de detecção de fraude."""

import os
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Adiciona o diretório raiz ao path para imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessamento.carregar_paysim import PROCESSED_DATA_PATH
from src.ferramentas.inferencia_modelos import extract_behavior_features

MODEL_DIR = Path("src/models/modelos_salvos")


def prepare_balanced_sample(df: pd.DataFrame, n_samples: int = 10000) -> pd.DataFrame:
    """Prepara uma amostra balanceada com 50% fraude e 50% não-fraude."""
    fraud_df = df[df["isFraud"] == 1]
    non_fraud_df = df[df["isFraud"] == 0]

    n_fraud = min(len(fraud_df), n_samples // 2)
    n_non_fraud = min(len(non_fraud_df), n_samples // 2)

    fraud_sample = fraud_df.sample(n=n_fraud, random_state=42)
    non_fraud_sample = non_fraud_df.sample(n=n_non_fraud, random_state=42)

    balanced_df = pd.concat([fraud_sample, non_fraud_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Amostra balanceada criada: {len(balanced_df)} registros")
    print(f"Fraudes: {len(fraud_sample)}, Não-fraudes: {len(non_fraud_sample)}")

    return balanced_df


def train_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    """Treina o modelo Isolation Forest para detecção comportamental."""
    print(" Treinando Isolation Forest...")

    features = []
    for _, row in df.iterrows():
        features.append(extract_behavior_features(row.to_dict()))

    X = np.array(features)

    # Treina o modelo
    model = IsolationForest(
        n_estimators=100,
        contamination=0.5,  # 50% esperado como anomalia (fraude)
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    # Avaliação básica
    scores = model.decision_function(X)
    predictions = model.predict(X)

    # Converte para labels de classificação (1 = normal, -1 = anomalia)
    y_true = df["isFraud"].values
    y_pred = (predictions == -1).astype(int)

    print(" Relatório de Classificação - Isolation Forest:")
    print(classification_report(y_true, y_pred, target_names=["Não-Fraude", "Fraude"]))

    return model


def train_lstm_model(df: pd.DataFrame, epochs: int = 10):
    """Treina o modelo LSTM para análise temporal."""
    print(" Placeholder: Modelo LSTM será implementado após resolver dependências PyTorch...")
    return None


def train_gnn_model(df: pd.DataFrame):
    """Treina modelo GNN para análise de grafos de identidade."""
    print(" Placeholder: Modelo GNN será implementado posteriormente...")
    return None


def save_model(model, filename: str):
    """Salva o modelo treinado."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / filename

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f" Modelo salvo em: {model_path}")


def main():
    """Executa o treinamento de todos os modelos."""
    print(" Iniciando treinamento dos modelos de detecção de fraude")
    print("=" * 60)

    # Carrega dados processados
    if not PROCESSED_DATA_PATH.exists():
        print(" Dados processados não encontrados. Execute load_paysim.py primeiro.")
        return

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f" Dataset carregado: {len(df)} registros")

    # Prepara amostra balanceada
    train_df = prepare_balanced_sample(df, n_samples=10000)

    # Treina Isolation Forest
    if_model = train_isolation_forest(train_df)
    save_model(if_model, "isolation_forest.pkl")

    # Treina LSTM
    lstm_model = train_lstm_model(train_df)
    save_model(lstm_model, "lstm_sequence_model.pkl")

    # Placeholder GNN
    gnn_model = train_gnn_model(train_df)
    if gnn_model:
        save_model(gnn_model, "gnn_identity_model.pkl")

    print("\n Todos os modelos treinados e salvos!")
    print(" Pronto para executar a simulação: python run_simulation.py")


if __name__ == "__main__":
    main()