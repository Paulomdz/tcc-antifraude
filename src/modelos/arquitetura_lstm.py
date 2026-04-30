"""Arquitetura e treinamento do modelo LSTM para análise temporal."""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# Suporte a PyTorch - versão simplificada que funciona com CPU
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("AVISO: PyTorch não disponível. LSTM será placeholder.")


if TORCH_AVAILABLE:
    class LSTMFraudDetector(nn.Module):
        """Modelo LSTM para detectar fraudes baseado em sequências temporais."""
        
        def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Camada LSTM bidirecional
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )
            
            # Camadas fully 
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.fc1 = nn.Linear(hidden_size * 2, 32)
            self.fc2 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            """
            Args:
                x: Tensor de shape (batch_size, seq_len, input_size)
            Returns:
                Tensor de shape (batch_size, 1) com scores entre 0 e 1
            """
            # LSTM 
            lstm_out, _ = self.lstm(x) 
            
            attention_weights = self.attention(lstm_out)
            attention_weights = self.sigmoid(attention_weights)
            
            weighted_output = lstm_out * attention_weights 
            context = weighted_output.sum(dim=1)
            
            # layers
            x = self.dropout(self.relu(self.fc1(context)))
            x = self.sigmoid(self.fc2(x))  # Score entre 0 e 1
            
            return x
    
    
    class TransactionSequenceDataset(Dataset):
        """Dataset que agrupa transações em sequências por janela de tempo global."""
        
        def __init__(self, df, seq_length=10, feature_cols=None):
            self.df = df.copy()
            self.seq_length = seq_length
            
            if feature_cols is None:
                feature_cols = [
                    'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'step_norm'
                ]
            self.feature_cols = feature_cols
            
            self.sequences = []
            self.labels = []
            
            # Usa uma janela deslizante global ordenada por tempo para gerar sequências.
            sorted_df = self.df.sort_values('step').reset_index(drop=True)
            for i in range(len(sorted_df) - self.seq_length):
                seq = sorted_df.iloc[i:i+self.seq_length][self.feature_cols].values
                label = sorted_df.iloc[i+self.seq_length]['isFraud']
                self.sequences.append(torch.tensor(seq, dtype=torch.float32))
                self.labels.append(torch.tensor([label], dtype=torch.float32))

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]


    def treinar_lstm(df_treino, epochs=20, batch_size=32, learning_rate=0.001):
        """Treina o modelo LSTM."""
        if not TORCH_AVAILABLE:
            print("  PyTorch não disponível. Retornando None.")
            return None
        
        print(" Treinando modelo LSTM...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {device}")
        
        dataset = TransactionSequenceDataset(df_treino, seq_length=10)
        
        if len(dataset) == 0:
            print(" Dataset vazio. Não há sequências suficientes para treino.")
            return None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = LSTMFraudDetector().to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Treinamento
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for sequences, labels in dataloader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            accuracy = 100 * correct / total if total > 0 else 0
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acurácia: {accuracy:.2f}%")
            
            scheduler.step(avg_loss)
        
        print("LSTM treinado com sucesso!")
        return model


def salvar_lstm(model, filepath="src/modelos/modelos_salvos/lstm_model.pt"):
    """Salva o modelo LSTM."""
    if not TORCH_AVAILABLE or model is None:
        print("Não é possível salvar modelo LSTM sem PyTorch.")
        return
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f" Modelo LSTM salvo em: {filepath}")


def carregar_lstm(filepath="src/modelos/modelos_salvos/lstm_model.pt"):
    """Carrega modelo LSTM treinado."""
    if not TORCH_AVAILABLE:
        print("PyTorch não disponível para carregar modelo.")
        return None
    
    if not Path(filepath).exists():
        print(f" Modelo não encontrado: {filepath}")
        return None
    
    model = LSTMFraudDetector()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()
    return model


if not TORCH_AVAILABLE:
    def treinar_lstm(*args, **kwargs):
        print("AVISO: PyTorch necessário para treinar LSTM. Usando placeholder.")
        return None
    
    def salvar_lstm(*args, **kwargs):
        pass
    
    def carregar_lstm(*args, **kwargs):
        return None
