"""Arquitetura e treinamento do modelo GNN para análise de grafos."""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    print("AVISO: PyTorch Geometric não disponível. GNN será placeholder.")


if TORCH_GEO_AVAILABLE:
    class GNNFraudDetector(nn.Module):
        """GNN baseado em Graph Attention Network para detecção de fraudes."""
        
        def __init__(self, in_channels=4, hidden_channels=32, out_channels=1, num_heads=4):
            super().__init__()
            
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
            
            self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, dropout=0.2)
            
            self.gcn = GCNConv(hidden_channels, hidden_channels)
            
            # Camadas fully connected
            self.fc1 = nn.Linear(hidden_channels, 16)
            self.fc2 = nn.Linear(16, out_channels)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x, edge_index):
            """
            Args:
                x: Features dos nós (num_nodes, in_channels)
                edge_index: Índices das arestas (2, num_edges)
            """
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            
            x = self.gat2(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            
            x = self.gcn(x, edge_index)
            x = F.elu(x)
            
            x = self.dropout(F.relu(self.fc1(x)))
            x = torch.sigmoid(self.fc2(x))
            
            return x


class TransactionGraphDataset:
    """Constrói grafo de transações para análise GNN."""
    
    def __init__(self, df, window_days=30):
        self.df = df
        self.window_days = window_days
        
        # Mapeia nomes de conta 
        self.account_to_idx = {}
        self.idx_to_account = {}
        
        # Constrói mapeamento
        unique_accounts = set(df['nameOrig'].unique()) | set(df['nameDest'].unique())
        for i, account in enumerate(sorted(unique_accounts)):
            self.account_to_idx[account] = i
            self.idx_to_account[i] = account
        
        self.num_nodes = len(self.account_to_idx)
    
    def build_graph(self, sample_df):

        node_features = self._compute_node_features(sample_df)
        
        # Arestas e pesos
        edge_list = []
        edge_weights = []
        
        for _, row in sample_df.iterrows():
            src = self.account_to_idx[row['nameOrig']]
            dst = self.account_to_idx[row['nameDest']]
            
            edge_list.append([src, dst])
            edge_weights.append(row['amount'])
        
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_weight = torch.FloatTensor(edge_weights)
        else:
            edge_index = torch.LongTensor(2, 0)
            edge_weight = torch.FloatTensor([])
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weight
        )
        
        return data
    
    def _compute_node_features(self, df):
        """Calcula features para cada nó (conta)."""
        features = []
        
        for node_idx in range(self.num_nodes):
            account = self.idx_to_account[node_idx]
            
            out_df = df[df['nameOrig'] == account]
            out_count = len(out_df)
            out_volume = out_df['amount'].sum() if len(out_df) > 0 else 0.0
            
            # Transações incoming
            in_df = df[df['nameDest'] == account]
            in_count = len(in_df)
            in_volume = in_df['amount'].sum() if len(in_df) > 0 else 0.0
            
            feature_vector = [
                float(out_count),
                float(in_count),
                float(out_volume),
                float(in_volume)
            ]
            features.append(feature_vector)
        
        return torch.FloatTensor(features)


def treinar_gnn(df_treino, epochs=20, batch_size=1):
    """Treina o modelo GNN."""
    if not TORCH_GEO_AVAILABLE:
        print(" PyTorch Geometric não disponível. Retornando None.")
        return None, None
    
    print("Treinando modelo GNN...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Constrói grafo
    dataset = TransactionGraphDataset(df_treino)
    graph_data = dataset.build_graph(df_treino)
    graph_data = graph_data.to(device)
    
    model = GNNFraudDetector(in_channels=4).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    node_fraud_labels = []
    for node_idx in range(dataset.num_nodes):
        account = dataset.idx_to_account[node_idx]
        
        out_df = df_treino[df_treino['nameOrig'] == account]
        fraud_rate = out_df['isFraud'].mean() if len(out_df) > 0 else 0.0
        
        node_fraud_labels.append(fraud_rate)
    
    y_nodes = torch.FloatTensor(node_fraud_labels).unsqueeze(1).to(device)
    
    # Treinamento
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out, y_nodes)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f" Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    print(" GNN treinado com sucesso!")
    return model, dataset


def salvar_gnn(model, dataset, filepath="src/modelos/modelos_salvos/gnn_model.pt"):
    """Salva modelo GNN e dataset."""
    if not TORCH_GEO_AVAILABLE or model is None:
        print("  Não é possível salvar modelo GNN sem PyTorch Geometric.")
        return
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), filepath)
    
    # Salva mapeamentos
    mapping_path = filepath.replace('.pt', '_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'account_to_idx': dataset.account_to_idx,
            'idx_to_account': dataset.idx_to_account
        }, f)
    
    print(f" Modelo GNN salvo em: {filepath}")


def carregar_gnn(filepath="src/modelos/modelos_salvos/gnn_model.pt"):
    """Carrega modelo GNN treinado."""
    if not TORCH_GEO_AVAILABLE:
        print("  PyTorch Geometric não disponível para carregar modelo.")
        return None, None
    
    if not Path(filepath).exists():
        print(f" Modelo não encontrado: {filepath}")
        return None, None
    
    model = GNNFraudDetector(in_channels=4)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()
    
    # Carrega mapeamentos
    mapping_path = filepath.replace('.pt', '_mapping.pkl')
    if Path(mapping_path).exists():
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
    else:
        mapping = {}
    
    return model, mapping


if not TORCH_GEO_AVAILABLE:
    def treinar_gnn(*args, **kwargs):
        print("AVISO: PyTorch Geometric necessário para treinar GNN. Usando placeholder.")
        return None, None
    
    def salvar_gnn(*args, **kwargs):
        pass
    
    def carregar_gnn(*args, **kwargs):
        return None, None
