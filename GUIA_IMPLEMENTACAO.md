# 📚 GUIA DE IMPLEMENTAÇÃO: LSTM, GNN e CrewAI Conversacional

## Status Atual do Projeto
✅ **Funcional:** Isolation Forest, Pré-processamento, Agentes básicos
🔄 **Para Implementar:** LSTM real, GNN real, CrewAI conversacional completo

---

## 1️⃣ IMPLEMENTAR LSTM (Long Short-Term Memory)

### O que é LSTM?
Rede neural recorrente especializada em capturar padrões temporais em sequências. No contexto de fraudes:
- **Entrada:** Histórico das últimas N transações de um cliente
- **Processo:** Analisa mudanças de padrão ao longo do tempo
- **Saída:** Score de anomalia baseado em desvios temporais

### Passo 1: Resolver Dependência PyTorch

O PyTorch oficial não suporta Python 3.13. Soluções:

**Opção A: Usar CPU-only versão nightly**
```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Opção B: Downgrade para Python 3.11 (Recomendado)**
```bash
# Criar novo venv com Python 3.11
python -m venv venv_py311
venv_py311\Scripts\activate
pip install torch
```

**Opção C: Usar TensorFlow/Keras (Alternativa)**
```bash
pip install tensorflow-cpu
# Keras já vem incluído
```

### Passo 2: Criar Modelo LSTM

Crie arquivo: `src/modelos/arquitetura_lstm.py`

```python
"""Arquitetura e treinamento do modelo LSTM para análise temporal."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

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
        
        # Camadas fully connected
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
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Mecanismo de atenção: foca nas transações mais relevantes
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = self.sigmoid(attention_weights)
        
        # Aplica atenção ao output
        weighted_output = lstm_out * attention_weights  # (batch, seq_len, hidden_size*2)
        context = weighted_output.sum(dim=1)  # (batch, hidden_size*2)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(context)))
        x = self.sigmoid(self.fc2(x))  # Score entre 0 e 1
        
        return x


class TransactionSequenceDataset(Dataset):
    """Dataset que agrupa transações em sequências por cliente."""
    
    def __init__(self, df, seq_length=10, feature_cols=None):
        self.df = df.copy()
        self.seq_length = seq_length
        
        if feature_cols is None:
            feature_cols = [
                'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'step_norm'
            ]
        self.feature_cols = feature_cols
        
        # Agrupa por cliente e cria sequências
        self.sequences = []
        self.labels = []
        
        for client, group in df.groupby('nameOrig'):
            group = group.sort_values('step').reset_index(drop=True)
            
            for i in range(len(group) - seq_length):
                seq = group.iloc[i:i+seq_length][feature_cols].values
                label = group.iloc[i+seq_length]['isFraud']
                
                self.sequences.append(torch.FloatTensor(seq))
                self.labels.append(torch.FloatTensor([label]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def treinar_lstm(df_treino, epochs=20, batch_size=32, learning_rate=0.001):
    """Treina o modelo LSTM."""
    print("🔧 Treinando modelo LSTM...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Dataset e DataLoader
    dataset = TransactionSequenceDataset(df_treino, seq_length=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo
    model = LSTMFraudDetector().to(device)
    
    # Loss e otimizador
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Treinamento
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estatísticas
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acurácia: {accuracy:.2f}%")
        
        scheduler.step(avg_loss)
    
    print("✅ LSTM treinado com sucesso!")
    return model


def salvar_lstm(model, filepath="src/modelos/modelos_salvos/lstm_model.pt"):
    """Salva o modelo LSTM."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"💾 Modelo LSTM salvo em: {filepath}")


def carregar_lstm(filepath="src/modelos/modelos_salvos/lstm_model.pt"):
    """Carrega modelo LSTM treinado."""
    model = LSTMFraudDetector()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()
    return model
```

### Passo 3: Integrar LSTM na Inferência

Editar `src/ferramentas/inferencia_modelos.py`:

```python
def lstm_sequence_score(sequence: list[Dict[str, Any]], model: Any = None) -> float:
    """Retorna score de anomalia temporal usando LSTM."""
    import torch
    
    if model is None:
        from src.modelos.arquitetura_lstm import carregar_lstm
        model = carregar_lstm()
    
    model.eval()
    
    # Converte sequência para tensor
    feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'step_norm'
    ]
    
    features = []
    for tx in sequence[-10:]:  # Pega últimas 10 transações
        feature_vector = [
            float(tx.get(col, 0.0)) for col in feature_cols
        ]
        features.append(feature_vector)
    
    # Padding se necessário
    while len(features) < 10:
        features.insert(0, [0.0] * len(feature_cols))
    
    features = torch.FloatTensor([features[-10:]])
    
    with torch.no_grad():
        output = model(features)
    
    return float(output[0, 0])
```

---

## 2️⃣ IMPLEMENTAR GNN (Graph Neural Network)

### O que é GNN?
Rede neural que analisa relações entre nós (contas) em um grafo. No contexto de fraudes:
- **Nós:** Contas (nameOrig, nameDest)
- **Arestas:** Transações entre contas
- **Objetivo:** Detectar padrões de rede fraudulenta

### Passo 1: Instalar Dependências

```bash
pip install torch-geometric networkx scikit-network
```

### Passo 2: Criar Modelo GNN

Crie arquivo: `src/modelos/arquitetura_gnn.py`

```python
"""Arquitetura e treinamento do modelo GNN para análise de grafos."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

class GNNFraudDetector(nn.Module):
    """GNN baseado em Graph Attention Network para detecção de fraudes."""
    
    def __init__(self, in_channels=4, hidden_channels=32, out_channels=1, num_heads=4):
        super().__init__()
        
        # Primeiro GAT com múltiplas cabeças de atenção
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        
        # Segundo GAT
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, dropout=0.2)
        
        # GCN para agregação global
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
        # GAT layers
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GCN layer
        x = self.gcn(x, edge_index)
        x = F.elu(x)
        
        # Fully connected
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        
        return x


class TransactionGraphDataset:
    """Constrói grafo de transações para análise GNN."""
    
    def __init__(self, df, window_days=30):
        self.df = df
        self.window_days = window_days
        
        # Mapeia nomes de conta para índices de nó
        self.account_to_idx = {}
        self.idx_to_account = {}
        
        # Constrói mapeamento
        unique_accounts = set(df['nameOrig'].unique()) | set(df['nameDest'].unique())
        for i, account in enumerate(sorted(unique_accounts)):
            self.account_to_idx[account] = i
            self.idx_to_account[i] = account
        
        self.num_nodes = len(self.account_to_idx)
    
    def build_graph(self, sample_df):
        """Constrói grafo PyTorch Geometric a partir das transações."""
        # Features dos nós: features agregadas por conta
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
        
        # Cria dataset PyG
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
            
            # Transações outgoing
            out_df = df[df['nameOrig'] == account]
            out_count = len(out_df)
            out_volume = out_df['amount'].sum() if len(out_df) > 0 else 0.0
            
            # Transações incoming
            in_df = df[df['nameDest'] == account]
            in_count = len(in_df)
            in_volume = in_df['amount'].sum() if len(in_df) > 0 else 0.0
            
            # Feature vector por conta
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
    print("🔧 Treinando modelo GNN...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Constrói grafo
    dataset = TransactionGraphDataset(df_treino)
    graph_data = dataset.build_graph(df_treino)
    graph_data = graph_data.to(device)
    
    # Modelo
    model = GNNFraudDetector(in_channels=4).to(device)
    
    # Loss e otimizador
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Labels dos nós (agregado de fraudes por conta)
    node_fraud_labels = []
    for node_idx in range(dataset.num_nodes):
        account = dataset.idx_to_account[node_idx]
        fraud_rate = df_treino[df_treino['nameOrig'] == account]['isFraud'].mean()
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
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    print("✅ GNN treinado com sucesso!")
    return model, dataset


def salvar_gnn(model, dataset, filepath="src/modelos/modelos_salvos/gnn_model.pt"):
    """Salva modelo GNN e dataset."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    import pickle
    torch.save(model.state_dict(), filepath)
    
    # Salva mapeamentos
    mapping_path = filepath.replace('.pt', '_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'account_to_idx': dataset.account_to_idx,
            'idx_to_account': dataset.idx_to_account
        }, f)
    
    print(f"💾 Modelo GNN salvo em: {filepath}")


def carregar_gnn(filepath="src/modelos/modelos_salvos/gnn_model.pt"):
    """Carrega modelo GNN treinado."""
    import pickle
    
    model = GNNFraudDetector(in_channels=4)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()
    
    # Carrega mapeamentos
    mapping_path = filepath.replace('.pt', '_mapping.pkl')
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    return model, mapping
```

### Passo 3: Integrar GNN na Inferência

Editar `src/ferramentas/inferencia_modelos.py`:

```python
def gnn_identity_score(transaction: Dict[str, Any], model: Any = None) -> float:
    """Retorna score de anomalia de identidade usando GNN."""
    import torch
    
    if model is None:
        from src.modelos.arquitetura_gnn import carregar_gnn
        model, mapping = carregar_gnn()
    else:
        mapping = model[1] if isinstance(model, tuple) else {}
        model = model[0] if isinstance(model, tuple) else model
    
    model.eval()
    
    # Pega contas
    origin = transaction.get('nameOrig')
    destination = transaction.get('nameDest')
    
    account_to_idx = mapping.get('account_to_idx', {})
    
    if origin not in account_to_idx or destination not in account_to_idx:
        return 0.5  # Score neutro para contas desconhecidas
    
    src_idx = account_to_idx[origin]
    dst_idx = account_to_idx[destination]
    
    # Cria edge simples
    edge_index = torch.LongTensor([[src_idx], [dst_idx]])
    
    # Features simples por conta (placeholder)
    x = torch.FloatTensor([[1.0, 0.0, 100.0, 0.0]])  # features por nó
    
    with torch.no_grad():
        output = model(x, edge_index)
    
    return float(output[0, 0])
```

---

## 3️⃣ INTEGRAR CrewAI COM AGENTES CONVERSACIONAIS

### O que precisa mudar?

Atualmente: Sistema local simplificado
Objetivo: Agentes reais conversando com Gemini

### Passo 1: Configurar CrewAI com Ferramentas

Editar `src/orquestração/definicoes_agentes.py`:

```python
"""Definições de agentes CrewAI com ferramentas e integração Gemini."""

import os
import json
from crewai import Agent, Tool
from crewai_tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ============== DEFINIÇÃO DE TOOLS ==============

@tool("Análise Comportamental")
def analisar_comportamento(transaction_json: str) -> str:
    """
    Analisa o padrão comportamental da transação.
    Retorna score e explicação de anomalia.
    """
    from src.ferramentas.inferencia_modelos import isolation_forest_score
    
    transaction = json.loads(transaction_json)
    score = isolation_forest_score(transaction)
    
    return json.dumps({
        "score": score,
        "tipo": "comportamental",
        "descricao": f"Score de anomalia: {score:.3f}",
        "risco": "ALTO" if score > 0.8 else "MÉDIO" if score > 0.5 else "BAIXO"
    })

@tool("Análise Temporal")
def analisar_temporal(transaction_json: str) -> str:
    """
    Analisa padrões temporais na sequência de transações.
    Detecta desvios do histórico do cliente.
    """
    from src.ferramentas.inferencia_modelos import lstm_sequence_score
    
    transaction = json.loads(transaction_json)
    
    # Simula sequência (em produção, viria do banco de dados)
    sequence = [transaction]
    score = lstm_sequence_score(sequence)
    
    return json.dumps({
        "score": score,
        "tipo": "temporal",
        "descricao": f"Score temporal: {score:.3f}",
        "analise": "Padrão consistente com histórico" if score < 0.5 else "Desvio detectado"
    })

@tool("Análise de Identidade")
def analisar_identidade(transaction_json: str) -> str:
    """
    Analisa conexões entre contas.
    Detecta redes suspeitas usando grafo de transações.
    """
    from src.ferramentas.inferencia_modelos import gnn_identity_score
    
    transaction = json.loads(transaction_json)
    score = gnn_identity_score(transaction)
    
    origin = transaction.get("nameOrig", "Desconhecido")
    dest = transaction.get("nameDest", "Desconhecido")
    
    return json.dumps({
        "score": score,
        "tipo": "identidade",
        "conexão": f"{origin} -> {dest}",
        "risco_conexão": "SUSPEITA" if score > 0.7 else "NORMAL"
    })

@tool("Buscar Histórico de Cliente")
def buscar_historico(client_id: str) -> str:
    """Busca transações anteriores do cliente para contexto."""
    # Em produção, isso viria de um banco de dados
    return json.dumps({
        "cliente": client_id,
        "transacoes_mes": 45,
        "fraudes_detectadas": 2,
        "valor_total_mes": 150000.00,
        "padroes": ["TRANSFER", "PAYMENT"]
    })

# ============== DEFINIÇÃO DE AGENTES ==============

class FraudSimulatorAgents:
    """Agentes CrewAI para análise multiagente de fraudes."""
    
    def especialista_comportamental(self):
        return Agent(
            role="Analista Comportamental (ML)",
            goal="Identificar anomalias estatísticas na transação atual.",
            backstory=(
                "Você é um especialista em Machine Learning com experiência em detecção "
                "de anomalias em dados financeiros. Sua função é interpretar scores do "
                "Isolation Forest e explicar por que uma transação parece suspeita."
            ),
            tools=[analisar_comportamento, buscar_historico],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True  # Mantém contexto de conversas
        )
    
    def especialista_temporal(self):
        return Agent(
            role="Analista de Séries Temporais (LSTM)",
            goal="Detectar desvios no padrão sequencial de transações.",
            backstory=(
                "Você é especialista em séries temporais e redes neurais recorrentes. "
                "Sua expertise é entender se uma transação faz sentido dado o histórico "
                "recente do cliente, analisando mudanças de comportamento ao longo do tempo."
            ),
            tools=[analisar_temporal, buscar_historico],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True
        )
    
    def especialista_identidade(self):
        return Agent(
            role="Analista de Redes (GNN)",
            goal="Verificar conexões suspeitas entre contas.",
            backstory=(
                "Você é um especialista em análise de grafos e redes de fraude. "
                "Você identifica padrões em como as contas se conectam, detectando "
                "redes organizadas de fraude e relacionamentos anormais."
            ),
            tools=[analisar_identidade, buscar_historico],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True
        )
    
    def juiz_final(self):
        return Agent(
            role="Juiz de Risco Financeiro",
            goal="Consolidar análises e dar decisão final balanceada.",
            backstory=(
                "Você é o autoridade máxima em decisões de risco. Com experiência "
                "em conformidade financeira e gestão de risco, você sintetiza os "
                "pareceres dos especialistas em uma decisão que equilibra segurança "
                "do sistema com experiência do cliente. Suas decisões são finais."
            ),
            tools=[analisar_comportamento, analisar_temporal, analisar_identidade],
            llm=llm,
            verbose=True,
            allow_delegation=True  # Pode questionar especialistas
        )
```

### Passo 2: Criar Workflow com Tarefas

Crie arquivo: `src/orquestração/tarefas_crewai.py`

```python
"""Definição de tarefas para o workflow CrewAI."""

from crewai import Task
from typing import Dict, Any
import json

def criar_tarefas_analise(transaction: Dict[str, Any], agents):
    """Cria tarefas de análise para uma transação."""
    
    transaction_str = json.dumps(transaction)
    
    # Tarefa 1: Análise Comportamental
    tarefa_comportamento = Task(
        description=(
            f"Analise a seguinte transação para anomalias comportamentais:\n"
            f"{transaction_str}\n\n"
            f"Use a ferramenta de análise comportamental e explique o score obtido. "
            f"Justifique sua avaliação em português."
        ),
        agent=agents.especialista_comportamental(),
        expected_output="Relatório detalhado com score e justificativa de anomalia comportamental"
    )
    
    # Tarefa 2: Análise Temporal
    tarefa_temporal = Task(
        description=(
            f"Analise os padrões temporais da transação:\n"
            f"{transaction_str}\n\n"
            f"Determine se é consistente com histórico do cliente. "
            f"Use a análise temporal e histórico. Responda em português."
        ),
        agent=agents.especialista_temporal(),
        expected_output="Análise de padrões temporais com score de desvio"
    )
    
    # Tarefa 3: Análise de Identidade
    tarefa_identidade = Task(
        description=(
            f"Analise a conexão entre as contas na transação:\n"
            f"{transaction_str}\n\n"
            f"Identifique se há padrões suspeitos de rede. Use grafo de transações. "
            f"Responda em português."
        ),
        agent=agents.especialista_identidade(),
        expected_output="Análise de conexões de rede com risco de fraude em grupo"
    )
    
    # Tarefa 4: Decisão Final
    tarefa_juiz = Task(
        description=(
            f"Com base nos relatórios dos especialistas, tome a decisão final sobre:\n"
            f"{transaction_str}\n\n"
            f"REGRAS DE DECISÃO:\n"
            f"- Se score médio > 0.8: BLOQUEADO\n"
            f"- Se score médio entre 0.5-0.8: REVISÃO MANUAL\n"
            f"- Se score médio < 0.5: APROVADO\n\n"
            f"Justifique detalhadamente em português. Formato final:\n"
            f"DECISÃO: [APROVADO/REVISÃO MANUAL/BLOQUEADO]\n"
            f"JUSTIFICATIVA: ..."
        ),
        agent=agents.juiz_final(),
        expected_output="Decisão final com justificativa detalhada"
    )
    
    return [tarefa_comportamento, tarefa_temporal, tarefa_identidade, tarefa_juiz]


def criar_crew_analise(transaction: Dict[str, Any], agents):
    """Cria e retorna um Crew com as tarefas."""
    from crewai import Crew
    
    tarefas = criar_tarefas_analise(transaction, agents)
    
    crew = Crew(
        agents=[
            agents.especialista_comportamental(),
            agents.especialista_temporal(),
            agents.especialista_identidade(),
            agents.juiz_final()
        ],
        tasks=tarefas,
        verbose=True,
        memory=True  # Ativa memória compartilhada entre agentes
    )
    
    return crew
```

### Passo 3: Novo Workflow de Orquestração

Editar `src/orquestração/fluxo_crewai.py`:

```python
"""Fluxo de orquestração com CrewAI conversacional."""

from typing import Dict, Any
import json
from src.orquestração.definicoes_agentes import FraudSimulatorAgents
from src.orquestração.tarefas_crewai import criar_crew_analise

def orchestrate_transaction_com_crewai(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquestra análise completa de transação usando CrewAI.
    Agentes conversam entre si e chegam a uma decisão.
    """
    print(f"\n🔄 Iniciando análise multiagente para transação...")
    print(f"   Transação: {transaction.get('type')} - ${transaction.get('amount'):.2f}")
    
    # Cria instância de agentes
    agents = FraudSimulatorAgents()
    
    # Cria crew com tarefas
    crew = criar_crew_analise(transaction, agents)
    
    # Executa workflow
    try:
        resultado = crew.kickoff()
        
        # Parse resultado
        return {
            "transaction": transaction,
            "resultado_crew": str(resultado),
            "status": "sucesso"
        }
    except Exception as e:
        print(f"❌ Erro no CrewAI: {str(e)}")
        return {
            "transaction": transaction,
            "erro": str(e),
            "status": "erro"
        }
```

### Passo 4: Atualizar run_simulation.py

```python
"""Script principal atualizado com CrewAI conversacional."""

from src.orquestração.fluxo_crewai import orchestrate_transaction_com_crewai
import pandas as pd
from src.preprocessamento.carregar_paysim import PROCESSED_DATA_PATH

def main():
    print("🚀 Simulador de Detecção de Fraudes com CrewAI Conversacional")
    print("=" * 60)
    
    # Carrega dados
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # Pega 3 transações de exemplo
    sample = df.sample(n=3, random_state=42)
    
    resultados = []
    for i, (_, transaction) in enumerate(sample.iterrows(), 1):
        print(f"\n{'='*60}")
        print(f"Transação {i}/3")
        print(f"{'='*60}")
        
        tx_dict = transaction.to_dict()
        resultado = orchestrate_transaction_com_crewai(tx_dict)
        resultados.append(resultado)
        
        print(f"\n✅ Análise concluída!")
        print(f"Resultado: {resultado['resultado_crew'][:200]}...")
    
    print(f"\n{'='*60}")
    print("✅ Simulação completa!")

if __name__ == "__main__":
    main()
```

---

## 4️⃣ CHECKLIST DE IMPLEMENTAÇÃO

### Fase 1: Setup Inicial ⏳
- [ ] Resolver PyTorch/TensorFlow
- [ ] Instalar torch-geometric
- [ ] Criar diretórios de modelos

### Fase 2: LSTM
- [ ] Criar `src/modelos/arquitetura_lstm.py`
- [ ] Treinar LSTM com dados balanceados
- [ ] Salvar modelo
- [ ] Integrar em `inferencia_modelos.py`

### Fase 3: GNN
- [ ] Criar `src/modelos/arquitetura_gnn.py`
- [ ] Treinar GNN com grafo de transações
- [ ] Salvar modelo e mapeamentos
- [ ] Integrar em `inferencia_modelos.py`

### Fase 4: CrewAI Conversacional
- [ ] Atualizar `definicoes_agentes.py` com ferramentas
- [ ] Criar `tarefas_crewai.py`
- [ ] Atualizar `fluxo_crewai.py`
- [ ] Testar com CrewAI localmente

### Fase 5: Aprimoramentos Finais
- [ ] Adicionar logging detalhado
- [ ] Criar dashboard Streamlit
- [ ] Testes unitários
- [ ] Documentação completa

---

## 5️⃣ COMANDOS RÁPIDOS

```bash
# Treinar LSTM
python -c "from src.modelos.arquitetura_lstm import treinar_lstm, salvar_lstm; import pandas as pd; df = pd.read_parquet('data/paysim_processed.parquet'); model = treinar_lstm(df, epochs=20); salvar_lstm(model)"

# Treinar GNN
python -c "from src.modelos.arquitetura_gnn import treinar_gnn, salvar_gnn; import pandas as pd; df = pd.read_parquet('data/paysim_processed.parquet'); model, dataset = treinar_gnn(df, epochs=20); salvar_gnn(model, dataset)"

# Testar CrewAI
python run_simulation.py
```

---

## 📚 Referências

- PyTorch LSTM: https://pytorch.org/tutorials/
- PyG (PyTorch Geometric): https://pytorch-geometric.readthedocs.io/
- CrewAI Tools: https://docs.crewai.com/
- Google Gemini: https://ai.google.dev/

---

**🎯 Objetivo Final:** Sistema funcional com agentes IA conversando em português, analisando fraudes com LSTM, GNN e Isolation Forest integrados via CrewAI e Gemini 1.5 Flash.