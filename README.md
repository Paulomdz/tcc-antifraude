# Simulador de Detecção de Fraudes Financeiras com IA Multiagente

Sistema avançado de detecção de fraudes financeiras usando múltiplos modelos de ML (Isolation Forest, LSTM, GNN) orquestrados por agentes IA conversacionais via CrewAI e Google Gemini 1.5 Flash.

## 🎯 Visão Geral

Um sistema que simula transações bancárias e detecta fraudes analisando:
- **Padrões Comportamentais**: Anomalias em valores e saldos (Isolation Forest)
- **Sequências Temporais**: Desvios no histórico de transações (LSTM)
- **Redes de Contas**: Conexões suspeitas entre contas (GNN)

Tudo integrado via agentes IA que conversam entre si usando Google Gemini 1.5 Flash.

## 📁 Estrutura do Projeto 

```
.
├── data/
│   ├── PS_20174392719_1491204439457_log.csv.zip  # PaySim (bruto)
│   └── paysim_processed.parquet                   # Processado
│
├── src/
│   ├── preprocessamento/              # Carregamento e limpeza
│   │   ├── carregar_paysim.py
│   │   └── __init__.py
│   │
│   ├── modelos/                       # Modelos de ML
│   │   ├── treinamento/
│   │   │   ├── treinar_modelos.py
│   │   │   └── __init__.py
│   │   ├── arquitetura_lstm.py
│   │   ├── arquitetura_gnn.py
│   │   ├── modelos_salvos/
│   │   └── __init__.py
│   │
│   ├── ferramentas/                   # Inferência
│   │   ├── inferencia_modelos.py
│   │   ├── ferramentas_crewai.py
│   │   └── __init__.py
│   │
│   └── orquestração/                  # CrewAI
│       ├── definicoes_agentes.py
│       ├── fluxo_crewai.py
│       ├── tarefas_crewai.py
│       └── __init__.py
│
├── .env                               # Chave API Gemini
├── run_simulation.py                  # Script principal
├── teste_sistema_completo.py         # Suite de testes
├── GUIA_IMPLEMENTACAO.md             # Guia detalhado
├── requirements.txt
└── README.md
```

## ⚙️ Instalação

### 1. Setup Básico
```bash
cd c:\Users\paulo\Desktop\Uni\TCC
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preparar Dados
```bash
# Descompacte PS_20174392719_1491204439457_log.csv.zip em data/
```

### 3. Configuração (Opcional)
Arquivo `.env` já contém chave Gemini. Para alterar:
```
GOOGLE_API_KEY=sua_chave_aqui
```

## 🚀 Como Usar

### ⚡ Opção 1: Suite Completa de Testes (Recomendado)
```bash
python teste_sistema_completo.py
```
Executa todos os testes e mostra o status de cada componente.

### 📊 Opção 2: Componentes Individuais

**Pré-processar:**
```bash
python src/preprocessamento/carregar_paysim.py
```

**Treinar Isolation Forest:**
```bash
python -c "
from src.modelos.treinamento.treinar_modelos import *
import pandas as pd
df = pd.read_parquet('data/paysim_processed.parquet')
modelo = train_isolation_forest(prepare_balanced_sample(df, 5000))
save_model(modelo, 'isolation_forest.pkl')
"
```

**Treinar LSTM (requer PyTorch):**
```bash
python -c "
from src.modelos.arquitetura_lstm import treinar_lstm, salvar_lstm
import pandas as pd
df = pd.read_parquet('data/paysim_processed.parquet')
model = treinar_lstm(df.sample(2000), epochs=20)
salvar_lstm(model) if model else None
"
```

**Treinar GNN (requer PyTorch Geometric):**
```bash
python -c "
from src.modelos.arquitetura_gnn import treinar_gnn, salvar_gnn
import pandas as pd
df = pd.read_parquet('data/paysim_processed.parquet')
model, ds = treinar_gnn(df.sample(2000), epochs=20)
salvar_gnn(model, ds) if model else None
"
```

**Executar Simulação:**
```bash
python run_simulation.py
```

## 📊 Status de Implementação

### [OK] Totalmente Funcional
- ✓ Pré-processamento PaySim
- ✓ Isolation Forest (58% acurácia)
- ✓ Sistema de inferência modular
- ✓ Tradução 100% em português
- ✓ Estrutura em português

### [PRONTO] Código Pronto (Com Fallbacks)
- [PRONTO] LSTM (Código completo, PyTorch opcional)
- [PRONTO] GNN (Código completo, PyTorch Geometric opcional)
- ⚠️ CrewAI (Estrutura pronta para ativar)

### 🔮 Próximas Fases
- [ ] Ativar CrewAI conversacional com agentes reais
- [ ] Dashboard Streamlit
- [ ] API REST em produção
- [ ] Testes unitários

## 🧠 Modelos de ML

| Modelo | Status | Entrada | Saída | Acurácia |
|--------|--------|---------|-------|----------|
| **Isolation Forest** | ✅ Ativo | Transação | Score 0-1 | 58% |
| **LSTM** | ⚠️ Pronto | Sequência | Score 0-1 | - |
| **GNN** | ⚠️ Pronto | Grafo | Score 0-1 | - |

## 📖 Documentação Completa

Veja `GUIA_IMPLEMENTACAO.md` para:
- Passo-a-passo LSTM e GNN
- Como integrar CrewAI conversacional
- Resolvendo dependências (PyTorch)
- Checklist de implementação
- Código completo de exemplo

## 🔧 Troubleshooting

**PyTorch não instala:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Dados não encontrados:**
- Descompacte: `data/PS_20174392719_1491204439457_log.csv.zip`

**CrewAI com erro:**
```bash
pip uninstall crewai -y && pip install crewai langchain-google-genai
```

**Verificar ambiente:**
```bash
python -c "import torch; print('✓ PyTorch'); import crewai; print('✓ CrewAI')"
```

## 📝 Exemplo de Uso

```python
from src.orquestração.fluxo_crewai import orchestrate_transaction
import pandas as pd

df = pd.read_parquet('data/paysim_processed.parquet')
transacao = df.iloc[0].to_dict()

resultado = orchestrate_transaction(transacao)

print(f"Decisão: {resultado['decision']['label']}")
print(f"Score: {resultado['decision']['fraud_score']:.3f}")
```

## 📚 Arquivos Principais

- **run_simulation.py** - Simulação completa
- **teste_sistema_completo.py** - Suite de testes
- **GUIA_IMPLEMENTACAO.md** - Documentação detalhada
- **src/modelos/arquitetura_lstm.py** - LSTM com atenção
- **src/modelos/arquitetura_gnn.py** - GNN com GAT+GCN
- **src/ferramentas/inferencia_modelos.py** - Carregamento de modelos

## 🎓 Referências

- PyTorch: https://pytorch.org/tutorials/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- CrewAI: https://docs.crewai.com/
- Google Gemini: https://ai.google.dev/
- Isolation Forest: https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest


---
