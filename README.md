# Simulador de Detecção de Fraudes Financeiras com IA Multiagente

Sistema de detecção de fraudes financeiras usando múltiplos modelos de ML (Isolation Forest, LSTM, GNN) orquestrados por agentes IA conversacionais via CrewAI e Google Gemini 1.5 Flash — com API REST, dashboard interativo e suíte de testes.

## 🎯 Visão Geral

O sistema analisa transações bancárias em três dimensões simultâneas:

- **Comportamental** — anomalias em valores e saldos (Isolation Forest)
- **Temporal** — desvios no histórico de transações (LSTM)
- **Identidade** — conexões suspeitas entre contas (GNN)

Um agente Juiz (Google Gemini 1.5 Flash) consolida os três scores e emite a decisão final: **APROVADO**, **REVISÃO** ou **BLOQUEADO**.

---

## 📁 Estrutura do Projeto

```text
tcc-antifraude/
├── data/
│   ├── PS_20174392719_1491204439457_log.csv   # PaySim bruto (1.3 GB)
│   └── paysim_dados_processados.parquet        # Processado (369 MB)
│
├── src/
│   ├── preprocessamento/
│   │   ├── carregar_paysim.py                  # Leitura e feature engineering
│   │   └── __init__.py
│   │
│   ├── modelos/
│   │   ├── arquitetura_lstm.py                 # LSTM bidirecional com atenção
│   │   ├── arquitetura_gnn.py                  # GNN com GAT + GCN
│   │   ├── treinamento/
│   │   │   └── treinar_modelos.py
│   │   ├── modelos_salvos/
│   │   │   └── isolation_forest.pkl            # Modelo treinado
│   │   └── __init__.py
│   │
│   ├── ferramentas/
│   │   ├── inferencia_modelos.py               # Carregamento e scoring
│   │   ├── ferramentas_crewai.py               # Tools dos agentes
│   │   └── __init__.py
│   │
│   ├── orquestração/
│   │   ├── definicoes_agentes.py               # Configuração dos agentes CrewAI
│   │   ├── fluxo_crewai.py                     # Pipeline de orquestração
│   │   └── __init__.py
│   │
│   ├── api/
│   │   ├── main.py                             # API REST (FastAPI)
│   │   └── __init__.py
│   │
│   └── dashboard/
│       ├── app.py                              # Dashboard (Streamlit)
│       └── __init__.py
│
├── tests/
│   ├── conftest.py                             # Fixtures compartilhadas
│   ├── test_preprocessamento.py               # 18 testes
│   ├── test_modelos.py                         # 20 testes
│   ├── test_api.py                             # 16 testes
│   └── test_dashboard.py                       # 19 testes
│
├── .env                                        # Chave API Gemini
├── run_simulation.py                           # Simulação via terminal
├── teste_sistema_completo.py                   # Suite legada
├── pytest.ini                                  # Configuração pytest (cov ≥ 80%)
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

```bash
# 1. Clonar e entrar no projeto
cd tcc-antifraude

# 2. Criar ambiente virtual
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar chave Gemini
# Edite .env e preencha:
# GOOGLE_API_KEY=sua_chave_aqui
```

---

## 🚀 Como Usar

### API REST

```bash
uvicorn src.api.main:app --reload
```

Acesse a documentação interativa em <http://localhost:8000/docs>

**Endpoints:**

| Método | Rota                 | Descrição                               |
|--------|----------------------|-----------------------------------------|
| GET    | `/`                  | Health check                            |
| POST   | `/analise_transacao` | Analisa uma transação e retorna decisão |

**Exemplo de requisição:**

```bash
curl -X POST http://localhost:8000/analise_transacao \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 50000.0,
    "type": "TRANSFER",
    "nameOrig": "C1234567",
    "oldbalanceOrg": 60000.0,
    "newbalanceOrig": 10000.0,
    "nameDest": "C9999999",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 50000.0,
    "step_norm": 0.5
  }'
```

**Resposta:**

```json
{
  "decisao": "BLOQUEADO",
  "score_final": 0.91,
  "justificativa": "Decisão: Recusada. O agente comportamental indicou 0.910...",
  "scores": {
    "comportamental": 0.91,
    "temporal": 0.72,
    "identidade": 0.65
  },
  "agentes": {
    "comportamental": { "score": 0.91, "label": "behavior", "explicacao": "..." },
    "temporal":       { "score": 0.72, "label": "temporal", "explicacao": "..." },
    "identidade":     { "score": 0.65, "label": "identity", "explicacao": "..." }
  }
}
```

---

### Dashboard Streamlit

```bash
streamlit run src/dashboard/app.py
```

Acesse em <http://localhost:8501>

| Aba                | Funcionalidade                                                  |
|--------------------|-----------------------------------------------------------------|
| Análise Individual | Formulário com todos os campos, exibe scores com barras visuais |
| Análise em Lote    | Upload de CSV/Parquet, analisa até 50 transações, tabela        |
| Estatísticas       | Pie chart de decisões, histograma de scores, boxplot por agente |

---

### Testes Unitários

```bash
pytest
```

Executa 73 testes com relatório de cobertura. O build falha se a cobertura for inferior a 80%.

```bash
# Apenas um módulo
pytest tests/test_api.py -v

# Com relatório HTML
pytest --cov-report=html
```

---

### Simulação via Terminal

```bash
python run_simulation.py
```

---

## 📊 Status de Implementação

| Componente       | Status    | Detalhe                              |
|------------------|-----------|--------------------------------------|
| Preprocessamento | Completo  | 6.362.620 transações, 22 features    |
| Isolation Forest | Treinado  | 58% acurácia, salvo em `.pkl`        |
| LSTM             | Standby   | Código completo, fallback heurístico |
| GNN              | Standby   | Código completo, fallback heurístico |
| Agentes CrewAI   | Funcional | 4 agentes, Gemini 1.5 Flash          |
| API REST         | Completo  | FastAPI, 2 endpoints, Pydantic       |
| Dashboard        | Completo  | Streamlit, 3 abas, Plotly            |
| Testes Unitários | Completo  | 73 testes, cobertura >= 80%          |

---

## 🧠 Modelos de ML

| Modelo           | Status     | Entrada   | Saída     | Acurácia |
|------------------|------------|-----------|-----------|----------|
| Isolation Forest | ✅ Ativo   | Transação | Score 0–1 | 58%      |
| LSTM             | 🔄 Standby | Sequência | Score 0–1 | —        |
| GNN              | 🔄 Standby | Grafo     | Score 0–1 | —        |

---

## 🤖 Fluxo dos Agentes

```text
Transação recebida
        ↓
┌───────────────────────────────────────────────────┐
│              Pipeline de Orquestração             │
│                                                   │
│  [Isolation Forest] → score comportamental        │
│  [LSTM]             → score temporal              │
│  [GNN]              → score de identidade         │
│                                                   │
│  [Juiz — Gemini 1.5 Flash]                        │
│    └─ max_score ≥ 0.8  → BLOQUEADO               │
│    └─ max_score ≥ 0.5  → REVISÃO                 │
│    └─ max_score < 0.5  → APROVADO                │
└───────────────────────────────────────────────────┘
        ↓
Decisão + justificativa em linguagem natural
```

---

## 🔧 Troubleshooting

**PyTorch não instala:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Dados não encontrados:**

```bash
# Certifique-se de que o CSV do PaySim está em data/
```

**CrewAI com erro:**

```bash
pip uninstall crewai -y && pip install crewai langchain-google-genai
```

**Verificar ambiente:**

```bash
python -c "import fastapi; print('✓ FastAPI'); import streamlit; print('✓ Streamlit'); import pytest; print('✓ pytest')"
```

---

## 📚 Arquivos Principais

| Arquivo                                    | Descrição                         |
|--------------------------------------------|-----------------------------------|
| `src/api/main.py`                          | API REST (FastAPI)                |
| `src/dashboard/app.py`                     | Dashboard (Streamlit)             |
| `src/orquestração/fluxo_crewai.py`         | Pipeline de orquestração          |
| `src/ferramentas/inferencia_modelos.py`    | Inferência dos modelos            |
| `src/modelos/arquitetura_lstm.py`          | LSTM bidirecional com atenção     |
| `src/modelos/arquitetura_gnn.py`           | GNN com GAT + GCN                 |
| `tests/conftest.py`                        | Fixtures de teste                 |
| `run_simulation.py`                        | Simulação via terminal            |

---

## 🎓 Referências

- PyTorch: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- PyTorch Geometric: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
- CrewAI: [docs.crewai.com](https://docs.crewai.com/)
- Google Gemini: [ai.google.dev](https://ai.google.dev/)
- FastAPI: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- Streamlit: [docs.streamlit.io](https://docs.streamlit.io/)
- Isolation Forest: [scikit-learn.org](https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest)

---

## TCC — Sistema Antifraude com IA Multiagente | 2026
