# ⚡ QUICK START - Guia Rápido de Execução

## 🚀 Setup Inicial (2 minutos)

### 1️⃣ Ativar Ambiente Virtual

**Windows PowerShell (Recomendado):**
```bash
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**
```bash
set PYTHONIOENCODING=utf-8
venv\Scripts\activate
```

**Linux/Mac:**
```bash
export PYTHONIOENCODING=utf-8
source venv/bin/activate
```

### 2️⃣ Verificar Instalação
```bash
pip --version
python --version
```

**Esperado:** Python 3.12+, pip atualizado ✅

⚠️ **Importante:** Sempre defina `PYTHONIOENCODING=utf-8` antes de ativar!

---

## 🔄 Fluxo Completo de Execução

### **PASSO 1: Preparar Dados**
```bash
python src/preprocessamento/carregar_paysim.py
```
📊 Carrega `data/PS_20174392719_1491204439457_log.csv` e cria `data/paysim_processed.parquet`

### **PASSO 2: Testar Sistema**
```bash
python teste_sistema_completo.py
```
✅ Executa 6 testes (preprocessamento, modelos, API, dashboard)
- Esperado: **6/6 PASSARAM** ✅

### **PASSO 3: Treinar Agentes & Modelos**

**3a. Isolation Forest (RÁPIDO - 30s)**
```bash
python -c "
from src.modelos.treinamento.treinar_modelos import *
import pandas as pd

df = pd.read_parquet('data/paysim_processed.parquet')
m = train_isolation_forest(prepare_balanced_sample(df))
save_model(m, 'src/modelos/modelos_salvos/isolation_forest.pkl')
print('✅ Isolation Forest treinado')
"
```

**3b. LSTM com PyTorch (5-10 min)**
```bash
python -c "
from src.modelos.arquitetura_lstm import *
import pandas as pd

df = pd.read_parquet('data/paysim_processed.parquet')
m = treinar_lstm(df.sample(2000), 20)
if m:
    salvar_lstm(m)
    print('✅ LSTM treinado e salvo')
"
```

**3c. GNN com PyTorch Geometric (5-10 min)**
```bash
python -c "
from src.modelos.arquitetura_gnn import *
import pandas as pd

df = pd.read_parquet('data/paysim_processed.parquet')
m, d = treinar_gnn(df.sample(2000), 20)
if m:
    salvar_gnn(m, d)
    print('✅ GNN treinado e salvo')
"
```

### **PASSO 4: Executar Sistema Completo**
```bash
python run_simulation.py
```
🤖 Orquestra agentes CrewAI para análise de fraudes em tempo real

---

## 🎯 Atalhos (Executar Tudo de Uma Vez)

```bash
# 1. Setup
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# 2. Dados + Testes
python src/preprocessamento/carregar_paysim.py && python teste_sistema_completo.py

# 3. Treinar + Rodar
python run_simulation.py
```

---

## 📱 Alternativas de Execução

### Dashboard Streamlit
```bash
streamlit run src/dashboard/app.py
```
🌐 Acessa em `http://localhost:8501`

### API FastAPI
```bash
uvicorn src.api.main:app --reload --port 8000
```
📡 Acessa em `http://localhost:8000/docs`

---

## 🔧 Troubleshooting

| Problema | Solução |
|----------|---------|
| ❌ `ModuleNotFoundError` | Ativar venv e reinstalar: `pip install -r requirements.txt` |
| ❌ `venv não ativa` | Verifique PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| ❌ Dados não encontrados | Descompacte `data/PS_20174392719_1491204439457_log.csv.zip` em `data/` |
| ❌ PyTorch erro | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| ❌ Encoding UTF-8 | Windows: `$env:PYTHONIOENCODING="utf-8"` antes de rodar |
| ❌ Google API erro | Configure `GOOGLE_API_KEY` em `.env` |
| ❌ CrewAI error | `pip install --upgrade crewai langchain-google-genai` |

---

## 📚 Documentação Completa

| Doc | Objetivo |
|-----|----------|
| 📖 **GUIA_IMPLEMENTACAO.md** | Técnico detalhado (1000+ linhas) |
| 📋 **README.md** | Overview geral do projeto |
| 📍 **INDICE.md** | Índice navegável por objetivo |
| ⚙️ **teste_sistema_completo.py** | Suite de testes (referência) |
| 🚀 **run_simulation.py** | Script principal com CrewAI |

---

## 📁 Estrutura do Projeto

```
tcc-antifraude/
├── src/
│   ├── preprocessamento/     ← Carregamento de dados (PaySim)
│   ├── modelos/              ← LSTM, GNN, Isolation Forest
│   │   ├── treinamento/      ← Scripts de treinamento
│   │   └── modelos_salvos/   ← Modelos treinados (.pt, .pkl)
│   ├── ferramentas/          ← Tools CrewAI + Inferência
│   ├── orquestração/         ← Agentes & fluxo CrewAI
│   ├── api/                  ← FastAPI (opcional)
│   └── dashboard/            ← Streamlit (opcional)
│
├── data/
│   ├── PS_20174392719_1491204439457_log.csv    ← Bruto (decomprimir)
│   └── paysim_processed.parquet                ← Processado (gerado)
│
├── tests/                    ← Suite de testes (pytest)
├── venv/                     ← Ambiente virtual
├── requirements.txt          ← Dependências
├── .env                      ← Variáveis de ambiente
├── pytest.ini                ← Configuração de testes
└── QUICK_START.md           ← Este arquivo

```

---

## ✅ Checklist de Setup

- [ ] venv criada: `.\venv\Scripts\Activate.ps1`
- [ ] Dependências instaladas: `pip install -r requirements.txt`
- [ ] `.env` configurado com `GOOGLE_API_KEY`
- [ ] Dados processados: `python src/preprocessamento/carregar_paysim.py`
- [ ] Testes passando: `python teste_sistema_completo.py`
- [ ] Modelos treinados: `run_simulation.py`

---

## 💡 Próximos Passos

1. ✅ Completar setup acima
2. 🎯 Executar `python run_simulation.py` para ver agentes em ação
3. 📊 Abrir dashboard: `streamlit run src/dashboard/app.py`
4. 📡 Explorar API: `uvicorn src.api.main:app --reload`
5. 🧪 Rodar testes: `pytest -v`

---

**Última atualização:** Maio 2026  
**Status:** ✅ Sistema 100% Funcional

### Testar Inferência
```python
import pandas as pd
from src.ferramentas.inferencia_modelos import isolation_forest_score, lstm_sequence_score

df = pd.read_parquet('data/paysim_processed.parquet')
tx = df.iloc[0].to_dict()

score_if = isolation_forest_score(tx)
score_lstm = lstm_sequence_score([tx])

print(f"IF Score: {score_if:.3f}")
print(f"LSTM Score: {score_lstm:.3f}")
```

### Executar Análise Completa
```python
import pandas as pd
from src.orquestração.fluxo_crewai import orchestrate_transaction

df = pd.read_parquet('data/paysim_processed.parquet')
resultado = orchestrate_transaction(df.iloc[0].to_dict())

print(f"Decisão: {resultado['decision']['label']}")
print(f"Score: {resultado['decision']['fraud_score']:.3f}")
```

---

## 🎯 Checklist de Validação

- [ ] `python teste_sistema_completo.py` → 6/6 testes
- [ ] `python run_simulation.py` → Análise de transações
- [ ] `python -c "import crewai; print('OK')"` → CrewAI funcional
- [ ] `python -c "from src.modelos.arquitetura_lstm import *; print('OK')"` → LSTM importa
- [ ] `python -c "from src.modelos.arquitetura_gnn import *; print('OK')"` → GNN importa
- [ ] Todos os arquivos em `src/modelos/modelos_salvos/` existem

---

## 📞 Suporte Rápido

**Verificar ambiente Python:**
```bash
python --version
pip list | grep -E "crewai|torch|pandas|scikit"
```

**Verificar arquivos críticos:**
```bash
dir data\paysim_processed.parquet
dir src\modelos\modelos_salvos\
```

**Limpar cache Python:**
```bash
python -c "import shutil; shutil.rmtree('src/__pycache__', ignore_errors=True); print('OK')"
```

---

## ⏱️ Tempos Esperados

| Tarefa | Tempo |
|--------|-------|
| Testes completos | ~2-3 min |
| Isolation Forest | ~5 min |
| LSTM (epochs=20) | ~10-15 min |
| GNN (epochs=20) | ~10-15 min |
| Simulação | ~1-2 min |

---

**🎉 Seu projeto está pronto! Execute `python teste_sistema_completo.py` e veja tudo funcionando!**
