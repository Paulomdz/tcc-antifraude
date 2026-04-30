# ⚡ QUICK START - Comandos Rápidos

## 🚀 Comece Aqui (30 segundos)

### 1. Ativar Ambiente Virtual
```bash
venv\Scripts\activate
```

### 2. Testar Tudo
```bash
python teste_sistema_completo.py
```

**Resultado esperado:** `6/6 testes PASSARAM ✅`

---

## 📊 Executar Componentes

### Pré-processar Dataset
```bash
python src/preprocessamento/carregar_paysim.py
```

### Treinar Isolation Forest
```bash
python -c "from src.modelos.treinamento.treinar_modelos import *; import pandas as pd; df = pd.read_parquet('data/paysim_processed.parquet'); m = train_isolation_forest(prepare_balanced_sample(df)); save_model(m, 'isolation_forest.pkl')"
```

### Treinar LSTM (requer PyTorch)
```bash
pip install torch
python -c "from src.modelos.arquitetura_lstm import *; import pandas as pd; df = pd.read_parquet('data/paysim_processed.parquet'); m = treinar_lstm(df.sample(2000), 20); salvar_lstm(m) if m else None"
```

### Treinar GNN (requer PyTorch Geometric)
```bash
pip install torch-geometric
python -c "from src.modelos.arquitetura_gnn import *; import pandas as pd; df = pd.read_parquet('data/paysim_processed.parquet'); m, d = treinar_gnn(df.sample(2000), 20); salvar_gnn(m, d) if m else None"
```

### Executar Simulação
```bash
python run_simulation.py
```

---

## 🔧 Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError` | Ativar venv: `venv\Scripts\activate` |
| PyTorch não instala | `pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu` |
| Dados não encontrados | Descompacte `PS_20174392719_1491204439457_log.csv.zip` em `data/` |
| Erro encoding | Use: `$env:PYTHONIOENCODING="utf-8"` antes de rodar |
| CrewAI com erro | `pip uninstall crewai -y && pip install crewai langchain-google-genai` |

---

## 📚 Documentação

- 📖 **GUIA_IMPLEMENTACAO.md** - Guia técnico completo (1000+ linhas)
- 📋 **RESUMO_IMPLEMENTACAO.md** - O que foi feito (esta sessão)
- 📄 **README_NOVO.md** - Novo README completo
- ⚙️ **teste_sistema_completo.py** - Suite de testes (referência)

---

## 📍 Estrutura de Pastas (Resumida)

```
src/
├── preprocessamento/        ← Carregamento de dados
├── modelos/                 ← LSTM, GNN, Isolation Forest
│   ├── treinamento/
│   └── modelos_salvos/
├── ferramentas/             ← Inferência e tools
└── orquestração/            ← CrewAI e agentes

data/
├── paysim_processed.parquet ← Dataset processado
└── PS_20174392719_1491204439457_log.csv.zip ← Bruto
```

---

## 💡 Exemplos Python

### Carregar e Processar Dados
```python
import pandas as pd
from src.preprocessamento.carregar_paysim import load_paysim, preprocess_paysim

df_raw = load_paysim()
df_processed = preprocess_paysim(df_raw)
print(f"Transações: {len(df_processed)}")
```

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
