# Relatório de Revisão de Código — tcc-antifraude-main

Revisão completa de todos os arquivos `.py` do projeto (28 arquivos), incluindo:
verificação de sintaxe (`py_compile`), análise estática (`pyflakes`), execução da
suíte de testes real (115 testes, `pytest`) e leitura manual de cada módulo em
busca de erros lógicos, inconsistências de dados e problemas de manutenção.

**Resumo geral:** não há erros de sintaxe e a suíte de testes (115 testes) passa
100%. Os problemas encontrados são, portanto, principalmente **erros lógicos e
de projeto** que os testes atuais (fortemente baseados em mocks) não cobrem.

---

## 🔴 Críticos (afetam a corretude do sistema)

### 1. O campo `step` é interpretado de 3 formas incompatíveis
- Em `src/preprocessamento/carregar_paysim.py`, `step` é tratado como o padrão
  real do PaySim: **1 unidade = 1 hora**, variando de 1 a 744 (31 dias), e
  `step_norm = step / step.max()`.
- Em `src/ferramentas/ferramentas_crewai.py` (usado por `behavior_specialist_tool`,
  `temporal_specialist_tool`, `identity_specialist_tool` e `judge_tool`), o mesmo
  campo é tratado como **minutos desde a meia-noite** (`hours = step // 60`,
  `minutes = step % 60`).
- Em `src/dashboard/app.py::random_transaction()`, `step` é gerado como
  `random.randint(0, 1439)` — minutos do dia — e `step_norm = step / 1440`.

Resultado: quando uma transação real do PaySim (`step` de 1 a 744) passa pelo
pipeline de decisão, o horário "calculado" (`hours`/`minutes`) não corresponde
à hora real da transação. Isso corrompe justamente a regra mais enfatizada do
sistema — bloqueio automático de transações de alto valor na madrugada
(00:00–05:59) — que aparece em `judge_tool`, nos três agentes e no dashboard.
Além disso, os modelos são treinados com `step_norm = step/744` mas usados em
produção (dashboard) com `step_norm = step/1440`: é um desvio real de
distribuição entre treino e inferência (train/serve skew) para essa feature.

### 2. O sistema não usa CrewAI/LLM em nenhum ponto executado
- `src/orquestração/fluxo_crewai.py::orchestrate_transaction` (usado pela API,
  por `run_simulation.py` e pelo dashboard) apenas chama funções Python
  determinísticas em `ferramentas_crewai.py`. Não há `Crew`, `Task` nem chamada
  de LLM em nenhum lugar desse fluxo.
- `src/orquestração/definicoes_agentes.py` é o único arquivo que de fato usa
  `crewai.Agent` e `ChatGoogleGenerativeAI` (Gemini) — mas **nunca é importado
  por nenhum outro módulo do projeto** (confirmado via busca em todo o
  código-fonte). É código morto.

Ou seja: toda a comunicação do projeto ("Sistema Multiagente de IA", "CrewAI +
Gemini", docstrings e prints como "Iniciando Simulador de Detecção de Fraudes
com CrewAI + Gemini") descreve um comportamento que o código não executa. O
que roda de fato é um conjunto de regras determinísticas ponderadas
(`0.35*comportamental + 0.30*temporal + 0.35*identidade` + regras rígidas de
madrugada). Vale muita atenção a isso especialmente por ser um TCC — a
descrição da arquitetura na monografia deve ser conferida contra o que o
código realmente faz.

### 3. `teste_sistema_completo.py::teste_simulacao()` lê chaves erradas
```python
decisao = resultado['decision']
label = decisao.get('label', 'DESCONHECIDO')       # chave real é "decision"
score = decisao.get('fraud_score', 0.0)            # chave real é "score"
```
O dicionário retornado por `judge_tool` tem as chaves `decision`, `score`,
`justification`, `specialist_outputs` — nunca `label` ou `fraud_score`. Este
teste, portanto, **sempre imprime "DESCONHECIDO" e score "N/A"**,
independentemente do resultado real, e mesmo assim é contado como "PASSOU".

### 4. `teste_lstm()` e `teste_gnn()` sempre retornam `True`
```python
except Exception as e:
    print(f" Aviso em LSTM: {str(e)}")
    return True
```
Qualquer exceção (não só "PyTorch ausente", mas também um bug real de código)
é silenciada e o teste é contado como aprovado no sumário final. Isso mascara
falhas reais de treinamento.

### 5. `execute_replacement.py` está quebrado e é específico de outra máquina
```python
work_dir = Path(r'c:\Users\paulo\Desktop\Uni\TCC\tcc-antifraude')
os.chdir(work_dir)
...
if (work_dir / 'requirements_new.txt').exists():
```
Caminho absoluto do Windows de outro desenvolvedor ("paulo"), e
`requirements_new.txt` não existe no repositório. O script não roda como está
e não deveria ter sido versionado (ou deveria ser apagado, já cumpriu seu
papel único de substituir o requirements.txt uma vez).

---

## 🟠 Médios

### 6. Isolation Forest treinado em amostra artificialmente balanceada
`train_isolation_forest` treina com `contamination=0.5` sobre uma amostra 50%
fraude / 50% não-fraude (`prepare_balanced_sample`). Em produção (API,
dashboard, dados reais), a taxa de fraude é muito menor que 50%. Isso é um
desvio de distribuição treino↔produção que compromete a validade dos scores
de anomalia do Isolation Forest fora do cenário de teste balanceado.

### 7. Incompatibilidade de versão do modelo serializado
Ao carregar `isolation_forest.pkl` neste ambiente, o scikit-learn emite:
```
InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.8.0
when using version 1.7.2.
```
O `requirements.txt` fixa apenas `scikit-learn>=1.3.0` (sem teto), então o
ambiente de qualquer pessoa pode acabar com uma versão diferente da que gerou
o `.pkl` — risco real de erro silencioso ou incompatibilidade ao carregar o
modelo.

### 8. Arquivo de modelo órfão e vazio
`src/modelos/modelos_salvos/lstm_sequence_model.pkl` tem **4 bytes** e é
literalmente `pickle.dump(None, f)`. Nenhum código no projeto o referencia
(confirmado via busca) — é lixo deixado de uma tentativa anterior.

### 9. `src/dashboard/app.py` executa `main()` no import do módulo
```python
if __name__ == "__main__":
    ...
main()   # linha 604 — fora do bloco acima, roda sempre que o módulo é importado
```
Por isso `tests/test_dashboard.py` precisa mockar `streamlit` e `plotly` no
`sys.modules` **antes** de importar `src.dashboard.app`, só para não quebrar.
Qualquer outra ferramenta que importe esse módulo (REPL, gerador de docs,
outro script) vai disparar a interface inteira do Streamlit fora de contexto.

### 10. Nome de pacote com caracteres acentuados: `src/orquestração`
Funciona no Linux/macOS, mas é um risco de portabilidade (Windows/encondings,
strings de `importlib.import_module("src.orquestração.fluxo_crewai")`,
suporte de ferramentas) — especialmente chamativo num projeto que também
inclui `activate_venv.bat`/`.ps1` para Windows.

---

## 🟡 Menores / qualidade e manutenção

- **Redefinições intencionais mas confusas**: `arquitetura_lstm.py` e
  `arquitetura_gnn.py` definem `treinar_*/salvar_*/carregar_*` uma vez dentro
  do `if TORCH_AVAILABLE:` e depois **de novo, incondicionalmente**, como
  placeholders (padrão funciona, mas o `pyflakes` acusa "redefinition of
  unused" e fica confuso de manter).
- **Imports não usados**: `os`/`Path` em `run_simulation.py`;
  `os`/`train_test_split` em `treinar_modelos.py`; `numpy`/`pandas`/`pickle`
  em `arquitetura_lstm.py`; `numpy`/`pandas`/`Dict`/`Tuple`/`defaultdict` em
  `arquitetura_gnn.py`; `pytest` em `test_api.py`; `numpy` em `conftest.py`.
- **f-strings sem placeholders** (six ocorrências em `teste_sistema_completo.py`,
  uma em `src/dashboard/app.py:267`) — provavelmente esqueceram de interpolar
  alguma variável.
- **Mensagem de erro desatualizada**: `load_pickle_model` aponta para
  `src/models/saved_models/` (nome antigo em inglês), mas a pasta real hoje é
  `src/modelos/modelos_salvos/` — resquício do processo de renomear o projeto
  para português (confirmado pelos `.pyc` órfãos em `__pycache__/` de módulos
  que não existem mais: `model_inference.py`, `crewai_workflow.py`,
  `train_models.py`, `agent_definitions.py`, `load_paysim.py`).
- **Checagem morta em `random_transaction()`**: `if amount > max_daily_limit`
  nunca é verdadeira, pois `amount` já foi sorteado com
  `random.uniform(10.0, _MAX_FINANCIAL_VALUE)` (mesmo limite).
- **`.gitignore` contém uma linha `` ``` `` solta** (sobra de colar conteúdo
  em Markdown) — inofensivo, mas é claramente um erro de copiar/colar.
- **`.coverage` e `pip_install_log.txt` versionados** como arquivos normais,
  sem entrada correspondente no `.gitignore`.
- **`requirements.txt.bak` é idêntico, byte a byte, a `requirements.txt`**
  (sobra do `execute_replacement.py`) — não agrega nada.
- **Inconsistência de moeda**: `run_simulation.py` imprime valores com `$`,
  enquanto todo o resto do projeto usa `R$` (o sistema é sobre regras de risco
  bancário brasileiro).
- **Ruído aleatório nos scores de fallback**: `lstm_sequence_score` e
  `gnn_identity_score` somam `random.uniform(-0.1, 0.1)` ao score quando
  PyTorch/PyTorch Geometric não estão instalados (que é justamente o estado
  deste ambiente de teste) — a mesma transação pode receber decisões
  diferentes em execuções distintas. Aceitável como heurística de fallback,
  mas vale deixar isso bem documentado/rotulado onde aparece.

---

## ✅ O que está OK
- Nenhum erro de sintaxe em nenhum dos 28 arquivos `.py` (`py_compile` limpo).
- Suíte de testes completa (115 testes em `tests/`) passa 100% após instalar
  as dependências leves (`pytest`, `fastapi`, `streamlit`, `plotly`,
  `scikit-learn`) — os testes cobrem bem as funções puras e os "caminhos
  felizes"/mockados, mas não pegam os problemas semânticos acima porque
  quase tudo é mockado.
- Nenhuma credencial ou segredo hard-coded encontrado no código.
- Pesos do `judge_tool` somam corretamente 1.0 (`0.35+0.30+0.35`).

---

## Observação
Notei que uma pasta pessoal do TCC (com o texto da monografia, proposta e
banner) foi conectada nesta sessão. Não abri esses documentos nesta revisão
(focada no código), mas se você quiser, posso conferir se a descrição da
arquitetura na monografia bate com o que o código realmente executa —
especialmente o ponto #2 acima (CrewAI/LLM não está de fato em uso no fluxo
de decisão).
