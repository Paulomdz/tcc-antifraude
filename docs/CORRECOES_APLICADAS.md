# Correções aplicadas (código funcional e condizente com a monografia)

Este arquivo resume as mudanças feitas para corrigir os problemas apontados em
`RELATORIO_REVISAO_CODIGO.md` e `RELATORIO_TESE_VS_CODIGO.md` (na pasta do
TCC). Todas as mudanças foram testadas: `pytest` passa 133/133 e `pyflakes`
está limpo (exceto um padrão intencional pré-existente em
`arquitetura_lstm.py`/`arquitetura_gnn.py`, documentado abaixo).

## 1. Agente Juiz real via CrewAI + Google Gemini (a correção principal)

- `src/orquestração/definicoes_agentes.py`: reescrito para usar `crewai.LLM`
  em vez de `langchain_google_genai.ChatGoogleGenerativeAI` — versões atuais
  do CrewAI (testado com 1.15.18) não aceitam mais um objeto langchain no
  parâmetro `llm` de `Agent` (dá erro de validação Pydantic). Confirmei isso
  na prática instalando o CrewAI real e testando a construção do Agent. O
  modelo Gemini agora é configurável via variável de ambiente `GEMINI_MODEL`
  (padrão `gemini-2.5-flash` — o `gemini-1.5-flash` original provavelmente já
  não está mais disponível).
- `src/ferramentas/ferramentas_crewai.py`: novo `llm_judge_tool()`, que monta
  uma `Task` do CrewAI com o parecer dos três especialistas e pede ao Gemini
  uma resposta estruturada (`DECISAO`/`SCORE`/`JUSTIFICATIVA`), faz o parsing
  e devolve exatamente o mesmo formato que `judge_tool` (determinístico) já
  devolvia — nada mais no projeto precisou saber a diferença entre os dois.
- `src/orquestração/fluxo_crewai.py`: `orchestrate_transaction` agora tenta o
  Juiz-LLM primeiro (padrão `use_llm_judge=True`) e cai automaticamente para
  a regra determinística (`judge_tool`) se: (a) não houver `GOOGLE_API_KEY`
  configurada, (b) o CrewAI não estiver instalado, ou (c) a chamada falhar
  por qualquer motivo (rede, formato de resposta inesperado etc.) — sempre
  registrando um aviso no console, nunca quebrando a análise.
- **Testei de ponta a ponta**: com uma chave falsa configurada, o sistema de
  fato chamou a API do Gemini (recebeu 403 Forbidden, como esperado de uma
  chave inválida) e caiu no fallback determinístico sem erro. Com uma chave
  real e válida no `.env`, a chamada deve funcionar normalmente.
- Como combinado, a **análise em lote** (dashboard e `/analise_lote`) usa
  `use_llm_judge=False` por padrão — evita uma chamada de API por transação
  em lotes de até 500 itens — enquanto a **análise individual** (dashboard e
  `/analise_transacao`) usa o Juiz-LLM real.
- Criei `.env.example` — copie para `.env` e preencha `GOOGLE_API_KEY` para
  ativar de fato o Gemini (sem isso, o sistema continua funcionando
  normalmente com a regra determinística).
- `requirements.txt`: removida a dependência `langchain-google-genai` (não é
  mais necessária); `crewai` atualizado para `>=1.15.0`.

## 2. Bug do horário (regra da madrugada)

- `src/ferramentas/ferramentas_crewai.py`: nova função `_hora_do_dia()`
  substitui o cálculo antigo (`step // 60`, que tratava `step` como minutos).
  Agora usa `step % 24`, que funciona corretamente tanto para dados reais do
  PaySim (`step` = horas desde o início da simulação, 1–744) quanto para
  entrada interativa (`step` = hora do dia, 0–23).
- `src/dashboard/app.py`: `random_transaction()` e o formulário manual agora
  geram/gravam `step` como a hora (0–23), não mais "minutos do dia" (0–1439).
  A precisão de minuto é preservada apenas em `step_norm` (feature auxiliar
  de ML), não no campo `step` usado pela regra de risco.
- `src/api/main.py`: campo `step` do schema agora aceita `0` (antes exigia
  `>=1`, o que rejeitava meia-noite) e a descrição documenta a convenção.
- Adicionei um teste de regressão (`test_regra_madrugada_usa_hora_correta`)
  garantindo que uma transação com `step=4` (madrugada real) é bloqueada.

## 3. Score do Isolation Forest normalizado em [0, 1]

- `src/ferramentas/inferencia_modelos.py::isolation_forest_score` agora
  aplica um sigmoid sobre o `decision_function` do modelo, garantindo saída
  sempre em [0, 1] — antes podia retornar valores negativos (como o
  `-0.037` do próprio exemplo da monografia, §4.1), contradizendo a
  normalização declarada em §3.6.

## 4. `run_simulation.py` não sobrescreve mais a decisão do Juiz

- Antes, `run_crewai_simulation()` recalculava a decisão a partir do score
  com limiares diferentes (0.75/0.45) dos usados pelo Juiz (0.80/0.50) e
  descartava a justificativa original. Agora ele só repassa o resultado de
  `orchestrate_transaction` — uma única fonte de verdade para a decisão.

## 5. Testes corrigidos

- `teste_sistema_completo.py::teste_simulacao()`: corrigidas as chaves
  erradas (`label`/`fraud_score` → `decision`/`score`), que faziam esse
  teste sempre mostrar "DESCONHECIDO" independente do resultado real.
- `teste_lstm()`/`teste_gnn()`: não retornam mais `True` para qualquer
  exceção — só o fallback esperado (PyTorch ausente) conta como sucesso;
  um erro real agora reprova o teste e imprime o traceback.
- `tests/`: adicionados testes para `llm_judge_tool`, `_parse_judge_output`,
  `_hora_do_dia` e o roteamento `use_llm_judge` em `orchestrate_transaction`
  (todos mockando o CrewAI — não fazem chamadas reais de API). Testes
  existentes ajustados para as novas assinaturas/comportamentos (ex.:
  `random_transaction()["step"]` agora é 0–23, não 0–1439).
- **Resultado: 133/133 testes passando.**

## 6. Nome do arquivo processado alinhado com a monografia

- `src/preprocessamento/carregar_paysim.py`: o nome padrão voltou a ser
  `data/paysim_dados_processados.parquet` (como documentado na monografia,
  §3.4); o nome em inglês (`paysim_processed.parquet`) virou apenas um
  fallback de leitura para compatibilidade.

## 7. Limpeza do repositório

- Removidos: `execute_replacement.py` (caminho hardcoded de outra máquina,
  já quebrado), `requirements.txt.bak` (idêntico ao atual),
  `lstm_sequence_model.pkl` (4 bytes, pickle de `None`, não usado por nada),
  `.coverage` e `pip_install_log.txt` (artefatos que não deveriam ser
  versionados).
- `.gitignore`: removida uma linha `` ``` `` solta (sobra de colar Markdown)
  e adicionadas entradas para `.coverage`, `htmlcov/` e `pip_install_log.txt`.
- Imports não usados removidos em vários arquivos (`run_simulation.py`,
  `arquitetura_lstm.py`, `arquitetura_gnn.py`, `treinar_modelos.py`,
  `conftest.py`, `test_api.py`); f-strings sem placeholder corrigidas.

## 8. Dataset real do PaySim processado e modelos retreinados

- O dataset real (`PS_20174392719_1491204439457_log.csv`, ~6,3M linhas /
  ~470MB) foi colocado em `data/` e processado de ponta a ponta.
- `src/preprocessamento/carregar_paysim.py`: adicionada uma nova função
  `process_paysim_in_chunks()` que lê o CSV em blocos (`pandas.read_csv(...,
  chunksize=...)`) e grava o Parquet processado incrementalmente via
  `pyarrow.parquet.ParquetWriter` — necessário porque carregar o CSV inteiro
  de uma vez estourava a memória disponível em ambientes com poucos GB de
  RAM. `main()` agora usa esse caminho por padrão; `load_paysim()` +
  `preprocess_paysim()` continuam existindo e são usados pelos testes/casos
  pequenos.
- **`step_norm` corrigido**: em vez de normalizar pelo máximo observado no
  lote/DataFrame em mãos (`df["step"].max()`), agora usa uma constante fixa
  `STEP_MAX = 744` (31 dias × 24h, a duração canônica da simulação PaySim),
  já usada em outros pontos do projeto (dashboard, testes, API). Isso
  resolve a inconsistência antes documentada aqui como limitação — o valor
  real observado no dataset (máximo de 743) confirma que 744 é a referência
  correta.
- `src/modelos/treinamento/treinar_modelos.py`: nova função
  `load_balanced_sample_from_parquet()` monta a amostra balanceada de
  treinamento lendo o Parquet processado um row-group por vez (via
  `pyarrow`), mantendo todas as fraudes (raras) e uma amostra proporcional de
  não-fraudes — evita carregar as ~6,3M linhas inteiras em memória só para
  extrair uma amostra de 10 mil.
- **Isolation Forest retreinado** com uma amostra balanceada real (5.000
  fraudes + 5.000 não-fraudes, das 8.213 fraudes existentes no dataset real)
  usando a versão do scikit-learn instalada no ambiente (1.7.2) — o aviso
  `InconsistentVersionWarning` (modelo antigo treinado com 1.8.0) não ocorre
  mais.
- `teste_sistema_completo.py`: ajustado para carregar amostras pequenas do
  Parquet processado (um row-group via `pyarrow`) em vez de `pd.read_parquet`
  do arquivo inteiro, pelo mesmo motivo de memória. Rodado de ponta a ponta
  com o dataset real: 6/6 testes passaram.

## 9. PyTorch e PyTorch Geometric instalados; LSTM e GNN treinados de verdade

- Instalados `torch==2.5.1` e `torch-geometric==2.8.0.post1` (build CPU para
  linux/aarch64 — versões anteriores/posteriores do torch nesse arquitetura
  ou vêm com dependências CUDA obrigatórias mesmo sem GPU, ou têm
  incompatibilidade de ABI com numpy 2.x; a 2.5.1 é a primeira a resolver
  ambos os problemas nesta plataforma).
- Rodei `python -m src.modelos.treinamento.treinar_modelos` com PyTorch
  disponível: os três modelos foram treinados de ponta a ponta sobre o
  dataset real do PaySim — Isolation Forest (amostra balanceada de 10 mil
  transações), LSTM (10 épocas, acurácia ~79% na própria amostra de treino)
  e GNN (10 épocas) — e salvos em `src/modelos/modelos_salvos/` (
  `isolation_forest.pkl`, `lstm_model.pt`, `gnn_model.pt`).
- `teste_sistema_completo.py` roda 6/6 com os três modelos reais ativos
  (antes, LSTM/GNN caíam no fallback por falta de PyTorch).
- Aviso não crítico observado: `torch.load(..., weights_only=False)` (usado
  em `inferencia_modelos.py` para carregar `lstm_model.pt`/`gnn_model.pt`)
  emite um `FutureWarning` do PyTorch sobre uma mudança de padrão de
  segurança em versões futuras — não afeta o funcionamento atual, mas é uma
  migração recomendada (`weights_only=True` ou `add_safe_globals`) para uma
  versão futura do PyTorch.

## O que NÃO foi alterado (limitações que continuam valendo)

- O Isolation Forest retreinado tem acurácia baixa (~59%) na própria amostra
  de treino — isso reflete as features simples usadas
  (`extract_behavior_features`: valores brutos de saldo/valor + `step_norm`)
  e o `contamination=0.5` fixo, não um bug de código. Ajustar/enriquecer as
  features ou o contamination é uma decisão de modelagem para a monografia,
  não uma correção de bug.
- LSTM e GNN foram treinados por poucas épocas (10) sobre uma amostra
  pequena (10 mil / 2 mil transações) — suficiente para validar que o
  pipeline funciona de ponta a ponta com PyTorch real, mas não uma
  otimização de hiperparâmetros nem uma avaliação formal (precisão, recall,
  F1) do desempenho desses modelos.
- O padrão de "redefinição" em `arquitetura_lstm.py`/`arquitetura_gnn.py`
  (funções definidas uma vez com PyTorch disponível, e de novo como
  placeholder quando não disponível) continua existindo no código — agora
  testado com PyTorch real disponível (o caminho "com PyTorch" foi exercido
  de fato nesta rodada), mas ainda esteticamente confuso para linters; não
  mexi na estrutura em si, só confirmei que os dois caminhos (com e sem
  PyTorch) funcionam corretamente.

## 10. `run_simulation.py` também estourava memória (OOM) — corrigido

Ao rodar o sistema de ponta a ponta numa máquina real (~4GB de RAM) para
validar o pipeline completo, `python run_simulation.py` foi morto pelo
kernel (OOM, exit code 137) logo na primeira execução. Causa: assim como em
`treinar_modelos.py` e `teste_sistema_completo.py` antes da correção
(ver seção 8), `load_sample_transactions()` usava `pd.read_parquet(PROCESSED_DATA_PATH)`
para carregar o Parquet inteiro (~6,36 milhões de linhas) só para amostrar
5-10 transações de exemplo.

Corrigido para ler apenas um row-group do Parquet via
`pyarrow.parquet.ParquetFile.read_row_group(0)` e amostrar a partir dele —
mesmo padrão já usado em `_load_sample_df()` (teste_sistema_completo.py) e
`load_balanced_sample_from_parquet()` (treinar_modelos.py). Após a correção,
`python run_simulation.py` e `python teste_sistema_completo.py` rodaram do
início ao fim com sucesso na máquina real, com os modelos Isolation
Forest/LSTM/GNN já treinados (ver seção 9) sendo usados de fato na
inferência.

Observação da execução real: ao retreinar o LSTM do zero em
`teste_sistema_completo.py` (5 épocas, amostra de ~2000 linhas), o modelo
atingiu "Loss: 0.0000, Acurácia: 100.00%" — um sinal claro de overfitting
numa amostra pequena e não representativo do desempenho real do modelo;
reforça o item já registrado na seção 9 sobre a necessidade de uma avaliação
formal (precisão/recall/F1) e de mais dados/épocas antes de reportar
métricas de desempenho na monografia.

## 11. Dashboard: StreamlitValueAboveMaxError no preview de saldo

Rodando o dashboard ao vivo numa maquina real e clicando repetidamente em
"Simular nova transacao", o campo somente-leitura "Novo saldo - Origem"
(preview calculado como saldo anterior + valor, para CASH_IN) podia superar
o teto do proprio widget (_MAX_FINANCIAL_VALUE = R$ 9.999.999,00) quando os
dois valores sorteados estavam proximos desse teto - a soma passava de
R$ 17 milhoes num caso observado, e o Streamlit lancava
StreamlitValueAboveMaxError, travando a tela.

Corrigido em `src/dashboard/app.py`: como esse campo e so uma
pre-visualizacao (o valor de fato usado na analise, `calculated_new_balance`,
e recalculado de forma independente no submit do formulario), os limites do
widget foram ampliados para o dobro do teto normal, cobrindo qualquer soma
possivel de dois valores no teto, sem alterar o calculo real usado na
analise.

## Para testar de verdade com o Gemini

1. Copie `.env.example` para `.env`.
2. Preencha `GOOGLE_API_KEY` com uma chave do Google AI Studio
   (https://aistudio.google.com/apikey).
3. Rode `python run_simulation.py` ou o dashboard normalmente — a análise
   individual vai chamar o Gemini de verdade; se algo falhar, o aviso
   `[AVISO] Juiz-LLM (CrewAI/Gemini) indisponível ou falhou (...)` aparece no
   console e a decisão determinística é usada no lugar, sem quebrar a
   aplicação.
