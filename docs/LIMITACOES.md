# Limitações do projeto
Documento vivo: atualizar a cada mudança que crie, resolva ou altere uma limitação. Os relatórios de etapa guardam a versão da época.

## 1. Dados

- **Base sintética**
  - O que é: os padrões de fraude seguem as regras do gerador e podem inflar as métricas.
  - Por que existe: a base foi construída de forma sintética, sem dados operacionais de instituição financeira.
  - Evidência: docs/secao_dataset_tcc.txt, seções 3.X.1 e 3.X.5.
  - Efeito no TCC: os resultados valem para esta base e não demonstram desempenho em produção.
- **Razão valor/saldo com alto poder de separação**
  - O que é: a mediana de amount / oldbalanceOrg é 1,0 nas fraudes e 0,006 nas legítimas.
  - Por que existe: as fraudes da base tendem a esvaziar a conta de origem.
  - Evidência: docs/secao_dataset_tcc.txt, seções 3.X.4 e 3.X.5; recálculo a partir do CSV: 1,0 e 0,00614.
  - Efeito no TCC: pode elevar as métricas; por isso os resultados são reportados também sem esse atributo (seção 2).
- **Valores de Pix Saque**
  - O que é: os valores de PIX_SAQUE não seguem as regras reais do Pix, segundo a revisão do grupo (regra real a confirmar em fonte oficial). Com step < 600, a mediana é R$ 81,93 e o máximo, R$ 361.985,58.
  - Por que existe: sem registro no repositório de que isso foi pensado.
  - Evidência: dataset/dataset_pix_banco.csv (29.607 linhas de PIX_SAQUE com step < 600).
  - Efeito no TCC: essa categoria não deve ser apresentada como reprodução fiel da modalidade real.
- **Regra de linha de base sem implementação**
  - O que é: a regra (≥ 90% do saldo e ≥ R$ 5.000) é descrita como "calculada à parte", mas não há implementação no repositório nem no histórico git.
  - Por que existe: sem registro no repositório.
  - Evidência: docs/secao_dataset_tcc.txt:103-106; busca no working tree e em git log --all, em 03/11/2026.
  - Efeito no TCC: o texto promete uma comparação que o repositório não permite reproduzir.

## 2. Modelos

- **GNN não convergiu**
  - O que é: atingiu o limite de 10 épocas com a validação ainda subindo (0,2931 → 0,3367).
  - Por que existe: limite de 10 épocas por restrição de tempo; ~145 s por época, treino final de 3.230 s, GPU de 8 GB e 15 GB de RAM.
  - Evidência: docs/etapas/etapa2.txt, seção 3; reports/etapa2_final_gnn.json; reports/final_gnn.log (local, fora do git), que traz avisos de alocação CUDA, sem erro.
  - Efeito no TCC: os resultados da GNN são parciais e podem não representar o potencial do modelo.
- **GNN sem grade e sem ablação**
  - O que é: usa uma única configuração e não mede o efeito da retirada da razão valor/saldo.
  - Por que existe: prazo; a ablação foi estimada em ~55 min.
  - Evidência: docs/etapas/etapa2.txt, seções 3, 4 e 8.
  - Efeito no TCC: a comparação entre modelos não é simétrica.
- **Grade do IF parcial**
  - O que é: 9 de 27 combinações, com max_samples fixo em auto. n_estimators=100 foi o menor testado e o melhor (100: 0,6626; 200: 0,6141; 300: 0,5771).
  - Por que existe: foi usado o primeiro passe da grade.
  - Evidência: reports/etapa2_grid_v2.csv; docs/etapas/etapa2.txt, seção 2; reports/grid_if_lstm.log (local, fora do git).
  - Efeito no TCC: o melhor valor está na borda da grade, e valores menores não foram testados.
- **Limites da LSTM**
  - O que é: 4 de 8 configurações tiveram a melhor época no teto de 10; no teste, precisão 0,949 e recall 0,230.
  - Por que existe: máximo de 10 épocas; a causa do recall baixo: sem registro no repositório.
  - Evidência: docs/etapas/etapa2.txt, seção 3; reports/etapa2_final_if_lstm.json; reports/grid_if_lstm.log (local, fora do git).
  - Efeito no TCC: a comparação entre configurações é limitada, e o modelo deixa passar a maior parte das fraudes no limiar escolhido.
- **Dependência da razão valor/saldo**
  - O que é: sem ela, o PR-AUC no teste cai de 0,6881 para 0,3641 no IF e de 0,3577 para 0,3462 na LSTM.
  - Por que existe: é o atributo com maior separação da base. A ideia de que o histórico compensa na LSTM é hipótese não verificada.
  - Evidência: reports/etapa2_ablacao.json; reports/etapa2_final_if_lstm.json.
  - Efeito no TCC: o desempenho do IF depende fortemente de um padrão ligado ao gerador.
- **Uma única semente e um único split**
  - O que é: semente 42 e um split temporal, sem estimativa de variância.
  - Por que existe: sem registro no repositório.
  - Evidência: src/modelos/treinamento/treinar_modelos.py:64; src/preprocessamento/carregar_dataset.py:254.
  - Efeito no TCC: as métricas são pontuais, sem intervalo de confiança; pequenas diferenças não devem ser tratadas como relevantes.

## 3. Avaliação

- **Sem métrica da decisão final**
  - O que é: só há métricas por modelo no teste (PR-AUC: IF 0,6881; GNN 0,4081; LSTM 0,3577), não da decisão final do sistema.
  - Por que existe: avaliar_scores.py foi adiado.
  - Evidência: reports/etapa2_final_if_lstm.json; reports/etapa2_final_gnn.json.
  - Efeito no TCC: não é possível afirmar o desempenho do sistema multiagente como um todo.
- **Sem métricas por tipo de transação**
  - O que é: os relatórios trazem só métricas globais por modelo.
  - Por que existe: sem registro no repositório.
  - Evidência: reports/etapa2_final_if_lstm.json; reports/etapa2_final_gnn.json.
  - Efeito no TCC: não se sabe em quais tipos os modelos acertam ou erram mais.
- **Sem comparação com a regra de linha de base do banco**
  - O que é: a comparação descrita no texto do dataset não foi feita.
  - Por que existe: a regra não está implementada (seção 1).
  - Evidência: docs/secao_dataset_tcc.txt:103-106.
  - Efeito no TCC: não há como dizer quanto os agentes superam uma regra simples.
- **avaliar_scores.py adiado**
  - O que é: o script de avaliação não foi feito.
  - Por que existe: decisão de adiamento na Etapa 3.
  - Evidência: docs/etapas/etapa3.md.
  - Efeito no TCC: mantém abertas as lacunas de avaliação desta seção.
- **Acurácia não serve como métrica**
  - O que é: com 5,56% de fraude, um modelo que nunca acusa fraude acerta 94,4%. O PR-AUC de acaso na validação é 0,0539.
  - Por que existe: a base é desbalanceada.
  - Evidência: README.md (25.247 fraudes em 454.117 registros); docs/etapas/etapa2.txt, seção 1.
  - Efeito no TCC: os resultados são reportados por PR-AUC, precisão, recall e F1.
- **Releitura do teste**
  - O que é: --final-if-lstm e --final-gnn podem ler o teste de novo se forem executados outra vez; só --ablacao recusa. test_read_count é fixo em 1 nos relatórios e não prova leitura única.
  - Por que existe: os modos finais não têm a checagem de --ablacao.
  - Evidência: src/modelos/treinamento/treinar_modelos.py:1499, :1942, :1561, :1992 e :1598.
  - Efeito no TCC: a leitura única nos modos finais depende de disciplina de execução, não de bloqueio no código.

## 4. Sistema multiagente e juiz

- **Juiz Gemini sem avaliação**
  - O que é: não há avaliação das decisões do juiz. Duas chamadas acidentais no início não foram avaliação.
  - Por que existe: a política proíbe chamar o Gemini em teste, treino ou ablação.
  - Evidência: docs/etapas/etapa0.txt, seção 0.4; docs/etapas/etapa2.txt, seção 6; docs/REGRAS_VIGENTES.md, seções 8 e 16 (local, fora do git).
  - Efeito no TCC: o modo production não tem medida de qualidade.
- **Pesos e limites de decisão sem calibração registrada**
  - O que é: pesos de agregação 0,35 / 0,30 / 0,35 e limites de decisão 0,50 / 0,80.
  - Por que existe: sem registro de calibração no repositório.
  - Evidência: src/ferramentas/ferramentas_agentes.py:145, :152 e :154.
  - Efeito no TCC: esses valores devem ser apresentados como parâmetros definidos, não como resultado de otimização.
- **Modo deep sem endpoint**
  - O que é: existe só como função de orquestração, sem endpoint na API. Os testes usam agentes simulados.
  - Por que existe: decisão D8: o lote usa fast, a análise individual usa production, e deep permanece só como função.
  - Evidência: src/ferramentas/ferramentas_agentes.py:297; tests/test_orquestacao.py:139 e :241; docs/etapas/etapa2_api.txt (D8).
  - Efeito no TCC: o modo deep não pode ser apresentado como avaliado nem como disponível na API.
- **Limite de R$ 10.000 na madrugada**
  - O que é: parâmetro de política, não calibrado com os dados. A referência regulatória do Pix é R$ 1.000 por conta entre pessoas físicas, das 20h às 6h (fonte a confirmar na resolução do Banco Central), com janela diferente de is_madrugada (0h a 5h).
  - Por que existe: sem registro no repositório da origem do valor.
  - Evidência:
    - Agente temporal: src/ferramentas/ferramentas_agentes.py:63 e :66.
    - Agregador do modo fast: src/ferramentas/ferramentas_agentes.py:122, :130 e :148.
    - Instruções dos agentes: src/orquestacao/definicoes_agentes.py:68, :81, :90, :110, :123, :132 e :133.
    - is_madrugada: src/preprocessamento/carregar_dataset.py:145, :184 e :236.
    - Contagem com step < 600 (366.650 transações, 5,54% de fraude): de 0h a 5h, amount >= 10.000 reúne 9.365 (82,77% de fraude) e amount >= 1.000, 17.276 (68,17%); das 20h às 6h, amount >= 10.000 reúne 10.086 (81,80%) e amount >= 1.000, 20.545 (61,69%).
  - Efeito no TCC: a contagem mostra só cortes de valor e horário em todos os tipos; não separa pessoa física nem limite por conta, então não mede a regra do Pix em si.
- **A regra não é evidência independente**
  - O que é: a regra usa padrões que o gerador colocou na base.
  - Por que existe: a base é sintética, com concentração noturna de fraudes definida na geração.
  - Evidência: docs/secao_dataset_tcc.txt, seções 3.X.4 e 3.X.5.
  - Efeito no TCC: bons resultados da regra nesta base não comprovam desempenho independente.
- **Texto do juiz alterado sem avaliação**
  - O que é: as menções a DEBIT e TRANSFER foram trocadas pelos tipos usados na regra do código; o efeito no Gemini não foi medido, nem antes nem depois.
  - Por que existe: os tipos não existem no dataset atual, e a política proíbe chamar o Gemini em teste, treino ou ablação.
  - Evidência: src/orquestacao/definicoes_agentes.py:132-133.
  - Efeito no TCC: o comportamento do juiz com o texto novo não tem medida.

## 5. API

- **Validada só com lote sintético**
  - O que é: o teste com artefatos reais usou 500 transações iguais, com IDs diferentes.
  - Por que existe: o objetivo era validar o contrato, sem chamar o Gemini.
  - Evidência: docs/etapas/etapa2.txt, seção 5.
  - Efeito no TCC: valida o contrato da API, não o desempenho dos modelos.
- **oldbalanceOrg obrigatório**
  - O que é: saldo 0 enviado explicitamente continua aceito, com razão alta. Sem chave e sem saldo, a API responde 422 antes do 503 do D7. No lote, um item sem saldo faz o lote inteiro responder 422.
  - Por que existe: o saldo ausente gera erro, sem valor estimado, e a validação da entrada ocorre antes do processamento.
  - Evidência: commit e2f1290; src/api/main.py:80.
  - Efeito no TCC: saldo zero ainda pode distorcer o score e o top-3, e um lote não é aceito parcialmente quando há item sem saldo.
- **Nome de teste que não descreve o que testa**
  - O que é: test_batch_counts_invalid_transaction_as_error simula falha do orquestrador com item válido.
  - Por que existe: sem registro no repositório.
  - Evidência: tests/test_api.py.
  - Efeito no TCC: o nome não deve ser citado como prova de tratamento de item inválido.

## 6. Explicabilidade (top-3)

- **Z-score não explica a decisão**
  - O que é: mostra o quanto a transação foge da média do treino, não o que pesou na decisão do modelo.
  - Por que existe: o cálculo usa só feature_stats.json, sem consultar os modelos.
  - Evidência: src/ferramentas/explicabilidade.py.
  - Efeito no TCC: o top-3 deve ser apresentado como desvio do padrão, não como atribuição causal.
- **Tipos raros**
  - O que é: quando a transação é de um tipo raro, esse tipo tende a entrar no top-3 por ter z alto.
  - Por que existe: nas colunas one-hot, a média baixa gera z alto (por exemplo, cerca de +7,66 em type_EMPRESTIMO).
  - Evidência: src/preprocessamento/feature_stats.json.
  - Efeito no TCC: esse item indica raridade, não necessariamente risco.
- **Cauda longa de amount**
  - O que é: valores comuns ficam com z pequeno.
  - Por que existe: o desvio (8.777) é muito maior que a média (1.567), e log_amount não é candidata.
  - Evidência: src/preprocessamento/feature_stats.json; src/ferramentas/explicabilidade.py.
  - Efeito no TCC: o valor da transação raramente se destaca, exceto em valores altos.
- **Cache de feature_stats.json**
  - O que é: só atualiza ao reiniciar a API.
  - Por que existe: o carregamento usa lru_cache.
  - Evidência: src/api/main.py:150.
  - Efeito no TCC: depois de regenerar as estatísticas, é preciso reiniciar a API.
- **--grade-if-lstm reescreve as estatísticas**
  - O que é: esse modo reescreve feature_stats.json e não foi protegido.
  - Por que existe: a proteção da Etapa 3 cobre só o treino padrão.
  - Evidência: src/modelos/treinamento/treinar_modelos.py:3396; commit 5712e27.
  - Efeito no TCC: as estatísticas podem mudar se esse modo for executado de novo.
- **Validação do top-3**
  - O que é: validado só com testes e dados sintéticos.
  - Por que existe: sem chamadas ao Gemini e sem avaliação com usuários.
  - Evidência: tests/test_explicabilidade.py; tests/test_api.py.
  - Efeito no TCC: não há medida da utilidade do top-3 para quem analisa as transações.
- **Só na análise individual**
  - O que é: o lote não calcula top-3.
  - Por que existe: a resposta do lote não tem campo de explicabilidade.
  - Evidência: src/api/main.py; tests/test_api.py (test_lote_nao_calcula_top3).
  - Efeito no TCC: o lote não explica suas decisões.

## 7. Dashboard

- **Não chama a API**
  - O que é: usa o orquestrador direto, então não mostra o top-3.
  - Por que existe: sem registro no repositório.
  - Evidência: src/dashboard/app.py:139-150.
  - Efeito no TCC: o top-3 não pode ser demonstrado pelo dashboard.
- **Justificativa vazia e nomes técnicos**
  - O que é: não trata justificativa vazia e mostra nomes técnicos de colunas e tipos.
  - Por que existe: sem registro no repositório.
  - Evidência: src/dashboard/app.py:326, :489-493, :588 e :605.
  - Efeito no TCC: a tela é menos clara para quem não conhece os dados.
- **Revisão adiada**
  - O que é: a revisão do dashboard fica para depois da Etapa 3.
  - Por que existe: decisão do projeto.
  - Evidência: docs/REGRAS_VIGENTES.md, seção 9 (local, fora do git).
  - Efeito no TCC: o dashboard não faz parte do que foi validado nesta etapa.

## 8. Repositório e processo

- **Chave Gemini antiga no histórico**
  - O que é: a chave antiga aparece no histórico do git. Foi apagada no Google Cloud, segundo o dev; a limpeza do histórico fica para depois.
  - Por que existe: o arquivo .env foi versionado no início do projeto.
  - Evidência: docs/etapas/etapa2.txt, seção 6, que diz "revogada".
  - Efeito no TCC: o histórico precisa de limpeza antes de ser tratado como concluído.
