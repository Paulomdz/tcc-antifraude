# Regras do sistema
Documento vivo: atualizar a cada mudança que crie, altere ou remova uma regra do sistema. Regras de trabalho do agente ficam em docs/REGRAS_VIGENTES.md (local).

## 1. Dados e split

- **Dataset**
  - Valor ou comportamento: dataset/dataset_pix_banco.csv, MD5 cfa6ae1176de2fd46466651fcc0663c7, 454.117 linhas.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:20.
  - Origem: Etapa 1 (docs/etapas/etapa1.txt).
- **Split temporal**
  - Valor ou comportamento: treino step < 550; validação 550 a 599; treino completo step < 600; teste step >= 600.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:263-266.
  - Origem: D3.
- **Uso único do teste**
  - Valor ou comportamento: o teste é lido depois da configuração fixa e não escolhe nada. --ablacao e avaliar_scores.py --split teste recusam nova leitura; --final-if-lstm e --final-gnn podem reler se forem executados outra vez.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:1598 (recusa); :1499 e :1942 (leitura nos modos finais); src/avaliacao/avaliar_scores.py:463 e :443-453.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 1).
- **Estatísticas**
  - Valor ou comportamento: feature_stats.json usa só step < 600; outro conteúdo gera erro.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:282 e :294-295.
  - Origem: Etapa 1 (docs/etapas/etapa1.txt, T1.8).

## 2. Features

- **Vetor do Isolation Forest**
  - Valor ou comportamento: 18 features em ordem fixa: amount, is_incoming, 12 colunas type_*, hora_norm, amount_to_balance_ratio, log_amount, is_madrugada.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:46.
  - Origem: Etapa 1 (docs/etapas/etapa1.txt).
- **Fórmulas**
  - Valor ou comportamento: is_incoming = 1 se nameOrig começa com B; hora_norm = (step % 24) / 23; amount_to_balance_ratio = amount / (|oldbalanceOrg| + 1); log_amount = log1p(amount).
  - Onde está no código: src/preprocessamento/carregar_dataset.py:127, :139, :143, :144 e :168-183.
  - Origem: D2 e D5; razão e log: Etapa 1.
- **Removidas por vazamento**
  - Valor ou comportamento: emptied_origin, balance_mismatch_orig e balance_mismatch_dest não são usadas, por dependerem de saldo pós-transação.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:43-45.
  - Origem: Etapa 1 (docs/etapas/etapa1.txt, "Correção de vazamento de features").
- **IDs proibidos como feature**
  - Valor ou comportamento: nameOrig e nameDest servem só para agrupar histórico e formar o grafo. A única informação derivada é a letra usada em is_incoming.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:67-68.
  - Origem: D5.
- **Features da LSTM e da GNN**
  - Valor ou comportamento: a LSTM usa 17 features, sem amount; a GNN usa 4.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:69 e :77.
  - Origem: Etapa 1.
- **Ablação**
  - Valor ou comportamento: retira só amount_to_balance_ratio: IF com 17 features e LSTM com 16 (16 só na ablação).
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:104-108.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 4).

## 3. Modelos e seleção

- **Critério de seleção**
  - Valor ou comportamento: PR-AUC na validação.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:564 e :1011.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 1).
- **Limiar de decisão de cada modelo**
  - Valor ou comportamento: maior F1 na validação.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:364, :1261, :1353 e :1872.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 1).
- **Contamination do IF na grade**
  - Valor ou comportamento: para cada par de n_estimators e max_samples, fica a contamination de maior F1 na validação, com limiar fixo de 0,5. Entre esses candidatos, vence o de maior PR-AUC na validação; o F1 desempata.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:352 (limiar 0,5), :715 (métricas) e :773-791 (seleção).
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 1); docs/REGRAS_VIGENTES.md, seção 4 (local, fora do git).
- **Configuração final do IF**
  - Valor ou comportamento: n_estimators=100, max_samples=auto, contamination=0,056.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:82.
  - Origem: grade da Etapa 2 (reports/etapa2_grid_v2.csv).
- **Configuração final da LSTM**
  - Valor ou comportamento: hidden_size=32, dropout=0,1, lr=1e-3, num_layers=2.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:87.
  - Origem: grade da Etapa 2 (reports/etapa2_grid_v2.csv).
- **Configuração final da GNN**
  - Valor ou comportamento: hidden_dim=32, dropout=0,1, lr=1e-3, num_layers=2, batch 4096, fanouts (10, 5), até 10 épocas, paciência 2.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:72, :75, :78, :79 e :96.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 3). Na Etapa 5 o modelo foi retreinado com o amostrador corrigido e todo o treino core (335.973 sementes, step < 550); melhor época 10, PR-AUC de validação 0,3464 (reports/etapa5_retreino_gnn.json).
- **Amostragem da GNN (dois saltos)**
  - Valor ou comportamento: os vizinhos de uma transação são as transações anteriores (step menor) da mesma conta de origem e da mesma conta de destino; fanouts 10 no primeiro salto e 5 no segundo; só o passado entra.
  - Onde está no código: src/modelos/arquitetura_gnn.py:174-194 (_sample_nodes), corrigido na Etapa 5 (linha 181).
  - Origem: Etapa 5.
- **Histórico da GNN por transação**
  - Valor ou comportamento: sem sequence explícita, a GNN monta o subgrafo do passado a partir do Parquet processado, com o mesmo amostrador do lote; o índice é montado uma vez e fica em cache. Na validação o limite é step < 600.
  - Onde está no código: src/ferramentas/historico.py (grafo_do_passado, GrafoPassado.subgrafo); src/ferramentas/inferencia_modelos.py (gnn_identity_score).
  - Origem: Etapa 5.
- **Histórico da LSTM por transação**
  - Valor ou comportamento: sem sequence explícita, a LSTM recebe as 5 transações anteriores (step menor) da mesma conta de origem, em ordem de step, com zeros à esquerda quando há menos de 5; conta vazia não tem histórico. É a mesma janela do lote (TransactionSequenceDataset), lida do mesmo índice da GNN.
  - Onde está no código: src/ferramentas/historico.py (GrafoPassado.janela_lstm); src/ferramentas/inferencia_modelos.py (lstm_sequence_score com sequence=None); src/ferramentas/ferramentas_agentes.py (temporal_specialist_tool).
  - Origem: Etapa 5.
- **Modelos carregados uma vez**
  - Valor ou comportamento: IF, LSTM e GNN são lidos do disco na primeira chamada e ficam em memória; o arquivo é lido de novo se mudar. Um arquivo ausente não fica em cache.
  - Onde está no código: src/ferramentas/inferencia_modelos.py (_modelo_em_cache).
  - Origem: Etapa 5.
- **Prova de equivalência na validação**
  - Valor ou comportamento: 2.000 transações da validação (seed 42), só step < 600; diferença máxima abaixo de 0,001 entre o sistema por transação e o lote B para IF, LSTM e GNN.
  - Onde está no código: src/avaliacao/equivalencia.py; reports/etapa5_equivalencia_validacao.json.
  - Origem: Etapa 5.

## 4. Agregação e decisão

- **Pesos do modo fast**
  - Valor ou comportamento: 0,35 comportamental, 0,30 temporal, 0,35 identidade.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:121.
  - Origem: sem registro no repositório.
- **Limites de decisão do modo fast**
  - Valor ou comportamento: score >= 0,80 Recusada; >= 0,50 Revisão Humana necessária; abaixo disso, Aprovada.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:123-128.
  - Origem: sem registro no repositório.
- **Mapeamento na API**
  - Valor ou comportamento: Aprovada → APROVADO; Revisão Humana necessária → REVISÃO; Recusada → BLOQUEADO. Os rótulos finais são aceitos como vieram; valor desconhecido gera erro.
  - Onde está no código: src/api/main.py:61.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).

## 5. Regras rígidas e de horário

- **is_madrugada**
  - Valor ou comportamento: hora de 0 a 5, com hora = step % 24.
  - Onde está no código: src/preprocessamento/carregar_dataset.py:145, :184 e :236.
  - Origem: Etapa 1 (docs/etapas/etapa1.txt).
- **Regra noturna do Banco Central (escopo B)**
  - Valor ou comportamento: das 20h às 6h (hora = step % 24), Pix entre pessoas (PIX_P2P), Pix QR Code, Pix Automático e TED de pessoa física para outra pessoa física têm limite de R$ 1.000,00 somados por conta pagadora e por noite. A transação que faz o total passar de R$ 1.000 vai, no mínimo, para Revisão Humana necessária. A regra não bloqueia, não muda o score e não rebaixa uma Recusada. Pix saque e troco ficam fora.
  - Onde está no código: src/ferramentas/regra_noturna.py; src/ferramentas/ferramentas_agentes.py:131-133 (modo fast) e aplicar_piso_noturno; src/orquestacao/fluxo_multiagente.py:78 e :97-101 (production e deep).
  - Origem: IN BCB 331/2022, alterada pelas IN BCB 512/2024 e 669/2025 (limite de R$ 1.000 e janela das 20h às 6h). Revisão como consequência, escopo B e total acumulado só com transações entre pessoas físicas: decisões do projeto em 06/11/2026, tomadas depois da primeira leitura do teste (Etapa 4), que usou a regra antiga.
- **Total noturno acumulado**
  - Valor ou comportamento: soma das transações elegíveis do mesmo pagador com step menor, desde as 20h em que começou a noite. Transações do mesmo step não se somam entre si. O histórico vem do Parquet processado, pelo mesmo índice da GNN.
  - Onde está no código: src/ferramentas/regra_noturna.py (total_noturno_anterior e, em lote, marcar_lote); src/ferramentas/historico.py.
  - Origem: Etapa 5.
- **Pessoa física pelo prefixo do ID**
  - Valor ou comportamento: ID começando com C é pessoa física; o recebedor precisa ser diferente do pagador. MEI não pode ser separado.
  - Onde está no código: src/ferramentas/regra_noturna.py (elegivel e marcar_elegiveis).
  - Origem: inferência a partir dos tipos de cada prefixo (Etapa 5); o significado dos prefixos não está documentado no gerador.
- **Regra antiga (removida na Etapa 5)**
  - Valor ou comportamento: até a Etapa 4, R$ 10.000 ou mais entre 0h e 5h em PIX_P2P, PIX_AUTOMATICO, TED ou TEF era bloqueado direto (Recusada, score 0,95), e nos outros tipos somava 0,25 ao score. O agente temporal acrescentava um "alerta crítico" à explicação.
  - Onde estava no código: src/ferramentas/ferramentas_agentes.py até o commit 3cbbcc5.
  - Origem: sem registro no repositório.
- **Textos dos agentes e do juiz**
  - Valor ou comportamento: os agentes e o juiz Gemini descrevem a regra noturna. O juiz recebe na tarefa se a regra se aplica, e o orquestrador aplica o mesmo piso de revisão nos modos production e deep.
  - Onde está no código: src/orquestacao/definicoes_agentes.py (especialistas e juiz_final); src/ferramentas/ferramentas_agentes.py (llm_judge_tool e deep_multiagent_tool).
  - Origem: Etapa 5.
- **Texto do dashboard**
  - Valor ou comportamento: a tela descreve a regra noturna em linguagem simples, junto com os limiares 0,50 e 0,80, e a ajuda do campo de hora cita a janela das 20h às 6h; o dashboard não calcula a regra.
  - Onde está no código: src/dashboard/app.py:74.
  - Origem: Etapa 5.
- **Parâmetro aplicar_regra_madrugada**
  - Valor ou comportamento: padrão True, que aplica a regra noturna; False a desliga. API, dashboard e lote usam o padrão; False é usado só na avaliação. O nome foi mantido por compatibilidade com a avaliação da Etapa 4.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:104 e :131; src/orquestacao/fluxo_multiagente.py:42 e :52.
  - Origem: Etapa 4, commit 77fb960; significado alterado na Etapa 5.

## 6. Modos de execução

- **fast**
  - Valor ou comportamento: modelos e agregação determinística, sem LLM. Usado em /analise_lote, /analise_lote/arquivo, em /analise_transacao?modo=fast e no dashboard (lote e aba "Aprenda Simulando").
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:41; src/api/main.py:293 e :338; src/dashboard/app.py:686.
  - Origem: D8.
- **production**
  - Valor ou comportamento: modelos e juiz Gemini. Padrão de /analise_transacao (sem o parâmetro modo). No dashboard, só o lote em modo production usa este modo: chama /analise_transacao uma vez por transação, com limite de 10 por lote (a aba "Aprenda Simulando" usa fast).
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:62 e :85; src/api/main.py:268; src/dashboard/app.py:571 e :69.
  - Origem: D8.
- **deep**
  - Valor ou comportamento: quatro agentes Gemini por transação (três especialistas e o juiz final, em sequência); exige chave e consome cota. Disponível em /analise_transacao?modo=deep e no lote deep do dashboard, com limite de 5 transações por lote.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:92; src/ferramentas/ferramentas_agentes.py:301; src/api/main.py:268 e :295; src/dashboard/app.py:70 e :571.
  - Origem: D8 (função); endpoint e lote no dashboard em 06/11/2026.
- **Sem fallback determinístico**
  - Valor ou comportamento: production não troca para decisão determinística.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:240.
  - Origem: D7.

## 7. API

- **Endpoints**
  - Valor ou comportamento: GET /; GET /config; POST /analise_transacao; POST /analise_lote; POST /analise_lote/arquivo; GET /historico; GET /historico/estatisticas.
  - Onde está no código: src/api/main.py:238, :243, :254, :297, :373, :420 e :429.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).
- **503 sem chave**
  - Valor ou comportamento: /analise_transacao com corpo válido e sem GOOGLE_API_KEY responde 503 com a mensagem do D7.
  - Onde está no código: src/api/main.py:37.
  - Origem: D7.
- **Limite do lote**
  - Valor ou comportamento: até 500 transações; acima disso, 422 com "Lote máximo de 500 transações.".
  - Onde está no código: src/api/main.py:39.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).
- **Upload**
  - Valor ou comportamento: CSV com vírgula ou TXT com ponto e vírgula, com colunas obrigatórias; isFraud é opcional.
  - Onde está no código: src/ferramentas/leitor_arquivo_lote.py:9 e :43.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).
- **Amostragem do upload**
  - Valor ou comportamento: seed 42; quantidade de 1 a 500, padrão 300.
  - Onde está no código: src/api/main.py:40 e :377.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).
- **oldbalanceOrg obrigatório**
  - Valor ou comportamento: sem o campo, 422; no lote, 422 para o lote inteiro; saldo 0 explícito é aceito.
  - Onde está no código: src/api/main.py:80.
  - Origem: Etapa 3, commit e2f1290.
- **/config**
  - Valor ou comportamento: informa disponibilidade do LLM, modelo e limite, sem expor a chave.
  - Onde está no código: src/api/main.py:243-247.
  - Origem: Etapa 2 (docs/etapas/etapa2_api.txt).
- **/historico**
  - Valor ou comportamento: funciona sem chave; limit de 1 a 1000, padrão 100.
  - Onde está no código: src/api/main.py:420-422.
  - Origem: D7.

## 8. Explicabilidade

- **Top-3 por z-score**
  - Valor ou comportamento: z = (valor - média) / desvio, com feature_stats.json; ranking por |z|; empate na ordem de IF_FEATURE_NAMES.
  - Onde está no código: src/ferramentas/explicabilidade.py:19.
  - Origem: Etapa 3, commit a608656.
- **17 candidatas**
  - Valor ou comportamento: as 18 features menos log_amount.
  - Onde está no código: src/ferramentas/explicabilidade.py:8.
  - Origem: Etapa 3, commit a608656.
- **Direção**
  - Valor ou comportamento: acima, abaixo ou na_media.
  - Onde está no código: src/ferramentas/explicabilidade.py:11.
  - Origem: Etapa 3, commit a608656.
- **desvio_zero**
  - Valor ou comportamento: desvio 0 retorna z = 0 e desvio_zero = true.
  - Onde está no código: src/ferramentas/explicabilidade.py:28.
  - Origem: Etapa 3, commit a608656.
- **Falha nas estatísticas**
  - Valor ou comportamento: responde 200 com top3 vazio e registra o erro no log.
  - Onde está no código: src/api/main.py:154.
  - Origem: Etapa 3, commit a608656.
- **Só na análise individual**
  - Valor ou comportamento: o lote não calcula top-3.
  - Onde está no código: src/api/main.py:200.
  - Origem: Etapa 3, commit a608656.

## 9. Treino

- **Modos dedicados**
  - Valor ou comportamento: --smoke-only, --grade-if-lstm (com --primeiro-passe e --grid-path), --final-if-lstm, --final-gnn e --ablacao.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:3483-3514.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 7).
- **Proteção da grade v2**
  - Valor ou comportamento: sem flag, o treino encerra antes de main() se a grade existir.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:67 e :3583.
  - Origem: Etapa 3, commit 5712e27.
- **--refazer-grade**
  - Valor ou comportamento: autoriza o treino padrão e não pode ser combinada com outras flags.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:3519 e :3533.
  - Origem: Etapa 3, commit 5712e27.
- **Ablação só com --ablacao**
  - Valor ou comportamento: IF e LSTM sem a razão, sem GNN; recusa nova execução se o relatório registrar leitura do teste.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:1594-1598.
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 4).

## 10. Gemini

- **Modelo**
  - Valor ou comportamento: lido de GEMINI_MODEL; padrão gemini-3.5-flash-lite.
  - Onde está no código: src/orquestacao/definicoes_agentes.py:24 e :33.
  - Origem: Etapa 0 (docs/etapas/etapa0.txt, seção 0.4).
- **Production**
  - Valor ou comportamento: uma execução do juiz por transação.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:287-288.
  - Origem: D8.
- **fast nunca chama**
  - Valor ou comportamento: usa só judge_tool, sem LLM.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:46.
  - Origem: D8.

## 11. Avaliação

- **Comando**
  - Valor ou comportamento: python -m src.avaliacao.avaliar_scores --split validacao ou --split teste; grava reports/etapa4_avaliacao_<split>.json.
  - Onde está no código: src/avaliacao/avaliar_scores.py:430 e :610.
  - Origem: Etapa 4, commit 77fb960.
- **Sistemas e comparações**
  - Valor ou comportamento: A usa orchestrate_ml_only, sem histórico; B calcula os scores com histórico, em lote, com as funções da Etapa 2, e passa pelo mesmo juiz. Seis comparações: A e B com e sem a regra da madrugada, regra_banco e if_sozinho.
  - Onde está no código: src/avaliacao/avaliar_scores.py:203, :288, :312 e :340.
  - Origem: Etapa 4.
- **Regra do banco**
  - Valor ou comportamento: amount ≥ 0,9 × oldbalanceOrg e amount ≥ 5.000 → BLOQUEADO; só na avaliação.
  - Onde está no código: src/avaliacao/avaliar_scores.py:79-82.
  - Origem: docs/secao_dataset_tcc.txt:103-106; implementada na Etapa 4.
- **Leituras e precisão sem bloqueio**
  - Valor ou comportamento: estrita (BLOQUEADO = fraude) e ampla (BLOQUEADO ou REVISÃO = fraude); sem nenhum positivo previsto, a precisão é gravada como 0,0.
  - Onde está no código: src/avaliacao/avaliar_scores.py:94-107.
  - Origem: Etapa 4.
- **Amostra de A na validação**
  - Valor ou comportamento: 2.000 transações, seed 42; no teste, A é completo.
  - Onde está no código: src/avaliacao/avaliar_scores.py:40-41 e :512.
  - Origem: Etapa 4.
- **Uma leitura do teste**
  - Valor ou comportamento: se reports/etapa4_avaliacao_teste.json existir, encerra sem ler nada. Antes da leitura, confere artefatos, limiar do IF e RAM (≥ 2 GiB) e reserva o relatório com criação exclusiva (status em_andamento). Uma falha grava status falhou, e o arquivo continua bloqueando nova execução.
  - Onde está no código: src/avaliacao/avaliar_scores.py:413-427, :443-453, :463-467 e :475-485.
  - Origem: Etapa 4.
- **Sem LLM**
  - Valor ou comportamento: durante a execução, a chave fica vazia e as funções do juiz Gemini e do modo deep recusam qualquer chamada; o relatório lista módulos de LLM carregados.
  - Onde está no código: src/avaliacao/avaliar_scores.py:381 e :389-410.
  - Origem: Etapa 4.
