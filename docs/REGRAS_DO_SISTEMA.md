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
  - Valor ou comportamento: o teste é lido depois da configuração fixa e não escolhe nada. Só --ablacao recusa nova leitura; --final-if-lstm e --final-gnn podem reler se forem executados outra vez.
  - Onde está no código: src/modelos/treinamento/treinar_modelos.py:1598 (recusa); :1499 e :1942 (leitura nos modos finais).
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
  - Origem: Etapa 2 (docs/etapas/etapa2.txt, seção 3).

## 4. Agregação e decisão

- **Pesos do modo fast**
  - Valor ou comportamento: 0,35 comportamental, 0,30 temporal, 0,35 identidade.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:145.
  - Origem: sem registro no repositório.
- **Limites de decisão do modo fast**
  - Valor ou comportamento: score >= 0,80 Recusada; >= 0,50 Revisão Humana necessária; abaixo disso, Aprovada.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:152-157.
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
- **Alerta do agente temporal**
  - Valor ou comportamento: madrugada e amount >= R$ 10.000 acrescentam um alerta crítico à explicação.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:61-66.
  - Origem: sem registro no repositório.
- **Bloqueio imediato do modo fast**
  - Valor ou comportamento: madrugada, amount >= R$ 10.000 e tipo PIX_P2P, PIX_AUTOMATICO, TED ou TEF (variável is_tipo_bloqueio) geram Recusada com score 0,95.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:121-133.
  - Origem: sem registro no repositório para o limite; tipos ajustados na Etapa 3, commit 03227b9.
- **Bônus de 0,25**
  - Valor ou comportamento: madrugada e amount >= R$ 10.000 somam 0,25 ao score, limitado a 1,0.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:148-149.
  - Origem: sem registro no repositório.
- **Texto do juiz**
  - Valor ou comportamento: as regras 1 e 2 citam PIX_P2P, PIX_AUTOMATICO, TED e TEF, com valor >= R$ 10.000 entre 00:00 e 05:59.
  - Onde está no código: src/orquestacao/definicoes_agentes.py:132-133.
  - Origem: Etapa 3, commit 03227b9; o limite: sem registro no repositório.
- **Outras instruções dos agentes**
  - Valor ou comportamento: os textos de outros agentes também citam o limite de R$ 10.000.
  - Onde está no código: src/orquestacao/definicoes_agentes.py:68, :81, :90, :110 e :123.
  - Origem: sem registro no repositório.
- **Texto do dashboard**
  - Valor ou comportamento: a tela descreve a regra; o dashboard não a calcula.
  - Onde está no código: src/dashboard/app.py:57.
  - Origem: sem registro no repositório.

## 6. Modos de execução

- **fast**
  - Valor ou comportamento: modelos e agregação determinística, sem LLM. Usado em /analise_lote, /analise_lote/arquivo e no lote do dashboard.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:39; src/api/main.py:319; src/dashboard/app.py:147.
  - Origem: D8.
- **production**
  - Valor ou comportamento: modelos e juiz Gemini. Usado em /analise_transacao e na análise individual do dashboard.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:57 e :77; src/api/main.py:274; src/dashboard/app.py:149.
  - Origem: D8.
- **deep**
  - Valor ou comportamento: quatro agentes LLM; só função de orquestração, sem endpoint.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:83; src/ferramentas/ferramentas_agentes.py:297.
  - Origem: D8.
- **Sem fallback determinístico**
  - Valor ou comportamento: production não troca para decisão determinística.
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:244.
  - Origem: D7.

## 7. API

- **Endpoints**
  - Valor ou comportamento: GET /; GET /config; POST /analise_transacao; POST /analise_lote; POST /analise_lote/arquivo; GET /historico; GET /historico/estatisticas.
  - Onde está no código: src/api/main.py:238, :243, :254, :294, :370, :417 e :426.
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
  - Onde está no código: src/api/main.py:417-419.
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
  - Onde está no código: src/ferramentas/ferramentas_agentes.py:283-284.
  - Origem: D8.
- **fast nunca chama**
  - Valor ou comportamento: usa só judge_tool, sem LLM.
  - Onde está no código: src/orquestacao/fluxo_multiagente.py:42.
  - Origem: D8.
