# ETAPA 4 — AVALIAÇÃO DO SISTEMA NO MODO FAST

## Feito
- Script src/avaliacao/avaliar_scores.py (commit 77fb960): `python -m src.avaliacao.avaliar_scores --split validacao|teste`; grava reports/etapa4_avaliacao_<split>.json.
- Parâmetro aplicar_regra_madrugada em judge_tool e orchestrate_ml_only, padrão True; com False, desliga o bloqueio imediato e o bônus de 0,25 da madrugada. API, dashboard e lote seguem com o padrão.
- Sistema A: o modo fast como roda hoje, por orchestrate_ml_only, uma transação por vez e sem histórico; a LSTM recebe uma janela 5 × 17 de zeros e a GNN, um nó sem arestas.
- Sistema B: os mesmos modelos com histórico, em lote, com as funções de avaliação da Etapa 2: até 5 transações anteriores da mesma conta de origem na LSTM; subgrafo só com arestas do passado, fanouts (10, 5), na GNN. Juiz igual ao de A.
- Seis comparações: A_com_regra, A_sem_regra, B_com_regra, B_sem_regra, regra_banco (valor ≥ 90% do saldo anterior e ≥ R$ 5.000 → BLOQUEADO) e if_sozinho (limiar salvo do IF, 0,5152).
- Duas leituras: estrita (BLOQUEADO = fraude) e ampla (BLOQUEADO ou REVISÃO = fraude).
- Validação: 30.677 linhas (step 550 a 599); B completo e A em amostra de 2.000 (seed 42).
- Teste lido uma vez, no commit 77fb960: 87.467 linhas (step 600 a 743), 4.921 fraudes; 10.169 s de execução; pico de 2,22 GiB de RAM e 1,74 GiB de VRAM; nenhum módulo de LLM carregado.

## Decisões
- A é o resultado principal, por ser o sistema entregue; B mostra os modelos com o histórico usado no treino.
- B em lote, com as mesmas funções da Etapa 2, para reproduzir os scores dos modelos.
- A regra do banco existe só na avaliação; o sistema não a usa.
- Pesos (0,35 / 0,30 / 0,35) e limites (0,50 / 0,80) do juiz sem alteração; nenhum modelo treinado.
- Validação usada só para conferir o script, porque os modelos finais treinaram com step < 600, que inclui a validação.
- Precisão por tipo sem nenhum bloqueio: o JSON grava 0,0; este relatório escreve "indefinida".

## Critérios
| Critério | Comando ou fonte | Resultado |
|---|---|---|
| Suíte não-live com cobertura | .venv/bin/python -m pytest -q -m 'not live' -p no:cacheprovider | 150 passed, 0 failed, 1 deselected, 1 warning; cobertura 89,75% (05/11/2026) |
| PR-AUC dos modelos em B contra a Etapa 2 | reports/etapa4_avaliacao_teste.json (pr_auc_modelos_b) | IF 0,688086 (0,6881); LSTM 0,357662 (0,3577); GNN 0,408056 (0,4081); diferenças < 0,0001 |
| IF igual em A e B | diferenca_max_if_a_b | 0,0 |
| Contagens | matrizes do teste | 87.467 linhas e 4.921 fraudes em cada comparação e na soma dos tipos; só as 3 decisões |
| Sem LLM | modulos_llm_carregados | nenhum |

## Resultados no teste (87.467 transações; 4.921 fraudes)
Decisões × fraude (fraude / legítima):

| Comparação | APROVADO | REVISÃO | BLOQUEADO |
|---|---|---|---|
| A_com_regra | 662 / 76.567 | 1.791 / 5.573 | 2.468 / 406 |
| A_sem_regra | 662 / 76.567 | 2.428 / 5.961 | 1.831 / 18 |
| B_com_regra | 950 / 74.797 | 2.058 / 7.365 | 1.913 / 384 |
| B_sem_regra | 1.070 / 74.967 | 2.978 / 7.579 | 873 / 0 |
| regra_banco | 3.181 / 82.544 | 0 / 0 | 1.740 / 2 |
| if_sozinho | 1.635 / 81.325 | 0 / 0 | 3.286 / 1.221 |

Métricas:

| Comparação | Estrita: precisão / recall / F1 | Ampla: precisão / recall / F1 | PR-AUC do score final | Taxa de REVISÃO |
|---|---|---|---|---|
| A_com_regra | 0,8587 / 0,5015 / 0,6332 | 0,4160 / 0,8655 / 0,5619 | 0,7653 | 8,42% |
| A_sem_regra | 0,9903 / 0,3721 / 0,5409 | 0,4160 / 0,8655 / 0,5619 | 0,8323 | 9,59% |
| B_com_regra | 0,8328 / 0,3887 / 0,5301 | 0,3388 / 0,8069 / 0,4773 | 0,6629 | 10,77% |
| B_sem_regra | 1,0000 / 0,1774 / 0,3013 | 0,3369 / 0,7826 / 0,4710 | 0,7038 | 12,07% |
| regra_banco | 0,9989 / 0,3536 / 0,5223 | igual à estrita | sem score | 0% |
| if_sozinho | 0,7291 / 0,6678 / 0,6971 | igual à estrita | 0,6881 | 0% |

Por tipo (recall na leitura ampla / precisão na leitura estrita; "indefinida" quando nada do tipo foi bloqueado):

| Tipo | Transações | Fraudes | A_com_regra | B_com_regra | regra_banco |
|---|---|---|---|---|---|
| BOLETO | 4.969 | 77 | 0,0390 / indefinida | 0,0390 / indefinida | 0,0000 / indefinida |
| CARTAO_ECOMMERCE | 15.080 | 1.132 | 0,7889 / 1,0000 | 0,6290 / 1,0000 | 0,0203 / 1,0000 |
| CARTAO_PRESENCIAL | 13.695 | 60 | 0,4667 / indefinida | 0,0167 / indefinida | 0,0000 / indefinida |
| EMPRESTIMO | 1.474 | 35 | 0,0571 / 0,0000 | 0,0286 / indefinida | 0,0000 / indefinida |
| PIX_AUTOMATICO | 3.018 | 61 | 0,2623 / 1,0000 | 0,6066 / indefinida | 0,0656 / 1,0000 |
| PIX_P2P | 18.216 | 1.791 | 0,9112 / 1,0000 | 0,9079 / 1,0000 | 0,5031 / 1,0000 |
| PIX_QRCODE | 2.849 | 77 | 0,5065 / 1,0000 | 0,7532 / 1,0000 | 0,1429 / 1,0000 |
| PIX_SAQUE | 7.028 | 1.058 | 0,9896 / 1,0000 | 0,9698 / 1,0000 | 0,5265 / 1,0000 |
| PIX_TROCO | 4.856 | 421 | 0,9952 / 1,0000 | 0,9976 / 1,0000 | 0,5796 / 1,0000 |
| SAQUE | 4.513 | 114 | 0,8246 / indefinida | 0,2368 / indefinida | 0,0000 / indefinida |
| TED | 5.945 | 58 | 0,8448 / 0,0025 | 0,4138 / 0,0000 | 0,0000 / 0,0000 |
| TEF | 5.824 | 37 | 1,0000 / indefinida | 0,9730 / 1,0000 | 0,0000 / indefinida |

## Leitura dos resultados
- A_com_regra leva 86,5% das fraudes (4.259 de 4.921) para REVISÃO ou BLOQUEADO, com 8,4% das transações em REVISÃO.
- O PR-AUC do score final de A (0,7653 com a regra; 0,8323 sem) supera o do IF sozinho (0,6881).
- Na leitura estrita, o IF sozinho tem o maior F1 (0,6971).
- Em A, a regra da madrugada bloqueia 637 fraudes e 388 legítimas a mais. Em TED, A_com_regra bloqueia 1 fraude e 399 legítimas; sem a regra, 1 e 18.
- B ficou abaixo de A em PR-AUC do score final, recall e F1, com e sem a regra. Causa não verificada; hipótese: com histórico, a LSTM (PR-AUC 0,3577) e a GNN (0,4081) têm desempenho baixo e diluem o IF (0,6881) no score final.
- A regra do banco tem precisão de 0,9989 (1.740 fraudes, 2 legítimas) e recall 0 em 6 dos 12 tipos, coerente com o padrão de esvaziamento de conta do gerador (docs/secao_dataset_tcc.txt:108-111).
- Os números valem para esta base sintética, com uma semente e um split; tipos com menos de 100 fraudes têm números instáveis.

## Limitações
- No sistema entregue, a LSTM recebe uma janela de zeros (score constante 0,8217) e a GNN roda sem grafo; a correção fica como melhoria futura.
- B abaixo de A, sem causa verificada.
- A regra do banco usa o mesmo padrão que o gerador colocou nas fraudes.
- O JSON grava 0,0 como precisão por tipo quando ela é indefinida.
- Validação otimista: os modelos finais treinaram com ela.
- O log do teste traz três avisos de falta de memória da GPU no cálculo de B, sem efeito nos scores.
- 7 dos 12 tipos têm menos de 100 fraudes no teste.
- O modo production (juiz Gemini) segue sem avaliação.
- Lista completa e atualizada: docs/LIMITACOES.md.
