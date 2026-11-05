# ETAPA 3 — EXPLICABILIDADE E AJUSTES

## Feito
- Grade: caminho padrão passa a reports/etapa2_grid_v2.csv (commit 0b40751).
- Teste de smoke: RAM fixada por monkeypatch; novo teste cobre skipped_memory (commit fe5ba3a).
- Top-3 na análise individual da API (commit a608656): z-score sobre 17 features (log_amount fora, por ser transformação de amount), com média e desvio de src/preprocessamento/feature_stats.json; retorno com nome, valor, z, direção (acima, abaixo ou na_media) e desvio_zero; top3=[] e erro no log se as estatísticas falharem; o lote não calcula top-3.
- Proteção do treino padrão: sem flag, o treino encerra antes de main() se a grade v2 existir, salvo com --refazer-grade; a flag não pode ser combinada com outras (commit 5712e27).
- oldbalanceOrg passa a ser obrigatório na entrada da API; sem o campo, a resposta é 422 (commit e2f1290).
- Removidos tipos herdados do PaySim (DEBIT, TRANSFER) da regra de bloqueio e do texto do juiz, e a constante STEP_MAX sem uso; variável renomeada para is_tipo_bloqueio (commit 03227b9).

## Decisões
- Z-score contra as estatísticas do treino, ranking por |z|, com direção.
- 18 features candidatas menos log_amount.
- Falha nas estatísticas devolve 200 com top3 vazio, para não perder a resposta já cobrada do Gemini.
- Treino padrão exige --refazer-grade quando a grade v2 existe.
- Saldo ausente gera 422, sem valor estimado, porque um banco sempre conhece o saldo no momento da transação.
- avaliar_scores.py adiado.

## Critérios
| Critério | Comando | Resultado |
|---|---|---|
| Suíte não-live com cobertura | .venv/bin/python -m pytest -q -m 'not live' -p no:cacheprovider | 119 passed, 0 failed, 1 deselected; cobertura 91,18% (05/11/2026) |

## Limitações
- O z-score mostra o quanto a transação foge da média do treino, não o que pesou na decisão do modelo.
- Quando a transação é de um tipo raro, esse tipo tende a entrar no top-3 por ter z alto.
- amount tem cauda longa.
- O cache de feature_stats.json só atualiza ao reiniciar a API.
- --grade-if-lstm também reescreve feature_stats.json; não foi protegido nesta etapa.
- Top-3 validado só com testes e dados sintéticos.
- A regra de linha de base descrita em docs/secao_dataset_tcc.txt não existe no repositório.
- O dashboard não chama a API e não mostra o top-3.
- avaliar_scores.py não foi feito.
- Saldo 0 enviado explicitamente continua aceito, e a razão valor/saldo fica alta nesse caso.
- Sem chave e sem saldo, a API responde 422 antes do 503 do D7.
- No lote, um item sem saldo faz o lote inteiro responder 422, como já acontecia com qualquer item inválido.
- O teste test_batch_counts_invalid_transaction_as_error simula falha do orquestrador com item válido; o nome não descreve o que testa.
- O limite de R$ 10.000 do sistema (agente temporal, agregador do modo fast e instruções dos agentes) é parâmetro de política, não calibrado com os dados. A referência regulatória do Pix é R$ 1.000 por conta entre pessoas físicas, das 20h às 6h (fonte a confirmar na resolução do Banco Central), com janela diferente de is_madrugada (0h a 5h). Com step < 600 (366.650 transações, 5,54% de fraude): de 0h a 5h, amount >= 10.000 reúne 9.365 transações (82,77% de fraude) e amount >= 1.000, 17.276 (68,17%); das 20h às 6h, amount >= 10.000 reúne 10.086 (81,80%) e amount >= 1.000, 20.545 (61,69%). A contagem mostra só cortes de valor e horário em todos os tipos; não separa pessoa física nem limite por conta, então não mede a regra do Pix em si.
- Lista completa e atualizada: docs/LIMITACOES.md.

## Pendências para a fase do dashboard
- CASH_IN em src/dashboard/app.py:248, :516 e :568 (e o teste em tests/test_dashboard.py:38).
- Código sem uso em src/dashboard/app.py: _MAX_UPLOAD_MB, _minutes_to_hh_mm, _format_currency, _is_valid_24h_time.
- Valores padrão de 5.000 e 10.000 do formulário.
- Rótulo "REVISÃO MANUAL" mapeado no dashboard e não na API.
