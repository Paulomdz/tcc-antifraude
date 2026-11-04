# Registro de mudanças
2026-11-02 22:24 | src/modelos/treinamento/treinar_modelos.py | Incluído o campo feature_count_decision no relatório de run_ablacao() | Registrar no relatório a decisão do LSTM com 16 features | aprovado: não; será removida
2026-11-02 22:34 | src/modelos/treinamento/treinar_modelos.py | Removido o campo feature_count_decision do relatório de run_ablacao() | Regra 13: decisões não ficam no código | aprovado por: ok do usuário em 02/11/2026
2026-11-02 hora não registrada | src/modelos/treinamento/treinar_modelos.py; tests/test_treinamento.py | Registro retroativo: implementados o modo --ablacao (run_ablacao) e os testes de ablação; versionados em 53e7514 | Rodar a ablação sem a rotina padrão
2026-11-02 hora não registrada | tests/test_modelos.py | Registro retroativo: adicionado test_lstm_janela_so_com_passado_estrito; versionado em 53e7514 | Verificar que a janela da LSTM usa só o passado | aprovação não registrada
2026-11-02 hora não registrada | reports/etapa2_grid_v2.csv | Registro retroativo: fim de linha convertido de CRLF para LF; FALHA DE PROCESSO: feita sem aprovação e sem registro | Motivo não registrado; sem evidência no git; relato do usuário | aprovado: não
2026-11-02 hora não registrada | reports/etapa2_grid_diagnostico_v1.csv | Registro retroativo: renomeação de reports/etapa2_grid.csv relatada pelo usuário, sem evidência no git | Separar o diagnóstico da política antiga | aprovação não registrada
2026-11-03 | src/modelos/treinamento/treinar_modelos.py | Caminho padrão da grade passa a ser etapa2_grid_v2.csv | Grade v2 é a usada na Etapa 2
2026-11-03 | docs/REGRAS_VIGENTES.md | Seção 13 passa a registrar mudanças sem campo de aprovação | Regra vale para qualquer documento
2026-11-03 | tests/test_treinamento.py | Teste de checkpoint do smoke test fixa a RAM disponível; novo teste cobre o status skipped_memory | Falha intermitente quando a RAM da máquina ficava abaixo de 2 GiB
2026-11-03 | src/ferramentas/explicabilidade.py; src/api/main.py; tests/test_api.py; tests/test_explicabilidade.py | top3 preenchido por z-score na análise individual | Explicabilidade da Etapa 3
2026-11-04 | src/modelos/treinamento/treinar_modelos.py; tests/test_treinamento.py | Treino padrão encerra se a grade v2 existir, salvo com --refazer-grade | Evitar renomear ou alterar a grade v2 e reescrever feature_stats.json por engano
2026-11-04 | src/api/main.py; tests/test_api.py | oldbalanceOrg passa a ser obrigatório na entrada da API | Sem saldo, a razão valor/saldo explodia e distorcia o score e o top-3
