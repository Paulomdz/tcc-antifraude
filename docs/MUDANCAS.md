# Registro de mudanças
2026-11-02 22:24 | src/modelos/treinamento/treinar_modelos.py | Incluído o campo feature_count_decision no relatório de run_ablacao() | Registrar no relatório a decisão do LSTM com 16 features | aprovado: não; será removida
2026-11-02 22:34 | src/modelos/treinamento/treinar_modelos.py | Removido o campo feature_count_decision do relatório de run_ablacao() | Regra 13: decisões não ficam no código | aprovado por: ok do usuário em 02/11/2026
2026-11-02 hora não registrada | src/modelos/treinamento/treinar_modelos.py; tests/test_treinamento.py | Registro retroativo: implementados o modo --ablacao (run_ablacao) e os testes de ablação; versionados em 53e7514 | Rodar a ablação sem a rotina padrão | aprovado por: Paulo (dev) em 03/11/2026
2026-11-02 hora não registrada | tests/test_modelos.py | Registro retroativo: adicionado test_lstm_janela_so_com_passado_estrito; versionado em 53e7514 | Verificar que a janela da LSTM usa só o passado | aprovação não registrada
hora não registrada | docs/etapas/etapa2.txt; docs/etapas/etapa2_api.txt | Registro retroativo: edições de documentação anteriores a este registro, versionadas em 53e7514 | Fechamento da Etapa 2 | aprovação das edições não registrada; commit pedido pelo usuário
2026-11-02 hora não registrada | reports/etapa2_grid_v2.csv | Registro retroativo: fim de linha convertido de CRLF para LF; FALHA DE PROCESSO: feita sem aprovação e sem registro | Motivo não registrado; sem evidência no git; relato do usuário | aprovado: não
2026-11-02 hora não registrada | reports/etapa2_grid_diagnostico_v1.csv | Registro retroativo: renomeação de reports/etapa2_grid.csv relatada pelo usuário, sem evidência no git | Separar o diagnóstico da política antiga | aprovação não registrada
2026-11-02 19:28 | venv | Registro retroativo: link venv removido do índice do git (git rm --cached), commit 6865eb0 | Link versionado por engano | aprovação não registrada
2026-11-03 01:51 | README.md; docs/etapas/etapa0.txt, etapa1.txt, etapa2.txt, etapa2_api.txt; docs/REGRAS_VIGENTES.md | Documentação da Etapa 2 atualizada
2026-11-03 02:31 | .gitignore | venv (link simbólico) ignorado
