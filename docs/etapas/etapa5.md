# Etapa 5: histórico no sistema entregue e regra noturna

Status em 06/11/2026: em andamento.
- Feito: histórico nos três modelos (LSTM, GNN e regra noturna); prova de equivalência na validação (2.000 transações); regra noturna do Banco Central; modelos versionados e carregados uma vez; modo deep disponível na API (`?modo=deep`) e no lote do dashboard (até 5 transações).
- Pendente: avaliação no teste com o sistema novo e juiz Gemini.

## 1. Problema encontrado

No modo fast, a transação chegava sem histórico. A LSTM recebia uma janela de zeros e dava sempre 0,8217. A GNN rodava com um único nó.

Ao investigar a GNN, achamos um erro no amostrador: `_sample_nodes` expandia a componente inteira do passado, em vez de dois saltos com fanouts 10 e 5. Uma transação chegava a cerca de 297 mil nós, e a GNN levava cerca de 1,9 s por transação. O modelo anterior foi treinado com essa entrada.

## 2. GNN

- **Correção do amostrador:** `src/modelos/arquitetura_gnn.py:181`. A expansão passa a usar só a fronteira de cada salto. Uma transação tem no máximo 7 nós (1 + 2 + 4).
- **Referência de teste:** `tests/test_modelos.py` usava uma cópia do amostrador antigo, com o mesmo erro; a cópia foi corrigida.
- **Retreino:** `src/modelos/treinamento/retreino_gnn.py`. Usa todo o treino core (335.973 sementes, step < 550) e escolhe a época na validação (550 a 599). Não lê step >= 600. Duração: 55 s.
  - Uma amostra de 100 mil sementes parou na época 1, com PR-AUC 0,2975. Com o treino completo, a PR-AUC chega a 0,3464 na época 10. O modelo final usa o treino completo: a lentidão vinha do amostrador, não do tamanho do treino.
  - Modelo anterior guardado em `src/modelos/modelos_salvos/gnn_model_etapa2_v1.pt`.
- **Índice do passado:** `src/ferramentas/historico.py`. Monta uma vez, por limite de step, o índice das contas de origem e de destino e dos vizinhos anteriores. `grafo_do_passado(600)` é usado na validação; `grafo_do_passado(None)` é o padrão do sistema.
- **Uso no sistema:** `src/ferramentas/inferencia_modelos.py` (`gnn_identity_score`). Sem `sequence` explícita, a GNN monta o subgrafo do passado. Com `sequence`, o comportamento antigo é mantido.

## 2.1 LSTM

- **Janela:** `GrafoPassado.janela_lstm` (`src/ferramentas/historico.py`) devolve as 5 transações anteriores (step menor) da mesma conta de origem, em ordem de step, com zeros à esquerda quando há menos de 5. Conta vazia não tem histórico. É a mesma regra do lote (`TransactionSequenceDataset`), lida do mesmo índice da GNN.
- **Uso no sistema:** o agente temporal (`temporal_specialist_tool`) chama `lstm_sequence_score` com `sequence=None` quando a transação não traz histórico, e a LSTM busca a janela no Parquet. Com `sequence`, o comportamento antigo é mantido.
- **Sem retreino:** a LSTM foi treinada com essa janela; o que faltava era entregá-la no sistema.
- **Modelos em memória:** IF, LSTM e GNN são lidos do disco na primeira chamada e ficam em cache (`_modelo_em_cache` em `src/ferramentas/inferencia_modelos.py`). Antes, cada transação lia os três arquivos.

## 2.2 Prova de equivalência (Fase C)

`python -m src.avaliacao.equivalencia` compara, em 2.000 transações da validação (step 550 a 599, seed 42), as funções que os agentes chamam com o cálculo em lote da Etapa 4 (`calcular_scores_b`). O índice do passado usa step < 600; o teste não é lido. Critério: diferença máxima menor que 0,001. Relatório: `reports/etapa5_equivalencia_validacao.json`.

| Modelo | Diferença máxima | Diferença média | Valores distintos (sistema / lote) | Passou |
|---|---|---|---|---|
| Isolation Forest | 0 | 0 | 1.966 / 1.966 | sim |
| LSTM | 0,000063 | 0,000002 | 1.391 / 1.391 | sim |
| GNN | 0,00000012 | 0,00000001 | 2.000 / 2.000 | sim |

A diferença pequena na LSTM vem de o lote rodar na GPU e o sistema na CPU. Tempo do sistema: 23 ms por transação nos três modelos; o lote leva 1,3 s para as 2.000. Pico de RAM: 1,77 GiB.

## 3. Regra noturna do Banco Central

Fontes:
- **Resolução BCB nº 142, de 23/09/2021, art. 2º, I:** limite de R$ 1.000 por conta, das 20h às 6h, para o total de transações entre pessoas físicas distintas, exceto empresários individuais. Não vale só para Pix: inclui TED, DOC, boletos e transferências entre contas. O cliente pode definir outros limites e cadastrar contas autorizadas (art. 3º).
- **IN BCB nº 185/2021:** limitou a janela a 20h–6h (padrão) ou 22h–6h.
- **IN BCB nº 331/2022, art. 10:** Pix Saque e Pix Troco têm limite de R$ 1.000 à noite; de dia, entre R$ 1.000 e R$ 3.000, salvo pedido expresso do usuário.

Na primeira versão desta etapa, a fonte citada era a IN BCB nº 331/2022, alterada pelas IN BCB nº 512/2024 e nº 669/2025. A fonte do limite noturno entre pessoas físicas é a Resolução BCB nº 142/2021; a IN 331/2022 trata do Pix Saque e do Pix Troco.

Como o projeto aplica a regra (`src/ferramentas/regra_noturna.py`):
- **Noite:** hora >= 20 ou hora < 6, com hora = step % 24 (mesma fonte de is_madrugada). A noite que começa às 20h termina às 6h do dia seguinte.
- **Quem entra:** pagador pessoa física e recebedor pessoa física diferente. Pessoa física é o ID que começa com C. MEI não pode ser separado nos dados.
- **Tipos (escopo B):** PIX_P2P, PIX_QRCODE, PIX_AUTOMATICO e TED. A Resolução também cobre TEF (transferência entre contas), boletos e DOC: TEF e boletos ficam fora, e DOC não existe nos dados. Incluir TEF e boletos somaria 557 revisões com step < 600 (65 fraudes e 492 legítimas, todas de TEF). Pix saque e troco ficam fora, embora a IN 331/2022, art. 10, também os limite à noite.
- **Janela e limite:** fixos, das 20h às 6h e R$ 1.000. Sem a opção das 22h (IN 185/2021), sem limite definido pelo cliente e sem contas autorizadas (art. 3º da Resolução).
- **Total:** acumulado por conta pagadora e por noite, somando só as transações elegíveis com step menor. Transações do mesmo step não se somam entre si.
- **Consequência:** a transação que faz o total passar de R$ 1.000 vai, no mínimo, para Revisão Humana necessária. É escolha do projeto, não da norma. A regra não bloqueia, não muda o score e não rebaixa uma Recusada.
- **Modos:** no fast, dentro de `judge_tool`. Em production e deep, o juiz Gemini recebe na tarefa se a regra se aplica, e o orquestrador aplica o mesmo piso depois da resposta.
- **Removido:** o bloqueio imediato de R$ 10.000 entre 0h e 5h (score 0,95), o bônus de 0,25 e o "alerta crítico" do agente temporal.
- **Textos atualizados:** agentes e juiz em `src/orquestacao/definicoes_agentes.py`; a descrição e a ajuda do campo de hora no dashboard (`src/dashboard/app.py`).

A regra foi revisada depois da primeira leitura do teste, feita na Etapa 4 com a regra antiga.

## 4. Resultados medidos

| Medida | Resultado |
|---|---|
| Tamanho do subgrafo da GNN por transação | antes: média de 297.510 nós; agora: no máximo 7 |
| Tempo da GNN por transação | antes: cerca de 1,9 s; agora: 8,8 ms (200 transações) |
| GNN por transação vs lote B, 200 transações da validação (step 550 a 599, seed 42), só step < 600 | diferença máxima 0,000000 |
| PR-AUC de validação da GNN nova | 0,3464 (época 10, limite de épocas) |
| PR-AUC de validação da GNN anterior | 0,3367 (época 10, com o amostrador antigo) |
| Regra noturna: lote vs uma transação por vez (4.926 transações, step < 600) | 0 divergências |
| Modo fast de ponta a ponta (modelos reais, histórico do Parquet) | 1ª chamada 2,2 s (carrega índice e modelos); depois, de 20 a 30 ms |
| LSTM no modo fast | antes: sempre 0,8217; agora: 1.391 valores distintos em 2.000 transações |
| Modo fast de ponta a ponta com os três modelos com histórico | 1ª chamada 2,3 s; depois, cerca de 22 ms |
| Testes | 272 passed, 0 failed, 1 deselected |
| Cobertura | 91,52% (1.887 instruções, 160 sem teste; configuração padrão do pytest.ini e do .coveragerc) |

A verificação da GNN com 200 transações foi a primeira; a prova completa, com 2.000 transações e os três modelos, está na seção 2.2.

## 5. Regra noturna: contagem (só leitura, step < 600)

Esta contagem usa a regra implementada (`marcar_lote`) e substitui as contagens antigas por transação, que não valem como regra do Banco Central.

| Tipo | Revisões | Fraude | Legítima |
|---|---|---|---|
| PIX_P2P | 4.086 | 4.069 | 17 |
| PIX_QRCODE | 16 | 16 | 0 |
| PIX_AUTOMATICO | 25 | 7 | 18 |
| TED | 2.540 | 31 | 2.509 |
| **Total** | **6.667** (1,82% das 366.650) | **4.123** (61,84%) | **2.544** |

Só na validação (step 550 a 599): 537 revisões, das quais 320 são fraude.

Comparação feita para escolher o escopo, com uma contagem mais simples (soma na ordem das linhas, inclusive no mesmo step):
- Escopo A (com saque e troco): 10.046 revisões, das quais 7.387 são fraude.
- Escopo B: 6.643 revisões, das quais 4.098 são fraude.
- Escopo C (só Pix): 7.506 revisões, das quais 7.355 são fraude.
- Somar no limite também os pagamentos a comerciantes quase não muda o resultado: no escopo B, 6.626 revisões em vez de 6.643.

A base é sintética: esses números não provam desempenho no mundo real.

## 6. Modelos no repositório

Até esta etapa, o `.gitignore` e as regras do projeto deixavam todos os modelos fora do git. Sem eles, o sistema não funciona e os resultados não se reproduzem. Desde 06/11/2026, quatro modelos são versionados; diagnósticos e ablações continuam fora.

| Arquivo | SHA-256 |
|---|---|
| `isolation_forest.pkl` | 645b9ae95c6e0cdbe2281d26ed49258fbe8513736f2ce78f192030e5aaa5c93b |
| `lstm_model.pt` | 615d574a8f987b95c145a8c43f5d68a64bf5e42d39f16135a7dfe7dc24476861 |
| `gnn_model.pt` (Etapa 5) | 35ecbb328d293847c3e58b694c73fdfa25730422c0eb82bf1f66f6fccd3d8fcc |
| `gnn_model_etapa2_v1.pt` (usado na Etapa 4) | 6e4e691a70c9a0898702489e74f0dd312fb6d8a1308b5ee82a7fcdd222a0f09b |

## 7. Limitações ainda abertas

- **Histórico só do Parquet:** contas que não estão no dataset do projeto, ou transações de um CSV enviado ao lote, não ganham histórico com as outras linhas do mesmo arquivo.
- **Leitura do teste:** o índice padrão do sistema (`grafo_do_passado(None)`) carrega a base inteira, inclusive step >= 600. Ele só usa o passado de cada transação, mas a leitura precisa ser declarada no relatório da avaliação no teste.
- **GNN ainda subindo:** a validação sobe na época 10; mais épocas não foram testadas.
- **Regra noturna:** escopo B sem saque e troco; só a transação que cruza o limite vai para revisão; pessoa física inferida pelo prefixo; MEI não separado; muitas revisões de TED legítimas. Detalhes em `docs/LIMITACOES.md`.
- **Textos do juiz:** mudaram sem avaliação com o Gemini.
- **Modo deep:** disponível, mas não avaliado; um lote de 5 transações no dashboard é demonstração, não avaliação.
- **Trecho da norma:** os docs citam os números das instruções normativas; o trecho do texto oficial ainda será anexado.
