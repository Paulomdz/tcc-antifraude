# Etapa 5: histórico no sistema entregue e regra noturna

Status em 06/11/2026: em andamento.
- Feito: GNN por transação com o passado; regra noturna do Banco Central; modelos versionados.
- Pendente: LSTM com histórico, prova de equivalência na validação (2.000 transações), leitura do teste e juiz Gemini.

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

## 3. Regra noturna do Banco Central

Fonte: IN BCB 331/2022, alterada pelas IN BCB 512/2024 e 669/2025. Limite de R$ 1.000,00 para Pix e TED entre pessoas físicas, das 20h às 6h.

Como o projeto aplica a regra (`src/ferramentas/regra_noturna.py`):
- **Noite:** hora >= 20 ou hora < 6, com hora = step % 24 (mesma fonte de is_madrugada). A noite que começa às 20h termina às 6h do dia seguinte.
- **Quem entra:** pagador pessoa física e recebedor pessoa física diferente. Pessoa física é o ID que começa com C. MEI não pode ser separado nos dados.
- **Tipos (escopo B):** PIX_P2P, PIX_QRCODE, PIX_AUTOMATICO e TED. Pix saque e troco ficam fora, embora a norma também os limite à noite.
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
| Testes | 259 passed, 0 failed, 1 deselected |
| Cobertura | 91,23% (1.779 instruções, 156 sem teste; configuração padrão do pytest.ini e do .coveragerc) |

A equivalência da GNN acima é uma verificação de 200 transações. A prova exigida para a Etapa 5 é a da Fase C, com 2.000 transações.

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

- **LSTM:** continua sem histórico no modo fast, com score constante 0,8217.
- **Leitura do teste:** o índice padrão do sistema (`grafo_do_passado(None)`) carrega a base inteira, inclusive step >= 600. Ele só usa o passado de cada transação, mas a leitura precisa ser declarada no relatório da avaliação no teste.
- **GNN ainda subindo:** a validação sobe na época 10; mais épocas não foram testadas.
- **Regra noturna:** escopo B sem saque e troco; só a transação que cruza o limite vai para revisão; pessoa física inferida pelo prefixo; MEI não separado; muitas revisões de TED legítimas. Detalhes em `docs/LIMITACOES.md`.
- **Textos do juiz:** mudaram sem avaliação com o Gemini.
- **Trecho da norma:** os docs citam os números das instruções normativas; o trecho do texto oficial ainda será anexado.
