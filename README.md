# SISTEMA MULTIAGENTE DE IA PARA DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS

Projeto acadêmico e experimental para detecção de fraudes financeiras com múltiplos modelos de Machine Learning, orquestração de agentes e interfaces para uso em linha de comando, API e dashboard.

Este sistema combina técnicas de análise comportamental, sequencial e de grafo para classificar transações como aprovadas, sujeitas a revisão ou bloqueadas, com base em scores de risco calculados por especialistas de IA.

## Visão Geral

O software foi desenvolvido para simular e analisar transações bancárias sintéticas do dataset PaySim, identificando padrões suspeitos que podem indicar fraude. A arquitetura integra:

- Isolation Forest para detecção de anomalias comportamentais;
- LSTM para análise temporal de sequências de transações;
- GNN para modelagem de relações entre contas;
- CrewAI e Gemini para orquestração multiagente;
- FastAPI e Streamlit para uso em API e interface visual.

## Objetivo

O objetivo principal é demonstrar como uma solução multiagente pode apoiar a detecção de risco financeiro em cenários de fraude, oferecendo uma base modular para pesquisa, apresentação acadêmica e prototipagem.

## Funcionalidades Principais

- Carregamento e pré-processamento de dados PaySim;
- Treinamento e inferência de modelos de fraude;
- Análise de transações individuais e em lote;
- Orquestração de especialistas com score consolidado;
- API REST para integração com sistemas externos;
- Dashboard interativo para visualização do resultado.

## Público-Alvo

- estudantes e pesquisadores em IA/ML;
- equipes de risco, compliance e fraude;
- desenvolvedores interessados em protótipos financeiros;
- projetos de TCC ou demonstrações acadêmicas.

## Problemas que Resolve

- reduz a dependência de revisão manual de transações;
- ajuda a identificar padrões atípicos em grandes volumes;
- oferece uma abordagem híbrida entre modelos clássicos e IA generativa;
- funciona como base para prototipação de sistemas de alerta antifraude.

## Arquitetura do Projeto

### Tecnologias e Bibliotecas

- Python
- pandas, numpy
- scikit-learn
- PyTorch e PyTorch Geometric (modelos LSTM/GNN)
- FastAPI
- Streamlit
- CrewAI
- Google Gemini (via suporte nativo do CrewAI, sem langchain-google-genai)
- python-dotenv
- pytest

### Estrutura Principal

```text
.
├── data/                           # Dados brutos e processados
├── src/
│   ├── api/                        # API FastAPI
│   ├── dashboard/                  # Interface Streamlit
│   ├── ferramentas/                # Inferência e ferramentas de agentes
│   ├── modelos/                    # LSTM, GNN e treinamento
│   └── preprocessamento/           # Leitura e limpeza do PaySim
├── tests/                          # Testes automatizados
├── run_simulation.py               # Simulação principal
├── teste_sistema_completo.py       # Verificação do sistema
└── requirements.txt                # Dependências do projeto
```

## Fluxo de Funcionamento

1. O dataset PaySim é carregado a partir de data/PS_20174392719_1491204439457_log.csv, em blocos (streaming), para lidar com o volume real (~6,36 milhões de transações) sem estourar memória.
2. O módulo de pré-processamento converte, limpa e gera features como amount_log, step_norm (normalizado por uma constante fixa de 744 steps, a duração canônica da simulação PaySim) e indicadores de saldo.
3. Os modelos de ML são treinados ou carregados a partir de src/modelos/modelos_salvos.
4. A inferência calcula scores de risco comportamental, temporal e de identidade.
5. A orquestração multiagente agrega esses scores e produz uma decisão final.
6. O resultado é entregue por linha de comando, API REST ou dashboard interativo.

## Como Executar

### 1. Preparar o ambiente

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Opcional, para ativar o Agente Juiz com Gemini de verdade) copie
`.env.example` para `.env` e preencha `GOOGLE_API_KEY`. Sem isso, o sistema
funciona normalmente usando a regra de decisão determinística como fallback.

### 2. Preparar os dados

O arquivo bruto do PaySim (dataset real, ~6,36 milhões de transações / ~470MB)
deve estar em:

```text
data/PS_20174392719_1491204439457_log.csv
```

Em seguida, execute:

```bash
python src/preprocessamento/carregar_paysim.py
```

O processamento é feito em blocos (streaming para Parquet via `pyarrow`), então
funciona mesmo em máquinas com pouca RAM disponível — não é necessário
carregar o CSV inteiro em memória de uma vez.

### 3. Executar a simulação

```bash
python run_simulation.py
```

### 4. Executar a API

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Acesse a documentação da API em:

```text
http://localhost:8000/docs
```

### 5. Executar o dashboard

```bash
python -m streamlit run src/dashboard/app.py --server.headless true --server.port 8502
```

## Exemplo de Uso

```python
import pandas as pd
from src.orquestração.fluxo_crewai import orchestrate_transaction

# Carrega transação processada
transacao = pd.read_parquet("data/paysim_dados_processados.parquet").iloc[0].to_dict()

resultado = orchestrate_transaction(transacao)
print("Decisão:", resultado["decision"]["decision"])
print("Score final:", round(resultado["decision"]["score"], 3))
```

## Status Atual do Projeto

### Funcionalidades já estruturadas

- Pré-processamento do PaySim
- Inferência modular por modelos
- API REST com FastAPI
- Dashboard com Streamlit
- Orquestração multiagente com CrewAI

### Componentes opcionais ou dependentes de instalação

- LSTM (requer PyTorch instalado)
- GNN (requer PyTorch Geometric instalado)
- Agente Juiz via LLM real (requer `GOOGLE_API_KEY` configurada — veja `.env.example`)

Todos esses módulos possuem fallback automático: sem PyTorch instalado, LSTM/GNN
retornam um placeholder controlado; sem `GOOGLE_API_KEY` configurada (ou se a
chamada ao Gemini falhar por qualquer motivo), o Agente Juiz usa uma regra de
decisão determinística equivalente, sem quebrar a aplicação. Veja
`docs/CORRECOES_APLICADAS.md` para o detalhamento de como essa integração foi
implementada e testada.

O `isolation_forest.pkl` incluído no repositório já foi retreinado com uma
amostra balanceada do dataset real do PaySim (5.000 fraudes e 5.000
não-fraudes, extraídas das 8.213 fraudes existentes entre as 6,36 milhões de
transações). LSTM e GNN continuam como placeholder porque PyTorch/PyTorch
Geometric não foram instalados no ambiente onde este treinamento rodou —
instale-os e rode `python -m src.modelos.treinamento.treinar_modelos`
novamente para treiná-los de fato.

## Casos de Uso

- demonstração de TCC em IA aplicada à fraude;
- análise de risco em transações financeiras sintéticas;
- prototipagem de sistemas antifraude;
- integração com painéis de compliance e análise de risco.

## Diferenciais

- arquitetura modular e extensível;
- combinação de múltiplos modelos de risco;
- integração entre IA tradicional e IA generativa;
- suporte para API e dashboard a partir da mesma base de código;
- foco em aprendizado, pesquisa e prototipação prática.

## Limitações

- o dataset é simulado (PaySim), não representando necessariamente dados reais de instituições financeiras;
- alguns componentes avançados dependem de bibliotecas pesadas, como PyTorch e PyTorch Geometric;
- a orquestração multiagente pode precisar de ajustes finos conforme a disponibilidade da API Gemini e das dependências instaladas;
- o Isolation Forest retreinado usa features simples (valores brutos de saldo/valor e step_norm) e `contamination` fixo em 0,5, resultando em acurácia modesta (~59% na própria amostra de treino) — enriquecer as features é uma melhoria de modelagem, não um bug de código;
- LSTM e GNN ainda não foram treinados com dados reais (dependem de PyTorch/PyTorch Geometric, não instalados no ambiente usado para o treinamento mais recente).

## Melhorias Futura

- adicionar métricas avançadas de avaliação (precision, recall, F1-score);
- treinar LSTM e GNN com o dataset real (requer instalar PyTorch/PyTorch Geometric);
- enriquecer as features do Isolation Forest para melhorar a acurácia;
- melhorar a robustez da API e do dashboard;
- ampliar a orquestração multiagente com agentes mais especializados;
- implementar monitoramento e histórico de análises.

## Documentação Adicional

- `docs/CORRECOES_APLICADAS.md` — histórico de correções aplicadas ao código
  para alinhá-lo à monografia (Agente Juiz real via CrewAI/Gemini, correção do
  cálculo de horário, normalização de scores etc.).
- `docs/RELATORIO_REVISAO_CODIGO.md` — relatório de revisão de código.

## Referências

- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- CrewAI: https://docs.crewai.com/
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/
- PaySim: https://www.compred.org/PaySim/

