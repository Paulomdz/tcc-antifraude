# Sistema Multiagente de IA para Detecção de Fraudes

Projeto acadêmico de TCC que aplica técnicas de Machine Learning e
orquestração multiagente à análise de risco de transações financeiras. O sistema
combina modelos especializados e disponibiliza os resultados por meio de uma
API REST e de um dashboard interativo.

## Visão geral

A solução analisa características financeiras e comportamentais das transações
para apoiar a identificação de operações suspeitas. Os resultados podem ser
classificados como **APROVADO**, **REVISÃO** ou **BLOQUEADO**, acompanhados de
scores e justificativas.

O projeto reúne:

- **Isolation Forest** para análise comportamental e detecção de anomalias;
- **LSTM** para análise de padrões sequenciais;
- **GNN** para análise de relações entre contas;
- **CrewAI** para coordenar agentes especialistas e consolidar a decisão;
- **FastAPI** para disponibilizar a análise por API REST;
- **Streamlit** para oferecer uma interface visual interativa.

## Conjunto de dados

O conjunto de dados de referência do projeto é
[`dataset/dataset_pix_banco.csv`](dataset/dataset_pix_banco.csv).
Ele contém transações financeiras sintéticas com categorias Pix e outros meios
de pagamento brasileiros. A base não contém registros operacionais de uma
instituição financeira nem dados oficiais do Banco Central.

Estatísticas do arquivo:

| Característica | Valor |
|---|---:|
| Registros | 454.117 |
| Colunas | 10 |
| Categorias de transação | 12 |
| Transações marcadas como fraude | 25.247 (5,56%) |
| Contas de origem | 8.257 |
| Valores ausentes | Nenhum |

As categorias de `type` incluem `PIX_P2P`, `PIX_QRCODE`, `PIX_SAQUE`,
`PIX_TROCO`, `PIX_AUTOMATICO`, `CARTAO_ECOMMERCE`, `CARTAO_PRESENCIAL`,
`TEF`, `BOLETO`, `SAQUE`, `TED` e `EMPRESTIMO`. A coluna `isFraud` identifica
os registros sintéticos marcados como fraude. A explicação técnica e a
distribuição detalhada por categoria estão documentadas em
[`docs/secao_dataset_tcc.txt`](docs/secao_dataset_tcc.txt).

## Arquitetura

```text
.
├── dataset/
│   ├── dataset_pix_banco.csv
│   └── dataset_pix_banco_processado.parquet
├── docs/
│   └── secao_dataset_tcc.txt
├── src/
│   ├── api/                 # API REST com FastAPI
│   ├── dashboard/           # Interface interativa com Streamlit
│   ├── ferramentas/         # Inferência e ferramentas dos agentes
│   ├── modelos/             # Modelos, treinamento e modelos salvos
│   ├── orquestacao/         # Definição e fluxo dos agentes
│   └── preprocessamento/    # Preparação dos dados e feature_stats
│       ├── carregar_dataset.py
│       └── feature_stats.json
├── tests/                   # Verificações do projeto
├── pytest.ini               # Configuração de coleta e cobertura
├── run_simulation_pix.py    # Execução de simulações Pix
├── run_sistema_completo.py  # Execução do fluxo integrado
└── requirements.txt         # Dependências Python
```

## Tecnologias

- Python 3.12 ou superior
- pandas e NumPy
- scikit-learn
- PyTorch e PyTorch Geometric
- FastAPI e Uvicorn
- Streamlit
- CrewAI
- PyArrow
- pytest

## Agente Juiz com Google Gemini

O Agente Juiz usa o Google Gemini para consolidar os pareceres dos três
especialistas. Para isso, são necessárias **duas configurações diferentes**:

1. A chave `GOOGLE_API_KEY` no arquivo `.env`, na raiz do projeto.
2. O provedor Google GenAI instalado junto com o CrewAI.

Ter a chave no `.env`, por si só, não instala o provedor. Neste projeto, a
mensagem `Google Gen AI native provider not available` indica que o CrewAI
estava instalado sem o extra `google-genai`. O carregamento da variável
também precisa acontecer antes de o sistema verificar se pode iniciar o
agente; o fluxo agora carrega o `.env` da raiz antes dessa verificação.

### Configurar ou corrigir a instalação

Confira se o `.env` contém estas linhas, substituindo o valor pelo segredo
válido da sua conta Google AI Studio:

```dotenv
GOOGLE_API_KEY=sua_chave_real
GEMINI_MODEL=gemini-3.5-flash-lite
```

Não publique, envie em mensagens ou versione o arquivo `.env`. Se a chave já
está configurada, não é necessário compartilhá-la para resolver o problema.

Na raiz do projeto, com o ambiente virtual ativado, instale o provedor:

```bash
pip install -r requirements.txt
```

O `requirements.txt` declara `crewai[google-genai]`, que instala o suporte
Google exigido pelo modelo Gemini. Se preferir atualizar manualmente apenas
essa dependência:

```bash
pip install "crewai[google-genai]>=1.15.0"
```

Depois, **encerre e reinicie a API e o Streamlit** para que os processos
carreguem o pacote e as variáveis de ambiente atualizados. Há três modos:
`fast` executa os três modelos e a agregação determinística sem LLM;
`production` executa os três modelos e o Juiz Gemini; `deep` usa quatro
agentes LLM. A análise individual da API e do dashboard exige a chave e usa
`production`. A análise em lote usa sempre `fast` e continua disponível sem
chave. Não há fallback de `production` ou `deep` para decisão determinística.

Se o agente não puder iniciar, verifique:

- se a chave é válida, está ativa e pertence ao Google AI Studio;
- se o modelo definido em `GEMINI_MODEL` está disponível para essa chave;
- se a máquina tem acesso à internet e consegue alcançar a API do Google;
- o aviso completo no terminal da API ou do Streamlit. Uma chave ausente ou
  inválida, limites de uso, rede ou modelo indisponível são problemas
  diferentes da ausência do pacote do provedor.

Falhas de configuração ou execução do Gemini em `production`/`deep` são
retornadas explicitamente. Use `fast` quando a análise sem LLM for a intenção.

## Execução das interfaces

### Versão necessária do Python

Use **Python 3.12 ou superior**. A versão 3.12 foi usada para validar este
projeto. Python é o interpretador que deve estar instalado no computador; não
é uma dependência que se instala com `pip` e, por isso, não aparece como uma
linha instalável no `requirements.txt`. Se tiver uma versão anterior a 3.12,
instale uma versão compatível antes de criar o ambiente virtual.

Instale as dependências a partir da raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare os dados e treine os modelos:

```bash
python -m src.preprocessamento.carregar_dataset
python -m src.modelos.treinamento.treinar_modelos
```

O primeiro comando gera `dataset/dataset_pix_banco_processado.parquet`,
usado pelo treinamento e pela simulação. Para executar os testes e a validação
integrada:

```bash
pytest -m 'not live'
python run_sistema_completo.py
```

### O que significa a cobertura dos testes

A configuração exige pelo menos 80% para a suite passar. Alguns módulos de
treinamento e definições de agentes estão explicitamente fora do cálculo em
`.coveragerc`, por isso o percentual não representa cada linha de todo o
projeto. A última execução completa não-live registrada nesta atualização
resultou em **82 passed, 1 deselected, 1 warning** e **91,47% de cobertura**.
Esse percentual mede as instruções percorridas pelos testes; **não** significa
que o modelo tem 91,47% de acurácia, nem garante todos os cenários possíveis.

Para iniciar a API:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

A documentação interativa da API fica disponível em
<http://localhost:8000/docs>.

### Contrato da API

- `POST /analise_transacao`: análise individual no modo `production`; exige
  `GOOGLE_API_KEY`.
- `POST /analise_lote`: análise `fast` sem Gemini, com 1 a 500 transações JSON.
  Acima do limite, retorna HTTP 422 com `Lote máximo de 500 transações.`.
- `POST /analise_lote/arquivo`: recebe `multipart/form-data` com o campo
  `arquivo` e aceita CSV separado por vírgula ou TXT separado por ponto e
  vírgula. As colunas obrigatórias são `step`, `type`, `amount`, `nameOrig`,
  `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest` e
  `newbalanceDest`; `isFraud` é opcional.
  O parâmetro `quantidade` aceita 1–500 e padrão 300. A amostra usa seed fixa
  42; se o arquivo tiver menos linhas, todas são processadas e a resposta
  informa as quantidades e a seed usadas.
- `GET /config`: informa disponibilidade do LLM, modelo configurado e limite
  do lote, sem expor a chave.

O nome oficial apresentado pela API é **Sistema Multiagente de IA para
Detecção de Fraudes**. O modo `fast` não consulta o Gemini e responde com
`llm_usado="NAO"` mesmo quando uma chave está configurada.

Para iniciar o dashboard, em outro terminal:

```bash
python -m streamlit run src/dashboard/app.py --server.headless true --server.port 8502
```

## Considerações finais

O sistema reúne análise de transações,
modelos de Machine Learning e coordenação multiagente em uma solução acadêmica
com interfaces de API e dashboard. Os resultados devem ser interpretados no
contexto experimental dos dados sintéticos utilizados.
