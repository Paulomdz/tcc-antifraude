"""API REST — ClearSafe Antifraude.

Executa com:
    uvicorn src.api.main:app --reload
"""

import importlib
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="SISTEMA MULTIAGENTE DE IA PARA DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS",
    description=(
        "Detecção de fraudes em transações financeiras usando três modelos de ML "
        "(Isolation Forest, LSTM e GNN) orquestrados por agentes especializados com IA. "
        "Implementação de regras rígidas de risco bancário brasileiro.\n\n"
        "**Decisões possíveis:** `APROVADO` · `REVISÃO` · `BLOQUEADO`"
    ),
    version="1.0.0",
    contact={"name": "TCC Antifraude"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_DECISION_MAP: Dict[str, str] = {
    "Aprovada": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "Recusada": "BLOQUEADO",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TransacaoInput(BaseModel):
    """Dados de entrada de uma transação financeira."""

    step: int = Field(default=1, ge=1, description="Step de tempo da simulação (1–744)")
    type: str = Field(default="TRANSFER", description="Tipo da transação")
    amount: float = Field(..., ge=0.0, description="Valor da transação em R$")
    nameOrig: str = Field(default="", description="Identificador da conta de origem")
    oldbalanceOrg: float = Field(default=0.0, ge=0.0, description="Saldo anterior da origem")
    newbalanceOrig: float = Field(default=0.0, ge=0.0, description="Novo saldo da origem")
    nameDest: str = Field(default="", description="Identificador da conta de destino")
    step_norm: float = Field(default=0.5, ge=0.0, le=1.0, description="Step normalizado [0, 1]")


class ScoreAgente(BaseModel):
    """Score e explicação de um agente especialista."""

    score: float
    label: str
    explicacao: str


class ResultadoAnalise(BaseModel):
    """Resultado completo da análise de fraude."""

    decisao: str
    score_final: float
    justificativa: str
    scores: Dict[str, float]
    agentes: Dict[str, ScoreAgente]


class LoteInput(BaseModel):
    """Lista de transações para análise em lote."""

    transacoes: List[TransacaoInput] = Field(
        ..., min_length=1, max_length=100, description="Lista de transações (máx. 100)"
    )


class ItemLote(BaseModel):
    """Resultado de uma transação no lote."""

    indice: int
    decisao: str
    score_final: float
    scores: Dict[str, float]
    erro: str = ""


class ResultadoLote(BaseModel):
    """Resultado agregado da análise em lote."""

    total: int
    aprovados: int
    revisoes: int
    bloqueados: int
    resultados: List[ItemLote]


# ---------------------------------------------------------------------------
# Utilitário interno
# ---------------------------------------------------------------------------


def _get_orchestrator():
    """Importa o módulo de orquestração em tempo de execução."""
    try:
        return importlib.import_module("src.orquestração.fluxo_crewai")
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Módulo de orquestração indisponível: {exc}",
        )


def _parse_resultado(resultado: dict) -> ResultadoAnalise:
    """Converte o dict do pipeline para o schema de resposta."""
    decision_raw = resultado["decision"]["decision"]
    decisao = _DECISION_MAP.get(decision_raw, decision_raw)

    behavior = resultado["behavior"]
    temporal = resultado["temporal"]
    identity = resultado["identity"]

    return ResultadoAnalise(
        decisao=decisao,
        score_final=float(resultado["decision"]["score"]),
        justificativa=resultado["decision"]["justification"],
        scores={
            "comportamental": float(behavior["score"]),
            "temporal": float(temporal["score"]),
            "identidade": float(identity["score"]),
        },
        agentes={
            "comportamental": ScoreAgente(
                score=float(behavior["score"]),
                label=behavior["label"],
                explicacao=behavior["details"]["explanation"],
            ),
            "temporal": ScoreAgente(
                score=float(temporal["score"]),
                label=temporal["label"],
                explicacao=temporal["details"]["explanation"],
            ),
            "identidade": ScoreAgente(
                score=float(identity["score"]),
                label=identity["label"],
                explicacao=identity["details"]["explanation"],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="Health check", tags=["Sistema"])
def health_check() -> Dict[str, str]:
    """Verifica se a API está operacional."""
    return {"status": "ok", "message": "ClearSafe Antifraude API operacional"}


@app.post(
    "/analise_transacao",
    response_model=ResultadoAnalise,
    summary="Analisar transação",
    tags=["Análise"],
)
def analise_transacao(transacao: TransacaoInput) -> ResultadoAnalise:
    """
    Analisa uma transação financeira com três agentes de ML e retorna a decisão.

    - **APROVADO** — score < 0.5, risco baixo
    - **REVISÃO** — score entre 0.5 e 0.8, requer revisão humana
    - **BLOQUEADO** — score ≥ 0.8, alto risco de fraude
    """
    fluxo = _get_orchestrator()
    try:
        resultado = fluxo.orchestrate_transaction(transacao.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Modelo não encontrado: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {exc}")

    return _parse_resultado(resultado)


@app.post(
    "/analise_lote",
    response_model=ResultadoLote,
    summary="Analisar lote de transações",
    tags=["Análise"],
)
def analise_lote(payload: LoteInput) -> ResultadoLote:
    """
    Analisa um lote de até 100 transações e retorna estatísticas agregadas.

    Transações que falham individualmente são registradas com campo `erro` preenchido
    e não interrompem o processamento do restante do lote.
    """
    fluxo = _get_orchestrator()
    resultados: List[ItemLote] = []
    aprovados = revisoes = bloqueados = 0

    for i, tx in enumerate(payload.transacoes):
        try:
            resultado = fluxo.orchestrate_transaction(tx.model_dump())
            parsed = _parse_resultado(resultado)
            if parsed.decisao == "APROVADO":
                aprovados += 1
            elif parsed.decisao == "REVISÃO":
                revisoes += 1
            else:
                bloqueados += 1
            resultados.append(
                ItemLote(
                    indice=i,
                    decisao=parsed.decisao,
                    score_final=parsed.score_final,
                    scores=parsed.scores,
                )
            )
        except Exception as exc:
            resultados.append(
                ItemLote(
                    indice=i,
                    decisao="ERRO",
                    score_final=0.0,
                    scores={},
                    erro=str(exc),
                )
            )

    return ResultadoLote(
        total=len(payload.transacoes),
        aprovados=aprovados,
        revisoes=revisoes,
        bloqueados=bloqueados,
        resultados=resultados,
    )
