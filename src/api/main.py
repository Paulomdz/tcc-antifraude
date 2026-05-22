"""API REST para análise de transações financeiras - TCC Antifraude."""

from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="TCC Antifraude API",
    description="Detecção de fraudes em transações financeiras usando ML multi-agente",
    version="1.0.0",
)

_DECISION_MAP = {
    "Aprovada": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "Recusada": "BLOQUEADO",
}


class TransacaoInput(BaseModel):
    """Dados de entrada de uma transação financeira."""

    step: int = Field(default=1, ge=1, description="Step de tempo da simulação")
    type: str = Field(default="TRANSFER", description="Tipo da transação")
    amount: float = Field(..., ge=0.0, description="Valor da transação em R$")
    nameOrig: str = Field(default="", description="Identificador da conta de origem")
    oldbalanceOrg: float = Field(default=0.0, ge=0.0, description="Saldo anterior da origem")
    newbalanceOrig: float = Field(default=0.0, ge=0.0, description="Novo saldo da origem")
    nameDest: str = Field(default="", description="Identificador da conta de destino")
    oldbalanceDest: float = Field(default=0.0, ge=0.0, description="Saldo anterior do destino")
    newbalanceDest: float = Field(default=0.0, ge=0.0, description="Novo saldo do destino")
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


@app.get("/", summary="Health check")
def health_check() -> Dict[str, str]:
    """Verifica se a API está operacional."""
    return {"status": "ok", "message": "TCC Antifraude API operacional"}


@app.post("/analise_transacao", response_model=ResultadoAnalise, summary="Analisar transação")
def analise_transacao(transacao: TransacaoInput) -> ResultadoAnalise:
    """
    Analisa uma transação financeira e retorna decisão com scores detalhados.

    Decisões possíveis: **APROVADO**, **REVISÃO**, **BLOQUEADO**.
    """
    try:
        import importlib
        fluxo = importlib.import_module("src.orquestração.fluxo_crewai")
        orchestrate_transaction = fluxo.orchestrate_transaction
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Módulo de orquestração indisponível: {exc}",
        )

    try:
        resultado = orchestrate_transaction(transacao.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Modelo não encontrado: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {exc}")

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
