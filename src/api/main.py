"""API REST do sistema multiagente antifraude."""

import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.historico.database import (
    DB_PATH,
    init_db,
    obter_estatisticas_24h,
    obter_historico,
    salvar_analise,
)
from src.ferramentas.explicabilidade import top3_por_zscore
from src.ferramentas.leitor_arquivo_lote import (
    SEED_AMOSTRAGEM,
    amostrar_lote,
    ler_arquivo_lote,
)
from src.preprocessamento.carregar_dataset import (
    FEATURE_STATS_PATH,
    IF_FEATURE_NAMES,
    compute_if_features,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)
GEMINI_NOT_CONFIGURED = "Gemini não configurado. Defina GOOGLE_API_KEY no .env."
SERVICE_NAME = "Sistema Multiagente de IA para Detecção de Fraudes"
LIMITE_LOTE = 500
SEED_AMOSTRAGEM_LOTE = SEED_AMOSTRAGEM

@asynccontextmanager
async def lifespan(_app):
    init_db(DB_PATH)
    yield


app = FastAPI(
    title=SERVICE_NAME,
    description=SERVICE_NAME,
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_DECISION_MAP = {
    "Aprovada": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "Recusada": "BLOQUEADO",
    "APROVADO": "APROVADO",
    "REVISÃO": "REVISÃO",
    "BLOQUEADO": "BLOQUEADO",
}


class TransacaoInput(BaseModel):
    """Dados de entrada de uma transação financeira."""

    # TCC / D2: step pode ser maior que 23; as features aplicam step % 24 antes
    # de hora_norm = hora / 23, tanto no treino quanto na inferência.
    step: int = Field(default=12, ge=0, description="Hora do dia (0–23) ou step do dataset.")
    type: str = Field(default="PIX_P2P", description="Tipo de transação")
    amount: float = Field(..., ge=0.0, description="Valor da transação em R$")
    nameOrig: str = Field(default="", description="Identificador da conta de origem")
    oldbalanceOrg: float = Field(..., ge=0.0)
    newbalanceOrig: float = Field(default=0.0, ge=0.0)
    nameDest: str = Field(default="", description="Identificador da conta de destino")
    oldbalanceDest: float = Field(default=0.0, ge=0.0)
    newbalanceDest: float = Field(default=0.0, ge=0.0)


class ScoreAgente(BaseModel):
    score: float
    label: str
    explicacao: str


class Explicabilidade(BaseModel):
    """Top-3 de features por |z| em relação a feature_stats.json."""

    top3: list[dict[str, Any]] = Field(default_factory=list)


class ResultadoAnalise(BaseModel):
    decisao: str
    score_final: float
    justificativa: str
    scores: dict[str, float]
    agentes: dict[str, ScoreAgente]
    explicabilidade: Explicabilidade
    modo: str = "production"


class LoteInput(BaseModel):
    transacoes: list[TransacaoInput] = Field(..., min_length=1)


class ItemLote(BaseModel):
    indice: int
    decisao: str
    score_final: float
    scores: dict[str, float]
    erro: str = ""


class ResultadoLote(BaseModel):
    total: int
    aprovados: int
    revisoes: int
    bloqueados: int
    erros: int
    resultados: list[ItemLote]
    llm_usado: str = "NAO"


class ResultadoLoteArquivo(ResultadoLote):
    quantidade_solicitada: int
    quantidade_usada: int
    total_arquivo: int
    seed: int
    arquivo_menor_que_quantidade: bool


def _get_orchestrator():
    try:
        return importlib.import_module("src.orquestacao.fluxo_multiagente")
    except ImportError as exc:
        logger.exception("Módulo de orquestração indisponível.")
        raise HTTPException(
            status_code=500, detail="Módulo de orquestração indisponível."
        ) from exc


@lru_cache(maxsize=1)
def _carregar_feature_stats() -> dict[str, dict[str, float]]:
    return json.loads(FEATURE_STATS_PATH.read_text(encoding="utf-8"))


def _calcular_top3(transacao: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        valores = dict(zip(IF_FEATURE_NAMES, compute_if_features(transacao)))
        return top3_por_zscore(valores, _carregar_feature_stats())
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Não foi possível calcular o top-3 de explicabilidade.")
        return []


def _parse_resultado(
    resultado: dict[str, Any], transacao: dict[str, Any] | None = None
) -> ResultadoAnalise:
    decision = resultado["decision"]
    raw_decision = decision["decision"]
    if not isinstance(raw_decision, str) or raw_decision not in _DECISION_MAP:
        raise ValueError(f"Decisão desconhecida: {raw_decision!r}")
    behavior = resultado["behavior"]
    temporal = resultado["temporal"]
    identity = resultado["identity"]
    return ResultadoAnalise(
        decisao=_DECISION_MAP[raw_decision],
        score_final=float(decision["score"]),
        justificativa=str(decision["justification"]),
        scores={
            "comportamental": float(behavior["score"]),
            "temporal": float(temporal["score"]),
            "identidade": float(identity["score"]),
        },
        agentes={
            "comportamental": ScoreAgente(
                score=float(behavior["score"]),
                label=str(behavior["label"]),
                explicacao=str(behavior["details"]["explanation"]),
            ),
            "temporal": ScoreAgente(
                score=float(temporal["score"]),
                label=str(temporal["label"]),
                explicacao=str(temporal["details"]["explanation"]),
            ),
            "identidade": ScoreAgente(
                score=float(identity["score"]),
                label=str(identity["label"]),
                explicacao=str(identity["details"]["explanation"]),
            ),
        },
        explicabilidade=Explicabilidade(
            top3=_calcular_top3(transacao) if transacao is not None else []
        ),
        modo=str(resultado.get("modo", "production")),
    )


def _transaction_hash(transaction: dict[str, Any]) -> str:
    payload = json.dumps(transaction, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_analysis(
    transaction: dict[str, Any],
    parsed: ResultadoAnalise | None,
    duration_ms: int,
    modo: str,
    error: str | None = None,
) -> None:
    scores = parsed.scores if parsed else {}
    transaction_hash = _transaction_hash(transaction)
    salvar_analise(
        {
            "transaction_hash": transaction_hash,
            "type": transaction.get("type"),
            "amount": transaction.get("amount", 0.0),
            "score_final": parsed.score_final if parsed else 0.0,
            "decisao": parsed.decisao if parsed else "ERRO",
            "score_behavior": scores.get("comportamental", 0.0),
            "score_temporal": scores.get("temporal", 0.0),
            "score_identity": scores.get("identidade", 0.0),
            "duracao_ms": duration_ms,
            "llm_usado": "SIM" if modo in {"production", "deep"} else "NAO",
            "erro": error,
        },
        db_path=DB_PATH,
    )


@app.get("/", summary="Health check", tags=["Sistema"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": SERVICE_NAME}


@app.get("/config", summary="Consultar configuração pública", tags=["Sistema"])
def obter_configuracao() -> dict[str, bool | str | int | None]:
    modelo = os.getenv("GEMINI_MODEL", "").strip() or None
    return {
        "llm_disponivel": bool(os.getenv("GOOGLE_API_KEY", "").strip()),
        "modelo_llm": modelo,
        "limite_lote": LIMITE_LOTE,
    }


@app.post(
    "/analise_transacao",
    response_model=ResultadoAnalise,
    summary="Analisar transação em production",
    tags=["Análise"],
)
@app.post(
    "/v1/analise_transacao",
    response_model=ResultadoAnalise,
    include_in_schema=False,
)
def analise_transacao(
    transacao: TransacaoInput,
    modo: Literal["production", "fast"] = Query(
        default="production",
        description="production usa o juiz Gemini (exige chave); fast não usa LLM.",
    ),
) -> ResultadoAnalise:
    """Production sempre exige Gemini e não troca para decisão determinística.

    O modo fast (opcional) roda os três modelos com a agregação determinística,
    sem Gemini e sem custo, e devolve o mesmo contrato, com top-3.
    """
    if modo == "production" and not os.getenv("GOOGLE_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail=GEMINI_NOT_CONFIGURED)

    payload = transacao.model_dump()
    started = time.perf_counter()
    resultado: dict[str, Any] | None = None
    parsed: ResultadoAnalise | None = None
    try:
        fluxo = _get_orchestrator()
        if modo == "fast":
            resultado = fluxo.orchestrate_ml_only(payload)
        else:
            resultado = fluxo.orchestrate_transaction(payload, modo="production")
        parsed = _parse_resultado(resultado, payload)
    except FileNotFoundError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        _save_analysis(payload, parsed, duration_ms, modo, str(exc))
        raise HTTPException(status_code=503, detail="Modelo treinado não encontrado.") from exc
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        _save_analysis(payload, parsed, duration_ms, modo, str(exc))
        logger.exception("Falha na análise %s.", modo)
        raise HTTPException(status_code=502, detail=f"Falha ao executar análise {modo}.") from exc

    duration_ms = round((time.perf_counter() - started) * 1000)
    _save_analysis(payload, parsed, duration_ms, modo)
    return parsed


@app.post(
    "/analise_lote",
    response_model=ResultadoLote,
    summary="Analisar lote de transações no modo fast",
    tags=["Análise"],
)
def analise_lote(payload: LoteInput) -> ResultadoLote:
    """Analisa até 500 transações em fast e persiste sucesso ou falha individual."""
    return _processar_lote(payload.transacoes)


def _processar_lote(transacoes: list[TransacaoInput]) -> ResultadoLote:
    if len(transacoes) > LIMITE_LOTE:
        raise HTTPException(
            status_code=422, detail="Lote máximo de 500 transações."
        )
    fluxo = _get_orchestrator()
    items: list[ItemLote] = []
    approved = reviews = blocked = errors = 0
    for index, tx in enumerate(transacoes):
        tx_data = tx.model_dump()
        started = time.perf_counter()
        resultado: dict[str, Any] | None = None
        parsed: ResultadoAnalise | None = None
        error: str | None = None
        try:
            resultado = fluxo.orchestrate_ml_only(tx_data)
            parsed = _parse_resultado(resultado)
            if parsed.decisao == "APROVADO":
                approved += 1
            elif parsed.decisao == "REVISÃO":
                reviews += 1
            else:
                blocked += 1
            items.append(
                ItemLote(
                    indice=index,
                    decisao=parsed.decisao,
                    score_final=parsed.score_final,
                    scores=parsed.scores,
                )
            )
        except Exception as exc:
            error = str(exc)
            errors += 1
            logger.exception("Falha na transação %d do lote.", index)
            items.append(
                ItemLote(
                    indice=index,
                    decisao="ERRO",
                    score_final=0.0,
                    scores={},
                    erro=error,
                )
            )
        duration_ms = round((time.perf_counter() - started) * 1000)
        try:
            _save_analysis(tx_data, parsed, duration_ms, "fast", error)
        except Exception:
            logger.exception(
                "Falha ao gravar histórico da transação %d do lote; "
                "o processamento continuará.",
                index,
            )

    return ResultadoLote(
        total=len(transacoes),
        aprovados=approved,
        revisoes=reviews,
        bloqueados=blocked,
        erros=errors,
        resultados=items,
        llm_usado="NAO",
    )


@app.post(
    "/analise_lote/arquivo",
    response_model=ResultadoLoteArquivo,
    summary="Analisar arquivo CSV/TXT com amostragem no modo fast",
    tags=["Análise"],
)
async def analise_lote_arquivo(
    arquivo: UploadFile = File(...),
    quantidade: int = Query(default=300, ge=1, le=LIMITE_LOTE),
) -> ResultadoLoteArquivo:
    """Lê e amostra um CSV/TXT usando o leitor compartilhado."""
    try:
        dataframe = ler_arquivo_lote(
            await arquivo.read(), arquivo.filename or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    total_arquivo = len(dataframe)
    quantidade_usada = min(quantidade, total_arquivo)
    amostra = amostrar_lote(dataframe, quantidade, SEED_AMOSTRAGEM_LOTE)

    transacoes: list[TransacaoInput] = []
    for indice, (_, linha) in enumerate(amostra.iterrows(), start=1):
        try:
            transacoes.append(TransacaoInput.model_validate(linha.to_dict()))
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Linha amostrada {indice} contém dados inválidos.",
            ) from exc

    resultado = _processar_lote(transacoes)
    return ResultadoLoteArquivo(
        **resultado.model_dump(),
        quantidade_solicitada=quantidade,
        quantidade_usada=quantidade_usada,
        total_arquivo=total_arquivo,
        seed=SEED_AMOSTRAGEM_LOTE,
        arquivo_menor_que_quantidade=total_arquivo < quantidade,
    )


@app.get("/historico", summary="Consultar histórico de análises", tags=["Histórico"])
def historico(
    limit: int = Query(default=100, ge=1, le=1000),
    decisao: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return obter_historico(limit=limit, decisao=decisao, db_path=DB_PATH)


@app.get(
    "/historico/estatisticas",
    summary="Estatísticas de análises nas últimas 24 horas",
    tags=["Histórico"],
)
def historico_estatisticas() -> dict[str, Any]:
    return obter_estatisticas_24h(db_path=DB_PATH)
