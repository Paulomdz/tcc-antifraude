"""Cliente HTTP do dashboard para a API do sistema antifraude."""

import os
import time
from typing import Any

import requests

URL_PADRAO = "http://127.0.0.1:8000"
COMANDO_SUBIR_API = "python -m uvicorn src.api.main:app --port 8000"

_CAMPOS = {
    "oldbalanceOrg": "saldo antes (origem)",
    "newbalanceOrig": "saldo depois (origem)",
    "oldbalanceDest": "saldo antes (destino)",
    "newbalanceDest": "saldo depois (destino)",
    "amount": "valor",
    "step": "hora",
    "type": "tipo",
}


class ErroAPI(Exception):
    """Erro devolvido pela API, já com mensagem em português."""

    def __init__(self, mensagem: str, status: int | None = None):
        super().__init__(mensagem)
        self.status = status


class APIForaDoAr(ErroAPI):
    """A API não respondeu."""


def url_base() -> str:
    return os.getenv("API_URL", URL_PADRAO).rstrip("/")


def mensagem_fora_do_ar() -> str:
    return (
        f"A API está fora do ar em {url_base()}. "
        f"Suba com: {COMANDO_SUBIR_API} (aguarde 'Application startup complete')."
    )


def _http(metodo: str, caminho: str, timeout: float = 30, **kwargs):
    """Ponto único de rede; os testes substituem esta função."""
    return requests.request(metodo, url_base() + caminho, timeout=timeout, **kwargs)


def traduzir_erro(status: int, detalhe: Any) -> str:
    """Converte o erro da API em uma frase clara."""
    if isinstance(detalhe, list):
        partes = []
        for item in detalhe:
            if not isinstance(item, dict):
                partes.append(str(item))
                continue
            campo = str(item.get("loc", [""])[-1]) if item.get("loc") else ""
            nome = _CAMPOS.get(campo, campo)
            if item.get("type") == "missing":
                partes.append(f"faltou informar o campo '{nome}'")
            else:
                partes.append(f"campo '{nome}' inválido ({item.get('msg', 'valor inválido')})")
        return "Dados inválidos: " + "; ".join(partes) + "."
    texto = str(detalhe or "").strip()
    if status == 503 and "Gemini" in texto:
        return (
            "O Gemini não está configurado na API. Defina GOOGLE_API_KEY no .env "
            "e reinicie a API."
        )
    if status == 503:
        return "Os modelos treinados não foram encontrados pela API. Treine ou restaure os modelos."
    if status == 502:
        return "A análise falhou dentro da API (agentes ou Gemini). Tente de novo; veja o log da API."
    if status == 422:
        return texto or "A API recusou os dados enviados."
    return texto or f"A API respondeu com erro {status}."


def _requisitar(metodo: str, caminho: str, timeout: float = 30, **kwargs) -> Any:
    try:
        resposta = _http(metodo, caminho, timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise APIForaDoAr(mensagem_fora_do_ar()) from exc
    except requests.Timeout as exc:
        raise ErroAPI("A API demorou demais para responder. Tente de novo.") from exc
    try:
        corpo = resposta.json()
    except ValueError:
        corpo = None
    if resposta.status_code >= 400:
        detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
        raise ErroAPI(traduzir_erro(resposta.status_code, detalhe), resposta.status_code)
    if corpo is None:
        raise ErroAPI("A API devolveu uma resposta que não é JSON.")
    return corpo


_cache_config: dict = {"t": 0.0, "v": None}


def obter_config(ttl: float = 10.0) -> dict:
    """GET /config, com cache curto para não consultar a cada interação."""
    agora = time.monotonic()
    if _cache_config["v"] is not None and agora - _cache_config["t"] < ttl:
        return _cache_config["v"]
    valor = _requisitar("GET", "/config", timeout=2)
    _cache_config.update(t=agora, v=valor)
    return valor


def limpar_cache() -> None:
    _cache_config.update(t=0.0, v=None)


def analisar_transacao(transacao: dict, modo: str = "fast") -> dict:
    """POST /analise_transacao. fast: sem Gemini e sem custo; production: usa o juiz Gemini."""
    return _requisitar(
        "POST", "/analise_transacao", timeout=180, params={"modo": modo}, json=transacao
    )


def analisar_lote_arquivo(conteudo: bytes, nome: str, quantidade: int) -> dict:
    """POST /analise_lote/arquivo (fast, sem Gemini)."""
    return _requisitar(
        "POST",
        "/analise_lote/arquivo",
        timeout=900,
        params={"quantidade": int(quantidade)},
        files={"arquivo": (nome, conteudo, "application/octet-stream")},
    )


def obter_historico(limite: int = 100, decisao: str | None = None) -> list[dict]:
    params: dict[str, Any] = {"limit": int(limite)}
    if decisao:
        params["decisao"] = decisao
    return _requisitar("GET", "/historico", timeout=30, params=params)


def obter_estatisticas() -> dict:
    return _requisitar("GET", "/historico/estatisticas", timeout=30)
