"""Avaliação do sistema completo no modo fast, sem LLM e sem treino.

A: como o sistema roda hoje, sem ``sequence``, transação por transação.
B: scores com histórico, em lote, com as mesmas funções da Etapa 2.
Cada sistema passa uma vez pelos modelos; o juiz roda com e sem a regra da
madrugada sobre os mesmos scores.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ferramentas.ferramentas_agentes import judge_tool
from src.ferramentas.inferencia_modelos import MODEL_DIR, load_pickle_model
from src.orquestacao import fluxo_multiagente
from src.orquestacao.fluxo_multiagente import orchestrate_ml_only
from src.preprocessamento.carregar_dataset import (
    GNN_FEATURE_NAMES,
    IF_FEATURE_NAMES,
    LSTM_FEATURE_NAMES,
    PROCESSED_DATA_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIR_REPORTS = PROJECT_ROOT / "reports"
SEED = 42
AMOSTRA_A_VALIDACAO = 2000
COLUNAS_TRANSACAO = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]
COLUNAS = [*COLUNAS_TRANSACAO, "isFraud"]
DECISOES = {
    "Aprovada": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "Recusada": "BLOQUEADO",
}
CLASSES = ("APROVADO", "REVISÃO", "BLOQUEADO")
ENTRADA_A = {
    "lstm": "matriz de zeros (5 x 17), sem histórico",
    "gnn": "um nó (a transação atual), sem arestas de histórico",
}
ENTRADA_B = {
    "lstm": "até 5 transações anteriores da mesma conta origem",
    "gnn": "subgrafo com arestas só do passado para a transação; fanouts (10, 5)",
}
PREFIXOS_LLM = (
    "crewai",
    "litellm",
    "openai",
    "anthropic",
    "langchain",
    "google.genai",
    "google.generativeai",
)


def regra_banco(transacao: dict[str, Any]) -> bool:
    """Linha de base: só para avaliação, fora do sistema."""
    valor = float(transacao["amount"])
    return valor >= 0.9 * float(transacao["oldbalanceOrg"]) and valor >= 5000.0


def matriz_decisoes(
    decisoes: Sequence[str], rotulos: Sequence[int]
) -> dict[str, dict[str, int]]:
    matriz = {classe: {"fraude": 0, "legitima": 0} for classe in CLASSES}
    for decisao, rotulo in zip(decisoes, rotulos, strict=True):
        matriz[decisao]["fraude" if rotulo else "legitima"] += 1
    return matriz


def metricas_binarias(
    previsto: Sequence[bool], rotulos: Sequence[int]
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for positivo, rotulo in zip(previsto, rotulos, strict=True):
        if positivo and rotulo:
            tp += 1
        elif positivo:
            fp += 1
        elif rotulo:
            fn += 1
        else:
            tn += 1
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    return {
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def pr_auc(rotulos: Sequence[int], scores: Sequence[float] | None) -> float | None:
    """PR-AUC do score; ``None`` sem score ou sem as duas classes."""
    if scores is None or len(set(int(rotulo) for rotulo in rotulos)) < 2:
        return None
    return float(average_precision_score(rotulos, scores))


def resumir(
    decisoes: Sequence[str],
    rotulos: Sequence[int],
    scores: Sequence[float] | None,
) -> dict[str, Any]:
    return {
        "n": len(decisoes),
        "fraudes": int(sum(int(rotulo) for rotulo in rotulos)),
        "matriz": matriz_decisoes(decisoes, rotulos),
        "bloqueado_e_fraude": metricas_binarias(
            [decisao == "BLOQUEADO" for decisao in decisoes], rotulos
        ),
        "bloqueado_ou_revisao_e_fraude": metricas_binarias(
            [decisao != "APROVADO" for decisao in decisoes], rotulos
        ),
        "pr_auc_score_final": pr_auc(rotulos, scores),
        "taxa_revisao": (
            sum(decisao == "REVISÃO" for decisao in decisoes) / len(decisoes)
            if len(decisoes)
            else 0.0
        ),
    }


def comparar(
    decisoes: Sequence[str],
    rotulos: Sequence[int],
    scores: Sequence[float] | None,
    tipos: Sequence[str],
) -> dict[str, Any]:
    """Resumo geral e por tipo de transação."""
    resultado = resumir(decisoes, rotulos, scores)
    resultado["por_tipo"] = {}
    for tipo in sorted(set(tipos)):
        posicoes = [i for i, valor in enumerate(tipos) if valor == tipo]
        resultado["por_tipo"][tipo] = resumir(
            [decisoes[i] for i in posicoes],
            [rotulos[i] for i in posicoes],
            None if scores is None else [scores[i] for i in posicoes],
        )
    return resultado


def carregar_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (contexto sem rótulos, alvos), como na Etapa 2.

    A validação nunca lê step >= 600. O contexto do teste contém transações
    posteriores a cada alvo, mas janelas e arestas só olham o passado.
    """
    passado = pd.read_parquet(
        PROCESSED_DATA_PATH, columns=COLUNAS, filters=[("step", "<", 600)]
    ).reset_index(drop=True)
    if not passado["step"].lt(600).all():
        raise ValueError("A leitura pré-teste incluiu step >= 600.")
    if split == "validacao":
        alvos = passado[passado["step"] >= 550].copy()
        contexto = passado.drop(columns=["isFraud"])
    elif split == "teste":
        alvos = pd.read_parquet(
            PROCESSED_DATA_PATH, columns=COLUNAS, filters=[("step", ">=", 600)]
        )
        if not alvos["step"].ge(600).all():
            raise ValueError("A leitura de teste incluiu step < 600.")
        alvos.index = pd.RangeIndex(len(passado), len(passado) + len(alvos))
        contexto = pd.concat(
            (passado.drop(columns=["isFraud"]), alvos.drop(columns=["isFraud"]))
        )
    else:
        raise ValueError("split deve ser 'validacao' ou 'teste'.")
    return (
        contexto.sort_values("step", kind="stable"),
        alvos.sort_values("step", kind="stable"),
    )


def calcular_scores_b(
    contexto: pd.DataFrame, alvos: pd.DataFrame
) -> dict[str, Any]:
    """Uma passada em lote, com as mesmas funções da Etapa 2 e os modelos salvos."""
    import torch
    from torch.utils.data import DataLoader

    from src.modelos.arquitetura_gnn import TransactionMiniBatchDataset, carregar_gnn
    from src.modelos.arquitetura_lstm import TransactionSequenceDataset, carregar_lstm
    from src.modelos.treinamento import treinar_modelos as etapa2

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_if = load_pickle_model("isolation_forest.pkl")
    modelo_lstm, _ = carregar_lstm(str(MODEL_DIR / "lstm_model.pt"))
    modelo_gnn, _ = carregar_gnn(str(MODEL_DIR / "gnn_model.pt"))
    rotulos = alvos["isFraud"].to_numpy(dtype=np.int64)

    comportamental = etapa2._if_scores(
        modelo_if, etapa2._feature_matrix(alvos, IF_FEATURE_NAMES)
    )
    janelas = DataLoader(
        TransactionSequenceDataset(
            alvos,
            seq_length=etapa2.LSTM_SEQUENCE_LENGTH,
            feature_cols=LSTM_FEATURE_NAMES,
            context_df=contexto,
        ),
        batch_size=etapa2.LSTM_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    _, rotulos_lstm, temporal = etapa2._evaluate_lstm(
        modelo_lstm.to(dispositivo),
        janelas,
        dispositivo,
        1.0,
        torch.nn.BCELoss(reduction="none"),
    )
    _, rotulos_gnn, identidade = etapa2._graph_scores(
        modelo_gnn.to(dispositivo),
        TransactionMiniBatchDataset(
            alvos, GNN_FEATURE_NAMES, context_df=contexto
        ),
        dispositivo,
        1.0,
        batch_size=etapa2.FINAL_GNN_BATCH_SIZE,
        fanouts=etapa2.FINAL_GNN_FANOUTS,
    )
    if not (
        np.array_equal(rotulos_lstm.astype(np.int64), rotulos)
        and np.array_equal(rotulos_gnn.astype(np.int64), rotulos)
    ):
        raise RuntimeError("Scores em lote fora da ordem dos alvos.")
    return {
        "comportamental": comportamental,
        "temporal": temporal,
        "identidade": identidade,
        "limiar_if": float(modelo_if.decision_threshold_),
        "dispositivo": str(dispositivo),
        "vram_pico_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def _julgar(
    transacao: dict[str, Any],
    comportamental: float,
    temporal: float,
    identidade: float,
    aplicar_regra_madrugada: bool,
) -> tuple[str, float]:
    resultado = judge_tool(
        transaction=transacao,
        behavior_score=comportamental,
        temporal_score=temporal,
        identity_score=identidade,
        specialist_outputs={},
        aplicar_regra_madrugada=aplicar_regra_madrugada,
    )
    return DECISOES[resultado["decision"]], float(resultado["score"])


def _rodar_b(
    transacoes: list[dict[str, Any]], scores: dict[str, Any]
) -> dict[str, list]:
    saida: dict[str, list] = {
        "com_decisoes": [],
        "com_scores": [],
        "sem_decisoes": [],
        "sem_scores": [],
    }
    for transacao, comportamental, temporal, identidade in zip(
        transacoes,
        scores["comportamental"],
        scores["temporal"],
        scores["identidade"],
        strict=True,
    ):
        trio = (float(comportamental), float(temporal), float(identidade))
        for prefixo, regra in (("com", True), ("sem", False)):
            decisao, score = _julgar(transacao, *trio, regra)
            saida[f"{prefixo}_decisoes"].append(decisao)
            saida[f"{prefixo}_scores"].append(score)
    return saida


def _rodar_a(transacoes: list[dict[str, Any]]) -> dict[str, list]:
    saida: dict[str, list] = {
        "com_decisoes": [],
        "com_scores": [],
        "sem_decisoes": [],
        "sem_scores": [],
        "comportamental": [],
        "temporal": [],
    }
    for numero, transacao in enumerate(transacoes, start=1):
        resultado = orchestrate_ml_only(transacao)
        trio = (
            float(resultado["behavior"]["score"]),
            float(resultado["temporal"]["score"]),
            float(resultado["identity"]["score"]),
        )
        saida["com_decisoes"].append(DECISOES[resultado["decision"]["decision"]])
        saida["com_scores"].append(float(resultado["decision"]["score"]))
        decisao, score = _julgar(transacao, *trio, False)
        saida["sem_decisoes"].append(decisao)
        saida["sem_scores"].append(score)
        saida["comportamental"].append(trio[0])
        saida["temporal"].append(trio[1])
        if numero % 500 == 0:
            print(f"A: {numero}/{len(transacoes)}", flush=True)
    return saida


def _linhas_de_base(
    transacoes: list[dict[str, Any]],
    rotulos: list[int],
    tipos: list[str],
    scores_if: Iterable[float],
    limiar_if: float,
) -> dict[str, Any]:
    scores_if = [float(score) for score in scores_if]
    return {
        "regra_banco": comparar(
            ["BLOQUEADO" if regra_banco(t) else "APROVADO" for t in transacoes],
            rotulos,
            None,
            tipos,
        ),
        "if_sozinho": comparar(
            ["BLOQUEADO" if s >= limiar_if else "APROVADO" for s in scores_if],
            rotulos,
            scores_if,
            tipos,
        ),
    }


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def _ram_pico_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _modulos_llm_carregados() -> list[str]:
    return sorted(
        nome
        for nome in sys.modules
        if any(nome == p or nome.startswith(f"{p}.") for p in PREFIXOS_LLM)
    )


@contextmanager
def _sem_llm():
    """Qualquer tentativa de usar o caminho com LLM derruba a execução."""

    def recusar(*_args, **_kwargs):
        raise RuntimeError("A avaliação não pode chamar LLM.")

    chave = os.environ.get("GOOGLE_API_KEY")
    juiz = fluxo_multiagente.llm_judge_tool
    profundo = fluxo_multiagente.deep_multiagent_tool
    os.environ["GOOGLE_API_KEY"] = ""
    fluxo_multiagente.llm_judge_tool = recusar
    fluxo_multiagente.deep_multiagent_tool = recusar
    try:
        yield
    finally:
        fluxo_multiagente.llm_judge_tool = juiz
        fluxo_multiagente.deep_multiagent_tool = profundo
        if chave is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = chave


def _conferir_ambiente() -> None:
    """Falha antes de reservar ou ler o teste se faltar algum artefato."""
    from src.modelos.treinamento import treinar_modelos as etapa2

    for caminho in (
        PROCESSED_DATA_PATH,
        MODEL_DIR / "lstm_model.pt",
        MODEL_DIR / "gnn_model.pt",
    ):
        if not caminho.exists():
            raise FileNotFoundError(f"Artefato ausente: {caminho}")
    if not hasattr(load_pickle_model("isolation_forest.pkl"), "decision_threshold_"):
        raise ValueError("isolation_forest.pkl não tem o limiar salvo.")
    if etapa2._ram_snapshot()["available_gib"] < 2:
        raise MemoryError("RAM disponível abaixo de 2 GiB.")


def _caminho_relatorio(split: str) -> Path:
    return DIR_REPORTS / f"etapa4_avaliacao_{split}.json"


def _gravar(caminho: Path, conteudo: dict[str, Any]) -> None:
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def _reservar_teste(caminho: Path, commit: str) -> None:
    """Cria o relatório antes de ler o teste; uma falha não libera nova leitura."""
    try:
        descritor = os.open(caminho, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as erro:
        raise SystemExit(
            f"{caminho} já existe: o teste só pode ser lido uma vez. "
            "Nada foi lido. Para decidir o que fazer, fale com o dev."
        ) from erro
    with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
        json.dump({"status": "em_andamento", "commit": commit}, arquivo)


def avaliar(split: str, amostra_a: int = AMOSTRA_A_VALIDACAO) -> dict[str, Any]:
    """Executa uma avaliação e grava o relatório do split."""
    if split not in {"validacao", "teste"}:
        raise ValueError("split deve ser 'validacao' ou 'teste'.")
    commit = _commit()
    caminho = _caminho_relatorio(split)
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    if split == "teste" and caminho.exists():
        raise SystemExit(
            f"{caminho} já existe: o teste só pode ser lido uma vez. "
            "Nada foi lido. Para decidir o que fazer, fale com o dev."
        )
    _conferir_ambiente()
    if split == "teste":
        _reservar_teste(caminho, commit)
    inicio = time.perf_counter()
    try:
        with _sem_llm():
            relatorio = _executar(split, amostra_a, commit, inicio)
    except Exception as erro:
        if split == "teste":
            _gravar(
                caminho,
                {
                    "status": "falhou",
                    "commit": commit,
                    "erro": f"{type(erro).__name__}: {erro}",
                },
            )
        raise
    _gravar(caminho, relatorio)
    print(f"Relatório salvo em {caminho}", flush=True)
    return relatorio


def _executar(
    split: str, amostra_a: int, commit: str, inicio: float
) -> dict[str, Any]:
    contexto, alvos = carregar_split(split)
    tempos: dict[str, float] = {"leitura": time.perf_counter() - inicio}

    marco = time.perf_counter()
    scores_b = calcular_scores_b(contexto, alvos)
    tempos["modelos_b"] = time.perf_counter() - marco
    del contexto

    transacoes = alvos[COLUNAS_TRANSACAO].to_dict("records")
    rotulos = [int(valor) for valor in alvos["isFraud"]]
    tipos = [str(valor) for valor in alvos["type"]]

    marco = time.perf_counter()
    juiz_b = _rodar_b(transacoes, scores_b)
    tempos["juiz_b"] = time.perf_counter() - marco

    if split == "validacao":
        escolhidas = np.sort(
            np.random.default_rng(SEED).choice(
                len(transacoes), size=min(amostra_a, len(transacoes)), replace=False
            )
        )
    else:
        escolhidas = np.arange(len(transacoes))
    transacoes_a = [transacoes[i] for i in escolhidas]
    rotulos_a = [rotulos[i] for i in escolhidas]
    tipos_a = [tipos[i] for i in escolhidas]

    marco = time.perf_counter()
    juiz_a = _rodar_a(transacoes_a)
    tempos["sistema_a"] = time.perf_counter() - marco

    comparacoes = {
        "A_com_regra": comparar(
            juiz_a["com_decisoes"], rotulos_a, juiz_a["com_scores"], tipos_a
        ),
        "A_sem_regra": comparar(
            juiz_a["sem_decisoes"], rotulos_a, juiz_a["sem_scores"], tipos_a
        ),
        "B_com_regra": comparar(
            juiz_b["com_decisoes"], rotulos, juiz_b["com_scores"], tipos
        ),
        "B_sem_regra": comparar(
            juiz_b["sem_decisoes"], rotulos, juiz_b["sem_scores"], tipos
        ),
        **_linhas_de_base(
            transacoes,
            rotulos,
            tipos,
            scores_b["comportamental"],
            scores_b["limiar_if"],
        ),
    }
    e_amostra = len(escolhidas) < len(transacoes)
    na_amostra = None
    if e_amostra:
        na_amostra = {
            "B_com_regra": comparar(
                [juiz_b["com_decisoes"][i] for i in escolhidas],
                rotulos_a,
                [juiz_b["com_scores"][i] for i in escolhidas],
                tipos_a,
            ),
            "B_sem_regra": comparar(
                [juiz_b["sem_decisoes"][i] for i in escolhidas],
                rotulos_a,
                [juiz_b["sem_scores"][i] for i in escolhidas],
                tipos_a,
            ),
            **_linhas_de_base(
                transacoes_a,
                rotulos_a,
                tipos_a,
                scores_b["comportamental"][escolhidas],
                scores_b["limiar_if"],
            ),
        }
    if_a = np.asarray(juiz_a["comportamental"], dtype=np.float64)
    if_b = np.asarray(scores_b["comportamental"], dtype=np.float64)[escolhidas]
    tempos["total"] = time.perf_counter() - inicio
    return {
        "status": "concluido",
        "data": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "split": split,
        "linhas": len(alvos),
        "step_min": int(alvos["step"].min()),
        "step_max": int(alvos["step"].max()),
        "sistema_a": {
            "linhas": len(escolhidas),
            "e_amostra": e_amostra,
            "seed_amostra": SEED if e_amostra else None,
            "entrada": ENTRADA_A,
            "scores_distintos_lstm": len(set(juiz_a["temporal"])),
        },
        "sistema_b": {
            "linhas": len(alvos),
            "entrada": ENTRADA_B,
            "dispositivo": scores_b["dispositivo"],
        },
        "limiar_if": scores_b["limiar_if"],
        "pr_auc_modelos_b": {
            "isolation_forest": pr_auc(rotulos, scores_b["comportamental"]),
            "lstm": pr_auc(rotulos, scores_b["temporal"]),
            "gnn": pr_auc(rotulos, scores_b["identidade"]),
        },
        "diferenca_max_if_a_b": float(np.abs(if_a - if_b).max()) if len(if_a) else None,
        "comparacoes": comparacoes,
        "comparacoes_na_amostra_a": na_amostra,
        "tempo_segundos": tempos,
        "ram_pico_gib": _ram_pico_gib(),
        "vram_pico_gib": scores_b["vram_pico_gib"],
        "modulos_llm_carregados": _modulos_llm_carregados(),
    }


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validacao", "teste"))
    avaliar(parser.parse_args(argv).split)


if __name__ == "__main__":
    _cli()
