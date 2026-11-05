"""Avaliação da Etapa 4, sem dados reais e sem modelos reais."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.avaliacao import avaliar_scores as av
from src.ferramentas import ferramentas_agentes as tools
from src.orquestacao import fluxo_multiagente as flow


def _chamar_juiz(step, amount, tipo="PIX_P2P", score=0.1, **kwargs):
    return tools.judge_tool(
        {"step": step, "amount": amount, "type": tipo}, score, score, score, {}, **kwargs
    )


@pytest.mark.parametrize(
    ("amount", "saldo", "esperado"),
    [
        (9000.0, 10000.0, True),  # 0,9 exato
        (8999.99, 10000.0, False),
        (5000.0, 5000.0, True),  # 5000 exato
        (4999.99, 4999.99, False),
        (5000.0, 6000.0, False),  # valor alto, mas abaixo de 90% do saldo
    ],
)
def test_regra_banco_nos_limites(amount, saldo, esperado):
    assert av.regra_banco({"amount": amount, "oldbalanceOrg": saldo}) is esperado


def test_metricas_em_exemplo_feito_a_mao():
    decisoes = ["BLOQUEADO", "BLOQUEADO", "REVISÃO", "REVISÃO", "APROVADO", "APROVADO"]
    rotulos = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]

    resumo = av.resumir(decisoes, rotulos, scores)

    assert resumo["matriz"] == {
        "APROVADO": {"fraude": 1, "legitima": 1},
        "REVISÃO": {"fraude": 1, "legitima": 1},
        "BLOQUEADO": {"fraude": 1, "legitima": 1},
    }
    so_bloqueio = resumo["bloqueado_e_fraude"]
    assert (so_bloqueio["tp"], so_bloqueio["fp"], so_bloqueio["fn"]) == (1, 1, 2)
    assert so_bloqueio["precisao"] == pytest.approx(1 / 2)
    assert so_bloqueio["recall"] == pytest.approx(1 / 3)
    assert so_bloqueio["f1"] == pytest.approx(2 / 5)
    com_revisao = resumo["bloqueado_ou_revisao_e_fraude"]
    assert (com_revisao["tp"], com_revisao["fp"], com_revisao["fn"]) == (2, 2, 1)
    assert com_revisao["recall"] == pytest.approx(2 / 3)
    assert resumo["taxa_revisao"] == pytest.approx(2 / 6)
    # Fraudes nas posições 1, 3 e 5: (1/1 + 2/3 + 3/5) / 3.
    assert resumo["pr_auc_score_final"] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)


def test_metricas_sem_positivos_nao_dividem_por_zero():
    resumo = av.resumir(["APROVADO", "APROVADO"], [0, 0], [0.1, 0.2])

    assert resumo["bloqueado_e_fraude"]["precisao"] == 0.0
    assert resumo["bloqueado_e_fraude"]["recall"] == 0.0
    assert resumo["pr_auc_score_final"] is None


def test_comparar_separa_por_tipo():
    resultado = av.comparar(
        ["BLOQUEADO", "APROVADO", "REVISÃO"], [1, 0, 1], [0.9, 0.1, 0.6], ["TED", "TED", "TEF"]
    )

    assert sorted(resultado["por_tipo"]) == ["TED", "TEF"]
    assert resultado["por_tipo"]["TED"]["n"] == 2
    assert resultado["por_tipo"]["TEF"]["taxa_revisao"] == 1.0


def test_juiz_com_regra_bloqueia_e_sem_regra_nao():
    com = _chamar_juiz(3, 20000.0)
    sem = _chamar_juiz(3, 20000.0, aplicar_regra_madrugada=False)

    assert (com["decision"], com["score"]) == ("Recusada", 0.95)
    assert sem["decision"] == "Aprovada"
    assert sem["score"] == pytest.approx(0.1)


def test_juiz_sem_regra_desliga_o_bonus():
    # BOLETO não bloqueia de imediato, mas recebe o bônus de 0,25.
    com = _chamar_juiz(3, 20000.0, tipo="BOLETO")
    sem = _chamar_juiz(3, 20000.0, tipo="BOLETO", aplicar_regra_madrugada=False)

    assert com["score"] == pytest.approx(0.35)
    assert sem["score"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("step", "amount", "tipo", "score", "decisao", "score_final"),
    [
        (3, 20000.0, "PIX_P2P", 0.1, "Recusada", 0.95),
        (3, 20000.0, "BOLETO", 0.1, "Aprovada", 0.35),
        (3, 20000.0, "BOLETO", 0.3, "Revisão Humana necessária", 0.55),
        (3, 20000.0, "BOLETO", 0.9, "Recusada", 1.0),
        (3, 9999.99, "PIX_P2P", 0.1, "Aprovada", 0.1),
        (12, 20000.0, "PIX_P2P", 0.6, "Revisão Humana necessária", 0.6),
        (12, 20000.0, "PIX_P2P", 0.9, "Recusada", 0.9),
    ],
)
def test_juiz_com_regra_mantem_o_comportamento_atual(
    step, amount, tipo, score, decisao, score_final
):
    padrao = _chamar_juiz(step, amount, tipo, score)
    explicito = _chamar_juiz(step, amount, tipo, score, aplicar_regra_madrugada=True)

    assert padrao == explicito
    assert padrao["decision"] == decisao
    assert padrao["score"] == pytest.approx(score_final)


def test_fora_da_madrugada_a_regra_nao_muda_nada():
    com = _chamar_juiz(12, 20000.0, score=0.6)
    sem = _chamar_juiz(12, 20000.0, score=0.6, aplicar_regra_madrugada=False)

    assert com == sem


def _especialistas_falsos(monkeypatch, chamadas=None):
    def especialista(nome, score):
        def _tool(_transacao):
            if chamadas is not None:
                chamadas[nome] = chamadas.get(nome, 0) + 1
            return {"score": score, "label": nome, "details": {"explanation": ""}}

        return _tool

    monkeypatch.setattr(flow, "behavior_specialist_tool", especialista("if", 0.4))
    monkeypatch.setattr(flow, "temporal_specialist_tool", especialista("lstm", 0.5))
    monkeypatch.setattr(flow, "identity_specialist_tool", especialista("gnn", 0.6))


def test_modo_fast_repassa_o_parametro_com_padrao_true(monkeypatch):
    _especialistas_falsos(monkeypatch)
    transacao = {"step": 3, "amount": 20000.0, "type": "PIX_P2P"}

    assert flow.orchestrate_ml_only(transacao)["decision"]["score"] == 0.95
    sem = flow.orchestrate_ml_only(transacao, aplicar_regra_madrugada=False)
    assert sem["decision"]["score"] == pytest.approx(0.35 * 0.4 + 0.3 * 0.5 + 0.35 * 0.6)


def _preparar_base(monkeypatch, tmp_path):
    linhas = []
    for indice, step in enumerate([10, 20, 551, 552, 553, 554, 600, 601, 602, 603]):
        linhas.append(
            {
                "step": step,
                "type": "PIX_P2P" if indice % 2 else "TED",
                "amount": 100.0 + indice,
                "nameOrig": f"C{indice}",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": f"C{indice + 100}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": indice % 2,
            }
        )
    caminho = tmp_path / "dados.parquet"
    pd.DataFrame(linhas).to_parquet(caminho)
    monkeypatch.setattr(av, "PROCESSED_DATA_PATH", caminho)
    monkeypatch.setattr(av, "DIR_REPORTS", tmp_path / "reports")
    monkeypatch.setattr(av, "_conferir_ambiente", lambda: None)
    monkeypatch.setattr(av, "_commit", lambda: "abc123")


def _scores_b_falsos(chamadas):
    def _calcular(_contexto, alvos):
        chamadas["modelos_b"] = chamadas.get("modelos_b", 0) + 1
        n = len(alvos)
        return {
            "comportamental": np.full(n, 0.4),
            "temporal": np.linspace(0.1, 0.9, n),
            "identidade": np.linspace(0.9, 0.1, n),
            "limiar_if": 0.5,
            "dispositivo": "cpu",
            "vram_pico_gib": 0.0,
        }

    return _calcular


def test_validacao_nunca_le_step_de_teste(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)

    contexto, alvos = av.carregar_split("validacao")

    assert contexto["step"].max() < 600
    assert alvos["step"].tolist() == [551, 552, 553, 554]
    assert "isFraud" not in contexto


def test_teste_usa_contexto_sem_rotulos_e_alvos_a_partir_de_600(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)

    contexto, alvos = av.carregar_split("teste")

    assert alvos["step"].min() == 600
    assert len(contexto) == 10
    assert "isFraud" not in contexto
    assert set(alvos.index) <= set(contexto.index)


def test_relatorio_identifica_comparacoes_e_entradas_de_a(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    _especialistas_falsos(monkeypatch)
    monkeypatch.setattr(av, "calcular_scores_b", _scores_b_falsos({}))

    relatorio = av.avaliar("validacao", amostra_a=2)

    assert sorted(relatorio["comparacoes"]) == [
        "A_com_regra",
        "A_sem_regra",
        "B_com_regra",
        "B_sem_regra",
        "if_sozinho",
        "regra_banco",
    ]
    assert relatorio["sistema_a"]["e_amostra"] is True
    assert relatorio["sistema_a"]["linhas"] == 2
    assert relatorio["sistema_a"]["seed_amostra"] == 42
    assert relatorio["sistema_a"]["entrada"]["lstm"].startswith("matriz de zeros")
    assert relatorio["sistema_a"]["entrada"]["gnn"].startswith("um nó")
    assert relatorio["comparacoes"]["A_com_regra"]["n"] == 2
    assert relatorio["comparacoes"]["B_com_regra"]["n"] == 4
    assert relatorio["comparacoes_na_amostra_a"]["B_com_regra"]["n"] == 2
    assert relatorio["commit"] == "abc123"
    assert relatorio["linhas"] == 4
    salvo = json.loads(av._caminho_relatorio("validacao").read_text(encoding="utf-8"))
    assert salvo["status"] == "concluido"
    assert "tempo_segundos" in salvo and "ram_pico_gib" in salvo and "data" in salvo


def test_cada_sistema_passa_uma_vez_pelos_modelos_e_o_juiz_duas(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    chamadas = {}
    _especialistas_falsos(monkeypatch, chamadas)
    monkeypatch.setattr(av, "calcular_scores_b", _scores_b_falsos(chamadas))
    regras = []
    juiz_real = tools.judge_tool

    def juiz_espiao(**kwargs):
        regras.append(kwargs.get("aplicar_regra_madrugada", True))
        return juiz_real(**kwargs)

    monkeypatch.setattr(av, "judge_tool", juiz_espiao)
    monkeypatch.setattr(flow, "judge_tool", juiz_espiao)

    av.avaliar("validacao", amostra_a=3)

    assert chamadas == {"if": 3, "lstm": 3, "gnn": 3, "modelos_b": 1}
    # B: 4 transações x 2. A: 3 no modo fast (com regra) + 3 sem regra.
    assert len(regras) == 8 + 6
    assert regras.count(True) == regras.count(False) == 7


def test_a_e_b_usam_o_mesmo_judge_tool():
    assert av.judge_tool is tools.judge_tool
    assert flow.judge_tool is tools.judge_tool


def test_if_da_o_mesmo_score_em_a_e_b():
    from src.ferramentas.inferencia_modelos import isolation_forest_score
    from src.modelos.treinamento import treinar_modelos as etapa2
    from src.preprocessamento.carregar_dataset import IF_FEATURE_NAMES

    class ModeloFalso:
        def decision_function(self, x):
            return np.asarray(x, dtype=np.float64).sum(axis=1) / 1000.0

    quadro = pd.DataFrame(
        [
            {"step": 3, "type": "TED", "amount": 120.5, "nameOrig": "C1", "oldbalanceOrg": 900.0},
            {"step": 12, "type": "PIX_P2P", "amount": 7.25, "nameOrig": "M2", "oldbalanceOrg": 0.0},
        ]
    )
    modelo = ModeloFalso()

    em_a = [isolation_forest_score(t, modelo) for t in quadro.to_dict("records")]
    em_b = etapa2._if_scores(modelo, etapa2._feature_matrix(quadro, IF_FEATURE_NAMES))

    assert em_a == pytest.approx(em_b.tolist(), abs=1e-12)


def test_teste_recusa_segunda_execucao_sem_ler_dados(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    _especialistas_falsos(monkeypatch)
    monkeypatch.setattr(av, "calcular_scores_b", _scores_b_falsos({}))

    primeiro = av.avaliar("teste")
    assert primeiro["sistema_a"]["e_amostra"] is False
    assert primeiro["sistema_a"]["linhas"] == 4
    assert primeiro["comparacoes_na_amostra_a"] is None

    monkeypatch.setattr(
        av, "carregar_split", lambda _split: pytest.fail("o teste foi lido de novo")
    )
    with pytest.raises(SystemExit, match="só pode ser lido uma vez"):
        av.avaliar("teste")


def test_falha_na_leitura_do_teste_nao_libera_nova_execucao(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    _especialistas_falsos(monkeypatch)

    def falhar(_contexto, _alvos):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(av, "calcular_scores_b", falhar)

    with pytest.raises(RuntimeError, match="falha simulada"):
        av.avaliar("teste")

    salvo = json.loads(av._caminho_relatorio("teste").read_text(encoding="utf-8"))
    assert salvo["status"] == "falhou"
    with pytest.raises(SystemExit, match="só pode ser lido uma vez"):
        av.avaliar("teste")


def test_conferencia_falha_antes_de_reservar_o_teste(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)

    def faltando():
        raise FileNotFoundError("Artefato ausente")

    monkeypatch.setattr(av, "_conferir_ambiente", faltando)

    with pytest.raises(FileNotFoundError):
        av.avaliar("teste")

    assert not av._caminho_relatorio("teste").exists()


def test_validacao_pode_repetir(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    _especialistas_falsos(monkeypatch)
    monkeypatch.setattr(av, "calcular_scores_b", _scores_b_falsos({}))

    av.avaliar("validacao", amostra_a=1)
    av.avaliar("validacao", amostra_a=1)

    assert av._caminho_relatorio("validacao").exists()


def test_nenhum_caminho_chama_llm(monkeypatch, tmp_path):
    _preparar_base(monkeypatch, tmp_path)
    _especialistas_falsos(monkeypatch)
    monkeypatch.setattr(av, "calcular_scores_b", _scores_b_falsos({}))
    monkeypatch.setenv("GOOGLE_API_KEY", "valor-de-teste")
    falhar = lambda *_a, **_k: pytest.fail("LLM não esperado")
    monkeypatch.setattr(flow, "llm_judge_tool", falhar)
    monkeypatch.setattr(flow, "deep_multiagent_tool", falhar)
    chaves = []
    original = av._executar

    def executar(*args):
        chaves.append(os.environ["GOOGLE_API_KEY"])
        with pytest.raises(RuntimeError, match="não pode chamar LLM"):
            flow.llm_judge_tool()
        return original(*args)

    monkeypatch.setattr(av, "_executar", executar)

    av.avaliar("validacao", amostra_a=2)

    assert chaves == [""]
    assert os.environ["GOOGLE_API_KEY"] == "valor-de-teste"
    assert flow.llm_judge_tool is falhar


def test_cli_exige_split_valido(monkeypatch):
    chamadas = []
    monkeypatch.setattr(av, "avaliar", chamadas.append)

    av._cli(["--split", "validacao"])
    assert chamadas == ["validacao"]
    with pytest.raises(SystemExit):
        av._cli(["--split", "treino"])
    with pytest.raises(SystemExit):
        av._cli([])
