"""Testes do dashboard: rótulos, cliente da API, lote, valores iniciais e telas."""

from pathlib import Path

import pandas as pd
import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api import main as api
from src.dashboard import app as dashboard
from src.dashboard import cliente_api, rotulos, valores_iniciais
from src.dashboard.cliente_api import APIForaDoAr, ErroAPI
from src.dashboard.lote import (
    calcular_metricas,
    montar_resultados,
    montar_resultados_individuais,
    transacao_da_linha,
)
from src.ferramentas.leitor_arquivo_lote import amostrar_lote, ler_arquivo_lote
from src.preprocessamento.carregar_dataset import TRANSACTION_TYPES

APP = str(Path(dashboard.__file__))
CONFIG_OK = {"llm_disponivel": True, "modelo_llm": "gemini-teste", "limite_lote": 500}


# ---------- rótulos ----------


def test_todos_os_tipos_tem_nome_e_descricao():
    assert set(rotulos.TIPOS) == set(TRANSACTION_TYPES)
    assert rotulos.nome_tipo("PIX_P2P") == "Pix entre pessoas"
    assert rotulos.nome_tipo("CARTAO_ECOMMERCE") == "Cartão em compra online"
    assert rotulos.nome_tipo("DESCONHECIDO") == "DESCONHECIDO"
    assert all(rotulos.descricao_tipo(tipo) for tipo in TRANSACTION_TYPES)


def test_rotulos_das_caracteristicas_do_top3():
    assert rotulos.rotulo_feature("type_CARTAO_PRESENCIAL") == "Pagamento presencial com cartão"
    assert rotulos.rotulo_feature("hora_norm") == "Horário da transação"
    assert rotulos.rotulo_feature("amount_to_balance_ratio") == "Valor em relação ao saldo"
    assert rotulos.rotulo_feature("type_PIX_P2P") == "Pix entre pessoas"
    assert rotulos.rotulo_feature("campo_novo") == "Campo novo"
    assert rotulos.valor_feature("hora_norm", 3 / 23) == "03h"
    assert rotulos.valor_feature("is_madrugada", 1.0) == "sim"
    assert rotulos.valor_feature("type_TED", 0.0) == "não"
    assert rotulos.valor_feature("amount", 1234.5) == "R$ 1.234,50"
    assert rotulos.valor_feature("amount_to_balance_ratio", 0.9375) == "93,8% do saldo"
    assert rotulos.valor_feature("outra", 1.23456) == "1,235"


def test_direcao_do_top3():
    assert rotulos.direcao_feature("acima") == "acima do normal"
    assert rotulos.direcao_feature("abaixo") == "abaixo do normal"
    assert rotulos.direcao_feature("na_media") == "na média"
    assert rotulos.direcao_feature("acima", desvio_zero=True) == "sem variação no treino"


def test_traduzir_colunas_do_arquivo():
    df = pd.DataFrame({"type": ["PIX_P2P"], "amount": [10.0], "nameOrig": ["C1"]})
    traduzido = rotulos.traduzir_colunas(df)
    assert list(traduzido.columns) == ["tipo", "valor", "conta de origem"]
    assert traduzido.loc[0, "tipo"] == "Pix entre pessoas"
    assert df.loc[0, "type"] == "PIX_P2P"  # entrada intacta


# ---------- cliente da API ----------


class Resposta:
    def __init__(self, status=200, corpo=None):
        self.status_code = status
        self._corpo = corpo

    def json(self):
        if self._corpo is None:
            raise ValueError("sem json")
        return self._corpo


def usar_http(monkeypatch, funcao):
    monkeypatch.setattr(cliente_api, "_http", funcao)
    cliente_api.limpar_cache()


def test_api_fora_do_ar_mostra_endereco_e_comando(monkeypatch):
    def falha(*_a, **_k):
        raise requests.ConnectionError("recusada")

    usar_http(monkeypatch, falha)
    with pytest.raises(APIForaDoAr) as erro:
        cliente_api.obter_config()
    assert cliente_api.url_base() in str(erro.value)
    assert cliente_api.COMANDO_SUBIR_API in str(erro.value)


def test_api_lenta_nao_e_tratada_como_fora_do_ar(monkeypatch):
    def lenta(*_a, **_k):
        raise requests.ReadTimeout("lenta")

    usar_http(monkeypatch, lenta)
    with pytest.raises(ErroAPI) as erro:
        cliente_api.analisar_transacao({})
    assert not isinstance(erro.value, APIForaDoAr)
    assert "demorou" in str(erro.value)


@pytest.mark.parametrize(
    "status, detalhe, trecho",
    [
        (422, [{"loc": ["body", "oldbalanceOrg"], "type": "missing", "msg": "x"}], "saldo antes (origem)"),
        (422, [{"loc": ["body", "amount"], "type": "greater_than_equal", "msg": "ruim"}], "campo 'valor' inválido"),
        (503, "Gemini não configurado. Defina GOOGLE_API_KEY no .env.", "GOOGLE_API_KEY"),
        (503, "Modelo treinado não encontrado.", "modelos treinados"),
        (502, "Falha ao executar análise production.", "falhou dentro da API"),
        (422, "Colunas obrigatórias ausentes: amount", "Colunas obrigatórias ausentes"),
        (500, None, "erro 500"),
    ],
)
def test_erros_da_api_viram_mensagem_em_portugues(monkeypatch, status, detalhe, trecho):
    usar_http(monkeypatch, lambda *_a, **_k: Resposta(status, {"detail": detalhe}))
    with pytest.raises(ErroAPI) as erro:
        cliente_api.analisar_transacao({})
    assert trecho in str(erro.value)
    assert erro.value.status == status


def test_resposta_sem_json_e_config_em_cache(monkeypatch):
    chamadas = []

    def http(metodo, caminho, **_k):
        chamadas.append(caminho)
        return Resposta(200, CONFIG_OK) if caminho == "/config" else Resposta(200, None)

    usar_http(monkeypatch, http)
    assert cliente_api.obter_config() == CONFIG_OK
    assert cliente_api.obter_config() == CONFIG_OK
    assert chamadas == ["/config"]
    with pytest.raises(ErroAPI, match="não é JSON"):
        cliente_api.obter_estatisticas()


def test_historico_envia_filtro_e_limite(monkeypatch):
    vistos = {}

    def http(metodo, caminho, **kwargs):
        vistos.update(caminho=caminho, **kwargs)
        return Resposta(200, [])

    usar_http(monkeypatch, http)
    assert cliente_api.obter_historico(50, "BLOQUEADO") == []
    assert vistos["params"] == {"limit": 50, "decisao": "BLOQUEADO"}
    cliente_api.obter_historico(10)
    assert vistos["params"] == {"limit": 10}


def test_url_da_api_vem_do_ambiente(monkeypatch):
    monkeypatch.setenv("API_URL", "http://exemplo:9000/")
    assert cliente_api.url_base() == "http://exemplo:9000"


# ---------- contrato com a API real (orquestrador simulado, sem Gemini) ----------


@pytest.fixture
def api_real(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "historico.db")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-teste")
    http = TestClient(api.app)

    def passar(metodo, caminho, timeout=None, **kwargs):
        return http.request(metodo, caminho, **kwargs)

    usar_http(monkeypatch, passar)

    class Orquestrador:
        def orchestrate_ml_only(self, tx):
            # decide pelo valor, para provar que cada índice casa com a linha certa
            decisao = "Recusada" if tx["amount"] >= 500 else "Aprovada"
            esp = {"score": 0.9 if decisao == "Recusada" else 0.1, "label": "x",
                   "details": {"explanation": "ok"}}
            return {"modo": "fast", "behavior": esp, "temporal": esp, "identity": esp,
                    "decision": {"decision": decisao, "score": esp["score"], "justification": "j"}}

    monkeypatch.setattr(api, "_get_orchestrator", lambda: Orquestrador())
    return http


def csv_lote(linhas=40):
    cab = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud"
    corpo = [
        f"{i % 24},{'TED' if i % 2 else 'PIX_P2P'},{i * 100 + 1},C{i:07d},9999,0,C{i + 500:07d},0,0,{int(i * 100 + 1 >= 500)}"
        for i in range(linhas)
    ]
    return "\n".join([cab, *corpo]).encode()


def test_lote_pelo_cliente_casa_indice_com_linha_amostrada(api_real):
    conteudo = csv_lote(40)
    resposta = cliente_api.analisar_lote_arquivo(conteudo, "lote.csv", 15)
    assert resposta["quantidade_usada"] == 15 and resposta["seed"] == 42

    df = ler_arquivo_lote(conteudo, "lote.csv")
    resultado = montar_resultados(amostrar_lote(df, 15, resposta["seed"]), resposta)

    assert len(resultado) == 15
    esperado = resultado["valor"].map(lambda v: "BLOQUEADO" if v >= 500 else "APROVADO")
    assert (resultado["decisao"] == esperado).all()
    metricas = calcular_metricas(resultado)
    assert metricas["estrita"]["precisao"] == 1.0 and metricas["estrita"]["recall"] == 1.0


def test_lote_pequeno_usa_todas_as_linhas_e_erros_sao_traduzidos(api_real):
    resposta = cliente_api.analisar_lote_arquivo(csv_lote(5), "lote.csv", 300)
    assert resposta["quantidade_usada"] == 5 and resposta["arquivo_menor_que_quantidade"]
    with pytest.raises(ErroAPI, match="Formato inválido"):
        cliente_api.analisar_lote_arquivo(b"a", "lote.pdf", 5)


def test_analise_individual_sem_chave_e_sem_saldo(api_real):
    with pytest.raises(ErroAPI, match="GOOGLE_API_KEY"):
        cliente_api.analisar_transacao({"amount": 10.0, "oldbalanceOrg": 100.0}, modo="production")
    with pytest.raises(ErroAPI, match="saldo antes"):
        cliente_api.analisar_transacao({"amount": 10.0})


def test_historico_e_config_pelo_cliente(api_real):
    cliente_api.analisar_lote_arquivo(csv_lote(6), "lote.csv", 6)
    assert cliente_api.obter_config()["modelo_llm"] == "gemini-teste"
    assert len(cliente_api.obter_historico(100)) == 6
    assert all(r["decisao"] == "BLOQUEADO" for r in cliente_api.obter_historico(100, "BLOQUEADO"))
    assert cliente_api.obter_estatisticas()["total_transacoes"] == 6


# ---------- lote: junção e métricas ----------


def amostra_e_resposta(decisoes, fraudes=None):
    amostra = pd.DataFrame(
        {"type": ["PIX_P2P"] * len(decisoes), "amount": range(1, len(decisoes) + 1),
         "nameOrig": "C1", "nameDest": "C2"}
    )
    if fraudes is not None:
        amostra["isFraud"] = fraudes
    itens = [
        {"indice": i, "decisao": d, "score_final": 0.5, "scores": {}, "erro": "falhou" if d == "ERRO" else ""}
        for i, d in enumerate(decisoes)
    ]
    return amostra, {"resultados": itens}


def test_montar_resultados_exige_mesmo_tamanho():
    amostra, resposta = amostra_e_resposta(["APROVADO", "BLOQUEADO"])
    with pytest.raises(ValueError, match="mesma versão"):
        montar_resultados(amostra.head(1), resposta)


def test_metricas_estrita_e_ampla():
    # real:       1          1          0          0          1
    decisoes = ["BLOQUEADO", "REVISÃO", "BLOQUEADO", "APROVADO", "ERRO"]
    amostra, resposta = amostra_e_resposta(decisoes, fraudes=[1, 1, 0, 0, 1])
    m = calcular_metricas(montar_resultados(amostra, resposta))

    assert m["avaliadas"] == 4 and m["excluidas"] == 1 and m["fraudes"] == 2
    assert m["estrita"]["precisao"] == 0.5 and m["estrita"]["recall"] == 0.5
    assert m["estrita"]["f1"] == 0.5
    assert m["ampla"]["precisao"] == pytest.approx(2 / 3) and m["ampla"]["recall"] == 1.0
    assert m["ampla"]["f1"] == pytest.approx(0.8)
    assert m["taxa_revisao"] == 0.25


def test_metricas_indefinidas_e_ausentes():
    amostra, resposta = amostra_e_resposta(["APROVADO", "APROVADO"], fraudes=[1, 0])
    m = calcular_metricas(montar_resultados(amostra, resposta))
    assert m["estrita"]["precisao"] is None and m["estrita"]["f1"] is None
    assert m["estrita"]["recall"] == 0.0

    sem_rotulo, resposta = amostra_e_resposta(["APROVADO"])
    assert calcular_metricas(montar_resultados(sem_rotulo, resposta)) is None
    so_erro, resposta = amostra_e_resposta(["ERRO"], fraudes=[1])
    assert calcular_metricas(montar_resultados(so_erro, resposta)) is None


# ---------- valores iniciais ----------


def test_valores_iniciais_vem_da_mediana_das_legitimas(tmp_path):
    csv = tmp_path / "base.csv"
    csv.write_text(
        "step,type,amount,oldbalanceOrg,isFraud\n"
        "1,TED,10,100,0\n2,TED,30,300,0\n3,TED,999,999999,1\n700,TED,5,5,0\n"
    )
    valores_iniciais.medianas_legitimas.cache_clear()
    try:
        medianas = valores_iniciais.medianas_legitimas(str(csv))
        assert medianas == {"TED": {"saldo": 200.0, "valor": 20.0}}
    finally:
        valores_iniciais.medianas_legitimas.cache_clear()


def test_valores_iniciais_sem_base_e_limitados(monkeypatch):
    monkeypatch.setattr(valores_iniciais, "medianas_legitimas", lambda: {})
    assert valores_iniciais.valores_do_tipo("TED", 1e6) == (100.0, 1000.0, False)
    monkeypatch.setattr(
        valores_iniciais, "medianas_legitimas", lambda: {"TED": {"saldo": 9e9, "valor": 9e9}}
    )
    assert valores_iniciais.valores_do_tipo("TED", 5000.0) == (5000.0, 5000.0, True)
    assert valores_iniciais.medianas_legitimas.__name__ == "<lambda>"


def test_valores_iniciais_arquivo_inexistente():
    valores_iniciais.medianas_legitimas.cache_clear()
    assert valores_iniciais.medianas_legitimas("/nao/existe.csv") == {}
    valores_iniciais.medianas_legitimas.cache_clear()


# ---------- auxiliares do app ----------


def test_novo_saldo_nao_fica_negativo():
    assert dashboard._novo_saldo(1000.0, 250.5) == 749.5
    assert dashboard._novo_saldo(100.0, 500.0) == 0.0


def test_transacao_aleatoria_e_valida():
    for _ in range(50):
        tx = dashboard.random_transaction()
        assert tx["type"] in TRANSACTION_TYPES and 0 <= tx["step"] <= 23
        assert 0 < tx["amount"] <= tx["oldbalanceOrg"] <= dashboard._MAX_FINANCIAL_VALUE


def test_codigo_sem_legado_paysim():
    texto = Path(APP).read_text(encoding="utf-8")
    for legado in ("CASH_IN", "REVISÃO MANUAL", "regra D2", "_minutes_to_hh_mm", "_MAX_UPLOAD_MB"):
        assert legado not in texto
    assert not hasattr(dashboard, "_is_valid_24h_time")
    assert not hasattr(dashboard, "run_analysis")


def test_formatar_historico_converte_horario_e_rotulos():
    df = dashboard.formatar_historico(
        [{"timestamp": "2026-11-06T15:00:00+00:00", "type": "TED", "amount": 10.0, "decisao": "APROVADO",
          "score_final": 0.1, "score_behavior": 0.1, "score_temporal": 0.1, "score_identity": 0.1,
          "duracao_ms": 5, "llm_usado": "NAO", "erro": None, "transaction_hash": "abc"}]
    )
    assert df.loc[0, "Data e hora"] == "06/11/2026 12:00:00"
    assert df.loc[0, "Usou Gemini"] == "não" and df.loc[0, "Erro"] == ""
    assert "transaction_hash" not in " ".join(df.columns)
    assert dashboard.formatar_historico([]).empty


# ---------- telas (AppTest com cliente da API simulado) ----------


def resposta_individual(**mudancas):
    base = {
        "decisao": "BLOQUEADO", "score_final": 0.91, "modo": "fast",
        "justificativa": "Bloqueada: valor alto na madrugada.",
        "scores": {"comportamental": 0.8, "temporal": 0.9, "identidade": 0.3},
        "agentes": {
            "comportamental": {"score": 0.8, "label": "behavior", "explicacao": "Explicação comportamental"},
            "temporal": {"score": 0.9, "label": "temporal", "explicacao": "Explicação temporal"},
            "identidade": {"score": 0.3, "label": "identity", "explicacao": "Explicação identidade"},
        },
        "explicabilidade": {"top3": [
            {"nome": "type_CARTAO_PRESENCIAL", "valor": 1.0, "z": 3.2, "direcao": "acima", "desvio_zero": False},
            {"nome": "hora_norm", "valor": 3 / 23, "z": -1.1, "direcao": "abaixo", "desvio_zero": False},
            {"nome": "amount_to_balance_ratio", "valor": 0.9, "z": 0.0, "direcao": "na_media", "desvio_zero": False},
        ]},
    }
    base.update(mudancas)
    return base


@pytest.fixture
def api_simulada(monkeypatch):
    """Cliente simulado; devolve o dicionário de chamadas para os testes."""
    st.cache_data.clear()
    estado = {"config": CONFIG_OK, "individual": resposta_individual(), "enviadas": [], "modos": [], "historico": [],
              "filtros": [], "stats": {"total_transacoes": 2, "score_medio": 0.4, "latencia_media_ms": 12.0,
                                       "contagem_por_decisao": {"APROVADO": 1, "BLOQUEADO": 1}}}

    def config():
        if isinstance(estado["config"], Exception):
            raise estado["config"]
        return estado["config"]

    def individual(tx, modo="fast"):
        estado["enviadas"].append(tx)
        estado["modos"].append(modo)
        if callable(estado["individual"]):
            return estado["individual"](tx, modo)
        if isinstance(estado["individual"], Exception):
            raise estado["individual"]
        return {**estado["individual"], "modo": modo}  # a API devolve o modo pedido

    def historico(limite, decisao=None):
        estado["filtros"].append((limite, decisao))
        return [r for r in estado["historico"] if decisao in (None, r["decisao"])]

    monkeypatch.setattr(cliente_api, "obter_config", config)
    monkeypatch.setattr(cliente_api, "analisar_transacao", individual)
    monkeypatch.setattr(cliente_api, "obter_historico", historico)
    def estatisticas():
        if isinstance(estado["stats"], Exception):
            raise estado["stats"]
        return estado["stats"]

    monkeypatch.setattr(cliente_api, "obter_estatisticas", estatisticas)
    return estado


def abrir():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def textos(elementos):
    return " ".join(str(e.value) for e in elementos)


def botao(at, trecho):
    return next(b for b in at.button if trecho in b.label)


def test_abas_na_ordem_e_chips(api_simulada):
    at = abrir()
    assert [t.label for t in at.tabs] == ["📦 Análise em Lote", "🎓 Aprenda Simulando", "🗂️ Histórico"]
    cabecalho = textos(at.markdown)
    assert "API online" in cabecalho and "Gemini ativo" in cabecalho
    assert not at.error
    assert "Os dados são sintéticos" in cabecalho


def test_api_fora_do_ar_avisa_e_desliga_botoes(api_simulada):
    api_simulada["config"] = APIForaDoAr(cliente_api.mensagem_fora_do_ar())
    at = abrir()
    assert "API offline" in textos(at.markdown)
    assert cliente_api.COMANDO_SUBIR_API in textos(at.error)
    assert botao(at, "Analisar (modo grátis").disabled
    assert "Suba a API" in textos(at.info)


def test_sem_gemini_a_simulacao_continua_gratuita_e_habilitada(api_simulada):
    api_simulada["config"] = {"llm_disponivel": False, "modelo_llm": None, "limite_lote": 500}
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    assert not botao(at, "Analisar (modo grátis").disabled
    assert api_simulada["modos"] == ["fast"]  # nunca pede production
    assert "GOOGLE_API_KEY" not in textos(at.warning)
    assert "Gemini sem chave" in textos(at.markdown)  # o chip continua informando


def test_analise_individual_envia_dados_e_mostra_resultado_na_ordem(api_simulada):
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    assert not at.exception and not at.error

    tx = api_simulada["enviadas"][0]
    assert tx["type"] == "PIX_P2P" and tx["step"] == 12
    contas = valores_iniciais.contas_do_dataset()
    assert tx["nameOrig"] in contas["origem"] and tx["nameDest"] in contas["destino"]
    assert api_simulada["modos"] == ["fast"]
    assert tx["oldbalanceOrg"] > 0 and tx["newbalanceOrig"] == round(tx["oldbalanceOrg"] - tx["amount"], 2)

    html = textos(at.markdown)
    posicoes = [html.index(t) for t in ("BLOQUEADO", "Bloqueada: valor alto", "Pagamento presencial com cartão",
                                        "Explicação comportamental")]
    assert posicoes == sorted(posicoes)  # decisão, justificativa, top-3, scores
    assert "acima do normal" in html and "abaixo do normal" in html and "na média" in html
    assert "03h" in html and "sim" in html
    assert "<b>não</b> é o motivo da decisão do modelo" in html  # aviso fixo do top-3
    assert "Modo usado: **fast**" in textos(at.caption)


def test_justificativa_vazia_mostra_aviso_sem_caixa(api_simulada):
    api_simulada["individual"] = resposta_individual(justificativa="   ")
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    assert "não devolveu justificativa" in textos(at.warning)
    assert '<div class="just-box"' not in textos(at.markdown)


def test_erro_da_api_na_analise_individual(api_simulada):
    api_simulada["individual"] = ErroAPI("Dados inválidos: faltou informar o campo 'saldo antes (origem)'.", 422)
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    assert "faltou informar" in textos(at.error)
    assert '<div class="just-box"' not in textos(at.markdown)


def test_trocar_tipo_muda_valores_iniciais(api_simulada, monkeypatch):
    monkeypatch.setattr(
        valores_iniciais, "medianas_legitimas",
        lambda: {"PIX_P2P": {"saldo": 3831.25, "valor": 36.53}, "TED": {"saldo": 41875.55, "valor": 956.01}},
    )
    at = abrir()
    assert at.number_input(key="f_old_PIX_P2P").value == 3831.25
    at.selectbox(key="f_tipo").select("TED").run()
    assert at.number_input(key="f_old_TED").value == 41875.55
    assert at.number_input(key="f_amount_TED").value == 956.01
    assert "mediana" in textos(at.caption)


def test_simular_preenche_formulario_e_limpa_resultado(api_simulada, monkeypatch):
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    assert "individual" in at.session_state
    botao(at, "Simular nova transação").click().run()
    assert not at.exception
    assert "individual" not in at.session_state
    tipo = at.session_state["f_tipo"]
    assert tipo in TRANSACTION_TYPES
    assert at.number_input(key=f"f_amount_{tipo}").value == at.session_state[f"f_amount_{tipo}"]


def df_lote():
    amostra, resposta = amostra_e_resposta(
        ["APROVADO", "REVISÃO", "BLOQUEADO", "BLOQUEADO", "ERRO"], fraudes=[0, 1, 1, 0, 1]
    )
    amostra["type"] = ["PIX_P2P", "TED", "TED", "BOLETO", "TEF"]
    amostra["comportamental"] = 0.5
    return montar_resultados(amostra, resposta), {
        "total": 5, "quantidade_usada": 5, "total_arquivo": 9, "seed": 42,
    }


def test_resultado_do_lote_mostra_totais_listas_e_metricas(api_simulada):
    df, resposta = df_lote()
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["lote"] = {"resposta": resposta, "df": df, "arquivo": "lote.csv"}
    at.run()
    assert not at.exception

    html = textos(at.markdown)
    for rotulo in ("Total analisado", "Aprovados", "Em revisão", "Bloqueados", "Erros"):
        assert rotulo in html
    assert "modo **fast** (sem Gemini" in textos(at.success)
    assert "seed 42" in textos(at.caption) and "não calcula top-3" in textos(at.caption)
    metricas = next(d.value for d in at.dataframe if "Leitura" in d.value.columns)
    assert list(metricas["Leitura"].str[:6]) == ["Estrit", "Ampla "]
    assert "indefinida" not in " ".join(metricas["Precisão"])
    rotulos_abas = [t.label for t in at.tabs]
    for trecho in ("Aprovado (1)", "Revisão (1)", "Bloqueado (2)", "Erro (1)"):
        assert any(trecho in r for r in rotulos_abas)
    nomes = " ".join(
        " ".join(d.value["Tipo"].astype(str)) for d in at.dataframe if "Tipo" in d.value.columns
    )
    assert "Pix entre pessoas" in nomes and "PIX_P2P" not in nomes


def test_lote_sem_resultado_mostra_orientacao(api_simulada):
    at = abrir()
    assert "Execute uma **Análise em Lote**" in textos(at.info)


def test_upload_invalido_mostra_erro_do_leitor(api_simulada):
    at = abrir()
    at.file_uploader[0].set_value(("lote.csv", b"amount,other\n1,2\n", "text/csv")).run()
    assert "Colunas obrigatórias ausentes" in textos(at.error)


def test_upload_e_analise_do_lote_completo(api_simulada, api_real):
    at = abrir()
    at.file_uploader[0].set_value(("lote.csv", csv_lote(30), "text/csv")).run()
    assert "seed 42" in textos(at.caption) or "menos linhas" in textos(at.caption)
    botao(at, "Análise em lote (sem Gemini, sem custo)").click().run()
    assert not at.exception and not at.error
    assert "modo **fast**" in textos(at.success)
    assert "lote" in at.session_state and len(at.session_state["lote"]["df"]) == 30


def test_historico_filtra_por_decisao_e_tipo(api_simulada):
    api_simulada["historico"] = [
        {"timestamp": "2026-11-06T15:00:00+00:00", "type": "TED", "amount": 10.0, "decisao": "APROVADO",
         "score_final": 0.1, "score_behavior": 0.1, "score_temporal": 0.1, "score_identity": 0.1,
         "duracao_ms": 5, "llm_usado": "NAO", "erro": None},
        {"timestamp": "2026-11-06T16:00:00+00:00", "type": "BOLETO", "amount": 99.0, "decisao": "BLOQUEADO",
         "score_final": 0.9, "score_behavior": 0.9, "score_temporal": 0.9, "score_identity": 0.9,
         "duracao_ms": 7, "llm_usado": "SIM", "erro": None},
    ]
    at = abrir()
    tabela = lambda: next(d.value for d in at.dataframe if "Data e hora" in d.value.columns)
    assert len(tabela()) == 2 and "Últimas 24 horas" in textos(at.markdown)

    at.selectbox(key="h_decisao").select("BLOQUEADO").run()
    assert api_simulada["filtros"][-1] == (100, "BLOQUEADO")
    assert list(tabela()["Tipo"]) == ["Boleto"]

    at.selectbox(key="h_decisao").select("Todas").run()
    at.selectbox(key="h_tipo").select("TED").run()
    assert list(tabela()["Tipo"]) == ["TED"]


def test_historico_vazio(api_simulada):
    at = abrir()
    assert "Nenhuma análise salva" in textos(at.info)


def test_historico_com_erro_da_api(api_simulada):
    api_simulada["stats"] = ErroAPI("A API demorou demais para responder. Tente de novo.")
    at = abrir()
    assert "demorou demais" in textos(at.error)


def test_listas_de_contas_vem_da_base_com_15_cada():
    valores_iniciais.contas_do_dataset.cache_clear()
    contas = valores_iniciais.contas_do_dataset()
    base = pd.read_csv(valores_iniciais.DATASET, usecols=["nameOrig", "nameDest"], dtype=str)
    assert len(contas["origem"]) == len(contas["destino"]) == 15
    assert len(set(contas["origem"])) == len(set(contas["destino"])) == 15
    assert set(contas["origem"]) <= set(base["nameOrig"])
    assert set(contas["destino"]) <= set(base["nameDest"])
    assert len({conta[0] for conta in contas["origem"]}) == 3  # 3 prefixos de conta
    valores_iniciais.contas_do_dataset.cache_clear()
    assert valores_iniciais.contas_do_dataset() == contas  # reprodutível (seed fixa)


def test_listas_de_contas_sem_base_usam_padrao():
    valores_iniciais.contas_do_dataset.cache_clear()
    try:
        contas = valores_iniciais.contas_do_dataset("/nao/existe.csv")
        assert contas == valores_iniciais.CONTAS_PADRAO
    finally:
        valores_iniciais.contas_do_dataset.cache_clear()


def test_contas_sao_selecoes_e_nao_texto_livre(api_simulada):
    at = abrir()
    contas = valores_iniciais.contas_do_dataset()
    assert list(at.selectbox(key="f_orig").options) == contas["origem"]
    assert list(at.selectbox(key="f_dest").options) == contas["destino"]
    assert not [t for t in at.text_input if t.key in ("f_orig", "f_dest")]
    # campos dentro do formulário só valem no envio: seleciona e envia na mesma execução
    at.selectbox(key="f_orig").select(contas["origem"][-1])
    botao(at, "Analisar (modo grátis").click().run()
    assert api_simulada["enviadas"][-1]["nameOrig"] == contas["origem"][-1]


def test_simular_escolhe_contas_da_lista(api_simulada):
    at = abrir()
    botao(at, "Simular nova transação").click().run()
    contas = valores_iniciais.contas_do_dataset()
    assert at.session_state["f_orig"] in contas["origem"]
    assert at.session_state["f_dest"] in contas["destino"]


def test_cliente_envia_modo_da_analise_individual(monkeypatch):
    vistos = []
    usar_http(monkeypatch, lambda m, c, **k: vistos.append(k["params"]) or Resposta(200, {"ok": 1}))
    cliente_api.analisar_transacao({})
    cliente_api.analisar_transacao({}, modo="production")
    assert vistos == [{"modo": "fast"}, {"modo": "production"}]


def test_analise_individual_fast_pelo_cliente_sem_chave(api_real, monkeypatch):
    class Orquestrador:
        def orchestrate_ml_only(self, tx):
            esp = {"score": 0.2, "label": "x", "details": {"explanation": "ok"}}
            return {"modo": "fast", "behavior": esp, "temporal": esp, "identity": esp,
                    "decision": {"decision": "Aprovada", "score": 0.2, "justification": "Aprovada pelo score."}}

    monkeypatch.setattr(api, "_get_orchestrator", lambda: Orquestrador())
    resposta = cliente_api.analisar_transacao({"amount": 10.0, "oldbalanceOrg": 100.0})
    assert resposta["modo"] == "fast" and len(resposta["explicabilidade"]["top3"]) == 3
    assert resposta["justificativa"] == "Aprovada pelo score."


# ---------- lote em dois modos ----------


def test_transacao_da_linha_omite_saldo_vazio_para_a_api_recusar():
    linha = pd.Series({"step": 30, "type": "TED", "amount": "10.5", "nameOrig": "C1", "nameDest": "C2",
                       "oldbalanceOrg": float("nan"), "newbalanceOrig": 5.0})
    tx = transacao_da_linha(linha)
    assert "oldbalanceOrg" not in tx
    assert tx["step"] == 30 and tx["amount"] == 10.5 and tx["oldbalanceDest"] == 0.0
    linha["oldbalanceOrg"] = 99.0
    assert transacao_da_linha(linha)["oldbalanceOrg"] == 99.0


def test_montar_resultados_individuais_com_erro_e_justificativa():
    amostra, _ = amostra_e_resposta(["x", "x"])
    df = montar_resultados_individuais(
        amostra, [resposta_individual(), {"erro": "Gemini fora"}]
    )
    assert list(df["decisao"]) == ["BLOQUEADO", "ERRO"]
    assert df.loc[0, "justificativa"].startswith("Bloqueada") and df.loc[1, "erro"] == "Gemini fora"
    assert df.loc[0, "comportamental"] == 0.8
    with pytest.raises(ValueError, match="respostas"):
        montar_resultados_individuais(amostra, [resposta_individual()])


def test_lote_comeca_no_modo_fast_com_limite_500(api_simulada):
    at = abrir()
    assert at.button_group(key="lote_modo").value == "fast"
    assert "Grátis, sem Gemini, até 500" in textos(at.caption)
    assert at.button_group(key="lote_modo").options == ["Fast", "Production"]  # o emoji inicial vira ícone
    at.file_uploader[0].set_value(("lote.csv", csv_lote(6), "text/csv")).run()
    assert at.number_input(key="lote_qtd_fast").max == 500
    assert botao(at, "sem Gemini, sem custo")


def test_modo_production_avisa_que_consome_cota_e_limita_a_10(api_simulada):
    at = abrir()
    at.button_group(key="lote_modo").set_value("production").run()
    assert "consome a cota" in textos(at.warning) and "uma vez por transação" in textos(at.warning)
    at.file_uploader[0].set_value(("lote.csv", csv_lote(30), "text/csv")).run()
    assert at.number_input(key="lote_qtd_production").max == 10
    assert botao(at, "usa Gemini, consome cota")


def test_modo_production_sem_chave_desliga_o_botao(api_simulada):
    api_simulada["config"] = {"llm_disponivel": False, "modelo_llm": None, "limite_lote": 500}
    at = abrir()
    at.button_group(key="lote_modo").set_value("production").run()
    at.file_uploader[0].set_value(("lote.csv", csv_lote(6), "text/csv")).run()
    assert botao(at, "usa Gemini, consome cota").disabled
    assert "GOOGLE_API_KEY" in textos(at.error)


def rodar_production(at, linhas=6):
    at.button_group(key="lote_modo").set_value("production").run()
    at.file_uploader[0].set_value(("lote.csv", csv_lote(linhas), "text/csv")).run()
    botao(at, "usa Gemini, consome cota").click().run()
    return at


def test_lote_production_chama_gemini_uma_vez_por_transacao(api_simulada):
    at = rodar_production(abrir(), linhas=6)
    assert not at.exception and not at.error
    assert api_simulada["modos"] == ["production"] * 6
    assert {tx["type"] for tx in api_simulada["enviadas"]} <= {"TED", "PIX_P2P"}
    estado = at.session_state["lote"]
    assert estado["modo"] == "production" and len(estado["df"]) == 6 and len(estado["detalhes"]) == 6
    assert "modo **production** (usou o Gemini)" in textos(at.success)
    # detalhe de uma transação mostra justificativa, top-3 e scores do Gemini
    html = textos(at.markdown)
    assert "Bloqueada: valor alto" in html and "Pagamento presencial com cartão" in html
    assert "Justificativa do juiz Gemini" in html
    tabelas = [d.value for d in at.dataframe if "Justificativa" in d.value.columns]
    assert tabelas and (tabelas[0]["Justificativa"] == "Bloqueada: valor alto na madrugada.").all()


def test_lote_production_para_no_503_e_guarda_o_que_ja_analisou(api_simulada):
    def resposta(tx, modo, chamadas=[]):
        chamadas.append(1)
        if len(chamadas) > 2:
            raise ErroAPI("O Gemini não está configurado na API.", 503)
        return resposta_individual()

    api_simulada["individual"] = resposta
    at = rodar_production(abrir(), linhas=6)
    assert len(api_simulada["enviadas"]) == 3  # parou na 3ª, sem gastar o resto
    estado = at.session_state["lote"]
    assert len(estado["df"]) == 3 and list(estado["df"]["decisao"]) == ["BLOQUEADO", "BLOQUEADO", "ERRO"]
    assert "Gemini não configurado" in textos(at.warning)


def test_lote_production_para_depois_de_3_erros_seguidos(api_simulada):
    def falha(tx, modo):
        raise ErroAPI("A análise falhou dentro da API.", 502)

    api_simulada["individual"] = falha
    at = rodar_production(abrir(), linhas=8)
    assert len(api_simulada["enviadas"]) == 3
    assert "3 erros seguidos" in textos(at.warning)
    assert set(at.session_state["lote"]["df"]["decisao"]) == {"ERRO"}


def test_lote_production_com_api_fora_do_ar_no_meio(api_simulada):
    def cai(tx, modo, vistas=[]):
        vistas.append(1)
        if len(vistas) == 2:
            raise APIForaDoAr(cliente_api.mensagem_fora_do_ar())
        return resposta_individual()

    api_simulada["individual"] = cai
    at = rodar_production(abrir(), linhas=5)
    assert len(at.session_state["lote"]["df"]) == 1
    assert "fora do ar" in textos(at.warning)


def test_lote_production_sem_nenhuma_resposta_mostra_erro(api_simulada):
    def cai(tx, modo):
        raise APIForaDoAr(cliente_api.mensagem_fora_do_ar())

    api_simulada["individual"] = cai
    at = rodar_production(abrir(), linhas=3)
    assert "lote" not in at.session_state
    assert "fora do ar" in textos(at.error)


def test_aprenda_simulando_nunca_usa_production(api_simulada):
    at = abrir()
    botao(at, "Analisar (modo grátis").click().run()
    botao(at, "Simular nova transação").click().run()
    botao(at, "Analisar (modo grátis").click().run()
    assert api_simulada["modos"] == ["fast", "fast"]
    assert "consome" not in textos(at.warning)
