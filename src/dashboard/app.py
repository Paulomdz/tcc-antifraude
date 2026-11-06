"""Dashboard Streamlit — Sistema Multiagente de IA para detecção de fraudes.

Toda análise passa pela API (src/api/main.py); o dashboard não carrega modelos.
"""

import html
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from src.dashboard import cliente_api, valores_iniciais  # noqa: E402
from src.dashboard.cliente_api import APIForaDoAr, ErroAPI  # noqa: E402
from src.dashboard.lote import (  # noqa: E402
    calcular_metricas,
    montar_resultados,
    montar_resultados_individuais,
    transacao_da_linha,
)
from src.dashboard.rotulos import (  # noqa: E402
    COLUNAS,
    DECISOES,
    ICONES,
    descricao_tipo,
    direcao_feature,
    moeda_br,
    nome_tipo,
    numero_br,
    rotulo_feature,
    traduzir_colunas,
    valor_feature,
)
from src.ferramentas.leitor_arquivo_lote import (  # noqa: E402
    SEED_AMOSTRAGEM,
    amostrar_lote,
    ler_arquivo_lote,
)
from src.preprocessamento.carregar_dataset import TRANSACTION_TYPES  # noqa: E402

# Constantes

_DECISION_BG = {
    "APROVADO": "rgba(22,163,74,0.14)",
    "REVISÃO": "rgba(217,119,6,0.16)",
    "BLOQUEADO": "rgba(220,38,38,0.14)",
    "ERRO": "rgba(128,128,128,0.18)",
}
_DECISION_FG = {
    "APROVADO": "#16a34a",
    "REVISÃO": "#d97706",
    "BLOQUEADO": "#dc2626",
    "ERRO": "#64748b",
}

_BRAND = "#2563eb"
_THRESHOLD_REVIEW = 0.5
_THRESHOLD_BLOCK = 0.8
_TX_TYPES = TRANSACTION_TYPES
_MAX_FINANCIAL_VALUE = 9_999_999.00
_LIMITE_LOTE = 500  # modo fast
_LIMITE_LOTE_PRODUCTION = 10  # modo production: cada transação chama o Gemini
_LIMITE_LOTE_DEEP = 5  # modo deep: quatro agentes Gemini por transação (mais cota)
_LIMITES_LOTE = {"fast": _LIMITE_LOTE, "production": _LIMITE_LOTE_PRODUCTION, "deep": _LIMITE_LOTE_DEEP}
_QUANTIDADE_PADRAO = 300

_RISK_RULES_TEXT = (
    "**Como o sistema decide.** Cada transação recebe três notas de 0 a 1 (comportamento, "
    "horário e identidade), combinadas em um score final: abaixo de 0,50 é **aprovado**; "
    "de 0,50 a 0,79, **revisão humana**; de 0,80 em diante, **bloqueado**. "
    "**Regra noturna (Banco Central):** das 20h às 6h, Pix (entre pessoas, QR Code ou "
    "Automático) e TED entre pessoas físicas têm limite de R$ 1.000 somados na noite, por "
    "conta pagadora. A transação que passa desse total vai para **revisão humana**; a regra "
    "não bloqueia e não muda o score."
)
_AVISOS_GERAIS = (
    "Os dados são sintéticos: os resultados valem para esta base e não mostram o desempenho "
    "em uma instituição real. Os pesos do juiz (0,35 / 0,30 / 0,35) e os limites 0,50 / 0,80 "
    "são parâmetros definidos, sem calibração; calibrá-los é uma melhoria futura."
)

_CSS = """
<style>
/* Neutro ao tema: usa só transparências, funciona no claro e no escuro */
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Cabeçalho da marca */
.cs-header {
    display: flex; flex-wrap: wrap; align-items: center;
    justify-content: space-between; gap: 14px;
    border: 1px solid rgba(37,99,235,0.30);
    background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(37,99,235,0.03));
    border-radius: 14px; padding: 18px 24px; margin-bottom: 14px;
}
.cs-header h1 { margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.3px; }
.cs-header p { margin: 2px 0 0 0; opacity: 0.7; font-size: 0.9rem; }

/* Chips de status */
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
    padding: 4px 12px; border-radius: 999px; border: 1px solid;
    font-size: 0.78rem; font-weight: 600; cursor: default;
}
.chip-ok   { color: #16a34a; background: rgba(22,163,74,0.12);  border-color: rgba(22,163,74,0.38); }
.chip-warn { color: #d97706; background: rgba(217,119,6,0.13);  border-color: rgba(217,119,6,0.40); }
.chip-bad  { color: #dc2626; background: rgba(220,38,38,0.12);  border-color: rgba(220,38,38,0.38); }
.chip-info { color: #2563eb; background: rgba(37,99,235,0.12);  border-color: rgba(37,99,235,0.38); }

/* Badge de decisão */
.decision-badge {
    display: inline-block; padding: 10px 22px; border-radius: 10px;
    font-size: 1.3rem; font-weight: 700; margin: 4px 0 12px 0;
}

/* KPIs */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 18px 0; }
.kpi {
    border: 1px solid rgba(128,128,128,0.25); border-top: 3px solid #2563eb;
    background: rgba(128,128,128,0.07); border-radius: 12px; padding: 14px 16px;
}
.kpi .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.65; }
.kpi .value { font-size: 1.7rem; font-weight: 700; line-height: 1.25; }
.kpi .sub { font-size: 0.78rem; opacity: 0.65; }

/* Cartões de agente e de feature */
.agent-card, .feat-card {
    border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 8px; height: 100%;
}
.agent-card .name, .feat-card .name { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; }
.agent-card .score { font-size: 1.8rem; font-weight: 700; line-height: 1.2; }
.agent-card .expl, .feat-card .meta { font-size: 0.82rem; opacity: 0.75; margin-top: 8px; }
.feat-card .rank { float: right; font-weight: 700; opacity: 0.45; }
.feat-card .z { font-size: 1.5rem; font-weight: 700; line-height: 1.25; }
.bar { height: 8px; border-radius: 999px; background: rgba(128,128,128,0.22); margin-top: 8px; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 999px; }

/* Botões */
div[data-testid="stButton"] button { border-radius: 8px; font-weight: 600; }

/* Rótulo de seção */
.section-label {
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; opacity: 0.6; margin: 20px 0 6px 0;
}
/* Botão de alternância do modo (vermelho): metade ativa escura, a outra clara */
.st-key-lote_modo button[kind="segmented_control"],
.st-key-lote_modo [data-testid="stBaseButton-segmented_control"] {
    background: #ffffff !important; color: #b91c1c !important;
    border: 1px solid #b91c1c !important; font-weight: 600; padding: 0.35rem 1.4rem;
}
.st-key-lote_modo button[kind="segmented_controlActive"],
.st-key-lote_modo [data-testid="stBaseButton-segmented_controlActive"] {
    background: #991b1b !important; color: #ffffff !important;
    border: 1px solid #991b1b !important; font-weight: 700; padding: 0.35rem 1.4rem;
}
.st-key-lote_modo button[kind="segmented_control"]:hover { background: #fee2e2 !important; }

/* Justificativa do juiz e avisos */
.just-box {
    border-left: 4px solid #2563eb; background: rgba(37,99,235,0.08);
    border-radius: 8px; padding: 14px 18px; font-size: 1.02rem; line-height: 1.55;
    white-space: pre-line;
}
.aviso { font-size: 0.82rem; opacity: 0.75; margin-top: 10px; }
</style>
"""


# Funções auxiliares puras (testáveis sem Streamlit)


def decision_icon(decision: str) -> str:
    return ICONES.get(decision, "⚪")


def _risk_color(score: float) -> str:
    """Cor do score segundo os limiares de revisão e bloqueio."""
    if score >= _THRESHOLD_BLOCK:
        return _DECISION_FG["BLOQUEADO"]
    if score >= _THRESHOLD_REVIEW:
        return _DECISION_FG["REVISÃO"]
    return _DECISION_FG["APROVADO"]


def _novo_saldo(saldo: float, valor: float) -> float:
    """Saldo da origem depois da transação (a API não aceita saldo negativo)."""
    return round(max(float(saldo) - float(valor), 0.0), 2)


def random_transaction() -> dict:
    """Transação aleatória para a aba de simulação (hora em 0–23)."""
    tipo = random.choice(_TX_TYPES)
    contas = valores_iniciais.contas_do_dataset()
    valor = round(min(10 ** random.uniform(1.0, 4.4), _MAX_FINANCIAL_VALUE), 2)
    saldo = round(min(valor * random.uniform(1.0, 6.0), _MAX_FINANCIAL_VALUE), 2)
    return {
        "step": random.randint(0, 23),
        "type": tipo,
        "amount": valor,
        "nameOrig": random.choice(contas["origem"]),
        "oldbalanceOrg": saldo,
        "nameDest": random.choice(contas["destino"]),
    }


# Componentes de renderização


def _chip(text: str, kind: str, tip: str = "") -> str:
    return f'<span class="chip chip-{kind}" title="{html.escape(tip)}">{text}</span>'


def _header_html(config: dict | None) -> str:
    if config is not None:
        chip_api = _chip("● API online", "ok", f"API em {cliente_api.url_base()}")
        if config.get("llm_disponivel"):
            modelo = config.get("modelo_llm") or "modelo padrão"
            chip_llm = _chip("● Gemini ativo", "ok", f"Modelo: {modelo}")
        else:
            chip_llm = _chip(
                "● Gemini sem chave", "warn",
                "Defina GOOGLE_API_KEY no .env e reinicie a API. O lote continua disponível.",
            )
    else:
        chip_api = _chip("● API offline", "bad", cliente_api.mensagem_fora_do_ar())
        chip_llm = _chip("● Gemini indisponível", "warn", "A API precisa estar no ar.")
    chip_rules = _chip("Regras de risco ativas", "info", "Veja 'Regras de risco e limites'.")
    return (
        '<div class="cs-header"><div>'
        "<h1>🛡️ Sistema Multiagente de IA para Detecção de Fraudes</h1>"
        "<p>Análise de risco de transações Pix e meios de pagamento, em tempo real.</p>"
        f'</div><div class="chips">{chip_api}{chip_llm}{chip_rules}</div></div>'
    )


def _decision_badge(decisao: str) -> None:
    bg = _DECISION_BG.get(decisao, "rgba(128,128,128,0.15)")
    fg = _DECISION_FG.get(decisao, "inherit")
    st.markdown(
        f'<div class="decision-badge" style="background:{bg};color:{fg};">'
        f"{decision_icon(decisao)}&ensp;{decisao}</div>",
        unsafe_allow_html=True,
    )


def _score_gauge(score: float):
    """Gauge de 0 a 1 com faixas de aprovação, revisão e bloqueio."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": "Score final de risco"},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, _THRESHOLD_REVIEW, _THRESHOLD_BLOCK, 1]},
                "bar": {"color": _risk_color(score), "thickness": 0.3},
                "steps": [
                    {"range": [0, _THRESHOLD_REVIEW], "color": "rgba(22,163,74,0.22)"},
                    {"range": [_THRESHOLD_REVIEW, _THRESHOLD_BLOCK], "color": "rgba(217,119,6,0.25)"},
                    {"range": [_THRESHOLD_BLOCK, 1], "color": "rgba(220,38,38,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def _render_agentes(agentes: dict) -> None:
    cols = st.columns(3)
    for col, chave, rotulo in zip(
        cols,
        ("comportamental", "temporal", "identidade"),
        ("Comportamental", "Temporal", "Identidade"),
    ):
        agente = agentes.get(chave, {})
        score = float(agente.get("score", 0.0))
        explicacao = html.escape(str(agente.get("explicacao") or "Sem explicação."))
        cor = _risk_color(score)
        pct = min(max(score, 0.0), 1.0) * 100
        with col:
            st.markdown(
                f'<div class="agent-card"><div class="name">{rotulo}</div>'
                f'<div class="score" style="color:{cor}">{score:.3f}</div>'
                f'<div class="bar"><span style="width:{pct:.0f}%;background:{cor}"></span></div>'
                f'<div class="expl">{explicacao}</div></div>',
                unsafe_allow_html=True,
            )


def _render_top3(top3: list[dict]) -> None:
    st.markdown(
        '<p class="section-label">O que mais fugiu do padrão (top-3)</p>',
        unsafe_allow_html=True,
    )
    if not top3:
        st.caption("Top-3 indisponível para esta transação.")
        return
    cols = st.columns(len(top3))
    for posicao, (col, item) in enumerate(zip(cols, top3), start=1):
        z = float(item.get("z", 0.0))
        cor = _risk_color(min(abs(z) / 5.0, 1.0))
        pct = min(abs(z) / 5.0, 1.0) * 100
        nome = item.get("nome", "")
        direcao = direcao_feature(item.get("direcao", ""), bool(item.get("desvio_zero")))
        with col:
            st.markdown(
                f'<div class="feat-card" title="z = {z:+.2f}"><span class="rank">#{posicao}</span>'
                f'<div class="name">{html.escape(rotulo_feature(nome))}</div>'
                f'<div class="z">{html.escape(valor_feature(nome, item.get("valor", 0.0)))}</div>'
                f'<div class="bar"><span style="width:{pct:.0f}%;background:{cor}"></span></div>'
                f'<div class="meta">{direcao}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<p class="aviso">O top-3 mostra o que foge do padrão dos dados de treino. '
        "Ele <b>não</b> é o motivo da decisão do modelo.</p>",
        unsafe_allow_html=True,
    )


def _render_resultado_individual(resposta: dict) -> None:
    """Ordem: decisão, justificativa, top-3, scores e modo usado."""
    decisao = resposta.get("decisao", "ERRO")
    score_final = float(resposta.get("score_final", 0.0))

    st.markdown('<p class="section-label">Decisão</p>', unsafe_allow_html=True)
    _decision_badge(decisao)

    juiz = "do juiz Gemini" if resposta.get("modo") in ("production", "deep") else "da decisão"
    st.markdown(
        f'<p class="section-label">Justificativa {juiz}</p>', unsafe_allow_html=True
    )
    justificativa = str(resposta.get("justificativa") or "").strip()
    if justificativa:
        st.markdown(f'<div class="just-box">{html.escape(justificativa)}</div>', unsafe_allow_html=True)
    else:
        st.warning("A análise não devolveu justificativa para esta decisão.")

    _render_top3(resposta.get("explicabilidade", {}).get("top3", []))

    st.markdown('<p class="section-label">Scores</p>', unsafe_allow_html=True)
    col_gauge, col_info = st.columns([1, 1.6])
    with col_gauge:
        try:
            st.plotly_chart(_score_gauge(score_final), width="stretch")
        except ImportError:
            st.metric("Score final", f"{score_final:.3f}")
    with col_info:
        st.metric("Score final", f"{score_final:.3f}")
        st.caption(
            f"Revisão a partir de {numero_br(_THRESHOLD_REVIEW, 2)} · "
            f"bloqueio a partir de {numero_br(_THRESHOLD_BLOCK, 2)}"
        )
    _render_agentes(resposta.get("agentes", {}))
    st.caption(f"Modo usado: **{resposta.get('modo', 'fast')}**")


# Estatísticas do lote


def _kpi_card(label: str, value: str, sub: str = "", color: str = _BRAND) -> str:
    return (
        f'<div class="kpi" style="border-top-color:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}&nbsp;</div></div>'
    )


def render_stats(df: pd.DataFrame) -> None:
    """KPIs e gráficos do lote (colunas de montar_resultados)."""
    if df is None or df.empty:
        st.info("Execute uma **Análise em Lote** para visualizar as estatísticas aqui.", icon="ℹ️")
        return
    try:
        import plotly.express as px
    except ImportError:
        st.warning("Instale `plotly` para visualizar os gráficos: `pip install plotly`")
        return

    total = len(df)
    n = {d: int((df["decisao"] == d).sum()) for d in DECISOES}
    pct = {d: (100 * v / total if total else 0.0) for d, v in n.items()}
    validos = df[df["decisao"] != "ERRO"]
    retido = float(df.loc[df["decisao"].isin(["REVISÃO", "BLOQUEADO"]), "valor"].sum())
    medio = float(validos["score_final"].mean()) if not validos.empty else 0.0

    cards = [
        _kpi_card("Total analisado", f"{total:,}".replace(",", "."), "transações"),
        _kpi_card("Aprovados", str(n["APROVADO"]), f"{pct['APROVADO']:.1f}% do lote", _DECISION_FG["APROVADO"]),
        _kpi_card("Em revisão", str(n["REVISÃO"]), f"{pct['REVISÃO']:.1f}% do lote", _DECISION_FG["REVISÃO"]),
        _kpi_card("Bloqueados", str(n["BLOQUEADO"]), f"{pct['BLOQUEADO']:.1f}% do lote", _DECISION_FG["BLOQUEADO"]),
        _kpi_card("Erros", str(n["ERRO"]), f"{pct['ERRO']:.1f}% do lote", _DECISION_FG["ERRO"]),
        _kpi_card("Score médio", f"{medio:.3f}", "sem contar erros", _risk_color(medio)),
        _kpi_card("Valor retido", moeda_br(retido), "revisão + bloqueio"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    cores = dict(_DECISION_FG)
    ordem = {"decisao": DECISOES}
    col_pie, col_tipo = st.columns(2)
    with col_pie:
        contagem = df["decisao"].value_counts().reset_index()
        contagem.columns = ["decisao", "quantidade"]
        fig = px.pie(
            contagem, values="quantidade", names="decisao", title="Distribuição de decisões",
            color="decisao", color_discrete_map=cores, hole=0.55,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with col_tipo:
        por_tipo = df.assign(tipo_nome=df["tipo"].map(nome_tipo)).groupby(
            ["tipo_nome", "decisao"]
        ).size().reset_index(name="quantidade")
        fig = px.bar(
            por_tipo, x="quantidade", y="tipo_nome", color="decisao", orientation="h",
            title="Decisões por tipo de transação", color_discrete_map=cores, category_orders=ordem,
            labels={"quantidade": "Quantidade", "tipo_nome": "Tipo"},
        )
        fig.update_layout(barmode="stack", showlegend=False, margin=dict(t=50, b=10))
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")

    if not validos.empty:
        col_hist, col_box = st.columns(2)
        with col_hist:
            fig = px.histogram(
                validos, x="score_final", nbins=20, color="decisao", title="Distribuição dos scores finais",
                color_discrete_map=cores, category_orders=ordem,
                labels={"score_final": "Score de risco", "count": "Frequência"},
            )
            fig.update_xaxes(range=[0, 1])
            fig.add_vline(x=_THRESHOLD_REVIEW, line_dash="dash", line_color=cores["REVISÃO"],
                          annotation_text=f"Revisão ({_THRESHOLD_REVIEW})")
            fig.add_vline(x=_THRESHOLD_BLOCK, line_dash="dash", line_color=cores["BLOQUEADO"],
                          annotation_text=f"Bloqueio ({_THRESHOLD_BLOCK})")
            fig.update_layout(showlegend=False, bargap=0.05, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with col_box:
            colunas = {"comportamental": "Comportamental", "temporal": "Temporal",
                       "identidade": "Identidade", "score_final": "Score final"}
            longo = validos[list(colunas)].rename(columns=colunas).melt(var_name="Agente", value_name="Score")
            fig = px.box(
                longo, x="Agente", y="Score", color="Agente", title="Scores por agente",
                color_discrete_sequence=["#2563eb", "#7c3aed", "#0891b2", "#64748b"],
            )
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")


def _tabela_lote(sub: pd.DataFrame, com_fraude: bool) -> pd.DataFrame:
    """Tabela amigável de um grupo de decisões."""
    saida = pd.DataFrame(
        {
            "Tipo": sub["tipo"].map(nome_tipo),
            "Valor": sub["valor"],
            "Conta de origem": sub["origem"],
            "Conta de destino": sub["destino"],
            "Score final": sub["score_final"].round(3),
            "Comportamental": sub["comportamental"].round(3),
            "Temporal": sub["temporal"].round(3),
            "Identidade": sub["identidade"].round(3),
        }
    )
    if com_fraude and "isFraud" in sub:
        saida["Fraude real"] = sub["isFraud"].map({1: "sim", 0: "não", True: "sim", False: "não"})
    return saida


def _pct(valor: float | None) -> str:
    return "indefinida" if valor is None else f"{valor * 100:.1f}%".replace(".", ",")


def _render_metricas(df: pd.DataFrame) -> None:
    metricas = calcular_metricas(df)
    if metricas is None:
        return
    st.markdown('<p class="section-label">Acertos contra a coluna de fraude real</p>', unsafe_allow_html=True)
    linhas = []
    for nome, chave in (("Estrita (só BLOQUEADO conta como fraude)", "estrita"),
                        ("Ampla (BLOQUEADO ou REVISÃO)", "ampla")):
        m = metricas[chave]
        linhas.append({"Leitura": nome, "Precisão": _pct(m["precisao"]),
                       "Recall": _pct(m["recall"]), "F1": _pct(m["f1"])})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.caption(
        f"{metricas['avaliadas']} transações avaliadas, {metricas['fraudes']} fraudes reais · "
        f"taxa de REVISÃO: {_pct(metricas['taxa_revisao'])}"
        + (f" · {metricas['excluidas']} fora da conta (erro ou sem rótulo)" if metricas["excluidas"] else "")
        + ". Em base desbalanceada a acurácia engana; use precisão, recall e F1."
    )


def _render_lote(estado: dict) -> None:
    resposta, df = estado["resposta"], estado["df"]
    modo = estado.get("modo", "fast")
    usa_gemini = modo != "fast"
    modo_texto = {
        "fast": "**fast** (sem Gemini)",
        "production": "**production** (usou o Gemini)",
        "deep": "**deep** (usou quatro agentes Gemini)",
    }[modo]
    st.success(
        f"Lote analisado: **{resposta['quantidade_usada']}** de {resposta['total_arquivo']} "
        f"transações do arquivo `{estado['arquivo']}` · modo {modo_texto}."
    )
    if estado.get("aviso"):
        st.warning(estado["aviso"])
    st.caption(
        f"A amostra é sorteada com seed {resposta['seed']}: o mesmo arquivo gera sempre as mesmas "
        "transações. "
        + (
            "Escolha uma transação abaixo para ver a justificativa do juiz e o top-3."
            if usa_gemini
            else "O modo fast não calcula top-3 nem usa a justificativa do juiz."
        )
    )
    render_stats(df)
    _render_metricas(df)

    st.markdown('<p class="section-label">Transações por decisão</p>', unsafe_allow_html=True)
    grupos = [d for d in DECISOES if (df["decisao"] == d).any()]
    abas = st.tabs([f"{ICONES[d]} {d.capitalize()} ({int((df['decisao'] == d).sum())})" for d in grupos])
    for aba, decisao in zip(abas, grupos):
        with aba:
            sub = df[df["decisao"] == decisao]
            if decisao == "ERRO":
                tabela = pd.DataFrame({"Tipo": sub["tipo"].map(nome_tipo), "Valor": sub["valor"],
                                       "Erro": sub["erro"]})
            else:
                ordenado = sub.sort_values("score_final", ascending=False)
                tabela = _tabela_lote(ordenado, True)
                if "justificativa" in ordenado:
                    tabela["Justificativa"] = ordenado["justificativa"]
            st.dataframe(tabela, hide_index=True, width="stretch",
                         column_config={"Valor": st.column_config.NumberColumn(format="R$ %.2f")})

    detalhes = estado.get("detalhes") or {}
    if detalhes:
        st.markdown('<p class="section-label">Detalhe de uma transação (Gemini)</p>', unsafe_allow_html=True)
        escolhida = st.selectbox(
            "Escolha a transação", list(detalhes), key="lote_detalhe",
            format_func=lambda i: (
                f"#{i + 1} · {nome_tipo(df.loc[i, 'tipo'])} · {moeda_br(df.loc[i, 'valor'])} "
                f"· {df.loc[i, 'decisao']}"
            ),
        )
        _render_resultado_individual(detalhes[escolhida])


# Abas


@st.cache_data(show_spinner=False)
def _ler_arquivo(conteudo: bytes, nome: str) -> pd.DataFrame:
    return ler_arquivo_lote(conteudo, nome)


def _rodar_lote_gemini(amostra: pd.DataFrame, modo: str) -> tuple[list[dict], str | None]:
    """Chama /analise_transacao (production ou deep) uma transação por vez, com progresso.

    Para cedo se a API cair, se faltar a chave (503) ou após 3 erros seguidos,
    para não gastar cota em chamadas que vão falhar de novo.
    """
    respostas: list[dict] = []
    aviso: str | None = None
    seguidos = 0
    total = len(amostra)
    barra = st.progress(0.0, text="Iniciando...")
    for posicao, (_, linha) in enumerate(amostra.iterrows()):
        barra.progress(posicao / total, text=f"Analisando {posicao + 1} de {total} com o Gemini…")
        try:
            respostas.append(
                cliente_api.analisar_transacao(transacao_da_linha(linha), modo=modo)
            )
            seguidos = 0
        except APIForaDoAr as exc:
            aviso = f"Lote interrompido na transação {posicao + 1}: {exc}"
            break
        except ErroAPI as exc:
            respostas.append({"erro": str(exc)})
            seguidos += 1
            if exc.status == 503 or seguidos >= 3:
                aviso = (
                    f"Lote interrompido na transação {posicao + 1} "
                    + ("(Gemini não configurado)." if exc.status == 503 else "(3 erros seguidos).")
                )
                break
    barra.empty()
    return respostas, aviso


_MODOS_LOTE = {"fast": "⚡ Fast", "production": "🤖 Production", "deep": "🧠 Deep"}
_MODOS_DESCRICAO = {
    "fast": f"Grátis, sem Gemini, até {_LIMITE_LOTE} transações.",
    "production": f"Usa o Gemini e consome cota, até {_LIMITE_LOTE_PRODUCTION} transações.",
    "deep": (
        f"Usa quatro agentes Gemini por transação e consome ainda mais cota, "
        f"até {_LIMITE_LOTE_DEEP} transações."
    ),
}
_AVISOS_GEMINI = {
    "production": "O modo production chama o Gemini **uma vez por transação**",
    "deep": "O modo deep chama o Gemini **várias vezes por transação** (três especialistas e o juiz final)",
}


def _aba_lote(api_ok: bool, gemini_ok: bool) -> None:
    st.header("Análise em lote")
    st.markdown(
        "Envie um arquivo **CSV** (separado por vírgula) ou **TXT** (separado por ponto e vírgula). "
        "Colunas obrigatórias: "
        + ", ".join(f"`{c}` ({COLUNAS[c]})" for c in COLUNAS if c != "isFraud")
        + ". Opcional: `isFraud` (fraude real, 0 ou 1), que habilita precisão, recall e F1."
    )
    modo = st.segmented_control(
        "Modo de análise", list(_MODOS_LOTE), default="fast", required=True,
        format_func=_MODOS_LOTE.get, key="lote_modo",
    ) or "fast"
    st.caption(_MODOS_DESCRICAO[modo])
    usa_gemini = modo != "fast"
    limite = _LIMITES_LOTE[modo]
    if usa_gemini:
        st.warning(
            f"{_AVISOS_GEMINI[modo]} e **consome a cota da chave**. "
            f"Por isso o lote fica limitado a {limite} transações e a demora é maior. "
            "Em troca, cada transação ganha justificativa do juiz e top-3."
        )
        if api_ok and not gemini_ok:
            st.error("O Gemini não está configurado na API: defina `GOOGLE_API_KEY` no `.env` e reinicie a API.")
    arquivo = st.file_uploader("Selecione o arquivo de transações", type=["csv", "txt"])
    if arquivo is not None:
        conteudo = arquivo.getvalue()
        try:
            df = _ler_arquivo(conteudo, arquivo.name)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.info(f"Arquivo carregado: **{len(df):,}** transações".replace(",", "."), icon="📄")
            st.dataframe(traduzir_colunas(df.head(10)), hide_index=True, width="stretch")
            quantidade = int(st.number_input(
                f"Quantas transações analisar? (1 a {limite})", min_value=1, max_value=limite,
                value=limite if usa_gemini else min(_QUANTIDADE_PADRAO, limite), step=1,
                key=f"lote_qtd_{modo}",
            ))
            if len(df) <= quantidade:
                st.caption("O arquivo tem menos linhas que o pedido: todas serão analisadas.")
            else:
                st.caption(f"A amostra será sorteada com seed {SEED_AMOSTRAGEM} (reprodutível).")
            rotulo = {
                "fast": "🚀 Análise em lote (sem Gemini, sem custo)",
                "production": "🤖 Análise em lote (usa Gemini, consome cota)",
                "deep": "🧠 Análise em lote Deep (vários agentes Gemini, consome mais cota)",
            }[modo]
            if st.button(rotulo, width="stretch", type="primary",
                         disabled=not api_ok or (usa_gemini and not gemini_ok)):
                if usa_gemini:
                    amostra = amostrar_lote(df, quantidade, SEED_AMOSTRAGEM)
                    respostas, aviso = _rodar_lote_gemini(amostra, modo)
                    if not respostas:
                        st.error(aviso or "Nenhuma transação foi analisada.")
                    else:
                        usada = amostra.head(len(respostas)).reset_index(drop=True)
                        st.session_state["lote"] = {
                            "resposta": {"quantidade_usada": len(respostas), "total_arquivo": len(df),
                                         "seed": SEED_AMOSTRAGEM},
                            "df": montar_resultados_individuais(usada, respostas),
                            "arquivo": arquivo.name, "modo": modo, "aviso": aviso,
                            "detalhes": {i: r for i, r in enumerate(respostas) if "decisao" in r},
                        }
                else:
                    with st.spinner("Analisando o lote na API..."):
                        try:
                            resposta = cliente_api.analisar_lote_arquivo(conteudo, arquivo.name, quantidade)
                            amostra = amostrar_lote(df, quantidade, resposta.get("seed", SEED_AMOSTRAGEM))
                            resultado = montar_resultados(amostra, resposta)
                        except (ErroAPI, ValueError) as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["lote"] = {
                                "resposta": resposta, "df": resultado, "arquivo": arquivo.name,
                                "modo": "fast",
                            }
    estado = st.session_state.get("lote")
    st.divider()
    if estado:
        _render_lote(estado)
    else:
        render_stats(None)


def _aplicar_simulacao() -> None:
    sim = random_transaction()
    tipo = sim["type"]
    st.session_state.update(
        {
            "f_tipo": tipo,
            f"f_amount_{tipo}": sim["amount"],
            f"f_old_{tipo}": sim["oldbalanceOrg"],
            "f_hora": sim["step"],
            "f_orig": sim["nameOrig"],
            "f_dest": sim["nameDest"],
        }
    )
    st.session_state.pop("individual", None)


def _aba_individual(api_ok: bool) -> None:
    st.header("Aprenda simulando")
    st.markdown(
        "Simule uma transação, ajuste os campos e veja **como o sistema decide**: a decisão, a "
        "justificativa, o que mais fugiu do padrão e a nota de cada agente. "
        "Esta aba usa o **modo gratuito (fast)**: roda os três modelos e a regra de decisão, "
        "**sem Gemini e sem custo**."
    )
    if not api_ok:
        st.warning("A API está fora do ar: não é possível analisar agora.")

    st.button("🎲 Simular nova transação", on_click=_aplicar_simulacao)

    tipo = st.selectbox("Tipo de transação", _TX_TYPES, key="f_tipo", format_func=nome_tipo)
    st.caption(descricao_tipo(tipo))
    valor_ini, saldo_ini, dos_dados = valores_iniciais.valores_do_tipo(tipo, _MAX_FINANCIAL_VALUE)
    st.session_state.setdefault(f"f_amount_{tipo}", valor_ini)
    st.session_state.setdefault(f"f_old_{tipo}", saldo_ini)
    if dos_dados:
        st.caption("Valores iniciais: mediana das transações legítimas deste tipo na base (step < 600).")
    contas = valores_iniciais.contas_do_dataset()

    with st.form("form_transacao"):
        col_a, col_b = st.columns(2)
        with col_a:
            valor = st.number_input(
                "Valor da transação (R$)", min_value=0.0, max_value=_MAX_FINANCIAL_VALUE,
                step=0.01, format="%.2f", key=f"f_amount_{tipo}",
            )
            saldo = st.number_input(
                "Saldo anterior da origem (R$) — obrigatório", min_value=0.0,
                max_value=_MAX_FINANCIAL_VALUE, step=0.01, format="%.2f", key=f"f_old_{tipo}",
                help="Quanto havia na conta de origem antes da transação. A API recusa a análise sem ele.",
            )
        with col_b:
            hora = st.number_input(
                "Hora da transação (0 a 23)", min_value=0, max_value=23, value=12, step=1, key="f_hora",
                help="Hora do dia em que a transação aconteceu (0 = meia-noite, 23 = 23h). "
                     "Das 20h às 6h vale a regra noturna: Pix e TED entre pessoas físicas acima de "
                     "R$ 1.000 no total da noite vão para revisão humana.",
            )
            origem = st.selectbox(
                "Conta de origem (15 contas da base)", contas["origem"], key="f_orig",
                help="Contas reais do dataset. As que começam com 'B' são de bancos "
                     "(salário, empréstimo); as demais são de clientes e comércios.",
            )
            destino = st.selectbox(
                "Conta de destino (15 contas da base)", contas["destino"], key="f_dest",
                help="Contas reais do dataset que aparecem como destino.",
            )
        enviado = st.form_submit_button(
            "Analisar (modo grátis, sem Gemini)", width="stretch", type="primary",
            disabled=not api_ok,
        )

    if enviado:
        tx = {
            "step": int(hora), "type": tipo, "amount": float(valor), "nameOrig": origem,
            "oldbalanceOrg": float(saldo), "newbalanceOrig": _novo_saldo(saldo, valor),
            "nameDest": destino,
        }
        with st.spinner("Analisando com os três modelos..."):
            try:
                st.session_state["individual"] = {"tx": tx, "resposta": cliente_api.analisar_transacao(tx, modo="fast")}
            except ErroAPI as exc:
                st.session_state.pop("individual", None)
                st.error(str(exc))
    atual = st.session_state.get("individual")
    if atual:
        st.divider()
        st.caption(
            f"Saldo depois: {moeda_br(atual['tx']['oldbalanceOrg'])} − "
            f"{moeda_br(atual['tx']['amount'])} = {moeda_br(atual['tx']['newbalanceOrig'])}"
        )
        _render_resultado_individual(atual["resposta"])


_COLUNAS_HISTORICO = {
    "timestamp": "Data e hora",
    "type": "Tipo",
    "amount": "Valor",
    "decisao": "Decisão",
    "score_final": "Score final",
    "score_behavior": "Comportamental",
    "score_temporal": "Temporal",
    "score_identity": "Identidade",
    "duracao_ms": "Duração (ms)",
    "llm_usado": "Usou Gemini",
    "erro": "Erro",
}


@st.cache_data(ttl=5, show_spinner=False)
def _carregar_historico(limite: int, decisao: str | None) -> list[dict]:
    return cliente_api.obter_historico(limite, decisao)


@st.cache_data(ttl=5, show_spinner=False)
def _carregar_estatisticas() -> dict:
    return cliente_api.obter_estatisticas()


def formatar_historico(registros: list[dict]) -> pd.DataFrame:
    """Histórico da API em tabela amigável (horário de Brasília)."""
    df = pd.DataFrame(registros)
    if df.empty:
        return df
    data = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = data.dt.tz_convert("America/Sao_Paulo").dt.strftime("%d/%m/%Y %H:%M:%S")
    df["llm_usado"] = df["llm_usado"].map({"SIM": "sim", "NAO": "não"}).fillna("")
    df["erro"] = df["erro"].fillna("")
    return df[[c for c in _COLUNAS_HISTORICO if c in df]].rename(columns=_COLUNAS_HISTORICO)


def _aba_historico(api_ok: bool) -> None:
    st.header("Histórico de análises")
    if not api_ok:
        st.info("O histórico fica na API. Suba a API para consultá-lo.")
        return
    try:
        estat = _carregar_estatisticas()
    except ErroAPI as exc:
        st.error(str(exc))
        return

    st.markdown('<p class="section-label">Últimas 24 horas</p>', unsafe_allow_html=True)
    contagem = estat.get("contagem_por_decisao", {})
    cards = [_kpi_card("Análises", str(estat.get("total_transacoes", 0)), "nas últimas 24 h"),
             _kpi_card("Score médio", f"{estat.get('score_medio', 0.0):.3f}", ""),
             _kpi_card("Latência média", f"{numero_br(estat.get('latencia_media_ms', 0.0), 0)} ms", "por análise")]
    for decisao in DECISOES:
        cards.append(_kpi_card(decisao.capitalize(), str(contagem.get(decisao, 0)), "",
                               _DECISION_FG[decisao]))
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    col_d, col_t, col_l = st.columns(3)
    decisao = col_d.selectbox("Decisão", ["Todas", *DECISOES], key="h_decisao")
    limite = col_l.number_input("Quantidade de registros", 1, 1000, 100, step=50, key="h_limite")
    try:
        registros = _carregar_historico(int(limite), None if decisao == "Todas" else decisao)
    except ErroAPI as exc:
        st.error(str(exc))
        return
    tipos = sorted({r["type"] for r in registros if r.get("type")})
    tipo = col_t.selectbox("Tipo", ["Todos", *tipos], key="h_tipo",
                           format_func=lambda c: c if c == "Todos" else nome_tipo(c))
    if tipo != "Todos":
        registros = [r for r in registros if r.get("type") == tipo]
    if not registros:
        st.info("Nenhuma análise salva com esses filtros. Rode um lote ou uma análise individual.")
        return
    tabela = formatar_historico(registros)
    tabela["Tipo"] = tabela["Tipo"].map(nome_tipo)
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={"Valor": st.column_config.NumberColumn(format="R$ %.2f")})
    st.caption(f"{len(tabela)} registros · horário de Brasília · o filtro de tipo vale para os registros já carregados.")


# Interface principal


def _consultar_config() -> tuple[dict | None, str | None]:
    try:
        return cliente_api.obter_config(), None
    except ErroAPI as exc:
        return None, str(exc)


def main() -> None:
    """Ponto de entrada do dashboard."""
    st.set_page_config(
        page_title="Detecção de Fraudes · Multiagente IA",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    config, erro_api = _consultar_config()
    api_ok = config is not None
    gemini_ok = bool(config and config.get("llm_disponivel"))

    st.markdown(_header_html(config), unsafe_allow_html=True)
    if erro_api:
        st.error(erro_api)
    with st.expander("ℹ️ Regras de risco e limites"):
        st.markdown(_RISK_RULES_TEXT)
        st.caption("Moeda em R$ · horário em 24h · lote de 1 a 500 transações.")

    tab_lote, tab_individual, tab_historico = st.tabs(
        ["📦 Análise em Lote", "🎓 Aprenda Simulando", "🗂️ Histórico"]
    )
    with tab_lote:
        _aba_lote(api_ok, gemini_ok)
    with tab_individual:
        _aba_individual(api_ok)
    with tab_historico:
        _aba_historico(api_ok)

    st.markdown(f'<p class="aviso">{html.escape(_AVISOS_GERAIS)}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
