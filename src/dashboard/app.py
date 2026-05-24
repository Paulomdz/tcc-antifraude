"""Dashboard Streamlit — ClearSafe Antifraude."""

import importlib
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_DECISION_MAP = {
    "Aprovada": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "Recusada": "BLOQUEADO",
}

_DECISION_ICON = {
    "APROVADO": "🟢",
    "REVISÃO": "🟡",
    "BLOQUEADO": "🔴",
}

_DECISION_BG = {
    "APROVADO": "#d4edda",
    "REVISÃO": "#fff3cd",
    "BLOQUEADO": "#f8d7da",
}

_DECISION_FG = {
    "APROVADO": "#155724",
    "REVISÃO": "#856404",
    "BLOQUEADO": "#721c24",
}

_TX_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]

_CSS = """
<style>
/* Fundo e corpo */
.block-container { padding-top: 1.5rem; max-width: 1100px; }
.stApp { background-color: #f8f9fb; }

/* Cabeçalho da marca */
.cs-header {
    background: linear-gradient(135deg, #ffffff 0%, #eef4fd 100%);
    border: 1px solid #d0e3f7;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.cs-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: -0.5px;
}
.cs-header p {
    margin: 6px 0 0 0;
    color: #5a7a9a;
    font-size: 1rem;
}

/* Badges de decisão */
.decision-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 8px 0 16px 0;
}

/* Cartão de agente */
.agent-card {
    background: #ffffff;
    border: 1px solid #e4ecf7;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-bottom: 8px;
}

/* Botões secundários */
div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
}

/* Separador de seção */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a94ae;
    margin: 20px 0 6px 0;
}
</style>
"""

# ---------------------------------------------------------------------------
# Funções auxiliares puras (testáveis sem Streamlit)
# ---------------------------------------------------------------------------


def map_decision(raw: str) -> str:
    """Converte decisão interna para label exibida no dashboard."""
    return _DECISION_MAP.get(raw, raw)


def decision_icon(decision: str) -> str:
    """Retorna emoji colorido para a decisão."""
    return _DECISION_ICON.get(decision, "⚪")


def run_analysis(transaction: dict) -> dict:
    """Executa análise de fraude chamando o pipeline de orquestração."""
    fluxo = importlib.import_module("src.orquestração.fluxo_crewai")
    return fluxo.orchestrate_transaction(transaction)


def build_result_row(tx: dict, resultado: dict) -> dict:
    """Monta dicionário resumido de uma análise para exibição tabular."""
    decisao = map_decision(resultado["decision"]["decision"])
    return {
        "Tipo": tx.get("type", ""),
        "Valor (R$)": tx.get("amount", 0.0),
        "Origem": tx.get("nameOrig", ""),
        "Destino": tx.get("nameDest", ""),
        "Comportamental": round(float(resultado["behavior"]["score"]), 3),
        "Temporal": round(float(resultado["temporal"]["score"]), 3),
        "Identidade": round(float(resultado["identity"]["score"]), 3),
        "Score Final": round(float(resultado["decision"]["score"]), 3),
        "Decisão": f"{decision_icon(decisao)} {decisao}",
    }


def random_transaction() -> dict:
    """Gera uma transação aleatória para simulação."""
    amount = round(random.uniform(100.0, 180_000.0), 2)
    bal_org = round(random.uniform(amount, amount * 4), 2)
    step = random.randint(1, 744)
    return {
        "step": step,
        "type": random.choice(_TX_TYPES),
        "amount": amount,
        "nameOrig": f"C{random.randint(1_000_000, 9_999_999)}",
        "oldbalanceOrg": bal_org,
        "newbalanceOrig": round(max(bal_org - amount, 0.0), 2),
        "nameDest": f"C{random.randint(1_000_000, 9_999_999)}",
        "oldbalanceDest": round(random.uniform(0.0, 40_000.0), 2),
        "newbalanceDest": round(random.uniform(0.0, 40_000.0) + amount, 2),
        "step_norm": step / 744.0,
    }


# ---------------------------------------------------------------------------
# Componentes de renderização
# ---------------------------------------------------------------------------


def _decision_badge(decisao: str) -> None:
    bg = _DECISION_BG.get(decisao, "#e9ecef")
    fg = _DECISION_FG.get(decisao, "#343a40")
    icon = decision_icon(decisao)
    st.markdown(
        f'<div class="decision-badge" style="background:{bg};color:{fg};">'
        f"{icon}&ensp;{decisao}</div>",
        unsafe_allow_html=True,
    )


def _render_agent_scores(resultado: dict) -> None:
    c1, c2, c3 = st.columns(3)
    for col, key, label in [
        (c1, "behavior", "Comportamental"),
        (c2, "temporal", "Temporal"),
        (c3, "identity", "Identidade"),
    ]:
        score = float(resultado[key]["score"])
        explanation = resultado[key]["details"]["explanation"]
        with col:
            st.markdown(f'<div class="agent-card">', unsafe_allow_html=True)
            st.metric(label=label, value=f"{score:.3f}")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(explanation)
            st.markdown("</div>", unsafe_allow_html=True)


def _render_single_result(resultado: dict) -> None:
    decision_raw = resultado["decision"]["decision"]
    decisao = map_decision(decision_raw)
    score_final = float(resultado["decision"]["score"])

    st.markdown('<p class="section-label">Decisão</p>', unsafe_allow_html=True)
    _decision_badge(decisao)

    col_score, col_tipo = st.columns(2)
    col_score.metric("Score Final", f"{score_final:.3f}")
    col_tipo.metric("Tipo de Transação", resultado["transaction"].get("type", "N/A"))

    st.info(resultado["decision"]["justification"], icon="ℹ️")

    st.markdown('<p class="section-label">Análise por Agente</p>', unsafe_allow_html=True)
    _render_agent_scores(resultado)


# ---------------------------------------------------------------------------
# Seção de Estatísticas (função pública para testabilidade)
# ---------------------------------------------------------------------------


def render_stats(resultados: list) -> None:
    """Renderiza a aba de estatísticas a partir de uma lista de resultados de lote."""
    if not resultados:
        st.info(
            "Execute uma **Análise em Lote** para visualizar as estatísticas aqui.",
            icon="ℹ️",
        )
        return

    try:
        import plotly.express as px

        valid = [r for r in resultados if "Decisão" in r and "Score Final" in r]
        if not valid:
            st.warning("Nenhum resultado válido encontrado.")
            return

        def _clean(label: str) -> str:
            for icon in ("🟢 ", "🟡 ", "🔴 "):
                label = label.replace(icon, "")
            return label

        decisoes = [_clean(r["Decisão"]) for r in valid]
        scores = [r["Score Final"] for r in valid]

        total = len(valid)
        aprovados = decisoes.count("APROVADO")
        revisoes = decisoes.count("REVISÃO")
        bloqueados = decisoes.count("BLOQUEADO")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analisado", total)
        c2.metric("✅ Aprovados", aprovados)
        c3.metric("⚠️ Em Revisão", revisoes)
        c4.metric("🚫 Bloqueados", bloqueados)

        st.divider()

        col_pie, col_hist = st.columns(2)

        with col_pie:
            counts = pd.Series(decisoes).value_counts().reset_index()
            counts.columns = ["Decisão", "Contagem"]
            fig_pie = px.pie(
                counts,
                values="Contagem",
                names="Decisão",
                title="Distribuição de Decisões",
                color="Decisão",
                color_discrete_map={
                    "APROVADO": "#2ecc71",
                    "REVISÃO": "#f39c12",
                    "BLOQUEADO": "#e74c3c",
                },
                hole=0.35,
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            fig_hist = px.histogram(
                x=scores,
                nbins=20,
                title="Distribuição de Scores Finais",
                labels={"x": "Score de Risco", "y": "Frequência"},
                color_discrete_sequence=["#4a90d9"],
            )
            fig_hist.add_vline(
                x=0.5, line_dash="dash", line_color="#f39c12", annotation_text="Revisão (0.5)"
            )
            fig_hist.add_vline(
                x=0.8, line_dash="dash", line_color="#e74c3c", annotation_text="Bloqueio (0.8)"
            )
            fig_hist.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        agent_cols = ["Comportamental", "Temporal", "Identidade", "Score Final"]
        score_data = [r for r in valid if "Comportamental" in r]
        if score_data:
            df_scores = pd.DataFrame(score_data)[agent_cols]
            fig_box = px.box(
                df_scores,
                title="Distribuição de Scores por Agente",
                labels={"variable": "Agente", "value": "Score"},
                color_discrete_sequence=["#4a90d9"],
            )
            st.plotly_chart(fig_box, use_container_width=True)

    except ImportError:
        st.warning("Instale `plotly` para visualizar os gráficos: `pip install plotly`")


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Ponto de entrada do dashboard ClearSafe Antifraude."""
    st.set_page_config(
        page_title="ClearSafe Antifraude",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    # Cabeçalho da marca
    st.markdown(
        """
        <div class="cs-header">
            <h1>🛡️ ClearSafe Antifraude</h1>
            <p>Detecção inteligente de fraudes em transações financeiras</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_individual, tab_lote, tab_stats = st.tabs(
        ["🔍 Análise Individual", "📦 Análise em Lote", "📊 Estatísticas"]
    )

    # ------------------------------------------------------------------
    # Aba 1: Análise Individual
    # ------------------------------------------------------------------
    with tab_individual:
        st.header("Analisar Transação")

        # Botão de simulação fora do form para não recarregar a página inteira
        col_sim, _ = st.columns([1, 3])
        if col_sim.button("🎲 Simular nova transação", use_container_width=True):
            sim = random_transaction()
            st.session_state.update(sim)

        st.divider()

        # Recupera valores da sessão (gerados pela simulação ou padrão)
        def _ss(key, default):
            return st.session_state.get(key, default)

        with st.form("form_transacao"):
            col_a, col_b = st.columns(2)

            with col_a:
                tipo = st.selectbox(
                    "Tipo de Transação",
                    _TX_TYPES,
                    index=_TX_TYPES.index(_ss("type", "TRANSFER")),
                )
                amount = st.number_input(
                    "Valor (R$)", min_value=0.0, value=float(_ss("amount", 5_000.0)), step=500.0
                )
                name_orig = st.text_input("Conta Origem", value=_ss("nameOrig", "C1234567"))
                old_bal_org = st.number_input(
                    "Saldo Anterior — Origem (R$)", value=float(_ss("oldbalanceOrg", 10_000.0)), step=500.0
                )
                new_bal_org = st.number_input(
                    "Novo Saldo — Origem (R$)", value=float(_ss("newbalanceOrig", 5_000.0)), step=500.0
                )

            with col_b:
                step = st.number_input(
                    "Step (hora da simulação)", min_value=1, max_value=744, value=int(_ss("step", 100))
                )
                name_dest = st.text_input("Conta Destino", value=_ss("nameDest", "C7654321"))
                old_bal_dest = st.number_input(
                    "Saldo Anterior — Destino (R$)", value=float(_ss("oldbalanceDest", 1_000.0)), step=500.0
                )
                new_bal_dest = st.number_input(
                    "Novo Saldo — Destino (R$)", value=float(_ss("newbalanceDest", 6_000.0)), step=500.0
                )

            submitted = st.form_submit_button(
                "🔍 Analisar Transação", use_container_width=True, type="primary"
            )

        if submitted:
            tx = {
                "step": int(step),
                "type": tipo,
                "amount": float(amount),
                "nameOrig": name_orig,
                "oldbalanceOrg": float(old_bal_org),
                "newbalanceOrig": float(new_bal_org),
                "nameDest": name_dest,
                "oldbalanceDest": float(old_bal_dest),
                "newbalanceDest": float(new_bal_dest),
                "step_norm": float(step) / 744.0,
            }
            with st.spinner("Analisando com agentes de IA..."):
                try:
                    resultado = run_analysis(tx)
                    st.success("Análise concluída.")
                    _render_single_result(resultado)
                except FileNotFoundError as exc:
                    st.error(f"Modelo não encontrado: {exc}")
                except Exception as exc:
                    st.error(f"Erro durante análise: {exc}")

    # ------------------------------------------------------------------
    # Aba 2: Análise em Lote
    # ------------------------------------------------------------------
    with tab_lote:
        st.header("Análise em Lote")
        st.markdown(
            "Envie um arquivo **CSV** ou **Parquet** com transações para análise automática. "
            "Colunas obrigatórias: `amount`, `nameOrig`, `nameDest`, `type`."
        )

        uploaded = st.file_uploader(
            "Selecione o arquivo de transações",
            type=["csv", "parquet"],
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_parquet(uploaded)

                st.info(
                    f"Arquivo carregado: **{len(df):,}** transações · "
                    f"Colunas: {', '.join(df.columns.tolist())}",
                    icon="📄",
                )
                st.dataframe(df.head(10), use_container_width=True)

                required_cols = {"amount", "nameOrig", "nameDest", "type"}
                missing = required_cols - set(df.columns)
                if missing:
                    st.warning(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
                else:
                    max_rows = st.slider(
                        "Quantas transações analisar?",
                        min_value=1,
                        max_value=min(len(df), 50),
                        value=min(5, len(df)),
                    )

                    if st.button("🚀 Iniciar Análise em Lote", use_container_width=True, type="primary"):
                        sample = df.head(max_rows)
                        rows: list[dict] = []
                        bar = st.progress(0, text="Iniciando...")

                        for i, (_, row) in enumerate(sample.iterrows()):
                            tx = row.to_dict()
                            tx.setdefault("step_norm", float(tx.get("step", 100)) / 744.0)
                            bar.progress(
                                (i + 1) / max_rows,
                                text=f"Analisando transação {i + 1} de {max_rows}…",
                            )
                            try:
                                res = run_analysis(tx)
                                rows.append(build_result_row(tx, res))
                            except Exception as exc:
                                rows.append({"Tipo": tx.get("type", ""), "Erro": str(exc)})

                        bar.empty()
                        st.success(f"✅ {max_rows} transações analisadas.")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        st.session_state["batch_results"] = rows

            except Exception as exc:
                st.error(f"Erro ao carregar arquivo: {exc}")

    # ------------------------------------------------------------------
    # Aba 3: Estatísticas
    # ------------------------------------------------------------------
    with tab_stats:
        st.header("Estatísticas das Análises")
        render_stats(st.session_state.get("batch_results", []))


main()
