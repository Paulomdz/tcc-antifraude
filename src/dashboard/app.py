"""Dashboard Streamlit para visualização e análise de fraudes - TCC Antifraude."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Garante que a raiz do projeto está no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Funções auxiliares puras (testáveis sem Streamlit)
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


# ---------------------------------------------------------------------------
# Componentes de renderização Streamlit
# ---------------------------------------------------------------------------

def _render_agent_scores(resultado: dict) -> None:
    """Exibe os scores dos três agentes em colunas."""
    c1, c2, c3 = st.columns(3)
    for col, key, label in [
        (c1, "behavior", "Comportamental"),
        (c2, "temporal", "Temporal"),
        (c3, "identity", "Identidade"),
    ]:
        score = float(resultado[key]["score"])
        explanation = resultado[key]["details"]["explanation"]
        with col:
            st.metric(label=label, value=f"{score:.3f}")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(explanation)


def _render_single_result(resultado: dict) -> None:
    """Exibe resultado completo de uma análise individual."""
    decision_raw = resultado["decision"]["decision"]
    decisao = map_decision(decision_raw)
    score_final = float(resultado["decision"]["score"])

    col_dec, col_score, col_tipo = st.columns(3)
    col_dec.metric("Decisão", f"{decision_icon(decisao)} {decisao}")
    col_score.metric("Score Final", f"{score_final:.3f}")
    col_tipo.metric("Tipo", resultado["transaction"].get("type", "N/A"))

    st.info(resultado["decision"]["justification"])

    st.subheader("Scores dos Agentes")
    _render_agent_scores(resultado)


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada do dashboard Streamlit."""
    st.set_page_config(
        page_title="TCC Antifraude",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Sistema Antifraude — Dashboard TCC")
    st.caption("Detecção de fraudes em transações financeiras com ML multi-agente")
    st.divider()

    tab_individual, tab_paysim, tab_lote, tab_stats = st.tabs(
        ["📋 Análise Individual", "📊 Dataset PaySim", "📦 Análise em Lote", "📈 Estatísticas"]
    )

    # ------------------------------------------------------------------
    # Aba 1: Análise Individual
    # ------------------------------------------------------------------
    with tab_individual:
        st.header("Analisar Transação")

        with st.form("form_transacao"):
            col_a, col_b = st.columns(2)

            with col_a:
                tipo = st.selectbox(
                    "Tipo de Transação",
                    ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
                )
                amount = st.number_input("Valor (R$)", min_value=0.0, value=5000.0, step=500.0)
                name_orig = st.text_input("Conta Origem", value="C1234567")
                old_bal_org = st.number_input("Saldo Anterior Origem (R$)", value=10000.0, step=500.0)
                new_bal_org = st.number_input("Novo Saldo Origem (R$)", value=5000.0, step=500.0)

            with col_b:
                step = st.number_input("Step (hora da simulação)", min_value=1, max_value=744, value=100)
                name_dest = st.text_input("Conta Destino", value="C7654321")
                old_bal_dest = st.number_input("Saldo Anterior Destino (R$)", value=1000.0, step=500.0)
                new_bal_dest = st.number_input("Novo Saldo Destino (R$)", value=6000.0, step=500.0)

            submitted = st.form_submit_button("🔍 Analisar Transação", use_container_width=True)

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
            with st.spinner("Analisando transação com agentes ML..."):
                try:
                    resultado = run_analysis(tx)
                    st.success("Análise concluída!")
                    _render_single_result(resultado)
                except FileNotFoundError as exc:
                    st.error(f"Modelo não encontrado: {exc}")
                except Exception as exc:
                    st.error(f"Erro durante análise: {exc}")

    # ------------------------------------------------------------------
    # Aba 2: Dataset PaySim
    # ------------------------------------------------------------------
    with tab_paysim:
        st.header("Dataset PaySim")
        st.markdown("Visualize e explore transações do dataset PaySim processado.")

        try:
            from pathlib import Path
            data_path = Path(__file__).resolve().parents[2] / "data" / "paysim_dados_processados.parquet"
            
            if data_path.exists():
                df_paysim = pd.read_parquet(data_path)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Total de Transações", f"{len(df_paysim):,}")
                col2.metric("🔴 Fraudes Detectadas", f"{df_paysim['isFraud'].sum():,}")
                col3.metric("✅ Transações Legítimas", f"{(df_paysim['isFraud'] == 0).sum():,}")
                
                st.subheader("Distribuição de Transações")
                tipo_counts = df_paysim["type"].value_counts()
                import plotly.express as px
                fig_tipo = px.bar(
                    x=tipo_counts.index,
                    y=tipo_counts.values,
                    labels={"x": "Tipo de Transação", "y": "Quantidade"},
                    title="Transações por Tipo",
                    color_discrete_sequence=["#3498db"]
                )
                st.plotly_chart(fig_tipo, use_container_width=True)
                
                st.subheader("Amostra de Transações")
                n_rows = st.slider("Mostrar primeiras N transações", min_value=5, max_value=100, value=10)
                st.dataframe(df_paysim.head(n_rows), use_container_width=True)
                
                st.subheader("Estatísticas Descritivas")
                st.dataframe(df_paysim.describe(), use_container_width=True)
                
            else:
                st.warning(f"Dataset não encontrado em: {data_path}")
                st.info("Execute: `python src/preprocessamento/carregar_paysim.py` para processar o dataset.")
        except Exception as exc:
            st.error(f"Erro ao carregar dataset: {exc}")

    # ------------------------------------------------------------------
    # Aba 3: Análise em Lote
    # ------------------------------------------------------------------
    with tab_lote:
        st.header("Análise em Lote")
        
        col_upload, col_dataset = st.columns(2)
        
        with col_upload:
            st.subheader("📤 Enviar Arquivo")
            st.markdown("Envie um arquivo **CSV** ou **Parquet** com transações para análise.")
            uploaded = st.file_uploader(
                "Selecione o arquivo de transações",
                type=["csv", "parquet"],
                help="Colunas obrigatórias: amount, nameOrig, nameDest, type",
            )
            
            if uploaded is not None:
                try:
                    if uploaded.name.endswith(".csv"):
                        df = pd.read_csv(uploaded)
                    else:
                        df = pd.read_parquet(uploaded)

                    st.info(f"Arquivo carregado: **{len(df):,}** transações | Colunas: {list(df.columns)}")
                    st.dataframe(df.head(10), use_container_width=True)

                    required_cols = {"amount", "nameOrig", "nameDest", "type"}
                    missing = required_cols - set(df.columns)
                    if missing:
                        st.warning(f"Colunas obrigatórias ausentes: {missing}")
                    else:
                        max_rows = st.slider(
                            "Número de transações a analisar",
                            min_value=1,
                            max_value=min(len(df), 50),
                            value=min(5, len(df)),
                        )

                        if st.button("🚀 Iniciar Análise em Lote", use_container_width=True):
                            sample = df.head(max_rows)
                            rows = []
                            progress_bar = st.progress(0, text="Iniciando...")

                            for i, (_, row) in enumerate(sample.iterrows()):
                                tx = row.to_dict()
                                tx.setdefault("step_norm", float(tx.get("step", 100)) / 744.0)
                                progress_bar.progress(
                                    (i + 1) / max_rows,
                                    text=f"Analisando transação {i + 1}/{max_rows}...",
                                )
                                try:
                                    res = run_analysis(tx)
                                    rows.append(build_result_row(tx, res))
                                except Exception as exc:
                                    rows.append(
                                        {"Tipo": tx.get("type", ""), "Erro": str(exc)}
                                    )

                            progress_bar.empty()
                            st.success(f"Análise de {max_rows} transações concluída!")
                            df_res = pd.DataFrame(rows)
                            st.dataframe(df_res, use_container_width=True)
                            st.session_state["batch_results"] = rows

                except Exception as exc:
                    st.error(f"Erro ao carregar arquivo: {exc}")
        
        with col_dataset:
            st.subheader("📊 Usar Dataset PaySim")
            st.markdown("Carregue transações diretamente do dataset processado.")
            
            try:
                from pathlib import Path
                data_path = Path(__file__).resolve().parents[2] / "data" / "paysim_dados_processados.parquet"
                
                if data_path.exists():
                    if st.button("📥 Carregar PaySim Processado", use_container_width=True):
                        df_paysim = pd.read_parquet(data_path)
                        
                        max_rows_paysim = st.slider(
                            "Quantas transações do PaySim analisar?",
                            min_value=1,
                            max_value=min(len(df_paysim), 100),
                            value=min(10, len(df_paysim)),
                            key="paysim_slider"
                        )
                        
                        if st.button("🚀 Analisar Amostra PaySim", use_container_width=True):
                            sample = df_paysim.head(max_rows_paysim)
                            rows = []
                            progress_bar = st.progress(0, text="Iniciando análise PaySim...")
                            
                            for i, (_, row) in enumerate(sample.iterrows()):
                                tx = row.to_dict()
                                tx.setdefault("step_norm", float(tx.get("step", 100)) / 744.0)
                                progress_bar.progress(
                                    (i + 1) / max_rows_paysim,
                                    text=f"Analisando {i + 1}/{max_rows_paysim}...",
                                )
                                try:
                                    res = run_analysis(tx)
                                    rows.append(build_result_row(tx, res))
                                except Exception as exc:
                                    rows.append(
                                        {"Tipo": tx.get("type", ""), "Erro": str(exc)}
                                    )
                            
                            progress_bar.empty()
                            st.success(f"✅ Análise de {max_rows_paysim} transações concluída!")
                            df_res = pd.DataFrame(rows)
                            st.dataframe(df_res, use_container_width=True)
                            st.session_state["batch_results"] = rows
                else:
                    st.warning("Dataset PaySim não encontrado.")
            except Exception as exc:
                st.error(f"Erro: {exc}")

    # ------------------------------------------------------------------
    # Aba 4: Estatísticas
    # ------------------------------------------------------------------
    with tab_stats:
        st.header("Estatísticas das Análises")

        resultados = st.session_state.get("batch_results", [])

        if not resultados:
            st.info(
                "Execute uma **Análise em Lote** para visualizar as estatísticas aqui.",
                icon="ℹ️",
            )
        else:
            try:
                import plotly.express as px

                valid = [r for r in resultados if "Decisão" in r and "Score Final" in r]
                if not valid:
                    st.warning("Nenhum resultado válido encontrado.")
                    return

                decisoes = [
                    r["Decisão"]
                    .replace("🟢 ", "")
                    .replace("🟡 ", "")
                    .replace("🔴 ", "")
                    for r in valid
                ]
                scores = [r["Score Final"] for r in valid]

                col1, col2 = st.columns(2)

                with col1:
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
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    fig_hist = px.histogram(
                        x=scores,
                        nbins=20,
                        title="Distribuição de Scores Finais",
                        labels={"x": "Score", "y": "Frequência"},
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_hist.add_vline(
                        x=0.5,
                        line_dash="dash",
                        line_color="#f39c12",
                        annotation_text="Revisão (0.5)",
                    )
                    fig_hist.add_vline(
                        x=0.8,
                        line_dash="dash",
                        line_color="#e74c3c",
                        annotation_text="Bloqueio (0.8)",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                agent_cols = ["Comportamental", "Temporal", "Identidade", "Score Final"]
                score_data = [r for r in valid if "Comportamental" in r]
                if score_data:
                    df_scores = pd.DataFrame(score_data)[agent_cols]
                    fig_box = px.box(
                        df_scores,
                        title="Distribuição de Scores por Agente",
                        labels={"variable": "Agente", "value": "Score"},
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            except ImportError:
                st.warning("Instale `plotly` para visualizar os gráficos: `pip install plotly`")


main()
