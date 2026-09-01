"""Dashboard Streamlit — Sistema Multiagente de IA para detecção de fraudes."""

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Constantes

_DECISION_MAP = {
    "Aprovada": "APROVADO",
    "APROVADO": "APROVADO",
    "Revisão Humana necessária": "REVISÃO",
    "REVISÃO MANUAL": "REVISÃO",
    "REVISÃO": "REVISÃO",
    "Recusada": "BLOQUEADO",
    "BLOQUEADO": "BLOQUEADO",
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
_MAX_FINANCIAL_VALUE = 9_999_999.00
_MAX_BATCH_ROWS = 500
_MAX_UPLOAD_MB = 1024
_RISK_RULES_TEXT = (
    "Regras de risco ativadas: débito/transferência ≥ R$ 10.000 entre 00:00 e 05:59 "
    "é bloqueado automaticamente com score 0.95; o dashboard utiliza formato brasileiro "
    "e horário 24h em toda a interface."
)
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

# Funções auxiliares puras (testáveis sem Streamlit)


def map_decision(raw: str) -> str:
    """Converte decisão interna para label exibida no dashboard."""
    return _DECISION_MAP.get(raw, raw)


def decision_icon(decision: str) -> str:
    """Retorna emoji colorido para a decisão."""
    return _DECISION_ICON.get(decision, "⚪")


def run_analysis(transaction: dict, use_llm_judge: bool = True) -> dict:
    """Executa a análise com a mesma lógica validada em run_simulation.

    ``use_llm_judge=True`` (padrão, usado na análise individual) tenta o
    Agente Juiz real via CrewAI/Gemini, com fallback automático para a regra
    determinística. ``use_llm_judge=False`` (usado na análise em lote) pula
    direto para a regra determinística, evitando uma chamada de API por
    transação em lotes de até 500 itens.
    """
    import run_simulation

    return run_simulation.run_crewai_simulation(transaction, use_llm_judge=use_llm_judge)


def build_result_row(tx: dict, resultado: dict) -> dict:
    """Monta dicionário resumido de uma análise para exibição tabular."""
    decisao = map_decision(resultado["decision"]["decision"])
    return {
        "Tipo": tx.get("type", ""),
        "Valor (R$)": float(tx.get("amount", 0.0)),
        "Origem": tx.get("nameOrig", ""),
        "Destino": tx.get("nameDest", ""),
        "Comportamental": round(float(resultado["behavior"]["score"]), 3),
        "Temporal": round(float(resultado["temporal"]["score"]), 3),
        "Identidade": round(float(resultado["identity"]["score"]), 3),
        "Score Final": round(float(resultado["decision"]["score"]), 3),
        "Decisão": f"{decision_icon(decisao)} {decisao}",
    }


def _normalize_step(step: float | int) -> float:
    """Normaliza HH:MM para o intervalo [0, 1] em 24h completos."""
    total_minutes = float(step) % 1440.0
    return total_minutes / 1440.0


def _time_to_minutes(value: str) -> int:
    """Converte string HH:MM em minutos do dia."""
    try:
        hh, mm = map(int, str(value).split(":", 1))
        return (hh % 24) * 60 + (mm % 60)
    except ValueError:
        return 0


def _coerce_step_to_hour(value) -> int:
    """Normaliza o campo ``step`` vindo de um upload em lote para hora do dia.

    Aceita: ausente/NaN (usa meio-dia como padrão), string "HH:MM" (upload
    manual) ou valor já numérico — que pode ser tanto a hora do dia (0-23)
    quanto o step bruto do PaySim (1-744, horas desde o início da simulação).
    Em todos os casos numéricos, o valor é mantido como está e a hora do dia
    correta é obtida depois via `% 24` (ver `_hora_do_dia` em
    `ferramentas_crewai.py`).
    """
    if value is None:
        return 12
    if isinstance(value, float) and pd.isna(value):
        return 12
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 12
        if ":" in stripped:
            return _time_to_minutes(stripped) // 60
        try:
            return int(float(stripped))
        except ValueError:
            return 12
    try:
        return int(value)
    except (TypeError, ValueError):
        return 12


def _minutes_to_hh_mm(minutes: int) -> str:
    """Converte minutos do dia para string HH:MM em formato 24h."""
    try:
        minutes = int(minutes) % 1440
        hh = minutes // 60
        mm = minutes % 60
        return f"{hh:02d}:{mm:02d}"
    except (ValueError, TypeError):
        return "00:00"


def _is_valid_24h_time(value: str) -> bool:
    """Valida se a string está no formato HH:MM de 24h."""
    try:
        hh, mm = map(int, str(value).strip().split(':', 1))
        return 0 <= hh < 24 and 0 <= mm < 60
    except (ValueError, AttributeError):
        return False


def _format_banking_amount(value: float) -> float:
    """Arredonda valores financeiros no padrão bancário real (2 casas decimais)."""
    return round(float(value), 2)


def _format_currency(value: float) -> str:
    """Formata valores com separadores de milhares e 2 casas decimais."""
    return f"{float(value):,.2f}".replace(",", "~").replace(".", ",").replace("~", ".")


def _format_currency_br(value: float) -> str:
    """Formata valores em estilo brasileiro para exibição no dashboard."""
    return f"R$ {float(value):,.2f}".replace(",", "~").replace(".", ",").replace("~", ".")


def _calculate_new_balance(old_balance: float, amount: float, tx_type: str) -> float:
    """Calcula o novo saldo conforme a natureza da transação."""
    old_balance = _format_banking_amount(old_balance)
    amount = _format_banking_amount(amount)
    if str(tx_type).upper() in {"CASH_IN"}:
        return _format_banking_amount(old_balance + amount)
    return _format_banking_amount(old_balance - amount)


def _safe_session_value(key: str, fallback: float) -> float:
    """Garante que valores antigos da sessão não ultrapassem o limite do widget."""
    try:
        value = float(st.session_state.get(key, fallback))
    except (TypeError, ValueError):
        value = float(fallback)
    return min(max(value, 0.0), _MAX_FINANCIAL_VALUE)


def random_transaction() -> dict:
    """Gera uma transação aleatória com valores bancários realistas.

    ``step`` representa a HORA do dia (0-23), a mesma convenção usada em todo
    o pipeline (ver ``_hora_do_dia`` em ``ferramentas_crewai.py``): tanto para
    dados reais do PaySim (step = horas desde o início da simulação, 1-744)
    quanto para entrada interativa, ``step % 24`` recupera a hora do dia.
    ``step_norm`` mantém precisão de minuto (fração do dia) como feature
    auxiliar para os modelos de ML.
    """
    tx_type = random.choice(_TX_TYPES)
    amount = _format_banking_amount(random.uniform(10.0, _MAX_FINANCIAL_VALUE))
    bal_org = _format_banking_amount(random.uniform(amount, amount * 6))
    step_minutes = random.randint(0, 1439)
    hour = step_minutes // 60
    return {
        "step": hour,
        "type": tx_type,
        "amount": amount,
        "nameOrig": f"C{random.randint(1_000_000, 9_999_999)}",
        "oldbalanceOrg": bal_org,
        "newbalanceOrig": _calculate_new_balance(bal_org, amount, tx_type),
        "nameDest": f"C{random.randint(1_000_000, 9_999_999)}",
        "step_norm": _normalize_step(step_minutes),
    }


# Componentes de renderização

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
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
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


# Seção de Estatísticas (função pública para testabilidade)


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


# Interface principal


def main() -> None:
    """Ponto de entrada do dashboard SISTEMA MULTIAGENTE DE IA PARA DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS."""
    st.set_page_config(
        page_title="SISTEMA MULTIAGENTE DE IA PARA DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    # Cabeçalho da marca
    st.markdown(
        """
        <div class="cs-header">
            <h1>🛡️ SISTEMA MULTIAGENTE DE IA PARA DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS</h1>
            <p>Identidade profissional, formato brasileiro e regras de risco bancário aplicadas em tempo real.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(_RISK_RULES_TEXT, icon="✅")
    st.caption("Moeda exibida em R$ · Horário em 24h · Limite de análise até R$ 9.999.999,00")

    tab_individual, tab_lote, tab_stats = st.tabs(
        ["🔍 Análise Individual", "📦 Análise em Lote", "📊 Estatísticas"]
    )


    # Aba 1: Análise Individual


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
                    "Valor da transação (R$)",
                    min_value=0.0,
                    max_value=_MAX_FINANCIAL_VALUE,
                    value=_safe_session_value("amount", 5_000.0),
                    step=0.01,
                    format="%.2f",
                )
                name_orig = st.text_input("Conta Origem", value=_ss("nameOrig", "C1234567"))
                old_bal_org = st.number_input(
                    "Saldo anterior — Origem (R$)",
                    min_value=0.0,
                    max_value=_MAX_FINANCIAL_VALUE,
                    value=_safe_session_value("oldbalanceOrg", 10_000.0),
                    step=0.01,
                    format="%.2f",
                )
                auto_new_balance = _calculate_new_balance(float(old_bal_org), float(amount), tipo)

                st.caption(
                    "Pré-visualização do saldo: "
                    f"{_format_currency_br(float(old_bal_org))} "
                    f"{'+' if str(tipo).upper() == 'CASH_IN' else '-'} "
                    f"{_format_currency_br(float(amount))} = "
                    f"{_format_currency_br(float(auto_new_balance))}"
                )

                # Campo somente leitura (o valor é sempre recalculado a partir do
                # saldo anterior, valor e tipo); não precisamos do retorno.
                st.number_input(
                    "Novo saldo — Origem (R$)",
                    min_value=-_MAX_FINANCIAL_VALUE,
                    max_value=_MAX_FINANCIAL_VALUE,
                    value=auto_new_balance,
                    step=0.01,
                    format="%.2f",
                    disabled=True,
                )

            with col_b:
                step_text = st.text_input(
                    "Hora da simulação (HH:MM)",
                    value=f"{int(_ss('step', 12)) % 24:02d}:00",
                    placeholder="00:00 até 23:59",
                )
                name_dest = st.text_input("Conta Destino", value=_ss("nameDest", "C7654321"))

            submitted = st.form_submit_button(
                "🔍 Analisar Transação", use_container_width=True, type="primary"
            )

        if submitted:
            if not _is_valid_24h_time(step_text):
                st.error("Informe a hora no formato 24h válido: HH:MM (ex.: 03:30).")
                submitted = False

        if submitted:
            calculated_new_balance = _calculate_new_balance(float(old_bal_org), float(amount), tipo)
            step_minutes_form = _time_to_minutes(step_text)
            tx = {
                "step": step_minutes_form // 60,
                "type": tipo,
                "amount": float(amount),
                "nameOrig": name_orig,
                "oldbalanceOrg": float(old_bal_org),
                "newbalanceOrig": calculated_new_balance,
                "nameDest": name_dest,
                "step_norm": _normalize_step(step_minutes_form),
            }
            st.caption(
                "Cálculo aplicado: "
                f"{_format_currency_br(float(old_bal_org))} "
                f"{'+' if str(tipo).upper() == 'CASH_IN' else '-'} "
                f"{_format_currency_br(float(amount))} = "
                f"{_format_currency_br(float(calculated_new_balance))}"
            )
            with st.spinner("Analisando com agentes de IA..."):
                try:
                    resultado = run_analysis(tx, use_llm_judge=True)
                    st.success("Análise concluída.")
                    _render_single_result(resultado)
                except FileNotFoundError as exc:
                    st.error(f"Modelo não encontrado: {exc}")
                except Exception as exc:
                    st.error(f"Erro durante análise: {exc}")

    # Aba 2: Análise em Lote

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
                        max_value=min(len(df), _MAX_BATCH_ROWS),
                        value=min(100, len(df)),
                    )

                    if st.button("🚀 Iniciar Análise em Lote", use_container_width=True, type="primary"):
                        sample = df.head(max_rows)
                        rows: list[dict] = []
                        bar = st.progress(0, text="Iniciando...")

                        for i, (_, row) in enumerate(sample.iterrows()):
                            tx = row.to_dict()
                            tx["step"] = _coerce_step_to_hour(tx.get("step"))
                            tx.setdefault(
                                "step_norm", _normalize_step((tx["step"] % 24) * 60)
                            )
                            bar.progress(
                                (i + 1) / max_rows,
                                text=f"Analisando transação {i + 1} de {max_rows}…",
                            )
                            try:
                                res = run_analysis(tx, use_llm_judge=False)
                                rows.append(build_result_row(tx, res))
                            except Exception as exc:
                                rows.append({"Tipo": tx.get("type", ""), "Erro": str(exc)})

                        bar.empty()
                        st.success(f"✅ {max_rows} transações analisadas.")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        st.session_state["batch_results"] = rows

            except Exception as exc:
                st.error(f"Erro ao carregar arquivo: {exc}")

    # Aba 3: Estatísticas
    with tab_stats:
        st.header("Estatísticas das Análises")
        render_stats(st.session_state.get("batch_results", []))


main()
