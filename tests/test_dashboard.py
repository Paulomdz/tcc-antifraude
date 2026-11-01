from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.dashboard import app as dashboard
from src.preprocessamento.carregar_dataset import TRANSACTION_TYPES


def test_dashboard_helper_functions_use_pix_dataset():
    assert dashboard._TX_TYPES == TRANSACTION_TYPES
    assert dashboard.hora_norm_from_step(0) == 0.0
    assert dashboard.hora_norm_from_step(23) == 1.0
    assert dashboard.hora_norm_from_step(100) == 4 / 23
    assert dashboard._time_to_minutes("03:45") == 225
    assert dashboard._time_to_minutes("25:10") == 70
    assert dashboard._is_valid_24h_time("23:59")
    assert not dashboard._is_valid_24h_time("24:00")
    assert dashboard._coerce_step_to_hour("08:30") == 8
    assert dashboard._coerce_step_to_hour("100") == 100
    assert dashboard._coerce_step_to_hour(None) == 12
    assert dashboard._coerce_step_to_hour("invalid") == 12


def test_dashboard_formatting_and_random_transaction(monkeypatch):
    monkeypatch.setattr(dashboard.random, "randint", lambda low, _high: low)
    monkeypatch.setattr(dashboard.random, "uniform", lambda low, _high: low)
    monkeypatch.setattr(dashboard.random, "choice", lambda values: values[0])

    transaction = dashboard.random_transaction()

    assert transaction["type"] in TRANSACTION_TYPES
    assert transaction["step"] == 0
    assert transaction["hora_norm"] == 0
    assert "step_norm" not in transaction
    assert dashboard._calculate_new_balance(100, 25, "PIX_P2P") == 75
    assert dashboard._calculate_new_balance(100, 25, "CASH_IN") == 125
    assert dashboard._format_currency_br(1234.5) == "R$ 1.234,50"
    assert dashboard.map_decision("Recusada") == "BLOQUEADO"
    assert dashboard.decision_icon("BLOQUEADO") == "🔴"


def test_build_result_row_formats_scores_and_decision():
    specialist = {"score": 0.25}
    result = {
        "decision": {"decision": "Aprovada", "score": 0.3},
        "behavior": specialist,
        "temporal": specialist,
        "identity": specialist,
    }

    row = dashboard.build_result_row(
        {"type": "PIX_P2P", "amount": 100, "nameOrig": "A", "nameDest": "B"},
        result,
    )

    assert row["Tipo"] == "PIX_P2P"
    assert row["Score Final"] == 0.3
    assert row["Decisão"] == "🟢 APROVADO"


def test_dashboard_renders_without_running_on_import():
    app_path = Path(dashboard.__file__)

    app_test = AppTest.from_file(str(app_path)).run()

    assert not app_test.exception
    assert any("Analisar Transação" in element.value for element in app_test.header)


class FakeProgress:
    def progress(self, *_args, **_kwargs):
        return self

    def empty(self):
        return None


class FakeColumn:
    def __init__(self, streamlit):
        self.streamlit = streamlit

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def button(self, label, **_kwargs):
        return label.startswith("🎲")

    def metric(self, *_args, **_kwargs):
        return None


class FakeStreamlit:
    def __init__(self, upload):
        self.session_state = {}
        self.upload = upload

    def __getattr__(self, name):
        if name in {
            "set_page_config",
            "markdown",
            "info",
            "caption",
            "divider",
            "header",
            "dataframe",
            "warning",
            "error",
            "success",
            "plotly_chart",
        }:
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)

    def tabs(self, _labels):
        return [FakeColumn(self) for _ in range(3)]

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [FakeColumn(self) for _ in range(count)]

    def form(self, *_args, **_kwargs):
        return nullcontext()

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def number_input(self, _label, **kwargs):
        return kwargs.get("value", 0.0)

    def text_input(self, _label, value="", **_kwargs):
        return value

    def form_submit_button(self, *_args, **_kwargs):
        return True

    def spinner(self, *_args, **_kwargs):
        return nullcontext()

    def file_uploader(self, *_args, **_kwargs):
        return self.upload

    def slider(self, _label, **kwargs):
        return kwargs["value"]

    def button(self, *_args, **_kwargs):
        return True

    def progress(self, *_args, **_kwargs):
        return FakeProgress()


class UploadedCsv(BytesIO):
    name = "transactions.csv"


def fake_analysis(transaction, use_llm_judge=True):
    specialist = {"score": 0.2, "details": {"explanation": "ok"}}
    return {
        "transaction": transaction,
        "decision": {
            "decision": "Aprovada",
            "score": 0.2,
            "justification": "ok",
        },
        "behavior": specialist,
        "temporal": specialist,
        "identity": specialist,
    }


def test_main_handles_manual_analysis_and_uploaded_batch(monkeypatch):
    upload = UploadedCsv(
        b"step,type,amount,nameOrig,nameDest\n4,PIX_P2P,125,C1,C2\n"
    )
    fake_st = FakeStreamlit(upload)
    monkeypatch.setattr(dashboard, "st", fake_st)
    monkeypatch.setattr(dashboard, "run_analysis", fake_analysis)
    monkeypatch.setattr(dashboard.random, "randint", lambda low, _high: low)

    dashboard.main()

    assert fake_st.session_state["step"] == 0
    assert fake_st.session_state["batch_results"][0]["Tipo"] == "PIX_P2P"


def test_main_reports_upload_missing_required_columns(monkeypatch):
    upload = UploadedCsv(b"amount,other\n125,value\n")
    fake_st = FakeStreamlit(upload)
    monkeypatch.setattr(dashboard, "st", fake_st)
    monkeypatch.setattr(dashboard, "run_analysis", fake_analysis)

    dashboard.main()
