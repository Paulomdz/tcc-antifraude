from src.orquestacao import definicoes_agentes


def test_gemini_model_uses_default_without_configured_model(monkeypatch):
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(definicoes_agentes, "LLM", fake_llm)

    definicoes_agentes.build_judge_llm()

    assert captured["model"] == "gemini/gemini-3.5-flash-lite"


def test_gemini_model_uses_env_value_and_normalizes_resource_name(monkeypatch):
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    monkeypatch.setattr(definicoes_agentes, "LLM", fake_llm)

    definicoes_agentes.build_judge_llm()

    assert captured["model"] == "gemini/gemini-2.5-flash"