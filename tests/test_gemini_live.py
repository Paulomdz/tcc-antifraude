import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google import genai

from src.orquestacao.definicoes_agentes import DEFAULT_GEMINI_MODEL


@pytest.mark.live
def test_gemini_model_accepts_one_live_request(request):
    if request.config.getoption("-m").strip() != "live":
        pytest.skip("Use explicitamente pytest -m live para o teste real.")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        pytest.skip("GOOGLE_API_KEY ausente; teste live não executado.")

    model_name = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    model_name = model_name.removeprefix("models/")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Responda apenas: OK",
            config={"max_output_tokens": 16},
        )
    finally:
        client.close()

    assert response.text