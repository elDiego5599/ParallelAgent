"""Contrato de proveedores: resolución, mocks y manejo de errores."""

import pytest

import providers
from providers import (
    GroqProvider,
    MockProvider,
    OpenRouterProvider,
    PollinationsProvider,
    ProviderError,
    resolve_provider,
)


def test_resolve_mock_variants():
    assert isinstance(resolve_provider("mock"), MockProvider)
    assert isinstance(resolve_provider("mock:3"), MockProvider)
    assert isinstance(resolve_provider("mock:3:llama"), MockProvider)
    assert resolve_provider("mock:3").rounds_before_consensus == 3
    assert resolve_provider("mock").rounds_before_consensus == 2


def test_resolve_prefixed():
    assert isinstance(resolve_provider("algo-modelo"), PollinationsProvider)
    with pytest.raises(ProviderError):
        resolve_provider("groq/llama-x")
    with pytest.raises(ProviderError):
        resolve_provider("openrouter/anthropic/claude")


def test_mock_follows_round_marker():
    m = resolve_provider("mock:3:claude")
    debating = m.chat([{"role": "user", "content": "[Ronda 2]\nTarea: x"}])
    assert "DEBATIENDO" in debating
    agreed = m.chat([{"role": "user", "content": "[Ronda 3]\nTarea: x"}])
    assert "CONSENSO_ALCANZADO" in agreed


def test_openai_compatible_wraps_transport_errors(monkeypatch):
    def boom(*a, **k):
        raise providers.requests.RequestException("caído")

    monkeypatch.setattr(providers.requests, "post", boom)
    p = GroqProvider.__new__(GroqProvider)
    p.model_id = "m"
    p.api_key = "k"
    p.URL = GroqProvider.URL
    with pytest.raises(ProviderError):
        p.chat([{"role": "user", "content": "hola"}])


def test_openai_compatible_rejects_malformed_payload(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"sin": "choices"}

    monkeypatch.setattr(
        providers.requests, "post", lambda *a, **k: Resp()
    )
    p = OpenRouterProvider.__new__(OpenRouterProvider)
    p.model_id = "m"
    p.api_key = "k"
    p.URL = OpenRouterProvider.URL
    with pytest.raises(ProviderError):
        p.chat([{"role": "user", "content": "hola"}])


def test_pollinations_accepts_json_choices(monkeypatch):
    class Resp:
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "  difflimpio  "}}]}

        text = "ignorado"

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp())
    p = PollinationsProvider("m")
    assert p.chat([{"role": "user", "content": "hola"}]) == "difflimpio"


def test_pollinations_rejects_empty_text(monkeypatch):
    class Resp:
        headers = {"content-type": "text/plain; charset=utf-8"}

        def raise_for_status(self):
            pass

        text = "   "

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(ProviderError):
        PollinationsProvider("m").chat([{"role": "user", "content": "hola"}])
