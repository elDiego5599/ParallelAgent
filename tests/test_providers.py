"""Contrato de proveedores: resolución, mocks y manejo de errores."""

from types import SimpleNamespace

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


def test_resolve_prefixed(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_API_KEY", "k")
    assert isinstance(resolve_provider("algo-modelo"), PollinationsProvider)
    monkeypatch.delenv("POLLINATIONS_API_KEY")
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


def test_pollinations_requires_key(monkeypatch):
    monkeypatch.delenv("POLLINATIONS_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="POLLINATIONS_API_KEY"):
        resolve_provider("mistral")


def test_pollinations_parses_choices(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "  difflimpio  "}}]}

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp())
    p = PollinationsProvider("mistral", api_key="k")
    assert p.chat([{"role": "user", "content": "hola"}]) == "difflimpio"


def test_http_error_surfaces_server_message(monkeypatch):

    def boom(*a, **k):
        err = providers.requests.HTTPError("401 Client Error")
        err.response = SimpleNamespace(text='{"error": "key requerida"}')
        raise err

    monkeypatch.setattr(providers.requests, "post", boom)
    p = PollinationsProvider("mistral", api_key="mala")
    with pytest.raises(ProviderError, match="key requerida"):
        p.chat([{"role": "user", "content": "hola"}])


def test_retries_429_then_succeeds(monkeypatch):
    calls = []
    slept = []

    class Resp429:
        status_code = 429
        headers = {"Retry-After": "0"}
        text = "límite"

        def raise_for_status(self):
            err = providers.requests.HTTPError("429 Too Many Requests")
            err.response = self
            raise err

    class RespOk:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "listo"}}]}

    def flaky(*a, **k):
        calls.append(1)
        return Resp429() if len(calls) == 1 else RespOk()

    monkeypatch.setattr(providers.requests, "post", flaky)
    monkeypatch.setattr(providers.time, "sleep", lambda s: slept.append(s))
    p = GroqProvider.__new__(GroqProvider)
    p.model_id = "m"
    p.api_key = "k"
    p.URL = GroqProvider.URL
    assert p.chat([{"role": "user", "content": "hola"}]) == "listo"
    assert len(calls) == 2 and slept == [0]


def test_gives_up_after_two_retries(monkeypatch):
    class Resp429:
        status_code = 429
        headers = {}
        text = "límite"

        def raise_for_status(self):
            err = providers.requests.HTTPError("429 Too Many Requests")
            err.response = self
            raise err

    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: Resp429())
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    p = GroqProvider.__new__(GroqProvider)
    p.model_id = "m"
    p.api_key = "k"
    p.URL = GroqProvider.URL
    with pytest.raises(ProviderError, match="límite"):
        p.chat([{"role": "user", "content": "hola"}])
