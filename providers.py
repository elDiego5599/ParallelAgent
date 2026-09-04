"""Adaptadores de proveedores de LLM para ParallelAgent.

Todos los proveedores implementan la interfaz común BaseProvider.
"""

from abc import ABC, abstractmethod
import os
import re
import time
from typing import Any, Dict, List
import requests


class ProviderError(Exception):
    """Excepción base para fallos de red, autenticación o formato en proveedores."""

    pass


def _retry_after_seconds(response: Any) -> float:
    """Lee el header Retry-After (segundos). Por defecto 20, tope 60."""
    try:
        wait = float(response.headers.get("Retry-After", "20"))
    except (ValueError, TypeError, AttributeError):
        wait = 20.0
    return max(0.0, min(wait, 60.0))


class BaseProvider(ABC):
    def __init__(self, model_id: str, **kwargs: Any):
        self.model_id = model_id
        self.kwargs = kwargs

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Envía una lista de mensajes [{'role': '...', 'content': '...'}]

        y devuelve la respuesta en texto.
        """
        pass


class MockProvider(BaseProvider):
    """Proveedor simulado para pruebas locales del orquestador sin costo ni red.

    Sintaxis de model_id:
      - 'mock' -> consensúa en ronda 2
      - 'mock:3' -> consensúa en ronda 3
      - 'mock:3:nombre' -> consensúa en ronda 3, identificado como 'nombre'
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.call_count = 0
        self.rounds_before_consensus = self._parse_target_round(model_id)

    def _parse_target_round(self, model_id: str) -> int:
        parts = model_id.split(":")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return 2

    def _extract_round_from_messages(
        self, messages: List[Dict[str, str]]
    ) -> int | None:
        """Intenta inferir la ronda actual buscando marcas '[Ronda X]' en el prompt."""
        for msg in reversed(messages):
            match = re.search(r"\[Ronda\s+(\d+)\]", msg.get("content", ""))
            if match:
                return int(match.group(1))
        return None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self.call_count += 1
        current_round = (
            self._extract_round_from_messages(messages) or self.call_count
        )

        if current_round < self.rounds_before_consensus:
            return (
                f"[{self.model_id}] Analicé la tarea. Propongo desacoplar la interfaz y "
                f"revisar el ciclo de vida para evitar fugas.\n\n"
                f"ESTADO: DEBATIENDO"
            )

        return (
            f"[{self.model_id}] Coincido plenamente con los ajustes propuestos por la mesa. "
            f"El diseño es seguro y cubre todos los casos borde.\n\n"
            f"ESTADO: CONSENSO_ALCANZADO"
        )


class OpenAICompatibleProvider(BaseProvider):
    """Clase base reutilizable para proveedores que siguen la spec de OpenAI."""

    URL: str = ""

    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id)
        self.api_key = api_key
        if not self.api_key:
            raise ProviderError(
                f"Falta API key requerida para el modelo '{model_id}'."
            )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.3,
        }
        retries = 0
        while True:
            try:
                response = requests.post(
                    self.URL, headers=self._get_headers(), json=payload, timeout=90
                )
                response.raise_for_status()
                data = response.json()

                # Validación defensiva del payload estándar OpenAI
                choices = data.get("choices")
                if not choices or not isinstance(choices, list):
                    raise ValueError(
                        f"Respuesta sin choices válidos: {str(data)[:200]}"
                    )

                content = choices[0].get("message", {}).get("content")
                if content is None:
                    raise ValueError(
                        f"Mensaje sin campo 'content': {str(data)[:200]}"
                    )

                return content.strip()

            except requests.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429 and retries < 2:
                    time.sleep(_retry_after_seconds(getattr(e, "response", None)))
                    retries += 1
                    continue
                detail = getattr(getattr(e, "response", None), "text", "") or ""
                raise ProviderError(
                    f"Fallo en API compatible ({self.model_id}): {e} {detail[:300]}".strip()
                )
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                raise ProviderError(
                    f"Fallo en API compatible ({self.model_id}): {e}"
                )


class GroqProvider(OpenAICompatibleProvider):
    """Proveedor Groq (requiere GROQ_API_KEY)."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model_id: str, api_key: str | None = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        super().__init__(model_id=model_id, api_key=key)


class OpenRouterProvider(OpenAICompatibleProvider):
    """Proveedor OpenRouter (requiere OPENROUTER_API_KEY)."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_id: str, api_key: str | None = None):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(model_id=model_id, api_key=key)

    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()
        headers["HTTP-Referer"] = "https://github.com/ParallelAgent"
        headers["X-Title"] = "ParallelAgent"
        return headers


class PollinationsProvider(OpenAICompatibleProvider):
    """Proveedor Pollinations vía gateway unificado (requiere clave gratuita).

    Clave gratuita en https://enter.pollinations.ai/keys -> POLLINATIONS_API_KEY.
    """

    URL = "https://gen.pollinations.ai/v1/chat/completions"

    def __init__(self, model_id: str, api_key: str | None = None):
        key = api_key or os.getenv("POLLINATIONS_API_KEY")
        if not key:
            raise ProviderError(
                "Falta POLLINATIONS_API_KEY. Consigue una gratuita en "
                "https://enter.pollinations.ai/keys"
            )
        super().__init__(model_id=model_id, api_key=key)


def _is_mock_spec(spec: str) -> bool:
    s = (spec or "").strip()
    return s == "mock" or s.startswith(("mock:", "mock/"))


def parse_model_spec(spec: str) -> tuple:
    """Separa `spec` en (api_id, alias|None).

    Sintaxis:
      - 'opus=arquitecto' -> ('opus', 'arquitecto')
      - 'opus:auditor'    -> ('opus', 'auditor') (solo no-mock)
      - 'mock', 'mock:2', 'mock:3:llama' -> (spec, None) sin partir
      - 'groq/llama=rapido' -> ('groq/llama', 'rapido')
    El alias permite distinguir gemelos en el transcript sin contaminar
    el slug que se envía a la API.
    """
    s = (spec or "").strip()
    if not s:
        return s, None
    if _is_mock_spec(s):
        return s, None
    if "=" in s:
        api, alias = s.split("=", 1)
        api, alias = api.strip(), alias.strip()
        if api and alias:
            return api, alias
        return s, None
    if ":" in s:
        api, alias = s.rsplit(":", 1)
        api, alias = api.strip(), alias.strip()
        if api and alias and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_\-]*", alias):
            return api, alias
        return s, None
    return s, None


def strip_alias(spec: str) -> str:
    """Devuelve el slug real para la API, sin alias de sala."""
    api, _ = parse_model_spec(spec)
    return api


def participant_labels(specs: List[str]) -> List[str]:
    """Etiquetas únicas para el transcript: alias si hay, si no el api_id.

    Los duplicados se sufijan como 'opus (1)', 'opus (2)' para romper el
    efecto espejo (mismo nombre dos veces = confusión de identidad).
    Sin duplicados devuelve los nombres tal cual (compat total).
    """
    bases: List[str] = []
    for spec in specs or []:
        api, alias = parse_model_spec(spec)
        bases.append(alias if alias else api)
    from collections import Counter
    counts = Counter(bases)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for b in bases:
        if counts[b] < 2:
            out.append(b)
        else:
            seen[b] = seen.get(b, 0) + 1
            out.append(f"{b} ({seen[b]})")
    return out


def resolve_provider(model_identifier: str) -> BaseProvider:
    """Fábrica para instanciar el proveedor correcto según el identificador.

    Acepta specs con alias ('opus=arquitecto'): el alias se pela y solo
    el slug real viaja a la API. Formatos soportados:
      - 'mock', 'mock:2', 'mock:3:llama' -> MockProvider
      - 'groq/<model>'                  -> GroqProvider
      - 'openrouter/<model>'            -> OpenRouterProvider
      - '<model>' sin prefijo           -> PollinationsProvider (default sin keys)
    """
    model_identifier = strip_alias(model_identifier.strip())

    if model_identifier == "mock" or model_identifier.startswith(
        ("mock:", "mock/")
    ):
        return MockProvider(model_id=model_identifier)

    if model_identifier.startswith("groq/"):
        real_model = model_identifier.replace("groq/", "", 1)
        return GroqProvider(model_id=real_model)

    if model_identifier.startswith("openrouter/"):
        real_model = model_identifier.replace("openrouter/", "", 1)
        return OpenRouterProvider(model_id=real_model)

    return PollinationsProvider(model_id=model_identifier)


_CATALOG_URLS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "pollinations": "https://gen.pollinations.ai/v1/models",
}

_CATALOG_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "pollinations": "POLLINATIONS_API_KEY",
}

_CATALOG_CACHE: Dict[str, Any] = {}


def _split_provider(model_identifier: str) -> tuple:
    mid = strip_alias(model_identifier.strip())
    for prefix in ("groq/", "openrouter/"):
        if mid.startswith(prefix):
            return prefix[:-1], mid[len(prefix):]
    if mid == "mock" or mid.startswith(("mock:", "mock/")):
        return "mock", mid
    return "pollinations", mid


def _fetch_catalog(provider: str) -> Any:
    """Descarga el catálogo de modelos (cacheado). None si no se puede verificar."""
    if provider in _CATALOG_CACHE:
        return _CATALOG_CACHE[provider]
    ids = None
    try:
        headers: Dict[str, str] = {}
        key = os.getenv(_CATALOG_KEY_ENV[provider])
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = requests.get(_CATALOG_URLS[provider], headers=headers, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = {m.get("id") for m in items if isinstance(m, dict) and m.get("id")}
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        ids = None
    _CATALOG_CACHE[provider] = ids
    return ids


def check_model_availability(model_identifier: str) -> tuple:
    """Verificación suave: (True, None) si existe o no se puede comprobar.

    Solo devuelve (False, aviso) con 200 OK y slug ausente. Nunca bloquea.
    """
    provider, real_id = _split_provider(model_identifier)
    if provider == "mock":
        return True, None
    ids = _fetch_catalog(provider)
    if ids is None or real_id in ids:
        return True, None
    return False, (
        f"El modelo '{model_identifier}' no figura en el catálogo vivo de {provider}. "
        "Es posible que la petición falle si el slug no existe."
    )


def warn_unknown_models(model_ids: List[str]) -> None:
    """Imprime [AVISO] por cada modelo ausente del catálogo. Nunca falla."""
    try:
        for mid in model_ids:
            ok, msg = check_model_availability(mid)
            if not ok:
                print(f"[AVISO] {msg}")
    except Exception:
        pass
