"""Adaptadores de proveedores de LLM para ParallelAgent.

Todos los proveedores implementan la interfaz común BaseProvider.
"""

from abc import ABC, abstractmethod
import os
import re
from typing import Any, Dict, List
import requests


class ProviderError(Exception):
    """Excepción base para fallos de red, autenticación o formato en proveedores."""

    pass


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


def resolve_provider(model_identifier: str) -> BaseProvider:
    """Fábrica para instanciar el proveedor correcto según el identificador.

    Formatos soportados:
      - 'mock', 'mock:2', 'mock:3:llama' -> MockProvider
      - 'groq/<model>'                  -> GroqProvider
      - 'openrouter/<model>'            -> OpenRouterProvider
      - '<model>' sin prefijo           -> PollinationsProvider (default sin keys)
    """
    model_identifier = model_identifier.strip()

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
