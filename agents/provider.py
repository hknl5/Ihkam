"""
The single seam between إحكام and any language model (§3 of the build plan).

Hard rule: no module outside this file may import an OpenAI / Gemini / AirLLM
SDK. Agents call ``get_provider()`` and talk to the ``LLMProvider`` interface
only. That is what makes the phase-2 switch to a local model a one-class change.

All provider SDKs are imported lazily, inside the concrete class, so the app
runs (and the other provider works) even if one SDK is absent.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

from django.conf import settings


class LLMError(RuntimeError):
    """Raised when a provider is misconfigured or a call cannot be completed."""


@dataclass
class LLMResponse:
    text: str
    raw: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Everything the rest of the system is allowed to ask a model to do."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """One completion. With ``json_mode`` the text is a JSON document."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each input string. Returns one vector per input, in order."""


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm:
        return vector
    return [v / norm for v in vector]


# --- OpenAI (phase 1 default) ----------------------------------------------


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.model = model or settings.OPENAI_MODEL
        self.embed_model = embed_model or settings.OPENAI_EMBED_MODEL
        self.embedding_dim = embedding_dim or settings.EMBEDDING_DIM

        key = api_key or settings.OPENAI_API_KEY
        if not key:
            raise LLMError("OPENAI_API_KEY is not set — add it to your .env file.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMError("The `openai` package is not installed.") from exc
        self._client = OpenAI(api_key=key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        text = (completion.choices[0].message.content or "").strip()
        return LLMResponse(text=text, raw=completion.model_dump())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embeddings.create(
            model=self.embed_model,
            input=texts,
            dimensions=self.embedding_dim,
        )
        ordered = sorted(result.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


# --- Gemini (alternate phase 1) ---------------------------------------------


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.model = model or settings.GEMINI_MODEL
        self.embed_model = embed_model or settings.GEMINI_EMBED_MODEL
        self.embedding_dim = embedding_dim or settings.EMBEDDING_DIM

        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY is not set — add it to your .env file.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMError("The `google-genai` package is not installed.") from exc
        self._types = types
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        return LLMResponse(text=(response.text or "").strip(), raw=response.model_dump())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.models.embed_content(
            model=self.embed_model,
            contents=texts,
            config=self._types.EmbedContentConfig(
                output_dimensionality=self.embedding_dim
            ),
        )
        # gemini-embedding-001 only returns unit-length vectors at its native
        # width; truncated outputs must be re-normalized before use.
        return [_l2_normalize(list(item.values)) for item in response.embeddings]


# --- AirLLM (phase 2, local) -------------------------------------------------


class AirLLMProvider(LLMProvider):
    """TODO (Stage 5, §8): local model served through AirLLM.

    Deliberately not implemented yet. When the API-key version is proven, fill
    in ``complete()`` and ``embed()`` against the local model, set
    ``LLM_PROVIDER=airllm`` in .env, and re-run the M2/M5/M7/M10 validation
    checks. No other code in the project should need to change.
    """

    name = "airllm"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AirLLMProvider is a phase-2 stub. Use LLM_PROVIDER=openai or gemini."
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


# --- Factory -----------------------------------------------------------------

PROVIDERS: dict[str, type[LLMProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    GeminiProvider.name: GeminiProvider,
    AirLLMProvider.name: AirLLMProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the configured provider instance.

    Reads ``settings.LLM_PROVIDER`` unless a name is passed explicitly (the
    ``llm_ping`` command uses that to test one provider without editing .env).
    Instances are cached per name so SDK clients are built once.
    """
    key = (name or settings.LLM_PROVIDER or "").strip().lower()
    if key not in PROVIDERS:
        raise LLMError(
            f"Unknown LLM_PROVIDER {key!r}. Choose one of: {', '.join(PROVIDERS)}."
        )
    return _build_provider(key)


@lru_cache(maxsize=None)
def _build_provider(key: str) -> LLMProvider:
    return PROVIDERS[key]()
