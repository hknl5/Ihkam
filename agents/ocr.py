"""
The single seam between إحكام and any OCR engine — the same shape as the
``LLMProvider`` seam in ``provider.py``, and the same hard rule: no module
outside this file may import an OCR or vision SDK. Callers ask for
``get_ocr_provider()`` and use ``ocr_page()`` only.

OCR here is done by a vision-language model rather than a classic OCR engine.
The material is Arabic and English, often on the same page, and VLMs read
Arabic script far better than traditional engines — they resolve connected
forms and right-to-left reading order instead of emitting reversed or
disconnected glyphs.

Gemini is the phase-1 provider: it reuses the API key already configured for
``LLMProvider``, needs no GPU and no new infrastructure. ``PaddleVLProvider``
is the local phase-2 path, stubbed the way ``AirLLMProvider`` is.
"""

from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Raised when a provider is misconfigured or a page cannot be read."""


#: What a provider must return when a page holds no readable text at all.
#: An explicit sentinel beats an empty string, which is indistinguishable from
#: a failed call — and storing a failed call as "this page is blank" is exactly
#: the dishonesty the extraction work set out to remove.
NO_TEXT_SENTINEL = "[[NO_TEXT]]"


@dataclass
class OCRResult:
    """One page, as read by a model."""

    text: str
    #: True when the model reported no readable text on the page. The caller
    #: keeps its "unreadable" flag rather than storing an empty page as read.
    is_empty: bool = False
    provider: str = ""
    model: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return bool(self.text.strip()) and not self.is_empty


class OCRProvider(ABC):
    """Everything the rest of the system may ask an OCR engine to do."""

    name: str = "base"

    @abstractmethod
    def ocr_page(
        self,
        image: bytes,
        *,
        mime_type: str = "image/png",
        language_hint: str = "",
    ) -> OCRResult:
        """Transcribe one rendered page image.

        ``language_hint`` is the course's content language ("ar", "en",
        "mixed") — a hint only; a provider must still transcribe whatever is
        actually on the page.
        """


# --- The transcription prompt ------------------------------------------------

OCR_SYSTEM_PROMPT = """\
You transcribe a single page from university lecture material. You are a \
transcriber, not a reader or a summariser.

Rules:
- Output only the text that appears on the page. No preamble, no commentary, \
no markdown fences, no description of images.
- Transcribe every piece of text: titles, body, bullet points, labels inside \
diagrams and screenshots, table cells, captions, headers and footers, and the \
page or slide number if one is shown.
- Preserve reading order. Arabic reads right to left; keep Arabic text in its \
correct logical order with connected letterforms, exactly as written. Latin \
words, numbers, code and formulas embedded in Arabic text stay as they are.
- Keep the page's line and paragraph structure. Keep bullet markers as "- ".
- Preserve tables as plain rows, one row per line, cells separated by " | ".
- Do not translate. Do not correct spelling. Do not fill in anything that is \
cut off or illegible — write nothing for it.
- If the page contains no readable text at all (a photograph, a blank page, \
pure decoration), output exactly: {sentinel}
""".format(sentinel=NO_TEXT_SENTINEL)


def _user_prompt(language_hint: str) -> str:
    hints = {
        "ar": "This course's material is mainly in Arabic.",
        "en": "This course's material is mainly in English.",
        "mixed": "This course's material mixes Arabic and English, often on the same page.",
    }
    hint = hints.get(language_hint, "")
    return f"Transcribe this page. {hint}".strip()


#: How long to wait when a provider says "too many requests". The server's own
#: retryDelay is preferred; otherwise back off exponentially.
_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?[:=]\s*['\"]?(\d+(?:\.\d+)?)")
_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota")
#: A *daily* quota does not come back in a minute. Waiting for one just makes
#: an upload hang for the full retry budget and still fail — measured: 12
#: minutes of waiting on an exhausted free-tier key.
_DAILY_QUOTA_MARKERS = ("PerDay", "per day", "PerProjectPerDay")
MAX_RETRY_DELAY_SECONDS = 75.0


class OCRQuotaExhausted(OCRError):
    """The provider's quota is spent for the day — retrying will not help."""


def _rate_limit_delay(exc: Exception, attempt: int = 1) -> float | None:
    """Seconds to wait before retrying, or None if waiting cannot help.

    A per-minute rate limit is worth waiting out: a free-tier key allows only
    a few requests a minute, and giving up would drop pages that are readable
    a moment later. A per-day quota is not — that is reported immediately.
    """
    message = str(exc)
    if not any(marker in message for marker in _RATE_LIMIT_MARKERS):
        return None
    if any(marker in message for marker in _DAILY_QUOTA_MARKERS):
        return None
    found = _RETRY_DELAY_PATTERN.search(message)
    if found:
        # A second of slack, so we do not come back a hair too early.
        return min(float(found.group(1)) + 1.0, MAX_RETRY_DELAY_SECONDS)
    return min(2.0**attempt + random.uniform(0, 1), MAX_RETRY_DELAY_SECONDS)


def _is_daily_quota(exc: Exception) -> bool:
    message = str(exc)
    return any(m in message for m in _RATE_LIMIT_MARKERS) and any(
        m in message for m in _DAILY_QUOTA_MARKERS
    )


# --- Gemini (phase 1 default) ------------------------------------------------


class GeminiOCRProvider(OCRProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.GEMINI_OCR_MODEL

        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise OCRError("GEMINI_API_KEY is not set — add it to your .env file.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise OCRError("The `google-genai` package is not installed.") from exc
        self._types = types
        self._client = genai.Client(api_key=key)

    def ocr_page(
        self,
        image: bytes,
        *,
        mime_type: str = "image/png",
        language_hint: str = "",
    ) -> OCRResult:
        config = self._types.GenerateContentConfig(
            system_instruction=OCR_SYSTEM_PROMPT,
            # Transcription is not a creative task: take the likeliest reading.
            temperature=0.0,
        )
        contents = [
            self._types.Part.from_bytes(data=image, mime_type=mime_type),
            _user_prompt(language_hint),
        ]
        attempts = max(1, getattr(settings, "OCR_MAX_ATTEMPTS", 5))
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except Exception as exc:  # noqa: BLE001 — SDK raises many shapes
                if _is_daily_quota(exc):
                    raise OCRQuotaExhausted(
                        "The Gemini daily free-tier quota is used up. OCR will work "
                        "again when it resets, or immediately on a billed key."
                    ) from exc
                delay = _rate_limit_delay(exc, attempt)
                if delay is None or attempt == attempts:
                    raise OCRError(f"Gemini could not read this page: {exc}") from exc
                # Free-tier keys allow only a handful of requests per minute.
                # Waiting is the difference between a read page and a lost one.
                logger.info(
                    "OCR rate-limited, waiting %.0fs (attempt %s/%s)", delay, attempt, attempts
                )
                time.sleep(delay)

        text = (response.text or "").strip()
        return OCRResult(
            text="" if text == NO_TEXT_SENTINEL else text,
            is_empty=(not text) or text == NO_TEXT_SENTINEL,
            provider=self.name,
            model=self.model,
            raw={"model": self.model},
        )


# --- PaddleOCR-VL (phase 2, local) -------------------------------------------


class PaddleVLProvider(OCRProvider):
    """TODO (phase 2, local): PaddleOCR-VL served locally.

    Deliberately not implemented yet — the same deferral as ``AirLLMProvider``.
    When the hosted version is proven and a GPU is available:

    - Use the **full server model** (PaddleOCR-VL), Apache-2.0. Do **not** use
      the mobile/lite variant: its Arabic accuracy is the reason we would be
      switching in the first place.
    - Serve it locally and implement ``ocr_page()`` against it, honouring the
      same contract: page image in, transcription out, ``is_empty`` set (and
      ``NO_TEXT_SENTINEL`` semantics) when the page holds no readable text.
    - Set ``OCR_PROVIDER=paddlevl`` in .env. Nothing else should change.
    - Re-run the extraction checks on the sample courses to confirm Arabic
      quality holds locally before trusting it.
    """

    name = "paddlevl"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "PaddleVLProvider is a phase-2 stub. Use OCR_PROVIDER=gemini."
        )

    def ocr_page(
        self,
        image: bytes,
        *,
        mime_type: str = "image/png",
        language_hint: str = "",
    ) -> OCRResult:
        raise NotImplementedError


# --- Factory -----------------------------------------------------------------

OCR_PROVIDERS: dict[str, type[OCRProvider]] = {
    GeminiOCRProvider.name: GeminiOCRProvider,
    PaddleVLProvider.name: PaddleVLProvider,
}


def get_ocr_provider(name: str | None = None) -> OCRProvider:
    """Return the configured OCR provider instance.

    Reads ``settings.OCR_PROVIDER`` unless a name is passed explicitly (the
    ``ocr_probe`` command uses that to test one without editing .env).
    """
    key = (name or settings.OCR_PROVIDER or "").strip().lower()
    if key not in OCR_PROVIDERS:
        raise OCRError(
            f"Unknown OCR_PROVIDER {key!r}. Choose one of: {', '.join(OCR_PROVIDERS)}."
        )
    return _build_ocr_provider(key)


@lru_cache(maxsize=None)
def _build_ocr_provider(key: str) -> OCRProvider:
    return OCR_PROVIDERS[key]()
