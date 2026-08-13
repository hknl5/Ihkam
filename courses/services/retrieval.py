"""Topic or query → the passages it actually lives in (M3).

This is the retrieval half of Agent 1A. Everything downstream — the reference
passages a blueprint row carries (M4), the material a question is grounded in
(M5) — asks this module for content, and never reads a whole course again.

Three rules shape it:

* **The query is re-embedded, always.** A topic's stored chunks were embedded
  as page text; the topic itself is a name and a handful of terms. Reusing a
  chunk's vector would search for a passage rather than for the subject, so
  both entry points — free text and a ``Topic`` — go through ``embed()``.
* **An excluded topic's passages are filtered out before ranking.** The
  instructor's "not taught in lectures" is a hard rule (see
  ``TopicQuerySet.included``). Filtering after the top-k would let an excluded
  passage take a slot from an eligible one and silently shorten the result;
  filtering in the query means it was never a candidate. A passage with no
  topic stays eligible — unclaimed is not the same as ruled out.
* **Weak matches are cut, not padded.** Below ``min_score`` a passage is not a
  worse answer, it is a different subject. A topic the material barely covers
  should come back with two passages, or none — never with ``k`` of them
  because ``k`` was asked for.

Pure retrieval: the only model call is ``embed()``. Nothing here writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from ..models import Chunk, Course, Topic

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """The query could not be embedded, or came back the wrong shape."""


@dataclass(frozen=True)
class Passage:
    """One retrieved passage, with everything a citation needs.

    ``score`` is cosine similarity in ``[-1, 1]``: 1 is identical direction,
    0 unrelated. It is carried rather than recomputed so the debug view and
    the generator are looking at the same number the ranking used.
    """

    chunk_id: int
    text: str
    page: int
    source_file: str
    score: float
    topic: str | None = None
    #: Whether the underlying page text is an OCR transcription rather than a
    #: text layer. A quote from a model's reading of a picture is weaker
    #: evidence, and M5 is entitled to know before it grounds a question in it.
    from_ocr: bool = False

    @property
    def page_ref(self) -> str:
        return f"{self.source_file} · page {self.page}"

    @property
    def score_display(self) -> str:
        return f"{self.score:.3f}"


# --- Composing the query text ------------------------------------------------


def query_text_for(query: str | Topic) -> str:
    """The text that gets embedded, for either shape of query.

    A topic becomes its name plus its key terms. The terms matter: "Chapter 4"
    on its own embeds to almost nothing, while "Chapter 4 — truth tables,
    tautology, logical equivalence" embeds to the subject the instructor
    actually means. Definitions and formulas are deliberately left out — they
    are long, and they drag the query vector toward one passage of the topic
    rather than the topic as a whole.
    """
    if isinstance(query, Topic):
        terms = [str(t).strip() for t in (query.key_terms or []) if str(t).strip()]
        parts = [query.name.strip(), *terms]
        return " — ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]
    return (query or "").strip()


def embed_query(text: str, *, provider=None) -> list[float]:
    """Embed one query string, checked against the stored width.

    The same check ``chunk_source_file`` makes before writing: a vector of the
    wrong width would not error, it would rank nonsense, and the failure would
    only surface as bad passages much later.
    """
    from agents.provider import LLMError, get_provider

    try:
        provider = provider or get_provider()
    except LLMError as exc:
        raise RetrievalError(str(exc)) from exc

    try:
        vectors = provider.embed([text])
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller, never swallowed
        raise RetrievalError(f"The embedding call failed: {exc}") from exc

    if not vectors:
        raise RetrievalError("The provider returned no embedding for the query.")
    vector = list(vectors[0])
    if len(vector) != settings.EMBEDDING_DIM:
        raise RetrievalError(
            f"The provider returned a {len(vector)}-dimension query embedding, but this "
            f"database stores {settings.EMBEDDING_DIM}."
        )
    return vector


# --- Retrieval ---------------------------------------------------------------


def retrieve(
    course: Course,
    query: str | Topic,
    *,
    k: int | None = None,
    min_score: float | None = None,
    provider=None,
) -> list[Passage]:
    """The passages of ``course`` that are about ``query``, best first.

    ``query`` is either free text or a ``Topic`` (composed by
    ``query_text_for``). Returns at most ``k`` passages, and fewer — possibly
    none — when the rest score below ``min_score``.

    Scoped to one course by construction: another instructor's material is not
    a weaker match here, it is not a candidate at all.
    """
    k = settings.RETRIEVAL_TOP_K if k is None else k
    min_score = settings.RETRIEVAL_MIN_SCORE if min_score is None else min_score

    text = query_text_for(query)
    if not text or k <= 0:
        return []

    vector = embed_query(text, provider=provider)

    rows = (
        Chunk.objects.filter(source_file__course=course)
        .usable()  # excluded topics are gone before anything is ranked
        .select_related("topic", "source_file")
        .annotate(distance=CosineDistance("embedding", vector))
        .order_by("distance", "pk")[:k]
    )

    passages = [
        Passage(
            chunk_id=chunk.pk,
            text=chunk.text,
            page=chunk.page,
            source_file=chunk.source_file.original_name,
            score=1.0 - float(chunk.distance),
            topic=chunk.topic.name if chunk.topic else None,
            from_ocr=chunk.is_from_ocr,
        )
        for chunk in rows
    ]
    kept = [p for p in passages if p.score >= min_score]
    if len(kept) < len(passages):
        logger.debug(
            "Retrieval for %r on %s: %d of %d passages scored below %.2f and were dropped.",
            text[:60],
            course.code,
            len(passages) - len(kept),
            len(passages),
            min_score,
        )
    return kept
