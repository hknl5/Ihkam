"""Readable page text → embedded passages (M2).

This is the foundation retrieval stands on in M3: Agent 1A never reads a whole
course, it reads the handful of passages a topic actually lives in. Chunks are
built here so they exist before anything needs them.

Two rules shape the whole module:

* **A passage never crosses a page boundary.** Every citation the system will
  make ("source: page 14") is only as good as the page number on the passage it
  quotes, and a passage spanning two pages has no honest answer.
* **Only readable pages contribute.** A page flagged as unread — no text layer,
  and OCR could not recover it — is not chunked. Its content is not missing
  from the index by accident; it was never readable in the first place, and
  quietly indexing the slide number left on it would hide that.

Embeddings go through ``LLMProvider.embed()`` — the same seam ``llm_ping``
uses — so the phase-2 switch to a local model covers retrieval too.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from django.conf import settings
from django.db import transaction

from ..models import Chunk, Course, ExtractedPage, SourceFile

logger = logging.getLogger(__name__)


class ChunkingError(RuntimeError):
    """Embeddings could not be produced, or came back the wrong shape."""


# --- Splitting (pure text, no model involved) --------------------------------

_PARAGRAPH = re.compile(r"\n\s*\n")
#: Sentence-ish boundaries for a paragraph too long to be one passage. Arabic
#: full stop and question mark are included — this material is bilingual.
_SENTENCE = re.compile(r"(?<=[.!?؟۔])\s+|\n")


def split_page(text: str, *, max_chars: int | None = None, min_chars: int | None = None) -> list[str]:
    """Split one page's text into passages, in reading order.

    Paragraphs are packed greedily up to ``max_chars``. A paragraph that would
    leave a passage shorter than ``min_chars`` is joined to the next one
    instead: a passage that is one heading embeds to nothing useful, and would
    only ever match itself.

    A single paragraph longer than ``max_chars`` is split on sentence ends, and
    then, if one "sentence" is still too long (an unpunctuated wall of text),
    on a hard character boundary — never dropped.
    """
    max_chars = max_chars or getattr(settings, "CHUNK_MAX_CHARS", 1200)
    min_chars = min_chars or getattr(settings, "CHUNK_MIN_CHARS", 200)

    paragraphs = [p.strip() for p in _PARAGRAPH.split(text or "") if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_hard_split(paragraph, max_chars) if len(paragraph) > max_chars else [paragraph])

    passages: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) + 2 <= max_chars or len(current) < min_chars:
            current = f"{current}\n\n{piece}"
        else:
            passages.append(current)
            current = piece
    if current:
        # A trailing scrap is appended to the last passage rather than kept as
        # a chunk of its own — unless it is all there is.
        if passages and len(current) < min_chars:
            passages[-1] = f"{passages[-1]}\n\n{current}"
        else:
            passages.append(current)
    return passages


def _hard_split(paragraph: str, max_chars: int) -> list[str]:
    """One over-long paragraph → sentence-sized pieces, nothing discarded."""
    sentences = [s.strip() for s in _SENTENCE.split(paragraph) if s and s.strip()]
    pieces: list[str] = []
    for sentence in sentences or [paragraph]:
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if sentence:
            pieces.append(sentence)
    return pieces


# --- Building chunks ---------------------------------------------------------


@dataclass
class ChunkRun:
    """What one chunking pass over a file did."""

    pages_chunked: int = 0
    pages_skipped: int = 0
    chunks: int = 0
    #: chunk source → count, so the file can say how much of its index rests on
    #: a transcription rather than a text layer.
    by_source: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def embed_texts(texts: list[str], provider=None) -> list[list[float]]:
    """Embed passages through the provider seam, in batches.

    The width is checked against ``EMBEDDING_DIM`` before anything is stored:
    a provider quietly returning its native size instead of the configured one
    would make every stored vector incomparable with the next file's, and the
    failure would not show up until retrieval returned nonsense in M3.
    """
    if not texts:
        return []

    from agents.provider import LLMError, get_provider  # the seam; see §3

    try:
        provider = provider or get_provider()
    except LLMError as exc:
        raise ChunkingError(str(exc)) from exc

    expected = settings.EMBEDDING_DIM
    batch_size = max(1, getattr(settings, "EMBED_BATCH_SIZE", 64))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            result = provider.embed(batch)
        except Exception as exc:  # noqa: BLE001 — surfaced, never half-stored
            raise ChunkingError(f"The embedding call failed: {exc}") from exc
        if len(result) != len(batch):
            raise ChunkingError(
                f"The provider returned {len(result)} embeddings for {len(batch)} passages."
            )
        for vector in result:
            if len(vector) != expected:
                raise ChunkingError(
                    f"The provider returned {len(vector)}-dimension embeddings, but this "
                    f"database stores {expected}. Check EMBEDDING_DIM and the embedding model."
                )
        vectors.extend(result)
    return vectors


def chunk_source_file(source_file: SourceFile, *, provider=None) -> ChunkRun:
    """Rebuild every chunk for one file from its readable pages.

    Replaces rather than adds: re-running after a re-extraction must not leave
    passages behind that no page says any more. Nothing is written unless every
    embedding came back at the right width — a half-embedded file is worse than
    an unembedded one, because it looks finished.
    """
    pages = list(source_file.pages.all())
    readable = [p for p in pages if p.is_readable]

    passages: list[tuple[ExtractedPage, int, str]] = []
    for page in readable:
        for position, text in enumerate(split_page(page.text)):
            passages.append((page, position, text))

    if not passages:
        with transaction.atomic():
            source_file.chunks.all().delete()
        return ChunkRun(pages_skipped=len(pages) - len(readable))

    try:
        vectors = embed_texts([text for _, _, text in passages], provider=provider)
    except ChunkingError as exc:
        logger.warning("Chunking %s failed: %s", source_file.original_name, exc)
        return ChunkRun(pages_skipped=len(pages) - len(readable), error=str(exc))

    by_source: dict[str, int] = {}
    with transaction.atomic():
        source_file.chunks.all().delete()
        Chunk.objects.bulk_create(
            Chunk(
                source_file=source_file,
                page=page.number,
                position=position,
                text=text,
                embedding=vector,
                source=page.source,
            )
            for (page, position, text), vector in zip(passages, vectors, strict=True)
        )
    for page, _, _ in passages:
        by_source[page.source] = by_source.get(page.source, 0) + 1

    return ChunkRun(
        pages_chunked=len({page.number for page, _, _ in passages}),
        pages_skipped=len(pages) - len(readable),
        chunks=len(passages),
        by_source=by_source,
    )


def chunk_course(course: Course, *, provider=None) -> ChunkRun:
    """Chunk every file of a course, reporting one total run."""
    total = ChunkRun()
    errors = []
    for source_file in course.files.all():
        run = chunk_source_file(source_file, provider=provider)
        total.pages_chunked += run.pages_chunked
        total.pages_skipped += run.pages_skipped
        total.chunks += run.chunks
        for key, count in run.by_source.items():
            total.by_source[key] = total.by_source.get(key, 0) + count
        if run.error:
            errors.append(f"{source_file.original_name}: {run.error}")
    total.error = " ".join(errors)
    return total


# --- Linking chunks to topics ------------------------------------------------


def link_chunks_to_topics(course: Course) -> int:
    """Attach each chunk to the topic whose page span covers its page.

    Deterministic, no model involved: a topic states the pages it was found on,
    and a passage on one of those pages belongs to it. The most specific match
    wins — a sub-topic's narrow span beats its chapter's wide one — because the
    narrower claim is the more informative one.

    A passage no topic claims keeps ``topic = None``. That is not a failure;
    it is a passage the syllabus does not name, and it stays retrievable.

    Returns how many chunks ended up attached.
    """
    spans = [
        topic
        for topic in course.topics.exclude(source_file__isnull=True).exclude(page_start__isnull=True)
    ]
    # Narrowest span first, so the first match is the most specific one. A
    # sub-topic wins a tie with its own chapter: when both claim the same page,
    # the sub-topic is the more informative of two true answers.
    spans.sort(
        key=lambda t: (
            (t.page_end or t.page_start) - t.page_start,
            0 if t.parent_id else 1,
            t.position,
        )
    )

    attached = 0
    updates = []
    for chunk in Chunk.objects.filter(source_file__course=course).select_related("source_file"):
        match = next(
            (
                topic
                for topic in spans
                if topic.source_file_id == chunk.source_file_id
                and topic.page_start <= chunk.page <= (topic.page_end or topic.page_start)
            ),
            None,
        )
        if chunk.topic_id != (match.pk if match else None):
            chunk.topic = match
            updates.append(chunk)
        if match:
            attached += 1
    if updates:
        Chunk.objects.bulk_update(updates, ["topic"])
    return attached
