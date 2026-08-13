"""Course text → a topic list the instructor then confirms (M2).

The model reads only the pages extraction could actually read, returns JSON,
and that JSON is validated twice before a row is written:

1. **Shape**, by Pydantic — the §3 rule: unvalidated model output never reaches
   the database. A malformed answer is retried once, then surfaced as an error.
2. **Truth about the material**, here — a file name must be a file this course
   really has, and a page span must lie inside the pages that were actually
   sent. A hallucinated citation is the failure this system exists to prevent,
   so a span that cannot be verified is dropped rather than stored.

Nothing produced here is trusted: it lands as `Topic` rows the instructor
renames, merges, deletes, adds to, and excludes on the review screen. That step
is the product, not a formality.

No model SDK is imported here — every call goes through ``get_provider()``.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from django.conf import settings
from django.db import transaction
from pydantic import BaseModel, Field, ValidationError

from ..models import Course, ExtractedPage, Topic

logger = logging.getLogger(__name__)


class TopicExtractionError(RuntimeError):
    """The model's answer could not be validated, twice. Nothing was stored."""


# --- The validated shape -----------------------------------------------------


class DefinitionOut(BaseModel):
    term: str = ""
    text: str = ""


class SubtopicOut(BaseModel):
    name: str
    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    key_terms: list[str] = Field(default_factory=list)
    definitions: list[DefinitionOut] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class ChapterOut(SubtopicOut):
    subtopics: list[SubtopicOut] = Field(default_factory=list)


class ExtractionOut(BaseModel):
    chapters: list[ChapterOut] = Field(default_factory=list)


# --- The document handed to the model ---------------------------------------


@dataclass
class CourseDocument:
    """The readable part of a course, as one string, plus what was left out."""

    text: str
    #: file id → the page numbers actually included, for verifying spans.
    pages_by_file: dict[int, set[int]] = field(default_factory=dict)
    #: original_name → file id, for resolving what the model names.
    files_by_name: dict[str, int] = field(default_factory=dict)
    pages_included: int = 0
    pages_skipped: int = 0
    pages_from_ocr: int = 0
    #: Pages dropped because the course text hit `TOPIC_EXTRACTION_MAX_CHARS`.
    pages_over_budget: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def build_course_document(course: Course) -> CourseDocument:
    """Assemble the course's readable pages into one prompt-ready document.

    Only readable pages are included — a page with no text layer that OCR could
    not recover contributes nothing, because it *is* nothing we can read, and
    letting its leftover slide number into the prompt would invite a topic
    invented out of a page number.

    OCR pages are included and labelled. Their text is a transcription of a
    picture, which is weaker evidence than a text layer, and the prompt says so
    rather than presenting the two as equivalent.
    """
    budget = getattr(settings, "TOPIC_EXTRACTION_MAX_CHARS", 150_000)
    document = CourseDocument(text="")
    parts: list[str] = []
    used = 0

    for source_file in course.files.all().order_by("uploaded_at"):
        pages = list(source_file.pages.all())
        readable = [p for p in pages if p.is_readable]
        document.pages_skipped += len(pages) - len(readable)
        if not readable:
            continue
        document.files_by_name[source_file.original_name] = source_file.pk
        included: set[int] = set()
        header = f"\n=== file: {source_file.original_name} ===\n"
        parts.append(header)
        used += len(header)
        for page in readable:
            label = " (OCR transcription)" if page.is_from_ocr else ""
            block = f"\n[page {page.number}{label}]\n{page.text.strip()}\n"
            if used + len(block) > budget:
                document.pages_over_budget += 1
                continue
            parts.append(block)
            used += len(block)
            included.add(page.number)
            document.pages_included += 1
            if page.is_from_ocr:
                document.pages_from_ocr += 1
        if included:
            document.pages_by_file[source_file.pk] = included

    document.text = "".join(parts).strip()
    return document


# --- Calling the model ------------------------------------------------------

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _loads(text: str) -> dict:
    """Parse a JSON answer, tolerating a code fence around it.

    `json_mode` is asked for on every call, but a fenced answer is a common
    enough provider quirk that failing the whole extraction over three
    backticks would be silly. Anything else malformed is a real failure.
    """
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def call_extraction(document: CourseDocument, course: Course, *, provider=None) -> ExtractionOut:
    """One completion, validated. Retried exactly once, then raised.

    The retry exists because a model occasionally returns prose around its JSON
    or drops a required field; asking again usually fixes it. It is bounded at
    one because a second failure is a signal about the prompt or the material,
    not bad luck, and silently looping would hide that.
    """
    from agents.prompts.topics import SYSTEM, build_user_prompt
    from agents.provider import LLMError, get_provider  # the seam; see §3

    try:
        provider = provider or get_provider()
    except LLMError as exc:
        raise TopicExtractionError(str(exc)) from exc

    user = build_user_prompt(course.name, course.content_language, document.text)
    last_error = ""
    for attempt in (1, 2):
        try:
            response = provider.complete(SYSTEM, user, json_mode=True, temperature=0.1)
            return ExtractionOut.model_validate(_loads(response.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            # The only thing worth asking twice for. A model that returned prose
            # around its JSON, or dropped a field, usually gets it right again.
            last_error = f"the answer was not the expected JSON ({exc.__class__.__name__})"
            logger.warning("Topic extraction attempt %s returned bad JSON: %s", attempt, exc)
        except Exception as exc:  # noqa: BLE001 — surfaced as itself, never stored
            # A call that never reached the model — no key, no credit, rate
            # limit, network — is not a malformed answer, and reporting it as
            # one sends the instructor looking in the wrong place. It is also
            # not worth a second call, which would fail identically.
            raise TopicExtractionError(
                f"The extraction call did not complete: {exc}"
            ) from exc

    raise TopicExtractionError(
        f"The model's answer could not be read after two attempts: {last_error}. "
        "Nothing was stored — no topic list is better than an unchecked one."
    )


# --- Verifying what came back against the material --------------------------


def _normalize_name(name: str) -> str:
    """For duplicate detection only. Never stored — the instructor sees the
    material's own wording, spacing and punctuation included."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name)).strip().casefold()


def _resolve_file(name: str | None, document: CourseDocument) -> int | None:
    """The file the model named, or None if it named one this course has not
    got. A made-up file name loses the citation; it never invents a link."""
    if not name:
        return None
    if name in document.files_by_name:
        return document.files_by_name[name]
    target = _normalize_name(name)
    for original, pk in document.files_by_name.items():
        if _normalize_name(original) == target:
            return pk
    return None


def _resolve_span(
    file_pk: int | None, start: int | None, end: int | None, document: CourseDocument
) -> tuple[int | None, int | None]:
    """Keep a page span only where it overlaps pages the model was shown.

    The span is intersected with those pages, not clamped to the file's length:
    the model cannot have found a topic on a page it never saw, so a span
    reaching outside is narrowed to the part that is real, and dropped entirely
    if none of it is.
    """
    if file_pk is None or start is None:
        return None, None
    pages = document.pages_by_file.get(file_pk)
    if not pages:
        return None, None
    end = end if end is not None and end >= start else start
    covered = [n for n in pages if start <= n <= end]
    if not covered:
        return None, None
    return min(covered), max(covered)


def _clean_strings(values) -> list[str]:
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def _clean_definitions(values) -> list[dict]:
    return [
        {"term": v.term.strip(), "text": v.text.strip()}
        for v in values
        if (v.term or "").strip() or (v.text or "").strip()
    ]


# --- Storing -----------------------------------------------------------------


@dataclass
class ExtractionRun:
    """What one extraction pass produced, in the instructor's terms."""

    chapters: int = 0
    subtopics: int = 0
    duplicates_dropped: int = 0
    spans_dropped: int = 0
    pages_included: int = 0
    pages_skipped: int = 0
    pages_from_ocr: int = 0
    pages_over_budget: int = 0

    @property
    def total(self) -> int:
        return self.chapters + self.subtopics


def extract_topics(course: Course, *, provider=None, replace: bool = True) -> ExtractionRun:
    """Extract this course's topics and store them as `Topic` rows.

    With ``replace`` (the default) the course's existing topics are removed
    first, because a second extraction re-reads the same material and merging
    two model opinions would produce a list neither of them meant. The screen
    that offers this says so before it runs: an instructor's edits are theirs,
    and losing them silently would break the one promise this milestone makes.

    Raises `TopicExtractionError` without writing anything if the material is
    unreadable or the answer cannot be validated.
    """
    document = build_course_document(course)
    if document.is_empty:
        raise TopicExtractionError(
            "There is no readable text in this course's material yet. Upload a file "
            "extraction can read, or check the pages flagged as unread."
        )

    result = call_extraction(document, course, provider=provider)

    run = ExtractionRun(
        pages_included=document.pages_included,
        pages_skipped=document.pages_skipped,
        pages_from_ocr=document.pages_from_ocr,
        pages_over_budget=document.pages_over_budget,
    )
    seen: set[tuple[int | None, str]] = set()
    position = 0

    with transaction.atomic():
        if replace:
            course.topics.all().delete()
        else:
            position = (course.topics.count() or 0) * 10
            seen = {
                (t.parent_id, _normalize_name(t.name))
                for t in course.topics.all()
            }

        for chapter in result.chapters:
            name = (chapter.name or "").strip()
            if not name:
                continue
            key = (None, _normalize_name(name))
            if key in seen:
                run.duplicates_dropped += 1
                continue
            seen.add(key)
            stored = _store(course, chapter, None, position, document, run)
            position += 1
            run.chapters += 1

            for sub in chapter.subtopics:
                sub_name = (sub.name or "").strip()
                if not sub_name:
                    continue
                sub_key = (stored.pk, _normalize_name(sub_name))
                if sub_key in seen:
                    run.duplicates_dropped += 1
                    continue
                seen.add(sub_key)
                _store(course, sub, stored, position, document, run)
                position += 1
                run.subtopics += 1

    return run


def _store(
    course: Course,
    item: SubtopicOut,
    parent: Topic | None,
    position: int,
    document: CourseDocument,
    run: ExtractionRun,
) -> Topic:
    file_pk = _resolve_file(item.source_file, document)
    start, end = _resolve_span(file_pk, item.page_start, item.page_end, document)
    if item.page_start is not None and start is None:
        run.spans_dropped += 1
    return Topic.objects.create(
        course=course,
        parent=parent,
        name=item.name.strip()[: Topic._meta.get_field("name").max_length],
        # The file link survives a dropped span: the model naming a real file
        # is still true, it just could not be pinned to pages.
        source_file_id=file_pk,
        page_start=start,
        page_end=end,
        key_terms=_clean_strings(item.key_terms),
        definitions=_clean_definitions(item.definitions),
        formulas=_clean_strings(item.formulas),
        examples=_clean_strings(item.examples),
        position=position,
    )


# --- Instructor edits --------------------------------------------------------


def merge_topics(keep: Topic, absorb: Topic) -> Topic:
    """Fold `absorb` into `keep` and delete it.

    The instructor said these are one topic, so nothing either of them knew is
    thrown away: the page span becomes the span covering both, detail lists are
    concatenated without duplicates, sub-topics and chunks are re-pointed. The
    surviving name is `keep`'s — they can rename it afterwards, and guessing a
    combined name for them would be presumptuous.

    A page span is only combined when both topics cite the *same* file; two
    spans from different files cannot be expressed as one range, so the
    surviving topic keeps its own rather than claiming pages it never had.
    """
    if keep.pk == absorb.pk or keep.course_id != absorb.course_id:
        raise ValueError("Two different topics of the same course are needed to merge.")

    if keep.source_file_id and keep.source_file_id == absorb.source_file_id:
        starts = [p for p in (keep.page_start, absorb.page_start) if p is not None]
        ends = [p for p in (keep.page_end or keep.page_start, absorb.page_end or absorb.page_start) if p is not None]
        if starts:
            keep.page_start = min(starts)
            keep.page_end = max(ends) if ends else None
    elif keep.source_file_id is None and absorb.source_file_id is not None:
        keep.source_file_id = absorb.source_file_id
        keep.page_start, keep.page_end = absorb.page_start, absorb.page_end

    keep.key_terms = _merge_lists(keep.key_terms, absorb.key_terms)
    keep.formulas = _merge_lists(keep.formulas, absorb.formulas)
    keep.examples = _merge_lists(keep.examples, absorb.examples)
    keep.definitions = _merge_definitions(keep.definitions, absorb.definitions)

    with transaction.atomic():
        keep.save()
        # A merged-away chapter's sub-topics move under the survivor; they are
        # part of what the instructor said belongs together.
        absorb.subtopics.update(parent=keep)
        absorb.chunks.update(topic=keep)
        absorb.delete()
    return keep


def _merge_lists(first, second) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in list(first or []) + list(second or []):
        key = _normalize_name(str(value))
        if key and key not in seen:
            seen.add(key)
            merged.append(value)
    return merged


def _merge_definitions(first, second) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for value in list(first or []) + list(second or []):
        if not isinstance(value, dict):
            continue
        key = _normalize_name(f"{value.get('term', '')} {value.get('text', '')}")
        if key and key not in seen:
            seen.add(key)
            merged.append(value)
    return merged


def delete_topic(topic: Topic) -> int:
    """Delete one topic, promoting its sub-topics rather than taking them too.

    Removing a heading is not a request to lose everything filed under it. The
    sub-topics become chapters in place; the instructor can re-file or delete
    them individually, which is a decision only they can make.
    """
    promoted = topic.subtopics.count()
    with transaction.atomic():
        topic.subtopics.update(parent=None)
        topic.delete()
    return promoted


def readable_page_count(course: Course) -> int:
    """How many pages of this course's material can actually be read."""
    return sum(
        1
        for page in ExtractedPage.objects.filter(source_file__course=course)
        if page.is_readable
    )


def unreadable_page_count(course: Course) -> int:
    """The counterpart: pages flagged as unread, which contribute nothing."""
    return sum(
        1
        for page in ExtractedPage.objects.filter(source_file__course=course)
        if not page.is_readable
    )


__all__ = [
    "CourseDocument",
    "ExtractionOut",
    "ExtractionRun",
    "TopicExtractionError",
    "build_course_document",
    "call_extraction",
    "delete_topic",
    "extract_topics",
    "merge_topics",
    "readable_page_count",
    "unreadable_page_count",
]
