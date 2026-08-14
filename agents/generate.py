"""Agent 2A — writing questions, strictly from the retrieved passages (M5).

Agent 1A finished by saying *which passages each planned question will be
written from*. This module writes them: one blueprint item at a time, in, one
batch of candidates out, validated before anything is stored.

Four decisions shape it:

* **A candidate that cannot name a supplied passage is dropped.** `source_ref`
  is resolved against the passages the model was actually given, the same way
  M2 resolves a topic's page span against the pages it was actually shown. A
  citation that resolves to nothing is not a formatting slip — it is the mark of
  a question written from the subject rather than from the course, which is the
  one failure this milestone exists to prevent. Dropped candidates are counted
  and reported, never silently discarded.
* **Over-generation, not exactness.** A row asking for N questions is generated
  as `ceil(N * 1.5)` candidates, so the instructor always has an alternative to
  the one they reject and M7's review loop has somewhere to go. Even N=1 gets 2.
* **The answer comes back with the question.** `correct` and `explanation` are
  part of the same validated object as the stem. M6 extends their shape by
  type; it must never be a second call, because an answer key produced by a
  later call is a key for a question the model has to re-read rather than one it
  wrote.
* **Retry once, and only for a malformed answer.** A call that never reached the
  model — no key, no credit, rate limit — is reported as itself. The M2 rule.

No model SDK is imported here; the call goes through ``get_provider()``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal

from django.db import transaction
from pydantic import BaseModel, Field, ValidationError, model_validator

from courses.services.retrieval import Passage
from exams.models import BlueprintRow, Question

logger = logging.getLogger(__name__)


class QuestionGenerationError(RuntimeError):
    """Generation could not produce trustworthy candidates. Nothing was stored."""


class UnsupportedQuestionType(QuestionGenerationError):
    """The blueprint asks for a type the MVP does not write (see `MVP_TYPES`)."""


# --- What the MVP writes -----------------------------------------------------

#: MCQ, true/false, short answer, and text-only numeric problems. Diagrams,
#: symbolic maths, code analysis, matching and image-based questions are
#: deferred: each needs a way to carry something that is not a paragraph of
#: text, and pretending otherwise produces a question whose figure does not
#: exist. A row asking for one is refused by name, not quietly turned into an
#: MCQ.
MVP_TYPES: frozenset[str] = frozenset(
    {
        BlueprintRow.QuestionType.MCQ,
        BlueprintRow.QuestionType.TRUE_FALSE,
        BlueprintRow.QuestionType.SHORT_ANSWER,
        BlueprintRow.QuestionType.NUMERIC,
    }
)

#: How many candidates one asked-for question is worth. 1.5 rounded up: a row of
#: 4 yields 6, and a row of 1 still yields 2, so there is always an alternative
#: to reject the first one in favour of. Named because M7 will want to tune it
#: once the review loop shows how many candidates actually survive.
OVER_GENERATION_MULTIPLIER = 1.5


def over_generated_count(count: int, multiplier: float = OVER_GENERATION_MULTIPLIER) -> int:
    """How many candidates to ask for, for a row of ``count`` questions."""
    if count <= 0:
        return 0
    return math.ceil(count * multiplier)


# --- The validated shape -----------------------------------------------------

_TRUE_FALSE = {"true": "True", "false": "False"}


class CandidateOut(BaseModel):
    """One candidate as the model must return it, checked before it is used.

    The per-type checks are here rather than in a later pass because they are
    what makes the JSON a *question*: an MCQ whose `correct` is not one of its
    options is not a slightly flawed candidate, it is unusable, and a retry
    usually fixes it.
    """

    stem: str
    type: str
    options: list[str] = Field(default_factory=list)
    correct: str
    explanation: str = ""
    source_ref: str

    @model_validator(mode="after")
    def _check_shape(self):
        self.stem = self.stem.strip()
        self.correct = self.correct.strip()
        self.explanation = self.explanation.strip()
        self.source_ref = self.source_ref.strip()
        self.type = self.type.strip().lower()
        self.options = [o.strip() for o in self.options if o and o.strip()]

        if not self.stem:
            raise ValueError("a question with no stem")
        if not self.correct:
            raise ValueError("a question with no answer")
        if not self.source_ref:
            raise ValueError("a question with no source_ref")
        if self.type not in MVP_TYPES:
            raise ValueError(f"unsupported question type {self.type!r}")

        if self.type == BlueprintRow.QuestionType.MCQ:
            if len(self.options) < 3:
                raise ValueError("a multiple-choice question needs at least 3 options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("a multiple-choice question repeats an option")
            if self.correct not in self.options:
                raise ValueError("`correct` is not one of the options")
        elif self.type == BlueprintRow.QuestionType.TRUE_FALSE:
            key = self.correct.strip().casefold()
            if key not in _TRUE_FALSE:
                raise ValueError("a true/false answer must be True or False")
            self.correct = _TRUE_FALSE[key]
            self.options = ["True", "False"]
        else:
            # Short answer and numeric are open questions; options would only be
            # a multiple choice wearing the wrong label.
            self.options = []
        return self


class GenerationOut(BaseModel):
    questions: list[CandidateOut] = Field(default_factory=list)


# --- The item Agent 2A is handed ---------------------------------------------


@dataclass(frozen=True)
class GenerationItem:
    """One blueprint item plus the passages it is written from.

    Deliberately not a `BlueprintRow`: generation needs the topic, type, level,
    marks, expected time and passages, and nothing about the database. That is
    what lets the test suite exercise this with four passages and no rows at
    all, and what lets M7 re-run one item for a replacement question without
    rebuilding the plan.
    """

    course_name: str
    topic_name: str
    question_type: str
    level: str
    marks: Decimal
    count: int
    passages: tuple[Passage, ...] = ()
    language: str = "en"
    expected_minutes: float | None = None
    row_id: int | None = None
    exam_id: int | None = None

    @property
    def candidates_wanted(self) -> int:
        return over_generated_count(self.count)

    @property
    def type_label(self) -> str:
        try:
            return BlueprintRow.QuestionType(self.question_type).label
        except ValueError:
            return self.question_type

    @property
    def level_label(self) -> str:
        try:
            return BlueprintRow.Level(self.level).label
        except ValueError:
            return self.level

    @property
    def is_supported(self) -> bool:
        return self.question_type in MVP_TYPES


def item_for_row(row: BlueprintRow, passages) -> GenerationItem:
    """Build the item for one blueprint row, given the passages M4 retrieved.

    Expected time per question is the exam's own arithmetic — its duration split
    over its questions — not a number a model is asked to guess.
    """
    exam = row.blueprint.exam
    minutes = (
        exam.duration_minutes / exam.question_count if exam.question_count else None
    )
    return GenerationItem(
        course_name=exam.course.name,
        topic_name=row.topic.name,
        question_type=row.question_type,
        level=row.level,
        marks=row.marks_per_question,
        count=row.count,
        passages=tuple(passages),
        language=exam.language,
        expected_minutes=minutes,
        row_id=row.pk,
        exam_id=exam.pk,
    )


# --- What comes back ---------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One generated question, already tied to the passage it came from."""

    stem: str
    question_type: str
    options: tuple[str, ...]
    correct: str
    explanation: str
    #: The supplied passage this candidate cited, resolved — not the model's
    #: string. A candidate only exists if this resolved.
    passage: Passage

    @property
    def source_ref(self) -> str:
        return self.passage.page_ref

    @property
    def from_ocr(self) -> bool:
        """Whether this question is quoting a transcription of a picture.

        A candidate cites exactly one passage, so "any source passage was OCR"
        is that passage. M6 and the review screen show it, because a question
        built on an OCR reading of a slide deserves the instructor's eye on the
        wording before it reaches a student.
        """
        return self.passage.from_ocr


@dataclass
class GenerationRun:
    """What one generation call produced, in terms worth reporting."""

    item: GenerationItem
    candidates: list[Candidate] = field(default_factory=list)
    #: Candidates the model returned that named no supplied passage. These are
    #: the M5 failure the success check counts: a question written from outside
    #: the material. Dropped, never stored, always reported.
    ungrounded: list[str] = field(default_factory=list)

    @property
    def wanted(self) -> int:
        return self.item.candidates_wanted

    @property
    def returned(self) -> int:
        return len(self.candidates) + len(self.ungrounded)

    @property
    def grounded_rate(self) -> float:
        return len(self.candidates) / self.returned if self.returned else 0.0


# --- Resolving a citation against the passages actually supplied -------------

_LABEL = re.compile(r"P\s*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")


def resolve_passage(source_ref: str, passages) -> Passage | None:
    """The supplied passage a candidate cited, or None if it cited nothing real.

    Tolerant about *form* and strict about *fact*: "P2", "2", "p 2" and a page
    reference copied out in full all resolve, because which of those a model
    emits is a formatting habit. A label past the end of the list, or a page the
    bundle does not contain, resolves to nothing — that is not a habit, it is a
    question about material the model was never given.
    """
    passages = list(passages)
    if not source_ref or not passages:
        return None

    label = _LABEL.search(source_ref)
    if label:
        index = int(label.group(1))
        if 1 <= index <= len(passages):
            return passages[index - 1]

    # A page reference, copied out whole or in part: match on the file name and
    # page when both are there, on the page alone when it is unambiguous.
    for number in (int(n) for n in _NUMBER.findall(source_ref)):
        on_page = [p for p in passages if p.page == number]
        named = [p for p in on_page if p.source_file.casefold() in source_ref.casefold()]
        if named:
            return named[0]
        if len(on_page) == 1:
            return on_page[0]
    return None


# --- The call ----------------------------------------------------------------

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _loads(text: str) -> dict:
    """Parse a JSON answer, tolerating a code fence around it (as in M2)."""
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def build_prompt(item: GenerationItem) -> tuple[str, str]:
    """The system and user halves of the generation call, composed."""
    from agents.prompts.generate import SYSTEM, build_user_prompt

    user = build_user_prompt(
        course_name=item.course_name,
        topic_name=item.topic_name,
        question_type=item.question_type,
        type_label=item.type_label,
        level=item.level,
        level_label=item.level_label,
        marks=item.marks,
        count=item.candidates_wanted,
        passages=item.passages,
        language=item.language,
        expected_minutes=item.expected_minutes,
    )
    return SYSTEM, user


def generate_candidates(item: GenerationItem, *, provider=None) -> GenerationRun:
    """Write `ceil(N * 1.5)` candidates for one blueprint item.

    Raises `UnsupportedQuestionType` for a type the MVP does not write,
    `QuestionGenerationError` if the item has no passages to be grounded in or
    the model's answer cannot be validated twice. Nothing partial is returned:
    on failure the caller has an error, not a half-batch.
    """
    from agents.provider import LLMError, get_provider  # the seam; see §3

    if not item.is_supported:
        raise UnsupportedQuestionType(
            f"إحكام does not write {item.type_label!r} questions yet — the MVP "
            f"covers {', '.join(sorted(MVP_TYPES))}. Change this blueprint row's "
            "type, or leave the row for a later milestone."
        )
    if not item.passages:
        raise QuestionGenerationError(
            f"There are no reference passages for '{item.topic_name}', so there is "
            "nothing to write a question from. A question written without them "
            "would be general knowledge wearing a citation."
        )
    if item.candidates_wanted <= 0:
        return GenerationRun(item=item)

    try:
        provider = provider or get_provider()
    except LLMError as exc:
        raise QuestionGenerationError(str(exc)) from exc

    system, user = build_prompt(item)
    last_error = ""
    for attempt in (1, 2):
        try:
            response = provider.complete(system, user, json_mode=True, temperature=0.4)
            result = GenerationOut.model_validate(_loads(response.text))
            return _collect(item, result)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"the answer was not the expected JSON ({exc.__class__.__name__})"
            logger.warning("Generation attempt %s returned bad JSON: %s", attempt, exc)
        except Exception as exc:  # noqa: BLE001 — surfaced as itself, never stored
            # Not a malformed answer: the call never reached the model. Reporting
            # it as bad JSON sends the instructor to the wrong place, and a
            # second attempt would fail identically. The M2 rule.
            raise QuestionGenerationError(
                f"The generation call did not complete: {exc}"
            ) from exc

    raise QuestionGenerationError(
        f"The model's answer could not be read after two attempts: {last_error}. "
        f"No candidates were produced for '{item.topic_name}' — none is better "
        "than unchecked ones."
    )


def _collect(item: GenerationItem, result: GenerationOut) -> GenerationRun:
    """Turn validated JSON into candidates, dropping any that cite nothing real."""
    run = GenerationRun(item=item)
    for candidate in result.questions:
        passage = resolve_passage(candidate.source_ref, item.passages)
        if passage is None:
            run.ungrounded.append(candidate.stem)
            logger.warning(
                "Dropped a candidate for '%s': source_ref %r names no supplied "
                "passage, so the question was not written from this course.",
                item.topic_name,
                candidate.source_ref,
            )
            continue
        if candidate.type != item.question_type:
            # The blueprint decided the type; a candidate of another type would
            # spend a row's marks on a question the plan did not ask for.
            run.ungrounded.append(candidate.stem)
            logger.warning(
                "Dropped a candidate for '%s': asked for %s, got %s.",
                item.topic_name,
                item.question_type,
                candidate.type,
            )
            continue
        run.candidates.append(
            Candidate(
                stem=candidate.stem,
                question_type=candidate.type,
                options=tuple(candidate.options),
                correct=candidate.correct,
                explanation=candidate.explanation,
                passage=passage,
            )
        )
    return run


# --- Storing -----------------------------------------------------------------


def save_candidates(run: GenerationRun, *, exam=None, row=None) -> list[Question]:
    """Store a run's candidates as `Question` rows, status `candidate`.

    Every row here has already been validated and grounded; nothing else is
    allowed to write a `Question` from model output. The instructor's approve /
    reject decision is what moves it off `candidate`.
    """
    from exams.models import Exam

    if row is None and run.item.row_id is not None:
        row = BlueprintRow.objects.filter(pk=run.item.row_id).first()
    if exam is None:
        exam = row.blueprint.exam if row else Exam.objects.filter(pk=run.item.exam_id).first()
    if exam is None:
        raise QuestionGenerationError("A candidate has to belong to an exam to be stored.")

    position = Question.objects.filter(exam=exam).count()
    stored = []
    with transaction.atomic():
        for offset, candidate in enumerate(run.candidates):
            stored.append(
                Question.objects.create(
                    exam=exam,
                    blueprint_row=row,
                    stem=candidate.stem,
                    question_type=candidate.question_type,
                    options=list(candidate.options),
                    correct=candidate.correct,
                    explanation=candidate.explanation,
                    source_ref=candidate.source_ref[
                        : Question._meta.get_field("source_ref").max_length
                    ],
                    source_chunk_id=candidate.passage.chunk_id,
                    from_ocr=candidate.from_ocr,
                    position=position + offset,
                )
            )
    return stored


__all__ = [
    "MVP_TYPES",
    "OVER_GENERATION_MULTIPLIER",
    "Candidate",
    "CandidateOut",
    "GenerationItem",
    "GenerationOut",
    "GenerationRun",
    "QuestionGenerationError",
    "UnsupportedQuestionType",
    "build_prompt",
    "generate_candidates",
    "item_for_row",
    "over_generated_count",
    "resolve_passage",
    "save_candidates",
]
