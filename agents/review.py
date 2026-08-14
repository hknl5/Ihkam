"""Agent 3A — judging a question before an instructor ever sees it (M7).

Agent 2A over-generates. This module is what makes that safe: every candidate is
checked, and a rejection carries a note specific enough for M8 to hand back to
2A as the brief for a replacement.

Four decisions shape it:

* **Agent 3A judges; it never rewrites.** A reviewer that returns a corrected
  question is a second author whose work nobody checked, and the loop meant to
  catch bad questions starts writing them. Nothing here edits a stem, an option,
  an answer or a key — the output is verdicts and notes.
* **What can be computed exactly is computed, not asked.** Whether an MCQ has
  two correct options, whether two options are the same string, whether the
  correct option is conspicuously the longest, whether a mark split adds up:
  these are string comparison and arithmetic. Sending them to a model would make
  a fixed answer probabilistic and cost a call to get a worse result. They run in
  Python. The model is asked only what needs language understanding — scope,
  clarity, level, whether the answer is actually right, whether a distractor is
  plausible rather than merely different.
* **A rejection is an instruction, not a complaint.** Every failed check carries
  a `reason` (what is wrong with this question) *and* a `requirement` (what a
  replacement must do differently). "Bad question" is not something a generator
  can act on; "requested multi-step but produced a definition — require a
  computation from the given values" is. M8 feeds these straight back to 2A.
* **All the checks run, even after the first failure.** The cheaper design is to
  stop at the first rejection, and it is the wrong one: 2A would fix the level,
  regenerate, and only then discover the scope problem. One review, one complete
  set of notes, one regeneration.

The deterministic maths hook lives here too: `MATH_CHECKERS` is the seam a real
symbolic checker plugs into later. M6's mark-sum flag is its first member — it is
surfaced here, not recomputed.

No model SDK is imported here; the call goes through ``get_provider()``.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError, model_validator

from agents.answer_key import NumericKey, ObjectiveKey, ShortAnswerKey, answer_key_from_dict
from courses.services.retrieval import Passage
from exams.models import BlueprintRow

logger = logging.getLogger(__name__)


class QuestionReviewError(RuntimeError):
    """The review could not be completed. No verdict is better than a guessed one."""


# --- The checks --------------------------------------------------------------

#: The five checks the plan names. A finding always names one of these, so the
#: instructor screen (M9) and the regeneration loop (M8) can group by check
#: rather than by the sentence a model happened to write.
CONTENT_LINK = "content_link"
CLARITY = "clarity"
ANSWER_CORRECTNESS = "answer_correctness"
OPTION_QUALITY = "option_quality"
LEVEL_MATCH = "level_match"

CHECKS: tuple[str, ...] = (
    CONTENT_LINK,
    CLARITY,
    ANSWER_CORRECTNESS,
    OPTION_QUALITY,
    LEVEL_MATCH,
)

CHECK_LABELS = {
    CONTENT_LINK: "Content link",
    CLARITY: "Clarity",
    ANSWER_CORRECTNESS: "Answer correctness",
    OPTION_QUALITY: "Option quality",
    LEVEL_MATCH: "Level match",
}

#: Where a verdict came from. Worth keeping on the finding: an instructor who
#: disagrees with "the correct option is conspicuously the longest" is arguing
#: with a threshold, and one who disagrees with "this is below the level asked
#: for" is arguing with a model. Those are different conversations.
BY_PYTHON = "python"
BY_MODEL = "model"

#: How much longer than the average wrong option the correct one may be before
#: length itself becomes the giveaway. 1.6 is deliberately loose: a correct
#: option that is a little more precise is normal, and a check that fires on
#: every well-written answer would be turned off within a week.
LONGEST_CORRECT_RATIO = 1.6

#: Below this many characters, the ratio means nothing — "42" against "7" is
#: three times the length and gives nothing away.
LONGEST_CORRECT_MIN_CHARS = 25


@dataclass(frozen=True)
class Finding:
    """One check's verdict on one question.

    `requirement` is the half M8 needs: not what went wrong, but what the next
    attempt has to do. A finding that fails without one is a complaint.
    """

    check: str
    passed: bool
    reason: str = ""
    requirement: str = ""
    decided_by: str = BY_PYTHON

    @property
    def label(self) -> str:
        return CHECK_LABELS.get(self.check, self.check)

    @property
    def note(self) -> str:
        """The sentence the instructor reads and the generator is given."""
        if self.passed:
            return ""
        parts = [f"{self.label}: {self.reason}".strip()]
        if self.requirement:
            parts.append(f"The replacement must: {self.requirement}")
        return " ".join(parts)


@dataclass
class ReviewResult:
    """What review concluded about one question, and why."""

    findings: list[Finding] = field(default_factory=list)
    #: Set when the model half was deliberately not run (`python_only`). The
    #: distinction matters: "passed every check" and "passed every check we ran"
    #: are not the same claim.
    model_checked: bool = True

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def rejected(self) -> bool:
        return bool(self.failures)

    @property
    def failed_checks(self) -> list[str]:
        return [f.check for f in self.failures]

    @property
    def notes(self) -> list[str]:
        """The rejection notes, in check order. M8's input to a regeneration."""
        return [f.note for f in self.failures]

    @property
    def note(self) -> str:
        return "\n".join(self.notes)

    def finding_for(self, check: str) -> Finding | None:
        for finding in self.findings:
            if finding.check == check:
                return finding
        return None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return "passed" if self.passed else f"rejected ({', '.join(self.failed_checks)})"


# --- What is reviewed --------------------------------------------------------


@dataclass(frozen=True)
class ReviewSubject:
    """One question as review sees it: what was asked for, what came back, from what.

    Deliberately not a `Question` and not a `Candidate`: M8 reviews candidates
    that are not stored yet, M9 reviews rows an instructor is looking at, and
    the test suite reviews neither. Both are adapted into this.
    """

    stem: str
    question_type: str
    options: tuple[str, ...] = ()
    correct: str = ""
    explanation: str = ""
    requested_level: str = ""
    marks: Decimal = Decimal("0")
    topic_name: str = ""
    course_name: str = ""
    passages: tuple[Passage, ...] = ()
    answer_key: object | None = None
    #: M6's verdict, carried rather than recomputed. `None` means there was no
    #: mark split to add up.
    mark_sum_ok: bool | None = None
    question_id: int | None = None

    @property
    def type_label(self) -> str:
        try:
            return BlueprintRow.QuestionType(self.question_type).label
        except ValueError:
            return self.question_type

    @property
    def level_label(self) -> str:
        try:
            return BlueprintRow.Level(self.requested_level).label
        except ValueError:
            return self.requested_level

    @property
    def is_objective(self) -> bool:
        return bool(self.options)

    @classmethod
    def from_candidate(cls, candidate, item) -> ReviewSubject:
        """A freshly generated candidate, with the item it was generated for."""
        return cls(
            stem=candidate.stem,
            question_type=candidate.question_type,
            options=tuple(candidate.options),
            correct=candidate.correct,
            explanation=candidate.explanation,
            requested_level=item.level,
            marks=item.marks,
            topic_name=item.topic_name,
            course_name=item.course_name,
            passages=tuple(item.passages),
            answer_key=candidate.answer_key,
            mark_sum_ok=candidate.mark_sum_ok,
        )

    @classmethod
    def from_question(cls, question, passages=None) -> ReviewSubject:
        """A stored `Question`, with the passage it cites if none is supplied.

        The stored question knows its chunk, so review can be re-run on a row an
        instructor is looking at without re-retrieving anything.
        """
        row = question.blueprint_row
        if passages is None:
            passages = _passages_for(question)
        return cls(
            stem=question.stem,
            question_type=question.question_type,
            options=tuple(question.options or ()),
            correct=question.correct,
            explanation=question.explanation,
            requested_level=row.level if row else "",
            marks=row.marks_per_question if row else Decimal("0"),
            topic_name=row.topic.name if row else "",
            course_name=question.exam.course.name,
            passages=tuple(passages),
            answer_key=answer_key_from_dict(question.answer_key),
            mark_sum_ok=question.mark_sum_ok,
            question_id=question.pk,
        )


def _passages_for(question) -> list[Passage]:
    """The passage a stored question cites, as a `Passage`."""
    chunk = question.source_chunk
    if chunk is None:
        return []
    return [
        Passage(
            chunk_id=chunk.pk,
            text=chunk.text,
            page=chunk.page,
            source_file=chunk.source_file.original_name,
            score=1.0,
            topic=chunk.topic.name if chunk.topic else None,
            from_ocr=question.from_ocr,
        )
    ]


# --- The deterministic checks ------------------------------------------------

_SPACES = re.compile(r"\s+")
#: Only the punctuation that is *presentation* is removed, and only at the edges:
#: a full stop or a wrapping quote. Interior symbols are left exactly as written,
#: because in this material they are the content — stripping them collapses
#: `(x y)′` and `x + y` into the same string and reports two different Boolean
#: expressions as one option repeated. (Found doing exactly that on the discrete
#: maths course.) Two options that differ only in an operator are two options.
_EDGE_LEAD = re.compile(r"^[\s\"'“”«»(\[]+")
_EDGE_TRAIL = re.compile(r"[\s\.,;:!?؟،\"”»]+$")


def normalise_option(text: str) -> str:
    """An option reduced to what it says, for comparing two of them.

    Case, edge punctuation, spacing and Unicode form are presentation. "A binary
    search." and "a binary search" are one option written twice, and an MCQ with
    one option written twice has three choices wearing four labels.
    """
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = _SPACES.sub(" ", text).strip()
    return _EDGE_TRAIL.sub("", _EDGE_LEAD.sub("", text)).strip()


def check_single_correct(subject: ReviewSubject) -> Finding | None:
    """Exactly one option matches the answer. Counting, not judgement."""
    if not subject.is_objective:
        return None
    matches = [o for o in subject.options if normalise_option(o) == normalise_option(subject.correct)]
    if len(matches) == 1:
        return Finding(check=OPTION_QUALITY, passed=True, decided_by=BY_PYTHON)
    if not matches:
        return Finding(
            check=OPTION_QUALITY,
            passed=False,
            reason=(
                f"none of the options is the stated answer ({subject.correct!r}), so "
                "there is no way for a student to answer correctly."
            ),
            requirement=(
                "write the correct answer as one of the options, word for word, and "
                "make the other options wrong"
            ),
            decided_by=BY_PYTHON,
        )
    return Finding(
        check=OPTION_QUALITY,
        passed=False,
        reason=(
            f"{len(matches)} options are the stated answer, so more than one choice "
            "is correct."
        ),
        requirement="give exactly one correct option and make every other option wrong",
        decided_by=BY_PYTHON,
    )


def check_duplicate_options(subject: ReviewSubject) -> Finding | None:
    """No two options are the same string once presentation is stripped away."""
    if not subject.is_objective:
        return None
    seen: dict[str, str] = {}
    for option in subject.options:
        key = normalise_option(option)
        if key in seen:
            return Finding(
                check=OPTION_QUALITY,
                passed=False,
                reason=(
                    f"two options say the same thing — {seen[key]!r} and {option!r} — "
                    "so the student is choosing between three answers, not four."
                ),
                requirement=(
                    "replace the repeated option with a different wrong answer drawn "
                    "from the same passage"
                ),
                decided_by=BY_PYTHON,
            )
        seen[key] = option
    return Finding(check=OPTION_QUALITY, passed=True, decided_by=BY_PYTHON)


def check_correct_not_longest(subject: ReviewSubject) -> Finding | None:
    """The correct option is not conspicuously the longest — the oldest tell there is."""
    if not subject.is_objective or len(subject.options) < 3:
        return None
    correct = next(
        (o for o in subject.options if normalise_option(o) == normalise_option(subject.correct)),
        None,
    )
    if correct is None:  # already reported by check_single_correct
        return None
    others = [o for o in subject.options if o is not correct]
    if not others:
        return None

    length = len(correct.strip())
    average = sum(len(o.strip()) for o in others) / len(others)
    if length < LONGEST_CORRECT_MIN_CHARS or not average:
        return Finding(check=OPTION_QUALITY, passed=True, decided_by=BY_PYTHON)
    if length > average * LONGEST_CORRECT_RATIO and length > max(len(o.strip()) for o in others):
        return Finding(
            check=OPTION_QUALITY,
            passed=False,
            reason=(
                f"the correct option is the longest by a wide margin ({length} "
                f"characters against an average of {average:.0f}), which lets a "
                "student pick it without reading the material."
            ),
            requirement=(
                "make the options roughly the same length — either shorten the "
                "correct one or give the distractors the same level of detail"
            ),
            decided_by=BY_PYTHON,
        )
    return Finding(check=OPTION_QUALITY, passed=True, decided_by=BY_PYTHON)


def check_answer_well_formed(subject: ReviewSubject) -> Finding | None:
    """The answer exists and is the shape its type needs. Structure, not truth.

    Whether the answer is *right* is a language judgement and belongs to the
    model; whether there is an answer at all, and whether a marker could use it,
    is structure and belongs here.
    """
    if not (subject.correct or "").strip():
        return Finding(
            check=ANSWER_CORRECTNESS,
            passed=False,
            reason="the question has no answer.",
            requirement="give the answer together with the question",
            decided_by=BY_PYTHON,
        )
    key = subject.answer_key
    if key is None:
        return Finding(
            check=ANSWER_CORRECTNESS,
            passed=False,
            reason="the question has no answer key, so it cannot be marked.",
            requirement="produce the answer key with the question, in the shape its type needs",
            decided_by=BY_PYTHON,
        )
    if isinstance(key, ShortAnswerKey) and not key.required_elements:
        return Finding(
            check=ANSWER_CORRECTNESS,
            passed=False,
            reason=(
                "the short-answer key lists no required elements, so a marker has "
                "nothing to mark against but resemblance to the model answer."
            ),
            requirement="list the ideas a student's answer must contain to earn the marks",
            decided_by=BY_PYTHON,
        )
    if isinstance(key, NumericKey) and not key.steps:
        return Finding(
            check=ANSWER_CORRECTNESS,
            passed=False,
            reason="the numeric key shows no worked steps, so the marks cannot be awarded partly.",
            requirement="show the solution steps and what each one is worth",
            decided_by=BY_PYTHON,
        )
    if isinstance(key, ObjectiveKey) and subject.is_objective:
        if normalise_option(key.answer) != normalise_option(subject.correct):
            return Finding(
                check=ANSWER_CORRECTNESS,
                passed=False,
                reason=(
                    f"the answer key says {key.answer!r} but the question's answer is "
                    f"{subject.correct!r}; a marker would not know which to use."
                ),
                requirement="give one answer, the same in the question and in the key",
                decided_by=BY_PYTHON,
            )
    return Finding(check=ANSWER_CORRECTNESS, passed=True, decided_by=BY_PYTHON)


# --- The deterministic maths hook --------------------------------------------

#: A maths checker takes a subject and returns findings. M6's mark-sum flag is
#: the first and, for now, only one: it is *surfaced* here, not recomputed, so
#: there is exactly one place in the system that decides whether a mark split
#: adds up. A later symbolic evaluator — one that actually recomputes the
#: arithmetic of a numeric answer rather than trusting the model's — registers
#: here and needs no change to `review_question`.
MathChecker = Callable[[ReviewSubject], list[Finding]]


def mark_sum_checker(subject: ReviewSubject) -> list[Finding]:
    """M6's mark-sum verdict, reported as a review finding."""
    if subject.mark_sum_ok is None:
        return []
    if subject.mark_sum_ok:
        return [Finding(check=ANSWER_CORRECTNESS, passed=True, decided_by=BY_PYTHON)]
    key = subject.answer_key
    total = key.total_marks if isinstance(key, NumericKey) else None
    found = f"total {total}" if total is not None else "do not total the marks"
    return [
        Finding(
            check=ANSWER_CORRECTNESS,
            passed=False,
            reason=(
                f"the marking scheme does not add up: the solution steps {found} "
                f"but the question is worth {subject.marks}."
            ),
            requirement=(
                f"split exactly {subject.marks} mark(s) across the solution steps"
            ),
            decided_by=BY_PYTHON,
        )
    ]


MATH_CHECKERS: list[MathChecker] = [mark_sum_checker]


def register_math_checker(checker: MathChecker) -> MathChecker:
    """Add a deterministic maths checker to the review (the M7 hook)."""
    MATH_CHECKERS.append(checker)
    return checker


def deterministic_findings(subject: ReviewSubject) -> list[Finding]:
    """Everything review can decide without a model. Exact, free, and repeatable."""
    findings: list[Finding] = []
    for check in (
        check_single_correct,
        check_duplicate_options,
        check_correct_not_longest,
        check_answer_well_formed,
    ):
        finding = check(subject)
        if finding is not None:
            findings.append(finding)
    for checker in MATH_CHECKERS:
        findings.extend(checker(subject))
    return findings


# --- The model's half --------------------------------------------------------

#: The model's verdict names map onto the plan's checks. Two of the model's
#: judgements land on checks Python also touches: whether the answer is actually
#: right joins the well-formedness of the key, and whether a distractor is
#: plausible joins the counting of the options. That is deliberate — one check
#: per line of the plan, whatever decided it.
MODEL_CHECK_MAP = {
    "content_link": CONTENT_LINK,
    "clarity": CLARITY,
    "answer_consistency": ANSWER_CORRECTNESS,
    "level_match": LEVEL_MATCH,
    "distractor_quality": OPTION_QUALITY,
}

#: What a replacement must do, per check, when the model fails one without
#: saying what to do about it. A rejection with no instruction is not something
#: M8 can act on, so there is always one.
DEFAULT_REQUIREMENTS = {
    CONTENT_LINK: (
        "write the question only from the supplied passages, using no fact, term or "
        "example that is not in them"
    ),
    CLARITY: "state the question so that it has exactly one reasonable interpretation",
    ANSWER_CORRECTNESS: "give one answer that the supplied passages support",
    OPTION_QUALITY: (
        "give exactly one correct option and make each wrong option plausible, "
        "distinct in meaning, and drawn from the same passage"
    ),
    LEVEL_MATCH: "write a question at the level that was requested",
}


class VerdictOut(BaseModel):
    """One check's verdict, as the model must return it."""

    ok: bool
    reason: str = ""
    requirement: str = ""

    @model_validator(mode="after")
    def _check(self):
        self.reason = (self.reason or "").strip()
        self.requirement = (self.requirement or "").strip()
        if not self.ok and not self.reason:
            # A rejection with no reason cannot be shown to an instructor or fed
            # back to the generator. Retrying is cheaper than inventing one.
            raise ValueError("a failed check with no reason")
        return self


class ReviewOut(BaseModel):
    """The model's half of the review, checked before it is used."""

    content_link: VerdictOut
    clarity: VerdictOut
    answer_consistency: VerdictOut
    level_match: VerdictOut
    distractor_quality: VerdictOut = Field(default_factory=lambda: VerdictOut(ok=True))


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _loads(text: str) -> dict:
    """Parse a JSON answer, tolerating a code fence around it (as in M2)."""
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def build_prompt(subject: ReviewSubject) -> tuple[str, str]:
    """The system and user halves of the review call, composed."""
    from agents.prompts.review import SYSTEM, build_user_prompt

    user = build_user_prompt(
        course_name=subject.course_name,
        topic_name=subject.topic_name,
        question_type=subject.question_type,
        type_label=subject.type_label,
        level=subject.requested_level,
        level_label=subject.level_label,
        marks=subject.marks,
        stem=subject.stem,
        options=subject.options,
        correct=subject.correct,
        explanation=subject.explanation,
        answer_key=subject.answer_key,
        passages=subject.passages,
    )
    return SYSTEM, user


def model_findings(subject: ReviewSubject, *, provider=None) -> list[Finding]:
    """The judgements that need language understanding, from one call.

    Raises `QuestionReviewError` if the answer cannot be validated twice, or if
    the call never reached the model — the M2 rule: a rate limit is reported as
    itself, not retried into the same wall, and never turned into a pass.
    """
    from agents.provider import LLMError, get_provider

    try:
        provider = provider or get_provider()
    except LLMError as exc:
        raise QuestionReviewError(str(exc)) from exc

    system, user = build_prompt(subject)
    last_error = ""
    for attempt in (1, 2):
        try:
            response = provider.complete(system, user, json_mode=True, temperature=0.0)
            result = ReviewOut.model_validate(_loads(response.text))
            return _to_findings(subject, result)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"the answer was not the expected JSON ({exc.__class__.__name__})"
            logger.warning("Review attempt %s returned bad JSON: %s", attempt, exc)
        except Exception as exc:  # noqa: BLE001 — surfaced as itself
            raise QuestionReviewError(f"The review call did not complete: {exc}") from exc

    raise QuestionReviewError(
        f"The reviewer's answer could not be read after two attempts: {last_error}. "
        f"'{subject.stem[:60]}' was not reviewed — an unreviewed question is not a "
        "passed one."
    )


def _to_findings(subject: ReviewSubject, result: ReviewOut) -> list[Finding]:
    findings = []
    for name, check in MODEL_CHECK_MAP.items():
        verdict: VerdictOut = getattr(result, name)
        if check == OPTION_QUALITY and not subject.is_objective:
            # An open question has no distractors to judge; a verdict either way
            # would be about something that does not exist.
            continue
        findings.append(
            Finding(
                check=check,
                passed=verdict.ok,
                reason=verdict.reason,
                requirement=verdict.requirement or DEFAULT_REQUIREMENTS[check],
                decided_by=BY_MODEL,
            )
        )
    return findings


# --- The review --------------------------------------------------------------


def review_question(
    subject: ReviewSubject, *, provider=None, python_only: bool = False
) -> ReviewResult:
    """Review one question: the deterministic checks, then the language ones.

    Every check runs, including after a failure, so that one review produces one
    complete set of notes and M8 regenerates once rather than once per fault.

    `python_only` runs the deterministic half alone — no call, no cost. It is
    what the debug screen and a bulk re-check use, and the result says so
    (`model_checked=False`) rather than claiming a clean bill of health it did
    not earn.
    """
    findings = deterministic_findings(subject)
    if not python_only:
        findings.extend(model_findings(subject, provider=provider))
    return ReviewResult(findings=findings, model_checked=not python_only)


def review_candidate(candidate, item, *, provider=None, python_only: bool = False) -> ReviewResult:
    """Review a freshly generated candidate, before it is stored (M8's entry point)."""
    return review_question(
        ReviewSubject.from_candidate(candidate, item),
        provider=provider,
        python_only=python_only,
    )


def review_stored_question(question, *, passages=None, provider=None, python_only: bool = False):
    """Review a stored `Question` — the review screen's entry point (M9)."""
    return review_question(
        ReviewSubject.from_question(question, passages=passages),
        provider=provider,
        python_only=python_only,
    )


__all__ = [
    "ANSWER_CORRECTNESS",
    "BY_MODEL",
    "BY_PYTHON",
    "CHECKS",
    "CHECK_LABELS",
    "CLARITY",
    "CONTENT_LINK",
    "DEFAULT_REQUIREMENTS",
    "LEVEL_MATCH",
    "LONGEST_CORRECT_RATIO",
    "MATH_CHECKERS",
    "OPTION_QUALITY",
    "Finding",
    "QuestionReviewError",
    "ReviewOut",
    "ReviewResult",
    "ReviewSubject",
    "VerdictOut",
    "build_prompt",
    "check_answer_well_formed",
    "check_correct_not_longest",
    "check_duplicate_options",
    "check_single_correct",
    "deterministic_findings",
    "mark_sum_checker",
    "model_findings",
    "normalise_option",
    "register_math_checker",
    "review_candidate",
    "review_question",
    "review_stored_question",
]
