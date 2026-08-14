"""The answer key, by question type (M6).

M5 ended with `correct` and `explanation` arriving in the same validated object
as the stem. That is enough for an MCQ and not enough for anything else: a short
answer needs to say *what a marker is looking for*, and a numeric problem needs
to say *how the marks are split*, or the key is a number an instructor has to
re-derive before they can use it.

Three decisions shape this module:

* **The key is part of the generate call, never a second one.** Nothing here
  talks to a provider; it parses and checks what the one call already returned.
  A key produced later would be a key for a question the model has to re-read
  rather than one it wrote, and would be free to disagree with the answer it
  already gave.
* **The model splits the marks; Python only adds them up.** How a numeric
  solution divides into steps is a teaching judgement — two markers would split
  it differently, and a student will not write the model's steps verbatim. What
  is *not* a judgement is whether the split totals the question's marks. That
  check is arithmetic, so it is done in Python, deterministically, and a
  mismatch is **flagged, not corrected**: `mark_sum_ok=False` puts the candidate
  in front of M7's review and the instructor rather than quietly rewriting a
  mark scheme neither of them chose. This is the first hook toward M7's
  deterministic maths checker.
* **The essay format exists before essay questions do.** Essays are post-MVP,
  but the shape they will store — rubric criteria with weights — is defined and
  validated here now, so that adding essay generation later is a change to what
  is written, not to what is stored or what review can read.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exams.models import BlueprintRow

# --- Kinds -------------------------------------------------------------------

#: A key kind is not a question type. Two types (MCQ, true/false) share one key
#: — a direct key — because from a marker's point of view they are the same act:
#: compare the student's choice with the right one. Essay is a kind with no type
#: behind it yet, on purpose.
OBJECTIVE = "objective"
SHORT_ANSWER = "short_answer"
NUMERIC = "numeric"
ESSAY = "essay"

KIND_FOR_TYPE: dict[str, str] = {
    BlueprintRow.QuestionType.MCQ: OBJECTIVE,
    BlueprintRow.QuestionType.TRUE_FALSE: OBJECTIVE,
    BlueprintRow.QuestionType.SHORT_ANSWER: SHORT_ANSWER,
    BlueprintRow.QuestionType.NUMERIC: NUMERIC,
    "essay": ESSAY,
}


def kind_for_type(question_type: str) -> str:
    """The key kind a question of this type must carry."""
    try:
        return KIND_FOR_TYPE[question_type]
    except KeyError:  # pragma: no cover - generation refuses these before here
        raise ValueError(f"no answer key format for question type {question_type!r}") from None


def _decimal(value) -> Decimal:
    """A mark figure, from whatever JSON shape the model wrote it in."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, AttributeError, TypeError):
        raise ValueError(f"{value!r} is not a number of marks") from None


def _cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


# --- Pieces ------------------------------------------------------------------


class SolutionStep(BaseModel):
    """One step of a worked solution, and what it is worth.

    `marks` is a *suggested* allocation for a human marker, not a threshold a
    student's answer is tested against.
    """

    text: str
    marks: Decimal = Decimal("0")

    @model_validator(mode="after")
    def _check(self):
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("a solution step with no text")
        self.marks = _cents(_decimal(self.marks))
        if self.marks < 0:
            raise ValueError("a solution step cannot be worth negative marks")
        return self


class RubricCriterion(BaseModel):
    """One row of an essay rubric: what is being judged, and its weight."""

    criterion: str
    weight: Decimal = Decimal("0")
    descriptor: str = ""

    @model_validator(mode="after")
    def _check(self):
        self.criterion = self.criterion.strip()
        self.descriptor = self.descriptor.strip()
        if not self.criterion:
            raise ValueError("a rubric criterion with no name")
        self.weight = _cents(_decimal(self.weight))
        if self.weight <= 0:
            raise ValueError("a rubric criterion must carry some weight")
        return self


# --- The keys ----------------------------------------------------------------


class AnswerKey(BaseModel):
    """What every key can be asked, whatever its type."""

    #: `model_answer` is the marker's phrase for it; pydantic reserves `model_`
    #: for its own methods, so the namespace guard is turned off here rather than
    #: renaming a field an instructor reads.
    model_config = ConfigDict(protected_namespaces=())

    #: Set by the subclass and stamped onto `kind` on validation, so a stored
    #: key says its own shape and `answer_key_from_dict` can read it back.
    KIND: ClassVar[str] = ""

    kind: str = ""
    explanation: str = ""

    @model_validator(mode="after")
    def _stamp_kind(self):
        self.kind = type(self).KIND
        self.explanation = (self.explanation or "").strip()
        return self

    def as_dict(self) -> dict:
        """The key as it is stored on `Question.answer_key`.

        `Decimal` is not JSON, so marks are written as strings rather than
        floats: half a mark has to come back out of the database as half a mark.
        """
        return _jsonable(self.model_dump())

    @property
    def display_answer(self) -> str:
        """The one line a list view shows beside the stem."""
        return ""


class ObjectiveKey(AnswerKey):
    """MCQ and true/false: the correct option, and the ones it beat."""

    KIND: ClassVar[str] = OBJECTIVE

    answer: str
    options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_objective(self):
        self.answer = self.answer.strip()
        self.options = [o.strip() for o in self.options if o and o.strip()]
        if not self.answer:
            raise ValueError("an objective key with no answer")
        if self.options and self.answer not in self.options:
            raise ValueError("the objective key's answer is not one of the options")
        return self

    @property
    def display_answer(self) -> str:
        return self.answer


class ShortAnswerKey(AnswerKey):
    """Short answer: the model answer, plus what an answer has to contain.

    `required_elements` is the part a marker actually uses. A model answer alone
    invites marking by resemblance; the elements say which ideas earn the marks,
    however the student phrased them.
    """

    KIND: ClassVar[str] = SHORT_ANSWER

    model_answer: str
    required_elements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_short(self):
        self.model_answer = self.model_answer.strip()
        self.required_elements = [e.strip() for e in self.required_elements if e and e.strip()]
        if not self.model_answer:
            raise ValueError("a short-answer key with no model answer")
        if not self.required_elements:
            raise ValueError(
                "a short-answer key must list the elements the answer has to contain"
            )
        return self

    @property
    def display_answer(self) -> str:
        return self.model_answer


class NumericKey(AnswerKey):
    """Numeric: the worked steps with their marks, and the final answer.

    `mark_sum_ok` is filled in by `check_mark_sum`, not by the model: it is
    `None` until the arithmetic has been checked, `True` when the steps total
    the question's marks, `False` when they do not. False is a flag for review,
    never a reason to drop the candidate or to rewrite the split.
    """

    KIND: ClassVar[str] = NUMERIC

    steps: list[SolutionStep] = Field(default_factory=list)
    final_answer: str
    mark_sum_ok: bool | None = None
    mark_sum_note: str = ""

    @model_validator(mode="after")
    def _check_numeric(self):
        self.final_answer = self.final_answer.strip()
        if not self.final_answer:
            raise ValueError("a numeric key with no final answer")
        if not self.steps:
            raise ValueError("a numeric key must show the steps that reach the answer")
        return self

    @property
    def total_marks(self) -> Decimal:
        return _cents(sum((step.marks for step in self.steps), Decimal("0")))

    @property
    def display_answer(self) -> str:
        return self.final_answer


class EssayKey(AnswerKey):
    """Essay: rubric criteria and their weights. Defined now, generated later.

    No essay candidate is produced by the MVP (`MVP_TYPES` does not include the
    type, and generation refuses it by name). This class exists so that the
    storage, the loader and review already understand an essay key on the day
    essays are switched on.
    """

    KIND: ClassVar[str] = ESSAY

    criteria: list[RubricCriterion] = Field(default_factory=list)
    model_answer: str = ""

    @model_validator(mode="after")
    def _check_essay(self):
        self.model_answer = self.model_answer.strip()
        if not self.criteria:
            raise ValueError("an essay key must have at least one rubric criterion")
        return self

    @property
    def total_weight(self) -> Decimal:
        return _cents(sum((c.weight for c in self.criteria), Decimal("0")))

    @property
    def display_answer(self) -> str:
        return ", ".join(c.criterion for c in self.criteria)


KEY_CLASSES: dict[str, type[AnswerKey]] = {
    OBJECTIVE: ObjectiveKey,
    SHORT_ANSWER: ShortAnswerKey,
    NUMERIC: NumericKey,
    ESSAY: EssayKey,
}


def _jsonable(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def answer_key_from_dict(data) -> AnswerKey | None:
    """Rebuild a stored key. Review reads the key back through this, not by hand."""
    if not data:
        return None
    kind = (data.get("kind") or "").strip()
    cls = KEY_CLASSES.get(kind)
    if cls is None:
        return None
    return cls.model_validate({k: v for k, v in data.items() if k != "kind"})


# --- The deterministic check -------------------------------------------------


def check_mark_sum(key: NumericKey, marks) -> NumericKey:
    """Do the per-step marks total the question's marks? Python decides, not a model.

    Sets `mark_sum_ok` and, when it is False, a note saying what was found
    against what was expected. The key is returned as it was otherwise: this
    function never edits a step, never rescales the split, never drops the
    candidate. A wrong total is something an instructor should see, and the
    right fix — more marks on the hard step, or a different question — is theirs.
    """
    expected = _cents(_decimal(marks))
    found = key.total_marks
    key.mark_sum_ok = found == expected
    key.mark_sum_note = (
        ""
        if key.mark_sum_ok
        else (
            f"The solution steps total {found} mark(s), but the question is worth "
            f"{expected}. The mark split needs an instructor's eye before this "
            "question is used."
        )
    )
    return key


__all__ = [
    "ESSAY",
    "KEY_CLASSES",
    "KIND_FOR_TYPE",
    "NUMERIC",
    "OBJECTIVE",
    "SHORT_ANSWER",
    "AnswerKey",
    "EssayKey",
    "NumericKey",
    "ObjectiveKey",
    "RubricCriterion",
    "ShortAnswerKey",
    "SolutionStep",
    "answer_key_from_dict",
    "check_mark_sum",
    "kind_for_type",
]
