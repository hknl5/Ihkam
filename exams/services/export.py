"""The deliverable: one form, out of إحكام, as a file (M11).

Two things live here, and keeping them apart is the whole design:

1. **The document** — what an exam paper *is*, as plain Python. A cover, an
   optional instruction block, an optional score distribution, and a numbered
   list of questions, each with its stem, its options and its marks. Built once
   from a `Form` and the instructor's options, by code that knows nothing about
   PDF.
2. **The exporter** — how that document becomes bytes. `PdfExporter` is the
   first one. Word, Moodle XML, QTI, Canvas and Blackboard are later additions
   that implement `render`, and none of them will need to re-derive what a paper
   is, re-read the answer-key formats, or re-decide what the exam PDF may not
   contain.

The rule that matters most is enforced in the document, not in the renderer:
**`ExamDocument` has no answers on it at all.** The exam paper is built by a
function that never reads `correct` or `answer_key`, so an exporter cannot leak
an answer onto a student's paper even by mistake — there is nothing in the
object it is handed to leak. The answer key is a separate document, built by a
separate function, rendered into a separate file. Two files, always, because one
file with a section at the back is one careless print away from a disaster.

`ExportOptions` is the instructor's, and every field on it is something they set
on the export screen: which form, in what question order, with or without the
institution's logo, the course data, the duration, the instructions, the score
distribution.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal

from ..models import BlueprintRow, Form


class ExportError(RuntimeError):
    """The file could not be produced. No half-written paper is returned."""


# --- What the instructor chose -----------------------------------------------


@dataclass(frozen=True)
class ExportOptions:
    """Everything the export screen offers, with the defaults it opens on.

    Frozen: an options object is what a particular export *was*, so a later
    screen can say how a file was produced without wondering whether something
    mutated it in between.
    """

    class Order:
        ASSEMBLED = "assembled"
        BY_TOPIC = "by_topic"
        SHUFFLED = "shuffled"

    #: Printed on the cover as "Form A". Off for a single-form exam, where a
    #: form letter is a label for a distinction that does not exist.
    show_form_label: bool = True
    question_order: str = Order.ASSEMBLED
    #: Only read when `question_order` is `SHUFFLED`. Stored so the same export
    #: produces the same paper twice — an instructor who re-exports after fixing
    #: a typo must not get a differently ordered exam.
    shuffle_seed: int = 20260815
    institution: str = ""
    logo_path: str = ""
    show_course_data: bool = True
    show_duration: bool = True
    show_score_distribution: bool = True
    instructions: str = ""
    #: Printed under each question as "(3 marks)". Separate from the score
    #: distribution table, because a paper can carry per-question marks without
    #: a summary and vice versa.
    show_marks_per_question: bool = True

    @property
    def orders(self) -> tuple:
        return (self.Order.ASSEMBLED, self.Order.BY_TOPIC, self.Order.SHUFFLED)


ORDER_LABELS = {
    ExportOptions.Order.ASSEMBLED: "As assembled",
    ExportOptions.Order.BY_TOPIC: "Grouped by topic",
    ExportOptions.Order.SHUFFLED: "Shuffled",
}


# --- What a paper is ---------------------------------------------------------


@dataclass(frozen=True)
class DocumentQuestion:
    """One numbered question, as it appears on a paper.

    Carries no answer. `AnswerBlock` on the key document carries those, and it
    is a different object reached from a different builder.
    """

    number: int
    stem: str
    options: tuple = ()
    marks: Decimal = Decimal("0")
    topic: str = ""
    question_type: str = ""
    level: str = ""
    expected_minutes: Decimal = Decimal("0")
    source_ref: str = ""

    @property
    def type_label(self) -> str:
        return dict(BlueprintRow.QuestionType.choices).get(
            self.question_type, self.question_type
        )

    @property
    def level_label(self) -> str:
        return dict(BlueprintRow.Level.choices).get(self.level, self.level)

    @property
    def option_labels(self) -> list:
        """A, B, C, … beside each option — how a student names their answer."""
        return [
            (chr(ord("A") + index), str(option)) for index, option in enumerate(self.options)
        ]


@dataclass(frozen=True)
class AnswerBlock:
    """One question's key, in whichever of M6's three shapes it has.

    Flattened into lines here rather than in the renderer: how a numeric key is
    laid out for a marker is a decision about the document, and every format
    that ever renders this must lay it out the same way.
    """

    number: int
    stem: str
    kind: str
    answer: str = ""
    required_elements: tuple = ()
    steps: tuple = ()  # (text, marks)
    marks: Decimal = Decimal("0")
    explanation: str = ""
    source_ref: str = ""
    mark_sum_ok: bool | None = None

    @property
    def needs_mark_review(self) -> bool:
        return self.mark_sum_ok is False


@dataclass(frozen=True)
class Cover:
    """The head of either document."""

    institution: str = ""
    course_code: str = ""
    course_name: str = ""
    exam_title: str = ""
    form_label: str = ""
    duration_minutes: int | None = None
    total_marks: Decimal = Decimal("0")
    question_count: int = 0
    instructions: str = ""
    logo_path: str = ""
    #: (topic, questions, marks, percent) — the score distribution, already
    #: totalled. A renderer prints it; it never computes it.
    distribution: tuple = ()

    @property
    def title(self) -> str:
        """The exam's own name. The form label is *not* joined in here.

        "Form A" is a phrase, and which language that phrase is in belongs to
        whoever is rendering the paper — see `export_pdf.CHROME`. A document
        model that hard-coded the English word would force every future format
        to print it too.
        """
        return self.exam_title


@dataclass(frozen=True)
class ExamDocument:
    """A student's paper. Has no answers on it, by construction."""

    cover: Cover
    questions: tuple = ()
    kind: str = "exam"
    language: str = "en"


@dataclass(frozen=True)
class AnswerKeyDocument:
    """A marker's copy: the typed keys, in question order."""

    cover: Cover
    answers: tuple = ()
    kind: str = "key"
    language: str = "en"


# --- Building the documents --------------------------------------------------


def _ordered_entries(form: Form, options: ExportOptions) -> list:
    entries = list(
        form.entries.select_related(
            "question", "blueprint_row", "blueprint_row__topic", "question__blueprint_row"
        ).order_by("position", "pk")
    )
    if options.question_order == ExportOptions.Order.BY_TOPIC:
        entries.sort(key=lambda entry: (_topic_of(entry), entry.position))
    elif options.question_order == ExportOptions.Order.SHUFFLED:
        # Seeded, so the same export is the same paper. An instructor who
        # re-exports after fixing a typo must not have to re-check the order.
        random.Random(options.shuffle_seed).shuffle(entries)
    return entries


def _topic_of(entry) -> str:
    row = entry.blueprint_row or entry.question.blueprint_row
    return row.topic.name if row and row.topic_id else ""


def _level_of(entry) -> str:
    row = entry.blueprint_row or entry.question.blueprint_row
    return row.level if row else ""


def _distribution(entries, total: Decimal) -> tuple:
    """Marks per topic, as counts, marks and share of the paper."""
    tally: dict = {}
    for entry in entries:
        topic = _topic_of(entry) or "—"
        count, marks = tally.get(topic, (0, Decimal("0")))
        tally[topic] = (count + 1, marks + Decimal(entry.marks or 0))
    rows = []
    for topic, (count, marks) in tally.items():
        percent = (marks / total * 100).quantize(Decimal("0.1")) if total else Decimal("0")
        rows.append((topic, count, marks, percent))
    return tuple(rows)


def _cover(form: Form, options: ExportOptions, entries) -> Cover:
    exam = form.exam
    course = exam.course
    total = sum((Decimal(entry.marks or 0) for entry in entries), Decimal("0"))
    return Cover(
        institution=options.institution,
        course_code=course.code if options.show_course_data else "",
        course_name=course.name if options.show_course_data else "",
        exam_title=exam.display_title,
        form_label=form.label if options.show_form_label else "",
        duration_minutes=exam.duration_minutes if options.show_duration else None,
        total_marks=total,
        question_count=len(entries),
        instructions=options.instructions,
        logo_path=options.logo_path,
        distribution=_distribution(entries, total) if options.show_score_distribution else (),
    )


def build_exam_document(form: Form, options: ExportOptions | None = None) -> ExamDocument:
    """The student's paper.

    This function does not read `Question.correct` or `Question.answer_key`, and
    that is the guarantee, not a convention: the object it returns has nowhere to
    put an answer, so no renderer can print one.
    """
    options = options or ExportOptions()
    entries = _ordered_entries(form, options)
    questions = tuple(
        DocumentQuestion(
            number=number,
            stem=entry.question.stem,
            options=tuple(str(option) for option in (entry.question.options or ())),
            marks=Decimal(entry.marks or 0) if options.show_marks_per_question else Decimal("0"),
            topic=_topic_of(entry),
            question_type=entry.question.question_type,
            level=_level_of(entry),
            expected_minutes=Decimal(entry.expected_minutes or 0),
            source_ref=entry.question.source_ref,
        )
        for number, entry in enumerate(entries, start=1)
    )
    return ExamDocument(
        cover=_cover(form, options, entries),
        questions=questions,
        language=form.exam.language,
    )


def _answer_block(number: int, entry) -> AnswerBlock:
    """One question's key, read through M6's typed formats.

    Falls back to the `correct` column when the stored key cannot be parsed: a
    marker needs *something*, and a key that fails validation is a reason to
    print the plain answer, not a reason to print nothing.
    """
    from agents.answer_key import NumericKey, ObjectiveKey, ShortAnswerKey

    question = entry.question
    marks = Decimal(entry.marks or 0)
    common = dict(
        number=number,
        stem=question.stem,
        marks=marks,
        explanation=question.explanation,
        source_ref=question.source_ref,
        mark_sum_ok=question.mark_sum_ok,
    )
    try:
        key = question.key
    except Exception:  # noqa: BLE001 — a bad key must not stop the marker's copy
        key = None

    if isinstance(key, ObjectiveKey):
        return AnswerBlock(kind="objective", answer=key.answer, **common)
    if isinstance(key, ShortAnswerKey):
        return AnswerBlock(
            kind="short_answer",
            answer=key.model_answer,
            required_elements=tuple(key.required_elements),
            **common,
        )
    if isinstance(key, NumericKey):
        return AnswerBlock(
            kind="numeric",
            answer=key.final_answer,
            steps=tuple((step.text, Decimal(step.marks)) for step in key.steps),
            **common,
        )
    return AnswerBlock(kind="plain", answer=question.correct or "", **common)


def build_key_document(form: Form, options: ExportOptions | None = None) -> AnswerKeyDocument:
    """The marker's copy — same questions, same order, with the typed keys."""
    options = options or ExportOptions()
    entries = _ordered_entries(form, options)
    return AnswerKeyDocument(
        cover=_cover(form, options, entries),
        answers=tuple(
            _answer_block(number, entry) for number, entry in enumerate(entries, start=1)
        ),
        language=form.exam.language,
    )


# --- The seam ----------------------------------------------------------------


@dataclass(frozen=True)
class ExportedFile:
    """One produced file: its bytes, its name, and what it is."""

    filename: str
    content: bytes
    content_type: str
    kind: str

    @property
    def size(self) -> int:
        return len(self.content)


class Exporter(ABC):
    """One output format. PDF is the first; the rest are additions, not rewrites.

    A new format implements `render_exam` and `render_key` over the document
    objects above and registers itself. It never reads a `Form`, a `Question` or
    an answer-key model — everything it needs has already been decided, in one
    place, for every format.
    """

    name: str = ""
    extension: str = ""
    content_type: str = "application/octet-stream"
    label: str = ""

    @abstractmethod
    def render_exam(self, document: ExamDocument) -> bytes: ...

    @abstractmethod
    def render_key(self, document: AnswerKeyDocument) -> bytes: ...


EXPORTERS: dict = {}


def register_exporter(exporter: Exporter) -> Exporter:
    EXPORTERS[exporter.name] = exporter
    return exporter


def get_exporter(name: str = "pdf") -> Exporter:
    try:
        return EXPORTERS[name]
    except KeyError:
        raise ExportError(
            f"إحكام does not export {name!r} yet. Available: "
            f"{', '.join(sorted(EXPORTERS)) or 'nothing'}."
        ) from None


def available_formats() -> list:
    return sorted(EXPORTERS.values(), key=lambda exporter: exporter.name)


def _slug(text: str) -> str:
    from django.utils.text import slugify

    return slugify(text, allow_unicode=False) or "exam"


def export_form(
    form: Form, *, options: ExportOptions | None = None, fmt: str = "pdf"
) -> list[ExportedFile]:
    """Both files for one form: the paper, then the key. Always both, always two.

    Returned rather than written: the view streams them, the tests read them,
    and nothing in this layer decides where a file belongs on disk.
    """
    options = options or ExportOptions()
    exporter = get_exporter(fmt)
    exam_document = build_exam_document(form, options)
    key_document = build_key_document(form, options)
    base = _slug(f"{form.exam.course.code}-{form.exam.display_title}")
    label = f"-form-{form.label.lower()}" if form.label else ""
    return [
        ExportedFile(
            filename=f"{base}{label}.{exporter.extension}",
            content=exporter.render_exam(exam_document),
            content_type=exporter.content_type,
            kind="exam",
        ),
        ExportedFile(
            filename=f"{base}{label}-answer-key.{exporter.extension}",
            content=exporter.render_key(key_document),
            content_type=exporter.content_type,
            kind="key",
        ),
    ]


def _register_builtin_exporters() -> None:
    """Import the shipped exporters for their registration side effect."""
    from . import export_pdf  # noqa: F401


_register_builtin_exporters()


__all__ = [
    "EXPORTERS",
    "ORDER_LABELS",
    "AnswerBlock",
    "AnswerKeyDocument",
    "Cover",
    "DocumentQuestion",
    "ExamDocument",
    "ExportError",
    "ExportOptions",
    "ExportedFile",
    "Exporter",
    "available_formats",
    "build_exam_document",
    "build_key_document",
    "export_form",
    "get_exporter",
    "register_exporter",
]
