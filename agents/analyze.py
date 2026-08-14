"""Agent 1A — analyze & plan. The last step: grounding the plan (M4).

The blueprint says *what* the exam will ask about. This module answers the only
question left before generation can start: **which passages of the course each
planned question will be written from.** Its output — a validated blueprint plus
one reference-passage bundle per planned question — is exactly what Agent 2A
(M5) consumes, one item at a time.

It generates nothing. The only model call in the whole path is the embedding
inside `retrieve`, and even that is reached through the provider seam.

Three decisions worth stating:

* **An invalid blueprint produces no plan.** Retrieving passages for a paper
  whose marks do not add up would spend embedding calls on a plan that cannot be
  used, and would let a broken blueprint reach M5 wearing the same shape as a
  sound one.
* **One retrieval per row, not per question.** Five MCQs on one topic at one
  level are the same query five times; asking five times would cost five
  embedding calls to get the same passages back. Each planned question still
  carries its own bundle — Agent 2A is handed one item at a time and must never
  have to look at its siblings to know what it is grounded in.
* **A row that retrieves nothing is reported, not dropped.** It means the course
  material does not cover a topic the blueprint plans to examine. That is a real
  finding for the instructor and the one thing a silent empty bundle would hide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from courses.services.retrieval import Passage, retrieve as _retrieve
from exams.models import Blueprint, BlueprintRow
from exams.services.blueprint import Report, validate_blueprint

logger = logging.getLogger(__name__)


class BlueprintNotReady(RuntimeError):
    """The blueprint does not add up, so there is nothing to ground.

    Carries the report so the caller can show the instructor *what* to fix
    rather than only that something is wrong.
    """

    def __init__(self, report: Report):
        self.report = report
        super().__init__(report.summary)


def _label(choices, value: str) -> str:
    """A choice's human label, falling back to the stored value rather than
    raising — a plan is for reading, and an unknown value is worth showing."""
    try:
        return choices(value).label
    except ValueError:
        return value


@dataclass(frozen=True)
class PlannedQuestion:
    """One question that does not exist yet, and the passages it will come from.

    This is the unit Agent 2A receives in M5: everything needed to write one
    question and nothing about any other.
    """

    row_id: int | None
    topic_id: int
    topic_name: str
    question_type: str
    level: str
    marks: Decimal
    #: 1-based position within its row — "the 2nd of 5 MCQs on Recursion".
    index_in_row: int
    passages: tuple[Passage, ...] = ()

    @property
    def has_passages(self) -> bool:
        return bool(self.passages)

    @property
    def question_type_label(self) -> str:
        return _label(BlueprintRow.QuestionType, self.question_type)

    @property
    def level_label(self) -> str:
        return _label(BlueprintRow.Level, self.level)

    @property
    def page_refs(self) -> list[str]:
        """The citations this question may draw on, for the instructor's eyes."""
        return [passage.page_ref for passage in self.passages]


@dataclass
class ExamPlan:
    """The whole of Agent 1A's output, ready to hand to Agent 2A."""

    blueprint: Blueprint
    report: Report
    questions: list[PlannedQuestion] = field(default_factory=list)
    #: Topics the blueprint plans to examine that the material has no passage
    #: for, above the retrieval score floor. Named, so the instructor can add
    #: material or drop the row.
    topics_without_passages: list[str] = field(default_factory=list)

    @property
    def exam(self):
        return self.blueprint.exam

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def passage_count(self) -> int:
        return sum(len(question.passages) for question in self.questions)

    @property
    def is_fully_grounded(self) -> bool:
        return bool(self.questions) and not self.topics_without_passages

    def questions_for_row(self, row_id: int) -> list[PlannedQuestion]:
        return [question for question in self.questions if question.row_id == row_id]


def plan_questions_for_row(row: BlueprintRow, *, k=None, retrieve=_retrieve) -> list[PlannedQuestion]:
    """The planned questions of one row, each with its reference passages.

    `retrieve` is injectable so this can be exercised without a provider — the
    test suite spends no API calls, and neither does any caller that already has
    passages in hand.
    """
    passages = tuple(retrieve(row.blueprint.exam.course, row.topic, k=k))
    return [
        PlannedQuestion(
            row_id=row.pk,
            topic_id=row.topic_id,
            topic_name=row.topic.name,
            question_type=row.question_type,
            level=row.level,
            marks=row.marks_per_question,
            index_in_row=position + 1,
            passages=passages,
        )
        for position in range(row.count)
    ]


def build_exam_plan(blueprint: Blueprint, *, k=None, retrieve=_retrieve) -> ExamPlan:
    """Ground a validated blueprint: one passage bundle per planned question.

    Raises `BlueprintNotReady` if the blueprint does not pass its own
    validation. On success the plan holds exactly `sum(row.count)` questions —
    the same number the exam is specified for, because that is what validation
    just guaranteed.
    """
    report = validate_blueprint(blueprint)
    if not report.is_valid:
        raise BlueprintNotReady(report)

    plan = ExamPlan(blueprint=blueprint, report=report)
    rows = blueprint.rows.select_related("topic", "blueprint__exam__course")
    for row in rows:
        questions = plan_questions_for_row(row, k=k, retrieve=retrieve)
        plan.questions.extend(questions)
        if row.count and not questions[0].has_passages:
            # Named once, however many rows sit on the topic — the instructor
            # has one gap to close, not three.
            if row.topic.name not in plan.topics_without_passages:
                plan.topics_without_passages.append(row.topic.name)
            logger.warning(
                "Blueprint row %s (%s) retrieved no passages: the material does not "
                "cover a topic the exam plans to examine.",
                row.pk,
                row.topic.name,
            )
    return plan
