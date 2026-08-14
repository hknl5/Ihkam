"""Blueprint arithmetic: pure Python, no LLM, no exceptions to that (M4).

Everything here can be computed exactly — sums, shares, rounding — so nothing
here is asked of a model. That is the hard rule from §1 of the build plan, and
this module is where it is most tempting to break and least excusable to.

Two things live here:

* **The validators.** Each one answers a different question about the same
  rows, and each returns *specific, fixable* issues rather than a boolean: which
  topic, what was expected, what is actually there. "Invalid blueprint" tells an
  instructor nothing; "Recursion is weighted 30% of a 40-mark exam, which is 12
  marks, but its rows carry 10" tells them what to type.
* **`auto_build`.** The first draft the instructor then edits: equal weight per
  eligible topic. It is written so that what it produces passes the validators
  above on creation — a proposal that arrives already flagged would teach the
  instructor to ignore the flags.

The validators run against `RowSpec`, not against saved rows, so the editor can
check what the instructor is *typing* without writing any of it to the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from courses.models import Topic

from ..models import Blueprint, BlueprintRow, Exam

CENT = Decimal("0.01")
HUNDRED = Decimal("100")

#: How far a topic's marks may sit from its weight's exact share before it is a
#: real mismatch rather than rounding. Marks are whole numbers in practice and a
#: weight is a percentage to two places, so an honest row lands within half a
#: mark; a mistyped one lands a whole mark or more away.
MARK_TOLERANCE = Decimal("0.5")


def _q(value) -> Decimal:
    return Decimal(value or 0).quantize(CENT, rounding=ROUND_HALF_UP)


def _plain(value: Decimal) -> str:
    """A Decimal as an instructor would write it: 12, 12.5 — never 12.00.

    Whole values are formatted from the integral value rather than
    `normalize()`, which renders 100 as "1E+2" — a message no instructor should
    have to read.
    """
    value = _q(value)
    if value == value.to_integral_value():
        return f"{value.to_integral_value():f}"
    return f"{value.normalize():f}"


# --- What gets validated -----------------------------------------------------


@dataclass(frozen=True)
class RowSpec:
    """One row's numbers, detached from the database.

    Built either from a saved `BlueprintRow` or straight from what the editor
    posted, so a row being typed is checked by exactly the same code as a row
    that was saved an hour ago.
    """

    topic_id: int | None
    topic_name: str
    count: int
    marks: Decimal
    weight_percent: Decimal
    question_type: str = BlueprintRow.QuestionType.MCQ
    level: str = BlueprintRow.Level.MEDIUM
    #: Whether this topic — or its chapter — is marked "not taught in lectures".
    #: Carried on the row because the instructor can exclude a topic *after*
    #: building the blueprint, and the row must then be caught rather than
    #: quietly generating from material that was ruled out.
    topic_excluded: bool = False

    @property
    def key(self):
        """What "the same topic" means when aggregating. Falls back to the name
        for a row whose topic was never chosen, so two blank rows do not merge."""
        return self.topic_id if self.topic_id is not None else f"name:{self.topic_name}"

    @classmethod
    def from_row(cls, row: BlueprintRow) -> "RowSpec":
        topic = row.topic
        return cls(
            topic_id=topic.pk,
            topic_name=topic.name,
            count=row.count,
            marks=_q(row.marks),
            weight_percent=_q(row.weight_percent),
            question_type=row.question_type,
            level=row.level,
            topic_excluded=topic.excluded or bool(topic.parent and topic.parent.excluded),
        )


@dataclass(frozen=True)
class Issue:
    """One thing that is wrong, said in a way that can be acted on.

    `code` is for the tests and the UI; `message` is for the instructor. The
    topic is carried separately so the editor can highlight the row.
    """

    code: str
    message: str
    topic_id: int | None = None
    topic_name: str = ""


@dataclass
class Report:
    """The result of checking one blueprint: totals, and everything wrong."""

    issues: list[Issue] = field(default_factory=list)
    total_count: int = 0
    total_marks: Decimal = Decimal("0")
    total_weight: Decimal = Decimal("0")
    expected_count: int = 0
    expected_marks: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def has(self, code: str) -> bool:
        return code in self.codes

    def _tone(self, ok: bool) -> str:
        """Maps to the design system's `.status--*` modifiers (§5)."""
        return "ok" if ok else "danger"

    @property
    def count_tone(self) -> str:
        return self._tone(not self.has("count_mismatch"))

    @property
    def marks_tone(self) -> str:
        return self._tone(not self.has("score_mismatch"))

    @property
    def weight_tone(self) -> str:
        return self._tone(not self.has("weight_mismatch"))

    @property
    def summary(self) -> str:
        if self.is_valid:
            return "This blueprint adds up. It is ready to be filled with questions."
        n = len(self.issues)
        return f"{n} thing{'s' if n != 1 else ''} to fix before questions can be generated."


# --- The checks --------------------------------------------------------------
#
# One function per error class, each independently callable and independently
# tested. They take the same shape so `validate` is only ever a list of calls.


def check_rows_exist(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """An empty blueprint is not a valid plan for a non-empty exam."""
    if rows:
        return []
    return [
        Issue(
            "no_rows",
            f"This blueprint has no rows, but the exam needs {exam.question_count} "
            f"questions worth {exam.total_score} marks. Build from your topics to start.",
        )
    ]


def check_question_count(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """The rows must plan exactly as many questions as the exam has."""
    total = sum(row.count for row in rows)
    if total == exam.question_count:
        return []
    difference = abs(total - exam.question_count)
    direction = "too many" if total > exam.question_count else "short"
    return [
        Issue(
            "count_mismatch",
            f"The rows plan {total} question{'s' if total != 1 else ''}, but this exam "
            f"is set to {exam.question_count}. That is {difference} "
            f"{direction} — change a row's count, or the exam's.",
        )
    ]


def check_total_score(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """The marks on the rows must be exactly the marks the exam is out of."""
    total = _q(sum((row.marks for row in rows), Decimal("0")))
    expected = Decimal(exam.total_score)
    if total == expected:
        return []
    difference = _plain(abs(total - expected))
    direction = "over" if total > expected else "unassigned"
    return [
        Issue(
            "score_mismatch",
            f"The rows carry {_plain(total)} marks, but this exam is out of "
            f"{exam.total_score}. {difference} mark(s) {direction}.",
        )
    ]


def check_weights(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """Topic weights must add up to 100% — no over, no under.

    Summed across rows, not topics: a topic split over an MCQ row and a
    short-answer row carries part of its weight on each, and the exam is fully
    planned only when every part of every topic adds to the whole.
    """
    total = _q(sum((row.weight_percent for row in rows), Decimal("0")))
    if total == HUNDRED:
        return []
    difference = _plain(abs(total - HUNDRED))
    direction = "too much" if total > HUNDRED else "missing"
    return [
        Issue(
            "weight_mismatch",
            f"The topic weights add up to {_plain(total)}%, not 100% — {difference}% "
            f"{direction}.",
        )
    ]


def check_topic_has_questions(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """A topic that is weighted but has no questions is a silent hole.

    It reads as covered on the blueprint and produces nothing on the paper,
    which is the failure this check exists for.
    """
    issues = []
    for spec, total_count, total_weight, _marks in _by_topic(rows):
        if total_count == 0 and total_weight > 0:
            issues.append(
                Issue(
                    "topic_without_questions",
                    f"“{spec.topic_name}” is weighted {_plain(total_weight)}% but has no "
                    f"questions. Give it at least one, or set its weight to 0% and "
                    f"redistribute it.",
                    topic_id=spec.topic_id,
                    topic_name=spec.topic_name,
                )
            )
    return issues


def check_marks_match_weights(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """Each topic's marks must be the share its weight claims.

    This is the check that catches a blueprint which sums perfectly and still
    lies: 100% of weight and the right total marks, distributed so a 30% topic
    carries 10 marks out of 40 instead of 12.
    """
    issues = []
    for spec, _count, total_weight, total_marks in _by_topic(rows):
        expected = _q(total_weight / HUNDRED * Decimal(exam.total_score))
        if abs(total_marks - expected) <= MARK_TOLERANCE:
            continue
        issues.append(
            Issue(
                "marks_weight_mismatch",
                f"“{spec.topic_name}” is weighted {_plain(total_weight)}% of a "
                f"{exam.total_score}-mark exam, which is {_plain(expected)} marks, but its "
                f"rows carry {_plain(total_marks)}.",
                topic_id=spec.topic_id,
                topic_name=spec.topic_name,
            )
        )
    return issues


def check_topics_are_usable(rows: list[RowSpec], *, exam: Exam) -> list[Issue]:
    """No row may sit on a topic the instructor said is not taught.

    Auto-build never picks one, but an instructor can exclude a topic *after*
    building the blueprint. Without this the row would survive, and retrieval
    would hand back nothing for it — the failure would surface in M5 as an
    unexplained empty question rather than here as a sentence.
    """
    issues = []
    for spec, _count, _weight, _marks in _by_topic(rows):
        if spec.topic_excluded:
            issues.append(
                Issue(
                    "excluded_topic",
                    f"“{spec.topic_name}” is marked not taught in lectures, so it cannot "
                    f"carry questions. Remove this row, or put the topic back in the "
                    f"syllabus.",
                    topic_id=spec.topic_id,
                    topic_name=spec.topic_name,
                )
            )
    return issues


CHECKS = (
    check_rows_exist,
    check_question_count,
    check_total_score,
    check_weights,
    check_topic_has_questions,
    check_marks_match_weights,
    check_topics_are_usable,
)


def _by_topic(rows: list[RowSpec]):
    """Rows folded to one entry per topic, in first-seen order.

    Yields `(first_row, total_count, total_weight, total_marks)`.
    """
    order: list = []
    seen: dict = {}
    for row in rows:
        if row.key not in seen:
            seen[row.key] = [row, 0, Decimal("0"), Decimal("0")]
            order.append(row.key)
        entry = seen[row.key]
        entry[1] += row.count
        entry[2] += _q(row.weight_percent)
        entry[3] += _q(row.marks)
    for key in order:
        spec, count, weight, marks = seen[key]
        yield spec, count, _q(weight), _q(marks)


def validate(rows: list[RowSpec], *, exam: Exam) -> Report:
    """Every check, against one exam's specification.

    All checks run — the instructor gets the whole list, not the first thing
    that failed, because fixing one number at a time through five reloads is
    how a validation screen becomes something people work around.
    """
    report = Report(
        total_count=sum(row.count for row in rows),
        total_marks=_q(sum((row.marks for row in rows), Decimal("0"))),
        total_weight=_q(sum((row.weight_percent for row in rows), Decimal("0"))),
        expected_count=exam.question_count,
        expected_marks=exam.total_score,
    )
    for check in CHECKS:
        report.issues.extend(check(rows, exam=exam))
    # An empty blueprint fails everything; saying so seven times is noise.
    if report.has("no_rows"):
        report.issues = [issue for issue in report.issues if issue.code == "no_rows"]
    return report


def validate_blueprint(blueprint: Blueprint) -> Report:
    """Validate what is actually saved."""
    rows = list(blueprint.rows.select_related("topic", "topic__parent"))
    return validate([RowSpec.from_row(row) for row in rows], exam=blueprint.exam)


# --- Auto-build --------------------------------------------------------------


def eligible_topics(course) -> list[Topic]:
    """The topics a blueprint may draw on, finest granularity first.

    Two rules, both the instructor's:

    * **Nothing excluded, and nothing under something excluded.** A sub-topic of
      a chapter marked "not taught" is out even if the sub-topic itself was
      never touched — the same inheritance `ChunkQuerySet.usable` applies to
      passages, applied here to rows, so the blueprint cannot plan a question
      that retrieval would then refuse to find material for.
    * **A chapter that has usable sub-topics is represented by them**, not by
      itself as well. Counting both would weight that chapter twice and spread
      the same material over two rows.
    """
    topics = list(course.topics.select_related("parent").order_by("position", "pk"))
    usable = [t for t in topics if not t.excluded and not (t.parent and t.parent.excluded)]
    chapters_with_children = {t.parent_id for t in usable if t.parent_id}
    return [t for t in usable if t.pk not in chapters_with_children]


def _largest_remainder(total: int, shares: int) -> list[int]:
    """Split `total` into `shares` whole parts that sum to exactly `total`.

    The remainder goes to the earliest parts one at a time, so the difference
    between the largest and smallest part is never more than one and the sum is
    exact. Rounding each share independently would leave the total short, which
    is precisely the error the blueprint is meant to catch — an auto-build that
    arrives already failing its own validation is worse than none.
    """
    if shares <= 0:
        return []
    base, remainder = divmod(total, shares)
    return [base + (1 if i < remainder else 0) for i in range(shares)]


def auto_build(exam: Exam, *, topics: list[Topic] | None = None) -> Blueprint:
    """Build (or rebuild) the first draft of `exam`'s blueprint: equal weight.

    Equal weight per topic is a starting point, not a recommendation — the
    instructor knows which chapter carried three weeks of lectures and this
    module cannot. What it guarantees is that the draft is *arithmetically
    sound*: counts and marks sum exactly, weights sum to 100%, and every topic
    that is weighted carries at least one question.

    Weights are derived from the marks rather than set to a flat 1/n, so the two
    can never disagree at birth: a 40-mark exam over 3 topics is 13/13/14 marks,
    which is 32.5% / 32.5% / 35% — not "33.33% each", which would claim a share
    the marks do not match.
    """
    topics = eligible_topics(exam.course) if topics is None else list(topics)

    blueprint, _ = Blueprint.objects.get_or_create(exam=exam)
    blueprint.rows.all().delete()
    blueprint.is_auto_built = True
    blueprint.save(update_fields=["is_auto_built", "updated_at"])

    if not topics:
        return blueprint

    # Never plan fewer questions than there are topics: a topic with zero
    # questions is an error by `check_topic_has_questions`, so when the exam has
    # fewer questions than topics only the first `question_count` topics get a
    # row. The rest are left out honestly rather than carried at zero.
    used = topics[: exam.question_count] if exam.question_count else []
    if not used:
        return blueprint

    counts = _largest_remainder(exam.question_count, len(used))
    marks = _largest_remainder(exam.total_score, len(used))
    weights = _weights_from_marks(marks, exam.total_score)

    BlueprintRow.objects.bulk_create(
        [
            BlueprintRow(
                blueprint=blueprint,
                topic=topic,
                question_type=BlueprintRow.QuestionType.MCQ,
                level=BlueprintRow.Level.MEDIUM,
                count=count,
                marks=Decimal(mark),
                weight_percent=weight,
                position=index,
            )
            for index, (topic, count, mark, weight) in enumerate(
                zip(used, counts, marks, weights, strict=True)
            )
        ]
    )
    return blueprint


def _weights_from_marks(marks: list[int], total_score: int) -> list[Decimal]:
    """Each mark share as a percentage, adjusted so the percentages sum to 100.

    Two-place percentages of a repeating share (a third of 30 marks) cannot all
    be exact, so the residue is added to the first row. It is at most 0.01% —
    far inside `MARK_TOLERANCE` — and it means the weight column reads 100%
    rather than 99.99%.
    """
    if not total_score:
        return [Decimal("0")] * len(marks)
    weights = [_q(Decimal(mark) / Decimal(total_score) * HUNDRED) for mark in marks]
    residue = HUNDRED - sum(weights)
    if residue:
        weights[0] = _q(weights[0] + residue)
    return weights
