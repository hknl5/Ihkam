"""Where each question of a new exam comes from — bank, or newly written (M12).

The instructor sets *one* ratio for the whole paper ("about 60% from the bank").
This module turns that sentence into a number per blueprint row, and it does it
in plain Python: which slots are filled from the bank is arithmetic over what
the bank actually holds, and no model is asked about it.

Four decisions:

* **One ratio for the exam, not one per row.** "60% bank" is a statement about
  the paper. Applying it row by row would round three-question rows to two and
  put the exam nowhere near 60% overall; it would also refuse to let a
  well-stocked topic carry for a thin one. So the share is taken over the exam's
  total demand and spread by largest remainder — the same rule
  `blueprint.auto_build` uses to split marks — then capped, row by row, by what
  the bank can actually supply.
* **A cap is not a failure; a cap is a report.** When a row's share cannot be
  filled, the unfillable slots are handed back to generation and the *reason* is
  written down naming the topic and both counts. M9's shortfall reporting set
  this rule: never emit a silently short exam, and never quietly substitute.
* **What the bank cannot cover, generation covers.** In `mix`, and in `bank`
  too, the remainder is generated. An exam is short only when the *material*
  cannot support it, which is M8's business and reported there — never because
  the bank ran out.
* **A question is drawn into one exam once.** `available_for_row` excludes
  anything already used in this exam, so re-running the sourcing pass after a
  partial run tops up rather than duplicating.

Nothing here writes until `fill_from_bank` is called, and `plan_sourcing` alone
is what the Generate screen shows *before* the instructor presses anything: the
split and its shortfalls are visible while they can still change the ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from exams.models import Exam

from ..models import BankQuestion, BankUsage
from .save import reuse_in_exam

logger = logging.getLogger(__name__)


def already_from_bank(exam, row) -> int:
    """Questions on this row that came out of the bank on an earlier press.

    Generate can be pressed twice. Without this the second press would draw the
    row's whole bank share again — an exam quietly draining the bank into one
    paper, which is the sort of silent behaviour M12 is supposed to be the
    opposite of. Rejected copies are not counted: the instructor threw those
    out, and the slot is open again.
    """
    from exams.models import Question

    return (
        Question.objects.filter(exam=exam, blueprint_row=row, bank_source__isnull=False)
        .exclude(status=Question.Status.REJECTED)
        .count()
    )


def available_for_row(exam, row):
    """The banked questions this row could be filled with, oldest first.

    Matched on the three things a blueprint row *is* — topic, question type,
    cognitive level — because a row is a promise about all three, and a question
    that matches two of them fills the slot with something the plan did not ask
    for. Same course by construction (`for_course`), and never a question this
    exam already carries.
    """
    used = BankUsage.objects.filter(exam=exam).values("bank_question_id")
    return (
        BankQuestion.objects.for_course(exam.course)
        .matching(topic=row.topic, question_type=row.question_type, level=row.level)
        .exclude(pk__in=used)
        .order_by("created_at", "pk")
    )


@dataclass
class RowSource:
    """How one blueprint row's questions are to be sourced.

    Three numbers, deliberately all kept: what the ratio *asked* of this row,
    what the bank *has* for it, and what was finally *assigned* to it after the
    exam-wide redistribution. Collapsing them into one would make the shortfall
    line lie — a row capped at 3 would report "3 needed, bank has 3".
    """

    row: object
    #: What the row asks for in total.
    required: int
    #: What the exam-wide ratio wanted from the bank here, before stock.
    asked: int
    #: How many matching questions the bank actually holds for this row.
    available: int
    #: What this row will really draw *now*, after capping and redistribution.
    assigned: int = 0
    #: Bank questions this row already carries from an earlier press.
    already: int = 0

    @property
    def topic_name(self) -> str:
        return self.row.topic.name

    @property
    def from_bank(self) -> int:
        """Bank questions this row will have once the pull is done."""
        return self.already + self.assigned

    @property
    def to_pull(self) -> int:
        """How many are actually pulled by this press."""
        return self.assigned

    @property
    def to_generate(self) -> int:
        """Everything the bank does not fill — including any shortfall."""
        return max(self.required - self.from_bank, 0)

    @property
    def short_by(self) -> int:
        """Bank slots asked of this row that the bank could not supply.

        Counted against everything the row can get from the bank — what it
        already holds plus what is still on the shelf.
        """
        return max(self.asked - (self.already + self.available), 0)

    @property
    def is_short(self) -> bool:
        return self.short_by > 0

    @property
    def shortfall_line(self) -> str:
        """The sentence the instructor reads. Names the topic and both counts."""
        return (
            f"Bank has {self.available} question{'s' if self.available != 1 else ''} for "
            f"{self.topic_name} ({self.row.get_question_type_display()} \u00b7 "
            f"{self.row.get_level_display()}), {self.asked} needed \u2014 "
            f"\u0625\u062d\u0643\u0627\u0645 will write the other {self.short_by}, or lower the bank share."
        )


@dataclass
class SourcingPlan:
    """The whole exam's split: what comes from the bank, what gets written.

    Read by the Generate screen before the run (free \u2014 it is database
    arithmetic) and by the run itself. The same object answers both, so what the
    instructor was shown is what happens.
    """

    exam: Exam
    rows: list[RowSource] = field(default_factory=list)
    #: The mode and share this plan was built from, kept so a plan can be read
    #: back without the exam beside it.
    mode: str = Exam.Sourcing.NEW
    share_percent: int = 0
    #: How many questions the ratio asked of the bank across the whole exam.
    target: int = 0

    @property
    def required(self) -> int:
        return sum(source.required for source in self.rows)

    @property
    def from_bank(self) -> int:
        return sum(source.from_bank for source in self.rows)

    @property
    def to_generate(self) -> int:
        return sum(source.to_generate for source in self.rows)

    @property
    def short_by(self) -> int:
        """What the bank could not supply of the ratio's ask, over the exam.

        Counted once, over the paper, because the ratio was set over the paper:
        a row capped at its stock while another row covers the difference is not
        a shortfall, it is the redistribution working.
        """
        return max(self.target - self.from_bank, 0)

    @property
    def is_short(self) -> bool:
        return self.short_by > 0

    @property
    def draws_on_bank(self) -> bool:
        return self.target > 0

    @property
    def shortfalls(self) -> list[str]:
        """Why the bank fell short, per row — empty unless it actually did.

        Silent when redistribution covered the gap: a report an instructor
        cannot act on is noise, and there is nothing to act on when the exam got
        the share it asked for.
        """
        if not self.is_short:
            return []
        return [source.shortfall_line for source in self.rows if source.is_short]

    @property
    def demand(self) -> dict[int, int]:
        """What generation is left to write, per blueprint row.

        The mapping the orchestrator runs on. A row of zero is a row the bank
        filled completely \u2014 it is kept in the mapping rather than dropped, so
        the loop still records an item for it.
        """
        return {source.row.pk: source.to_generate for source in self.rows}

    @property
    def actual_share(self) -> int:
        """The share of the paper the bank really supplies, as a whole percent."""
        if not self.required:
            return 0
        return round(100 * self.from_bank / self.required)

    @property
    def summary(self) -> str:
        if not self.draws_on_bank:
            return (
                f"All {self.required} question{'s' if self.required != 1 else ''} will be "
                f"written from your content."
            )
        note = (
            f"{self.from_bank} of {self.required} question"
            f"{'s' if self.required != 1 else ''} from the bank ({self.actual_share}%), "
            f"{self.to_generate} written from your content."
        )
        if self.is_short:
            note += (
                f" The bank was {self.short_by} short of the "
                f"{self.share_percent}% asked for."
            )
        return note


def largest_remainder(total: int, weights: list[int]) -> list[int]:
    """Split `total` over `weights` so the parts add back to exactly `total`.

    The same rule the blueprint uses on marks. A 60% bank share over rows of
    5, 3 and 2 is 6 = 3 + 2 + 1, never three roundings that come to 7.
    """
    weight_total = sum(weights)
    if total <= 0 or weight_total <= 0:
        return [0 for _ in weights]

    exact = [total * weight / weight_total for weight in weights]
    parts = [int(value) for value in exact]
    remainder = total - sum(parts)
    # Biggest fractional part first; ties go to the earlier row, so the split is
    # the same every time it is computed for the same blueprint.
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - parts[i]), i))
    for index in order[:remainder]:
        parts[index] += 1
    return [min(part, weight) for part, weight in zip(parts, weights, strict=True)]


def _share_of(total: int, percent: int) -> int:
    """`percent` of `total`, rounded half *up*.

    Python's `round` is banker's rounding, so half of a five-question exam
    would come back as two. An instructor who typed 50% and got 40% would be
    right to call that a bug, so the rounding is spelled out.
    """
    value = Decimal(total) * Decimal(percent) / Decimal(100)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def plan_sourcing(exam: Exam) -> SourcingPlan:
    """Decide, without writing anything, where each of this exam's questions comes from.

    Two passes over the rows. The first spreads the exam-wide share by largest
    remainder; the second caps each row at what the bank holds and hands the
    slots that fell off to the rows that still have stock \u2014 a well-supplied
    topic covering for a thin one is exactly what "one ratio for the whole exam"
    means, and refusing to do it would report a shortfall the bank does not
    actually have.
    """
    board = getattr(exam, "blueprint", None)
    if board is None:
        return SourcingPlan(exam=exam, mode=exam.sourcing, share_percent=exam.bank_share)

    rows = list(board.rows.select_related("topic"))
    counts = [row.count for row in rows]
    stock = [available_for_row(exam, row).count() for row in rows]
    held = [already_from_bank(exam, row) for row in rows]

    share = exam.bank_share
    target = _share_of(sum(counts), share)
    asked = largest_remainder(target, counts)

    # What is still to be pulled: the ask, less what the row already holds,
    # capped by the shelf.
    assigned = [
        min(max(asked[index] - held[index], 0), stock[index]) for index in range(len(rows))
    ]
    spare = sum(asked) - sum(held[index] + assigned[index] for index in range(len(rows)))
    for index in range(len(rows)):
        if spare <= 0:
            break
        headroom = (
            min(stock[index] + held[index], counts[index]) - held[index] - assigned[index]
        )
        if headroom > 0:
            moved = min(headroom, spare)
            assigned[index] += moved
            spare -= moved

    plan = SourcingPlan(
        exam=exam,
        mode=exam.sourcing,
        share_percent=share,
        target=target,
        rows=[
            RowSource(
                row=row,
                required=counts[index],
                asked=asked[index],
                available=stock[index],
                assigned=assigned[index],
                already=held[index],
            )
            for index, row in enumerate(rows)
        ],
    )
    if plan.is_short:
        logger.info(
            "Bank sourcing for %s is %s short of the %s%% asked for.",
            exam.display_title,
            plan.short_by,
            share,
        )
    return plan


def fill_from_bank(exam: Exam, plan: SourcingPlan | None = None) -> list:
    """Pull the planned bank questions into the exam. This one writes.

    Runs before generation, so the loop is asked only for what is genuinely
    left to write. Every pulled question arrives approved and locked against the
    loop (`Question.bank_source`), which is what stops the next generation pass
    treating a reused question as a slot it still owes.
    """
    plan = plan_sourcing(exam) if plan is None else plan
    pulled = []
    for source in plan.rows:
        if not source.to_pull:
            continue
        for banked in available_for_row(exam, source.row)[: source.to_pull]:
            pulled.append(reuse_in_exam(banked, exam=exam, row=source.row))
    if pulled:
        logger.info(
            "Pulled %s banked question(s) into %s.", len(pulled), exam.display_title
        )
    return pulled
