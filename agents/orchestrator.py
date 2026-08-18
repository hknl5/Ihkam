"""The closed correction loop — 1A → 2A → 3A, in one automated pass (M8).

The three agents already work. This module is what makes them a *system*: for
each blueprint item, generate a batch, review every candidate, and — when fewer
candidates passed than the row asks for — hand Agent 3A's own rejection notes
back to Agent 2A as the brief for the replacements. Repeat until the row is
satisfied or the retry cap stops it.

Five decisions shape it:

* **The loop fills a gap; it does not chase a candidate.** The obvious design
  regenerates each rejected question individually, and it is the wrong one: a
  row of 5 that got 6 candidates and passed 5 of them is *finished*, and the
  sixth rejection is not a problem to solve. The condition is therefore
  `passing < N`, counted per row, and a round asks only for the shortfall. Over-
  satisfying is fine — the surplus is kept as the alternatives an instructor
  rejects the first choice in favour of.
* **A rejection is a brief, not a retry.** `generate_candidates` is called again
  with the notes Agent 3A wrote, and those notes reach the model (see
  `agents.prompts.generate.format_rejections`). Without them a gap-fill round is
  the same dice rolled twice, and the quality story of إحكام is that a rejected
  question comes back *steered*.
* **Three gap-fill rounds, then a human.** A row the model cannot satisfy is
  usually a row whose material does not support the question the blueprint asks
  for — a multi-step numeric on a topic with only definitions. No number of
  retries fixes that, so the item stops and surfaces as **needs manual
  attention**, with its whole attempt log attached.
* **Every attempt is written down, including the ones that never became
  questions.** Passed, rejected, and dropped-before-review are all recorded with
  their notes. M5 drops an ungrounded candidate silently by design; here that
  drop has to be visible, or a row that came back short has no explanation.
  `attempts == approved + rejected + dropped`, per item, and the tests hold it.
* **An outage is not a rejection.** A call that never reached the model raises
  `GenerationCallFailed` / `ReviewCallFailed`, and the loop reports it as itself
  without spending one of the item's three rounds. The M2 rule, all the way up.

Nothing here decides that a question is on the paper. A candidate that passes
review is stored as a `Question` with status `candidate`, exactly as M5 stores
one — review passing is إحكام's opinion, and the instructor's approval is a
different act. What M8 guarantees is only this: nothing reaches M9 that was not
reviewed.

**M11 adds a sixth rule: an instructor-edited question is untouchable here.**
`locked_questions` are carried through a run unchanged and stay in the pool. The
loop does not regenerate over them, does not ask, and does not say anything —
the instructor did not request this run, so it has nothing to tell them. A
regeneration they *do* request goes through `exams/services/revision.py`, which
warns first and then obeys.

**M12 adds a seventh: the loop is told what is left to write.** An exam can be
sourced partly from the course's question bank, and those slots are filled
before this module runs. `run_plan(demand=...)` therefore asks each row only for
the questions that do not exist yet — down to none — and a reused question is
carried through exactly as an edited one is, for the same reason: it carries an
approval the instructor already gave.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.db import transaction

from agents.analyze import ExamPlan, build_exam_plan
from agents.generate import (
    Candidate,
    GenerationCallFailed,
    GenerationItem,
    GenerationRun,
    QuestionGenerationError,
    UnsupportedQuestionType,
    item_for_row,
    over_generated_count,
    save_candidates,
)
from agents.review import (
    ReviewCallFailed,
    ReviewResult,
    QuestionReviewError,
    review_candidate,
)
from exams.models import ItemRun, Question, QuestionAttempt

logger = logging.getLogger(__name__)


class OrchestrationError(RuntimeError):
    """The loop could not be run at all (a blueprint that does not add up)."""


#: How many times an item may ask for replacements before it stops and asks for
#: an instructor. Three, because the second round already carries the first
#: round's notes: if two briefed attempts and one blind one all fail, the fault
#: is very rarely the model's phrasing, and a fourth round is a fourth bill for
#: the same answer.
MAX_GAP_FILL_ROUNDS = 3

PASSED = ItemRun.Status.PASSED
NEEDS_ATTENTION = ItemRun.Status.NEEDS_ATTENTION

#: What the attempt log says about a question this run did not write. It is in
#: the log because it is in the pool, and it is in the pool because an
#: instructor edited it and إحكام does not overwrite that.
CARRIED_NOTE = "Carried unchanged: this question was edited by the instructor."

#: The same, for a question that came out of the course's bank (M12). It is in
#: the pool because the instructor approved it on an earlier exam, and this run
#: neither wrote it nor is entitled to write over it.
BANK_NOTE = "Carried unchanged: this question was reused from the course's bank."


# --- What one attempt was ----------------------------------------------------


@dataclass
class Attempt:
    """One candidate the loop produced, and what became of it.

    Held in memory as well as stored, because the probe and the tests read a run
    without a database, and because M9 reads the stored form.
    """

    round: int
    outcome: str
    stem: str
    candidate: Candidate | None = None
    review: ReviewResult | None = None
    question: Question | None = None

    @property
    def notes(self) -> list[str]:
        return list(self.review.notes) if self.review else []

    @property
    def failed_checks(self) -> list[str]:
        return list(self.review.failed_checks) if self.review else []

    @property
    def passed(self) -> bool:
        return self.outcome == QuestionAttempt.Outcome.PASSED

    @property
    def rejected(self) -> bool:
        return self.outcome == QuestionAttempt.Outcome.REJECTED

    @property
    def dropped(self) -> bool:
        return self.outcome == QuestionAttempt.Outcome.DROPPED


@dataclass
class ItemResult:
    """What the loop did about one blueprint item.

    `approved` is the pool: candidates that were reviewed and passed. There is
    no path into it that skips review — that is the guardrail M8 exists to hold.
    """

    item: GenerationItem
    attempts: list[Attempt] = field(default_factory=list)
    rounds: int = 0
    status: str = NEEDS_ATTENTION
    error: str = ""
    #: Set when the loop stopped because the provider was unreachable. Such a
    #: stop costs no round, and the item is worth re-running as it stands rather
    #: than investigating.
    unreachable: bool = False
    record: ItemRun | None = None
    #: Instructor-edited questions this run preserved rather than regenerated.
    #: Empty on every run that met no locked question, which is most of them.
    carried: list = field(default_factory=list)

    @property
    def required(self) -> int:
        return self.item.count

    @property
    def approved(self) -> list[Attempt]:
        return [a for a in self.attempts if a.passed]

    @property
    def rejected(self) -> list[Attempt]:
        return [a for a in self.attempts if a.rejected]

    @property
    def dropped(self) -> list[Attempt]:
        return [a for a in self.attempts if a.dropped]

    @property
    def approved_count(self) -> int:
        return len(self.approved)

    @property
    def questions(self) -> list[Question]:
        return [a.question for a in self.approved if a.question is not None]

    @property
    def gap_fill_rounds(self) -> int:
        return max(self.rounds - 1, 0)

    @property
    def shortfall(self) -> int:
        return max(self.required - self.approved_count, 0)

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    @property
    def needs_attention(self) -> bool:
        return self.status == NEEDS_ATTENTION

    @property
    def surplus(self) -> int:
        """Passing questions beyond what the row asked for — kept, not discarded."""
        return max(self.approved_count - self.required, 0)

    @property
    def counts_reconcile(self) -> bool:
        """Every attempt is exactly one of approved, rejected, dropped."""
        return len(self.attempts) == len(self.approved) + len(self.rejected) + len(self.dropped)

    @property
    def notes(self) -> list[str]:
        """Every rejection note this item produced, in order, deduplicated."""
        seen: list[str] = []
        for attempt in self.attempts:
            for note in attempt.notes:
                if note not in seen:
                    seen.append(note)
        return seen

    def notes_from_round(self, round_number: int) -> list[str]:
        seen: list[str] = []
        for attempt in self.attempts:
            if attempt.round != round_number:
                continue
            for note in attempt.notes:
                if note not in seen:
                    seen.append(note)
        return seen

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"{self.item.topic_name} · {self.approved_count}/{self.required} "
            f"in {self.rounds} round(s) · {self.status}"
        )


@dataclass
class OrchestrationRun:
    """The whole pass: what M9 consumes and what the later metrics read."""

    exam: object = None
    items: list[ItemResult] = field(default_factory=list)
    plan: ExamPlan | None = None
    #: Set when the pass stopped early because the provider was unreachable.
    #: The remaining items were not attempted — and are not "failed", which is
    #: why they have no `ItemResult` rather than a bad one.
    aborted: str = ""

    @property
    def approved_count(self) -> int:
        return sum(item.approved_count for item in self.items)

    @property
    def required(self) -> int:
        return sum(item.required for item in self.items)

    @property
    def attempt_count(self) -> int:
        return sum(len(item.attempts) for item in self.items)

    @property
    def rejected_count(self) -> int:
        return sum(len(item.rejected) for item in self.items)

    @property
    def dropped_count(self) -> int:
        return sum(len(item.dropped) for item in self.items)

    @property
    def items_needing_attention(self) -> list[ItemResult]:
        return [item for item in self.items if item.needs_attention]

    @property
    def passed_first_round(self) -> list[ItemResult]:
        """Items satisfied by the first batch — no gap-fill round needed."""
        return [item for item in self.items if item.passed and item.rounds <= 1]

    @property
    def is_complete(self) -> bool:
        return bool(self.items) and not self.items_needing_attention and not self.aborted

    @property
    def counts_reconcile(self) -> bool:
        return all(item.counts_reconcile for item in self.items)

    @property
    def questions(self) -> list[Question]:
        """The approved pool, in item order. Every one of these passed review."""
        return [question for item in self.items for question in item.questions]


# --- One item ----------------------------------------------------------------


def run_item(
    item: GenerationItem,
    *,
    provider=None,
    max_rounds: int = MAX_GAP_FILL_ROUNDS,
    python_only: bool = False,
    persist: bool = True,
    exam=None,
    row=None,
) -> ItemResult:
    """Generate, review, and fill the gap until the item is satisfied or capped.

    Round 1 is the over-generated batch M5 already produces. Every round after
    it is a gap-fill: it asks for as many alternatives as the row is short, and
    it carries the notes Agent 3A wrote about the attempts so far.

    `persist=False` runs the whole loop and stores nothing — what the probe uses
    for a dry run, and what lets the suite exercise the loop with no database.
    """
    result = ItemResult(item=item)

    if item.count <= 0:
        # Nothing left to write: this row was filled from the bank (M12). The
        # item is still recorded, and `persist_item` carries the reused
        # questions into it — a row with no `ItemRun` would be a row M9 cannot
        # see a pool for, which is a paper missing a section.
        result.status = PASSED
        logger.info(
            "Nothing to generate for '%s' — the bank filled the row.", item.topic_name
        )
        if persist:
            persist_item(result, exam=exam, row=row)
        return result

    try:
        _run_round(result, 1, provider=provider, python_only=python_only)
        while result.shortfall and result.rounds <= max_rounds:
            _run_round(result, result.rounds + 1, provider=provider, python_only=python_only)
    except (GenerationCallFailed, ReviewCallFailed) as exc:
        # Never reached the model. The round it happened in is not a round the
        # item spent: nothing was learned about the questions, and re-running
        # the item once the provider is back should find it exactly as it was.
        result.unreachable = True
        result.error = str(exc)
        logger.warning("Item '%s' stopped: %s", item.topic_name, exc)
    except (UnsupportedQuestionType, QuestionGenerationError, QuestionReviewError) as exc:
        # A real failure about this item: an unsupported type, no passages, or
        # an answer that could not be read twice. It is the instructor's to fix.
        result.error = str(exc)
        logger.warning("Item '%s' could not be generated: %s", item.topic_name, exc)

    result.status = PASSED if not result.shortfall and not result.error else NEEDS_ATTENTION
    if result.needs_attention:
        logger.warning(
            "Item '%s' needs manual attention: %s of %s passed review after %s round(s). %s",
            item.topic_name,
            result.approved_count,
            result.required,
            result.rounds,
            result.error or "The retry cap was reached.",
        )
    if persist:
        persist_item(result, exam=exam, row=row)
    return result


def _run_round(result: ItemResult, number: int, *, provider=None, python_only: bool = False):
    """One generation round and the review of everything it produced."""
    item = result.item
    if number == 1:
        wanted, notes, stems = item.candidates_wanted, (), ()
    else:
        # Only the shortfall, over-generated the same way a first batch is — a
        # row short by one still gets two tries at it. The notes are every
        # rejection this item has collected, not only the last round's: a
        # replacement that fixes the level and reintroduces the scope problem is
        # exactly what a one-round memory produces.
        wanted = over_generated_count(result.shortfall)
        notes = result.notes
        stems = [attempt.stem for attempt in result.rejected]

    logger.info(
        "Round %s for '%s': asking for %s candidate(s)%s",
        number,
        item.topic_name,
        wanted,
        f", briefed with {len(notes)} rejection note(s)" if notes else "",
    )
    run: GenerationRun = _generate(
        item, provider=provider, wanted=wanted, notes=notes, rejected_stems=stems
    )

    for stem in run.ungrounded:
        # Never reviewed because it never became a candidate. Recorded anyway:
        # this is the difference between "the row is short" and "the row is
        # short because three of six questions were not written from the course".
        result.attempts.append(
            Attempt(round=number, outcome=QuestionAttempt.Outcome.DROPPED, stem=stem)
        )

    for candidate in run.candidates:
        review = review_candidate(candidate, item, provider=provider, python_only=python_only)
        outcome = (
            QuestionAttempt.Outcome.PASSED
            if review.passed
            else QuestionAttempt.Outcome.REJECTED
        )
        result.attempts.append(
            Attempt(
                round=number,
                outcome=outcome,
                stem=candidate.stem,
                candidate=candidate,
                review=review,
            )
        )
        if review.rejected:
            logger.info(
                "Rejected on %s: %s",
                ", ".join(review.failed_checks),
                candidate.stem[:70],
            )

    # Counted only once the round finished. A round the provider cut short is
    # not a round the item spent — that is the guardrail, held here.
    result.rounds = number


def _generate(item, *, provider, wanted, notes, rejected_stems) -> GenerationRun:
    """The 2A call, imported lazily so the seam stays the only import path."""
    from agents.generate import generate_candidates

    return generate_candidates(
        item,
        provider=provider,
        wanted=wanted,
        notes=notes,
        rejected_stems=rejected_stems,
    )


# --- Storing the run ---------------------------------------------------------


def persist_item(result: ItemResult, *, exam=None, row=None) -> ItemRun:
    """Store the approved questions and the whole attempt log.

    Idempotent per item: the `ItemRun` is updated rather than duplicated, its
    previous attempts are replaced by this run's, and a passing candidate whose
    stem is already stored for this row reuses that `Question` instead of adding
    a second copy. Re-running an item therefore costs model calls, not a
    polluted pool.
    """
    from exams.models import Exam

    item = result.item
    if row is None and item.row_id is not None:
        row = _row(item.row_id)
    if exam is None:
        exam = row.blueprint.exam if row else Exam.objects.filter(pk=item.exam_id).first()
    if exam is None:
        raise OrchestrationError(
            "An orchestration run has to belong to an exam to be stored."
        )

    with transaction.atomic():
        record, _ = ItemRun.objects.update_or_create(
            exam=exam,
            blueprint_row=row,
            defaults=dict(
                topic_name=item.topic_name,
                question_type=item.question_type,
                level=item.level,
                required=result.required,
                approved_count=result.approved_count,
                rounds=result.rounds,
                status=result.status,
                error=result.error,
            ),
        )
        record.attempts.all().delete()
        _store_approved(result, exam=exam, row=row)
        QuestionAttempt.objects.bulk_create(
            [
                QuestionAttempt(
                    item_run=record,
                    round=attempt.round,
                    outcome=attempt.outcome,
                    stem=attempt.stem,
                    question=attempt.question,
                    notes=attempt.notes,
                    failed_checks=attempt.failed_checks,
                    model_checked=attempt.review.model_checked if attempt.review else False,
                )
                for attempt in result.attempts
            ]
        )
        # M11's hard rule, held at the one place that could break it. Carried
        # attempts are marked round 0 — they belong to no round of this run,
        # because this run did not produce them — so "how many rounds did this
        # item take" stays a true answer.
        carried = _carry_locked(result, exam=exam, row=row)
        if carried:
            QuestionAttempt.objects.bulk_create(
                [
                    QuestionAttempt(
                        item_run=record,
                        round=0,
                        outcome=QuestionAttempt.Outcome.PASSED,
                        stem=question.stem,
                        question=question,
                        notes=[BANK_NOTE if question.is_from_bank else CARRIED_NOTE],
                        failed_checks=[],
                        model_checked=False,
                    )
                    for question in carried
                ]
            )
            record.approved_count = record.approved_count + len(carried)
            record.save(update_fields=["approved_count", "updated_at"])
            logger.info(
                "Carried %s instructor-edited question(s) for '%s' — the loop left them alone.",
                len(carried),
                item.topic_name,
            )
        result.carried = carried
    result.record = record
    return record


def _row(row_id):
    from exams.models import BlueprintRow

    return BlueprintRow.objects.filter(pk=row_id).select_related("blueprint__exam").first()


def locked_questions(*, exam, row):
    """Questions on this row the loop must leave alone (M11's hard rule, M12's).

    The automatic loop must leave these exactly as they are. It never rewrote a
    stored question in the first place, but that alone was not enough: this
    module replaces an item's attempt log on every run, and M9 reads the pool
    *through* those attempts. An instructor-edited question whose stem no longer
    matches anything the model wrote would therefore have quietly dropped out of
    the pool on the next automatic pass — an edit lost by bookkeeping rather
    than by an overwrite, which is the same thing to the instructor.

    So they are carried: kept in the log, kept in the pool, and never touched.
    Silently, because the instructor did not ask for this run.

    M12 puts a second kind of question under the same protection: one pulled out
    of the course's bank. It carries an approval the instructor gave on an
    earlier exam, so regenerating over it would discard a decision for exactly
    the reason M11 forbids — the loop did not ask, and the instructor did not
    offer.
    """
    from django.db.models import Q

    if row is None or exam is None:
        return []
    return list(
        Question.objects.filter(exam=exam, blueprint_row=row)
        .filter(Q(instructor_edited=True) | Q(bank_source__isnull=False))
        .exclude(status=Question.Status.REJECTED)
        .order_by("position", "pk")
    )


def _carry_locked(result: ItemResult, *, exam, row) -> list[Question]:
    """The locked questions this run must preserve, minus any it re-produced."""
    already = {question.pk for question in result.questions if question is not None}
    return [
        question
        for question in locked_questions(exam=exam, row=row)
        if question.pk not in already
    ]


def _store_approved(result: ItemResult, *, exam, row) -> None:
    """Store the passing candidates as `Question` rows, without duplicating.

    Only attempts that passed review are stored. A rejected candidate is a fact
    in the log, not a row an instructor could stumble into approving.

    An instructor-edited question is never in `fresh` and never updated here:
    matching is by stem, and a question the instructor rewrote is matched by its
    own current stem or not at all. Either way this function only ever *creates*
    rows, so a locked question cannot be overwritten by it.
    """
    from agents.review import normalise_option

    existing = {
        normalise_option(question.stem): question
        for question in Question.objects.filter(exam=exam, blueprint_row=row)
    }

    fresh = []
    for attempt in result.approved:
        known = existing.get(normalise_option(attempt.stem))
        if known is not None:
            attempt.question = known
            continue
        fresh.append(attempt)

    if not fresh:
        return
    stored = save_candidates(
        GenerationRun(item=result.item, candidates=[a.candidate for a in fresh]),
        exam=exam,
        row=row,
    )
    for attempt, question in zip(fresh, stored, strict=True):
        attempt.question = question


# --- A whole blueprint -------------------------------------------------------


def items_for_plan(
    plan: ExamPlan, *, demand: dict[int, int] | None = None
) -> list[tuple[object, GenerationItem]]:
    """One generation item per blueprint row, carrying the row's passages.

    Agent 1A plans per *question*; Agent 2A is handed one *row* at a time and
    over-generates within it. The passages are the row's, so they are taken from
    its first planned question rather than retrieved a second time.

    `demand` overrides how many questions a row still needs *written* (M12).
    A row whose slots came out of the bank asks for fewer, or for none — the
    row still wants `row.count` questions, and some of them already exist.
    Anything not named in the mapping is generated in full, so a caller with no
    bank in play passes nothing and gets M8's behaviour exactly.
    """
    from dataclasses import replace

    pairs = []
    for row in plan.blueprint.rows.select_related("topic", "blueprint__exam__course"):
        planned = plan.questions_for_row(row.pk)
        passages = planned[0].passages if planned else ()
        item = item_for_row(row, passages)
        if demand is not None and row.pk in demand:
            item = replace(item, count=max(int(demand[row.pk]), 0))
        pairs.append((row, item))
    return pairs


def run_plan(
    plan: ExamPlan,
    *,
    provider=None,
    max_rounds: int = MAX_GAP_FILL_ROUNDS,
    python_only: bool = False,
    persist: bool = True,
    demand: dict[int, int] | None = None,
) -> OrchestrationRun:
    """Run the correction loop over every row of a grounded blueprint.

    Stops early if the provider becomes unreachable: with no calls completing,
    the remaining items would each report the same outage, and an item nobody
    attempted is more honest than one marked as having failed.
    """
    run = OrchestrationRun(exam=plan.exam, plan=plan)
    for row, item in items_for_plan(plan, demand=demand):
        result = run_item(
            item,
            provider=provider,
            max_rounds=max_rounds,
            python_only=python_only,
            persist=persist,
            exam=plan.exam,
            row=row,
        )
        run.items.append(result)
        if result.unreachable:
            run.aborted = result.error
            logger.warning(
                "Stopping the run after '%s': the provider is unreachable. "
                "%s of %s items were attempted.",
                item.topic_name,
                len(run.items),
                len(plan.blueprint.rows.all()),
            )
            break
    return run


def run_exam(
    exam,
    *,
    provider=None,
    max_rounds: int = MAX_GAP_FILL_ROUNDS,
    python_only: bool = False,
    persist: bool = True,
    k=None,
    retrieve=None,
    demand: dict[int, int] | None = None,
) -> OrchestrationRun:
    """Ground the exam's blueprint (1A), then run the loop over it (2A → 3A).

    The one entry point that goes from "an instructor has a blueprint" to "there
    is a reviewed pool of questions", which is what M9 starts from.
    """
    from agents.analyze import BlueprintNotReady

    if not exam.has_blueprint:
        raise OrchestrationError(
            f"{exam.display_title} has no blueprint, so there is nothing to generate from."
        )
    kwargs = {"k": k}
    if retrieve is not None:
        kwargs["retrieve"] = retrieve
    try:
        plan = build_exam_plan(exam.blueprint, **kwargs)
    except BlueprintNotReady as exc:
        raise OrchestrationError(
            f"The blueprint does not add up, so nothing was generated: {exc}"
        ) from exc
    return run_plan(
        plan,
        provider=provider,
        max_rounds=max_rounds,
        python_only=python_only,
        persist=persist,
        demand=demand,
    )


__all__ = [
    "BANK_NOTE",
    "CARRIED_NOTE",
    "MAX_GAP_FILL_ROUNDS",
    "NEEDS_ATTENTION",
    "PASSED",
    "Attempt",
    "ItemResult",
    "OrchestrationError",
    "OrchestrationRun",
    "items_for_plan",
    "persist_item",
    "run_exam",
    "run_item",
    "run_plan",
]
