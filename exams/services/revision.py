"""Regenerate, make easier, make harder, clarify — the instructor's own asks (M11).

Four buttons on the review card, one path: the *existing* 2A → 3A path. The
instructor's request becomes a note, the note is handed to Agent 2A exactly the
way M8 hands back Agent 3A's rejection notes, and the candidate that comes back
is reviewed by Agent 3A before it is allowed anywhere near the paper. There is
no second generation prompt and no "quick" route that skips review — a question
an instructor asked for is held to the same bar as one the loop wrote, because
it ends up on the same paper.

Four decisions shape this module:

* **The replacement lands in the same `Question` row.** A form points at
  questions (M9), so replacing the row rather than creating a new one keeps the
  paper intact: Form A's question 7 is still Form A's question 7, with its
  position, its marks and its expected time. Creating a new row would leave the
  instructor to re-place it by hand, and would leave the old one lying in the
  pool waiting to be picked by mistake.
* **A revision is reviewed, and a failed review is reported, not stored.** If
  Agent 3A rejects the replacement, the original question is left exactly as it
  was and the instructor is told why. The alternative — storing a candidate that
  failed review because a human asked for it — would put an unreviewed question
  on a paper, which is the one thing M8 exists to prevent.
* **The instructor's edit is discarded only on their word.** `revise` refuses to
  touch an instructor-edited question unless `confirmed=True`. The screen asks
  first ("this question was edited manually — regenerating will discard your
  edit"), and only then calls again with the confirmation. The automatic loop
  never gets this far: it skips locked questions in silence (see
  `agents.orchestrator.locked_questions`). Manual request warns then obeys;
  automatic cycle respects the lock without asking.
* **Every revision is written into the attempt log.** A replaced question is a
  fact about how this paper was built, and the log is where M8 already keeps
  those. The instructor's mode is recorded with it, so "why is this question not
  the one review passed?" has an answer.

Nothing here writes a prompt. The briefs live in `agents/prompts/revision.py`
beside the generation prompt they travel into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from django.db import transaction

from agents.prompts.revision import BRIEFS, LABELS, MODES, brief_for
from courses.services.retrieval import RetrievalError

from ..models import ItemRun, Question, QuestionAttempt

logger = logging.getLogger(__name__)


class RevisionError(RuntimeError):
    """The revision could not be completed. The question is left as it was."""


class RevisionNeedsConfirmation(RevisionError):
    """The question carries an instructor edit that this would discard.

    Its own class rather than a boolean, because the screen has to tell these
    two apart: a revision that *failed* is an error, and a revision that is
    *waiting for a word* is a question being asked. They read differently and
    they are recovered from differently.
    """

    def __init__(self, question):
        self.question = question
        super().__init__(
            "This question was edited manually — regenerating will discard your edit."
        )


@dataclass
class RevisionResult:
    """What one revision did, for the toast and for the tests."""

    question: Question
    mode: str
    previous_stem: str
    review: object = None
    replaced: bool = False
    message: str = ""

    @property
    def verb(self) -> str:
        """The past-tense label §5 requires the toast to use."""
        return LABELS.get(self.mode, ("Revised", "Revised"))[1]


def _item_for(question: Question, *, retrieve=None):
    """The generation item this question would be written from today.

    Built from the blueprint row, not from the question: the row is what the
    exam's arithmetic was validated against, and a revision must not be free to
    drift off the plan it was planned under.
    """
    from agents.analyze import plan_questions_for_row
    from agents.generate import item_for_row

    row = question.blueprint_row
    if row is None:
        raise RevisionError(
            "This question is no longer attached to a blueprint row, so there is "
            "nothing to write a replacement from. Re-plan the blueprint, or edit "
            "the question by hand."
        )
    try:
        planned = (
            plan_questions_for_row(row, retrieve=retrieve)
            if retrieve is not None
            else plan_questions_for_row(row)
        )
    except RetrievalError as exc:
        raise RevisionError(f"The passages could not be retrieved: {exc}") from exc

    passages = planned[0].passages if planned else ()
    if not passages:
        raise RevisionError(
            f"There are no passages for “{row.topic.name}” any more, so a "
            "replacement would not be grounded in this course."
        )
    return row, item_for_row(row, passages)


def revise(
    question: Question,
    mode: str,
    *,
    provider=None,
    confirmed: bool = False,
    retrieve=None,
    python_only: bool = False,
) -> RevisionResult:
    """Ask Agent 2A for a replacement, review it, and put it in this row.

    Raises `RevisionNeedsConfirmation` when the question carries an instructor
    edit and `confirmed` is False — nothing is generated in that case, so asking
    the question costs nothing.
    """
    from agents.generate import (
        GenerationCallFailed,
        QuestionGenerationError,
        UnsupportedQuestionType,
        generate_candidates,
    )
    from agents.review import QuestionReviewError, ReviewCallFailed, review_candidate

    if mode not in MODES:
        raise RevisionError(f"There is no “{mode}” revision.")
    if question.instructor_edited and not confirmed:
        raise RevisionNeedsConfirmation(question)

    row, item = _item_for(question, retrieve=retrieve)
    notes = brief_for(mode, stem=question.stem)

    try:
        run = generate_candidates(
            item,
            provider=provider,
            wanted=1,
            notes=notes,
            rejected_stems=[question.stem],
        )
    except (GenerationCallFailed, UnsupportedQuestionType, QuestionGenerationError) as exc:
        raise RevisionError(str(exc)) from exc

    if not run.candidates:
        raise RevisionError(
            "Nothing came back that was written from this course's material, so "
            "the question is unchanged."
        )
    candidate = run.candidates[0]

    try:
        review = review_candidate(candidate, item, provider=provider, python_only=python_only)
    except (ReviewCallFailed, QuestionReviewError) as exc:
        raise RevisionError(
            f"The replacement could not be reviewed, so it was not stored: {exc}"
        ) from exc

    previous_stem = question.stem
    if review.rejected:
        _log_attempt(question, row, stem=candidate.stem, review=review, mode=mode, stored=False)
        raise RevisionError(
            "Agent 3A rejected the replacement, so the question is unchanged. "
            + " ".join(review.notes)
        )

    with transaction.atomic():
        question.stem = candidate.stem
        question.question_type = candidate.question_type
        question.options = list(candidate.options)
        question.correct = candidate.correct
        question.explanation = candidate.explanation
        question.source_ref = candidate.source_ref[
            : Question._meta.get_field("source_ref").max_length
        ]
        question.source_chunk_id = candidate.passage.chunk_id if candidate.passage else None
        question.from_ocr = candidate.from_ocr
        question.answer_key = candidate.answer_key.as_dict() if candidate.answer_key else {}
        question.mark_sum_ok = candidate.mark_sum_ok
        # A regenerated question is the model's again, so the lock comes off and
        # the status goes back to undecided: approval was given to the question
        # that used to be here, and it does not transfer to a different one.
        question.instructor_edited = False
        question.edited_at = None
        question.status = Question.Status.CANDIDATE
        question.save()
        _log_attempt(question, row, stem=candidate.stem, review=review, mode=mode, stored=True)

    logger.info("Revised question %s (%s).", question.pk, mode)
    return RevisionResult(
        question=question,
        mode=mode,
        previous_stem=previous_stem,
        review=review,
        replaced=True,
        message=f"{LABELS[mode][1]}. The replacement passed review and is a candidate again.",
    )


def _log_attempt(question, row, *, stem, review, mode, stored: bool) -> None:
    """Record the revision in M8's attempt log, stored or not.

    A revision that was rejected is as much a fact about this paper as one that
    landed — and it is the only record that the instructor asked for something
    the material could not support.
    """
    run = ItemRun.objects.filter(exam=question.exam, blueprint_row=row).first()
    if run is None:
        return
    QuestionAttempt.objects.create(
        item_run=run,
        round=0,
        outcome=(
            QuestionAttempt.Outcome.PASSED
            if stored and review.passed
            else QuestionAttempt.Outcome.REJECTED
        ),
        stem=stem,
        question=question if stored else None,
        notes=[f"Instructor asked for: {BRIEFS[mode]}", *review.notes],
        failed_checks=list(review.failed_checks),
        model_checked=review.model_checked,
    )


__all__ = [
    "MODES",
    "RevisionError",
    "RevisionNeedsConfirmation",
    "RevisionResult",
    "revise",
]
