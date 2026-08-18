"""Saving an approved question to the bank, and pulling one back out (M12).

Two operations, deliberately in one module: they are the two halves of the same
copy, and the list of fields that must survive the round trip is written once
here rather than twice in two files that drift apart.

The rule both halves hold: **nothing is dropped**. A banked question carries the
stem, its typed key, the topic, the type, the level, the marks, the citation and
its provenance — everything the review screen showed the instructor when they
approved it. A copy that lost the answer key would be a question someone has to
write a key for later, which is the failure M6 exists to prevent; a copy that
lost the citation would be a question nobody can check against the material.
"""

from __future__ import annotations

import logging

from django.db import transaction

from exams.models import Question

from ..models import BankQuestion, BankUsage

logger = logging.getLogger(__name__)


class BankError(RuntimeError):
    """The question could not be banked or reused — said in the instructor's words."""


#: The fields copied verbatim in both directions. `topic`, the citation link and
#: the provenance are handled separately because their *names* differ between a
#: `Question` (which hangs off a blueprint row) and a `BankQuestion` (which
#: cannot, because the row belongs to an exam that may be deleted).
SHARED_FIELDS = (
    "stem",
    "question_type",
    "options",
    "correct",
    "explanation",
    "answer_key",
    "mark_sum_ok",
    "source_ref",
    "from_ocr",
)


def _page_of(question: Question) -> int | None:
    """The page the citation points at, when the chunk is still there to say."""
    return question.source_chunk.page if question.source_chunk_id else None


def save_to_bank(question: Question, *, provider=None) -> BankQuestion:
    """Copy one approved question into its course's bank.

    Refuses anything the instructor has not approved. That is the whole
    admission rule: the bank is a record of their decisions, so a candidate that
    merely passed Agent 3A has no business in it — review passing is إحكام's
    opinion, and this table is not for opinions.

    Idempotent per question: banking the same question twice returns the row
    that is already there rather than adding a second copy of it.

    The stem is embedded here, once, so the bank's search is a database query
    rather than a scan. A provider that cannot be reached does **not** fail the
    save: the question is banked without a vector and simply does not take part
    in ranking until it is re-embedded. Losing an instructor's approval because
    an embedding call timed out would be the worse trade by far.
    """
    if not question.is_approved:
        raise BankError(
            "Only a question you have approved can be saved to the bank. Approve it "
            "first — the bank is a record of your decisions, not of إحكام's."
        )

    existing = BankQuestion.objects.filter(origin_question=question).first()
    if existing is not None:
        return existing

    row = question.blueprint_row
    exam = question.exam
    with transaction.atomic():
        banked = BankQuestion.objects.create(
            course=exam.course,
            topic=row.topic if row else None,
            topic_name=row.topic.name if row else "",
            level=row.level if row else "",
            marks=row.marks_per_question if row else 0,
            source_page=_page_of(question),
            source_chunk=question.source_chunk,
            origin_question=question,
            origin_exam=exam,
            origin_exam_title=exam.display_title,
            **{name: getattr(question, name) for name in SHARED_FIELDS},
        )
        # The exam it was approved on *used* it. Recording that here is what
        # makes "which exams have used this" true from the first day rather
        # than counting only reuses, and it is what stops the sourcing pass
        # pulling a question back into the very exam it came from.
        BankUsage.objects.create(
            bank_question=banked,
            exam=exam,
            exam_title=exam.display_title,
            question=question,
            is_origin=True,
        )

    _embed(banked, provider=provider)
    logger.info("Banked question %s for %s.", banked.pk, exam.course.code)
    return banked


def _embed(banked: BankQuestion, *, provider=None) -> None:
    """Give the banked stem its vector, or leave it unindexed and say so."""
    from courses.services.retrieval import RetrievalError, embed_query

    try:
        banked.embedding = embed_query(banked.stem, provider=provider)
    except RetrievalError as exc:
        logger.warning(
            "Banked question %s was saved without an embedding, so it will not appear "
            "in bank search until it is re-embedded: %s",
            banked.pk,
            exc,
        )
        return
    banked.save(update_fields=["embedding", "updated_at"])


def reuse_in_exam(banked: BankQuestion, *, exam, row=None, position: int | None = None) -> Question:
    """Pull a banked question into an exam as a question of that exam.

    A *copy*, on purpose. The instructor may want to reword it for this paper,
    and an edit made here must not reach back and change what is in the bank —
    the bank holds the question as it was approved, and this exam holds the
    question as it is being asked.

    Two refusals, both about honesty rather than safety:

    * a bank question from another course is not reusable material, it is
      someone else's subject — reuse is same-course, per M12's success check;
    * the same banked question is not pulled into one exam twice, because a
      paper asking the same question in two slots is a defect, not a plan.

    The copy is stored **approved**: the instructor approved this question once
    already, and asking them to approve it a second time would be إحكام
    forgetting a decision it recorded. It is theirs to reject on the review
    screen like any other.
    """
    if banked.course_id != exam.course_id:
        raise BankError(
            f"That question belongs to {banked.course.code}'s bank, and this exam is "
            f"for {exam.course.code}. Questions are reused within a course, never "
            f"across two."
        )
    used = BankUsage.objects.filter(bank_question=banked, exam=exam).first()
    if used is not None:
        raise BankError(
            f"{exam.display_title} already carries this banked question. One question "
            f"cannot fill two slots on the same paper."
        )

    with transaction.atomic():
        copy = Question.objects.create(
            exam=exam,
            blueprint_row=row,
            source_chunk=banked.source_chunk,
            status=Question.Status.APPROVED,
            bank_source=banked,
            position=exam.questions.count() if position is None else position,
            **{name: getattr(banked, name) for name in SHARED_FIELDS},
        )
        BankUsage.objects.create(
            bank_question=banked,
            exam=exam,
            exam_title=exam.display_title,
            question=copy,
            is_origin=False,
        )
    logger.info(
        "Reused banked question %s in %s.", banked.pk, exam.display_title
    )
    return copy
