"""The question bank: an approved question, kept for the next exam (M12).

Everything before this milestone was one exam's work. A question was written
for an exam, reviewed for that exam, approved on that exam's screen, and died
with it. M12 is the milestone where an instructor's approval stops being
disposable: a question they said yes to is saved to their course's bank and is
sourcing material for every exam they build afterwards.

Three decisions shape this module:

* **A banked question is a copy, not a pointer.** `BankQuestion` carries the
  stem, the typed key, the topic name, the level, the marks and the citation on
  its own row rather than following a foreign key to the `Question` it came
  from. The origin exam is the instructor's to delete — a paper from two terms
  ago is clutter — and deleting it must not take the bank with it. So the
  foreign keys back to the exam side are all `SET_NULL`, and the bank row still
  reads as a whole question after everything it was born from is gone.
* **A bank belongs to a course, not to an instructor.** Reuse is *for the same
  course*: a question about binary search written from CS210's lectures is not
  material for a database course, even though the same person teaches both. The
  scope is enforced in the query, not in the screen — see
  `bank.services.sourcing.available_for_row`.
* **Usage is recorded, not counted.** "Which exams have used this" is a list of
  rows (`BankUsage`), not an integer that goes up: an instructor deciding
  whether a question is over-used needs to know *which* papers carried it, and
  a counter cannot be corrected when an exam is deleted. The count is derived.

The embedding is here for one reason: the bank's search is the same `embed()`
every other search in إحكام goes through (M2's rule — one embedding path, one
provider seam), so a banked stem is embedded when it is saved and searched with
cosine distance in the same database. There is no second search stack.
"""

from __future__ import annotations

from decimal import Decimal

from django.conf import settings
from django.db import models
from pgvector.django import VectorField

from courses.models import Course, Topic
from exams.models import BlueprintRow


class BankQuestionQuerySet(models.QuerySet):
    def for_course(self, course):
        """The only way the bank is ever read. Reuse is same-course, always."""
        return self.filter(course=course)

    def indexed(self):
        """Rows that can take part in a semantic search.

        A question saved while the provider was unreachable has no vector. It
        is a perfectly good bank question — browsable, pullable, complete — and
        it is simply not a candidate for ranking until it is re-embedded. The
        search screen says how many of those there are rather than pretending
        the result list is the whole bank.
        """
        return self.filter(embedding__isnull=False)

    def matching(self, *, topic=None, question_type="", level=""):
        """The bank filtered the way a blueprint row asks for questions."""
        rows = self
        if topic is not None:
            rows = rows.filter(topic=topic)
        if question_type:
            rows = rows.filter(question_type=question_type)
        if level:
            rows = rows.filter(level=level)
        return rows


class BankQuestion(models.Model):
    """One approved question, kept for reuse in a later exam of the same course.

    Written only by `bank.services.save.save_to_bank`, and only from a question
    the instructor approved: the bank is a record of decisions they made, so
    nothing a model merely passed can reach it.
    """

    course = models.ForeignKey(
        Course, on_delete=models.CASCADE, related_name="bank_questions"
    )
    #: The topic as a link *and* as text. The link is what a blueprint row
    #: matches on; the copy is what the card still says after a re-extraction
    #: of the syllabus has renamed or removed the topic.
    topic = models.ForeignKey(
        Topic, on_delete=models.SET_NULL, null=True, blank=True, related_name="bank_questions"
    )
    topic_name = models.CharField(max_length=200, blank=True)

    stem = models.TextField()
    question_type = models.CharField(
        max_length=16, choices=BlueprintRow.QuestionType.choices
    )
    level = models.CharField(max_length=12, choices=BlueprintRow.Level.choices)
    #: What the question was worth on the exam it came from. A later blueprint
    #: may price it differently — the row's arithmetic wins there — but what it
    #: carried when it was approved is part of what was approved.
    marks = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal("0"))

    options = models.JSONField(default=list, blank=True)
    correct = models.TextField()
    explanation = models.TextField(blank=True)
    #: The typed key (M6), copied whole. A bank question without its key would
    #: be half a question, and re-deriving one later is exactly the "key written
    #: afterwards" that M6 exists to forbid.
    answer_key = models.JSONField(default=dict, blank=True)
    mark_sum_ok = models.BooleanField(null=True, blank=True)

    #: The citation, kept as the human sentence and — while the chunk survives —
    #: as the link that lets a screen show the passage beside the question.
    source_ref = models.CharField(max_length=300, blank=True)
    source_page = models.PositiveIntegerField(null=True, blank=True)
    source_chunk = models.ForeignKey(
        "courses.Chunk",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="bank_questions",
    )
    from_ocr = models.BooleanField(default=False)

    #: Where this came from, for the audit trail — both `SET_NULL`, because the
    #: bank outliving its origin is the whole point.
    origin_question = models.ForeignKey(
        "exams.Question",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="banked_as",
    )
    origin_exam = models.ForeignKey(
        "exams.Exam",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="banked_questions",
    )
    #: Copied for the same reason `ItemRun.topic_name` is: the card still reads
    #: as a sentence after the exam it names has been deleted.
    origin_exam_title = models.CharField(max_length=200, blank=True)

    #: Null when the stem could not be embedded at save time. See `indexed()`.
    embedding = VectorField(dimensions=settings.EMBEDDING_DIM, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = BankQuestionQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at", "-pk"]
        indexes = [models.Index(fields=["course", "question_type", "level"])]

    def __str__(self) -> str:
        return self.stem[:80]

    @property
    def type_label(self) -> str:
        try:
            return BlueprintRow.QuestionType(self.question_type).label
        except ValueError:  # pragma: no cover - a type that left the choices
            return self.question_type

    @property
    def level_label(self) -> str:
        try:
            return BlueprintRow.Level(self.level).label
        except ValueError:  # pragma: no cover - as above
            return self.level

    @property
    def topic_label(self) -> str:
        return self.topic.name if self.topic else (self.topic_name or "—")

    @property
    def usage_count(self) -> int:
        """How many exams have carried this question, origin exam included."""
        return self.usages.count()

    @property
    def used_by(self) -> list[str]:
        """The exams that carried it, named — including ones since deleted."""
        return [usage.exam_label for usage in self.usages.all()]

    @property
    def is_indexed(self) -> bool:
        return self.embedding is not None

    @property
    def key(self):
        """The stored key, parsed back into its typed form — as `Question.key`."""
        from agents.answer_key import answer_key_from_dict

        return answer_key_from_dict(self.answer_key)


class BankUsage(models.Model):
    """One exam that has carried one banked question.

    Written when the question is banked (the exam it was approved on used it)
    and again every time it is pulled into a later exam. It is what stops the
    same banked question being drawn twice into one paper, and what lets a
    browse screen say "used in 3 exams" honestly.
    """

    bank_question = models.ForeignKey(
        BankQuestion, on_delete=models.CASCADE, related_name="usages"
    )
    #: Null once the exam is deleted — the *fact* that it was used there is not
    #: undone by deleting the paper, so the row stays and `exam_title` carries
    #: the name.
    exam = models.ForeignKey(
        "exams.Exam", on_delete=models.SET_NULL, null=True, blank=True, related_name="bank_usages"
    )
    exam_title = models.CharField(max_length=200, blank=True)
    #: The copy this pull created on that exam, when it was a pull. Null for the
    #: origin usage, whose question is `BankQuestion.origin_question`.
    question = models.ForeignKey(
        "exams.Question",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="bank_usages",
    )
    #: True for the exam the question was approved on, false for a later reuse.
    is_origin = models.BooleanField(default=False)
    used_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["used_at", "pk"]
        constraints = [
            models.UniqueConstraint(
                fields=["bank_question", "exam"], name="one_usage_per_exam_per_bank_question"
            )
        ]

    def __str__(self) -> str:
        return f"{self.exam_label} — {self.bank_question.stem[:50]}"

    @property
    def exam_label(self) -> str:
        if self.exam is not None:
            return self.exam.display_title
        return self.exam_title or "an exam since deleted"
