"""Run Agent 3A on one question and print the verdict — M7's manual test.

Two ways in. A stored question:

    uv run python manage.py review_probe --question 41 --provider gemini

Or a hand-crafted one, judged against a real topic's retrieved passages — which
is how a *deliberately* bad question gets reviewed, since Agent 2A does not
write them on request:

    uv run python manage.py review_probe --topic 7 --level multi_step \\
        --stem "What is precision?" --correct "TP / (TP + FP)" \\
        --option "TP / (TP + FP)" --option "TP / (TP + FN)" --option "TP / N"

Nothing is stored and nothing is changed: Agent 3A judges only.
"""

from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError

from agents.review import (
    BY_PYTHON,
    QuestionReviewError,
    ReviewSubject,
    review_question,
)
from courses.models import Topic
from courses.services.retrieval import RetrievalError, retrieve
from exams.models import Question


class Command(BaseCommand):
    help = "Review one question (stored or hand-crafted) and print the findings."

    def add_arguments(self, parser):
        parser.add_argument("--question", type=int, help="Review a stored Question by id.")
        parser.add_argument("--topic", type=int, help="Topic to retrieve passages from.")
        parser.add_argument("--stem", help="The question, for a hand-crafted review.")
        parser.add_argument("--correct", default="", help="The answer.")
        parser.add_argument("--option", action="append", default=[], help="An option (repeatable).")
        parser.add_argument("--type", default="mcq", help="Question type.")
        parser.add_argument("--level", default="multi_step", help="The level that was asked for.")
        parser.add_argument("--marks", default="2", help="Marks the question is worth.")
        parser.add_argument("--explanation", default="", help="The writer's explanation.")
        parser.add_argument("--provider", help="Override LLM_PROVIDER for this run.")
        parser.add_argument(
            "--python-only", action="store_true", help="Deterministic checks only, no call."
        )

    def handle(self, *args, **options):
        provider = self._provider(options.get("provider"))
        subject = self._subject(options, provider)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Reviewing: {subject.stem}"))
        self.stdout.write(
            f"  asked for: {subject.type_label} · {subject.level_label} · "
            f"{subject.marks} mark(s) · topic {subject.topic_name!r}"
        )
        for position, passage in enumerate(subject.passages, start=1):
            mark = " (OCR)" if passage.from_ocr else ""
            self.stdout.write(f"  P{position}  {passage.page_ref}{mark}  {passage.text[:80]}…")

        try:
            result = review_question(
                subject, provider=provider, python_only=options["python_only"]
            )
        except QuestionReviewError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write("")
        for finding in result.findings:
            if finding.passed:
                self.stdout.write(
                    self.style.SUCCESS(f"  ✓ {finding.label} ({finding.decided_by})")
                )
        for finding in result.failures:
            self.stdout.write("")
            self.stdout.write(self.style.ERROR(f"  ✗ {finding.label} ({finding.decided_by})"))
            self.stdout.write(f"      why: {finding.reason}")
            self.stdout.write(f"      replacement must: {finding.requirement}")

        self.stdout.write("")
        if result.passed:
            checked = "every check" if result.model_checked else "every deterministic check"
            self.stdout.write(self.style.SUCCESS(f"  PASSED {checked}."))
        else:
            self.stdout.write(
                self.style.ERROR(
                    f"  REJECTED on {', '.join(result.failed_checks)} "
                    f"({sum(1 for f in result.failures if f.decided_by == BY_PYTHON)} "
                    "decided in Python)."
                )
            )
            self.stdout.write("")
            self.stdout.write("  The note Agent 2A is handed (M8):")
            for note in result.notes:
                self.stdout.write(f"    · {note}")

    def _subject(self, options, provider=None):
        if options["question"]:
            question = Question.objects.filter(pk=options["question"]).first()
            if question is None:
                raise CommandError("No such question.")
            return ReviewSubject.from_question(question)

        if not (options["topic"] and options["stem"]):
            raise CommandError("Pass --question, or --topic with --stem.")
        topic = Topic.objects.filter(pk=options["topic"]).select_related("course").first()
        if topic is None:
            raise CommandError("No such topic.")
        try:
            passages = retrieve(topic.course, topic, provider=provider)
        except RetrievalError as exc:
            raise CommandError(str(exc)) from exc
        if not passages:
            raise CommandError("No passages retrieved — nothing to judge the question against.")

        options_given = tuple(options["option"])
        correct = options["correct"] or (options_given[0] if options_given else "")
        return ReviewSubject(
            stem=options["stem"],
            question_type=options["type"],
            options=options_given,
            correct=correct,
            explanation=options["explanation"],
            requested_level=options["level"],
            marks=Decimal(options["marks"]),
            topic_name=topic.name,
            course_name=topic.course.name,
            passages=tuple(passages),
            answer_key=self._key(options["type"], correct, options_given),
        )

    def _key(self, question_type, correct, options):
        """A key for the hand-crafted question, so the key checks have something to read."""
        from agents.answer_key import ObjectiveKey, ShortAnswerKey

        if not correct:
            return None
        if question_type in {"mcq", "true_false"}:
            return ObjectiveKey(answer=correct, options=list(options))
        if question_type == "short_answer":
            return ShortAnswerKey(model_answer=correct, required_elements=[correct])
        return None

    def _provider(self, name):
        from agents.provider import LLMError, get_provider

        try:
            return get_provider(name) if name else None
        except LLMError as exc:
            raise CommandError(str(exc)) from exc
