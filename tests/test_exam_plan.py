"""M4: Agent 1A's output — a validated blueprint, grounded in real passages.

The contract this file holds down is the one M5 depends on: **one reference
passage bundle per planned question**, never per row and never per topic. A row
of five questions is five items Agent 2A will be handed separately, and each has
to arrive knowing what it is written from.

No network: retrieval is injected in most tests, and the one test that goes
through the real `retrieve` patches the provider. A completion model is never
reachable from this path at all — the fake provider raises if one is asked for.
"""

from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from agents.analyze import (
    BlueprintNotReady,
    ExamPlan,
    build_exam_plan,
    plan_questions_for_row,
)
from courses.models import Chunk, Course, SourceFile, Topic
from courses.services.retrieval import Passage
from exams.models import Blueprint, BlueprintRow, Exam

PASSWORD = "quiet-precision-42"
DIM = 1536


def unit(axis: int, dim: int = DIM):
    vector = [0.0] * dim
    vector[axis] = 1.0
    return vector


class OneAxisProvider:
    """Embeds everything onto the same axis, so every passage matches.

    Deliberately blunt: this file is testing the wiring, not the ranking (which
    `test_retrieval.py` already pins down). Completion is unreachable.
    """

    name = "one-axis"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return [unit(0) for _ in texts]

    def complete(self, *args, **kwargs):  # pragma: no cover - the failure is the point
        raise AssertionError("Agent 1A must never call a completion model.")


def a_passage(text="Truth tables assign a value to every row.", page=3, score=0.81):
    return Passage(
        chunk_id=1, text=text, page=page, source_file="lecture.pdf", score=score, topic="Logic"
    )


class FakeRetrieve:
    """A stand-in for M3's `retrieve` that records exactly how it was called."""

    def __init__(self, passages=None, *, by_topic=None):
        self.passages = passages if passages is not None else [a_passage()]
        self.by_topic = by_topic or {}
        self.calls = []

    def __call__(self, course, query, *, k=None):
        self.calls.append((course, query, k))
        if isinstance(query, Topic) and query.name in self.by_topic:
            return list(self.by_topic[query.name])
        return list(self.passages)


class PlanFixture(TestCase):
    """A course, an exam, and a blueprint that adds up."""

    def setUp(self):
        user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(instructor=user, name="Discrete maths", code="CS310")
        self.exam = Exam.objects.create(
            course=self.course, total_score=40, question_count=8, duration_minutes=60
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.logic = Topic.objects.create(course=self.course, name="Logic", position=0)
        self.recursion = Topic.objects.create(course=self.course, name="Recursion", position=1)
        self.row_a = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.logic,
            count=5,
            marks=Decimal("25"),
            weight_percent=Decimal("62.50"),
            position=0,
        )
        self.row_b = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.recursion,
            count=3,
            marks=Decimal("15"),
            weight_percent=Decimal("37.50"),
            position=1,
        )


class RowGroundingTests(PlanFixture):
    def test_a_row_of_five_yields_five_planned_questions(self):
        retrieve = FakeRetrieve()

        questions = plan_questions_for_row(self.row_a, retrieve=retrieve)

        self.assertEqual(len(questions), 5)
        self.assertEqual([q.index_in_row for q in questions], [1, 2, 3, 4, 5])

    def test_each_planned_question_carries_its_own_bundle(self):
        retrieve = FakeRetrieve([a_passage(), a_passage(page=4)])

        questions = plan_questions_for_row(self.row_a, retrieve=retrieve)

        for question in questions:
            self.assertEqual(len(question.passages), 2)
            self.assertTrue(question.has_passages)

    def test_the_row_is_retrieved_once_not_once_per_question(self):
        """Five identical queries would be five embedding calls for one answer."""
        retrieve = FakeRetrieve()

        plan_questions_for_row(self.row_a, retrieve=retrieve)

        self.assertEqual(len(retrieve.calls), 1)

    def test_it_retrieves_for_the_topic_within_its_own_course(self):
        retrieve = FakeRetrieve()

        plan_questions_for_row(self.row_a, retrieve=retrieve)

        course, query, _k = retrieve.calls[0]
        self.assertEqual(course, self.course)
        self.assertEqual(query, self.logic)

    def test_a_planned_question_knows_what_it_is_worth(self):
        retrieve = FakeRetrieve()

        questions = plan_questions_for_row(self.row_a, retrieve=retrieve)

        self.assertEqual(questions[0].marks, Decimal("5.00"))
        self.assertEqual(questions[0].topic_name, "Logic")

    def test_page_refs_are_carried_for_the_instructor(self):
        retrieve = FakeRetrieve([a_passage(page=14)])

        questions = plan_questions_for_row(self.row_a, retrieve=retrieve)

        self.assertEqual(questions[0].page_refs, ["lecture.pdf · page 14"])


class BuildExamPlanTests(PlanFixture):
    def test_a_valid_blueprint_emits_one_bundle_per_planned_question(self):
        """The success check of M4."""
        retrieve = FakeRetrieve()

        plan = build_exam_plan(self.blueprint, retrieve=retrieve)

        self.assertIsInstance(plan, ExamPlan)
        self.assertEqual(plan.question_count, 8)
        self.assertEqual(plan.question_count, self.exam.question_count)
        self.assertTrue(all(q.has_passages for q in plan.questions))
        self.assertEqual(plan.passage_count, 8)

    def test_one_retrieval_per_row(self):
        retrieve = FakeRetrieve()

        build_exam_plan(self.blueprint, retrieve=retrieve)

        self.assertEqual(len(retrieve.calls), 2)

    def test_the_questions_of_each_row_can_be_found_again(self):
        retrieve = FakeRetrieve()

        plan = build_exam_plan(self.blueprint, retrieve=retrieve)

        self.assertEqual(len(plan.questions_for_row(self.row_a.pk)), 5)
        self.assertEqual(len(plan.questions_for_row(self.row_b.pk)), 3)

    def test_an_invalid_blueprint_is_refused_before_a_single_call(self):
        self.row_b.count = 2  # the exam wants 8 questions; this plans 7
        self.row_b.save()
        retrieve = FakeRetrieve()

        with self.assertRaises(BlueprintNotReady) as caught:
            build_exam_plan(self.blueprint, retrieve=retrieve)

        self.assertEqual(retrieve.calls, [])
        self.assertTrue(caught.exception.report.has("count_mismatch"))
        self.assertIn("7 questions", str(caught.exception.report.issues[0].message))

    def test_a_blueprint_on_an_excluded_topic_is_refused(self):
        self.recursion.excluded = True
        self.recursion.save()

        with self.assertRaises(BlueprintNotReady) as caught:
            build_exam_plan(self.blueprint, retrieve=FakeRetrieve())

        self.assertTrue(caught.exception.report.has("excluded_topic"))

    def test_a_topic_the_material_does_not_cover_is_reported_not_hidden(self):
        retrieve = FakeRetrieve(by_topic={"Recursion": []})

        plan = build_exam_plan(self.blueprint, retrieve=retrieve)

        self.assertEqual(plan.topics_without_passages, ["Recursion"])
        self.assertFalse(plan.is_fully_grounded)
        # The questions are still planned — the gap is named, not deleted.
        self.assertEqual(plan.question_count, 8)
        self.assertEqual(len(plan.questions_for_row(self.row_b.pk)), 3)

    def test_a_fully_covered_plan_says_so(self):
        plan = build_exam_plan(self.blueprint, retrieve=FakeRetrieve())

        self.assertTrue(plan.is_fully_grounded)
        self.assertEqual(plan.topics_without_passages, [])

    def test_the_plan_carries_the_exam_it_belongs_to(self):
        plan = build_exam_plan(self.blueprint, retrieve=FakeRetrieve())

        self.assertEqual(plan.exam, self.exam)
        self.assertTrue(plan.report.is_valid)


class RealRetrievalWiringTests(PlanFixture):
    """The same wiring, through M3's actual `retrieve` — with a fake provider.

    Injecting retrieval everywhere would leave the join between M3 and M4
    untested, which is exactly where a wiring bug would live.
    """

    def setUp(self):
        super().setUp()
        self.source_file = SourceFile.objects.create(
            course=self.course,
            original_name="lecture.pdf",
            kind=SourceFile.Kind.PDF,
            page_count=20,
        )
        for index, topic in enumerate([self.logic, self.recursion]):
            Chunk.objects.create(
                source_file=self.source_file,
                page=index + 1,
                position=0,
                text=f"A passage about {topic.name}.",
                embedding=unit(0),
                topic=topic,
            )

    def test_passages_come_back_through_the_real_retrieval_path(self):
        provider = OneAxisProvider()

        with patch("agents.provider.get_provider", return_value=provider):
            plan = build_exam_plan(self.blueprint)

        self.assertEqual(plan.question_count, 8)
        self.assertTrue(all(q.has_passages for q in plan.questions))
        # One embedding call per row, not per question.
        self.assertEqual(provider.calls, 2)

    def test_an_excluded_topics_passages_never_reach_a_planned_question(self):
        """The M2 exclusion rule, still holding at the last step of Agent 1A."""
        stray = Topic.objects.create(course=self.course, name="Ethics", excluded=True)
        Chunk.objects.create(
            source_file=self.source_file,
            page=9,
            position=0,
            text="An excluded passage about Ethics.",
            embedding=unit(0),
            topic=stray,
        )
        provider = OneAxisProvider()

        with patch("agents.provider.get_provider", return_value=provider):
            plan = build_exam_plan(self.blueprint)

        texts = [passage.text for question in plan.questions for passage in question.passages]
        self.assertTrue(texts)
        self.assertNotIn("An excluded passage about Ethics.", texts)
