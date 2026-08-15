"""M9: two forms out of one blueprint, and what happens when the pool is thin.

No provider is faked here because none is reachable: form assembly is
arithmetic, and the one test that goes looking (`test_assembly_calls_no_model`)
makes `get_provider` explode to prove it. The suite has made no network call
since M2 and this milestone does not start.

What is pinned down:

* the two forms match on **every** dimension M9 names — count, marks, topic,
  type, level, multi-step, numeric, expected time — and seven of the eight match
  *exactly*, because the construction gives no room for them to drift;
* fully-separate forms share nothing at all when the pool can cover both;
* a pool that cannot cover both is **reported per row**, naming the form, the
  topic and the missing count, and no form is written;
* sharing-allowed reuses a question rather than leaving a hole — and the forms
  still balance;
* the cap is two forms, at the service and on the spec screen;
* the sharing question is hidden for one form and revealed for two.
"""

import random
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from courses.models import Course, Topic
from exams.models import (
    Blueprint,
    BlueprintRow,
    Exam,
    Form,
    FormQuestion,
    ItemRun,
    Question,
    QuestionAttempt,
)
from exams.services.forms import (
    MAX_FORMS,
    Assembly,
    FormAssemblyError,
    FormPlan,
    Item,
    RowDemand,
    assemble_forms,
    compare,
    distribute,
    estimate_minutes,
    greedy_balanced,
    pool_for_exam,
    save_assembly,
)

PASSWORD = "quiet-precision-42"

MCQ = BlueprintRow.QuestionType.MCQ
NUMERIC = BlueprintRow.QuestionType.NUMERIC
SHORT = BlueprintRow.QuestionType.SHORT_ANSWER
DIRECT = BlueprintRow.Level.DIRECT
MEDIUM = BlueprintRow.Level.MEDIUM
MULTI_STEP = BlueprintRow.Level.MULTI_STEP


def an_item(question_id, *, topic="Recursion", question_type=MCQ, level=MEDIUM, marks="2",
            minutes="1.5") -> Item:
    return Item(
        question_id=question_id,
        topic_id=1,
        topic_name=topic,
        question_type=question_type,
        level=level,
        marks=Decimal(marks),
        minutes=Decimal(minutes),
        stem=f"Question {question_id}",
    )


class ExpectedTimeTests(SimpleTestCase):
    """The one number M9 estimates rather than derives — and it is only that."""

    def test_type_and_level_both_move_the_estimate(self):
        easy = estimate_minutes(question_type=MCQ, level=DIRECT, stem="One two three")
        hard = estimate_minutes(question_type=NUMERIC, level=MULTI_STEP, stem="One two three")
        self.assertLess(easy, hard)

    def test_a_longer_stem_costs_reading_time(self):
        short = estimate_minutes(question_type=MCQ, level=MEDIUM, stem="Short stem")
        long = estimate_minutes(
            question_type=MCQ, level=MEDIUM, stem=" ".join(["word"] * 360)
        )
        self.assertEqual(long - short, Decimal("2"))

    def test_worked_steps_cost_time_beyond_the_first(self):
        one = estimate_minutes(
            question_type=NUMERIC, level=MEDIUM, answer_key={"steps": [{"marks": 1}]}
        )
        three = estimate_minutes(
            question_type=NUMERIC,
            level=MEDIUM,
            answer_key={"steps": [{"marks": 1}, {"marks": 1}, {"marks": 1}]},
        )
        self.assertEqual(three - one, Decimal("1"))

    def test_the_estimate_is_deterministic(self):
        kwargs = dict(question_type=SHORT, level=MEDIUM, stem="Define a stack and give one use")
        self.assertEqual(estimate_minutes(**kwargs), estimate_minutes(**kwargs))


class AllocatorTests(SimpleTestCase):
    """The seam itself: one row's pool, dealt across two forms."""

    def test_it_balances_expected_time_rather_than_dealing_in_order(self):
        # Dealt in order, Form A would take both long questions. Balanced, it
        # takes one of each — which is the whole point of scoring here.
        items = [
            an_item(1, minutes="5"),
            an_item(2, minutes="5"),
            an_item(3, minutes="1"),
            an_item(4, minutes="1"),
        ]
        a, b = greedy_balanced(items, per_form=2, form_count=2, allow_sharing=False)
        self.assertEqual(sum(i.minutes for i in a), sum(i.minutes for i in b))

    def test_it_never_reuses_a_question_unless_sharing_is_allowed(self):
        items = [an_item(i) for i in range(1, 4)]
        a, b = greedy_balanced(items, per_form=2, form_count=2, allow_sharing=False)
        self.assertFalse({i.question_id for i in a} & {i.question_id for i in b})
        # Three questions cannot fill two forms of two without reuse, so one
        # bucket comes back short rather than being padded.
        self.assertEqual(sorted([len(a), len(b)]), [1, 2])

    def test_sharing_fills_from_questions_already_placed(self):
        items = [an_item(i) for i in range(1, 4)]
        a, b = greedy_balanced(items, per_form=2, form_count=2, allow_sharing=True)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)
        self.assertTrue({i.question_id for i in a} & {i.question_id for i in b})

    def test_no_form_carries_the_same_question_twice(self):
        # One question, two forms of two: sharing can put it on both papers, and
        # cannot put it on either one twice.
        a, b = greedy_balanced([an_item(1)], per_form=2, form_count=2, allow_sharing=True)
        for bucket in (a, b):
            ids = [i.question_id for i in bucket]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 1)

    def test_the_same_pool_always_assembles_the_same_way(self):
        items = [an_item(i, minutes=str(1 + i % 3)) for i in range(1, 9)]
        first = greedy_balanced(items, per_form=3, form_count=2, allow_sharing=False)
        second = greedy_balanced(
            list(reversed(items)), per_form=3, form_count=2, allow_sharing=False
        )
        self.assertEqual(
            [[i.question_id for i in b] for b in first],
            [[i.question_id for i in b] for b in second],
        )


def seeded_demands_and_pool(*, per_form=3, surplus=2, seed=9):
    """A blueprint's worth of rows and a reviewed pool with room to choose from.

    The pool is varied on purpose — different stem lengths, so different
    expected times — because a pool of identical questions would make the time
    balance true by accident rather than by the allocator.
    """
    rng = random.Random(seed)
    plan = [
        ("Recursion", MCQ, MEDIUM, "2"),
        ("Recursion", SHORT, MULTI_STEP, "4"),
        ("Sorting", MCQ, DIRECT, "1"),
        ("Complexity", NUMERIC, MULTI_STEP, "5"),
    ]
    demands, pool, next_id = [], {}, 1
    for position, (topic, question_type, level, marks) in enumerate(plan):
        row_id = position + 1
        demands.append(
            RowDemand(
                row_id=row_id,
                topic_id=row_id,
                topic_name=topic,
                question_type=question_type,
                level=level,
                count=per_form,
                marks_per_question=Decimal(marks),
                position=position,
            )
        )
        items = []
        for _ in range(per_form * 2 + surplus):
            items.append(
                Item(
                    question_id=next_id,
                    topic_id=row_id,
                    topic_name=topic,
                    question_type=question_type,
                    level=level,
                    marks=Decimal(marks),
                    minutes=Decimal(rng.choice(["0.75", "1.5", "2.25", "3", "4.5"])),
                    stem=f"{topic} question {next_id}",
                )
            )
            next_id += 1
        pool[row_id] = items
    return demands, pool


class DistributionTests(SimpleTestCase):
    """Two forms, one blueprint, every dimension M9 lists."""

    def setUp(self):
        self.demands, self.pool = seeded_demands_and_pool()

    def test_every_dimension_matches_within_tolerance(self):
        assembly = distribute(self.demands, self.pool)
        self.assertTrue(assembly.is_complete)
        self.assertTrue(assembly.matches, assembly.failing_dimensions)

    def test_the_seven_structural_dimensions_match_exactly(self):
        # Not "within tolerance" — exactly. These cannot drift, because the
        # assembly works a row at a time and a row is one topic, one type, one
        # level, one price. A non-zero spread here is a bug, not a near miss.
        assembly = distribute(self.demands, self.pool)
        exact = {
            dimension.name: dimension
            for dimension in assembly.dimensions
            if dimension.name != "expected_minutes"
        }
        self.assertEqual(len(exact), 7)
        for name, dimension in exact.items():
            with self.subTest(dimension=name):
                self.assertEqual(dimension.spread, Decimal("0"))
                self.assertEqual(dimension.tolerance, Decimal("0"))

    def test_expected_time_is_balanced_not_merely_tolerated(self):
        assembly = distribute(self.demands, self.pool)
        time = next(d for d in assembly.dimensions if d.name == "expected_minutes")
        self.assertLessEqual(time.spread, time.tolerance)
        self.assertFalse(time.by_construction)

    def test_each_form_carries_the_planned_count_and_marks(self):
        assembly = distribute(self.demands, self.pool)
        expected_count = sum(d.count for d in self.demands)
        expected_marks = sum(
            (d.marks_per_question * d.count for d in self.demands), Decimal("0")
        )
        for form in assembly.forms:
            self.assertEqual(form.question_count, expected_count)
            self.assertEqual(form.total_marks, expected_marks)

    def test_topics_carry_the_same_weight_on_both_forms(self):
        assembly = distribute(self.demands, self.pool)
        a, b = assembly.forms
        self.assertEqual(a.marks_by_topic(), b.marks_by_topic())
        self.assertEqual(a.counts_by("topic_name"), b.counts_by("topic_name"))

    def test_a_row_asking_for_nothing_is_skipped_not_flagged(self):
        demands = [RowDemand(1, 1, "Recursion", MCQ, MEDIUM, 0, Decimal("0"), 0)]
        assembly = distribute(demands, {1: []})
        self.assertEqual(assembly.shortfalls, [])
        self.assertEqual(assembly.forms[0].question_count, 0)


class FullySeparateTests(SimpleTestCase):
    """A ≠ B: no question on both papers, or an honest account of why not."""

    def test_no_question_appears_on_both_forms(self):
        demands, pool = seeded_demands_and_pool()
        assembly = distribute(demands, pool, sharing_allowed=False)
        a, b = assembly.forms
        self.assertFalse(a.question_ids & b.question_ids)
        self.assertEqual(assembly.shared_count, 0)
        self.assertIn("No question appears on both", assembly.summary)

    def test_a_thin_pool_is_reported_per_row_and_never_padded(self):
        demands, pool = seeded_demands_and_pool(per_form=3, surplus=0)
        # Four questions where two whole forms need six: Form B comes up short.
        pool[1] = pool[1][:4]
        assembly = distribute(demands, pool, sharing_allowed=False)

        self.assertFalse(assembly.is_complete)
        self.assertEqual(len(assembly.shortfalls), 1)
        shortfall = assembly.shortfalls[0]
        self.assertEqual(shortfall.form_label, "B")
        self.assertEqual(shortfall.topic_name, "Recursion")
        self.assertEqual(shortfall.missing, 2)
        self.assertEqual(shortfall.available_in_pool, 4)

        # The sentence names the form, the count, the topic, and both ways out.
        self.assertIn("Form B is short 2 questions", shortfall.message)
        self.assertIn("Recursion", shortfall.message)
        self.assertIn("Run generation again", shortfall.message)
        self.assertIn("share questions", shortfall.message)

        # Form A is filled and Form B is left short — nothing was invented to
        # fill it, and the gap is concentrated rather than spread over both
        # papers. (Recursion has two rows in this blueprint, an MCQ row and a
        # short-answer one; only the MCQ row is thin, so A carries 3 + 3 and B
        # carries 1 + 3.)
        a, b = assembly.forms
        self.assertEqual(a.counts_by("topic_name")["Recursion"], 6)
        self.assertEqual(b.counts_by("topic_name")["Recursion"], 4)

    def test_every_short_row_is_named_not_just_the_first(self):
        demands, pool = seeded_demands_and_pool(per_form=2, surplus=0)
        pool[1] = pool[1][:3]
        pool[3] = pool[3][:3]
        assembly = distribute(demands, pool, sharing_allowed=False)
        self.assertEqual(len(assembly.shortfalls), 2)
        self.assertEqual(
            {s.topic_name for s in assembly.shortfalls}, {"Recursion", "Sorting"}
        )

    def test_a_pool_too_thin_for_even_one_form_shorts_both(self):
        demands, pool = seeded_demands_and_pool(per_form=3, surplus=0)
        pool[1] = pool[1][:2]
        assembly = distribute(demands, pool, sharing_allowed=False)
        self.assertEqual({s.form_label for s in assembly.shortfalls}, {"A", "B"})


class SharingAllowedTests(SimpleTestCase):
    """Sharing draws from the same pool, and the papers still line up."""

    def test_it_reuses_a_question_when_the_pool_runs_out(self):
        demands, pool = seeded_demands_and_pool(per_form=3, surplus=0)
        pool[1] = pool[1][:4]
        assembly = distribute(demands, pool, sharing_allowed=True)

        self.assertTrue(assembly.is_complete)
        self.assertEqual(assembly.shortfalls, [])
        self.assertEqual(assembly.shared_count, 2)
        self.assertTrue(assembly.matches, assembly.failing_dimensions)

    def test_it_does_not_reuse_while_alternatives_are_unused(self):
        # Sharing is permission, not preference: a pool that covers both papers
        # produces two disjoint papers even with sharing on.
        demands, pool = seeded_demands_and_pool()
        assembly = distribute(demands, pool, sharing_allowed=True)
        self.assertEqual(assembly.shared_count, 0)

    def test_sharing_cannot_rescue_a_row_short_for_one_paper(self):
        demands, pool = seeded_demands_and_pool(per_form=3, surplus=0)
        pool[1] = pool[1][:2]
        assembly = distribute(demands, pool, sharing_allowed=True)
        self.assertFalse(assembly.is_complete)
        shortfall = assembly.shortfalls[0]
        self.assertEqual(shortfall.missing, 1)
        # And it says so — telling an instructor to turn on a setting that is
        # already on would be the least useful sentence on the screen.
        self.assertIn("Sharing is already on", shortfall.message)
        self.assertNotIn("(Exam settings", shortfall.message)


class FormCapTests(SimpleTestCase):
    """Two forms is the MVP, and it is enforced where forms are made."""

    def test_the_cap_is_two(self):
        self.assertEqual(MAX_FORMS, 2)

    def test_a_third_form_is_refused(self):
        demands, pool = seeded_demands_and_pool()
        with self.assertRaises(FormAssemblyError) as caught:
            distribute(demands, pool, form_count=3)
        self.assertIn("at most 2 forms", str(caught.exception))

    def test_zero_forms_is_refused(self):
        with self.assertRaises(FormAssemblyError):
            distribute([], {}, form_count=0)

    def test_one_form_is_assembled_without_complaint(self):
        demands, pool = seeded_demands_and_pool()
        assembly = distribute(demands, pool, form_count=1)
        self.assertEqual(assembly.form_count, 1)
        self.assertTrue(assembly.is_complete)


class OptimizerSeamTests(SimpleTestCase):
    """The greedy rule is replaceable — that is the point of it being a seam."""

    def test_a_different_allocator_can_be_supplied(self):
        calls = []

        def take_in_order(items, *, per_form, form_count, allow_sharing):
            calls.append(per_form)
            return [items[i * per_form:(i + 1) * per_form] for i in range(form_count)]

        demands, pool = seeded_demands_and_pool()
        assembly = distribute(demands, pool, allocate=take_in_order)
        self.assertEqual(len(calls), len(demands))
        self.assertTrue(assembly.is_complete)
        # The structural dimensions hold whatever the allocator does, because
        # they are the row's, not the allocator's.
        for dimension in assembly.dimensions:
            if dimension.by_construction:
                self.assertEqual(dimension.spread, Decimal("0"))


class DistributionSummaryTests(SimpleTestCase):
    """The machine-readable summary M10 will render."""

    def test_it_reports_every_dimension_for_every_form(self):
        demands, pool = seeded_demands_and_pool()
        summary = distribute(demands, pool).distribution

        self.assertEqual([f["label"] for f in summary["forms"]], ["A", "B"])
        self.assertTrue(summary["matches"])
        self.assertTrue(summary["is_complete"])
        self.assertEqual(summary["shortfalls"], [])
        self.assertEqual(
            [d["name"] for d in summary["dimensions"]],
            [
                "question_count",
                "total_marks",
                "topic_distribution",
                "type_distribution",
                "level_spread",
                "multi_step_count",
                "numeric_count",
                "expected_minutes",
            ],
        )
        for form in summary["forms"]:
            for key in ("topics", "types", "levels", "topic_marks"):
                self.assertTrue(form[key])

    def test_shortfalls_are_carried_in_the_summary(self):
        demands, pool = seeded_demands_and_pool(per_form=3, surplus=0)
        pool[1] = pool[1][:4]
        summary = distribute(demands, pool).distribution
        self.assertFalse(summary["is_complete"])
        self.assertEqual(summary["shortfalls"][0]["missing"], 2)
        self.assertEqual(summary["shortfalls"][0]["topic"], "Recursion")

    def test_no_forms_compares_to_nothing(self):
        self.assertEqual(compare([]), [])
        self.assertFalse(Assembly().matches)


# --- Against the database ----------------------------------------------------


class ExamFixture(TestCase):
    """One exam, one blueprint, and an M8-style reviewed pool behind it."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, code="CS201", name="Data Structures"
        )
        self.exam = Exam.objects.create(
            course=self.course,
            title="Midterm",
            total_score=20,
            question_count=6,
            number_of_forms=2,
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.rows = []
        for position, (name, question_type, level, marks) in enumerate(
            [
                ("Recursion", MCQ, MEDIUM, "6"),
                ("Sorting", NUMERIC, MULTI_STEP, "14"),
            ]
        ):
            topic = Topic.objects.create(course=self.course, name=name, position=position)
            self.rows.append(
                BlueprintRow.objects.create(
                    blueprint=self.blueprint,
                    topic=topic,
                    question_type=question_type,
                    level=level,
                    count=3,
                    marks=Decimal(marks),
                    weight_percent=Decimal("30") if position == 0 else Decimal("70"),
                    position=position,
                )
            )

    def fill_pool(self, row, count, *, run=None):
        """`count` reviewed questions for `row`, recorded the way M8 records them."""
        run = run or ItemRun.objects.create(
            exam=self.exam,
            blueprint_row=row,
            topic_name=row.topic.name,
            question_type=row.question_type,
            level=row.level,
            required=row.count,
            approved_count=count,
            rounds=1,
            status=ItemRun.Status.PASSED,
        )
        made = []
        for index in range(count):
            question = Question.objects.create(
                exam=self.exam,
                blueprint_row=row,
                stem=f"{row.topic.name} question {index} " + "word " * (index * 20),
                question_type=row.question_type,
                correct="A",
                source_ref="lecture.pdf · page 1",
                position=index,
            )
            QuestionAttempt.objects.create(
                item_run=run,
                round=1,
                outcome=QuestionAttempt.Outcome.PASSED,
                stem=question.stem,
                question=question,
            )
            made.append(question)
        return run, made


class PoolTests(ExamFixture):
    """What may reach a paper, and what may not."""

    def test_the_pool_is_read_through_review_not_through_questions(self):
        self.fill_pool(self.rows[0], 4)
        # A question nobody reviewed exists, and stays out of the pool.
        Question.objects.create(
            exam=self.exam,
            blueprint_row=self.rows[0],
            stem="Never reviewed",
            question_type=MCQ,
            correct="A",
            source_ref="lecture.pdf · page 2",
        )
        pool = pool_for_exam(self.exam)
        self.assertEqual(len(pool[self.rows[0].pk]), 4)
        self.assertNotIn("Never reviewed", [item.stem for item in pool[self.rows[0].pk]])

    def test_a_question_the_instructor_rejected_is_out(self):
        _run, questions = self.fill_pool(self.rows[0], 4)
        questions[0].status = Question.Status.REJECTED
        questions[0].save(update_fields=["status"])
        self.assertEqual(len(pool_for_exam(self.exam)[self.rows[0].pk]), 3)

    def test_items_are_priced_by_the_row_not_by_the_question(self):
        self.fill_pool(self.rows[1], 6)
        item = pool_for_exam(self.exam)[self.rows[1].pk][0]
        self.assertEqual(item.marks, self.rows[1].marks_per_question)
        self.assertEqual(item.level, MULTI_STEP)
        self.assertTrue(item.is_multi_step)
        self.assertTrue(item.is_numeric)


class AssembleFromExamTests(ExamFixture):
    """End to end, on a real exam: assemble, refuse, save."""

    def test_it_assembles_two_matching_forms(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        assembly = assemble_forms(self.exam)

        self.assertTrue(assembly.is_complete)
        self.assertTrue(assembly.matches, assembly.failing_dimensions)
        a, b = assembly.forms
        self.assertEqual(a.question_count, self.exam.question_count)
        self.assertEqual(b.question_count, self.exam.question_count)
        self.assertEqual(a.total_marks, Decimal(self.exam.total_score))
        self.assertEqual(b.total_marks, Decimal(self.exam.total_score))
        self.assertFalse(a.question_ids & b.question_ids)

    def test_the_mode_comes_from_the_exam_not_the_caller(self):
        for row in self.rows:
            self.fill_pool(row, 4)
        self.assertFalse(assemble_forms(self.exam).is_complete)

        self.exam.form_sharing = Exam.FormSharing.SHARED
        self.exam.save(update_fields=["form_sharing"])
        shared = assemble_forms(self.exam)
        self.assertTrue(shared.is_complete)
        self.assertTrue(shared.sharing_allowed)
        self.assertTrue(shared.shared_count)

    def test_a_one_form_exam_never_shares_whatever_is_stored(self):
        for row in self.rows:
            self.fill_pool(row, 3)
        self.exam.number_of_forms = 1
        self.exam.form_sharing = Exam.FormSharing.SHARED
        self.exam.save(update_fields=["number_of_forms", "form_sharing"])
        assembly = assemble_forms(self.exam)
        self.assertEqual(assembly.form_count, 1)
        self.assertFalse(assembly.sharing_allowed)

    def test_an_exam_without_a_blueprint_is_refused(self):
        other = Exam.objects.create(course=self.course, title="Quiz")
        with self.assertRaises(FormAssemblyError):
            assemble_forms(other)

    def test_a_complete_assembly_is_saved_as_forms_of_placements(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        assembly = assemble_forms(self.exam)
        saved = save_assembly(self.exam, assembly)

        self.assertEqual([form.label for form in saved], ["A", "B"])
        self.assertEqual(FormQuestion.objects.count(), self.exam.question_count * 2)
        form_a = Form.objects.get(exam=self.exam, label="A")
        self.assertEqual(form_a.question_count, self.exam.question_count)
        self.assertEqual(form_a.total_marks, Decimal(self.exam.total_score))
        self.assertEqual(form_a.expected_minutes, assembly.forms[0].expected_minutes)
        # The questions are pointed at, not copied — one edit, every form.
        self.assertEqual(Question.objects.count(), 16)

    def test_saving_again_replaces_rather_than_accumulates(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        save_assembly(self.exam, assemble_forms(self.exam))
        save_assembly(self.exam, assemble_forms(self.exam))
        self.assertEqual(Form.objects.filter(exam=self.exam).count(), 2)
        self.assertEqual(FormQuestion.objects.count(), self.exam.question_count * 2)

    def test_an_incomplete_assembly_is_never_written(self):
        self.fill_pool(self.rows[0], 8)
        self.fill_pool(self.rows[1], 4)
        assembly = assemble_forms(self.exam)

        self.assertFalse(assembly.is_complete)
        with self.assertRaises(FormAssemblyError) as caught:
            save_assembly(self.exam, assembly)
        self.assertIn("no form was saved", str(caught.exception))
        self.assertEqual(Form.objects.count(), 0)
        self.assertEqual(FormQuestion.objects.count(), 0)

    def test_assembly_calls_no_model(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        with patch(
            "agents.provider.get_provider",
            side_effect=AssertionError("form assembly must never call a model"),
        ):
            self.assertTrue(assemble_forms(self.exam).is_complete)


# --- The spec screen ---------------------------------------------------------


class SharingOptionTests(TestCase):
    """The option appears exactly when it means something."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, code="CS201", name="Data Structures"
        )
        self.client.login(username="nadia", password=PASSWORD)
        self.url = reverse("exams:exam_sharing_option", args=[self.course.pk])

    def test_it_is_hidden_for_one_form(self):
        response = self.client.post(self.url, {"number_of_forms": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "How the forms relate")
        self.assertContains(response, 'id="sharing-option"')

    def test_it_is_revealed_for_two_forms(self):
        response = self.client.post(self.url, {"number_of_forms": "2"})
        self.assertContains(response, "How the forms relate")
        self.assertContains(response, "Fully separate")
        self.assertContains(response, "Sharing allowed")

    def test_a_half_typed_count_reveals_nothing(self):
        for raw in ("", "two", "-"):
            with self.subTest(raw=raw):
                response = self.client.post(self.url, {"number_of_forms": raw})
                self.assertNotContains(response, "How the forms relate")

    def test_the_spec_screen_starts_with_it_hidden(self):
        response = self.client.get(reverse("exams:list", args=[self.course.pk]))
        self.assertContains(response, 'id="sharing-option"')
        self.assertNotContains(response, "How the forms relate")

    def test_another_instructors_course_has_no_such_endpoint(self):
        other = Course.objects.create(
            instructor=User.objects.create_user("omar", password=PASSWORD),
            code="CS999",
            name="Elsewhere",
        )
        response = self.client.post(
            reverse("exams:exam_sharing_option", args=[other.pk]), {"number_of_forms": "2"}
        )
        self.assertEqual(response.status_code, 404)

    def test_the_choice_is_saved_with_the_exam(self):
        self.client.post(
            reverse("exams:list", args=[self.course.pk]),
            {
                "title": "Midterm",
                "kind": Exam.Kind.MIDTERM,
                "total_score": "20",
                "question_count": "6",
                "duration_minutes": "60",
                "language": Exam.Language.ENGLISH,
                "number_of_forms": "2",
                "form_sharing": Exam.FormSharing.SHARED,
            },
        )
        exam = Exam.objects.get(title="Midterm")
        self.assertEqual(exam.form_sharing, Exam.FormSharing.SHARED)
        self.assertTrue(exam.forms_may_share)

    def test_a_one_form_exam_saves_the_strict_default_not_an_error(self):
        response = self.client.post(
            reverse("exams:list", args=[self.course.pk]),
            {
                "title": "Quiz",
                "kind": Exam.Kind.QUIZ,
                "total_score": "10",
                "question_count": "5",
                "duration_minutes": "20",
                "language": Exam.Language.ENGLISH,
                "number_of_forms": "1",
            },
        )
        self.assertEqual(response.status_code, 302)
        exam = Exam.objects.get(title="Quiz")
        self.assertEqual(exam.form_sharing, Exam.FormSharing.SEPARATE)
        self.assertFalse(exam.forms_may_share)

    def test_three_forms_is_refused_on_the_spec_screen(self):
        response = self.client.post(
            reverse("exams:list", args=[self.course.pk]),
            {
                "title": "Final",
                "kind": Exam.Kind.FINAL,
                "total_score": "40",
                "question_count": "20",
                "duration_minutes": "90",
                "language": Exam.Language.ENGLISH,
                "number_of_forms": "3",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "at most 2 forms")
        self.assertFalse(Exam.objects.filter(title="Final").exists())


class FormScreenTests(ExamFixture):
    """The diagnostic screen — enough to eyeball, not M10's comparison."""

    def setUp(self):
        super().setUp()
        self.client.login(username="nadia", password=PASSWORD)
        self.url = reverse("exams:forms", args=[self.course.pk, self.exam.pk])

    def test_it_shows_both_forms_and_every_dimension(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        response = self.client.get(self.url)
        self.assertContains(response, "Form A")
        self.assertContains(response, "Form B")
        self.assertContains(response, "Expected time")
        self.assertContains(response, "Cognitive-level spread")

    def test_a_short_pool_shows_the_shortfall_and_saves_nothing(self):
        self.fill_pool(self.rows[0], 8)
        self.fill_pool(self.rows[1], 4)
        response = self.client.get(self.url)
        self.assertContains(response, "Form B is short")
        self.assertContains(response, "Sorting")

        self.client.post(self.url, follow=True)
        self.assertEqual(Form.objects.count(), 0)

    def test_saving_a_complete_assembly_writes_the_forms(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        response = self.client.post(self.url, follow=True)
        self.assertEqual(Form.objects.filter(exam=self.exam).count(), 2)
        self.assertContains(response, "assembled")

    def test_another_instructors_exam_does_not_exist(self):
        self.client.logout()
        User.objects.create_user("omar", password=PASSWORD)
        self.client.login(username="omar", password=PASSWORD)
        self.assertEqual(self.client.get(self.url).status_code, 404)
