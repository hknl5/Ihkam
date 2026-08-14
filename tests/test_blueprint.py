"""M4: the blueprint's arithmetic — every check, and the draft it validates.

Nothing in this file touches a model or the network, because nothing in
`exams/services/blueprint.py` is allowed to. What is asserted is not only that a
broken blueprint fails, but that it fails *with the sentence the instructor
needs*: which topic, what was expected, what is there instead. A validator that
returns False is a validator the instructor has to guess their way past.
"""

from decimal import Decimal

from django.contrib.auth.models import User
from django.test import TestCase

from courses.models import Course, Topic
from exams.models import Blueprint, BlueprintRow, Exam
from exams.services.blueprint import (
    MARK_TOLERANCE,
    RowSpec,
    auto_build,
    check_marks_match_weights,
    check_question_count,
    check_rows_exist,
    check_topic_has_questions,
    check_topics_are_usable,
    check_total_score,
    check_weights,
    eligible_topics,
    validate,
    validate_blueprint,
)

PASSWORD = "quiet-precision-42"


def a_course(code="CS310", **kwargs):
    user = User.objects.create_user(f"nadia-{code}", password=PASSWORD)
    return Course.objects.create(instructor=user, name="Discrete maths", code=code, **kwargs)


def an_exam(course=None, *, total_score=40, question_count=20, **kwargs):
    return Exam.objects.create(
        course=course or a_course(),
        total_score=total_score,
        question_count=question_count,
        **kwargs,
    )


def spec(name="Logic", *, count=1, marks="10", weight="25", topic_id=1, excluded=False):
    return RowSpec(
        topic_id=topic_id,
        topic_name=name,
        count=count,
        marks=Decimal(marks),
        weight_percent=Decimal(weight),
        topic_excluded=excluded,
    )


def a_balanced_pair(exam):
    """Two rows that add up perfectly against a 40-mark, 20-question exam."""
    return [
        spec("Logic", topic_id=1, count=10, marks="20", weight="50"),
        spec("Recursion", topic_id=2, count=10, marks="20", weight="50"),
    ]


class ValidBlueprintTests(TestCase):
    def test_a_blueprint_that_adds_up_has_no_issues(self):
        exam = an_exam()

        report = validate(a_balanced_pair(exam), exam=exam)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.total_count, 20)
        self.assertEqual(report.total_marks, Decimal("40.00"))
        self.assertEqual(report.total_weight, Decimal("100.00"))

    def test_every_total_reads_ok_when_it_adds_up(self):
        exam = an_exam()

        report = validate(a_balanced_pair(exam), exam=exam)

        self.assertEqual(
            (report.count_tone, report.marks_tone, report.weight_tone), ("ok", "ok", "ok")
        )

    def test_one_topic_split_over_two_rows_is_one_topic(self):
        """Five MCQs and five short answers on one chapter is an ordinary plan."""
        exam = an_exam()
        rows = [
            spec("Logic", topic_id=1, count=5, marks="10", weight="25"),
            spec("Logic", topic_id=1, count=5, marks="10", weight="25"),
            spec("Recursion", topic_id=2, count=10, marks="20", weight="50"),
        ]

        report = validate(rows, exam=exam)

        self.assertTrue(report.is_valid, report.codes)


class QuestionCountCheckTests(TestCase):
    def test_too_few_questions_names_both_numbers_and_the_gap(self):
        exam = an_exam(question_count=20)
        rows = [spec(count=18, marks="40", weight="100")]

        issues = check_question_count(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["count_mismatch"])
        message = issues[0].message
        self.assertIn("18 questions", message)
        self.assertIn("set to 20", message)
        self.assertIn("2 short", message)

    def test_too_many_questions_says_too_many(self):
        exam = an_exam(question_count=20)

        issues = check_question_count([spec(count=23, marks="40", weight="100")], exam=exam)

        self.assertIn("23 questions", issues[0].message)
        self.assertIn("3 too many", issues[0].message)

    def test_the_right_number_of_questions_is_not_an_issue(self):
        exam = an_exam(question_count=20)

        self.assertEqual(check_question_count([spec(count=20)], exam=exam), [])


class TotalScoreCheckTests(TestCase):
    def test_unassigned_marks_are_named_with_the_shortfall(self):
        exam = an_exam(total_score=40)
        rows = [spec(marks="38", weight="100", count=20)]

        issues = check_total_score(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["score_mismatch"])
        self.assertIn("38 marks", issues[0].message)
        self.assertIn("out of 40", issues[0].message)
        self.assertIn("2 mark(s) unassigned", issues[0].message)

    def test_marks_over_the_total_are_named_as_over(self):
        exam = an_exam(total_score=40)

        issues = check_total_score([spec(marks="44.5")], exam=exam)

        self.assertIn("44.5 marks", issues[0].message)
        self.assertIn("4.5 mark(s) over", issues[0].message)

    def test_exact_marks_pass(self):
        exam = an_exam(total_score=40)

        self.assertEqual(check_total_score([spec(marks="40")], exam=exam), [])

    def test_half_marks_are_ordinary(self):
        exam = an_exam(total_score=40)
        rows = [spec(marks="20.5"), spec(marks="19.5", topic_id=2)]

        self.assertEqual(check_total_score(rows, exam=exam), [])


class WeightCheckTests(TestCase):
    def test_weights_summing_to_110_are_flagged_as_10_too_much(self):
        """The manual test of M4, as a unit test."""
        exam = an_exam()
        rows = [
            spec("Logic", topic_id=1, weight="60"),
            spec("Recursion", topic_id=2, weight="50"),
        ]

        issues = check_weights(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["weight_mismatch"])
        self.assertIn("110%", issues[0].message)
        self.assertIn("not 100%", issues[0].message)
        self.assertIn("10% too much", issues[0].message)

    def test_weights_under_100_say_what_is_missing(self):
        exam = an_exam()
        rows = [spec("Logic", topic_id=1, weight="40"), spec("Recursion", topic_id=2, weight="35")]

        issues = check_weights(rows, exam=exam)

        self.assertIn("75%", issues[0].message)
        self.assertIn("25% missing", issues[0].message)

    def test_exactly_100_passes(self):
        exam = an_exam()
        rows = [spec(weight="33.33"), spec(weight="33.33", topic_id=2), spec(weight="33.34", topic_id=3)]

        self.assertEqual(check_weights(rows, exam=exam), [])


class TopicWithoutQuestionsCheckTests(TestCase):
    def test_a_weighted_topic_with_no_questions_is_named(self):
        exam = an_exam()
        rows = [
            spec("Propositional logic", topic_id=1, count=0, marks="10", weight="25"),
            spec("Recursion", topic_id=2, count=20, marks="30", weight="75"),
        ]

        issues = check_topic_has_questions(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["topic_without_questions"])
        self.assertIn("Propositional logic", issues[0].message)
        self.assertIn("25%", issues[0].message)
        self.assertIn("no questions", issues[0].message)
        self.assertEqual(issues[0].topic_id, 1)

    def test_a_topic_whose_rows_together_have_questions_is_fine(self):
        """One row of zero is not a hole if a sibling row covers the topic."""
        exam = an_exam()
        rows = [
            spec("Logic", topic_id=1, count=0, marks="0", weight="0"),
            spec("Logic", topic_id=1, count=20, marks="40", weight="100"),
        ]

        self.assertEqual(check_topic_has_questions(rows, exam=exam), [])

    def test_an_unweighted_topic_with_no_questions_is_not_an_error(self):
        """A row zeroed out on purpose is a decision, not a mistake."""
        exam = an_exam()
        rows = [spec("Ethics", topic_id=1, count=0, marks="0", weight="0")]

        self.assertEqual(check_topic_has_questions(rows, exam=exam), [])


class MarksMatchWeightsCheckTests(TestCase):
    def test_the_brief_example_is_caught(self):
        """30% of a 40-mark exam is 12 marks. 10 is not."""
        exam = an_exam(total_score=40)
        rows = [spec("Recursion", topic_id=1, count=5, marks="10", weight="30")]

        issues = check_marks_match_weights(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["marks_weight_mismatch"])
        message = issues[0].message
        self.assertIn("Recursion", message)
        self.assertIn("30%", message)
        self.assertIn("40-mark", message)
        self.assertIn("12 marks", message)
        self.assertIn("carry 10", message)

    def test_a_blueprint_can_sum_perfectly_and_still_be_caught_here(self):
        """The failure this check exists for: right totals, wrong distribution."""
        exam = an_exam(total_score=40, question_count=20)
        rows = [
            spec("Logic", topic_id=1, count=10, marks="10", weight="30"),
            spec("Recursion", topic_id=2, count=10, marks="30", weight="70"),
        ]

        report = validate(rows, exam=exam)

        self.assertEqual(report.total_marks, Decimal("40.00"))
        self.assertEqual(report.total_weight, Decimal("100.00"))
        self.assertEqual(report.total_count, 20)
        self.assertEqual(report.codes, ["marks_weight_mismatch", "marks_weight_mismatch"])

    def test_marks_matching_the_weight_pass(self):
        exam = an_exam(total_score=40)
        rows = [spec("Recursion", topic_id=1, marks="12", weight="30")]

        self.assertEqual(check_marks_match_weights(rows, exam=exam), [])

    def test_rounding_within_half_a_mark_is_not_a_mismatch(self):
        """13 marks of a 40-mark exam is 32.5%; a third is 33.33%. Both are 13."""
        exam = an_exam(total_score=30)
        rows = [spec("Logic", topic_id=1, marks="10", weight="33.34")]

        self.assertEqual(check_marks_match_weights(rows, exam=exam), [])

    def test_a_whole_mark_off_is_a_mismatch(self):
        exam = an_exam(total_score=40)
        rows = [spec("Logic", topic_id=1, marks="11", weight="30")]

        issues = check_marks_match_weights(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["marks_weight_mismatch"])
        self.assertGreater(Decimal("12") - Decimal("11"), MARK_TOLERANCE)


class ExcludedTopicCheckTests(TestCase):
    def test_a_row_on_an_excluded_topic_is_refused_by_name(self):
        exam = an_exam()
        rows = [spec("Ethics", topic_id=1, excluded=True, count=20, marks="40", weight="100")]

        issues = check_topics_are_usable(rows, exam=exam)

        self.assertEqual([i.code for i in issues], ["excluded_topic"])
        self.assertIn("Ethics", issues[0].message)
        self.assertIn("not taught in lectures", issues[0].message)

    def test_usable_topics_pass(self):
        exam = an_exam()

        self.assertEqual(check_topics_are_usable([spec()], exam=exam), [])


class EmptyBlueprintTests(TestCase):
    def test_no_rows_is_its_own_error(self):
        exam = an_exam(question_count=20, total_score=40)

        issues = check_rows_exist([], exam=exam)

        self.assertEqual([i.code for i in issues], ["no_rows"])
        self.assertIn("20 questions", issues[0].message)
        self.assertIn("40 marks", issues[0].message)

    def test_an_empty_blueprint_says_one_thing_not_seven(self):
        """Every other check also fails on no rows. Saying so is noise."""
        exam = an_exam()

        report = validate([], exam=exam)

        self.assertEqual(report.codes, ["no_rows"])


class ValidateBlueprintTests(TestCase):
    """The same checks, against what is actually saved."""

    def setUp(self):
        self.course = a_course()
        self.exam = an_exam(self.course, total_score=40, question_count=20)
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.topic = Topic.objects.create(course=self.course, name="Logic")

    def _row(self, **kwargs):
        defaults = {
            "blueprint": self.blueprint,
            "topic": self.topic,
            "count": 20,
            "marks": Decimal("40"),
            "weight_percent": Decimal("100"),
        }
        return BlueprintRow.objects.create(**{**defaults, **kwargs})

    def test_a_saved_blueprint_that_adds_up_passes(self):
        self._row()

        self.assertTrue(validate_blueprint(self.blueprint).is_valid)

    def test_excluding_a_topic_after_the_blueprint_was_built_is_caught(self):
        """The row survives the exclusion; the validator is what notices."""
        self._row()
        self.topic.excluded = True
        self.topic.save()

        report = validate_blueprint(self.blueprint)

        self.assertTrue(report.has("excluded_topic"))
        self.assertIn("Logic", report.issues[0].message)

    def test_a_row_under_an_excluded_chapter_is_caught_too(self):
        chapter = Topic.objects.create(course=self.course, name="Chapter 3", excluded=True)
        self.topic.parent = chapter
        self.topic.save()
        self._row()

        self.assertTrue(validate_blueprint(self.blueprint).has("excluded_topic"))

    def test_marks_per_question_is_the_row_divided_by_its_count(self):
        row = self._row(count=8, marks=Decimal("20"))

        self.assertEqual(row.marks_per_question, Decimal("2.50"))

    def test_a_row_with_no_questions_is_worth_nothing_per_question(self):
        row = self._row(count=0)

        self.assertEqual(row.marks_per_question, Decimal("0"))


class EligibleTopicTests(TestCase):
    """Which topics a blueprint may draw on at all."""

    def setUp(self):
        self.course = a_course()

    def test_an_excluded_topic_is_not_eligible(self):
        Topic.objects.create(course=self.course, name="Logic")
        Topic.objects.create(course=self.course, name="Ethics", excluded=True)

        self.assertEqual([t.name for t in eligible_topics(self.course)], ["Logic"])

    def test_a_subtopic_of_an_excluded_chapter_is_not_eligible(self):
        chapter = Topic.objects.create(course=self.course, name="Chapter 3", excluded=True)
        Topic.objects.create(course=self.course, name="3.1 Trees", parent=chapter)
        Topic.objects.create(course=self.course, name="Chapter 4")

        self.assertEqual([t.name for t in eligible_topics(self.course)], ["Chapter 4"])

    def test_a_chapter_with_usable_subtopics_is_represented_by_them(self):
        chapter = Topic.objects.create(course=self.course, name="Chapter 1", position=0)
        Topic.objects.create(course=self.course, name="1.1 Sets", parent=chapter, position=1)
        Topic.objects.create(course=self.course, name="1.2 Relations", parent=chapter, position=2)

        self.assertEqual(
            [t.name for t in eligible_topics(self.course)], ["1.1 Sets", "1.2 Relations"]
        )

    def test_a_chapter_whose_only_subtopic_is_excluded_stands_for_itself(self):
        chapter = Topic.objects.create(course=self.course, name="Chapter 1", position=0)
        Topic.objects.create(
            course=self.course, name="1.1 Sets", parent=chapter, position=1, excluded=True
        )

        self.assertEqual([t.name for t in eligible_topics(self.course)], ["Chapter 1"])


class AutoBuildTests(TestCase):
    """The first draft: equal weight, and sound arithmetic on arrival."""

    def setUp(self):
        self.course = a_course()
        self.topics = [
            Topic.objects.create(course=self.course, name=name, position=i)
            for i, name in enumerate(["Logic", "Recursion", "Graphs"])
        ]

    def test_an_auto_built_blueprint_passes_its_own_validation(self):
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        self.assertTrue(validate_blueprint(blueprint).is_valid, validate_blueprint(blueprint).codes)

    def test_it_splits_evenly_and_sums_exactly_when_nothing_divides(self):
        """40 marks over 3 topics is 14/13/13 — not 13.33 three times."""
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        rows = list(blueprint.rows.all())
        self.assertEqual([r.count for r in rows], [7, 7, 6])
        self.assertEqual([int(r.marks) for r in rows], [14, 13, 13])
        self.assertEqual(sum(r.count for r in rows), 20)
        self.assertEqual(sum(r.marks for r in rows), Decimal("40"))
        self.assertEqual(sum(r.weight_percent for r in rows), Decimal("100"))

    def test_weights_are_the_share_the_marks_actually_are(self):
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        rows = list(blueprint.rows.all())
        self.assertEqual([str(r.weight_percent) for r in rows], ["35.00", "32.50", "32.50"])

    def test_it_stays_sound_across_awkward_numbers(self):
        """Whatever the split, the draft it produces must never arrive flagged."""
        for total_score, question_count, topic_count in [
            (10, 3, 3), (37, 7, 3), (100, 25, 3), (40, 20, 2), (13, 13, 3), (50, 4, 3)
        ]:
            with self.subTest(total_score=total_score, questions=question_count, topics=topic_count):
                exam = an_exam(
                    self.course, total_score=total_score, question_count=question_count
                )

                blueprint = auto_build(exam, topics=self.topics[:topic_count])

                report = validate_blueprint(blueprint)
                self.assertTrue(report.is_valid, f"{report.codes}: {[i.message for i in report.issues]}")

    def test_it_never_builds_a_row_on_an_excluded_topic(self):
        self.topics[1].excluded = True
        self.topics[1].save()
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        names = [row.topic.name for row in blueprint.rows.all()]
        self.assertEqual(names, ["Logic", "Graphs"])
        self.assertTrue(validate_blueprint(blueprint).is_valid)

    def test_it_never_builds_a_row_on_a_subtopic_of_an_excluded_chapter(self):
        chapter = Topic.objects.create(
            course=self.course, name="Chapter 9", position=10, excluded=True
        )
        Topic.objects.create(course=self.course, name="9.1 Out of index", parent=chapter, position=11)
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        names = [row.topic.name for row in blueprint.rows.all()]
        self.assertNotIn("9.1 Out of index", names)
        self.assertNotIn("Chapter 9", names)
        self.assertEqual(names, ["Logic", "Recursion", "Graphs"])

    def test_fewer_questions_than_topics_leaves_the_rest_out_rather_than_at_zero(self):
        """A topic carried at zero questions is an error, not a plan."""
        exam = an_exam(self.course, total_score=10, question_count=2)

        blueprint = auto_build(exam)

        self.assertEqual(blueprint.rows.count(), 2)
        self.assertTrue(validate_blueprint(blueprint).is_valid)

    def test_a_course_with_nothing_usable_builds_no_rows(self):
        for topic in self.topics:
            topic.excluded = True
            topic.save()
        exam = an_exam(self.course)

        blueprint = auto_build(exam)

        self.assertEqual(blueprint.rows.count(), 0)

    def test_rebuilding_replaces_the_rows_rather_than_adding_to_them(self):
        exam = an_exam(self.course, total_score=40, question_count=20)
        auto_build(exam)

        blueprint = auto_build(exam)

        self.assertEqual(blueprint.rows.count(), 3)
        self.assertTrue(blueprint.is_auto_built)

    def test_every_auto_built_row_carries_at_least_one_question(self):
        exam = an_exam(self.course, total_score=40, question_count=20)

        blueprint = auto_build(exam)

        self.assertTrue(all(row.count >= 1 for row in blueprint.rows.all()))
