"""M4's screens: the exam spec and the blueprint editor.

Two rules are asserted here more than anything else:

* **Every screen belongs to one instructor.** Another instructor's exam is not
  forbidden, it does not exist.
* **Live validation writes nothing.** The endpoint the editor calls on every
  keystroke must be able to describe a broken blueprint without saving one, so
  each test that posts nonsense also asserts the database did not move.
"""

from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import Blueprint, BlueprintRow, Exam

PASSWORD = "quiet-precision-42"
DIM = 1536


def unit(axis=0):
    vector = [0.0] * DIM
    vector[axis] = 1.0
    return vector


class OneAxisProvider:
    name = "one-axis"

    def embed(self, texts):
        return [unit() for _ in texts]

    def complete(self, *args, **kwargs):  # pragma: no cover - the failure is the point
        raise AssertionError("The blueprint screens must never call a completion model.")


class ExamScreenFixture(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, name="Discrete maths", code="CS310"
        )
        self.logic = Topic.objects.create(course=self.course, name="Logic", position=0)
        self.recursion = Topic.objects.create(course=self.course, name="Recursion", position=1)
        self.exam = Exam.objects.create(
            course=self.course, total_score=40, question_count=20, duration_minutes=60
        )
        self.client.login(username="nadia", password=PASSWORD)

    def url(self, name, *args):
        return reverse(f"exams:{name}", args=[self.course.pk, *args])

    def a_blueprint(self):
        blueprint = Blueprint.objects.create(exam=self.exam)
        BlueprintRow.objects.create(
            blueprint=blueprint,
            topic=self.logic,
            count=20,
            marks=Decimal("40"),
            weight_percent=Decimal("100"),
        )
        return blueprint

    def table_post(self, rows, *, total=None):
        """The editor's table, as the browser posts it."""
        data = {
            "rows-TOTAL_FORMS": str(total if total is not None else len(rows)),
            "rows-INITIAL_FORMS": "0",
            "rows-MIN_NUM_FORMS": "0",
            "rows-MAX_NUM_FORMS": "1000",
        }
        for index, row in enumerate(rows):
            for field, value in row.items():
                data[f"rows-{index}-{field}"] = str(value)
        return data


class AuthorisationTests(ExamScreenFixture):
    def test_every_screen_needs_a_login(self):
        self.client.logout()

        for name, args in [
            ("list", ()),
            ("blueprint", (self.exam.pk,)),
            ("plan", (self.exam.pk,)),
        ]:
            with self.subTest(view=name):
                response = self.client.get(self.url(name, *args))
                self.assertEqual(response.status_code, 302)
                self.assertIn("/login/", response["Location"])

    def test_the_live_validation_endpoint_needs_a_login_too(self):
        self.client.logout()

        response = self.client.post(self.url("blueprint_validate", self.exam.pk), {})

        self.assertEqual(response.status_code, 302)

    def test_another_instructors_exam_does_not_exist(self):
        other = User.objects.create_user("omar", password=PASSWORD)
        their_course = Course.objects.create(instructor=other, name="OOP", code="CS201")
        their_exam = Exam.objects.create(course=their_course)

        response = self.client.get(
            reverse("exams:blueprint", args=[their_course.pk, their_exam.pk])
        )

        self.assertEqual(response.status_code, 404)

    def test_validation_is_post_only(self):
        response = self.client.get(self.url("blueprint_validate", self.exam.pk))

        self.assertEqual(response.status_code, 405)


class ExamSpecTests(ExamScreenFixture):
    def test_the_spec_screen_lists_this_instructors_exams(self):
        response = self.client.get(self.url("list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Midterm")

    def test_defining_an_exam_leads_straight_to_its_blueprint(self):
        response = self.client.post(
            self.url("list"),
            {
                "title": "Midterm — week 7",
                "kind": Exam.Kind.MIDTERM,
                "total_score": 50,
                "question_count": 25,
                "duration_minutes": 90,
                "language": Exam.Language.ENGLISH,
                "number_of_forms": 2,
            },
        )

        exam = Exam.objects.get(title="Midterm — week 7")
        self.assertRedirects(response, self.url("blueprint", exam.pk))
        self.assertEqual(exam.course, self.course)
        self.assertEqual(exam.total_score, 50)

    def test_an_exam_out_of_nothing_is_refused(self):
        response = self.client.post(
            self.url("list"),
            {
                "kind": Exam.Kind.QUIZ,
                "total_score": 0,
                "question_count": 10,
                "duration_minutes": 30,
                "language": Exam.Language.ENGLISH,
                "number_of_forms": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "out of at least one mark")
        self.assertEqual(Exam.objects.filter(kind=Exam.Kind.QUIZ).count(), 0)


class BlueprintEditorTests(ExamScreenFixture):
    def test_an_exam_with_no_blueprint_invites_building_one(self):
        response = self.client.get(self.url("blueprint", self.exam.pk))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No blueprint yet")
        self.assertContains(response, "Build from topics")

    def test_a_course_with_no_usable_topics_says_so_instead(self):
        for topic in [self.logic, self.recursion]:
            topic.excluded = True
            topic.save()

        response = self.client.get(self.url("blueprint", self.exam.pk))

        self.assertContains(response, "no topics a blueprint can draw on")

    def test_auto_build_fills_the_table_and_it_adds_up(self):
        response = self.client.post(self.url("blueprint_autobuild", self.exam.pk))

        self.assertRedirects(response, self.url("blueprint", self.exam.pk))
        blueprint = self.exam.blueprint
        self.assertEqual(blueprint.rows.count(), 2)
        self.assertEqual(sum(row.count for row in blueprint.rows.all()), 20)
        self.assertEqual(sum(row.marks for row in blueprint.rows.all()), Decimal("40"))

    def test_the_editor_shows_the_saved_rows_and_a_verdict(self):
        self.a_blueprint()

        response = self.client.get(self.url("blueprint", self.exam.pk))

        self.assertContains(response, "Logic")
        self.assertContains(response, "bp-totals")
        self.assertContains(response, "This blueprint adds up")

    def test_saving_the_table_writes_the_rows(self):
        blueprint = self.a_blueprint()
        row = blueprint.rows.get()

        response = self.client.post(
            self.url("blueprint", self.exam.pk),
            {
                **self.table_post(
                    [
                        {
                            "id": row.pk,
                            "topic": self.logic.pk,
                            "question_type": BlueprintRow.QuestionType.MCQ,
                            "level": BlueprintRow.Level.MEDIUM,
                            "count": 12,
                            "marks": "24",
                            "weight_percent": "60",
                        },
                        {
                            "topic": self.recursion.pk,
                            "question_type": BlueprintRow.QuestionType.SHORT_ANSWER,
                            "level": BlueprintRow.Level.MULTI_STEP,
                            "count": 8,
                            "marks": "16",
                            "weight_percent": "40",
                        },
                    ]
                ),
                "rows-INITIAL_FORMS": "1",
            },
        )

        self.assertRedirects(response, self.url("blueprint", self.exam.pk))
        self.assertEqual(blueprint.rows.count(), 2)
        row.refresh_from_db()
        self.assertEqual(row.count, 12)
        self.assertEqual(
            blueprint.rows.get(topic=self.recursion).question_type,
            BlueprintRow.QuestionType.SHORT_ANSWER,
        )

    def test_an_edited_blueprint_stops_claiming_to_be_auto_built(self):
        self.client.post(self.url("blueprint_autobuild", self.exam.pk))
        blueprint = self.exam.blueprint
        self.assertTrue(blueprint.is_auto_built)
        rows = list(blueprint.rows.all())

        self.client.post(
            self.url("blueprint", self.exam.pk),
            {
                **self.table_post(
                    [
                        {
                            "id": row.pk,
                            "topic": row.topic_id,
                            "question_type": row.question_type,
                            "level": row.level,
                            "count": row.count,
                            "marks": str(row.marks),
                            "weight_percent": str(row.weight_percent),
                        }
                        for row in rows
                    ]
                ),
                "rows-INITIAL_FORMS": str(len(rows)),
            },
        )

        blueprint.refresh_from_db()
        self.assertFalse(blueprint.is_auto_built)

    def test_the_topic_column_does_not_offer_an_excluded_topic(self):
        self.recursion.excluded = True
        self.recursion.save()
        self.a_blueprint()

        response = self.client.get(self.url("blueprint", self.exam.pk))

        html = response.content.decode()
        self.assertIn(f'value="{self.logic.pk}"', html)
        self.assertNotIn(f'value="{self.recursion.pk}"', html)


class LiveValidationTests(ExamScreenFixture):
    """The HTMX endpoint: same checks, unsaved input, nothing written."""

    def test_a_weight_that_sums_to_110_comes_back_flagged_by_name(self):
        """The M4 manual test, driven through the screen the instructor uses."""
        response = self.client.post(
            self.url("blueprint_validate", self.exam.pk),
            self.table_post(
                [
                    {
                        "topic": self.logic.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 10,
                        "marks": "20",
                        "weight_percent": "60",
                    },
                    {
                        "topic": self.recursion.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 10,
                        "marks": "20",
                        "weight_percent": "50",
                    },
                ]
            ),
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "110")
        self.assertContains(response, "10% too much")
        self.assertContains(response, "status--danger")

    def test_live_validation_saves_nothing(self):
        self.client.post(
            self.url("blueprint_validate", self.exam.pk),
            self.table_post(
                [
                    {
                        "topic": self.logic.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 99,
                        "marks": "999",
                        "weight_percent": "410",
                    }
                ]
            ),
        )

        self.assertEqual(BlueprintRow.objects.count(), 0)
        self.assertEqual(Blueprint.objects.count(), 0)

    def test_a_table_that_adds_up_comes_back_clean(self):
        response = self.client.post(
            self.url("blueprint_validate", self.exam.pk),
            self.table_post(
                [
                    {
                        "topic": self.logic.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 20,
                        "marks": "40",
                        "weight_percent": "100",
                    }
                ]
            ),
        )

        self.assertContains(response, "adds up")
        self.assertContains(response, "status--ok")
        self.assertNotContains(response, "status--danger")

    def test_the_empty_row_waiting_at_the_bottom_is_not_a_row(self):
        response = self.client.post(
            self.url("blueprint_validate", self.exam.pk),
            self.table_post(
                [
                    {
                        "topic": self.logic.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 20,
                        "marks": "40",
                        "weight_percent": "100",
                    },
                    {"topic": "", "count": "", "marks": "", "weight_percent": ""},
                ]
            ),
        )

        self.assertContains(response, "adds up")

    def test_a_cell_that_is_not_a_number_is_named_rather_than_counted_as_zero(self):
        response = self.client.post(
            self.url("blueprint_validate", self.exam.pk),
            self.table_post(
                [
                    {
                        "topic": self.logic.pk,
                        "question_type": BlueprintRow.QuestionType.MCQ,
                        "level": BlueprintRow.Level.MEDIUM,
                        "count": 20,
                        "marks": "4o",
                        "weight_percent": "100",
                    }
                ]
            ),
        )

        self.assertContains(response, "is not a number")

    def test_an_empty_table_says_there_is_nothing_to_check(self):
        response = self.client.post(
            self.url("blueprint_validate", self.exam.pk), self.table_post([])
        )

        self.assertContains(response, "no rows")


class PlanScreenTests(ExamScreenFixture):
    """Agent 1A's output, on screen."""

    def setUp(self):
        super().setUp()
        self.source_file = SourceFile.objects.create(
            course=self.course,
            original_name="lecture.pdf",
            kind=SourceFile.Kind.PDF,
            page_count=10,
        )
        Chunk.objects.create(
            source_file=self.source_file,
            page=3,
            position=0,
            text="Truth tables assign a value to every row.",
            embedding=unit(),
            topic=self.logic,
        )

    def test_a_valid_blueprint_shows_a_bundle_for_every_planned_question(self):
        self.a_blueprint()

        with patch("agents.provider.get_provider", return_value=OneAxisProvider()):
            response = self.client.get(self.url("plan", self.exam.pk))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "reference passage")
        self.assertContains(response, "Truth tables assign a value to every row.")
        self.assertEqual(response.context["plan"].question_count, 20)

    def test_an_invalid_blueprint_is_sent_back_to_be_fixed(self):
        blueprint = Blueprint.objects.create(exam=self.exam)
        BlueprintRow.objects.create(
            blueprint=blueprint,
            topic=self.logic,
            count=3,  # the exam wants 20
            marks=Decimal("40"),
            weight_percent=Decimal("100"),
        )

        response = self.client.get(self.url("plan", self.exam.pk))

        self.assertRedirects(response, self.url("blueprint", self.exam.pk))

    def test_an_exam_with_no_blueprint_is_sent_to_build_one(self):
        response = self.client.get(self.url("plan", self.exam.pk))

        self.assertRedirects(response, self.url("blueprint", self.exam.pk))
