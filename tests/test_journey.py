"""M11.5: the journey is walkable with buttons — including the one that was missing.

Every milestone up to M11 worked when driven from a management command or a
typed URL. This file asserts the thing an instructor actually needs: that each
screen carries a control to the next one, and that pressing Generate runs M8's
loop over this exam's blueprint and lands on Review with what it wrote.

No provider here is real. `get_provider` is patched for the whole generate path,
so the loop runs end to end — retrieval, Agent 2A, Agent 3A — and makes no
network call and spends no credit, which has been the rule since M2.
"""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from agents.prompts.generate import SYSTEM as GENERATE_SYSTEM
from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Question
from exams.templatetags.journey import STEPS

from tests.test_generate import DIM, mcq, payload
from tests.test_review import all_ok

PASSWORD = "not-a-real-password"


class JourneyProvider:
    """Both agents and the embedder, faked. Told apart by the system prompt.

    `embed` returns the one axis every stored chunk also sits on, so retrieval
    ranks and returns them rather than depending on a real vector space.
    """

    name = "journey"

    def __init__(self, *batches):
        self.batches = list(batches)
        self.generation_calls = 0

    def embed(self, texts):
        vector = [0.0] * DIM
        vector[0] = 1.0
        return [vector for _ in texts]

    def complete(self, system, user, **kwargs):
        from agents.provider import LLMResponse

        if system == GENERATE_SYSTEM:
            self.generation_calls += 1
            if not self.batches:
                raise AssertionError("The loop generated more times than expected.")
            return LLMResponse(text=payload(*self.batches.pop(0)))
        return LLMResponse(text=json.dumps(all_ok()))


def batch(*stems):
    return [mcq(stem=stem) for stem in stems]


class JourneyFixture(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, name="Data Structures", code="CS210"
        )
        self.topic = Topic.objects.create(course=self.course, name="Binary search", position=0)
        self.exam = Exam.objects.create(
            course=self.course,
            total_score=4,
            question_count=2,
            duration_minutes=60,
            number_of_forms=2,
        )
        self.client.login(username="nadia", password=PASSWORD)

    # --- building blocks -------------------------------------------------

    def url(self, name, *args):
        return reverse(f"exams:{name}", args=[self.course.pk, *args])

    def a_chunk(self):
        source_file, _ = SourceFile.objects.get_or_create(
            course=self.course,
            original_name="lecture-3.pdf",
            defaults=dict(kind=SourceFile.Kind.PDF, page_count=12),
        )
        vector = [0.0] * DIM
        vector[0] = 1.0
        return Chunk.objects.create(
            source_file=source_file,
            page=7,
            position=0,
            text="A binary search halves the interval at every step.",
            embedding=vector,
        )

    def a_blueprint(self, *, count=2, marks="4.00"):
        board = Blueprint.objects.create(exam=self.exam)
        BlueprintRow.objects.create(
            blueprint=board,
            topic=self.topic,
            count=count,
            marks=Decimal(marks),
            weight_percent=Decimal("100.00"),
        )
        return board

    def a_question(self, stem="What does binary search halve?", **kwargs):
        """One question on this exam, as the review screen sees it."""
        return Question.objects.create(
            exam=self.exam,
            stem=stem,
            question_type=BlueprintRow.QuestionType.SHORT_ANSWER,
            correct="The interval",
            source_ref="lecture-3.pdf · page 7",
            answer_key={"answer": "The interval", "elements": []},
            **kwargs,
        )

    def a_reviewed_pool(self):
        """A pool built the way the product builds one: by pressing Generate.

        Assembly reads questions that passed Agent 3A, so a pool assembled by
        hand would be testing a state the journey cannot produce.
        """
        self.a_chunk()
        self.a_blueprint()
        stems = [f"Binary search question {index}" for index in range(4)]
        self.press_generate(JourneyProvider(batch(*stems)))
        return list(self.exam.questions.all())

    def press_generate(self, provider):
        with patch("agents.provider.get_provider", return_value=provider):
            return self.client.post(self.url("generate", self.exam.pk))


# --- The gap M11.5 exists to close --------------------------------------------


class GenerateButtonTests(JourneyFixture):
    def test_the_generate_button_runs_the_loop_and_review_shows_what_it_wrote(self):
        self.a_chunk()
        self.a_blueprint()
        provider = JourneyProvider(batch("What does binary search halve?", "Halving: how many steps?"))

        response = self.press_generate(provider)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], self.url("review", self.exam.pk))
        self.assertTrue(provider.generation_calls, "The loop never reached Agent 2A.")
        self.assertEqual(self.exam.questions.count(), 2)

        landed = self.client.get(response["Location"])
        self.assertContains(landed, "What does binary search halve?")

    def test_nothing_generated_is_an_exam_question_until_the_instructor_says_so(self):
        self.a_chunk()
        self.a_blueprint()

        self.press_generate(JourneyProvider(batch("Q1", "Q2")))

        self.assertEqual(
            set(self.exam.questions.values_list("status", flat=True)),
            {Question.Status.CANDIDATE},
        )

    def test_generate_is_refused_on_a_blueprint_that_does_not_add_up(self):
        self.a_chunk()
        self.a_blueprint(count=99)  # the marks no longer match the count
        provider = JourneyProvider()

        response = self.press_generate(provider)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], self.url("blueprint", self.exam.pk))
        self.assertEqual(provider.generation_calls, 0)
        self.assertEqual(self.exam.questions.count(), 0)
        self.assertIn("does not add up", self._only_message(response))

    def test_generate_is_refused_when_nothing_has_been_saved_to_generate_from(self):
        provider = JourneyProvider()

        response = self.press_generate(provider)

        self.assertEqual(response["Location"], self.url("blueprint", self.exam.pk))
        self.assertEqual(provider.generation_calls, 0)
        self.assertIn("no blueprint", self._only_message(response))

    def test_a_row_with_no_passages_comes_back_saying_so_rather_than_empty(self):
        # No chunk was ever stored, so retrieval has nothing to hand Agent 2A.
        self.a_blueprint()
        provider = JourneyProvider()

        response = self.press_generate(provider)

        # Not sent to an empty Review screen — kept here, with the reason.
        self.assertEqual(response["Location"], self.url("generate", self.exam.pk))
        self.assertEqual(self.exam.questions.count(), 0)
        message = self._only_message(response)
        self.assertIn("needs manual attention", message)
        self.assertIn("Binary search", message)
        self.assertIn("no reference passages", message)

    def test_the_blueprint_screen_carries_the_button(self):
        self.a_chunk()
        self.a_blueprint()

        response = self.client.get(self.url("blueprint", self.exam.pk))

        self.assertContains(response, self.url("generate", self.exam.pk))
        self.assertContains(response, "Generate questions")
        self.assertNotContains(response, "disabled")

    def test_the_button_is_disabled_while_the_blueprint_is_not_ready(self):
        self.a_blueprint(count=99)

        response = self.client.get(self.url("generate", self.exam.pk))

        self.assertContains(response, "disabled")
        self.assertContains(response, "Fix the blueprint")

    def test_the_button_carries_the_waiting_state_it_needs(self):
        """The press blocks for minutes; a screen that looks frozen gets reloaded,
        and a reload here spends the calls a second time."""
        self.a_chunk()
        self.a_blueprint()

        response = self.client.get(self.url("generate", self.exam.pk))

        self.assertContains(response, "data-generate-button")
        self.assertContains(response, 'class="spinner"')
        self.assertContains(response, "js/generate.js")
        self.assertContains(response, "leave this page open")

    def test_the_generate_screen_needs_a_login_and_belongs_to_one_instructor(self):
        other = User.objects.create_user("omar", password=PASSWORD)
        their_course = Course.objects.create(instructor=other, name="OOP", code="CS201")
        their_exam = Exam.objects.create(course=their_course)

        response = self.client.get(
            reverse("exams:generate", args=[their_course.pk, their_exam.pk])
        )
        self.assertEqual(response.status_code, 404)

        self.client.logout()
        response = self.client.get(self.url("generate", self.exam.pk))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    def _only_message(self, response):
        from django.contrib.messages import get_messages

        return " ".join(str(m) for m in get_messages(response.wsgi_request))


# --- The spine ----------------------------------------------------------------


class StepperTests(JourneyFixture):
    """The stepper is the map. Every step it shows has to be somewhere to go."""

    def test_the_current_step_is_marked_and_everything_before_it_is_done(self):
        self.a_blueprint()

        response = self.client.get(self.url("blueprint", self.exam.pk))
        html = response.content.decode()

        self.assertIn('data-step="blueprint"', html)
        self.assertIn('aria-current="step"', html)
        # Course, Upload, Topics and Spec come before Blueprint.
        for slug in ("course", "upload", "topics", "spec"):
            self.assertRegex(html, rf'class="is-done"[^>]*data-step="{slug}"')
        # Generate comes after it, so it is neither done nor current.
        self.assertRegex(html, r'class=""\s+data-step="generate"')

    def test_every_step_of_an_exam_journey_is_a_link_that_resolves(self):
        self.a_blueprint()

        response = self.client.get(self.url("review", self.exam.pk))
        html = response.content.decode()

        for slug, label, name, scope in STEPS:
            with self.subTest(step=slug):
                args = (
                    [self.course.pk, self.exam.pk] if scope == "exam" else [self.course.pk]
                )
                target = reverse(name, args=args)
                if slug == "review":
                    continue  # the current step is text, by design
                self.assertIn(f'href="{target}', html)

    def test_the_exam_steps_are_plain_text_before_an_exam_exists(self):
        response = self.client.get(reverse("courses:topics", args=[self.course.pk]))
        html = response.content.decode()

        self.assertIn('data-step="generate"', html)
        # Nothing to point at, so nothing pretends to.
        self.assertNotIn("/generate/", html)

    def test_every_screen_of_the_journey_shows_the_stepper(self):
        self.a_chunk()
        self.a_blueprint()
        self.a_question()

        screens = [
            reverse("courses:detail", args=[self.course.pk]),
            reverse("courses:topics", args=[self.course.pk]),
            self.url("list"),
            self.url("blueprint", self.exam.pk),
            self.url("generate", self.exam.pk),
            self.url("review", self.exam.pk),
            self.url("forms", self.exam.pk),
            self.url("compare", self.exam.pk),
        ]
        for screen in screens:
            with self.subTest(screen=screen):
                response = self.client.get(screen)
                self.assertContains(response, 'class="stepper"')


# --- No screen is a dead end --------------------------------------------------


class ForwardControlTests(JourneyFixture):
    def test_review_offers_the_way_on_to_forms_compare_and_export(self):
        self.a_question()

        response = self.client.get(self.url("review", self.exam.pk))

        self.assertContains(response, self.url("forms", self.exam.pk))
        self.assertContains(response, self.url("compare", self.exam.pk))
        self.assertContains(response, self.url("export", self.exam.pk))

    def test_an_exam_with_no_questions_sends_the_instructor_to_generate(self):
        self.a_blueprint()

        response = self.client.get(self.url("review", self.exam.pk))

        self.assertContains(response, "Generate questions")
        self.assertContains(response, self.url("generate", self.exam.pk))

    def test_the_assemble_button_saves_the_forms_and_compare_reads_them(self):
        self.a_reviewed_pool()

        screen = self.client.get(self.url("forms", self.exam.pk))
        self.assertContains(screen, "Assemble &amp; save these forms")

        saved = self.client.post(self.url("forms", self.exam.pk))
        self.assertEqual(saved.status_code, 302)
        self.assertTrue(self.exam.forms.exists())

        landed = self.client.get(self.url("forms", self.exam.pk))
        self.assertContains(landed, self.url("compare", self.exam.pk))

    def test_compare_carries_a_button_to_the_export(self):
        self.a_reviewed_pool()
        self.client.post(self.url("forms", self.exam.pk))

        response = self.client.get(self.url("compare", self.exam.pk))

        self.assertContains(response, self.url("export", self.exam.pk))

    def test_export_is_reached_by_button_and_says_so_when_there_is_nothing_to_export(self):
        self.a_question()

        response = self.client.get(self.url("export", self.exam.pk))

        # No saved forms: sent to the screen that can make them, not to an error.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], self.url("forms", self.exam.pk))

    def test_an_exam_opens_at_the_step_it_is_actually_at(self):
        # No blueprint: the only way on is to plan it.
        response = self.client.get(self.url("list"))
        self.assertContains(response, self.url("blueprint", self.exam.pk))
        self.assertNotContains(response, self.url("generate", self.exam.pk))

        # A blueprint but nothing written: Generate.
        self.a_blueprint()
        response = self.client.get(self.url("list"))
        self.assertContains(response, self.url("generate", self.exam.pk))
        self.assertNotContains(response, self.url("review", self.exam.pk))

        # Questions written: Review.
        self.a_question()
        response = self.client.get(self.url("list"))
        self.assertContains(response, self.url("review", self.exam.pk))

    def test_the_course_screen_leads_to_its_exams(self):
        SourceFile.objects.create(
            course=self.course, original_name="lecture.pdf",
            kind=SourceFile.Kind.PDF, page_count=3,
        )

        response = self.client.get(reverse("courses:detail", args=[self.course.pk]))

        self.assertContains(response, self.url("list"))
