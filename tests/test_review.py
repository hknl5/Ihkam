"""M7: Agent 3A judges a question, and says what a replacement would have to do.

Every provider here is a fake: the suite makes no network call and spends no API
credit, the rule since M2.

What is pinned down:

* the deterministic checks are decided **in Python** — two correct options, a
  repeated option, a conspicuously long correct option and a mark split that
  does not add up are all caught with no provider present at all;
* the language checks are decided by the model, and their rejections arrive as
  notes that name the failing check *and* what the replacement must do;
* the plan's two worked examples are both caught with the right reason: a
  multi-step row that came back as a definition, and a question that depends on
  something the passages never said;
* review never rewrites anything;
* a malformed review is retried once, then raised — an unreviewed question is
  not a passed one.
"""

import json
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from agents.answer_key import NumericKey, ObjectiveKey, ShortAnswerKey
from agents.generate import GenerationItem, generate_candidates
from agents.prompts.review import ACTIONABLE_RULE, JUDGE_ONLY_RULE
from agents.review import (
    ANSWER_CORRECTNESS,
    BY_MODEL,
    BY_PYTHON,
    CLARITY,
    CONTENT_LINK,
    LEVEL_MATCH,
    OPTION_QUALITY,
    QuestionReviewError,
    ReviewSubject,
    build_prompt,
    deterministic_findings,
    normalise_option,
    review_candidate,
    review_question,
    review_stored_question,
)
from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Question

from tests.test_answer_key import numeric_candidate
from tests.test_generate import DIM, ScriptedProvider, a_passage, payload


def a_subject(**kwargs):
    """An MCQ that passes every deterministic check, unless a test breaks it."""
    defaults = dict(
        stem="Which step does a binary search repeat?",
        question_type="mcq",
        options=(
            "Halve the interval",
            "Scan every element",
            "Sort the list first",
            "Hash the key",
        ),
        correct="Halve the interval",
        explanation="The passage says the interval is halved at every step.",
        requested_level="medium",
        marks=Decimal("2.00"),
        topic_name="Binary search",
        course_name="Data Structures",
        passages=(a_passage(),),
        answer_key=ObjectiveKey(answer="Halve the interval", options=["Halve the interval"]),
    )
    defaults.update(kwargs)
    return ReviewSubject(**defaults)


def all_ok():
    return {
        "content_link": {"ok": True, "reason": "", "requirement": ""},
        "clarity": {"ok": True, "reason": "", "requirement": ""},
        "answer_consistency": {"ok": True, "reason": "", "requirement": ""},
        "level_match": {"ok": True, "reason": "", "requirement": ""},
        "distractor_quality": {"ok": True, "reason": "", "requirement": ""},
    }


def verdicts(**failures):
    """A clean review with the named checks failed."""
    result = all_ok()
    result.update(failures)
    return json.dumps(result)


# --- The deterministic half --------------------------------------------------


class PythonChecksTests(SimpleTestCase):
    """No provider is passed to any test here: these checks never need one."""

    def _failures(self, subject):
        return [f for f in deterministic_findings(subject) if not f.passed]

    def test_a_clean_question_passes_every_deterministic_check(self):
        self.assertEqual(self._failures(a_subject()), [])

    def test_two_correct_options_are_caught_in_python(self):
        subject = a_subject(
            options=("Halve the interval", "halve the interval.", "Sort the list", "Hash the key")
        )
        failures = self._failures(subject)

        self.assertIn(OPTION_QUALITY, [f.check for f in failures])
        finding = next(f for f in failures if "2 options" in f.reason)
        self.assertEqual(finding.decided_by, BY_PYTHON)
        self.assertIn("exactly one correct option", finding.requirement)

    def test_an_answer_that_is_not_among_the_options_is_caught(self):
        subject = a_subject(correct="Something else entirely")
        failures = self._failures(subject)

        finding = next(f for f in failures if f.check == OPTION_QUALITY)
        self.assertIn("none of the options is the stated answer", finding.reason)
        self.assertEqual(finding.decided_by, BY_PYTHON)

    def test_two_options_that_are_the_same_string_are_caught(self):
        subject = a_subject(
            options=(
                "Halve the interval",
                "Scan every element",
                "scan every element!",
                "Hash the key",
            )
        )
        finding = next(f for f in self._failures(subject) if "same thing" in f.reason)

        self.assertEqual(finding.check, OPTION_QUALITY)
        self.assertEqual(finding.decided_by, BY_PYTHON)
        self.assertIn("replace the repeated option", finding.requirement)

    def test_normalisation_is_about_presentation_not_meaning(self):
        self.assertEqual(normalise_option("A binary search."), normalise_option("a binary search"))
        self.assertEqual(normalise_option("Scan every element!"), normalise_option("scan every element"))
        self.assertNotEqual(normalise_option("halve"), normalise_option("double"))

    def test_two_boolean_expressions_are_not_one_option_repeated(self):
        """Interior symbols are the content, not presentation — caught on real material.

        Stripping punctuation everywhere collapsed `(x y)′` and `x + y` into the
        same string and reported a NAND question's options as duplicates.
        """
        subject = a_subject(
            options=("(xy)'", "(x y)′", "x + y", "x · y"),
            correct="(xy)'",
            answer_key=None,
        )
        self.assertEqual([f for f in self._failures(subject) if "same thing" in f.reason], [])

    def test_a_conspicuously_long_correct_option_is_caught(self):
        subject = a_subject(
            options=(
                "It halves the search interval at every step, discarding the half that "
                "cannot contain the key, which is why it is logarithmic",
                "It scans",
                "It sorts",
                "It hashes",
            ),
            correct=(
                "It halves the search interval at every step, discarding the half that "
                "cannot contain the key, which is why it is logarithmic"
            ),
            answer_key=None,
        )
        finding = next(f for f in self._failures(subject) if "longest" in f.reason)

        self.assertEqual(finding.check, OPTION_QUALITY)
        self.assertEqual(finding.decided_by, BY_PYTHON)
        self.assertIn("roughly the same length", finding.requirement)

    def test_a_correct_option_that_is_merely_a_little_longer_is_left_alone(self):
        subject = a_subject(
            options=(
                "It halves the search interval at every step",
                "It scans every element in order",
                "It sorts the list before searching",
                "It hashes the key into a bucket",
            ),
            correct="It halves the search interval at every step",
            answer_key=None,
        )
        self.assertEqual([f for f in self._failures(subject) if "longest" in f.reason], [])

    def test_short_options_are_not_judged_on_length(self):
        """"42" against "7" is three times the length and gives nothing away."""
        subject = a_subject(options=("42", "7", "8", "9"), correct="42", answer_key=None)
        self.assertEqual([f for f in self._failures(subject) if "longest" in f.reason], [])

    def test_a_mark_split_that_does_not_add_up_is_surfaced_from_m6(self):
        subject = a_subject(
            question_type="numeric",
            options=(),
            correct="6 gates",
            marks=Decimal("5.00"),
            answer_key=NumericKey(
                steps=[{"text": "count the NOT gates", "marks": 2}, {"text": "add", "marks": 5}],
                final_answer="6 gates",
            ),
            mark_sum_ok=False,  # M6's verdict, carried not recomputed
        )
        finding = next(f for f in self._failures(subject) if "does not add up" in f.reason)

        self.assertEqual(finding.check, ANSWER_CORRECTNESS)
        self.assertEqual(finding.decided_by, BY_PYTHON)
        self.assertIn("total 7.00", finding.reason)
        self.assertIn("split exactly 5.00 mark(s)", finding.requirement)

    def test_a_mark_split_that_adds_up_is_not_a_failure(self):
        subject = a_subject(question_type="numeric", options=(), mark_sum_ok=True, answer_key=None)
        self.assertEqual([f for f in self._failures(subject) if "add up" in f.reason], [])

    def test_a_question_with_no_answer_key_cannot_be_marked(self):
        finding = next(
            f for f in self._failures(a_subject(answer_key=None)) if f.check == ANSWER_CORRECTNESS
        )
        self.assertIn("no answer key", finding.reason)

    def test_a_short_answer_key_with_no_required_elements_is_caught(self):
        subject = a_subject(
            question_type="short_answer",
            options=(),
            correct="It halves the interval.",
            answer_key=ShortAnswerKey.model_construct(
                kind="short_answer",
                model_answer="It halves the interval.",
                required_elements=[],
                explanation="",
            ),
        )
        finding = next(f for f in self._failures(subject) if f.check == ANSWER_CORRECTNESS)
        self.assertIn("no required elements", finding.reason)

    def test_a_key_that_disagrees_with_the_question_s_own_answer_is_caught(self):
        subject = a_subject(
            answer_key=ObjectiveKey(answer="Scan every element", options=["Scan every element"])
        )
        finding = next(f for f in self._failures(subject) if f.check == ANSWER_CORRECTNESS)
        self.assertIn("a marker would not know which to use", finding.reason)

    def test_the_deterministic_half_can_run_with_no_provider_at_all(self):
        result = review_question(a_subject(correct="Not an option"), python_only=True)

        self.assertTrue(result.rejected)
        self.assertFalse(result.model_checked)
        self.assertTrue(all(f.decided_by == BY_PYTHON for f in result.findings))


# --- The language half -------------------------------------------------------


class ModelChecksTests(SimpleTestCase):
    def test_a_clean_review_passes(self):
        provider = ScriptedProvider(verdicts())
        result = review_question(a_subject(), provider=provider)

        self.assertTrue(result.passed)
        self.assertEqual(result.notes, [])
        self.assertTrue(result.model_checked)
        self.assertEqual(len(provider.calls), 1)

    def test_the_level_mismatch_example_from_the_plan_is_caught(self):
        """Asked for multi-step, got a definition — the brief's worked example."""
        provider = ScriptedProvider(
            verdicts(
                level_match={
                    "ok": False,
                    "reason": (
                        "the row asked for a multi-step question but this asks the "
                        "student to state the definition of precision, which is recall "
                        "of one sentence of the passage"
                    ),
                    "requirement": (
                        "require the student to compute precision from the confusion "
                        "matrix given in the passage, not to define it"
                    ),
                }
            )
        )
        subject = a_subject(
            stem="What is precision?",
            options=(
                "The share of predicted positives that are correct",
                "The share of actual positives that are found",
                "The share of all predictions that are correct",
                "The share of negatives that are correct",
            ),
            correct="The share of predicted positives that are correct",
            requested_level="multi_step",
            topic_name="Evaluation metrics",
            answer_key=ObjectiveKey(
                answer="The share of predicted positives that are correct",
                options=["The share of predicted positives that are correct"],
            ),
        )
        result = review_question(subject, provider=provider)

        self.assertTrue(result.rejected)
        self.assertEqual(result.failed_checks, [LEVEL_MATCH])
        self.assertIn("multi-step", result.notes[0])
        self.assertIn("compute precision from the confusion matrix", result.notes[0])

    def test_level_mismatch_note_names_the_gap_and_what_to_do(self):
        provider = ScriptedProvider(
            verdicts(
                level_match={
                    "ok": False,
                    "reason": (
                        "multi-step was requested but the question asks for a "
                        "definition, which is direct recall"
                    ),
                    "requirement": (
                        "require a computation from the values in the passage rather "
                        "than a definition"
                    ),
                }
            )
        )
        result = review_question(
            a_subject(stem="What is precision?", requested_level="multi_step"),
            provider=provider,
        )

        self.assertTrue(result.rejected)
        self.assertEqual(result.failed_checks, [LEVEL_MATCH])
        finding = result.finding_for(LEVEL_MATCH)
        self.assertEqual(finding.decided_by, BY_MODEL)
        note = result.notes[0]
        self.assertIn("Level match:", note)
        self.assertIn("definition", note)
        self.assertIn("The replacement must:", note)
        self.assertIn("computation", note)

    def test_the_out_of_scope_example_is_caught_and_named(self):
        provider = ScriptedProvider(
            verdicts(
                content_link={
                    "ok": False,
                    "reason": (
                        "the question depends on the master theorem, which none of the "
                        "supplied passages mention"
                    ),
                    "requirement": (
                        "write the question only from the supplied passages, using the "
                        "halving argument they state"
                    ),
                }
            )
        )
        result = review_question(a_subject(), provider=provider)

        self.assertEqual(result.failed_checks, [CONTENT_LINK])
        note = result.notes[0]
        self.assertIn("Content link:", note)
        self.assertIn("master theorem", note)
        self.assertIn("The replacement must:", note)

    def test_every_model_rejection_carries_an_instruction_even_when_it_omits_one(self):
        provider = ScriptedProvider(
            verdicts(clarity={"ok": False, "reason": "the stem asks two things at once"})
        )
        result = review_question(a_subject(), provider=provider)

        finding = result.finding_for(CLARITY)
        self.assertFalse(finding.passed)
        self.assertIn("exactly one reasonable interpretation", finding.requirement)
        self.assertIn("The replacement must:", finding.note)

    def test_a_rejection_with_no_reason_is_not_a_review(self):
        """It could be shown to nobody and acted on by nothing — retry instead."""
        provider = ScriptedProvider(
            verdicts(clarity={"ok": False, "reason": "", "requirement": ""}), verdicts()
        )
        result = review_question(a_subject(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(result.passed)

    def test_a_malformed_review_is_retried_once_then_raised(self):
        provider = ScriptedProvider("not json", "still not json")
        with self.assertRaises(QuestionReviewError) as caught:
            review_question(a_subject(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertIn("two attempts", str(caught.exception))
        self.assertIn("not a passed one", str(caught.exception))

    def test_a_call_that_never_reached_the_model_is_reported_as_itself(self):
        class NoCredit:
            name = "no-credit"
            calls = 0

            def complete(self, *args, **kwargs):
                type(self).calls += 1
                raise RuntimeError("429 insufficient_quota")

            def embed(self, texts):  # pragma: no cover
                raise AssertionError

        with self.assertRaises(QuestionReviewError) as caught:
            review_question(a_subject(), provider=NoCredit())

        self.assertEqual(NoCredit.calls, 1)  # not retried
        self.assertIn("insufficient_quota", str(caught.exception))

    def test_an_open_question_is_not_judged_on_distractors_it_does_not_have(self):
        provider = ScriptedProvider(
            verdicts(
                distractor_quality={
                    "ok": False,
                    "reason": "a verdict about options that do not exist",
                    "requirement": "ignore me",
                }
            )
        )
        subject = a_subject(
            question_type="short_answer",
            options=(),
            correct="It halves the interval.",
            answer_key=ShortAnswerKey(
                model_answer="It halves the interval.", required_elements=["halving"]
            ),
        )
        result = review_question(subject, provider=provider)

        self.assertTrue(result.passed)

    def test_all_the_checks_run_so_one_review_produces_one_complete_set_of_notes(self):
        provider = ScriptedProvider(
            verdicts(
                content_link={"ok": False, "reason": "outside the passages", "requirement": "x"},
                clarity={"ok": False, "reason": "two questions at once", "requirement": "y"},
            )
        )
        result = review_question(a_subject(correct="Not an option"), provider=provider)

        self.assertIn(OPTION_QUALITY, result.failed_checks)  # python
        self.assertIn(CONTENT_LINK, result.failed_checks)  # model
        self.assertIn(CLARITY, result.failed_checks)
        self.assertGreaterEqual(len(result.notes), 3)


# --- The prompt --------------------------------------------------------------


class PromptTests(SimpleTestCase):
    def test_the_reviewer_is_told_never_to_rewrite(self):
        system, user = build_prompt(a_subject())
        self.assertIn(JUDGE_ONLY_RULE, system)
        self.assertIn(JUDGE_ONLY_RULE, user)

    def test_the_reviewer_is_told_to_say_what_a_replacement_needs(self):
        system, user = build_prompt(a_subject())
        self.assertIn(ACTIONABLE_RULE, system)
        self.assertIn(ACTIONABLE_RULE, user)

    def test_the_requested_level_and_its_meaning_reach_the_model(self):
        _, user = build_prompt(a_subject(requested_level="multi_step"))
        self.assertIn("Multi-step", user)
        self.assertIn("two or more connected steps", user)

    def test_the_passages_the_question_must_come_from_reach_the_model(self):
        _, user = build_prompt(a_subject())
        self.assertIn("[P1] lecture-3.pdf · page 7", user)
        self.assertIn("halves the interval", user)

    def test_the_answer_key_is_shown_so_the_key_is_judged_too(self):
        _, user = build_prompt(
            a_subject(
                question_type="numeric",
                options=(),
                correct="6 gates",
                answer_key=NumericKey(
                    steps=[{"text": "count the gates", "marks": 2}], final_answer="6 gates"
                ),
            )
        )
        self.assertIn("Worked solution:", user)
        self.assertIn("count the gates", user)
        self.assertIn("Final answer: 6 gates", user)

    def test_review_never_returns_a_rewritten_question(self):
        """The output has no field a corrected question could arrive in."""
        from agents.review import ReviewOut

        for field_name in ReviewOut.model_fields:
            self.assertNotIn("stem", field_name)
            self.assertNotIn("option", field_name)


# --- Adapting what is reviewed ----------------------------------------------


class SubjectTests(SimpleTestCase):
    def test_a_generated_candidate_is_reviewed_against_the_item_it_came_from(self):
        item = GenerationItem(
            course_name="Discrete Maths",
            topic_name="Logic gates",
            question_type="numeric",
            level="multi_step",
            marks=Decimal("4.00"),
            count=1,
            passages=(a_passage(),),
        )
        run = generate_candidates(item, provider=ScriptedProvider(payload(numeric_candidate())))
        candidate = run.candidates[0]

        provider = ScriptedProvider(verdicts())
        result = review_candidate(candidate, item, provider=provider)

        self.assertTrue(result.passed)
        _, user = build_prompt(ReviewSubject.from_candidate(candidate, item))
        self.assertIn("Multi-step", user)
        self.assertIn("Logic gates", user)

    def test_reviewing_a_candidate_needs_no_second_generation_call(self):
        item = GenerationItem(
            course_name="Discrete Maths",
            topic_name="Logic gates",
            question_type="numeric",
            level="multi_step",
            marks=Decimal("4.00"),
            count=1,
            passages=(a_passage(),),
        )
        generation = ScriptedProvider(payload(numeric_candidate()))
        run = generate_candidates(item, provider=generation)
        review_candidate(run.candidates[0], item, provider=ScriptedProvider(verdicts()))

        self.assertEqual(len(generation.calls), 1)


class StoredQuestionTests(TestCase):
    def setUp(self):
        instructor = User.objects.create_user("hana", password="quiet-precision-42")
        self.course = Course.objects.create(
            instructor=instructor, name="Data Structures", code="CS201", content_language="en"
        )
        self.topic = Topic.objects.create(course=self.course, name="Binary search")
        self.exam = Exam.objects.create(
            course=self.course, total_score=40, question_count=20, duration_minutes=60
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.row = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.topic,
            question_type="mcq",
            level="multi_step",
            count=1,
            marks=Decimal("2.00"),
        )
        source_file = SourceFile.objects.create(
            course=self.course,
            original_name="lecture-3.pdf",
            kind=SourceFile.Kind.PDF,
            page_count=12,
        )
        self.chunk = Chunk.objects.create(
            source_file=source_file,
            topic=self.topic,
            page=7,
            position=0,
            text="A binary search halves the interval at every step.",
            embedding=[0.0] * DIM,
        )
        self.question = Question.objects.create(
            exam=self.exam,
            blueprint_row=self.row,
            stem="Which step does a binary search repeat?",
            question_type="mcq",
            options=["Halve the interval", "Scan every element", "Sort the list"],
            correct="Halve the interval",
            explanation="Stated in the passage.",
            source_ref="lecture-3.pdf · page 7",
            source_chunk=self.chunk,
            answer_key=ObjectiveKey(
                answer="Halve the interval", options=["Halve the interval"]
            ).as_dict(),
        )

    def test_a_stored_question_is_reviewed_against_its_own_passage_and_row(self):
        provider = ScriptedProvider(verdicts())
        result = review_stored_question(self.question, provider=provider)

        self.assertTrue(result.passed)
        system, user = build_prompt(ReviewSubject.from_question(self.question))
        self.assertIn("A binary search halves the interval", user)
        self.assertIn("Multi-step", user)  # the row's requested level
        self.assertIn("lecture-3.pdf · page 7", user)

    def test_review_stores_nothing_and_changes_nothing(self):
        """Agent 3A judges. It does not touch the question it judged."""
        before = Question.objects.values().get()
        review_stored_question(
            self.question,
            provider=ScriptedProvider(
                verdicts(clarity={"ok": False, "reason": "ambiguous", "requirement": "be clear"})
            ),
        )
        self.question.refresh_from_db()

        self.assertEqual(Question.objects.values().get(), before)
        self.assertEqual(Question.objects.count(), 1)
