"""M6: the answer key is produced with the question, and typed by question type.

Every provider here is a fake, as since M2: the suite makes no network call and
spends no API credit. The fakes count their calls, because the single-call
guarantee — the key arrives with the stem, never from a second call — is the
thing this milestone is actually measured on and the easiest thing to lose
quietly later.

What is pinned down:

* each type produces its own key shape, and only that shape;
* a short answer without required elements, or a numeric without steps, does not
  validate — it is not a candidate with a missing field, it is a question whose
  marking scheme someone would have to write by hand;
* the numeric mark-sum check is arithmetic in Python: it passes when the steps
  total the question's marks, flags when they do not, and never rewrites either;
* the essay key format validates, and no essay question is generated;
* one item, one provider call, keys on every candidate.
"""

import json
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from agents.answer_key import (
    ESSAY,
    NUMERIC,
    OBJECTIVE,
    SHORT_ANSWER,
    EssayKey,
    NumericKey,
    ObjectiveKey,
    ShortAnswerKey,
    answer_key_from_dict,
    check_mark_sum,
    kind_for_type,
)
from agents.generate import (
    MVP_TYPES,
    GenerationItem,
    UnsupportedQuestionType,
    build_prompt,
    generate_candidates,
    item_for_row,
    save_candidates,
)
from agents.prompts.generate import ANSWER_KEY_RULE, KEY_RULES
from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Question

from tests.test_generate import DIM, ScriptedProvider, a_passage, an_item, payload


# --- The candidates the model returns ----------------------------------------


def mcq_candidate(source_ref="P1"):
    return {
        "stem": "Which step does a binary search repeat?",
        "type": "mcq",
        "options": ["Halve the interval", "Scan every element", "Sort the list", "Hash the key"],
        "correct": "Halve the interval",
        "explanation": "The passage says the interval is halved at every step.",
        "source_ref": source_ref,
        "answer_key": {"answer": "Halve the interval"},
    }


def true_false_candidate():
    return {
        "stem": "A binary search halves the interval at every step.",
        "type": "true_false",
        "options": [],
        "correct": "true",
        "explanation": "Stated in the passage.",
        "source_ref": "P1",
        "answer_key": {"answer": "True"},
    }


def short_answer_candidate(elements=("halving the interval", "on a sorted list")):
    return {
        "stem": "What does a binary search do at every step, and what does it require?",
        "type": "short_answer",
        "options": [],
        "correct": "It halves the search interval, which requires the list to be sorted.",
        "explanation": "Both are stated in the passage.",
        "source_ref": "P1",
        "answer_key": {
            "model_answer": "It halves the search interval, which requires a sorted list.",
            "required_elements": list(elements),
        },
    }


def numeric_candidate(steps=None, final="4 comparisons"):
    """A numeric candidate whose steps total 4 marks by default."""
    return {
        "stem": "How many comparisons does a binary search need on 16 sorted items?",
        "type": "numeric",
        "options": [],
        "correct": final,
        "explanation": "Each comparison halves the interval.",
        "source_ref": "P1",
        "answer_key": {
            "steps": steps
            if steps is not None
            else [
                {"text": "Each comparison halves the interval: 16 → 8 → 4 → 2 → 1.", "marks": 2},
                {"text": "Count the halvings: log2(16) = 4.", "marks": 2},
            ],
            "final_answer": final,
        },
    }


def a_numeric_item(marks="4.00", count=1, passages=None):
    """A numeric item worth `marks` — what the mark-sum check is measured against."""
    return GenerationItem(
        course_name="Data Structures",
        topic_name="Binary search",
        question_type="numeric",
        level="multi_step",
        marks=Decimal(marks),
        count=count,
        passages=tuple(passages if passages is not None else [a_passage()]),
    )


# --- Key shapes --------------------------------------------------------------


class KeyShapeTests(SimpleTestCase):
    def test_each_question_type_maps_to_its_key_kind(self):
        self.assertEqual(kind_for_type("mcq"), OBJECTIVE)
        self.assertEqual(kind_for_type("true_false"), OBJECTIVE)
        self.assertEqual(kind_for_type("short_answer"), SHORT_ANSWER)
        self.assertEqual(kind_for_type("numeric"), NUMERIC)
        self.assertEqual(kind_for_type("essay"), ESSAY)

    def test_an_mcq_arrives_with_a_direct_key(self):
        provider = ScriptedProvider(payload(mcq_candidate()))
        key = generate_candidates(an_item(), provider=provider).candidates[0].answer_key

        self.assertIsInstance(key, ObjectiveKey)
        self.assertEqual(key.kind, OBJECTIVE)
        self.assertEqual(key.answer, "Halve the interval")
        self.assertIn("Scan every element", key.options)
        self.assertEqual(key.display_answer, "Halve the interval")

    def test_a_true_false_key_is_the_normalised_answer(self):
        provider = ScriptedProvider(payload(true_false_candidate()))
        run = generate_candidates(an_item(question_type="true_false"), provider=provider)

        key = run.candidates[0].answer_key
        self.assertIsInstance(key, ObjectiveKey)
        self.assertEqual(key.answer, "True")
        self.assertEqual(key.options, ["True", "False"])

    def test_an_objective_key_is_formalised_when_the_model_omits_it(self):
        """`correct` and the options already are the key — M6 formalises them."""
        candidate = mcq_candidate()
        del candidate["answer_key"]
        provider = ScriptedProvider(payload(candidate))
        key = generate_candidates(an_item(), provider=provider).candidates[0].answer_key

        self.assertEqual(len(provider.calls), 1)  # not a second call to fill it in
        self.assertIsInstance(key, ObjectiveKey)
        self.assertEqual(key.answer, "Halve the interval")

    def test_a_short_answer_arrives_with_a_model_answer_and_required_elements(self):
        provider = ScriptedProvider(payload(short_answer_candidate()))
        run = generate_candidates(an_item(question_type="short_answer"), provider=provider)

        key = run.candidates[0].answer_key
        self.assertIsInstance(key, ShortAnswerKey)
        self.assertEqual(key.kind, SHORT_ANSWER)
        self.assertIn("halves the search interval", key.model_answer)
        self.assertEqual(key.required_elements, ["halving the interval", "on a sorted list"])

    def test_a_short_answer_without_required_elements_does_not_validate(self):
        """A model answer alone is marking by resemblance; the elements are the key."""
        keyless = short_answer_candidate(elements=())
        provider = ScriptedProvider(payload(keyless), payload(short_answer_candidate()))
        run = generate_candidates(an_item(question_type="short_answer"), provider=provider)

        self.assertEqual(len(provider.calls), 2)  # retried, not stored keyless
        self.assertEqual(len(run.candidates), 1)
        self.assertTrue(run.candidates[0].answer_key.required_elements)

    def test_a_numeric_arrives_with_steps_marks_and_a_final_answer(self):
        provider = ScriptedProvider(payload(numeric_candidate()))
        run = generate_candidates(a_numeric_item(), provider=provider)

        key = run.candidates[0].answer_key
        self.assertIsInstance(key, NumericKey)
        self.assertEqual(key.kind, NUMERIC)
        self.assertEqual(key.final_answer, "4 comparisons")
        self.assertEqual(len(key.steps), 2)
        self.assertEqual([s.marks for s in key.steps], [Decimal("2.00"), Decimal("2.00")])
        self.assertEqual(key.total_marks, Decimal("4.00"))

    def test_a_numeric_without_steps_does_not_validate(self):
        provider = ScriptedProvider(
            payload(numeric_candidate(steps=[])), payload(numeric_candidate())
        )
        run = generate_candidates(a_numeric_item(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(run.candidates[0].answer_key.steps)

    def test_half_marks_survive_the_round_trip(self):
        """Half marks are ordinary in a real paper; floats are not how they are kept."""
        key = NumericKey.model_validate(
            {
                "steps": [{"text": "a", "marks": 0.5}, {"text": "b", "marks": "1.5"}],
                "final_answer": "3 m/s",
            }
        )
        self.assertEqual(key.total_marks, Decimal("2.00"))
        self.assertEqual(key.as_dict()["steps"][0]["marks"], "0.50")

        json.dumps(key.as_dict())  # storable as JSON, which Decimal is not
        restored = answer_key_from_dict(key.as_dict())
        self.assertIsInstance(restored, NumericKey)
        self.assertEqual(restored.total_marks, Decimal("2.00"))

    def test_a_stored_key_is_read_back_as_its_own_type(self):
        for key in (
            ObjectiveKey(answer="A", options=["A", "B"]),
            ShortAnswerKey(model_answer="Because.", required_elements=["cause"]),
            NumericKey(steps=[{"text": "a", "marks": 1}], final_answer="7"),
        ):
            with self.subTest(kind=key.kind):
                self.assertIsInstance(answer_key_from_dict(key.as_dict()), type(key))

    def test_an_unreadable_stored_key_is_none_rather_than_a_wrong_type(self):
        self.assertIsNone(answer_key_from_dict({}))
        self.assertIsNone(answer_key_from_dict(None))
        self.assertIsNone(answer_key_from_dict({"kind": "diagram", "answer": "x"}))


# --- The deterministic mark-sum check ----------------------------------------


class MarkSumTests(SimpleTestCase):
    def test_the_check_passes_when_the_steps_total_the_marks(self):
        key = check_mark_sum(
            NumericKey(
                steps=[{"text": "a", "marks": 1.5}, {"text": "b", "marks": 2.5}],
                final_answer="7",
            ),
            Decimal("4.00"),
        )
        self.assertIs(key.mark_sum_ok, True)
        self.assertEqual(key.mark_sum_note, "")

    def test_the_check_flags_when_they_do_not(self):
        key = check_mark_sum(
            NumericKey(
                steps=[{"text": "a", "marks": 1}, {"text": "b", "marks": 1}],
                final_answer="7",
            ),
            Decimal("5.00"),
        )
        self.assertIs(key.mark_sum_ok, False)
        self.assertIn("2.00", key.mark_sum_note)
        self.assertIn("5.00", key.mark_sum_note)

    def test_a_flagged_split_is_never_rewritten(self):
        """The model splits the marks; Python only adds them up."""
        key = check_mark_sum(
            NumericKey(steps=[{"text": "a", "marks": 1}], final_answer="7"), Decimal("4.00")
        )
        self.assertEqual([s.marks for s in key.steps], [Decimal("1.00")])
        self.assertEqual(key.total_marks, Decimal("1.00"))

    def test_the_check_runs_on_every_generated_numeric_candidate(self):
        provider = ScriptedProvider(payload(numeric_candidate()))
        run = generate_candidates(a_numeric_item(marks="4.00"), provider=provider)
        candidate = run.candidates[0]

        self.assertIs(candidate.mark_sum_ok, True)
        self.assertEqual(run.mark_sum_flagged, [])

    def test_a_mismatched_split_is_flagged_and_kept_not_dropped(self):
        """Caught, reported, and still offered: the stem is not the arithmetic's fault."""
        provider = ScriptedProvider(payload(numeric_candidate()))
        run = generate_candidates(a_numeric_item(marks="6.00"), provider=provider)
        candidate = run.candidates[0]

        self.assertEqual(len(run.candidates), 1)
        self.assertIs(candidate.mark_sum_ok, False)
        self.assertIn("total 4.00", candidate.mark_sum_note)
        self.assertIn("worth 6.00", candidate.mark_sum_note)
        self.assertEqual(run.mark_sum_flagged, [candidate.stem])

    def test_no_other_type_claims_a_check_that_never_ran(self):
        provider = ScriptedProvider(payload(mcq_candidate()))
        candidate = generate_candidates(an_item(), provider=provider).candidates[0]

        self.assertIsNone(candidate.mark_sum_ok)
        self.assertEqual(candidate.mark_sum_note, "")


# --- Essay: the format now, the questions later ------------------------------


class EssayFormatTests(SimpleTestCase):
    def test_an_essay_key_validates_as_rubric_criteria_with_weights(self):
        key = EssayKey.model_validate(
            {
                "criteria": [
                    {"criterion": "Uses the course's definition", "weight": 4},
                    {"criterion": "Argument is supported", "weight": "6", "descriptor": "..."},
                ],
                "model_answer": "An answer that would earn full marks.",
            }
        )
        self.assertEqual(key.kind, ESSAY)
        self.assertEqual(key.total_weight, Decimal("10.00"))
        self.assertIsInstance(answer_key_from_dict(key.as_dict()), EssayKey)

    def test_an_essay_key_with_no_criteria_is_not_a_rubric(self):
        with self.assertRaises(ValueError):
            EssayKey.model_validate({"criteria": []})

    def test_a_criterion_with_no_weight_is_refused(self):
        with self.assertRaises(ValueError):
            EssayKey.model_validate({"criteria": [{"criterion": "Style", "weight": 0}]})

    def test_no_essay_question_is_generated(self):
        self.assertNotIn("essay", MVP_TYPES)
        provider = ScriptedProvider(payload(mcq_candidate()))
        with self.assertRaises(UnsupportedQuestionType):
            generate_candidates(an_item(question_type="essay"), provider=provider)
        self.assertEqual(provider.calls, [])


# --- One call ----------------------------------------------------------------


class SingleCallTests(SimpleTestCase):
    def test_every_candidate_carries_a_key_from_one_call(self):
        """The M6 success check: 100% keyed, and no second call to key them."""
        provider = ScriptedProvider(
            payload(
                numeric_candidate(),
                numeric_candidate(final="5 comparisons"),
                numeric_candidate(final="3 comparisons"),
            )
        )
        run = generate_candidates(a_numeric_item(count=2), provider=provider)

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(run.candidates), 3)
        self.assertEqual(run.keyed_rate, 1.0)
        for candidate in run.candidates:
            self.assertIsNotNone(candidate.answer_key)
            self.assertEqual(candidate.key_kind, NUMERIC)

    def test_reading_a_key_back_needs_no_provider_at_all(self):
        """Stage-3 review inspects stem and key together, without a model call."""
        provider = ScriptedProvider(payload(short_answer_candidate()))
        run = generate_candidates(an_item(question_type="short_answer"), provider=provider)

        self.assertEqual(provider.texts, [])  # the script is exhausted
        key = run.candidates[0].answer_key
        self.assertTrue(key.model_answer and key.required_elements)

    def test_the_prompt_asks_for_the_key_with_the_question(self):
        _, user = build_prompt(a_numeric_item())
        system, _ = build_prompt(a_numeric_item())

        self.assertIn(ANSWER_KEY_RULE, system)
        self.assertIn(ANSWER_KEY_RULE, user)
        self.assertIn(KEY_RULES["numeric"], user)
        self.assertIn("must add up to exactly the marks", user)

    def test_the_prompt_asks_for_the_key_of_the_row_s_own_type(self):
        _, user = build_prompt(an_item(question_type="short_answer"))
        self.assertIn("required_elements", user)
        self.assertNotIn("final_answer", user)


# --- Storage -----------------------------------------------------------------


class KeyStorageTests(TestCase):
    def setUp(self):
        instructor = User.objects.create_user("hana", password="quiet-precision-42")
        self.course = Course.objects.create(
            instructor=instructor, name="Data Structures", code="CS201", content_language="en"
        )
        self.topic = Topic.objects.create(course=self.course, name="Binary search")
        self.exam = Exam.objects.create(
            course=self.course,
            total_score=40,
            question_count=20,
            duration_minutes=60,
            language="en",
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.row = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.topic,
            question_type="numeric",
            level="multi_step",
            count=1,
            marks=Decimal("4.00"),
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

    def _run(self, *candidates):
        passage = a_passage(chunk_id=self.chunk.pk)
        provider = ScriptedProvider(payload(*candidates))
        return generate_candidates(item_for_row(self.row, [passage]), provider=provider)

    def test_the_key_is_stored_with_the_question_and_read_back_typed(self):
        stored = save_candidates(self._run(numeric_candidate()))
        question = stored[0]
        question.refresh_from_db()

        self.assertEqual(question.answer_key["kind"], NUMERIC)
        self.assertEqual(question.answer_key["final_answer"], "4 comparisons")
        self.assertEqual(len(question.answer_key["steps"]), 2)
        self.assertIs(question.mark_sum_ok, True)
        self.assertFalse(question.needs_mark_review)

        key = question.key
        self.assertIsInstance(key, NumericKey)
        self.assertEqual(key.total_marks, Decimal("4.00"))

    def test_review_can_find_the_questions_whose_marks_do_not_add_up(self):
        """The flag is a column, not a note inside JSON, so review can query it."""
        self.row.marks = Decimal("9.00")  # the 2 + 2 split no longer totals
        self.row.save()
        save_candidates(self._run(numeric_candidate()))

        flagged = Question.objects.filter(mark_sum_ok=False)
        self.assertEqual(flagged.count(), 1)
        self.assertTrue(flagged.get().needs_mark_review)
        self.assertIn("instructor's eye", flagged.get().key.mark_sum_note)

    def test_a_question_of_another_type_stores_no_mark_sum_verdict(self):
        self.row.question_type = "mcq"
        self.row.save()
        stored = save_candidates(self._run(mcq_candidate()))

        question = stored[0]
        self.assertIsNone(question.mark_sum_ok)
        self.assertEqual(question.answer_key["kind"], OBJECTIVE)
        self.assertEqual(question.key.answer, question.correct)
