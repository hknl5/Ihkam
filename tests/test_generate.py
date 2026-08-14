"""M5: Agent 2A writes questions, and only from the passages it was given.

Every provider in this file is a fake. The suite spends no API calls and makes
no network connection — the rule since M2 — and the fakes are written so that a
call reaching a real model would fail loudly rather than quietly cost money.

What is pinned down here:

* over-generation is `ceil(N * 1.5)`, for the row sizes a real blueprint has;
* a malformed answer is retried exactly once, then raised;
* a call that never reached the model is reported as itself, not retried;
* a candidate whose `source_ref` names no supplied passage never becomes a
  candidate — the M5 failure mode, counted rather than hidden;
* the grounding instruction is in the prompt that is actually composed.
"""

import json
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from agents.generate import (
    MVP_TYPES,
    Candidate,
    GenerationItem,
    QuestionGenerationError,
    UnsupportedQuestionType,
    build_prompt,
    generate_candidates,
    item_for_row,
    over_generated_count,
    resolve_passage,
    save_candidates,
)
from agents.prompts.generate import GROUNDING_RULE
from courses.models import Chunk, Course, SourceFile, Topic
from courses.services.retrieval import Passage
from exams.models import Blueprint, BlueprintRow, Exam, Question

DIM = 1536


def a_passage(
    text="A binary search halves the interval at every step.",
    page=7,
    source_file="lecture-3.pdf",
    chunk_id=1,
    from_ocr=False,
):
    return Passage(
        chunk_id=chunk_id,
        text=text,
        page=page,
        source_file=source_file,
        score=0.78,
        topic="Searching",
        from_ocr=from_ocr,
    )


def an_item(question_type="mcq", count=1, passages=None, **kwargs):
    return GenerationItem(
        course_name="Data Structures",
        topic_name="Binary search",
        question_type=question_type,
        level="medium",
        marks=Decimal("2.00"),
        count=count,
        passages=tuple(passages if passages is not None else [a_passage()]),
        **kwargs,
    )


def mcq(stem="Which step does a binary search repeat?", source_ref="P1"):
    return {
        "stem": stem,
        "type": "mcq",
        "options": ["Halve the interval", "Scan every element", "Sort the list", "Hash the key"],
        "correct": "Halve the interval",
        "explanation": "The passage says the interval is halved at every step.",
        "source_ref": source_ref,
    }


class ScriptedProvider:
    """Returns prepared texts in order. A call past the script is a test bug."""

    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, **kwargs):
        from agents.provider import LLMResponse

        self.calls.append((system, user))
        if not self.texts:
            raise AssertionError("The generator called the model more times than expected.")
        return LLMResponse(text=self.texts.pop(0))

    def embed(self, texts):  # pragma: no cover - generation never embeds
        raise AssertionError("Agent 2A must not embed; retrieval already happened.")


def payload(*candidates):
    return json.dumps({"questions": list(candidates)})


# --- Over-generation ---------------------------------------------------------


class OverGenerationTests(SimpleTestCase):
    def test_asks_for_ceil_of_one_and_a_half_times_n(self):
        self.assertEqual(
            [over_generated_count(n) for n in (1, 2, 3, 4)],
            [2, 3, 5, 6],
        )

    def test_one_question_still_gets_an_alternative(self):
        """The point of the rounding: a row of 1 must not yield exactly 1."""
        self.assertGreater(over_generated_count(1), 1)

    def test_no_questions_asked_for_means_none_generated(self):
        self.assertEqual(over_generated_count(0), 0)

    def test_multiplier_is_tunable(self):
        self.assertEqual(over_generated_count(4, multiplier=2.0), 8)

    def test_the_item_reports_what_it_will_ask_for(self):
        self.assertEqual(an_item(count=4).candidates_wanted, 6)

    def test_the_prompt_asks_the_model_for_the_over_generated_count(self):
        _, user = build_prompt(an_item(count=4))
        self.assertIn("Write exactly 6 question(s)", user)


# --- The prompt --------------------------------------------------------------


class PromptTests(SimpleTestCase):
    def test_the_grounding_instruction_reaches_the_model(self):
        system, user = build_prompt(an_item())
        self.assertIn(GROUNDING_RULE, system)
        self.assertIn(GROUNDING_RULE, user)

    def test_passages_are_labelled_so_a_candidate_can_cite_one(self):
        _, user = build_prompt(
            an_item(passages=[a_passage(), a_passage(text="Linear search scans.", page=8)])
        )
        self.assertIn("[P1] lecture-3.pdf · page 7", user)
        self.assertIn("[P2] lecture-3.pdf · page 8", user)
        self.assertIn("Linear search scans.", user)

    def test_an_ocr_passage_is_marked_as_a_transcription(self):
        _, user = build_prompt(an_item(passages=[a_passage(from_ocr=True)]))
        self.assertIn("(OCR transcription)", user)

    def test_the_type_and_level_rules_are_the_row_s_own(self):
        _, user = build_prompt(an_item(question_type="true_false"))
        self.assertIn("True / false", user)
        self.assertNotIn("exactly 4 options", user)


# --- Validation --------------------------------------------------------------


class ValidationTests(SimpleTestCase):
    def test_a_valid_candidate_parses(self):
        provider = ScriptedProvider(payload(mcq()))
        run = generate_candidates(an_item(), provider=provider)

        self.assertEqual(len(run.candidates), 1)
        candidate = run.candidates[0]
        self.assertEqual(candidate.question_type, "mcq")
        self.assertEqual(candidate.correct, "Halve the interval")
        self.assertEqual(len(candidate.options), 4)
        self.assertTrue(candidate.explanation)

    def test_a_fenced_answer_is_still_read(self):
        provider = ScriptedProvider(f"```json\n{payload(mcq())}\n```")
        self.assertEqual(len(generate_candidates(an_item(), provider=provider).candidates), 1)

    def test_a_malformed_answer_is_retried_once_then_accepted(self):
        provider = ScriptedProvider("not json at all", payload(mcq()))
        run = generate_candidates(an_item(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(run.candidates), 1)

    def test_two_malformed_answers_raise_and_store_nothing(self):
        provider = ScriptedProvider("nonsense", "still nonsense")
        with self.assertRaises(QuestionGenerationError) as caught:
            generate_candidates(an_item(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertIn("two attempts", str(caught.exception))

    def test_an_mcq_whose_answer_is_not_an_option_is_a_validation_failure(self):
        broken = mcq() | {"correct": "Something the options do not say"}
        provider = ScriptedProvider(payload(broken), payload(mcq()))
        run = generate_candidates(an_item(), provider=provider)

        self.assertEqual(len(provider.calls), 2)  # retried, not passed through
        self.assertEqual(len(run.candidates), 1)

    def test_true_false_answers_are_normalised(self):
        candidate = {
            "stem": "A binary search halves the interval at every step.",
            "type": "true_false",
            "options": [],
            "correct": "true",
            "explanation": "Stated in the passage.",
            "source_ref": "P1",
        }
        provider = ScriptedProvider(payload(candidate))
        run = generate_candidates(an_item(question_type="true_false"), provider=provider)

        self.assertEqual(run.candidates[0].correct, "True")
        self.assertEqual(run.candidates[0].options, ("True", "False"))

    def test_an_open_question_carries_no_options(self):
        candidate = {
            "stem": "What does a binary search do at every step?",
            "type": "short_answer",
            "options": ["a distractor that does not belong here"],
            "correct": "It halves the interval.",
            "explanation": "The passage states it.",
            "source_ref": "P1",
        }
        provider = ScriptedProvider(payload(candidate))
        run = generate_candidates(an_item(question_type="short_answer"), provider=provider)

        self.assertEqual(run.candidates[0].options, ())

    def test_a_call_that_never_reached_the_model_is_reported_as_itself(self):
        class NoCredit:
            name = "no-credit"
            calls = 0

            def complete(self, *args, **kwargs):
                type(self).calls += 1
                raise RuntimeError("429 insufficient_quota")

            def embed(self, texts):  # pragma: no cover
                raise AssertionError

        provider = NoCredit()
        with self.assertRaises(QuestionGenerationError) as caught:
            generate_candidates(an_item(), provider=provider)

        self.assertEqual(NoCredit.calls, 1)  # not retried
        self.assertIn("did not complete", str(caught.exception))
        self.assertIn("insufficient_quota", str(caught.exception))


# --- Grounding ---------------------------------------------------------------


class GroundingTests(SimpleTestCase):
    def test_every_candidate_carries_a_source_ref_to_a_supplied_passage(self):
        passages = [a_passage(), a_passage(text="Linear search scans.", page=8, chunk_id=2)]
        provider = ScriptedProvider(
            payload(mcq(), mcq(stem="Which search scans every element?", source_ref="P2"))
        )
        run = generate_candidates(an_item(count=2, passages=passages), provider=provider)

        self.assertEqual(len(run.candidates), 2)
        self.assertEqual(run.candidates[0].source_ref, "lecture-3.pdf · page 7")
        self.assertEqual(run.candidates[1].source_ref, "lecture-3.pdf · page 8")
        for candidate in run.candidates:
            self.assertIn(candidate.passage, passages)

    def test_a_candidate_citing_a_passage_that_was_not_supplied_is_dropped(self):
        provider = ScriptedProvider(
            payload(mcq(), mcq(stem="From somewhere else entirely", source_ref="P9"))
        )
        run = generate_candidates(an_item(count=2), provider=provider)

        self.assertEqual(len(run.candidates), 1)
        self.assertEqual(run.ungrounded, ["From somewhere else entirely"])
        self.assertEqual(run.returned, 2)
        self.assertAlmostEqual(run.grounded_rate, 0.5)

    def test_a_candidate_with_no_source_ref_never_validates(self):
        provider = ScriptedProvider(
            payload(mcq() | {"source_ref": ""}), payload(mcq())
        )
        run = generate_candidates(an_item(), provider=provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(run.candidates), 1)

    def test_a_candidate_of_the_wrong_type_is_dropped(self):
        provider = ScriptedProvider(payload(mcq()))
        run = generate_candidates(an_item(question_type="short_answer"), provider=provider)

        self.assertEqual(run.candidates, [])
        self.assertEqual(len(run.ungrounded), 1)

    def test_generation_without_passages_never_calls_the_model(self):
        provider = ScriptedProvider(payload(mcq()))
        with self.assertRaises(QuestionGenerationError) as caught:
            generate_candidates(an_item(passages=[]), provider=provider)

        self.assertEqual(provider.calls, [])
        self.assertIn("nothing to write a question from", str(caught.exception))

    def test_a_citation_written_out_in_full_still_resolves(self):
        passages = [a_passage(), a_passage(page=8, chunk_id=2)]
        self.assertIs(resolve_passage("lecture-3.pdf · page 8", passages), passages[1])
        self.assertIs(resolve_passage("p 1", passages), passages[0])

    def test_a_citation_to_a_page_that_was_not_supplied_resolves_to_nothing(self):
        self.assertIsNone(resolve_passage("page 42", [a_passage()]))
        self.assertIsNone(resolve_passage("", [a_passage()]))
        self.assertIsNone(resolve_passage("P1", []))


class FromOcrTests(SimpleTestCase):
    def test_from_ocr_propagates_from_the_cited_passage(self):
        passages = [a_passage(from_ocr=True), a_passage(page=8, chunk_id=2, from_ocr=False)]
        provider = ScriptedProvider(
            payload(mcq(), mcq(stem="Second one", source_ref="P2"))
        )
        run = generate_candidates(an_item(count=2, passages=passages), provider=provider)

        self.assertTrue(run.candidates[0].from_ocr)
        self.assertFalse(run.candidates[1].from_ocr)


# --- Types the MVP does not write -------------------------------------------


class UnsupportedTypeTests(SimpleTestCase):
    def test_the_mvp_writes_four_types(self):
        self.assertEqual(MVP_TYPES, {"mcq", "true_false", "short_answer", "numeric"})

    def test_a_deferred_type_is_refused_by_name_without_a_call(self):
        provider = ScriptedProvider(payload(mcq()))
        item = an_item(question_type="matching")
        with self.assertRaises(UnsupportedQuestionType) as caught:
            generate_candidates(item, provider=provider)

        self.assertEqual(provider.calls, [])
        self.assertIn("matching", str(caught.exception))
        self.assertFalse(item.is_supported)

    def test_an_unsupported_type_is_not_silently_turned_into_an_mcq(self):
        provider = ScriptedProvider(payload(mcq()))
        with self.assertRaises(UnsupportedQuestionType):
            generate_candidates(an_item(question_type="diagram"), provider=provider)
        self.assertEqual(provider.calls, [])


# --- Wiring to a real blueprint row ------------------------------------------


class RowToItemTests(TestCase):
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
            question_type="mcq",
            level="medium",
            count=4,
            marks=Decimal("8.00"),
        )

    def test_the_item_carries_the_row_s_plan_and_the_exam_s_arithmetic(self):
        item = item_for_row(self.row, [a_passage()])

        self.assertEqual(item.topic_name, "Binary search")
        self.assertEqual(item.question_type, "mcq")
        self.assertEqual(item.count, 4)
        self.assertEqual(item.candidates_wanted, 6)
        self.assertEqual(item.marks, Decimal("2.00"))
        self.assertEqual(item.expected_minutes, 3)  # 60 minutes over 20 questions
        self.assertEqual(item.language, "en")
        self.assertEqual(item.row_id, self.row.pk)

    def test_candidates_are_stored_as_candidate_questions(self):
        source_file = SourceFile.objects.create(
            course=self.course,
            original_name="lecture-3.pdf",
            kind=SourceFile.Kind.PDF,
            page_count=12,
        )
        chunk = Chunk.objects.create(
            source_file=source_file,
            topic=self.topic,
            page=7,
            position=0,
            text="A binary search halves the interval at every step.",
            embedding=[0.0] * DIM,
        )
        passage = a_passage(chunk_id=chunk.pk, from_ocr=True)
        provider = ScriptedProvider(payload(mcq()))
        run = generate_candidates(item_for_row(self.row, [passage]), provider=provider)

        stored = save_candidates(run)

        self.assertEqual(len(stored), 1)
        question = Question.objects.get()
        self.assertEqual(question.status, Question.Status.CANDIDATE)
        self.assertTrue(question.is_candidate)
        self.assertEqual(question.exam, self.exam)
        self.assertEqual(question.blueprint_row, self.row)
        self.assertEqual(question.source_ref, "lecture-3.pdf · page 7")
        self.assertEqual(question.source_chunk, chunk)
        self.assertTrue(question.from_ocr)
        self.assertEqual(question.correct, "Halve the interval")
        self.assertEqual(len(question.options), 4)

    def test_a_question_survives_the_blueprint_row_it_came_from(self):
        question = Question.objects.create(
            exam=self.exam,
            blueprint_row=self.row,
            stem="Kept",
            question_type="mcq",
            correct="Halve the interval",
            source_ref="lecture-3.pdf · page 7",
        )
        self.row.delete()
        question.refresh_from_db()

        self.assertIsNone(question.blueprint_row)
        self.assertEqual(question.stem, "Kept")
