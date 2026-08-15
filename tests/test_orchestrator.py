"""M8: the closed correction loop — generate, review, and go back with the note.

Every provider here is a fake. The suite makes no network call and spends no API
credit — the rule since M2 — and a call the script did not expect fails loudly
rather than quietly reaching a real model.

What is pinned down:

* a row whose first batch already passes never asks for a replacement;
* a row that comes back short asks again, for the shortfall only, and **the
  reviewer's own note reaches the second generation call** — the loop is steered
  regeneration, not a second roll of the same dice;
* a row nothing satisfies stops after exactly three gap-fill rounds and surfaces
  as "needs manual attention" instead of looping;
* every attempt is recorded with its note — passed, rejected, and dropped before
  review — and the counts reconcile;
* nothing unreviewed reaches the approved pool, a provider outage is reported as
  itself and costs the item no round, and re-running an item does not duplicate
  the pool.
"""

import json
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from agents.generate import GenerationItem, item_for_row, over_generated_count
from agents.orchestrator import (
    MAX_GAP_FILL_ROUNDS,
    NEEDS_ATTENTION,
    PASSED,
    OrchestrationError,
    items_for_plan,
    run_exam,
    run_item,
    run_plan,
)
from agents.prompts.generate import REGENERATION_HEADER
from agents.prompts.generate import SYSTEM as GENERATE_SYSTEM
from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import (
    Blueprint,
    BlueprintRow,
    Exam,
    ItemRun,
    Question,
    QuestionAttempt,
)

from tests.test_generate import DIM, a_passage, mcq, payload
from tests.test_review import all_ok

PASSWORD = "not-a-real-password"


def an_item(count=1, **kwargs):
    defaults = dict(
        course_name="Data Structures",
        topic_name="Binary search",
        question_type="mcq",
        level="medium",
        marks=Decimal("2.00"),
        passages=(a_passage(),),
    )
    return GenerationItem(count=count, **(defaults | kwargs))


def batch(*stems):
    """One generation answer: an MCQ per stem, each citing the supplied passage."""
    return [mcq(stem=stem) for stem in stems]


def rejection(check="level_match", reason="it asks for a definition, not a computation"):
    """One failing check, as the reviewing model returns it."""
    return {
        check: {
            "ok": False,
            "reason": reason,
            "requirement": "require a computation from the values in the passage",
        }
    }


class LoopProvider:
    """One fake standing in for both agents, told apart by the system prompt.

    Generation answers come from a script of batches, in order. Review answers
    are clean unless the stem being reviewed is in `reject`, which maps a stem to
    the check it fails — that is how a test says "this question is bad" without
    having to write a bad question.
    """

    name = "loop"

    def __init__(self, *batches, reject=None, generation_fails_on=None):
        self.batches = list(batches)
        self.reject = dict(reject or {})
        #: 1-based index of the generation call that never reaches the model.
        self.generation_fails_on = generation_fails_on
        self.generation_prompts: list[str] = []
        self.review_prompts: list[str] = []

    def complete(self, system, user, **kwargs):
        from agents.provider import LLMResponse

        if system == GENERATE_SYSTEM:
            self.generation_prompts.append(user)
            if len(self.generation_prompts) == self.generation_fails_on:
                raise RuntimeError("429 insufficient_quota")
            if not self.batches:
                raise AssertionError("The loop generated more times than expected.")
            return LLMResponse(text=payload(*self.batches.pop(0)))

        self.review_prompts.append(user)
        for stem, failure in self.reject.items():
            if stem in user:
                return LLMResponse(text=json.dumps(all_ok() | failure))
        return LLMResponse(text=json.dumps(all_ok()))

    def embed(self, texts):  # pragma: no cover - the loop never embeds
        raise AssertionError("The orchestrator must not embed; retrieval already happened.")


# --- The happy path ----------------------------------------------------------


class FirstRoundTests(SimpleTestCase):
    def test_enough_passing_candidates_means_no_gap_fill_round(self):
        provider = LoopProvider(batch("Q1", "Q2"))
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(result.status, PASSED)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.gap_fill_rounds, 0)
        self.assertEqual(len(provider.generation_prompts), 1)
        self.assertEqual(result.approved_count, 2)

    def test_the_first_round_asks_for_the_over_generated_count(self):
        provider = LoopProvider(batch("Q1", "Q2", "Q3"))
        run_item(an_item(count=2), provider=provider, persist=False)

        self.assertIn("Write exactly 3 question(s)", provider.generation_prompts[0])

    def test_the_first_round_carries_no_rejection_brief(self):
        provider = LoopProvider(batch("Q1", "Q2"))
        run_item(an_item(count=1), provider=provider, persist=False)

        self.assertNotIn(REGENERATION_HEADER, provider.generation_prompts[0])

    def test_surplus_passes_are_kept_as_alternatives(self):
        """A row of 2 that passes 3 keeps the third — it is the alternative."""
        provider = LoopProvider(batch("Q1", "Q2", "Q3"))
        result = run_item(an_item(count=2), provider=provider, persist=False)

        self.assertEqual(result.approved_count, 3)
        self.assertEqual(result.surplus, 1)
        self.assertEqual(result.status, PASSED)

    def test_every_candidate_is_reviewed_even_once_the_row_is_satisfied(self):
        """The surplus is only usable as an alternative if it was judged too."""
        provider = LoopProvider(batch("Q1", "Q2", "Q3"))
        run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(len(provider.review_prompts), 3)


# --- The gap-fill loop -------------------------------------------------------


class GapFillTests(SimpleTestCase):
    def test_a_short_row_asks_again_and_the_note_reaches_agent_2a(self):
        provider = LoopProvider(
            batch("Define a binary search.", "Q2"),
            batch("Compute the number of steps for n = 32.", "Q4"),
            reject={
                "Define a binary search.": rejection(),
                "Q2": rejection(),
            },
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(result.status, PASSED)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.gap_fill_rounds, 1)

        brief = provider.generation_prompts[1]
        self.assertIn(REGENERATION_HEADER, brief)
        self.assertIn("it asks for a definition, not a computation", brief)
        self.assertIn("require a computation from the values in the passage", brief)
        self.assertIn("Level match", brief)

    def test_the_rejected_question_itself_goes_back_with_its_note(self):
        provider = LoopProvider(
            batch("Define a binary search."),
            batch("Compute the number of steps for n = 32."),
            reject={"Define a binary search.": rejection()},
        )
        run_item(an_item(count=1), provider=provider, persist=False)

        self.assertIn("Define a binary search.", provider.generation_prompts[1])

    def test_a_gap_fill_round_asks_only_for_the_shortfall(self):
        """Three of four passed on a row of four: ask for one more, not four."""
        provider = LoopProvider(
            batch("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
            batch("Q7", "Q8"),
            reject={f"Q{n}": rejection() for n in (1, 2, 3)},
        )
        result = run_item(an_item(count=4), provider=provider, persist=False)

        self.assertIn("Write exactly 2 question(s)", provider.generation_prompts[1])
        self.assertEqual(over_generated_count(1), 2)  # the shortfall, over-generated
        self.assertEqual(result.status, PASSED)
        self.assertEqual(result.approved_count, 5)

    def test_notes_from_every_round_are_carried_not_just_the_last(self):
        """Otherwise round three fixes the level and reintroduces the scope fault."""
        provider = LoopProvider(
            batch("Q1"),
            batch("Q2"),
            batch("Q3"),
            reject={
                "Q1": rejection("level_match", "far too easy for a multi-step row"),
                "Q2": rejection("content_link", "uses a term the passages never state"),
            },
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        third = provider.generation_prompts[2]
        self.assertIn("far too easy for a multi-step row", third)
        self.assertIn("uses a term the passages never state", third)
        self.assertEqual(result.status, PASSED)

    def test_the_loop_stops_the_moment_the_row_is_satisfied(self):
        provider = LoopProvider(
            batch("Q1", "Q2"),
            batch("Q3", "Q4"),
            reject={"Q1": rejection(), "Q2": rejection()},
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(len(provider.generation_prompts), 2)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.approved_count, 2)


# --- The cap -----------------------------------------------------------------


class RetryCapTests(SimpleTestCase):
    def _stubborn(self):
        """A row nothing satisfies: every candidate of every round is rejected."""
        rounds = [batch(f"R{n}a", f"R{n}b") for n in range(1, 10)]
        reject = {f"R{n}{letter}": rejection() for n in range(1, 10) for letter in "ab"}
        return LoopProvider(*rounds, reject=reject)

    def test_a_stubborn_item_stops_after_exactly_three_gap_fill_rounds(self):
        provider = self._stubborn()
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(result.gap_fill_rounds, MAX_GAP_FILL_ROUNDS)
        self.assertEqual(result.gap_fill_rounds, 3)
        self.assertEqual(result.rounds, 4)  # the first batch, then three retries
        self.assertEqual(len(provider.generation_prompts), 4)

    def test_a_stubborn_item_is_marked_needs_manual_attention(self):
        result = run_item(an_item(count=1), provider=self._stubborn(), persist=False)

        self.assertEqual(result.status, NEEDS_ATTENTION)
        self.assertTrue(result.needs_attention)
        self.assertEqual(result.approved_count, 0)
        self.assertEqual(result.shortfall, 1)

    def test_the_cap_is_tunable_without_touching_the_loop(self):
        provider = self._stubborn()
        result = run_item(an_item(count=1), provider=provider, max_rounds=1, persist=False)

        self.assertEqual(result.gap_fill_rounds, 1)
        self.assertEqual(len(provider.generation_prompts), 2)

    def test_a_capped_item_keeps_the_partial_pool_it_did_earn(self):
        """Two of three passed on a row of three: the two are still questions."""
        rounds = [batch("Q1", "Q2", "Q3", "Q4", "Q5")] + [batch(f"R{n}") for n in range(2, 9)]
        reject = {"Q3": rejection(), "Q4": rejection(), "Q5": rejection()}
        reject |= {f"R{n}": rejection() for n in range(2, 9)}
        result = run_item(an_item(count=3), provider=LoopProvider(*rounds, reject=reject),
                          persist=False)

        self.assertEqual(result.status, NEEDS_ATTENTION)
        self.assertEqual(result.approved_count, 2)
        self.assertEqual(result.shortfall, 1)


# --- Traceability ------------------------------------------------------------


class AttemptLogTests(SimpleTestCase):
    def test_every_attempt_is_recorded_with_its_outcome(self):
        provider = LoopProvider(
            batch("Q1", "Q2"),
            batch("Q3", "Q4"),
            reject={"Q1": rejection(), "Q2": rejection()},
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(len(result.attempts), 4)
        self.assertEqual([a.stem for a in result.rejected], ["Q1", "Q2"])
        self.assertEqual([a.stem for a in result.approved], ["Q3", "Q4"])
        self.assertTrue(result.counts_reconcile)

    def test_a_rejected_attempt_carries_the_note_that_rejected_it(self):
        provider = LoopProvider(
            batch("Q1"),
            batch("Q2"),
            reject={"Q1": rejection()},
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        rejected = result.rejected[0]
        self.assertIn("level_match", rejected.failed_checks)
        self.assertIn("it asks for a definition, not a computation", rejected.notes[0])
        self.assertIn("The replacement must:", rejected.notes[0])

    def test_an_ungrounded_candidate_is_recorded_rather_than_silently_dropped(self):
        """M5 drops it before review. A short row must still say why it is short."""
        ungrounded = mcq(stem="Written from somewhere else", source_ref="P9")
        provider = LoopProvider([mcq(stem="Q1"), ungrounded], batch("Q2"))
        result = run_item(an_item(count=2), provider=provider, persist=False)

        self.assertEqual([a.stem for a in result.dropped], ["Written from somewhere else"])
        self.assertEqual(len(result.attempts), 3)
        self.assertTrue(result.counts_reconcile)

    def test_the_round_each_attempt_belongs_to_is_recorded(self):
        provider = LoopProvider(batch("Q1"), batch("Q2"), reject={"Q1": rejection()})
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual([(a.stem, a.round) for a in result.attempts], [("Q1", 1), ("Q2", 2)])

    def test_a_passing_attempt_has_no_notes(self):
        result = run_item(an_item(count=1), provider=LoopProvider(batch("Q1", "Q2")),
                          persist=False)

        self.assertEqual(result.approved[0].notes, [])
        self.assertEqual(result.notes, [])


# --- The guardrails ----------------------------------------------------------


class OutageTests(SimpleTestCase):
    def test_an_outage_is_reported_as_itself_and_costs_no_round(self):
        provider = LoopProvider(
            batch("Q1"),
            batch("Q2"),
            reject={"Q1": rejection()},
            generation_fails_on=2,  # the gap-fill call never reaches the model
        )
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertTrue(result.unreachable)
        self.assertIn("did not complete", result.error)
        self.assertIn("insufficient_quota", result.error)
        self.assertEqual(result.rounds, 1)  # the round it died in was never spent
        self.assertEqual(result.gap_fill_rounds, 0)
        self.assertEqual(result.status, NEEDS_ATTENTION)

    def test_an_outage_does_not_pass_an_unreviewed_question(self):
        provider = LoopProvider(batch("Q1"), generation_fails_on=1)
        result = run_item(an_item(count=1), provider=provider, persist=False)

        self.assertEqual(result.approved, [])
        self.assertEqual(result.attempts, [])

    def test_an_item_with_no_passages_is_reported_not_generated(self):
        provider = LoopProvider(batch("Q1"))
        result = run_item(an_item(count=1, passages=()), provider=provider, persist=False)

        self.assertEqual(provider.generation_prompts, [])
        self.assertEqual(result.status, NEEDS_ATTENTION)
        self.assertIn("nothing to write a question from", result.error)

    def test_an_unsupported_type_is_refused_by_name(self):
        result = run_item(
            an_item(count=1, question_type="matching"),
            provider=LoopProvider(),
            persist=False,
        )

        self.assertEqual(result.status, NEEDS_ATTENTION)
        self.assertIn("does not write", result.error)


class ApprovedPoolTests(SimpleTestCase):
    def test_only_reviewed_candidates_reach_the_pool(self):
        ungrounded = mcq(stem="Written from somewhere else", source_ref="P9")
        provider = LoopProvider(
            [mcq(stem="Q1"), mcq(stem="Q2"), ungrounded],
            batch("Q3"),
            reject={"Q1": rejection()},
        )
        result = run_item(an_item(count=2), provider=provider, persist=False)

        pooled = {a.stem for a in result.approved}
        self.assertEqual(pooled, {"Q2", "Q3"})
        for attempt in result.approved:
            self.assertIsNotNone(attempt.review)
            self.assertTrue(attempt.review.passed)
            self.assertTrue(attempt.review.model_checked)

    def test_python_only_review_still_gates_the_pool_and_says_so(self):
        provider = LoopProvider(batch("Q1", "Q2"))
        result = run_item(an_item(count=1), provider=provider, python_only=True, persist=False)

        self.assertEqual(provider.review_prompts, [])  # no review call was made
        self.assertEqual(result.approved_count, 2)
        self.assertFalse(result.approved[0].review.model_checked)


# --- Persistence -------------------------------------------------------------


class PersistenceTests(TestCase):
    def setUp(self):
        user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=user, name="Data Structures", code="CS201"
        )
        self.exam = Exam.objects.create(
            course=self.course, total_score=40, question_count=20, duration_minutes=60
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.topic = Topic.objects.create(course=self.course, name="Binary search", position=0)
        self.row = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.topic,
            question_type="mcq",
            level="medium",
            count=2,
            marks=Decimal("4.00"),
        )
        self.source_file = SourceFile.objects.create(
            course=self.course,
            original_name="lecture-3.pdf",
            kind=SourceFile.Kind.PDF,
            page_count=12,
        )
        self.chunk = Chunk.objects.create(
            source_file=self.source_file,
            topic=self.topic,
            page=7,
            position=0,
            text="A binary search halves the interval at every step.",
            embedding=[0.0] * DIM,
        )
        self.passage = a_passage(chunk_id=self.chunk.pk)

    def _item(self):
        return item_for_row(self.row, [self.passage])

    def _provider(self):
        """One rejection, one drop, and enough passes to satisfy a row of two."""
        ungrounded = mcq(stem="Written from somewhere else", source_ref="P9")
        return LoopProvider(
            [mcq(stem="Q1"), mcq(stem="Q2"), ungrounded],
            batch("Q3"),
            reject={"Q1": rejection()},
        )

    def test_the_whole_attempt_log_is_stored(self):
        result = run_item(self._item(), provider=self._provider())

        record = ItemRun.objects.get()
        self.assertEqual(record.exam, self.exam)
        self.assertEqual(record.blueprint_row, self.row)
        self.assertEqual(record.topic_name, "Binary search")
        self.assertEqual(record.required, 2)
        self.assertEqual(record.approved_count, 2)
        self.assertEqual(record.rounds, 2)
        self.assertEqual(record.gap_fill_rounds, 1)
        self.assertEqual(record.status, ItemRun.Status.PASSED)
        self.assertEqual(record.attempts.count(), 4)
        self.assertEqual(result.record.pk, record.pk)

    def test_stored_counts_reconcile(self):
        run_item(self._item(), provider=self._provider())

        record = ItemRun.objects.get()
        outcomes = record.attempts.values_list("outcome", flat=True)
        approved = sum(1 for o in outcomes if o == QuestionAttempt.Outcome.PASSED)
        rejected = sum(1 for o in outcomes if o == QuestionAttempt.Outcome.REJECTED)
        dropped = sum(1 for o in outcomes if o == QuestionAttempt.Outcome.DROPPED)

        self.assertEqual(record.attempts.count(), approved + rejected + dropped)
        self.assertEqual((approved, rejected, dropped), (2, 1, 1))
        self.assertEqual(approved, record.approved_count)

    def test_a_stored_rejection_keeps_the_note_it_was_rejected_with(self):
        run_item(self._item(), provider=self._provider())

        attempt = QuestionAttempt.objects.get(outcome=QuestionAttempt.Outcome.REJECTED)
        self.assertEqual(attempt.stem, "Q1")
        self.assertEqual(attempt.round, 1)
        self.assertEqual(attempt.failed_checks, ["level_match"])
        self.assertIn("it asks for a definition, not a computation", attempt.note)
        self.assertIsNone(attempt.question)

    def test_a_dropped_candidate_is_stored_with_no_question_behind_it(self):
        run_item(self._item(), provider=self._provider())

        attempt = QuestionAttempt.objects.get(outcome=QuestionAttempt.Outcome.DROPPED)
        self.assertEqual(attempt.stem, "Written from somewhere else")
        self.assertIsNone(attempt.question)
        self.assertEqual(attempt.notes, [])

    def test_only_passing_candidates_become_questions(self):
        run_item(self._item(), provider=self._provider())

        self.assertEqual(sorted(Question.objects.values_list("stem", flat=True)), ["Q2", "Q3"])
        for question in Question.objects.all():
            self.assertEqual(question.status, Question.Status.CANDIDATE)
            self.assertEqual(question.blueprint_row, self.row)
            self.assertEqual(question.source_chunk, self.chunk)

    def test_a_stored_question_is_reachable_from_the_attempt_that_passed_it(self):
        run_item(self._item(), provider=self._provider())

        record = ItemRun.objects.get()
        self.assertEqual(
            sorted(record.approved_questions.values_list("stem", flat=True)), ["Q2", "Q3"]
        )
        for attempt in record.attempts.filter(outcome=QuestionAttempt.Outcome.PASSED):
            self.assertIsNotNone(attempt.question)

    def test_re_running_an_item_does_not_duplicate_the_pool(self):
        run_item(self._item(), provider=self._provider())
        run_item(self._item(), provider=self._provider())

        self.assertEqual(Question.objects.count(), 2)
        self.assertEqual(ItemRun.objects.count(), 1)
        self.assertEqual(QuestionAttempt.objects.count(), 4)

    def test_a_capped_item_is_stored_as_needing_manual_attention(self):
        rounds = [batch(f"R{n}") for n in range(1, 9)]
        reject = {f"R{n}": rejection() for n in range(1, 9)}
        run_item(self._item(), provider=LoopProvider(*rounds, reject=reject))

        record = ItemRun.objects.get()
        self.assertEqual(record.status, ItemRun.Status.NEEDS_ATTENTION)
        self.assertTrue(record.needs_attention)
        self.assertEqual(record.shortfall, 2)
        self.assertEqual(record.rounds, 4)
        self.assertEqual(Question.objects.count(), 0)

    def test_the_log_survives_the_blueprint_row_it_describes(self):
        run_item(self._item(), provider=self._provider())
        self.row.delete()

        record = ItemRun.objects.get()
        self.assertIsNone(record.blueprint_row)
        self.assertEqual(record.topic_name, "Binary search")
        self.assertEqual(record.attempts.count(), 4)


# --- A whole blueprint -------------------------------------------------------


class WholeBlueprintTests(TestCase):
    def setUp(self):
        user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=user, name="Data Structures", code="CS201"
        )
        self.exam = Exam.objects.create(
            course=self.course, total_score=40, question_count=2, duration_minutes=60
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.searching = Topic.objects.create(course=self.course, name="Searching", position=0)
        self.sorting = Topic.objects.create(course=self.course, name="Sorting", position=1)
        for position, topic in enumerate([self.searching, self.sorting]):
            BlueprintRow.objects.create(
                blueprint=self.blueprint,
                topic=topic,
                question_type="mcq",
                level="medium",
                count=1,
                marks=Decimal("20.00"),
                weight_percent=Decimal("50.00"),
                position=position,
            )

    def _plan(self):
        from agents.analyze import build_exam_plan

        return build_exam_plan(self.blueprint, retrieve=lambda *a, **k: [a_passage()])

    def test_one_generation_item_per_row_carrying_the_row_s_passages(self):
        pairs = items_for_plan(self._plan())

        self.assertEqual([item.topic_name for _, item in pairs], ["Searching", "Sorting"])
        self.assertEqual([len(item.passages) for _, item in pairs], [1, 1])

    def test_the_run_reports_every_item_and_the_pool_it_produced(self):
        provider = LoopProvider(batch("Q1", "Q2"), batch("Q3", "Q4"))
        run = run_plan(self._plan(), provider=provider, persist=False)

        self.assertEqual(len(run.items), 2)
        self.assertEqual(run.required, 2)
        self.assertEqual(run.approved_count, 4)
        self.assertEqual(len(run.passed_first_round), 2)
        self.assertTrue(run.is_complete)
        self.assertTrue(run.counts_reconcile)
        self.assertEqual(run.items_needing_attention, [])

    def test_an_item_that_needs_attention_does_not_stop_the_others(self):
        rounds = [batch(f"R{n}") for n in range(1, 5)] + [batch("Q1")]
        reject = {f"R{n}": rejection() for n in range(1, 5)}
        run = run_plan(
            self._plan(), provider=LoopProvider(*rounds, reject=reject), persist=False
        )

        self.assertEqual(len(run.items), 2)
        self.assertEqual([item.status for item in run.items], [NEEDS_ATTENTION, PASSED])
        self.assertFalse(run.is_complete)
        self.assertEqual(run.rejected_count, 4)

    def test_an_outage_stops_the_run_rather_than_failing_every_remaining_item(self):
        provider = LoopProvider(batch("Q1"), batch("Q2"), generation_fails_on=1)
        run = run_plan(self._plan(), provider=provider, persist=False)

        self.assertEqual(len(run.items), 1)  # the second row was never attempted
        self.assertIn("insufficient_quota", run.aborted)
        self.assertFalse(run.is_complete)

    def test_run_exam_grounds_the_blueprint_then_runs_the_loop(self):
        provider = LoopProvider(batch("Q1", "Q2"), batch("Q3", "Q4"))
        run = run_exam(
            self.exam,
            provider=provider,
            retrieve=lambda *a, **k: [a_passage(chunk_id=self._chunk().pk)],
        )

        self.assertEqual(run.approved_count, 4)
        self.assertEqual(Question.objects.count(), 4)
        self.assertEqual(ItemRun.objects.count(), 2)
        self.assertEqual(len(run.questions), 4)

    def test_run_exam_refuses_a_blueprint_that_does_not_add_up(self):
        self.blueprint.rows.all().delete()
        BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.searching,
            count=99,
            marks=Decimal("1.00"),
            weight_percent=Decimal("100.00"),
        )
        with self.assertRaises(OrchestrationError) as caught:
            run_exam(self.exam, provider=LoopProvider(), retrieve=lambda *a, **k: [a_passage()])

        self.assertIn("does not add up", str(caught.exception))

    def test_run_exam_without_a_blueprint_is_refused_before_any_call(self):
        exam = Exam.objects.create(course=self.course, total_score=10, question_count=1)
        with self.assertRaises(OrchestrationError) as caught:
            run_exam(exam, provider=LoopProvider())

        self.assertIn("no blueprint", str(caught.exception))

    def _chunk(self):
        source_file, _ = SourceFile.objects.get_or_create(
            course=self.course,
            original_name="lecture-3.pdf",
            defaults=dict(kind=SourceFile.Kind.PDF, page_count=12),
        )
        chunk, _ = Chunk.objects.get_or_create(
            source_file=source_file,
            page=7,
            position=0,
            defaults=dict(text="A binary search halves the interval.", embedding=[0.0] * DIM),
        )
        return chunk
