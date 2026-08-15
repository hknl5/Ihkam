"""M11: the decision surface, and the edit that survives everything (M11).

Every provider here is a fake — the suite has made no network call since M2 and
the decision screen does not start. Most of these tests pass no provider at all,
because approving a question is not a model's business.

The rule this file exists to hold is the hard one:

* an **automatic** correction-loop pass leaves an instructor-edited question
  exactly as it is, keeps it in the pool, and says nothing — the instructor did
  not ask for that run;
* a revision the **instructor asks for** on that same question stops and asks
  first, and changes nothing until they confirm;
* a question nobody edited is regenerated normally, with no confirmation, so the
  lock is a lock on *edits* rather than a lock on the pipeline.

The rest is the card: every action drives the Decision Rail to the right state,
and the state is always paired with a word.
"""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from agents.orchestrator import CARRIED_NOTE, locked_questions, run_item
from agents.prompts.revision import BRIEFS, CLARIFY, EASIER, HARDER, MODES, REGENERATE
from courses.models import Chunk, Course, SourceFile, Topic
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
from exams.services.revision import (
    RevisionError,
    RevisionNeedsConfirmation,
    revise,
)
from tests.test_generate import ScriptedProvider, a_passage, an_item, mcq, payload
from tests.test_review import verdicts

PASSWORD = "quiet-precision-42"

MCQ = BlueprintRow.QuestionType.MCQ
MEDIUM = BlueprintRow.Level.MEDIUM


def a_provider(*stems, review="pass"):
    """A provider scripted to write each stem and then have it reviewed.

    Generation and review alternate, which is the order `revise` calls them in:
    one candidate, one verdict.
    """
    texts = []
    for stem in stems:
        texts.append(payload(mcq(stem=stem)))
        texts.append(verdicts() if review == "pass" else verdicts(clarity={
            "ok": False,
            "reason": "two readings of the same sentence",
            "requirement": "state it so it has one interpretation",
        }))
    return ScriptedProvider(*texts)


def no_retrieval(course, topic, **kwargs):
    """Retrieval, replaced. The revision path must not need a vector database.

    The chunk id is real, because a stored question carries a foreign key to the
    passage it cites and a citation pointing at nothing is not a citation.
    """
    return [a_passage(chunk_id=Chunk.objects.values_list("pk", flat=True).first())]


class ExamFixture(TestCase):
    """One exam, one row, one stored question, on one form."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, code="CS201", name="Data Structures"
        )
        self.exam = Exam.objects.create(
            course=self.course,
            title="Midterm",
            total_score=10,
            question_count=2,
            duration_minutes=60,
            number_of_forms=2,
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.topic = Topic.objects.create(course=self.course, name="Binary search")
        self.source_file = SourceFile.objects.create(
            course=self.course, original_name="lecture-3.pdf", kind=SourceFile.Kind.PDF
        )
        self.chunk = Chunk.objects.create(
            source_file=self.source_file,
            page=7,
            position=0,
            text="A binary search halves the interval at every step.",
            embedding=[0.0] * 1536,
        )
        self.row = BlueprintRow.objects.create(
            blueprint=self.blueprint,
            topic=self.topic,
            question_type=MCQ,
            level=MEDIUM,
            count=1,
            marks=Decimal("5"),
            weight_percent=Decimal("100"),
        )
        self.run = ItemRun.objects.create(
            exam=self.exam,
            blueprint_row=self.row,
            topic_name=self.topic.name,
            question_type=MCQ,
            level=MEDIUM,
            required=1,
            approved_count=1,
            rounds=1,
            status=ItemRun.Status.PASSED,
        )
        self.question = self.make_question("Which step does a binary search repeat?")
        self.form_a = Form.objects.create(exam=self.exam, label="A", position=0)
        self.form_b = Form.objects.create(exam=self.exam, label="B", position=1)
        FormQuestion.objects.create(
            form=self.form_a,
            question=self.question,
            blueprint_row=self.row,
            position=0,
            marks=Decimal("5"),
            expected_minutes=Decimal("2"),
        )

    def make_question(self, stem, **kwargs):
        question = Question.objects.create(
            exam=self.exam,
            blueprint_row=self.row,
            stem=stem,
            question_type=MCQ,
            options=["Halve the interval", "Scan every element", "Sort the list", "Hash the key"],
            correct="Halve the interval",
            source_ref="lecture-3.pdf · page 7",
            answer_key={"kind": "objective", "answer": "Halve the interval"},
            **kwargs,
        )
        QuestionAttempt.objects.create(
            item_run=self.run,
            round=1,
            outcome=QuestionAttempt.Outcome.PASSED,
            stem=question.stem,
            question=question,
        )
        return question

    def login(self):
        self.client.login(username="nadia", password=PASSWORD)

    def action_url(self, question=None):
        question = question or self.question
        return reverse(
            "exams:question_action", args=[self.course.pk, self.exam.pk, question.pk]
        )

    def review_url(self):
        return reverse("exams:review", args=[self.course.pk, self.exam.pk])


# --- The rail ----------------------------------------------------------------


class DecisionRailTests(ExamFixture):
    """Every action drives the rail, and the rail is never colour alone."""

    def test_a_fresh_candidate_is_grey(self):
        self.assertEqual(self.question.rail_state, "candidate")

    def test_a_flagged_candidate_is_amber(self):
        self.question.mark_sum_ok = False
        self.assertEqual(self.question.rail_state, "attention")
        self.question.mark_sum_ok = None
        self.question.from_ocr = True
        self.assertEqual(self.question.rail_state, "attention")

    def test_approving_turns_it_teal(self):
        self.login()
        self.client.post(self.action_url(), {"action": "approve"})
        self.question.refresh_from_db()
        self.assertEqual(self.question.status, Question.Status.APPROVED)
        self.assertEqual(self.question.rail_state, "approved")

    def test_rejecting_turns_it_red_and_outranks_a_flag(self):
        self.question.from_ocr = True
        self.question.save(update_fields=["from_ocr"])
        self.login()
        self.client.post(self.action_url(), {"action": "reject"})
        self.question.refresh_from_db()
        self.assertEqual(self.question.status, Question.Status.REJECTED)
        self.assertEqual(self.question.rail_state, "rejected")

    def test_the_decision_outranks_the_flag_in_both_directions(self):
        """An instructor's decision is not overridden by إحكام's flag."""
        self.question.from_ocr = True
        self.question.status = Question.Status.APPROVED
        self.assertEqual(self.question.rail_state, "approved")

    def test_the_card_pairs_every_colour_with_a_word(self):
        self.login()
        response = self.client.get(self.review_url())
        body = response.content.decode()
        self.assertIn("is-candidate", body)
        self.assertIn("Candidate", body)
        self.assertIn("rail-legend", body)

    def test_deleting_removes_the_question(self):
        self.login()
        self.client.post(self.action_url(), {"action": "delete"})
        self.assertFalse(Question.objects.filter(pk=self.question.pk).exists())

    def test_an_unknown_action_changes_nothing(self):
        self.login()
        response = self.client.post(self.action_url(), {"action": "bless"}, follow=True)
        self.question.refresh_from_db()
        self.assertEqual(self.question.status, Question.Status.CANDIDATE)
        self.assertContains(response, "There is no")


class ReviewScreenTests(ExamFixture):
    def test_it_renders_the_card_with_the_rail_and_the_notes(self):
        self.question.mark_sum_ok = False
        self.question.save(update_fields=["mark_sum_ok"])
        self.login()
        response = self.client.get(self.review_url())
        self.assertContains(response, "card--rail")
        self.assertContains(response, "is-attention")
        self.assertContains(response, "System notes")
        self.assertContains(response, "do not add up")
        self.assertContains(response, "lecture-3.pdf")

    def test_it_offers_every_action_the_milestone_names(self):
        self.login()
        body = self.client.get(self.review_url()).content.decode()
        for label in (
            "Approve",
            "Reject",
            "Regenerate",
            "Make easier",
            "Make harder",
            "Clarify wording",
            "Delete",
            "Edit this question",
            "Move to form",
        ):
            self.assertIn(label, body)

    def test_it_makes_no_model_call_on_load(self):
        self.login()
        with patch("agents.provider.get_provider", side_effect=AssertionError("no calls")):
            self.assertEqual(self.client.get(self.review_url()).status_code, 200)

    def test_another_instructors_exam_does_not_exist(self):
        User.objects.create_user("omar", password=PASSWORD)
        self.client.login(username="omar", password=PASSWORD)
        self.assertEqual(self.client.get(self.review_url()).status_code, 404)
        self.assertEqual(
            self.client.post(self.action_url(), {"action": "approve"}).status_code, 404
        )

    def test_moving_a_question_to_the_other_form(self):
        self.login()
        self.client.post(self.action_url(), {"action": "move", "form_id": self.form_b.pk})
        entry = self.question.form_entries.get()
        self.assertEqual(entry.form, self.form_b)


# --- The edit, and the lock it sets ------------------------------------------


class EditTests(ExamFixture):
    def test_saving_an_edit_locks_the_question(self):
        self.login()
        self.client.post(
            self.action_url(),
            {
                "action": "edit",
                f"q{self.question.pk}-stem": "Which interval does a binary search halve?",
                f"q{self.question.pk}-correct": "Halve the interval",
                f"q{self.question.pk}-options_text": (
                    "Halve the interval\nScan every element\nSort the list\nHash the key"
                ),
                f"q{self.question.pk}-explanation": "",
            },
        )
        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, "Which interval does a binary search halve?")
        self.assertTrue(self.question.instructor_edited)
        self.assertIsNotNone(self.question.edited_at)
        self.assertTrue(self.question.is_locked)

    def test_the_lock_is_visible_on_the_card(self):
        self.question.mark_edited()
        self.login()
        self.assertContains(self.client.get(self.review_url()), "locked")

    def test_an_answer_that_is_not_an_option_is_refused(self):
        self.login()
        response = self.client.post(
            self.action_url(),
            {
                "action": "edit",
                f"q{self.question.pk}-stem": "Edited",
                f"q{self.question.pk}-correct": "Something else entirely",
                f"q{self.question.pk}-options_text": "Halve the interval\nScan every element",
                f"q{self.question.pk}-explanation": "",
            },
            follow=True,
        )
        self.question.refresh_from_db()
        self.assertNotEqual(self.question.stem, "Edited")
        self.assertFalse(self.question.instructor_edited)
        self.assertContains(response, "one of the options")

    def test_an_edit_keeps_the_objective_key_pointing_at_the_answer(self):
        self.login()
        self.client.post(
            self.action_url(),
            {
                "action": "edit",
                f"q{self.question.pk}-stem": "Which step is repeated?",
                f"q{self.question.pk}-correct": "Scan every element",
                f"q{self.question.pk}-options_text": "Halve the interval\nScan every element",
                f"q{self.question.pk}-explanation": "",
            },
        )
        self.question.refresh_from_db()
        self.assertEqual(self.question.answer_key["answer"], "Scan every element")

    def test_approving_is_not_editing(self):
        """A decision is not a rewrite: approving must not set the lock."""
        self.login()
        self.client.post(self.action_url(), {"action": "approve"})
        self.question.refresh_from_db()
        self.assertFalse(self.question.instructor_edited)


# --- The hard rule: the automatic loop never overwrites an edit ---------------


class AutomaticLoopRespectsTheLockTests(ExamFixture):
    """M11's hard rule, tested where it could actually be broken."""

    def _run_the_loop(self, *stems):
        """One automatic pass over this row, writing `stems`."""
        item = an_item(
            question_type=MCQ,
            count=1,
            passages=(a_passage(chunk_id=self.chunk.pk),),
            row_id=self.row.pk,
            exam_id=self.exam.pk,
        )
        return run_item(
            item,
            provider=a_provider(*stems),
            exam=self.exam,
            row=self.row,
            max_rounds=0,
        )

    def test_an_edited_question_is_not_overwritten_and_stays_in_the_pool(self):
        self.question.stem = "My own wording, which no model wrote."
        self.question.save(update_fields=["stem"])
        self.question.mark_edited()

        self._run_the_loop("A completely different question about binary search?")

        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, "My own wording, which no model wrote.")
        self.assertTrue(self.question.instructor_edited)
        # Still in M9's pool, which is read through the attempt log the run just
        # replaced — the way an edit would have been lost without the carry.
        self.run.refresh_from_db()
        self.assertIn(self.question, list(self.run.approved_questions))

    def test_the_carry_is_recorded_as_carried_not_as_a_round(self):
        self.question.mark_edited()
        self._run_the_loop("Another question?")
        carried = self.run.attempts.filter(round=0)
        self.assertEqual(carried.count(), 1)
        self.assertIn(CARRIED_NOTE, carried.get().notes)

    def test_the_loop_says_nothing_to_the_instructor_about_it(self):
        """Silently: they did not ask for this run, so it has nothing to tell them."""
        self.question.mark_edited()
        result = self._run_the_loop("Another question?")
        self.assertEqual(result.error, "")
        self.assertEqual([question.pk for question in result.carried], [self.question.pk])

    def test_an_unedited_question_is_not_carried(self):
        """The lock is on edits, not on the pipeline."""
        result = self._run_the_loop("A fresh question about binary search?")
        self.assertEqual(result.carried, [])
        self.assertEqual(self.run.attempts.filter(round=0).count(), 0)

    def test_a_rejected_edit_is_not_carried_back_into_the_pool(self):
        """An instructor's rejection outranks their earlier edit."""
        self.question.mark_edited()
        self.question.status = Question.Status.REJECTED
        self.question.save(update_fields=["status"])
        self.assertEqual(locked_questions(exam=self.exam, row=self.row), [])


# --- The instructor's own revision -------------------------------------------


class RevisionTests(ExamFixture):
    def test_an_unedited_question_is_regenerated_without_a_confirmation(self):
        provider = a_provider("A different question about binary search entirely?")
        result = revise(
            self.question, REGENERATE, provider=provider, retrieve=no_retrieval
        )
        self.question.refresh_from_db()
        self.assertTrue(result.replaced)
        self.assertEqual(
            self.question.stem, "A different question about binary search entirely?"
        )

    def test_a_locked_question_stops_and_asks_first(self):
        self.question.mark_edited()
        original = self.question.stem

        class NoCalls:
            name = "none"

            def complete(self, *args, **kwargs):
                raise AssertionError("nothing may be generated before the instructor confirms")

            def embed(self, texts):
                raise AssertionError

        with self.assertRaises(RevisionNeedsConfirmation) as caught:
            revise(self.question, REGENERATE, provider=NoCalls(), retrieve=no_retrieval)

        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, original)
        self.assertTrue(self.question.instructor_edited)
        self.assertIn("edited manually", str(caught.exception))

    def test_confirming_obeys_and_replaces_the_question_in_place(self):
        self.question.mark_edited()
        entry_pk = self.question.form_entries.get().pk

        revise(
            self.question,
            REGENERATE,
            provider=a_provider("The replacement question about binary search?"),
            confirmed=True,
            retrieve=no_retrieval,
        )

        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, "The replacement question about binary search?")
        # The lock comes off, the approval does not transfer, and the paper is
        # intact: the same placement, so Form A's question 1 is still there.
        self.assertFalse(self.question.instructor_edited)
        self.assertIsNone(self.question.edited_at)
        self.assertEqual(self.question.status, Question.Status.CANDIDATE)
        self.assertEqual(self.question.form_entries.get().pk, entry_pk)

    def test_every_mode_sends_its_own_brief_to_agent_2a(self):
        for mode in (REGENERATE, EASIER, HARDER, CLARIFY):
            with self.subTest(mode=mode):
                provider = a_provider(f"A {mode} question about binary search?")
                revise(self.question, mode, provider=provider, retrieve=no_retrieval)
                generation_prompt = provider.calls[0][1]
                self.assertIn(BRIEFS[mode][:60], generation_prompt)

    def test_a_replacement_that_fails_review_is_not_stored(self):
        original = self.question.stem
        provider = a_provider("An ambiguous replacement?", review="reject")
        with self.assertRaises(RevisionError) as caught:
            revise(self.question, CLARIFY, provider=provider, retrieve=no_retrieval)
        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, original)
        self.assertIn("rejected the replacement", str(caught.exception))

    def test_the_rejected_replacement_is_still_written_into_the_log(self):
        provider = a_provider("An ambiguous replacement?", review="reject")
        with self.assertRaises(RevisionError):
            revise(self.question, CLARIFY, provider=provider, retrieve=no_retrieval)
        attempt = self.run.attempts.order_by("-pk").first()
        self.assertEqual(attempt.outcome, QuestionAttempt.Outcome.REJECTED)
        self.assertIn("Instructor asked for", attempt.note)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(RevisionError):
            revise(self.question, "make_it_sing", retrieve=no_retrieval)

    def test_every_mode_has_a_brief_and_a_pair_of_labels(self):
        from agents.prompts.revision import LABELS

        for mode in MODES:
            self.assertTrue(BRIEFS[mode])
            self.assertEqual(len(LABELS[mode]), 2)


class RevisionThroughTheScreenTests(ExamFixture):
    """The two presses the hard rule describes, as an instructor makes them."""

    def test_the_first_press_warns_and_changes_nothing(self):
        self.question.mark_edited()
        original = self.question.stem
        self.login()

        with patch("agents.provider.get_provider", side_effect=AssertionError("no calls")):
            response = self.client.post(self.action_url(), {"action": "regenerate"})

        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, original)
        self.assertIn(f"confirm={self.question.pk}", response["Location"])
        self.assertIn("mode=regenerate", response["Location"])

    def test_the_warning_is_shown_on_that_card(self):
        self.question.mark_edited()
        self.login()
        response = self.client.get(
            self.review_url(), {"confirm": self.question.pk, "mode": "regenerate"}
        )
        self.assertContains(response, "edited manually")
        self.assertContains(response, "Discard my edit")
        self.assertContains(response, "Keep my edit")

    def test_the_second_press_obeys(self):
        self.question.mark_edited()
        self.login()
        provider = a_provider("The confirmed replacement about binary search?")
        with patch("agents.provider.get_provider", return_value=provider), patch(
            "agents.analyze.plan_questions_for_row"
        ) as planned:
            planned.return_value = [
                type("P", (), {"passages": (a_passage(chunk_id=self.chunk.pk),)})()
            ]
            self.client.post(
                self.action_url(), {"action": "regenerate", "confirmed": "1"}
            )

        self.question.refresh_from_db()
        self.assertEqual(
            self.question.stem, "The confirmed replacement about binary search?"
        )
        self.assertFalse(self.question.instructor_edited)

    def test_an_unedited_question_needs_no_confirmation_from_the_screen(self):
        self.login()
        provider = a_provider("A fresh replacement about binary search?")
        with patch("agents.provider.get_provider", return_value=provider), patch(
            "agents.analyze.plan_questions_for_row"
        ) as planned:
            planned.return_value = [
                type("P", (), {"passages": (a_passage(chunk_id=self.chunk.pk),)})()
            ]
            response = self.client.post(self.action_url(), {"action": "regenerate"})

        self.question.refresh_from_db()
        self.assertEqual(self.question.stem, "A fresh replacement about binary search?")
        self.assertNotIn("confirm=", response["Location"])
