"""M12: an approved question becomes a reusable asset (the last MVP milestone).

What is held here is the claim M12 makes: an instructor's approval outlives the
exam it was given on. A question saved to the bank keeps everything the review
screen showed — stem, key, topic, type, level, marks, source — survives the
deletion of the exam it came from, and can be pulled into a later exam of the
**same course**, either one at a time or by a ratio applied to the whole paper.

Three rules are asserted rather than assumed:

* **Never a silently short exam.** When the bank cannot cover the share asked
  for, the shortfall is reported naming the topic and both counts, and the
  remainder is generated. M9 set this rule for forms; M12 holds it for sourcing.
* **Same course only.** Reuse is scoped by construction, and the test drives it
  through the view rather than the queryset — a scope enforced only in a screen
  is not a scope.
* **One search stack.** Bank search goes through `embed()` like everything else.
  The test asserts the provider's `embed` was called *and* that the ranking used
  the vectors it returned — which is what makes "no separate search backend"
  checkable rather than a promise in a docstring.

No network, as since M2: every provider is a fake, and the ones that must not be
called raise if they are.
"""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.test import TestCase
from django.urls import reverse

from agents.prompts.generate import SYSTEM as GENERATE_SYSTEM
from bank.models import BankQuestion, BankUsage
from bank.services.save import BankError, reuse_in_exam, save_to_bank
from bank.services.search import search_bank
from bank.services.sourcing import fill_from_bank, largest_remainder, plan_sourcing
from courses.models import Chunk, Course, SourceFile, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Question

from tests.test_generate import DIM, mcq, payload
from tests.test_review import all_ok

PASSWORD = "not-a-real-password"


# --- Fakes -------------------------------------------------------------------


def axis(index: int, value: float = 1.0) -> list[float]:
    """A unit vector on one axis — a "subject" the fake embedder can point at."""
    vector = [0.0] * DIM
    vector[index] = value
    return vector


class SubjectProvider:
    """An embedder that puts each subject on its own axis, and counts its calls.

    Real embeddings would make "does search rank the right question first?"
    depend on an API. Here the geometry is decided by the test: a stem about
    intervals and a query about intervals sit on the same axis, a stem about
    stacks sits on another, and cosine distance does the rest — in postgres,
    through the same code path production uses.
    """

    name = "subjects"
    #: word → axis. Anything matching none of them lands on the last axis, so an
    #: unrelated query is genuinely unrelated rather than accidentally close.
    SUBJECTS = ("interval", "stack", "graph")

    def __init__(self):
        self.embed_calls = 0
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embed_calls += 1
        self.embedded.extend(texts)
        vectors = []
        for text in texts:
            lowered = (text or "").lower()
            index = next(
                (i for i, word in enumerate(self.SUBJECTS) if word in lowered),
                len(self.SUBJECTS),
            )
            vectors.append(axis(index))
        return vectors

    def complete(self, system, user, **kwargs):  # pragma: no cover - never wanted
        raise AssertionError("Searching or banking a question must not call a model.")


class BankingProvider(SubjectProvider):
    """The subject embedder plus both agents, for the tests that generate.

    Told apart by the system prompt, as `JourneyProvider` is: generation returns
    the batches it was given, review passes everything. `generation_calls`
    is what the sourcing tests read to prove the bank *saved* calls.
    """

    name = "banking"

    def __init__(self, *batches):
        super().__init__()
        self.batches = list(batches)
        self.generation_calls = 0

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


class UnreachableProvider:
    """A provider that cannot be reached — for the save that must not fail."""

    name = "unreachable"

    def embed(self, texts):
        raise RuntimeError("the embedding service is down")

    def complete(self, system, user, **kwargs):  # pragma: no cover
        raise AssertionError("nothing here should call a model")


# --- Fixture -----------------------------------------------------------------


class BankFixture(TestCase):
    """One course, one topic, and helpers that build exams the product's way."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, name="Data Structures", code="CS210"
        )
        # The key terms are what `query_text_for` embeds, and they are what puts
        # the topic on the same axis as the stored chunks in `SubjectProvider`.
        self.topic = Topic.objects.create(
            course=self.course,
            name="Binary search",
            key_terms=["halving the interval"],
            position=0,
        )
        self.client.login(username="nadia", password=PASSWORD)

    # --- building blocks -------------------------------------------------

    def an_exam(self, *, count=2, score=4, **kwargs):
        return Exam.objects.create(
            course=self.course,
            total_score=score,
            question_count=count,
            duration_minutes=60,
            **kwargs,
        )

    def a_chunk(self, text="A binary search halves the interval at every step."):
        source_file, _ = SourceFile.objects.get_or_create(
            course=self.course,
            original_name="lecture-3.pdf",
            defaults=dict(kind=SourceFile.Kind.PDF, page_count=12),
        )
        return Chunk.objects.create(
            source_file=source_file,
            page=7,
            position=Chunk.objects.count(),
            text=text,
            embedding=axis(0),
        )

    def a_blueprint(self, exam, *, count=2, marks="4.00", topic=None, **row_kwargs):
        board = Blueprint.objects.create(exam=exam)
        BlueprintRow.objects.create(
            blueprint=board,
            topic=topic or self.topic,
            count=count,
            marks=Decimal(marks),
            weight_percent=Decimal("100.00"),
            **row_kwargs,
        )
        return board

    def an_approved_question(self, exam, *, row=None, stem="What does binary search halve?"):
        """A question in the state the bank admits: approved by the instructor."""
        if row is None:
            row = exam.blueprint.rows.first()
        return Question.objects.create(
            exam=exam,
            blueprint_row=row,
            stem=stem,
            question_type=BlueprintRow.QuestionType.MCQ,
            options=["The interval", "The list", "The key", "The index"],
            correct="The interval",
            explanation="Each step discards half of what is left.",
            source_ref="lecture-3.pdf · page 7",
            source_chunk=self.a_chunk(),
            answer_key={"answer": "The interval", "elements": []},
            status=Question.Status.APPROVED,
        )

    def bank_it(self, question, provider=None):
        return save_to_bank(question, provider=provider or SubjectProvider())

    def press_generate(self, exam, provider):
        with patch("agents.provider.get_provider", return_value=provider):
            return self.client.post(
                reverse("exams:generate", args=[self.course.pk, exam.pk])
            )

    def messages_of(self, response):
        return " ".join(str(m) for m in get_messages(response.wsgi_request))


# --- 1. Saving carries everything, and keeps it ------------------------------


class SaveToBankTests(BankFixture):
    def test_saving_carries_the_whole_question_and_its_metadata(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)

        banked = self.bank_it(question)

        self.assertEqual(banked.course, self.course)
        self.assertEqual(banked.stem, question.stem)
        self.assertEqual(banked.correct, "The interval")
        self.assertEqual(banked.options, question.options)
        self.assertEqual(banked.explanation, question.explanation)
        # The typed key (M6) — a banked question without one would be a question
        # someone has to write a key for later.
        self.assertEqual(banked.answer_key, question.answer_key)
        self.assertEqual(banked.topic, self.topic)
        self.assertEqual(banked.topic_name, "Binary search")
        self.assertEqual(banked.question_type, BlueprintRow.QuestionType.MCQ)
        self.assertEqual(banked.level, BlueprintRow.Level.MEDIUM)
        self.assertEqual(banked.marks, Decimal("2.00"))  # 4 marks over 2 questions
        self.assertEqual(banked.source_ref, "lecture-3.pdf · page 7")
        self.assertEqual(banked.source_page, 7)
        self.assertEqual(banked.origin_exam, exam)
        self.assertEqual(banked.origin_exam_title, exam.display_title)

    def test_only_an_approved_question_can_be_banked(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)
        question.status = Question.Status.CANDIDATE
        question.save(update_fields=["status"])

        with self.assertRaises(BankError) as caught:
            self.bank_it(question)

        self.assertIn("approved", str(caught.exception))
        self.assertEqual(BankQuestion.objects.count(), 0)

    def test_banking_the_same_question_twice_does_not_make_two_copies(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)

        first = self.bank_it(question)
        second = self.bank_it(question)

        self.assertEqual(first.pk, second.pk)
        self.assertEqual(BankQuestion.objects.count(), 1)

    def test_the_banked_question_survives_the_deletion_of_its_exam(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        banked = self.bank_it(self.an_approved_question(exam))
        title = exam.display_title

        exam.delete()

        banked.refresh_from_db()
        self.assertEqual(BankQuestion.objects.count(), 1)
        self.assertIsNone(banked.origin_exam)
        self.assertIsNone(banked.origin_question)
        # Still a whole question, and still readable as a sentence.
        self.assertEqual(banked.stem, "What does binary search halve?")
        self.assertEqual(banked.answer_key, {"answer": "The interval", "elements": []})
        self.assertEqual(banked.topic_name, "Binary search")
        self.assertEqual(banked.origin_exam_title, title)
        self.assertIn(title, banked.used_by)

    def test_editing_the_origin_question_does_not_change_the_banked_copy(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)
        banked = self.bank_it(question)

        question.stem = "Rewritten for this paper only"
        question.save(update_fields=["stem"])

        banked.refresh_from_db()
        self.assertEqual(banked.stem, "What does binary search halve?")

    def test_usage_records_which_exams_have_used_it(self):
        first = self.an_exam(title="Midterm")
        self.a_blueprint(first)
        banked = self.bank_it(self.an_approved_question(first))

        second = self.an_exam(title="Final")
        self.a_blueprint(second)
        reuse_in_exam(banked, exam=second, row=second.blueprint.rows.first())

        self.assertEqual(banked.usage_count, 2)
        self.assertEqual(banked.used_by, ["Midterm", "Final"])
        self.assertTrue(banked.usages.filter(exam=first, is_origin=True).exists())
        self.assertTrue(banked.usages.filter(exam=second, is_origin=False).exists())

    def test_the_same_banked_question_is_not_pulled_into_one_exam_twice(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        banked = self.bank_it(self.an_approved_question(exam))
        later = self.an_exam(title="Final")
        self.a_blueprint(later)
        row = later.blueprint.rows.first()

        reuse_in_exam(banked, exam=later, row=row)
        with self.assertRaises(BankError) as caught:
            reuse_in_exam(banked, exam=later, row=row)

        self.assertIn("already carries", str(caught.exception))
        self.assertEqual(later.questions.count(), 1)

    def test_a_question_the_embedder_could_not_reach_is_still_banked(self):
        """The approval is the thing being saved. Losing it to a timed-out
        embedding call would be the worse trade by far."""
        exam = self.an_exam()
        self.a_blueprint(exam)

        banked = save_to_bank(self.an_approved_question(exam), provider=UnreachableProvider())

        self.assertEqual(BankQuestion.objects.count(), 1)
        self.assertFalse(banked.is_indexed)
        # And it is still perfectly reusable — only unrankable.
        later = self.an_exam(title="Final")
        self.a_blueprint(later)
        reuse_in_exam(banked, exam=later, row=later.blueprint.rows.first())
        self.assertEqual(later.questions.count(), 1)

    def test_a_reused_question_is_a_copy_the_instructor_can_rework(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        banked = self.bank_it(self.an_approved_question(exam))
        later = self.an_exam(title="Final")
        self.a_blueprint(later)

        copy = reuse_in_exam(banked, exam=later, row=later.blueprint.rows.first())
        copy.stem = "Reworded for the final"
        copy.save(update_fields=["stem"])

        banked.refresh_from_db()
        self.assertEqual(banked.stem, "What does binary search halve?")
        # It arrives approved: the instructor approved it once already.
        self.assertEqual(copy.status, Question.Status.APPROVED)
        self.assertTrue(copy.is_from_bank)
        self.assertTrue(copy.is_locked)


# --- 2. Reuse is scoped to one course ----------------------------------------


class SameCourseTests(BankFixture):
    def setUp(self):
        super().setUp()
        self.other_course = Course.objects.create(
            instructor=self.user, name="Databases", code="CS340"
        )
        self.other_topic = Topic.objects.create(
            course=self.other_course, name="Normalization", position=0
        )

    def test_a_banked_question_cannot_be_reused_in_another_courses_exam(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        banked = self.bank_it(self.an_approved_question(exam))
        theirs = Exam.objects.create(course=self.other_course, question_count=2, total_score=4)
        Blueprint.objects.create(exam=theirs)
        row = BlueprintRow.objects.create(
            blueprint=theirs.blueprint,
            topic=self.other_topic,
            count=2,
            marks=Decimal("4.00"),
            weight_percent=Decimal("100.00"),
        )

        with self.assertRaises(BankError) as caught:
            reuse_in_exam(banked, exam=theirs, row=row)

        self.assertIn("CS210", str(caught.exception))
        self.assertEqual(theirs.questions.count(), 0)

    def test_the_pull_view_refuses_a_bank_question_from_another_course(self):
        """Scope is enforced in the query, not only on the screen."""
        exam = self.an_exam()
        self.a_blueprint(exam)
        banked = self.bank_it(self.an_approved_question(exam))
        theirs = Exam.objects.create(course=self.other_course, question_count=2, total_score=4)

        response = self.client.post(
            reverse("bank:use", args=[self.other_course.pk, banked.pk]),
            {"exam": theirs.pk, "row": 0},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(theirs.questions.count(), 0)

    def test_another_courses_bank_is_not_sourcing_material(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        self.bank_it(self.an_approved_question(exam))

        theirs = Exam.objects.create(
            course=self.other_course,
            question_count=2,
            total_score=4,
            sourcing=Exam.Sourcing.BANK,
        )
        Blueprint.objects.create(exam=theirs)
        BlueprintRow.objects.create(
            blueprint=theirs.blueprint,
            topic=self.other_topic,
            count=2,
            marks=Decimal("4.00"),
            weight_percent=Decimal("100.00"),
        )

        plan = plan_sourcing(theirs)

        self.assertEqual(plan.from_bank, 0)
        self.assertEqual(plan.to_generate, 2)
        self.assertEqual(fill_from_bank(theirs, plan), [])

    def test_the_browse_screen_shows_only_this_courses_bank(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        self.bank_it(self.an_approved_question(exam))

        response = self.client.get(reverse("bank:browse", args=[self.other_course.pk]))

        self.assertNotContains(response, "What does binary search halve?")
        self.assertContains(response, "Nothing in the bank yet")

    def test_another_instructors_bank_is_not_reachable_at_all(self):
        omar = User.objects.create_user("omar", password=PASSWORD)
        their_course = Course.objects.create(instructor=omar, name="OOP", code="CS201")

        response = self.client.get(reverse("bank:browse", args=[their_course.pk]))

        self.assertEqual(response.status_code, 404)


# --- 3. Search: the same embed(), and no second stack ------------------------


class BankSearchTests(BankFixture):
    def a_bank_of_two(self):
        exam = self.an_exam(count=2)
        self.a_blueprint(exam, count=2)
        row = exam.blueprint.rows.first()
        provider = SubjectProvider()
        intervals = self.bank_it(
            self.an_approved_question(
                exam, row=row, stem="What does binary search halve at every step? interval"
            ),
            provider,
        )
        stacks = self.bank_it(
            self.an_approved_question(
                exam, row=row, stem="Which order does a stack pop its elements in?"
            ),
            provider,
        )
        return intervals, stacks, provider

    def test_search_returns_the_relevant_banked_question_first(self):
        intervals, stacks, _ = self.a_bank_of_two()
        provider = SubjectProvider()

        result = search_bank(self.course, "halving the interval", provider=provider)

        self.assertEqual([hit.question.pk for hit in result.hits], [intervals.pk])
        self.assertNotIn(stacks.pk, [hit.question.pk for hit in result.hits])
        self.assertEqual(result.searched, 2)

    def test_search_goes_through_the_same_embed_as_everything_else(self):
        """One embedding path — no separate search backend to run or migrate."""
        self.a_bank_of_two()
        provider = SubjectProvider()

        with patch("agents.provider.get_provider", return_value=provider) as seam:
            result = search_bank(self.course, "stack")

        self.assertTrue(seam.called, "The search did not go through get_provider().")
        self.assertEqual(provider.embed_calls, 1, "One query, one embedding call.")
        self.assertEqual(provider.embedded, ["stack"])
        # And the ranking used those vectors: the stack question came back.
        self.assertEqual(result.found, 1)
        self.assertIn("stack", result.hits[0].question.stem.lower())

    def test_search_ranks_over_the_stored_vectors_rather_than_the_words(self):
        """No text index is consulted: a query sharing no word with the stem
        still finds it, because the vectors are what is compared."""
        intervals, _stacks, _ = self.a_bank_of_two()

        result = search_bank(self.course, "the interval", provider=SubjectProvider())

        self.assertEqual([hit.question.pk for hit in result.hits], [intervals.pk])

    def test_an_unembedded_banked_question_is_reported_rather_than_hidden(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        save_to_bank(self.an_approved_question(exam), provider=UnreachableProvider())

        result = search_bank(self.course, "interval", provider=SubjectProvider())

        self.assertEqual(result.found, 0)
        self.assertEqual(result.unindexed, 1)
        self.assertIn("not embedded", result.summary)

    def test_a_failed_embedding_call_degrades_the_screen_instead_of_breaking_it(self):
        self.a_bank_of_two()

        result = search_bank(self.course, "interval", provider=UnreachableProvider())

        self.assertTrue(result.error)
        self.assertEqual(result.found, 0)

    def test_the_browse_screen_searches_and_shows_the_metadata(self):
        intervals, _stacks, _ = self.a_bank_of_two()
        provider = SubjectProvider()

        with patch("agents.provider.get_provider", return_value=provider):
            response = self.client.post(
                reverse("bank:browse", args=[self.course.pk]),
                {"query": "halving the interval"},
            )

        self.assertContains(response, intervals.stem)
        self.assertContains(response, "Binary search")  # topic
        self.assertContains(response, "Multiple choice")  # type
        self.assertContains(response, "Applied")  # level
        self.assertContains(response, "lecture-3.pdf")  # source
        self.assertContains(response, "used in")  # usage count


# --- 4. Sourcing a new exam: new, bank, and a mix ----------------------------


class SourcingTests(BankFixture):
    def a_stocked_bank(self, how_many=4, *, level=BlueprintRow.Level.MEDIUM):
        """`how_many` approved, banked questions on this course's one topic."""
        exam = self.an_exam(count=how_many, score=how_many * 2, title="Source exam")
        self.a_blueprint(exam, count=how_many, marks=Decimal(how_many * 2), level=level)
        row = exam.blueprint.rows.first()
        provider = SubjectProvider()
        return [
            self.bank_it(
                self.an_approved_question(exam, row=row, stem=f"Banked interval question {i}"),
                provider,
            )
            for i in range(how_many)
        ]

    def test_fully_new_generates_everything_and_touches_no_bank_question(self):
        self.a_stocked_bank(4)
        exam = self.an_exam(count=2, title="New exam")
        self.a_blueprint(exam, count=2)
        self.a_chunk()
        provider = BankingProvider(batch("Fresh question 1", "Fresh question 2"))

        self.press_generate(exam, provider)

        self.assertEqual(exam.questions.count(), 2)
        self.assertEqual(exam.questions.filter(bank_source__isnull=False).count(), 0)
        self.assertTrue(provider.generation_calls)

    def test_from_bank_fills_the_slots_from_the_bank_and_generates_nothing(self):
        banked = self.a_stocked_bank(4)
        exam = self.an_exam(count=2, title="Reuse exam", sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=2)
        self.a_chunk()
        provider = BankingProvider()  # any generation call would raise

        response = self.press_generate(exam, provider)

        self.assertEqual(provider.generation_calls, 0, "The bank covered it; nothing to write.")
        self.assertEqual(exam.questions.count(), 2)
        self.assertEqual(exam.questions.filter(bank_source__isnull=False).count(), 2)
        self.assertEqual(
            set(exam.questions.values_list("bank_source", flat=True)),
            {banked[0].pk, banked[1].pk},
        )
        self.assertIn("came from the bank", self.messages_of(response))

    def test_a_mix_splits_the_exam_at_the_ratio_that_was_set(self):
        self.a_stocked_bank(4)
        exam = self.an_exam(
            count=5,
            score=10,
            title="Mixed exam",
            sourcing=Exam.Sourcing.MIX,
            bank_share_percent=60,
        )
        self.a_blueprint(exam, count=5, marks="10.00")
        self.a_chunk()
        # 60% of 5 is 3 from the bank, so 2 are written.
        provider = BankingProvider(batch("Fresh 1", "Fresh 2"))

        plan = plan_sourcing(exam)
        self.assertEqual(plan.from_bank, 3)
        self.assertEqual(plan.to_generate, 2)

        self.press_generate(exam, provider)

        self.assertEqual(exam.questions.count(), 5)
        self.assertEqual(exam.questions.filter(bank_source__isnull=False).count(), 3)
        self.assertEqual(exam.questions.filter(bank_source__isnull=True).count(), 2)

    def test_the_ratio_is_applied_over_the_exam_and_not_row_by_row(self):
        """Rows of 3 and 2 at 50% is 3 questions from the bank overall — not one
        rounding decision per row that lands somewhere else entirely."""
        second_topic = Topic.objects.create(
            course=self.course, name="Sorting", key_terms=["graph order"], position=1
        )
        self.a_stocked_bank(3)

        exam = self.an_exam(
            count=5, score=10, sourcing=Exam.Sourcing.MIX, bank_share_percent=50
        )
        board = self.a_blueprint(exam, count=3, marks="6.00")
        BlueprintRow.objects.create(
            blueprint=board,
            topic=second_topic,
            count=2,
            marks=Decimal("4.00"),
            weight_percent=Decimal("40.00"),
        )

        plan = plan_sourcing(exam)

        # Round(5 × 50%) = 3, and only the first topic has stock — so the whole
        # 3 comes from there rather than 2 + a row that cannot supply one.
        self.assertEqual(plan.target, 3)
        self.assertEqual(plan.from_bank, 3)
        self.assertEqual(plan.to_generate, 2)
        self.assertFalse(plan.is_short)

    def test_largest_remainder_splits_a_share_without_losing_a_question(self):
        self.assertEqual(sum(largest_remainder(6, [5, 3, 2])), 6)
        self.assertEqual(largest_remainder(6, [5, 3, 2]), [3, 2, 1])
        self.assertEqual(largest_remainder(0, [5, 3]), [0, 0])
        # A part never exceeds the row it is filling.
        self.assertEqual(largest_remainder(10, [2, 2]), [2, 2])

    def test_a_bank_question_pulled_in_is_not_regenerated_over(self):
        """The M11 rule, extended: a reuse carries an approval already given."""
        self.a_stocked_bank(4)
        exam = self.an_exam(
            count=4, score=8, sourcing=Exam.Sourcing.MIX, bank_share_percent=50
        )
        self.a_blueprint(exam, count=4, marks="8.00")
        self.a_chunk()

        self.press_generate(exam, BankingProvider(batch("Written 1", "Written 2")))
        reused = set(
            exam.questions.filter(bank_source__isnull=False).values_list("stem", flat=True)
        )
        self.assertEqual(len(reused), 2)

        # A second generation over the same rows: the reused questions are
        # carried, not rewritten, and the log says why.
        self.press_generate(exam, BankingProvider(batch("Written 3", "Written 4")))

        self.assertTrue(
            reused.issubset(set(exam.questions.values_list("stem", flat=True))),
            "A reused question was regenerated over.",
        )
        notes = [
            note
            for attempt in exam.item_runs.first().attempts.all()
            for note in attempt.notes
        ]
        self.assertIn(
            "Carried unchanged: this question was reused from the course's bank.", notes
        )

    def test_pressing_generate_twice_does_not_drain_the_bank_into_one_exam(self):
        """The share is a share of the paper, not of every press of the button."""
        self.a_stocked_bank(4)
        exam = self.an_exam(count=2, sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=2)
        self.a_chunk()

        self.press_generate(exam, BankingProvider())
        self.press_generate(exam, BankingProvider())

        self.assertEqual(exam.questions.count(), 2)
        self.assertEqual(exam.questions.filter(bank_source__isnull=False).count(), 2)

    def test_reused_questions_reach_the_pool_the_forms_are_assembled_from(self):
        """A reused question that M9 cannot see would be a paper with a hole."""
        from exams.services.forms import pool_for_exam

        self.a_stocked_bank(4)
        exam = self.an_exam(count=2, sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=2)
        self.a_chunk()

        self.press_generate(exam, BankingProvider())

        pool = pool_for_exam(exam)
        self.assertEqual(sum(len(items) for items in pool.values()), 2)


# --- 5. An insufficient bank is reported, never silently absorbed ------------


class InsufficientBankTests(BankFixture):
    def test_the_shortfall_names_the_topic_and_both_counts(self):
        exam = self.an_exam(count=5, score=10, sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=5, marks="10.00")
        # Three banked questions against five needed.
        source = self.an_exam(count=3, score=6, title="Source")
        self.a_blueprint(source, count=3, marks="6.00")
        row = source.blueprint.rows.first()
        provider = SubjectProvider()
        for index in range(3):
            self.bank_it(
                self.an_approved_question(source, row=row, stem=f"Banked {index}"), provider
            )

        plan = plan_sourcing(exam)

        self.assertTrue(plan.is_short)
        self.assertEqual(plan.short_by, 2)
        line = plan.shortfalls[0]
        self.assertIn("Binary search", line)
        self.assertIn("Bank has 3", line)
        self.assertIn("5 needed", line)

    def test_the_remainder_is_generated_rather_than_left_out(self):
        exam = self.an_exam(count=4, score=8, sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=4, marks="8.00")
        source = self.an_exam(count=2, score=4, title="Source")
        self.a_blueprint(source, count=2, marks="4.00")
        row = source.blueprint.rows.first()
        provider = SubjectProvider()
        for index in range(2):
            self.bank_it(
                self.an_approved_question(source, row=row, stem=f"Banked {index}"), provider
            )
        self.a_chunk()

        response = self.press_generate(
            exam, BankingProvider(batch("Written 1", "Written 2"))
        )

        # Four questions: two reused, two written. Never a short exam.
        self.assertEqual(exam.questions.count(), 4)
        self.assertEqual(exam.questions.filter(bank_source__isnull=False).count(), 2)
        self.assertEqual(exam.questions.filter(bank_source__isnull=True).count(), 2)
        message = self.messages_of(response)
        self.assertIn("Bank has 2", message)
        self.assertIn("4 needed", message)
        self.assertIn("writing the remainder", message)

    def test_an_empty_bank_reports_itself_and_the_exam_is_still_written(self):
        exam = self.an_exam(count=2, sourcing=Exam.Sourcing.BANK)
        self.a_blueprint(exam, count=2)
        self.a_chunk()

        response = self.press_generate(exam, BankingProvider(batch("Written 1", "Written 2")))

        self.assertEqual(exam.questions.count(), 2)
        self.assertIn("short", self.messages_of(response))

    def test_the_generate_screen_shows_the_split_before_anything_is_pressed(self):
        exam = self.an_exam(count=4, score=8, sourcing=Exam.Sourcing.MIX, bank_share_percent=50)
        self.a_blueprint(exam, count=4, marks="8.00")
        source = self.an_exam(count=1, score=2, title="Source")
        self.a_blueprint(source, count=1, marks="2.00")
        self.bank_it(self.an_approved_question(source, row=source.blueprint.rows.first()))

        response = self.client.get(reverse("exams:generate", args=[self.course.pk, exam.pk]))

        self.assertContains(response, "Where the questions come from")
        self.assertContains(response, "The bank cannot cover that share")
        self.assertContains(response, "Bank has 1 question for Binary search")

    def test_no_shortfall_is_reported_when_the_share_was_actually_met(self):
        """A report an instructor cannot act on is noise."""
        second_topic = Topic.objects.create(
            course=self.course, name="Sorting", key_terms=["graph order"], position=1
        )
        source = self.an_exam(count=3, score=6, title="Source")
        self.a_blueprint(source, count=3, marks="6.00")
        row = source.blueprint.rows.first()
        provider = SubjectProvider()
        for index in range(3):
            self.bank_it(
                self.an_approved_question(source, row=row, stem=f"Banked {index}"), provider
            )

        exam = self.an_exam(count=5, score=10, sourcing=Exam.Sourcing.MIX, bank_share_percent=50)
        board = self.a_blueprint(exam, count=3, marks="6.00")
        BlueprintRow.objects.create(
            blueprint=board,
            topic=second_topic,
            count=2,
            marks=Decimal("4.00"),
            weight_percent=Decimal("40.00"),
        )

        plan = plan_sourcing(exam)

        self.assertFalse(plan.is_short)
        self.assertEqual(plan.shortfalls, [])


# --- 6. The journey: every M12 act is reachable by button --------------------


class BankJourneyTests(BankFixture):
    def test_the_review_screen_carries_save_to_bank_on_an_approved_question(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        self.an_approved_question(exam)

        response = self.client.get(reverse("exams:review", args=[self.course.pk, exam.pk]))

        self.assertContains(response, 'value="save_to_bank"')
        self.assertContains(response, "Save to bank")

    def test_save_to_bank_is_not_offered_on_a_question_nobody_has_approved(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)
        question.status = Question.Status.CANDIDATE
        question.save(update_fields=["status"])

        response = self.client.get(reverse("exams:review", args=[self.course.pk, exam.pk]))

        self.assertNotContains(response, 'value="save_to_bank"')

    def test_pressing_save_to_bank_banks_the_question(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)

        with patch("agents.provider.get_provider", return_value=SubjectProvider()):
            response = self.client.post(
                reverse(
                    "exams:question_action", args=[self.course.pk, exam.pk, question.pk]
                ),
                {"action": "save_to_bank"},
            )

        self.assertEqual(BankQuestion.objects.filter(course=self.course).count(), 1)
        self.assertIn("Saved to the CS210 bank", self.messages_of(response))

    def test_pressing_save_to_bank_on_an_unapproved_question_says_why_not(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        question = self.an_approved_question(exam)
        question.status = Question.Status.CANDIDATE
        question.save(update_fields=["status"])

        response = self.client.post(
            reverse("exams:question_action", args=[self.course.pk, exam.pk, question.pk]),
            {"action": "save_to_bank"},
        )

        self.assertEqual(BankQuestion.objects.count(), 0)
        self.assertIn("Approve it first", self.messages_of(response))

    def test_the_bank_is_reachable_by_button_from_review_generate_and_the_exam_list(self):
        exam = self.an_exam()
        self.a_blueprint(exam)
        self.an_approved_question(exam)
        bank_url = reverse("bank:browse", args=[self.course.pk])

        for name, args in (
            ("exams:review", [self.course.pk, exam.pk]),
            ("exams:generate", [self.course.pk, exam.pk]),
            ("exams:list", [self.course.pk]),
        ):
            with self.subTest(screen=name):
                response = self.client.get(reverse(name, args=args))
                self.assertContains(response, bank_url)
                self.assertContains(response, "the bank")

    def test_the_sourcing_choice_is_on_the_spec_screen_and_the_generate_screen(self):
        exam = self.an_exam()
        self.a_blueprint(exam)

        spec = self.client.get(reverse("exams:list", args=[self.course.pk]))
        self.assertContains(spec, "Where the questions come from")
        self.assertContains(spec, "From the bank")

        generate = self.client.get(reverse("exams:generate", args=[self.course.pk, exam.pk]))
        self.assertContains(generate, 'value="sourcing"')
        self.assertContains(generate, "Save sourcing")

    def test_the_sourcing_choice_can_be_saved_from_the_generate_screen(self):
        exam = self.an_exam()
        self.a_blueprint(exam)

        response = self.client.post(
            reverse("exams:generate", args=[self.course.pk, exam.pk]),
            {"action": "sourcing", "sourcing": Exam.Sourcing.MIX, "bank_share_percent": 60},
        )

        exam.refresh_from_db()
        self.assertEqual(exam.sourcing, Exam.Sourcing.MIX)
        self.assertEqual(exam.bank_share_percent, 60)
        self.assertEqual(exam.bank_share, 60)
        self.assertIn("Sourcing saved", self.messages_of(response))

    def test_the_sourcing_choice_is_saved_with_a_new_exam_spec(self):
        response = self.client.post(
            reverse("exams:list", args=[self.course.pk]),
            {
                "kind": Exam.Kind.MIDTERM,
                "total_score": 20,
                "question_count": 10,
                "duration_minutes": 60,
                "language": Exam.Language.ENGLISH,
                "number_of_forms": 1,
                "sourcing": Exam.Sourcing.MIX,
                "bank_share_percent": 40,
            },
        )

        self.assertEqual(response.status_code, 302)
        exam = Exam.objects.latest("pk")
        self.assertEqual(exam.sourcing, Exam.Sourcing.MIX)
        self.assertEqual(exam.bank_share, 40)

    def test_a_spec_with_no_sourcing_answer_is_a_fully_new_exam(self):
        """Silence is not a choice — it is the behaviour every earlier milestone had."""
        self.client.post(
            reverse("exams:list", args=[self.course.pk]),
            {
                "kind": Exam.Kind.QUIZ,
                "total_score": 10,
                "question_count": 5,
                "duration_minutes": 30,
                "language": Exam.Language.ENGLISH,
                "number_of_forms": 1,
            },
        )

        exam = Exam.objects.latest("pk")
        self.assertEqual(exam.sourcing, Exam.Sourcing.NEW)
        self.assertEqual(exam.bank_share, 0)

    def test_a_question_can_be_pulled_into_a_slot_from_the_browse_screen(self):
        source = self.an_exam(title="Midterm")
        self.a_blueprint(source)
        banked = self.bank_it(self.an_approved_question(source))

        later = self.an_exam(title="Final")
        self.a_blueprint(later)
        row = later.blueprint.rows.first()

        screen = self.client.get(
            reverse("bank:browse", args=[self.course.pk]), {"exam": later.pk}
        )
        self.assertContains(screen, "Use in this slot")
        self.assertContains(screen, reverse("bank:use", args=[self.course.pk, banked.pk]))

        response = self.client.post(
            reverse("bank:use", args=[self.course.pk, banked.pk]),
            {"exam": later.pk, "row": row.pk},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response["Location"], reverse("exams:review", args=[self.course.pk, later.pk])
        )
        copy = later.questions.get()
        self.assertEqual(copy.bank_source, banked)
        self.assertEqual(copy.blueprint_row, row)
        self.assertEqual(copy.status, Question.Status.APPROVED)
        self.assertEqual(copy.answer_key, banked.answer_key)

    def test_the_review_screen_says_a_question_came_from_the_bank(self):
        source = self.an_exam(title="Midterm")
        self.a_blueprint(source)
        banked = self.bank_it(self.an_approved_question(source))
        later = self.an_exam(title="Final")
        self.a_blueprint(later)
        reuse_in_exam(banked, exam=later, row=later.blueprint.rows.first())

        response = self.client.get(reverse("exams:review", args=[self.course.pk, later.pk]))

        self.assertContains(response, "From the bank · reused")

    def test_a_pull_with_no_row_says_so_instead_of_guessing(self):
        source = self.an_exam()
        self.a_blueprint(source)
        banked = self.bank_it(self.an_approved_question(source))
        later = self.an_exam(title="Final")

        response = self.client.post(
            reverse("bank:use", args=[self.course.pk, banked.pk]), {"exam": later.pk}
        )

        self.assertEqual(later.questions.count(), 0)
        self.assertIn("Pick the blueprint row", self.messages_of(response))

    def test_the_bank_screen_needs_a_login(self):
        self.client.logout()

        response = self.client.get(reverse("bank:browse", args=[self.course.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])


# --- 7. No network, ever -----------------------------------------------------


class NoNetworkTests(BankFixture):
    def test_banking_and_reusing_a_question_calls_no_model(self):
        """One embedding call to index the stem, and not one completion."""
        exam = self.an_exam()
        self.a_blueprint(exam)
        provider = SubjectProvider()  # `complete` raises if it is ever reached

        banked = self.bank_it(self.an_approved_question(exam), provider)
        later = self.an_exam(title="Final")
        self.a_blueprint(later)
        reuse_in_exam(banked, exam=later, row=later.blueprint.rows.first())

        self.assertEqual(provider.embed_calls, 1)

    def test_planning_the_sourcing_of_an_exam_costs_nothing(self):
        """The Generate screen shows the split on every visit, so it must be free."""
        exam = self.an_exam(count=2, sourcing=Exam.Sourcing.MIX, bank_share_percent=50)
        self.a_blueprint(exam, count=2)
        provider = SubjectProvider()

        with patch("agents.provider.get_provider", return_value=provider):
            plan = plan_sourcing(exam)
            self.client.get(reverse("exams:generate", args=[self.course.pk, exam.pk]))

        self.assertEqual(provider.embed_calls, 0)
        self.assertEqual(plan.required, 2)
