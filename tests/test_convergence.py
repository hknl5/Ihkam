"""M10: how close two papers are, and the claims this milestone refuses to make.

Every provider here is a fake and every vector is written by hand: the suite has
made no network call since M2 and this milestone does not start. Two of the
fakes exist to prove specific things — `NoProvider` proves the deterministic
half of the comparison needs no model at all, and `LeakProvider` lets the same
pair of questions be judged twice, once each way, on identical vectors.

What is pinned down:

* the leakage pre-filter is **lenient**: a seeded pair that leaks completely and
  embeds nowhere near itself still reaches the model, because the echo channel
  does not go through vectors at all;
* the **model's verdict is the verdict** — the same pair, the same distance, two
  answers, two different reports. Cosine distance decides what is read and
  nothing else;
* **similarity is not leakage**: a near-duplicate pair is reported as its own
  finding and does not become a leak because it is close;
* expected difficulty is four proxies, computed from what the questions are;
* expected time is M9's `estimate_minutes`, summed — not a second estimator;
* **no equivalence percentage and no "actual difficulty" appears anywhere**, in
  the data or in the rendered screen. `dishonest_claims` is run over both.
"""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from courses.models import Course, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Form, FormQuestion, Question
from exams.services.convergence import (
    LEAKAGE_PREFILTER,
    MAX_JUDGED_PAIRS,
    ConvergenceError,
    Note,
    Placed,
    QuestionRef,
    build_rows,
    check_coverage,
    check_time,
    confirm_leaks,
    cosine,
    dishonest_claims,
    find_similar_pairs,
    profiles_from_forms,
    report_for_exam,
    report_for_forms,
    shortlist_leak_candidates,
)
from exams.services.forms import assemble_forms, estimate_minutes, save_assembly
from tests.test_forms import ExamFixture

PASSWORD = "quiet-precision-42"

MCQ = BlueprintRow.QuestionType.MCQ
NUMERIC = BlueprintRow.QuestionType.NUMERIC
SHORT = BlueprintRow.QuestionType.SHORT_ANSWER
DIRECT = BlueprintRow.Level.DIRECT
MEDIUM = BlueprintRow.Level.MEDIUM
MULTI_STEP = BlueprintRow.Level.MULTI_STEP

DIM = 8


def vector(index: int, dim: int = DIM) -> list[float]:
    """A unit vector along one axis — two of these are exactly orthogonal."""
    return [1.0 if position == index else 0.0 for position in range(dim)]


class NoProvider:
    """Any use of this is a test failure. The deterministic half must not call."""

    name = "none"

    def complete(self, *args, **kwargs):  # pragma: no cover - the point is not reaching it
        raise AssertionError("The deterministic comparison must not call the model.")

    def embed(self, texts):  # pragma: no cover - same
        raise AssertionError("The deterministic comparison must not embed.")


class LeakProvider:
    """Vectors by stem prefix, and a leakage verdict per pair of question codes.

    `verdicts` maps a frozenset of codes ({"Q1A", "Q2A"}) to the JSON the model
    would return. Anything not listed comes back as "no leak", which is what a
    real model does for most of a shortlist.
    """

    name = "leak-fake"

    def __init__(self, vectors: dict, verdicts: dict | None = None):
        self.vectors = vectors
        self.verdicts = verdicts or {}
        self.embedded: list[list[str]] = []
        self.judged: list[str] = []

    def embed(self, texts):
        self.embedded.append(list(texts))
        out = []
        for text in texts:
            match = next(
                (v for prefix, v in self.vectors.items() if text.startswith(prefix)), None
            )
            if match is None:
                raise AssertionError(f"no vector prepared for {text[:40]!r}")
            out.append(match)
        return out

    def complete(self, system, user, **kwargs):
        from agents.provider import LLMResponse

        codes = frozenset(part for part in _codes_in(user))
        self.judged.append(" / ".join(sorted(codes)))
        payload = self.verdicts.get(
            codes, {"leaks": False, "direction": "none", "reason": ""}
        )
        return LLMResponse(text=json.dumps(payload))


def _codes_in(text: str) -> list[str]:
    import re

    return re.findall(r"\bQ\d+[A-Z]\b", text)


def a_placed(
    number,
    form_label="A",
    *,
    question_id=None,
    stem="What is a stack?",
    options=(),
    answer="LIFO",
    topic="Recursion",
    question_type=MCQ,
    level=MEDIUM,
    marks="2",
    minutes="1.5",
    steps=0,
    needs_mark_review=False,
    from_ocr=False,
) -> Placed:
    return Placed(
        ref=QuestionRef(
            form_label=form_label,
            number=number,
            question_id=question_id if question_id is not None else number * 10,
        ),
        topic_name=topic,
        question_type=question_type,
        level=level,
        marks=Decimal(marks),
        minutes=Decimal(minutes),
        stem=stem,
        options=tuple(options),
        answer_text=answer,
        steps=steps,
        needs_mark_review=needs_mark_review,
        from_ocr=from_ocr,
    )


# --- The naming rule ---------------------------------------------------------


class HonestyRuleTests(SimpleTestCase):
    """The hard rule, checked as a rule rather than trusted as a habit."""

    def test_it_catches_the_forbidden_claims(self):
        self.assertTrue(dishonest_claims("Actual difficulty: 4.2"))
        self.assertTrue(dishonest_claims("The forms are 97% equivalent."))
        self.assertTrue(dishonest_claims("Equivalence score: 0.94"))
        self.assertTrue(dishonest_claims("These forms are 88% similar"))
        self.assertTrue(dishonest_claims("actual time taken"))

    def test_it_leaves_honest_sentences_alone(self):
        self.assertEqual(dishonest_claims("Expected difficulty: 4 worked steps"), [])
        self.assertEqual(dishonest_claims("Recursion is 30% of the paper's marks"), [])
        self.assertEqual(dishonest_claims("Expected time: about 58 minutes"), [])


# --- Expected difficulty -----------------------------------------------------


class ExpectedDifficultyTests(SimpleTestCase):
    """Four proxies, each counted from the question rather than judged."""

    def setUp(self):
        self.long_stem = "A student runs the algorithm on the array " + "value " * 60
        self.questions = [
            a_placed(1, stem="Define recursion.", answer="a function calling itself"),
            a_placed(
                2,
                stem="Given n = 12, compute T(n) = 2T(n/2) + n and state the order.",
                answer="O(n log n)",
                level=MULTI_STEP,
                question_type=NUMERIC,
                steps=4,
            ),
            a_placed(
                3,
                stem=self.long_stem,
                options=("one", "two", "three", "four", "five", "six"),
                answer="one",
            ),
        ]

    def _profile(self):
        from exams.services.convergence import FormProfile

        return FormProfile(label="A", questions=self.questions)

    def test_worked_steps_come_from_the_answer_key(self):
        profile = self._profile()
        self.assertEqual(profile.total_worked_steps, 3)  # 4 steps, 3 past the first

    def test_formula_presence_is_detected_in_the_stem(self):
        profile = self._profile()
        self.assertEqual(profile.formula_count, 1)
        self.assertTrue(self.questions[1].has_formula)
        self.assertFalse(self.questions[0].has_formula)

    def test_option_load_counts_only_options_past_four(self):
        self.assertEqual(self.questions[2].option_load, 2)
        self.assertEqual(self.questions[0].option_load, 0)

    def test_reading_length_counts_stem_and_options_together(self):
        self.assertGreater(self.questions[2].words, self.questions[0].words)
        self.assertEqual(self.questions[0].words, 2)

    def test_demanding_items_are_named_not_scored(self):
        profile = self._profile()
        codes = [question.ref.code for question in profile.demanding_questions]
        self.assertIn("Q3A", codes)
        # And there is no single difficulty number anywhere on the profile.
        self.assertFalse(hasattr(profile, "difficulty_score"))

    def test_the_rows_label_it_expected_difficulty(self):
        from exams.services.convergence import FormProfile

        profiles = [
            FormProfile(label="A", questions=self.questions),
            FormProfile(label="B", questions=self.questions[:2]),
        ]
        rows = build_rows(
            profiles, timing=check_time(profiles, limit_minutes=60), dimensions=[]
        )
        groups = {row.group for row in rows}
        self.assertIn("Expected difficulty", groups)
        self.assertNotIn("Actual difficulty", groups)
        self.assertEqual(dishonest_claims(json.dumps([row.as_dict() for row in rows])), [])


# --- Similarity --------------------------------------------------------------


class SimilarityTests(SimpleTestCase):
    """A near-duplicate is its own finding, and never becomes a leak by itself."""

    def test_high_similarity_pairs_are_flagged(self):
        first = a_placed(1, stem="Define a stack.")
        second = a_placed(2, stem="What is a stack?")
        far = a_placed(3, stem="Sort this array by merge sort.")
        vectors = {
            first.ref.question_id: vector(0),
            second.ref.question_id: vector(0),
            far.ref.question_id: vector(1),
        }
        pairs = find_similar_pairs([first, second, far], vectors)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].first.code, pairs[0].second.code}, {"Q1A", "Q2A"})
        self.assertEqual(pairs[0].similarity, Decimal("1"))

    def test_a_similar_pair_says_it_is_not_leakage(self):
        first = a_placed(1, stem="Define a stack.")
        second = a_placed(2, stem="What is a stack?")
        vectors = {first.ref.question_id: vector(0), second.ref.question_id: vector(0)}
        pair = find_similar_pairs([first, second], vectors)[0]
        self.assertIn("not leakage", pair.message)

    def test_the_same_question_on_both_forms_is_not_a_similar_pair(self):
        shared = a_placed(1, question_id=99, stem="Define a stack.")
        again = a_placed(4, form_label="B", question_id=99, stem="Define a stack.")
        vectors = {99: vector(0)}
        self.assertEqual(find_similar_pairs([shared, again], vectors), [])


# --- Leakage, stage one ------------------------------------------------------


class LeakagePrefilterTests(SimpleTestCase):
    """The lenient net: what reaches the model, and why."""

    def _leaking_pair(self):
        """A real leak that embeds nowhere near itself.

        Q1A mentions in passing that the tree's height is 7; Q2A asks for that
        height. They share almost no vocabulary and are orthogonal in vector
        space — the case a cosine-only pre-filter is guaranteed to miss.
        """
        stated = a_placed(
            1,
            stem=(
                "A binary search tree of height 7 stores the module's enrolment "
                "records. Which traversal prints the records in ascending order?"
            ),
            answer="in-order traversal",
        )
        asked = a_placed(
            2,
            stem="State the height of the enrolment tree described in this paper.",
            answer="7",
        )
        return stated, asked

    def test_a_leaking_pair_that_is_not_semantically_close_still_reaches_the_model(self):
        from exams.services.convergence import FormProfile

        stated, asked = self._leaking_pair()
        vectors = {stated.ref.question_id: vector(0), asked.ref.question_id: vector(1)}

        # Proof the pair is genuinely far apart: the cosine is below the
        # pre-filter threshold, so the similarity channel did not let it in.
        self.assertEqual(cosine(vector(0), vector(1)), Decimal("0"))
        self.assertLess(cosine(vector(0), vector(1)), LEAKAGE_PREFILTER)

        profile = FormProfile(label="A", questions=[stated, asked])
        candidates = shortlist_leak_candidates([profile], vectors)

        self.assertEqual(len(candidates), 1)
        self.assertIn("answer_echo", candidates[0].signals)
        self.assertNotIn("similarity", candidates[0].signals)

    def test_leakage_is_checked_within_a_paper_not_across_the_two(self):
        from exams.services.convergence import FormProfile

        stated, asked = self._leaking_pair()
        asked_b = a_placed(
            2,
            form_label="B",
            question_id=77,
            stem=asked.stem,
            answer=asked.answer_text,
        )
        vectors = {
            stated.ref.question_id: vector(0),
            asked.ref.question_id: vector(1),
            77: vector(1),
        }
        profiles = [
            FormProfile(label="A", questions=[stated]),
            FormProfile(label="B", questions=[asked_b]),
        ]
        # A student sits one paper; only what is printed on it can help them.
        self.assertEqual(shortlist_leak_candidates(profiles, vectors), [])

    def test_a_close_pair_gets_in_on_the_similarity_channel_alone(self):
        from exams.services.convergence import FormProfile

        first = a_placed(1, stem="Explain the quicksort partition step.", answer="pivot")
        second = a_placed(
            2, stem="Describe how merge sort divides an array.", answer="halves"
        )
        vectors = {first.ref.question_id: vector(0), second.ref.question_id: vector(0)}
        profile = FormProfile(label="A", questions=[first, second])
        candidates = shortlist_leak_candidates([profile], vectors)
        self.assertEqual(len(candidates), 1)
        self.assertIn("similarity", candidates[0].signals)

    def test_the_budget_is_split_so_neither_channel_buries_the_other(self):
        """What a real paper taught: one ranking cannot serve both channels.

        Many weak echo pairs must not push the closest pairs off the list, and
        a very close pair must not push off the one pair that states another's
        answer. Half the budget each, and whatever one channel does not use goes
        to the other.
        """
        from exams.services.convergence import Pair, select_pairs_to_judge

        def pair(number, *, echo, similarity):
            return Pair(
                first=a_placed(number, question_id=number),
                second=a_placed(number + 100, question_id=number + 100),
                similarity=Decimal(str(similarity)),
                signals=("answer_echo", "similarity") if echo else ("similarity",),
            )

        echoes = [pair(n, echo=True, similarity=0.40) for n in range(1, 31)]
        close = [pair(n, echo=False, similarity=0.95) for n in range(31, 61)]
        read, unread = select_pairs_to_judge(echoes + close, limit=16)

        self.assertEqual(len(read), 16)
        self.assertEqual(len(unread), 44)
        self.assertEqual(sum(1 for p in read if "answer_echo" in p.signals), 8)
        self.assertEqual(sum(1 for p in read if "answer_echo" not in p.signals), 8)

    def test_an_unused_half_of_the_budget_goes_to_the_other_channel(self):
        from exams.services.convergence import Pair, select_pairs_to_judge

        def pair(number, *, echo):
            return Pair(
                first=a_placed(number, question_id=number),
                second=a_placed(number + 100, question_id=number + 100),
                similarity=Decimal("0.9"),
                signals=("answer_echo", "similarity") if echo else ("similarity",),
            )

        candidates = [pair(1, echo=True)] + [pair(n, echo=False) for n in range(2, 30)]
        read, _unread = select_pairs_to_judge(candidates, limit=16)
        self.assertEqual(len(read), 16)
        self.assertIn("answer_echo", read[0].signals)

    def test_nothing_is_dropped_when_the_shortlist_fits(self):
        from exams.services.convergence import Pair, select_pairs_to_judge

        candidates = [
            Pair(
                first=a_placed(1),
                second=a_placed(2),
                similarity=Decimal("0.5"),
                signals=("similarity",),
            )
        ]
        read, unread = select_pairs_to_judge(candidates, limit=16)
        self.assertEqual(read, candidates)
        self.assertEqual(unread, [])

    def test_the_echo_pair_is_ranked_ahead_of_the_merely_similar_ones(self):
        """What the cap protects: the pair most likely to leak is read first."""
        from exams.services.convergence import FormProfile

        stated, asked = self._leaking_pair()
        filler = [
            a_placed(
                index,
                question_id=1000 + index,
                stem=f"Compare the traversal orders of two balanced trees, part {index}.",
                answer=f"variant {index}",
            )
            for index in range(3, 3 + 6)
        ]
        vectors = {q.ref.question_id: vector(0) for q in filler}
        vectors[stated.ref.question_id] = vector(2)
        vectors[asked.ref.question_id] = vector(3)

        profile = FormProfile(label="A", questions=[stated, asked, *filler])
        candidates = shortlist_leak_candidates([profile], vectors)

        self.assertEqual(candidates[0].codes, "Q1A / Q2A")
        self.assertIn("answer_echo", candidates[0].signals)
        self.assertGreater(len(candidates), 1)


# --- Leakage, stage two ------------------------------------------------------


class LeakageVerdictTests(SimpleTestCase):
    """The model decides. The vectors only decided what it read."""

    def _pair(self):
        from exams.services.convergence import FormProfile

        stated = a_placed(
            1,
            stem="A binary search tree of height 7 stores the enrolment records.",
            answer="in-order traversal",
        )
        asked = a_placed(2, stem="State the height of the enrolment tree.", answer="7")
        vectors = {stated.ref.question_id: vector(0), asked.ref.question_id: vector(1)}
        profile = FormProfile(label="A", questions=[stated, asked])
        return shortlist_leak_candidates([profile], vectors)

    def test_a_confirmed_leak_is_reported_with_the_pair(self):
        provider = LeakProvider(
            {},
            {
                frozenset({"Q1A", "Q2A"}): {
                    "leaks": True,
                    "direction": "a_reveals_b",
                    "reason": "Q1A states the height is 7, which is what Q2A asks for.",
                }
            },
        )
        leaks, unjudged = confirm_leaks(self._pair(), course_name="CS", provider=provider)
        self.assertEqual(unjudged, [])
        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0].source.code, "Q1A")
        self.assertEqual(leaks[0].target.code, "Q2A")
        self.assertEqual(leaks[0].message, "Q1A may help answer Q2A.")

    def test_the_model_stage_is_what_flips_the_verdict(self):
        """Same pair, same distance, two answers — and two different reports."""
        candidates = self._pair()
        clean = LeakProvider({})
        confirmed = LeakProvider(
            {},
            {
                frozenset({"Q1A", "Q2A"}): {
                    "leaks": True,
                    "direction": "a_reveals_b",
                    "reason": "the height is stated outright",
                }
            },
        )
        self.assertEqual(confirm_leaks(candidates, course_name="CS", provider=clean)[0], [])
        self.assertEqual(
            len(confirm_leaks(candidates, course_name="CS", provider=confirmed)[0]), 1
        )
        # Both providers were actually asked — the shortlist alone reported nothing.
        self.assertEqual(len(clean.judged), 1)
        self.assertEqual(len(confirmed.judged), 1)

    def test_both_directions_are_reported_as_two_notes(self):
        provider = LeakProvider(
            {},
            {
                frozenset({"Q1A", "Q2A"}): {
                    "leaks": True,
                    "direction": "both",
                    "reason": "each states what the other asks for",
                }
            },
        )
        leaks, _ = confirm_leaks(self._pair(), course_name="CS", provider=provider)
        self.assertEqual(
            {(leak.source.code, leak.target.code) for leak in leaks},
            {("Q1A", "Q2A"), ("Q2A", "Q1A")},
        )

    def test_an_unreadable_verdict_is_unjudged_not_clean(self):
        class Garbage:
            name = "garbage"

            def complete(self, *args, **kwargs):
                from agents.provider import LLMResponse

                return LLMResponse(text="not json at all")

            def embed(self, texts):  # pragma: no cover
                raise AssertionError

        leaks, unjudged = confirm_leaks(self._pair(), course_name="CS", provider=Garbage())
        self.assertEqual(leaks, [])
        self.assertEqual(len(unjudged), 1)
        self.assertIn("not as clean", unjudged[0][1])

    def test_a_leak_claimed_with_no_reason_is_not_accepted(self):
        provider = LeakProvider(
            {}, {frozenset({"Q1A", "Q2A"}): {"leaks": True, "direction": "a_reveals_b"}}
        )
        leaks, unjudged = confirm_leaks(self._pair(), course_name="CS", provider=provider)
        self.assertEqual(leaks, [])
        self.assertEqual(len(unjudged), 1)

    def test_a_call_that_never_landed_is_raised_not_swallowed(self):
        from exams.services.convergence import judge_pair

        class NoCredit:
            name = "broke"

            def complete(self, *args, **kwargs):
                raise RuntimeError("insufficient quota")

            def embed(self, texts):  # pragma: no cover
                raise AssertionError

        with self.assertRaises(ConvergenceError):
            judge_pair(self._pair()[0], course_name="CS", provider=NoCredit())


class LeakagePromptTests(SimpleTestCase):
    """The prompt has to say the thing the whole two-stage design rests on."""

    def test_it_tells_the_model_leakage_is_not_similarity(self):
        from agents.prompts.leakage import NOT_SIMILARITY_RULE, SYSTEM, build_user_prompt

        self.assertIn(NOT_SIMILARITY_RULE, SYSTEM)
        user = build_user_prompt(
            course_name="CS201",
            form_label="A",
            first=a_placed(1, stem="one", answer="alpha"),
            second=a_placed(2, stem="two", answer="beta"),
        )
        self.assertIn(NOT_SIMILARITY_RULE, user)
        # The answers are in the prompt: the judgement is about whether one
        # question hands the other's answer over, which needs both answers.
        self.assertIn("alpha", user)
        self.assertIn("beta", user)

    def test_it_forbids_rewriting_the_questions(self):
        from agents.prompts.leakage import JUDGE_ONLY_RULE, SYSTEM

        self.assertIn(JUDGE_ONLY_RULE, SYSTEM)
        self.assertIn("never rewrite", JUDGE_ONLY_RULE)


# --- Coverage and expected time ----------------------------------------------


class CoverageTests(SimpleTestCase):
    def test_matching_coverage_is_even(self):
        from exams.services.convergence import FormProfile

        profiles = [
            FormProfile(
                label=label,
                questions=[a_placed(1, label, topic="Recursion"),
                           a_placed(2, label, topic="Sorting", question_id=hash(label) % 97)],
            )
            for label in ("A", "B")
        ]
        coverage = check_coverage(profiles)
        self.assertEqual(coverage.topics, ["Recursion", "Sorting"])
        self.assertTrue(coverage.is_even)

    def test_a_topic_missing_from_one_form_is_named(self):
        from exams.services.convergence import FormProfile

        profiles = [
            FormProfile(label="A", questions=[a_placed(1, "A", topic="Recursion"),
                                              a_placed(2, "A", topic="Sorting")]),
            FormProfile(label="B", questions=[a_placed(1, "B", topic="Recursion",
                                                       question_id=55)]),
        ]
        coverage = check_coverage(profiles)
        self.assertEqual(coverage.missing["B"], ["Sorting"])
        self.assertFalse(coverage.is_even)


class TimingAgainstLimitTests(SimpleTestCase):
    def test_a_form_over_the_limit_is_named(self):
        from exams.services.convergence import FormProfile

        profiles = [
            FormProfile(label="A", questions=[a_placed(1, "A", minutes="70")]),
            FormProfile(label="B", questions=[a_placed(1, "B", question_id=42, minutes="55")]),
        ]
        timing = check_time(profiles, limit_minutes=60)
        self.assertEqual(timing.over, ["A"])
        self.assertEqual(timing.under, [])

    def test_a_form_far_under_the_limit_is_named_too(self):
        from exams.services.convergence import FormProfile

        profiles = [FormProfile(label="A", questions=[a_placed(1, "A", minutes="20")])]
        timing = check_time(profiles, limit_minutes=60)
        self.assertEqual(timing.under, ["A"])


# --- Against a real exam -----------------------------------------------------


class ComparisonFixture(TestCase):
    """Two saved forms, built by hand so the questions can be seeded."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, code="CS201", name="Data Structures"
        )
        self.exam = Exam.objects.create(
            course=self.course,
            title="Midterm",
            total_score=20,
            question_count=4,
            duration_minutes=60,
            number_of_forms=2,
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.topics = {}
        self.rows = {}
        for position, (name, question_type, level, marks) in enumerate(
            [("Recursion", MCQ, MEDIUM, "6"), ("Sorting", NUMERIC, MULTI_STEP, "14")]
        ):
            topic = Topic.objects.create(course=self.course, name=name, position=position)
            self.topics[name] = topic
            self.rows[name] = BlueprintRow.objects.create(
                blueprint=self.blueprint,
                topic=topic,
                question_type=question_type,
                level=level,
                count=1,
                marks=Decimal(marks),
                weight_percent=Decimal("30") if position == 0 else Decimal("70"),
                position=position,
            )
        self.forms = {
            label: Form.objects.create(
                exam=self.exam,
                label=label,
                position=position,
                expected_minutes=Decimal("0"),
                total_marks=Decimal("0"),
            )
            for position, label in enumerate(("A", "B"))
        }

    def place(
        self,
        form_label,
        *,
        topic="Recursion",
        stem,
        answer="A",
        options=(),
        marks="3",
        minutes="2",
        answer_key=None,
        mark_sum_ok=None,
        from_ocr=False,
    ):
        row = self.rows[topic]
        question = Question.objects.create(
            exam=self.exam,
            blueprint_row=row,
            stem=stem,
            question_type=row.question_type,
            options=list(options),
            correct=answer,
            answer_key=answer_key or {},
            mark_sum_ok=mark_sum_ok,
            from_ocr=from_ocr,
            source_ref="lecture.pdf · page 1",
        )
        form = self.forms[form_label]
        FormQuestion.objects.create(
            form=form,
            question=question,
            blueprint_row=row,
            position=form.entries.count(),
            expected_minutes=Decimal(minutes),
            marks=Decimal(marks),
        )
        return question


class ReportTests(ComparisonFixture):
    def _even_forms(self):
        for label in ("A", "B"):
            self.place(label, stem=f"Define recursion, version {label}.", answer="base case")
            self.place(
                label,
                topic="Sorting",
                stem=f"Sort the array using merge sort, version {label}.",
                answer="12",
                marks="7",
                minutes="5",
            )

    def test_the_deterministic_report_makes_no_call_at_all(self):
        self._even_forms()
        with patch("agents.provider.get_provider", side_effect=AssertionError("no calls")):
            report = report_for_exam(self.exam)
        self.assertEqual(len(report.profiles), 2)
        self.assertFalse(report.semantic_ran)
        self.assertEqual(report.leaks, [])

    def test_expected_time_is_m9s_estimator_summed(self):
        """The screen never re-estimates: it adds up what M9 already wrote down."""
        for label in ("A", "B"):
            self.place(
                label,
                topic="Sorting",
                stem="Compute the number of comparisons merge sort makes on 8 items.",
                answer="24",
                marks="14",
                minutes=str(
                    estimate_minutes(
                        question_type=NUMERIC,
                        level=MULTI_STEP,
                        stem="Compute the number of comparisons merge sort makes on 8 items.",
                        answer_key={"steps": [{"text": "one"}, {"text": "two"}]},
                    )
                ),
                answer_key={"steps": [{"text": "one"}, {"text": "two"}], "final_answer": "24"},
            )
        report = report_for_exam(self.exam)
        expected = estimate_minutes(
            question_type=NUMERIC,
            level=MULTI_STEP,
            stem="Compute the number of comparisons merge sort makes on 8 items.",
            answer_key={"steps": [{"text": "one"}, {"text": "two"}]},
        )
        for profile in report.profiles:
            self.assertEqual(profile.expected_minutes, Decimal(expected))
        self.assertEqual(report.timing.limit_minutes, 60)

    def test_the_report_carries_no_equivalence_claim_anywhere(self):
        self._even_forms()
        report = report_for_exam(self.exam)
        payload = json.dumps(report.as_dict(), ensure_ascii=False)
        self.assertEqual(dishonest_claims(payload), [])
        self.assertNotIn("equivalent", payload.lower())
        self.assertIn("expected_difficulty", payload)
        self.assertIn("expected_time", payload)

    def test_an_extra_multi_step_question_becomes_a_note(self):
        self._even_forms()
        self.place(
            "B",
            topic="Sorting",
            stem="Derive the recurrence for merge sort and solve it.",
            answer="O(n log n)",
            marks="7",
            minutes="5",
        )
        report = report_for_exam(self.exam)
        texts = [note.text for note in report.notes]
        self.assertTrue(
            any("Form B has one extra multi-step question" in text for text in texts), texts
        )

    def test_a_key_whose_steps_do_not_add_up_becomes_a_clarity_note(self):
        self._even_forms()
        question = self.place(
            "B",
            topic="Sorting",
            stem="Work through the partition step.",
            answer="7",
            marks="7",
            minutes="5",
            mark_sum_ok=False,
        )
        report = report_for_exam(self.exam)
        code = next(
            q.ref.code
            for profile in report.profiles
            for q in profile.questions
            if q.ref.question_id == question.pk
        )
        self.assertTrue(
            any(f"{code} needs a clarity check" in note.text for note in report.notes)
        )

    def test_the_semantic_half_finds_the_seeded_leak_and_the_near_duplicate(self):
        # Form A carries the leak: Q1A states the height, Q2A asks for it.
        stated = self.place(
            "A",
            stem=(
                "A binary search tree of height 7 stores the enrolment records. "
                "Which traversal prints them in ascending order?"
            ),
            answer="in-order traversal",
        )
        asked = self.place(
            "A",
            topic="Sorting",
            stem="State the height of the enrolment tree used above.",
            answer="7",
            marks="7",
            minutes="5",
        )
        # Form B carries a near-duplicate pair instead: same question, twice over.
        first_b = self.place("B", stem="Define recursion in your own words.", answer="base case")
        second_b = self.place(
            "B",
            topic="Sorting",
            stem="Explain recursion in your own words.",
            answer="base case",
            marks="7",
            minutes="5",
        )

        provider = LeakProvider(
            {
                "A binary search tree of height 7": vector(0),
                "State the height of the enrolment tree": vector(1),
                "Define recursion in your own words": vector(2),
                "Explain recursion in your own words": vector(2),
            },
            {
                frozenset({"Q1A", "Q2A"}): {
                    "leaks": True,
                    "direction": "a_reveals_b",
                    "reason": "Q1A states the height is 7, which is exactly what Q2A asks for.",
                }
            },
        )
        report = report_for_forms(
            list(self.exam.forms.all()),
            exam=self.exam,
            provider=provider,
            check_semantics=True,
        )

        self.assertTrue(report.semantic_ran)
        self.assertEqual(len(report.leaks), 1)
        self.assertEqual(report.leaks[0].source.question_id, stated.pk)
        self.assertEqual(report.leaks[0].target.question_id, asked.pk)

        # The near-duplicate is reported as similarity, not as a leak.
        self.assertEqual(len(report.similar_pairs), 1)
        pair = report.similar_pairs[0]
        self.assertEqual(
            {pair.first.question_id, pair.second.question_id}, {first_b.pk, second_b.pk}
        )
        self.assertNotIn(
            {first_b.pk, second_b.pk},
            [{leak.source.question_id, leak.target.question_id} for leak in report.leaks],
        )

        kinds = {note.kind for note in report.notes}
        self.assertIn("leakage", kinds)
        self.assertIn("similarity", kinds)
        self.assertEqual(dishonest_claims(json.dumps(report.as_dict(), ensure_ascii=False)), [])

    def test_pairs_past_the_cap_are_reported_as_unchecked_not_as_clean(self):
        """The cap bounds what a report costs; it never launders a pair clean."""
        for index in range(MAX_JUDGED_PAIRS + 3):
            self.place(
                "A",
                stem=f"Explain how a balanced tree keeps its height bounded, case {index}.",
                answer=f"rotation {index}",
                marks="1",
                minutes="1",
            )
        provider = LeakProvider({"Explain how a balanced tree": vector(0)})
        report = report_for_forms(
            list(self.exam.forms.all()),
            exam=self.exam,
            provider=provider,
            check_semantics=True,
        )
        self.assertGreater(report.shortlisted_pairs, MAX_JUDGED_PAIRS)
        self.assertEqual(report.judged_pairs, MAX_JUDGED_PAIRS)
        self.assertEqual(len(provider.judged), MAX_JUDGED_PAIRS)
        self.assertEqual(
            len(report.unread_pairs), report.shortlisted_pairs - MAX_JUDGED_PAIRS
        )
        # One note for all of them: 150 identical notes would bury the findings
        # the instructor is here to read.
        unchecked = [note for note in report.notes if note.kind == "leakage_unjudged"]
        self.assertEqual(len(unchecked), 1)
        self.assertIn("not read", unchecked[0].text)
        self.assertIn(str(len(report.unread_pairs)), unchecked[0].text)

    def test_two_matching_forms_produce_no_attention_notes(self):
        """Notes point at findings. Nothing found is a legitimate result."""
        self._even_forms()
        # A limit the paper actually fills, so the timing check has nothing to
        # say either — this fixture's four questions are seven expected minutes.
        self.exam.duration_minutes = 10
        self.exam.save(update_fields=["duration_minutes"])
        report = report_for_exam(self.exam)
        self.assertEqual([note for note in report.notes if note.tone != "info"], [])

    def test_ocr_sourced_questions_are_one_note_not_one_each(self):
        for label in ("A", "B"):
            for index in range(4):
                self.place(
                    label,
                    stem=f"Question {index} on form {label} from a scanned page.",
                    answer="A",
                    from_ocr=True,
                )
        report = report_for_exam(self.exam)
        ocr_notes = [note for note in report.notes if "OCR" in note.text]
        self.assertEqual(len(ocr_notes), 1)
        self.assertIn("8 questions cite a scanned page", ocr_notes[0].text)

    def test_a_provider_failure_is_reported_not_swallowed(self):
        self._even_forms()

        class Broken:
            name = "broken"

            def embed(self, texts):
                raise RuntimeError("rate limited")

            def complete(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError

        report = report_for_forms(
            list(self.exam.forms.all()), exam=self.exam, provider=Broken(), check_semantics=True
        )
        self.assertFalse(report.semantic_ran)
        self.assertIn("rate limited", report.semantic_error)


class ExpectedTimeFromAssemblyTests(ExamFixture):
    """The same check, on forms M9 actually assembled and saved."""

    def test_the_saved_forms_expected_time_is_the_estimator_summed(self):
        for row in self.rows:
            self.fill_pool(row, 8)
        save_assembly(self.exam, assemble_forms(self.exam))

        report = report_for_exam(self.exam)
        for form, profile in zip(self.exam.forms.all(), report.profiles, strict=True):
            recomputed = sum(
                estimate_minutes(
                    question_type=entry.question.question_type,
                    level=(entry.blueprint_row or entry.question.blueprint_row).level,
                    stem=entry.question.stem,
                    options=entry.question.options,
                    answer_key=entry.question.answer_key,
                )
                for entry in form.entries.select_related("question", "blueprint_row")
            )
            self.assertEqual(profile.expected_minutes, Decimal(recomputed))


# --- The screen --------------------------------------------------------------


class CompareScreenTests(ComparisonFixture):
    def setUp(self):
        super().setUp()
        self.client.login(username="nadia", password=PASSWORD)
        self.url = reverse("exams:compare", args=[self.course.pk, self.exam.pk])

    def _fill(self):
        for label in ("A", "B"):
            self.place(label, stem=f"Define recursion, version {label}.", answer="base case")
            self.place(
                label,
                topic="Sorting",
                stem=f"Sort the array using merge sort, version {label}.",
                answer="12",
                marks="7",
                minutes="5",
            )

    def test_it_renders_both_forms_with_honest_labels(self):
        self._fill()
        with patch("agents.provider.get_provider", side_effect=AssertionError("no calls")):
            response = self.client.get(self.url)
        self.assertContains(response, "Form A")
        self.assertContains(response, "Form B")
        self.assertContains(response, "Expected difficulty")
        self.assertContains(response, "Expected time")

    def test_the_screen_makes_no_equivalence_claim(self):
        self._fill()
        response = self.client.get(self.url)
        body = response.content.decode()
        self.assertEqual(dishonest_claims(body), [])
        self.assertNotIn("% equivalent", body)
        self.assertNotIn("Actual difficulty", body)
        self.assertNotIn("actual difficulty", body)

    def test_matching_forms_color_nothing(self):
        self._fill()
        response = self.client.get(self.url)
        body = response.content.decode()
        self.assertNotIn("cell--divergent", body)
        self.assertIn("data--compare", body)

    def test_only_the_divergent_cells_are_colored(self):
        self._fill()
        # One extra question on B: the counts diverge, the per-chapter rows for
        # Recursion do not.
        self.place(
            "B",
            topic="Sorting",
            stem="Derive the recurrence for merge sort and solve it, step by step.",
            answer="O(n log n)",
            marks="7",
            minutes="9",
        )
        response = self.client.get(self.url)
        report = response.context["report"]
        divergent = {row.key for row in report.rows if row.divergent}
        self.assertIn("question_count", divergent)
        self.assertNotIn("topic:Recursion", divergent)
        self.assertIn("cell--divergent", response.content.decode())

    def test_the_notes_are_shown(self):
        self._fill()
        self.place(
            "B",
            topic="Sorting",
            stem="Derive the recurrence for merge sort and solve it.",
            answer="O(n log n)",
            marks="7",
            minutes="5",
        )
        response = self.client.get(self.url)
        self.assertContains(response, "إحكام notes")
        self.assertContains(response, "Form B has one extra")

    def test_a_get_never_reaches_a_provider(self):
        self._fill()
        with patch("agents.provider.get_provider", side_effect=AssertionError("no calls")):
            self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_the_post_runs_the_semantic_checks(self):
        self._fill()
        provider = LeakProvider(
            {
                "Define recursion, version A": vector(0),
                "Sort the array using merge sort, version A": vector(1),
                "Define recursion, version B": vector(2),
                "Sort the array using merge sort, version B": vector(3),
            }
        )
        with patch("agents.provider.get_provider", return_value=provider):
            response = self.client.post(self.url)
        self.assertTrue(response.context["report"].semantic_ran)
        self.assertEqual(len(provider.embedded), 1)
        self.assertContains(response, "Leakage and similarity")

    def test_with_no_saved_forms_it_says_so_instead_of_comparing(self):
        Form.objects.filter(exam=self.exam).delete()
        response = self.client.get(self.url)
        self.assertContains(response, "No forms have been saved yet")

    def test_another_instructors_exam_does_not_exist(self):
        self.client.logout()
        User.objects.create_user("omar", password=PASSWORD)
        self.client.login(username="omar", password=PASSWORD)
        self.assertEqual(self.client.get(self.url).status_code, 404)


class NoteLabelTests(SimpleTestCase):
    def test_every_note_kind_has_a_written_label(self):
        for kind in Note.LABELS:
            self.assertTrue(Note(kind=kind, text="x").label)
        self.assertEqual(Note(kind="leakage", text="x").label, "Answer leakage")
