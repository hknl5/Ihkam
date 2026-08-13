"""M3: retrieval — the passages a topic actually lives in.

Embeddings are faked, so the suite spends no API calls, but the fake is not
arbitrary: each concept gets its own axis in the vector space, so cosine
similarity between "logic" text and "logic" query really is 1.0 and between
"logic" and "recursion" really is 0.0. That makes the ranking assertions below
statements about retrieval, not about the mock.

The rule this file exists to hold down: a passage under a topic the instructor
marked "not taught" can never come back, not even when it is the single best
match in the course.
"""

from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from courses.models import Chunk, Course, SourceFile, Topic
from courses.services.retrieval import (
    Passage,
    RetrievalError,
    query_text_for,
    retrieve,
)

PASSWORD = "quiet-precision-42"

#: One axis per concept. Anything on the same axis matches perfectly, anything
#: on another is orthogonal — a clean stand-in for "about the same subject".
CONCEPTS = ["logic", "recursion", "ethics", "arrays"]


def vec(concept: str, *, blend: str | None = None, weight: float = 0.0, dim: int = 1536):
    """A unit vector for ``concept``, optionally leaning ``weight`` toward another.

    ``blend`` is how the mid-range scores are produced: a passage that is mostly
    about one subject and partly about another, which is exactly the case the
    score floor has to judge.
    """
    import math

    vector = [0.0] * dim
    vector[CONCEPTS.index(concept)] = 1.0
    if blend:
        vector[CONCEPTS.index(blend)] = weight
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class ConceptProvider:
    """Embeds by looking for a concept word. Refuses to complete anything."""

    name = "concept"

    def __init__(self):
        self.embedded = []

    def embed(self, texts):
        self.embedded.extend(texts)
        out = []
        for text in texts:
            lowered = text.lower()
            match = next((c for c in CONCEPTS if c in lowered), None)
            out.append(vec(match) if match else [0.0] * 1536)
        return out

    def complete(self, *args, **kwargs):  # pragma: no cover - the failure is the point
        raise AssertionError("Retrieval must never call a completion model.")


def a_course_with_file(user=None, code="CS310", name="ML"):
    user = user or User.objects.create_user(f"nadia-{code}", password=PASSWORD)
    course = Course.objects.create(instructor=user, name=name, code=code)
    source_file = SourceFile.objects.create(
        course=course, original_name="lecture.pdf", kind=SourceFile.Kind.PDF, page_count=20
    )
    return course, source_file


def a_chunk(source_file, text, concept, *, page=1, position=None, topic=None, **kwargs):
    position = source_file.chunks.filter(page=page).count() if position is None else position
    return Chunk.objects.create(
        source_file=source_file,
        page=page,
        position=position,
        text=text,
        embedding=vec(concept, **kwargs),
        topic=topic,
    )


class QueryCompositionTests(TestCase):
    """Both query shapes end up as one piece of text to embed."""

    def test_free_text_is_the_query(self):
        self.assertEqual(query_text_for("  truth tables  "), "truth tables")

    def test_a_topic_becomes_its_name_and_key_terms(self):
        topic = Topic(name="Propositional logic", key_terms=["truth table", "tautology"])

        self.assertEqual(
            query_text_for(topic), "Propositional logic — truth table, tautology"
        )

    def test_a_topic_with_no_terms_is_just_its_name(self):
        self.assertEqual(query_text_for(Topic(name="Recursion")), "Recursion")

    def test_blank_terms_are_dropped_rather_than_embedded(self):
        topic = Topic(name="Ethics", key_terms=["", "   ", "bias"])

        self.assertEqual(query_text_for(topic), "Ethics — bias")


class RetrieveTests(TestCase):
    """Ranking, scoping, the score floor, and the exclusion rule."""

    def setUp(self):
        self.course, self.file = a_course_with_file()
        self.provider = ConceptProvider()

        self.logic_topic = Topic.objects.create(
            course=self.course, name="Propositional logic", key_terms=["truth table"],
            source_file=self.file, page_start=1, page_end=3, position=0,
        )
        self.on_topic = a_chunk(
            self.file, "A truth table enumerates every assignment.", "logic",
            page=1, topic=self.logic_topic,
        )
        self.off_topic = a_chunk(
            self.file, "A recursive call needs a base case.", "recursion", page=8
        )

    def retrieve(self, query, **kwargs):
        kwargs.setdefault("provider", self.provider)
        return retrieve(self.course, query, **kwargs)

    # --- both query shapes ---------------------------------------------------

    def test_free_text_returns_the_passage_about_it(self):
        passages = self.retrieve("how does a truth table prove logic equivalence")

        self.assertEqual([p.chunk_id for p in passages], [self.on_topic.pk])
        self.assertAlmostEqual(passages[0].score, 1.0, places=5)

    def test_a_topic_returns_the_passages_of_its_subject(self):
        passages = self.retrieve(self.logic_topic)

        self.assertEqual([p.chunk_id for p in passages], [self.on_topic.pk])

    def test_a_topic_is_embedded_from_its_name_and_terms_not_its_stored_vector(self):
        self.retrieve(self.logic_topic)

        self.assertEqual(
            self.provider.embedded, ["Propositional logic — truth table"]
        )

    def test_an_empty_query_asks_no_model_and_returns_nothing(self):
        self.assertEqual(self.retrieve("   "), [])
        self.assertEqual(self.provider.embedded, [])

    # --- what comes back -----------------------------------------------------

    def test_a_passage_carries_its_page_file_topic_and_score(self):
        passage = self.retrieve("logic")[0]

        self.assertIsInstance(passage, Passage)
        self.assertEqual(passage.page, 1)
        self.assertEqual(passage.source_file, "lecture.pdf")
        self.assertEqual(passage.topic, "Propositional logic")
        self.assertEqual(passage.text, self.on_topic.text)
        self.assertEqual(passage.page_ref, "lecture.pdf · page 1")

    def test_a_passage_no_topic_claims_is_still_retrievable(self):
        passages = self.retrieve("recursion")

        self.assertEqual([p.chunk_id for p in passages], [self.off_topic.pk])
        self.assertIsNone(passages[0].topic)

    def test_passages_come_back_best_first(self):
        near = a_chunk(
            self.file, "Logic gates, mostly.", "logic", page=2, blend="arrays", weight=0.9
        )

        passages = self.retrieve("logic", min_score=0.0)

        self.assertEqual([p.chunk_id for p in passages][:2], [self.on_topic.pk, near.pk])
        self.assertGreater(passages[0].score, passages[1].score)

    # --- course scoping ------------------------------------------------------

    def test_another_course_s_passages_never_leak_in(self):
        other_course, other_file = a_course_with_file(code="CS999", name="Other")
        a_chunk(other_file, "An identical truth table page.", "logic")

        passages = self.retrieve("logic")

        self.assertEqual([p.chunk_id for p in passages], [self.on_topic.pk])

    def test_another_course_of_the_same_instructor_does_not_leak_either(self):
        same_owner = Course.objects.create(
            instructor=self.course.instructor, name="Second", code="CS311"
        )
        second_file = SourceFile.objects.create(
            course=same_owner, original_name="other.pdf", kind=SourceFile.Kind.PDF
        )
        a_chunk(second_file, "Truth tables again, other course.", "logic")

        self.assertEqual([p.chunk_id for p in self.retrieve("logic")], [self.on_topic.pk])

    # --- the score floor -----------------------------------------------------

    def test_weak_matches_are_cut_rather_than_returned(self):
        # Both chunks exist; only one is about the query.
        passages = self.retrieve("logic", k=8, min_score=0.5)

        self.assertEqual(len(passages), 1)
        self.assertNotIn(self.off_topic.pk, [p.chunk_id for p in passages])

    def test_the_result_is_not_padded_out_to_k(self):
        for i in range(5):
            a_chunk(self.file, f"Unrelated ethics passage {i}", "ethics", page=10 + i)

        passages = self.retrieve("logic", k=8)

        self.assertEqual(len(passages), 1)

    def test_a_lower_floor_lets_a_partial_match_through(self):
        partial = a_chunk(
            self.file, "Arrays, with a note on logic.", "arrays", page=4,
            blend="logic", weight=0.5,
        )

        strict = [p.chunk_id for p in self.retrieve("logic", min_score=0.8)]
        lenient = [p.chunk_id for p in self.retrieve("logic", min_score=0.3)]

        self.assertNotIn(partial.pk, strict)
        self.assertIn(partial.pk, lenient)

    def test_k_caps_the_result(self):
        for i in range(4):
            a_chunk(self.file, f"More logic, page {i}", "logic", page=20 + i)

        self.assertEqual(len(self.retrieve("logic", k=2)), 2)

    def test_k_of_zero_asks_no_model(self):
        self.assertEqual(self.retrieve("logic", k=0), [])
        self.assertEqual(self.provider.embedded, [])

    # --- the exclusion rule (the point of this file) -------------------------

    def test_a_passage_of_an_excluded_topic_never_comes_back(self):
        excluded = Topic.objects.create(
            course=self.course, name="Set theory", excluded=True, position=1
        )
        hidden = a_chunk(
            self.file, "The perfect match, but not taught.", "logic",
            page=5, topic=excluded,
        )

        passages = self.retrieve("logic", min_score=0.0)

        self.assertNotIn(hidden.pk, [p.chunk_id for p in passages])

    def test_an_excluded_passage_does_not_even_take_a_top_k_slot(self):
        """Filtered before ranking, not after — or the result would be empty."""
        excluded = Topic.objects.create(
            course=self.course, name="Set theory", excluded=True, position=1
        )
        for i in range(3):
            a_chunk(
                self.file, f"Excluded but perfectly on topic {i}", "logic",
                page=5 + i, topic=excluded,
            )

        passages = self.retrieve("logic", k=3, min_score=0.0)

        self.assertIn(self.on_topic.pk, [p.chunk_id for p in passages])

    def test_excluding_a_topic_takes_its_passages_out_immediately(self):
        before = [p.chunk_id for p in self.retrieve("logic")]

        self.logic_topic.excluded = True
        self.logic_topic.save(update_fields=["excluded"])

        self.assertIn(self.on_topic.pk, before)
        self.assertEqual(self.retrieve("logic"), [])

    def test_excluding_a_chapter_also_takes_out_its_subtopics_passages(self):
        """"I do not teach this chapter" is not "except section 1.1".

        Found on the real Arabic course: the excluded chapter covered pages
        4-6, but a sub-topic claimed page 6 more narrowly, so M2 linked page 6's
        passages to the sub-topic — and they were still retrievable.
        """
        chapter = Topic.objects.create(
            course=self.course, name="Set theory", excluded=True, position=1
        )
        section = Topic.objects.create(
            course=self.course, name="Venn diagrams", parent=chapter, position=2
        )
        inherited = a_chunk(
            self.file, "Under a chapter that is not taught.", "logic",
            page=6, topic=section,
        )

        passages = self.retrieve("logic", min_score=0.0)

        self.assertFalse(section.excluded)  # the row itself was never touched
        self.assertNotIn(inherited.pk, [p.chunk_id for p in passages])

    def test_querying_the_excluded_topic_itself_returns_none_of_its_own_pages(self):
        self.logic_topic.excluded = True
        self.logic_topic.save(update_fields=["excluded"])

        passages = self.retrieve(self.logic_topic, min_score=0.0)

        self.assertNotIn(self.on_topic.pk, [p.chunk_id for p in passages])

    # --- failure modes -------------------------------------------------------

    def test_a_wrong_width_embedding_is_refused_not_ranked(self):
        class NarrowProvider(ConceptProvider):
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

        with self.assertRaises(RetrievalError):
            self.retrieve("logic", provider=NarrowProvider())

    def test_a_failing_provider_surfaces_as_a_retrieval_error(self):
        class BrokenProvider(ConceptProvider):
            def embed(self, texts):
                raise RuntimeError("rate limited")

        with self.assertRaises(RetrievalError) as caught:
            self.retrieve("logic", provider=BrokenProvider())
        self.assertIn("rate limited", str(caught.exception))

    def test_retrieval_never_calls_a_completion_model(self):
        # ConceptProvider.complete raises; a clean run proves it was untouched.
        self.retrieve("logic")


class RetrievalViewTests(TestCase):
    """The debug screen: the instructor's own course, and nobody else's."""

    def setUp(self):
        self.course, self.file = a_course_with_file()
        self.url = reverse("courses:retrieval", args=[self.course.pk])

        self.topic = Topic.objects.create(
            course=self.course, name="Propositional logic", key_terms=["truth table"],
            source_file=self.file, page_start=1, page_end=3,
        )
        self.chunk = a_chunk(
            self.file, "A truth table enumerates every assignment.", "logic",
            page=7, topic=self.topic,
        )
        self.excluded_topic = Topic.objects.create(
            course=self.course, name="Set theory", excluded=True, position=1
        )
        self.excluded_chunk = a_chunk(
            self.file, "Not taught, but a perfect match.", "logic",
            page=9, topic=self.excluded_topic,
        )

    def login(self):
        self.client.login(username=self.course.instructor.username, password=PASSWORD)

    def get(self, **params):
        with patch("agents.provider.get_provider", return_value=ConceptProvider()):
            return self.client.get(self.url, params)

    def test_logging_in_is_required(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response.url)

    def test_another_instructor_cannot_open_it(self):
        User.objects.create_user("intruder", password=PASSWORD)
        self.client.login(username="intruder", password=PASSWORD)

        self.assertEqual(self.client.get(self.url).status_code, 404)

    def test_it_opens_without_a_query(self):
        self.login()

        response = self.get()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["ran"])
        self.assertEqual(response.context["passages"], [])

    def test_a_free_text_query_renders_its_passages(self):
        self.login()

        response = self.get(q="truth table logic")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [p.chunk_id for p in response.context["passages"]], [self.chunk.pk]
        )
        self.assertContains(response, "A truth table enumerates")
        self.assertContains(response, "Propositional logic")

    def test_a_selected_topic_runs_the_topic_query(self):
        self.login()

        response = self.get(topic=self.topic.pk)

        self.assertEqual(response.context["selected_topic"], self.topic)
        self.assertEqual(
            response.context["query_text"], "Propositional logic — truth table"
        )
        self.assertEqual(
            [p.chunk_id for p in response.context["passages"]], [self.chunk.pk]
        )

    def test_the_excluded_topic_s_passage_is_never_rendered(self):
        self.login()

        response = self.get(q="logic")

        self.assertNotIn(
            self.excluded_chunk.pk, [p.chunk_id for p in response.context["passages"]]
        )
        self.assertNotContains(response, "Not taught, but a perfect match.")

    def test_the_screen_says_how_many_passages_are_out_of_the_index(self):
        self.login()

        response = self.get()

        self.assertEqual(response.context["chunk_count"], 2)
        self.assertEqual(response.context["usable_count"], 1)
        self.assertEqual(response.context["excluded_count"], 1)

    def test_the_page_number_reaches_the_screen(self):
        self.login()

        response = self.get(q="logic")

        self.assertEqual(response.context["passages"][0].page, 7)
        self.assertContains(response, "page")

    def test_a_query_that_matches_nothing_says_so_rather_than_padding(self):
        self.login()

        response = self.get(q="ethics and bias in deployment")

        self.assertTrue(response.context["ran"])
        self.assertEqual(response.context["passages"], [])
        self.assertContains(response, "Nothing scored above the floor")

    def test_a_provider_failure_is_reported_not_raised(self):
        self.login()

        class BrokenProvider(ConceptProvider):
            def embed(self, texts):
                raise RuntimeError("no credits")

        with patch("agents.provider.get_provider", return_value=BrokenProvider()):
            response = self.client.get(self.url, {"q": "logic"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("no credits", response.context["error"])
        self.assertEqual(response.context["passages"], [])
