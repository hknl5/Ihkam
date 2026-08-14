"""M2 success check, as code: topics extracted, then confirmed by a person.

The provider is faked throughout — the suite spends no API calls, the same
discipline M1 uses with `OCR_ENABLED=False`. What the real providers return on
the real files is checked by hand against ch10.3.pdf and the Arabic guidelines
file; what is locked in here is everything that must be true regardless of
which model answered.
"""

import json
import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from courses.models import Chunk, Course, ExtractedPage, SourceFile, Topic
from courses.services.ingest import ingest_source_file
from courses.services.topics import (
    TopicExtractionError,
    build_course_document,
    delete_topic,
    extract_topics,
    merge_topics,
)

from .test_courses import make_pdf, upload

PASSWORD = "quiet-precision-42"


#: What a well-behaved model returns for the three-page fixture PDF.
GOOD_ANSWER = {
    "chapters": [
        {
            "name": "Evaluation metrics",
            "source_file": "lecture.pdf",
            "page_start": 1,
            "page_end": 1,
            "key_terms": ["precision"],
            "definitions": [
                {"term": "Precision", "text": "the share of predicted positives that are correct"}
            ],
            "formulas": ["P = TP / (TP + FP)"],
            "examples": [],
            "subtopics": [
                {
                    "name": "Precision",
                    "source_file": "lecture.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "key_terms": ["true positive"],
                    "definitions": [],
                    "formulas": [],
                    "examples": [],
                }
            ],
        },
        {
            "name": "Confusion matrix",
            "source_file": "lecture.pdf",
            "page_start": 2,
            "page_end": 3,
            "key_terms": [],
            "definitions": [],
            "formulas": [],
            "examples": ["Given TP = 8 and FP = 2, precision is 0.8."],
            "subtopics": [],
        },
    ]
}


class FakeLLMProvider:
    """Stands in for OpenAI/Gemini. Records what it was asked."""

    name = "fake"

    def __init__(self, answers=None, dim=8):
        # A list, so a test can make the first answer bad and the second good.
        self.answers = list(answers) if answers is not None else [json.dumps(GOOD_ANSWER)]
        self.dim = dim
        self.prompts = []
        self.embedded = []

    def complete(self, system, user, *, json_mode=False, temperature=0.2):
        from agents.provider import LLMResponse

        self.prompts.append((system, user, json_mode))
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        return LLMResponse(text=answer)

    def embed(self, texts):
        self.embedded.extend(texts)
        # Deterministic and distinct per text, so similarity is meaningful.
        return [[(hash(t) % 100) / 100 + i * 0.0 for i in range(self.dim)] for t in texts]


def a_course(user=None, **kwargs):
    user = user or User.objects.create_user("nadia", password=PASSWORD)
    return Course.objects.create(
        instructor=user, name=kwargs.pop("name", "ML"), code=kwargs.pop("code", "CS310"), **kwargs
    )


def a_file(course, name="lecture.pdf", content=None):
    source_file = SourceFile.objects.create(
        course=course,
        file=SimpleUploadedFile(name, content or make_pdf()),
        original_name=name,
        kind=SourceFile.Kind.PDF,
    )
    ingest_source_file(source_file, run_ocr=False, run_chunking=False)
    return source_file


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class TopicExtractionTests(TestCase):
    """Extraction produces validated rows, or nothing at all."""

    def setUp(self):
        self.course = a_course()
        self.source_file = a_file(self.course)

    def test_a_valid_answer_becomes_chapters_and_subtopics(self):
        run = extract_topics(self.course, provider=FakeLLMProvider())

        self.assertEqual(run.chapters, 2)
        self.assertEqual(run.subtopics, 1)
        self.assertEqual(self.course.topics.count(), 3)

        metrics = self.course.topics.get(name="Evaluation metrics")
        self.assertIsNone(metrics.parent)
        self.assertEqual(metrics.source_file, self.source_file)
        self.assertEqual((metrics.page_start, metrics.page_end), (1, 1))
        self.assertEqual(metrics.formulas, ["P = TP / (TP + FP)"])
        self.assertEqual(metrics.definitions[0]["term"], "Precision")

        precision = self.course.topics.get(name="Precision")
        self.assertEqual(precision.parent, metrics)

    def test_the_material_reaches_the_prompt_with_its_page_numbers(self):
        provider = FakeLLMProvider()

        extract_topics(self.course, provider=provider)

        _, user, json_mode = provider.prompts[0]
        self.assertTrue(json_mode)  # §3: structured output, always
        self.assertIn("lecture.pdf", user)
        self.assertIn("[page 1]", user)
        self.assertIn("Evaluation metrics", user)
        self.assertIn("[page 3]", user)

    def test_a_malformed_answer_is_retried_once_then_accepted(self):
        provider = FakeLLMProvider(answers=["not json at all", json.dumps(GOOD_ANSWER)])

        run = extract_topics(self.course, provider=provider)

        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(run.chapters, 2)

    def test_two_bad_answers_store_nothing_and_say_so(self):
        provider = FakeLLMProvider(answers=["not json", '{"chapters": "not a list"}'])

        with self.assertRaises(TopicExtractionError) as caught:
            extract_topics(self.course, provider=provider)

        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(self.course.topics.count(), 0)
        self.assertIn("Nothing was stored", str(caught.exception))

    def test_a_call_that_never_reached_the_model_is_reported_as_itself(self):
        # A spent API credit is not a malformed answer. Calling it one sends
        # the instructor looking in the wrong place — and a second call would
        # fail identically, so it is not retried either.
        class BrokenProvider:
            calls = 0

            def complete(self, system, user, **kwargs):
                type(self).calls += 1
                raise RuntimeError("Error code: 429 — you have no credits remaining")

        with self.assertRaises(TopicExtractionError) as caught:
            extract_topics(self.course, provider=BrokenProvider())

        self.assertEqual(BrokenProvider.calls, 1)
        self.assertIn("did not complete", str(caught.exception))
        self.assertIn("no credits remaining", str(caught.exception))
        self.assertEqual(self.course.topics.count(), 0)

    def test_a_fenced_json_answer_is_still_read(self):
        provider = FakeLLMProvider(answers=[f"```json\n{json.dumps(GOOD_ANSWER)}\n```"])

        run = extract_topics(self.course, provider=provider)

        self.assertEqual(len(provider.prompts), 1)
        self.assertEqual(run.chapters, 2)

    def test_a_page_span_outside_the_material_is_dropped_not_stored(self):
        answer = {
            "chapters": [
                {"name": "Invented chapter", "source_file": "lecture.pdf",
                 "page_start": 40, "page_end": 44, "subtopics": []}
            ]
        }
        run = extract_topics(self.course, provider=FakeLLMProvider([json.dumps(answer)]))

        topic = self.course.topics.get()
        self.assertIsNone(topic.page_start)
        self.assertIsNone(topic.page_end)
        self.assertEqual(run.spans_dropped, 1)

    def test_a_span_reaching_past_the_material_keeps_only_the_real_part(self):
        answer = {
            "chapters": [
                {"name": "Metrics", "source_file": "lecture.pdf",
                 "page_start": 2, "page_end": 90, "subtopics": []}
            ]
        }
        extract_topics(self.course, provider=FakeLLMProvider([json.dumps(answer)]))

        topic = self.course.topics.get()
        self.assertEqual((topic.page_start, topic.page_end), (2, 3))

    def test_a_file_this_course_has_not_got_gets_no_citation(self):
        answer = {
            "chapters": [
                {"name": "Metrics", "source_file": "somebody-elses.pdf",
                 "page_start": 1, "page_end": 2, "subtopics": []}
            ]
        }
        extract_topics(self.course, provider=FakeLLMProvider([json.dumps(answer)]))

        topic = self.course.topics.get()
        self.assertIsNone(topic.source_file)
        self.assertIsNone(topic.page_start)

    def test_the_same_topic_twice_is_stored_once(self):
        answer = {
            "chapters": [
                {"name": "Evaluation metrics", "subtopics": []},
                {"name": "evaluation   METRICS", "subtopics": []},
            ]
        }
        run = extract_topics(self.course, provider=FakeLLMProvider([json.dumps(answer)]))

        self.assertEqual(run.chapters, 1)
        self.assertEqual(run.duplicates_dropped, 1)
        # The material's own wording survives, not the normalised form.
        self.assertEqual(self.course.topics.get().name, "Evaluation metrics")

    def test_re_extracting_replaces_the_list_rather_than_doubling_it(self):
        extract_topics(self.course, provider=FakeLLMProvider())
        extract_topics(self.course, provider=FakeLLMProvider())

        self.assertEqual(self.course.topics.count(), 3)

    def test_a_course_with_nothing_readable_refuses_to_guess(self):
        empty = a_course(user=self.course.instructor, name="Empty", code="CS999")

        with self.assertRaises(TopicExtractionError) as caught:
            extract_topics(empty, provider=FakeLLMProvider())

        self.assertIn("no readable text", str(caught.exception))


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class ContentSourceDisciplineTests(TestCase):
    """Only readable pages contribute, and an OCR page says that it is one."""

    def setUp(self):
        self.course = a_course()
        self.source_file = a_file(self.course)

    def _mark(self, number, **fields):
        page = self.source_file.pages.get(number=number)
        for key, value in fields.items():
            setattr(page, key, value)
        page.save()
        return page

    def test_an_unreadable_page_contributes_nothing_to_the_prompt(self):
        self._mark(2, is_image_only=True, ocr_reason=ExtractedPage.OCRReason.IMAGE_ONLY)

        document = build_course_document(self.course)

        self.assertNotIn("Confusion matrix", document.text)
        self.assertNotIn("[page 2]", document.text)
        self.assertEqual(document.pages_included, 2)
        self.assertEqual(document.pages_skipped, 1)

    def test_a_blank_page_contributes_nothing_either(self):
        self._mark(3, text="")

        document = build_course_document(self.course)

        self.assertEqual(document.pages_included, 2)
        self.assertEqual(document.pages_skipped, 1)

    def test_an_ocr_page_is_used_but_labelled_as_a_transcription(self):
        self._mark(2, source=ExtractedPage.Source.OCR,
                   ocr_reason=ExtractedPage.OCRReason.IMAGE_ONLY)

        document = build_course_document(self.course)

        self.assertIn("[page 2 (OCR transcription)]", document.text)
        self.assertIn("Confusion matrix", document.text)
        self.assertEqual(document.pages_from_ocr, 1)
        self.assertEqual(document.pages_included, 3)

    def test_the_prompt_tells_the_model_a_transcription_is_weaker_evidence(self):
        from agents.prompts.topics import SYSTEM

        self.assertIn("OCR transcription", SYSTEM)
        self.assertIn("transcription errors", SYSTEM)

    def test_a_partially_read_page_still_counts_as_readable(self):
        # Its text is real, only incomplete. Dropping it would lose the part
        # that *was* read; the reader marks it, and its chunks keep the page.
        page = self._mark(2, ocr_reason=ExtractedPage.OCRReason.MIXED)

        self.assertTrue(page.is_partially_unread)
        self.assertTrue(page.is_readable)
        self.assertIn("[page 2]", build_course_document(self.course).text)

    @override_settings(TOPIC_EXTRACTION_MAX_CHARS=200)
    def test_pages_that_do_not_fit_are_counted_not_silently_dropped(self):
        document = build_course_document(self.course)

        self.assertGreater(document.pages_over_budget, 0)
        self.assertEqual(
            document.pages_included + document.pages_over_budget, 3
        )


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class TopicEditTests(TestCase):
    """The instructor's edits — the point of the milestone — persist."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.client.login(username="nadia", password=PASSWORD)
        self.course = a_course(self.user)
        self.source_file = a_file(self.course)
        extract_topics(self.course, provider=FakeLLMProvider())
        self.url = reverse("courses:topics", args=[self.course.pk])
        self.metrics = self.course.topics.get(name="Evaluation metrics")
        self.matrix = self.course.topics.get(name="Confusion matrix")

    def test_the_screen_shows_what_was_extracted(self):
        response = self.client.get(self.url)

        self.assertContains(response, "Evaluation metrics")
        self.assertContains(response, "Confusion matrix")
        self.assertContains(response, "pages 2–3")

    def test_renaming_persists(self):
        self.client.post(
            reverse("courses:topic_rename", args=[self.course.pk, self.metrics.pk]),
            {"name": "  Evaluating a classifier  "},
        )
        self.metrics.refresh_from_db()

        self.assertEqual(self.metrics.name, "Evaluating a classifier")
        self.assertContains(self.client.get(self.url), "Evaluating a classifier")

    def test_a_blank_rename_changes_nothing(self):
        response = self.client.post(
            reverse("courses:topic_rename", args=[self.course.pk, self.metrics.pk]),
            {"name": "   "},
            follow=True,
        )
        self.metrics.refresh_from_db()

        self.assertEqual(self.metrics.name, "Evaluation metrics")
        self.assertContains(response, "needs a name")

    def test_adding_a_topic_by_hand_persists_without_a_page_reference(self):
        self.client.post(
            reverse("courses:topic_add", args=[self.course.pk]),
            {"name": "Cross-validation", "parent": "", "source_file": "",
             "page_start": "", "page_end": ""},
        )

        added = self.course.topics.get(name="Cross-validation")
        self.assertIsNone(added.source_file)
        self.assertIsNone(added.page_start)
        self.assertTrue(added.is_chapter)
        self.assertContains(self.client.get(self.url), "Cross-validation")

    def test_a_hand_added_topic_can_sit_under_a_chapter(self):
        self.client.post(
            reverse("courses:topic_add", args=[self.course.pk]),
            {"name": "Recall", "parent": self.metrics.pk, "source_file": "",
             "page_start": "", "page_end": ""},
        )

        self.assertEqual(self.course.topics.get(name="Recall").parent, self.metrics)

    def test_a_backwards_page_span_is_refused(self):
        response = self.client.post(
            reverse("courses:topic_add", args=[self.course.pk]),
            {"name": "Nonsense", "parent": "", "source_file": self.source_file.pk,
             "page_start": 3, "page_end": 1},
        )

        self.assertContains(response, "comes before the first page", status_code=400)
        self.assertFalse(self.course.topics.filter(name="Nonsense").exists())

    def test_a_page_beyond_the_file_is_refused(self):
        response = self.client.post(
            reverse("courses:topic_add", args=[self.course.pk]),
            {"name": "Page 90", "parent": "", "source_file": self.source_file.pk,
             "page_start": 90, "page_end": ""},
        )

        self.assertContains(response, "has only 3 pages", status_code=400)

    def test_deleting_a_topic_persists(self):
        self.client.post(
            reverse("courses:topic_delete", args=[self.course.pk, self.matrix.pk])
        )

        self.assertFalse(self.course.topics.filter(name="Confusion matrix").exists())

    def test_deleting_a_chapter_keeps_its_subtopics(self):
        # Removing a heading is not a request to lose what is filed under it.
        promoted = delete_topic(self.metrics)

        self.assertEqual(promoted, 1)
        precision = self.course.topics.get(name="Precision")
        self.assertIsNone(precision.parent)

    def test_merging_two_topics_persists_and_keeps_both_spans(self):
        self.client.post(
            reverse("courses:topics_merge", args=[self.course.pk]),
            {"topic": [self.metrics.pk, self.matrix.pk]},
        )

        self.assertFalse(self.course.topics.filter(pk=self.matrix.pk).exists())
        self.metrics.refresh_from_db()
        self.assertEqual((self.metrics.page_start, self.metrics.page_end), (1, 3))
        self.assertEqual(
            self.metrics.examples, ["Given TP = 8 and FP = 2, precision is 0.8."]
        )
        self.assertEqual(self.metrics.name, "Evaluation metrics")  # the survivor's

    def test_merging_needs_exactly_two(self):
        for selection in ([self.metrics.pk], [self.metrics.pk, self.matrix.pk,
                                              self.course.topics.get(name="Precision").pk]):
            response = self.client.post(
                reverse("courses:topics_merge", args=[self.course.pk]),
                {"topic": selection},
                follow=True,
            )
            self.assertContains(response, "exactly two")
        self.assertEqual(self.course.topics.count(), 3)

    def test_merging_a_chapter_into_another_moves_its_subtopics(self):
        merge_topics(self.matrix, self.metrics)

        self.matrix.refresh_from_db()
        self.assertEqual(
            list(self.matrix.subtopics.values_list("name", flat=True)), ["Precision"]
        )

    def test_excluding_a_topic_persists_and_can_be_undone(self):
        exclude_url = reverse("courses:topic_exclude", args=[self.course.pk, self.matrix.pk])

        self.client.post(exclude_url)
        self.matrix.refresh_from_db()
        self.assertTrue(self.matrix.excluded)

        self.client.post(exclude_url)
        self.matrix.refresh_from_db()
        self.assertFalse(self.matrix.excluded)

    def test_the_screen_names_an_excluded_topic_in_words_not_only_colour(self):
        self.client.post(
            reverse("courses:topic_exclude", args=[self.course.pk, self.matrix.pk])
        )

        response = self.client.get(self.url)
        self.assertContains(response, "Not taught in lectures")

    def test_extracting_again_will_not_quietly_discard_edits(self):
        self.client.post(
            reverse("courses:topic_rename", args=[self.course.pk, self.metrics.pk]),
            {"name": "Mine now"},
        )

        response = self.client.post(
            reverse("courses:topics_extract", args=[self.course.pk]), follow=True
        )

        self.assertContains(response, "already has topics")
        self.assertTrue(self.course.topics.filter(name="Mine now").exists())

    def test_every_edit_needs_a_post(self):
        for name, args in (
            ("courses:topics_extract", [self.course.pk]),
            ("courses:topic_add", [self.course.pk]),
            ("courses:topics_merge", [self.course.pk]),
            ("courses:topic_rename", [self.course.pk, self.metrics.pk]),
            ("courses:topic_delete", [self.course.pk, self.metrics.pk]),
            ("courses:topic_exclude", [self.course.pk, self.metrics.pk]),
        ):
            self.assertEqual(self.client.get(reverse(name, args=args)).status_code, 405)

    def test_another_instructor_cannot_see_or_edit_these_topics(self):
        User.objects.create_user("omar", password=PASSWORD)
        self.client.login(username="omar", password=PASSWORD)

        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(
            self.client.post(
                reverse("courses:topic_rename", args=[self.course.pk, self.metrics.pk]),
                {"name": "hijacked"},
            ).status_code,
            404,
        )
        self.metrics.refresh_from_db()
        self.assertEqual(self.metrics.name, "Evaluation metrics")

    def test_the_screen_invites_a_first_extraction_when_there_are_none(self):
        self.course.topics.all().delete()

        response = self.client.get(self.url)
        self.assertContains(response, "No topics yet")
        self.assertContains(response, "Extract topics")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class ExcludedTopicsTests(TestCase):
    """An excluded topic must never flow downstream. This is the hard one."""

    def setUp(self):
        self.course = a_course()
        self.source_file = a_file(self.course)
        extract_topics(self.course, provider=FakeLLMProvider())
        self.matrix = self.course.topics.get(name="Confusion matrix")

    def test_included_leaves_excluded_topics_out(self):
        self.matrix.excluded = True
        self.matrix.save()

        names = set(self.course.topics.included().values_list("name", flat=True))

        self.assertNotIn("Confusion matrix", names)
        self.assertEqual(names, {"Evaluation metrics", "Precision"})

    def test_chunks_of_an_excluded_topic_are_not_usable(self):
        from courses.services.chunking import chunk_source_file, link_chunks_to_topics

        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))
        link_chunks_to_topics(self.course)
        self.assertTrue(Chunk.objects.filter(topic=self.matrix).exists())

        self.matrix.excluded = True
        self.matrix.save()

        usable = Chunk.objects.usable()
        self.assertFalse(usable.filter(topic=self.matrix).exists())
        self.assertTrue(usable.exists())  # the rest of the course still is

    def test_a_chunk_with_no_topic_stays_usable(self):
        from courses.services.chunking import chunk_source_file

        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertEqual(Chunk.objects.filter(topic__isnull=True).count(),
                         Chunk.objects.usable().filter(topic__isnull=True).count())


class OutOfIndexSubtopicTests(TestCase):
    """A sub-topic under an excluded chapter stays visible, and says what it is.

    The instructor's decision was about the chapter, so the sub-topic is not
    hidden or deleted — but it inherits the exclusion (`ChunkQuerySet.usable`),
    and a row that reads "in the syllabus" while contributing to nothing is the
    exact confusion this marker was added to prevent.
    """

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = a_course(self.user)
        # The review screen only lists topics once the course has material.
        SourceFile.objects.create(
            course=self.course, original_name="lecture.pdf", kind=SourceFile.Kind.PDF
        )
        self.chapter = Topic.objects.create(course=self.course, name="Chapter 3", position=0)
        self.subtopic = Topic.objects.create(
            course=self.course, name="3.1 Precision", parent=self.chapter, position=1
        )
        self.client.login(username="nadia", password=PASSWORD)

    def _topics_page(self):
        return self.client.get(reverse("courses:topics", args=[self.course.pk]))

    def test_an_ordinary_subtopic_reads_as_in_the_syllabus(self):
        response = self._topics_page()

        self.assertContains(response, "3.1 Precision")
        self.assertNotContains(response, "out of index")
        self.assertContains(response, "In the syllabus")

    def test_a_subtopic_of_an_excluded_chapter_is_still_listed(self):
        self.chapter.excluded = True
        self.chapter.save()

        response = self._topics_page()

        self.assertContains(response, "3.1 Precision")

    def test_it_is_marked_out_of_index_rather_than_reading_as_ordinary(self):
        self.chapter.excluded = True
        self.chapter.save()

        response = self._topics_page()

        html = response.content.decode()
        self.assertIn("Out of index — parent chapter excluded", html)
        self.assertNotIn("In the syllabus", html)

    def test_putting_the_chapter_back_clears_the_marker(self):
        self.chapter.excluded = True
        self.chapter.save()

        self.client.post(
            reverse("courses:topic_exclude", args=[self.course.pk, self.chapter.pk])
        )

        html = self._topics_page().content.decode()
        self.assertNotIn("Out of index", html)
        self.assertIn("In the syllabus", html)
