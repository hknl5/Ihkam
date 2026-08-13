"""M2: passages and embeddings — the foundation retrieval stands on in M3.

The provider is faked, so the suite spends no API calls. What is locked in here
is the shape of what gets stored: the right dimension, the right page, the
right provenance, and nothing at all from a page that could not be read.
"""

import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from courses.models import Chunk, Course, ExtractedPage, SourceFile
from courses.services.chunking import (
    ChunkingError,
    chunk_source_file,
    embed_texts,
    link_chunks_to_topics,
    split_page,
)
from courses.services.ingest import ingest_source_file
from courses.services.topics import extract_topics

from .test_courses import make_pdf
from .test_topics import FakeLLMProvider, a_course

PASSWORD = "quiet-precision-42"


class SplittingTests(TestCase):
    """Pure text work: no page is ever silently dropped or merged across."""

    def test_a_short_page_is_one_passage(self):
        self.assertEqual(split_page("A single short slide."), ["A single short slide."])

    def test_an_empty_page_produces_nothing(self):
        self.assertEqual(split_page(""), [])
        self.assertEqual(split_page("   \n\n  "), [])

    def test_paragraphs_are_packed_up_to_the_maximum(self):
        page = "\n\n".join(["x" * 300] * 6)

        passages = split_page(page, max_chars=700, min_chars=100)

        self.assertGreater(len(passages), 1)
        self.assertTrue(all(len(p) <= 700 for p in passages))

    def test_a_heading_is_not_left_as_a_passage_of_its_own(self):
        # A one-line passage embeds to something that only ever matches itself.
        page = "Chapter 3\n\n" + "y" * 400

        passages = split_page(page, max_chars=1200, min_chars=200)

        self.assertEqual(len(passages), 1)
        self.assertTrue(passages[0].startswith("Chapter 3"))

    def test_one_enormous_paragraph_is_split_not_truncated(self):
        page = "z" * 3000

        passages = split_page(page, max_chars=1000, min_chars=100)

        self.assertEqual("".join(passages), page)
        self.assertTrue(all(len(p) <= 1000 for p in passages))

    def test_arabic_sentence_ends_are_boundaries_too(self):
        sentence = "هذه جملة عربية طويلة جدا تحتوي على كلمات كثيرة. "
        passages = split_page(sentence * 40, max_chars=400, min_chars=100)

        self.assertGreater(len(passages), 1)
        self.assertTrue(all(len(p) <= 400 for p in passages))

    def test_nothing_is_lost_between_passages(self):
        page = "\n\n".join(f"Paragraph number {i} with some real words in it." for i in range(20))

        rejoined = " ".join(split_page(page, max_chars=200, min_chars=50))

        for i in range(20):
            self.assertIn(f"Paragraph number {i}", rejoined)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class ChunkBuildingTests(TestCase):
    def setUp(self):
        self.course = a_course()
        self.source_file = SourceFile.objects.create(
            course=self.course,
            file=SimpleUploadedFile("lecture.pdf", make_pdf()),
            original_name="lecture.pdf",
            kind=SourceFile.Kind.PDF,
        )
        ingest_source_file(self.source_file, run_ocr=False, run_chunking=False)

    def test_every_readable_page_gets_a_chunk_at_the_configured_dimension(self):
        run = chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertTrue(run.ok)
        self.assertEqual(run.pages_chunked, 3)
        chunks = list(self.source_file.chunks.all())
        self.assertEqual(len(chunks), run.chunks)
        self.assertEqual(sorted({c.page for c in chunks}), [1, 2, 3])
        for chunk in chunks:
            self.assertEqual(len(chunk.embedding), 1536)

    def test_a_chunk_carries_the_page_it_came_from(self):
        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        page_two = self.source_file.chunks.filter(page=2).first()
        self.assertIn("Confusion matrix", page_two.text)
        self.assertNotIn("Confusion matrix", self.source_file.chunks.get(page=1).text)

    def test_provenance_records_a_text_layer_and_an_ocr_page_separately(self):
        page = self.source_file.pages.get(number=2)
        page.source = ExtractedPage.Source.OCR
        page.ocr_reason = ExtractedPage.OCRReason.IMAGE_ONLY
        page.save()

        run = chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertEqual(
            self.source_file.chunks.get(page=2).source, ExtractedPage.Source.OCR
        )
        self.assertEqual(
            self.source_file.chunks.get(page=1).source, ExtractedPage.Source.TEXT_LAYER
        )
        self.assertEqual(run.by_source[ExtractedPage.Source.OCR], 1)
        self.assertTrue(self.source_file.chunks.get(page=2).is_from_ocr)

    def test_an_unreadable_page_is_not_indexed(self):
        page = self.source_file.pages.get(number=3)
        page.is_image_only = True
        page.ocr_reason = ExtractedPage.OCRReason.IMAGE_ONLY
        page.save()

        run = chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertEqual(run.pages_skipped, 1)
        self.assertFalse(self.source_file.chunks.filter(page=3).exists())

    def test_rebuilding_replaces_chunks_rather_than_adding_more(self):
        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))
        first = self.source_file.chunks.count()

        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertEqual(self.source_file.chunks.count(), first)

    def test_a_wrong_width_is_refused_and_nothing_is_stored(self):
        # A provider quietly returning its native size would make every stored
        # vector incomparable, and the damage would only show up in M3.
        run = chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=64))

        self.assertFalse(run.ok)
        self.assertIn("64-dimension", run.error)
        self.assertEqual(self.source_file.chunks.count(), 0)

    def test_embed_texts_refuses_a_short_answer(self):
        class Miscounting:
            def embed(self, texts):
                return [[0.0] * 1536]

        with self.assertRaises(ChunkingError):
            embed_texts(["one", "two"], provider=Miscounting())

    def test_embedding_nothing_costs_no_call(self):
        provider = FakeLLMProvider(dim=1536)
        self.assertEqual(embed_texts([], provider=provider), [])
        self.assertEqual(provider.embedded, [])

    def test_a_file_with_no_readable_pages_stores_no_chunks(self):
        self.source_file.pages.update(is_image_only=True)

        run = chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))

        self.assertEqual(run.chunks, 0)
        self.assertEqual(self.source_file.chunks.count(), 0)

    def test_chunking_runs_at_the_end_of_ingest_when_it_is_enabled(self):
        provider = FakeLLMProvider(dim=1536)

        with override_settings(EMBEDDINGS_ENABLED=True):
            ingest_source_file(
                self.source_file, run_ocr=False, embed_provider=provider
            )

        self.assertGreater(self.source_file.chunks.count(), 0)
        self.assertGreater(len(provider.embedded), 0)

    def test_a_failing_embedding_provider_does_not_fail_the_upload(self):
        # The file is extracted and readable either way; build_chunks retries.
        with override_settings(EMBEDDINGS_ENABLED=True):
            ingest_source_file(
                self.source_file, run_ocr=False, embed_provider=FakeLLMProvider(dim=8)
            )

        self.assertEqual(self.source_file.status, SourceFile.Status.READY)
        self.assertEqual(self.source_file.chunks.count(), 0)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False, EMBEDDINGS_ENABLED=False)
class ChunkTopicLinkTests(TestCase):
    """Chunks are attached to topics by page span — no model involved."""

    def setUp(self):
        self.course = a_course()
        self.source_file = SourceFile.objects.create(
            course=self.course,
            file=SimpleUploadedFile("lecture.pdf", make_pdf()),
            original_name="lecture.pdf",
            kind=SourceFile.Kind.PDF,
        )
        ingest_source_file(self.source_file, run_ocr=False, run_chunking=False)
        chunk_source_file(self.source_file, provider=FakeLLMProvider(dim=1536))
        extract_topics(self.course, provider=FakeLLMProvider())

    def test_a_chunk_is_attached_to_the_topic_whose_pages_cover_it(self):
        link_chunks_to_topics(self.course)

        page_two = self.source_file.chunks.get(page=2)
        self.assertEqual(page_two.topic.name, "Confusion matrix")

    def test_the_narrower_span_wins_over_its_chapter(self):
        # "Precision" claims page 1 alone; its chapter claims page 1 too. The
        # more specific claim is the more informative one.
        link_chunks_to_topics(self.course)

        self.assertEqual(self.source_file.chunks.get(page=1).topic.name, "Precision")

    def test_a_page_no_topic_claims_keeps_no_topic(self):
        self.course.topics.all().delete()

        link_chunks_to_topics(self.course)

        self.assertFalse(self.source_file.chunks.exclude(topic__isnull=True).exists())

    def test_merging_moves_the_absorbed_topic_s_chunks(self):
        from courses.services.topics import merge_topics

        link_chunks_to_topics(self.course)
        metrics = self.course.topics.get(name="Evaluation metrics")
        matrix = self.course.topics.get(name="Confusion matrix")

        merge_topics(metrics, matrix)

        self.assertFalse(Chunk.objects.filter(topic_id=matrix.pk).exists())
        self.assertEqual(self.source_file.chunks.get(page=2).topic, metrics)

    def test_deleting_a_topic_leaves_its_chunks_in_place(self):
        link_chunks_to_topics(self.course)
        matrix = self.course.topics.get(name="Confusion matrix")

        matrix.delete()

        chunk = self.source_file.chunks.get(page=2)
        self.assertIsNone(chunk.topic)
        self.assertIn("Confusion matrix", chunk.text)
