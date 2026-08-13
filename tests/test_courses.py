"""M1 success check, as code: create a course, upload a PDF, read the text.

Sample PDFs are generated here rather than committed, so the fixtures stay
readable and the expected text lives next to the assertion.
"""

import io
import tempfile
import threading

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from courses.forms import MAX_UPLOAD_BYTES
from courses.models import Course, ExtractedPage, SourceFile
from agents.ocr import OCRResult
from courses.services.ingest import (
    UnsupportedFormatError,
    extract_pages,
    extract_pdf,
    ingest_source_file,
    normalize_whitespace,
    repair_lam_alef,
)

PAGE_LINES = [
    ["Chapter 1 — Evaluation metrics", "Precision is the share of predicted", "positives that are correct."],
    ["Chapter 2 — Confusion matrix", "A confusion matrix counts true and", "false positives and negatives."],
    ["Chapter 3 — Worked example", "Given TP = 8 and FP = 2, precision", "is 0.8."],
]


def make_pdf(pages=PAGE_LINES) -> bytes:
    """A small text-based PDF: one page per entry, each line drawn as text."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 24
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_blank_pdf(page_count=2) -> bytes:
    """Stands in for a scanned document: real pages, no text layer."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for _ in range(page_count):
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def upload(name="lecture.pdf", content=None) -> SimpleUploadedFile:
    return SimpleUploadedFile(
        name, content if content is not None else make_pdf(), content_type="application/pdf"
    )


class CourseModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")

    def test_course_belongs_to_its_instructor(self):
        course = Course.objects.create(
            instructor=self.user,
            name="Introduction to Machine Learning",
            code="CS310",
            level=Course.Level.INTERMEDIATE,
            content_language=Course.ContentLanguage.MIXED,
        )
        self.assertEqual(course.instructor, self.user)
        self.assertEqual(list(self.user.courses.all()), [course])
        self.assertEqual(str(course), "CS310 — Introduction to Machine Learning")
        self.assertEqual(course.get_absolute_url(), f"/courses/{course.pk}/")

    def test_code_is_unique_per_instructor_but_not_globally(self):
        Course.objects.create(instructor=self.user, name="ML", code="CS310")
        other = User.objects.create_user("omar", password="quiet-precision-42")
        # Another instructor may use the same code.
        Course.objects.create(instructor=other, name="ML", code="CS310")

        form_page = self.client
        form_page.login(username="nadia", password="quiet-precision-42")
        response = form_page.post(
            reverse("courses:dashboard"),
            {"name": "ML again", "code": "cs310", "level": "introductory", "content_language": "en"},
        )
        self.assertContains(response, "You already have a course with this code.")
        self.assertEqual(Course.objects.filter(instructor=self.user).count(), 1)

    def test_source_file_kind_is_detected_from_the_extension(self):
        self.assertEqual(SourceFile.kind_for_filename("week1.PDF"), SourceFile.Kind.PDF)
        self.assertEqual(SourceFile.kind_for_filename("deck.pptx"), SourceFile.Kind.POWERPOINT)
        self.assertEqual(SourceFile.kind_for_filename("notes.docx"), SourceFile.Kind.WORD)
        self.assertIsNone(SourceFile.kind_for_filename("archive.zip"))


class ExtractionTests(TestCase):
    """The M1 success check: legible text, complete, page numbers preserved."""

    def test_pdf_extraction_returns_one_page_per_page_in_order(self):
        pages = extract_pdf(io.BytesIO(make_pdf()))

        self.assertEqual(len(pages), 3)
        self.assertEqual([p.number for p in pages], [1, 2, 3])
        self.assertTrue(all(p.text.strip() for p in pages))

    def test_extracted_text_is_legible_and_complete(self):
        pages = extract_pdf(io.BytesIO(make_pdf()))

        self.assertIn("Evaluation metrics", pages[0].text)
        self.assertIn("Precision is the share of predicted", pages[0].text)
        self.assertIn("Confusion matrix", pages[1].text)
        # Content stays on its own page — this is what makes a citation possible.
        self.assertNotIn("Confusion matrix", pages[0].text)
        self.assertIn("TP = 8", pages[2].text)

    def test_normalize_whitespace_tidies_without_changing_the_words(self):
        messy = "Precision   is\t\tthe   share \r\n\n\n\nof positives.  \n"
        self.assertEqual(
            normalize_whitespace(messy), "Precision is the share\n\nof positives."
        )
        self.assertEqual(normalize_whitespace(""), "")

    def test_unimplemented_formats_raise_the_shared_stub_error(self):
        for kind in (SourceFile.Kind.POWERPOINT, SourceFile.Kind.WORD, SourceFile.Kind.TEXT):
            with self.assertRaises(UnsupportedFormatError):
                extract_pages(io.BytesIO(b"anything"), kind)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False)
class UploadFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")
        self.client.login(username="nadia", password="quiet-precision-42")
        self.course = Course.objects.create(instructor=self.user, name="ML", code="CS310")
        self.url = reverse("courses:detail", args=[self.course.pk])

    def test_upload_extracts_and_lands_on_the_file(self):
        response = self.client.post(self.url, {"file": upload()}, follow=True)

        source_file = SourceFile.objects.get()
        self.assertRedirects(response, source_file.get_absolute_url())
        self.assertEqual(source_file.course, self.course)
        self.assertEqual(source_file.kind, SourceFile.Kind.PDF)
        self.assertEqual(source_file.original_name, "lecture.pdf")
        self.assertEqual(source_file.status, SourceFile.Status.READY)
        self.assertEqual(source_file.page_count, 3)
        self.assertGreater(source_file.size_bytes, 0)

        pages = list(source_file.pages.all())
        self.assertEqual([p.number for p in pages], [1, 2, 3])
        self.assertIn("Evaluation metrics", pages[0].text)

    def test_pages_are_read_one_at_a_time_with_their_number_shown(self):
        self.client.post(self.url, {"file": upload()})
        source_file = SourceFile.objects.get()

        first = self.client.get(source_file.get_absolute_url())
        self.assertContains(first, "Evaluation metrics")
        self.assertNotContains(first, "Confusion matrix")

        second = self.client.get(source_file.get_absolute_url(), {"page": 2})
        self.assertContains(second, "Confusion matrix")
        self.assertContains(second, "Page")
        self.assertNotContains(second, "Evaluation metrics")

    def test_re_ingesting_replaces_pages_rather_than_duplicating_them(self):
        self.client.post(self.url, {"file": upload()})
        source_file = SourceFile.objects.get()

        ingest_source_file(source_file)

        self.assertEqual(ExtractedPage.objects.filter(source_file=source_file).count(), 3)

    def test_scanned_pdf_is_reported_not_crashed_on(self):
        response = self.client.post(
            self.url, {"file": upload("scan.pdf", make_blank_pdf(2))}, follow=True
        )
        source_file = SourceFile.objects.get()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(source_file.status, SourceFile.Status.NO_TEXT)
        self.assertEqual(source_file.page_count, 2)  # page numbers survive regardless
        self.assertIn("scanned", source_file.status_detail.lower())
        self.assertContains(response, "No text layer")

    def test_a_file_that_is_not_a_pdf_fails_without_taking_the_page_down(self):
        # pypdf prints its own complaint to stderr here — that noise is expected.
        response = self.client.post(
            self.url, {"file": upload("broken.pdf", b"this is not a PDF at all")}, follow=True
        )
        source_file = SourceFile.objects.get()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(source_file.status, SourceFile.Status.FAILED)
        self.assertEqual(source_file.pages.count(), 0)

    def test_powerpoint_uploads_are_stored_but_marked_unsupported(self):
        self.client.post(self.url, {"file": SimpleUploadedFile("deck.pptx", b"stub")})
        source_file = SourceFile.objects.get()

        self.assertEqual(source_file.kind, SourceFile.Kind.POWERPOINT)
        self.assertEqual(source_file.status, SourceFile.Status.UNSUPPORTED)
        self.assertIn("not extracted yet", source_file.status_detail)

    def test_unknown_extensions_are_rejected_at_the_form(self):
        response = self.client.post(self.url, {"file": SimpleUploadedFile("notes.zip", b"stub")})

        self.assertContains(response, "Unsupported file type")
        self.assertEqual(SourceFile.objects.count(), 0)

    def test_oversized_uploads_are_rejected(self):
        big = SimpleUploadedFile("huge.pdf", b"0" * (MAX_UPLOAD_BYTES + 1))
        response = self.client.post(self.url, {"file": big})

        self.assertContains(response, "larger than")
        self.assertEqual(SourceFile.objects.count(), 0)

    def test_another_instructor_cannot_reach_the_course_or_its_files(self):
        self.client.post(self.url, {"file": upload()})
        source_file = SourceFile.objects.get()

        User.objects.create_user("omar", password="quiet-precision-42")
        self.client.login(username="omar", password="quiet-precision-42")

        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.client.get(source_file.get_absolute_url()).status_code, 404)


class DashboardTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")
        self.client.login(username="nadia", password="quiet-precision-42")

    def test_dashboard_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("courses:dashboard"))
        self.assertRedirects(
            response, f"{reverse('accounts:login')}?next={reverse('courses:dashboard')}"
        )

    def test_creating_a_course_lands_on_it(self):
        response = self.client.post(
            reverse("courses:dashboard"),
            {
                "name": "Introduction to Machine Learning",
                "code": "cs310",
                "level": "intermediate",
                "content_language": "ar",
            },
            follow=True,
        )
        course = Course.objects.get()

        self.assertRedirects(response, course.get_absolute_url())
        self.assertEqual(course.instructor, self.user)
        self.assertEqual(course.code, "CS310")  # normalised on the way in
        self.assertContains(response, "Introduction to Machine Learning")

    def test_dashboard_lists_only_your_own_courses(self):
        Course.objects.create(instructor=self.user, name="Mine", code="CS310")
        other = User.objects.create_user("omar", password="quiet-precision-42")
        Course.objects.create(instructor=other, name="Theirs", code="CS999")

        response = self.client.get(reverse("courses:dashboard"))
        self.assertContains(response, "Mine")
        self.assertNotContains(response, "Theirs")

    def test_empty_state_invites_the_first_course(self):
        response = self.client.get(reverse("courses:dashboard"))
        self.assertContains(response, "No courses yet")
        self.assertContains(response, "Create a course")


class ArabicLigatureTests(TestCase):
    """The lam-alef repair, from the real GenAI guidelines PDF.

    PDFium reports the single lam-alef glyph as its two letters in visual
    order, both with the same character box. Without the repair, every
    "الاصطناعي" in that document read "االصطناعي".
    """

    def test_a_ligature_pair_is_put_back_into_logical_order(self):
        box = (427.2, 175.9, 432.3, 194.7)
        chars = [["ا", box], ["ل", box], ["ص", (417.2, 175.9, 426.2, 194.7)]]

        repaired = repair_lam_alef(chars)

        self.assertEqual(repaired, 1)
        self.assertEqual("".join(c for c, _ in chars), "لاص")

    def test_the_definite_article_is_left_alone(self):
        # Two real glyphs, two different boxes — this is "ال", not a ligature.
        chars = [
            ["ا", (433.6, 175.9, 434.5, 194.7)],
            ["ل", (427.2, 175.9, 432.3, 194.7)],
            ["ذ", (420.0, 175.9, 426.0, 194.7)],
        ]

        repaired = repair_lam_alef(chars)

        self.assertEqual(repaired, 0)
        self.assertEqual("".join(c for c, _ in chars), "الذ")

    def test_characters_without_boxes_are_skipped(self):
        chars = [["ا", None], ["ل", None], ["\n", None]]

        self.assertEqual(repair_lam_alef(chars), 0)

    def test_alef_variants_are_repaired_too(self):
        box = (10.0, 10.0, 20.0, 20.0)
        for alef in ("ا", "أ", "إ", "آ"):
            chars = [[alef, box], ["ل", box]]
            self.assertEqual(repair_lam_alef(chars), 1)
            self.assertEqual("".join(c for c, _ in chars), f"ل{alef}")


def make_image_pdf(text_pages=1, image_pages=2) -> bytes:
    """A deck like the real ch10.3.pdf: some real slides, some that are
    screenshots with nothing but the slide number as text."""
    from reportlab.lib.utils import ImageReader

    picture = ImageReader(io.BytesIO(_png_bytes()))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for i in range(text_pages):
        pdf.drawString(72, 800, f"{i + 1}")
        pdf.drawString(72, 760, "Logical Gates and Combinatorial Circuits")
        pdf.drawString(72, 730, "In circuitry theory, NOT, AND and OR gates are the basic gates.")
        pdf.showPage()
    for i in range(image_pages):
        pdf.drawString(72, 800, f"{text_pages + i + 1}")  # only the slide number
        pdf.drawImage(picture, 72, 300, width=400, height=300)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 200, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=False)
class ImageOnlyPageTests(TestCase):
    """The reported bug: pages whose content is a screenshot came back as the
    slide number and were reported as fully extracted."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")
        self.client.login(username="nadia", password="quiet-precision-42")
        self.course = Course.objects.create(instructor=self.user, name="Discrete Maths", code="CS210")
        self.url = reverse("courses:detail", args=[self.course.pk])

    def test_a_screenshot_page_is_flagged_not_passed_off_as_extracted(self):
        self.client.post(self.url, {"file": upload("deck.pdf", make_image_pdf())})
        source_file = SourceFile.objects.get()

        self.assertEqual(source_file.status, SourceFile.Status.PARTIAL_TEXT)
        self.assertEqual(source_file.page_count, 3)
        self.assertEqual(source_file.pages_without_text, 2)

        pages = list(source_file.pages.all())
        self.assertFalse(pages[0].is_image_only)  # a real slide
        self.assertTrue(pages[1].is_image_only)
        self.assertTrue(pages[2].is_image_only)

    def test_the_reader_says_the_page_needs_ocr(self):
        self.client.post(self.url, {"file": upload("deck.pdf", make_image_pdf())})
        source_file = SourceFile.objects.get()

        response = self.client.get(source_file.get_absolute_url(), {"page": 2})

        self.assertContains(response, "no text layer")
        self.assertContains(response, "OCR")

    def test_pages_with_real_text_and_pictures_are_not_flagged(self):
        self.client.post(
            self.url, {"file": upload("all-good.pdf", make_image_pdf(text_pages=2, image_pages=0))}
        )
        source_file = SourceFile.objects.get()

        self.assertEqual(source_file.status, SourceFile.Status.READY)
        self.assertEqual(source_file.pages_without_text, 0)

    def test_the_upload_message_names_what_was_missed(self):
        response = self.client.post(
            self.url, {"file": upload("deck.pdf", make_image_pdf())}, follow=True
        )
        self.assertContains(response, "1 of 3 pages extracted")


class FakeOCRProvider:
    """Stands in for Gemini. Records what it was asked, one call per page."""

    name = "fake"
    model = "fake-vision"

    def __init__(self, text="Transcribed slide body.", fail_on=()):
        self.text = text
        self.fail_on = fail_on
        self.calls = []
        self._lock = threading.Lock()

    def ocr_page(self, image, *, mime_type="image/png", language_hint=""):
        with self._lock:
            self.calls.append((len(image), language_hint))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("provider exploded")
        if self.text is None:
            return OCRResult(text="", is_empty=True, provider=self.name, model=self.model)
        return OCRResult(text=self.text, provider=self.name, model=self.model)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=True)
class OCRIntegrationTests(TestCase):
    """OCR runs inline on upload and fills the pages extraction could not read."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")
        self.course = Course.objects.create(instructor=self.user, name="Discrete Maths", code="CS210")

    def _upload(self, provider, **kwargs):
        source_file = SourceFile.objects.create(
            course=self.course,
            file=SimpleUploadedFile("deck.pdf", make_image_pdf(text_pages=1, image_pages=3)),
            original_name="deck.pdf",
            kind=SourceFile.Kind.PDF,
        )
        return ingest_source_file(source_file, ocr_provider=provider, **kwargs)

    def test_image_only_pages_are_read_and_marked_as_ocr(self):
        provider = FakeOCRProvider()

        source_file = self._upload(provider)

        self.assertEqual(len(provider.calls), 3)  # one call per image-only page
        self.assertEqual(source_file.pages_from_ocr, 3)
        self.assertEqual(source_file.pages_without_text, 0)
        self.assertEqual(source_file.status, SourceFile.Status.READY)
        self.assertEqual(source_file.ocr_engine, "fake/fake-vision")

        pages = list(source_file.pages.all())
        self.assertEqual(pages[0].source, ExtractedPage.Source.TEXT_LAYER)
        for page in pages[1:]:
            self.assertEqual(page.source, ExtractedPage.Source.OCR)
            self.assertIn("Transcribed slide body.", page.text)
            self.assertFalse(page.is_image_only)

    def test_one_image_per_call_never_a_batch(self):
        provider = FakeOCRProvider()

        self._upload(provider)

        # Three separate calls, each carrying exactly one rendered page.
        self.assertEqual(len(provider.calls), 3)
        self.assertTrue(all(size > 0 for size, _ in provider.calls))

    def test_the_course_content_language_is_passed_as_a_hint(self):
        self.course.content_language = Course.ContentLanguage.ARABIC
        self.course.save()
        provider = FakeOCRProvider()

        self._upload(provider)

        self.assertEqual({hint for _, hint in provider.calls}, {"ar"})

    def test_an_empty_transcription_leaves_the_page_flagged(self):
        # Never store junk: a page the model could not read stays unreadable.
        provider = FakeOCRProvider(text=None)

        source_file = self._upload(provider)

        self.assertEqual(source_file.pages_from_ocr, 0)
        self.assertEqual(source_file.pages_without_text, 3)
        self.assertEqual(source_file.status, SourceFile.Status.PARTIAL_TEXT)
        self.assertEqual(source_file.ocr_engine, "")
        self.assertTrue(all(p.is_image_only for p in source_file.pages.all()[1:]))

    def test_one_failing_page_does_not_lose_the_others(self):
        provider = FakeOCRProvider(fail_on=(2,))

        source_file = self._upload(provider)

        self.assertEqual(source_file.pages_from_ocr, 2)
        self.assertEqual(source_file.pages_without_text, 1)

    @override_settings(OCR_MAX_PAGES_PER_FILE=2)
    def test_the_cap_reads_what_it_can_and_says_what_it_skipped(self):
        provider = FakeOCRProvider()

        source_file = self._upload(provider)

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(source_file.pages_from_ocr, 2)
        self.assertEqual(source_file.pages_without_text, 1)  # left flagged, not lost
        self.assertEqual(source_file.status, SourceFile.Status.PARTIAL_TEXT)
        self.assertIn("over the 2-page OCR limit", source_file.status_detail)

    @override_settings(OCR_ENABLED=False)
    def test_ocr_can_be_turned_off_entirely(self):
        provider = FakeOCRProvider()

        source_file = self._upload(provider, run_ocr=False)

        self.assertEqual(provider.calls, [])
        self.assertEqual(source_file.pages_without_text, 3)

    def test_the_reader_marks_an_ocr_page_as_a_transcription(self):
        source_file = self._upload(FakeOCRProvider())
        self.client.login(username="nadia", password="quiet-precision-42")

        response = self.client.get(source_file.get_absolute_url(), {"page": 2})

        self.assertContains(response, "Read by OCR")
        self.assertContains(response, "model transcription")
        self.assertContains(response, "Transcribed slide body.")

    def test_re_ingesting_does_not_leave_stale_ocr_provenance(self):
        source_file = self._upload(FakeOCRProvider())
        self.assertEqual(source_file.pages_from_ocr, 3)

        ingest_source_file(source_file, run_ocr=False)

        self.assertEqual(source_file.pages_from_ocr, 0)
        self.assertEqual(source_file.ocr_engine, "")
        self.assertFalse(source_file.pages.filter(source=ExtractedPage.Source.OCR).exists())


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=True)
class OCRQuotaTests(TestCase):
    """A spent daily quota stops the pass instead of grinding through it."""

    class QuotaBurntProvider(FakeOCRProvider):
        def ocr_page(self, image, *, mime_type="image/png", language_hint=""):
            from agents.ocr import OCRQuotaExhausted

            with self._lock:
                self.calls.append((len(image), language_hint))
                spent = len(self.calls) > 1
            if spent:
                raise OCRQuotaExhausted("The Gemini daily free-tier quota is used up.")
            return OCRResult(text="First page only.", provider=self.name, model=self.model)

    def test_pages_left_unread_stay_flagged_and_the_reason_is_recorded(self):
        user = User.objects.create_user("nadia", password="quiet-precision-42")
        course = Course.objects.create(instructor=user, name="Discrete Maths", code="CS210")
        source_file = SourceFile.objects.create(
            course=course,
            file=SimpleUploadedFile("deck.pdf", make_image_pdf(text_pages=1, image_pages=3)),
            original_name="deck.pdf",
            kind=SourceFile.Kind.PDF,
        )

        ingest_source_file(source_file, ocr_provider=self.QuotaBurntProvider())

        self.assertEqual(source_file.pages_from_ocr, 1)
        self.assertEqual(source_file.pages_without_text, 2)
        self.assertEqual(source_file.status, SourceFile.Status.PARTIAL_TEXT)
        self.assertIn("daily quota", source_file.status_detail)
        # Nothing junk stored for the pages that were never read.
        for page in source_file.pages.filter(is_image_only=True):
            self.assertEqual(page.source, ExtractedPage.Source.TEXT_LAYER)


# --- The three shapes of page that must be re-read --------------------------
#
# Thresholds were measured on the real uploaded files (ch10.3.pdf and the
# Arabic guidelines file); the numbers and the pages that set them are recorded
# in config/settings.py. These lock in the decision at each boundary, including
# the pages that must *not* be re-read.

REASON = ExtractedPage.OCRReason


def signals(letters=200, total_chars=0, unmappable=0, image_coverage=0.0, has_content=True):
    from courses.services.ingest import PageSignals

    return PageSignals(
        letters=letters,
        total_chars=total_chars or max(letters, 1),
        unmappable=unmappable,
        image_coverage=image_coverage,
        has_content=has_content,
    )


class OCRTriggerTests(TestCase):
    """Which pages `classify_ocr_need` sends for OCR, and which it leaves alone."""

    def classify(self, **kwargs):
        from courses.services.ingest import classify_ocr_need

        return classify_ocr_need(signals(**kwargs))

    def test_a_page_with_nothing_readable_is_image_only(self):
        self.assertEqual(self.classify(letters=1, image_coverage=0.8), REASON.IMAGE_ONLY)

    def test_a_genuinely_blank_page_is_not_sent_for_ocr(self):
        # Honesty runs both ways: a blank page is blank, not "unread".
        self.assertEqual(self.classify(letters=0, has_content=False), "")

    def test_a_title_over_a_body_image_is_mixed(self):
        # ch10.3 pages 3, 4, 5: 36 letters of title, body inside an image.
        self.assertEqual(self.classify(letters=36, image_coverage=0.27), REASON.MIXED)

    def test_a_full_page_of_text_beside_a_large_figure_is_left_alone(self):
        # ch10.3 page 31: 0.49 coverage — more than the broken pages — but the
        # body prose is all in the text layer. Coverage alone would re-read it.
        self.assertEqual(self.classify(letters=154, image_coverage=0.49), "")

    def test_a_normal_page_with_one_small_figure_is_left_alone(self):
        # ch10.3 page 43: a real page, one small image.
        self.assertEqual(self.classify(letters=287, image_coverage=0.18), "")

    def test_a_short_page_with_no_image_is_left_alone(self):
        # ch10.3 page 37: 47 letters, diagram drawn as vector art with its
        # labels in the text layer. Letters alone would re-read it.
        self.assertEqual(self.classify(letters=47, image_coverage=0.0), "")

    def test_a_page_of_font_junk_is_defective(self):
        # The Arabic file's page 16: 39 unmappable chars in its heading.
        self.assertEqual(
            self.classify(letters=991, total_chars=1223, unmappable=39),
            REASON.DEFECTIVE_FONT,
        )

    def test_one_or_two_junk_characters_are_not_worth_an_ocr_call(self):
        for count in (1, 2, 4):
            self.assertEqual(
                self.classify(letters=202, total_chars=279, unmappable=count),
                "",
                f"{count} junk chars should not trigger OCR",
            )

    def test_a_few_stray_glyphs_on_a_very_long_page_do_not_trigger(self):
        self.assertEqual(self.classify(letters=9000, total_chars=12000, unmappable=8), "")

    def test_greek_maths_notation_is_not_mistaken_for_font_junk(self):
        # Measured on ch10.3.pdf: 57 real α, β, γ, δ. Counting those as junk
        # reported a clean file as broken and would re-read every maths page.
        from courses.services.ingest import UNMAPPABLE_GLYPHS

        self.assertEqual(UNMAPPABLE_GLYPHS.findall("α(x) = x′, β, γ, δ"), [])
        # The Arabic file's real defect: glyphs mapped into Latin Extended.
        self.assertEqual(len(UNMAPPABLE_GLYPHS.findall("كيŚ ũŻكŮ الاĸتřادę")), 7)

    def test_a_page_with_no_text_layer_at_all_beats_the_other_reasons(self):
        # Recorded as the reason that describes the page best.
        self.assertEqual(
            self.classify(letters=0, unmappable=40, image_coverage=0.9), REASON.IMAGE_ONLY
        )


def make_mixed_pdf() -> bytes:
    """A deck like ch10.3 pages 3-5: the title has a text layer, the body of
    the slide is a pasted image, so most of the page is silently unread."""
    from reportlab.lib.utils import ImageReader

    picture = ImageReader(io.BytesIO(_png_bytes()))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    # A complete slide: real body text, and a small figure beside it.
    pdf.drawString(72, 800, "1")
    pdf.drawString(72, 770, "Logical Gates and Combinatorial Circuits")
    pdf.drawString(72, 740, "A NOT gate can be implemented using a NAND gate, and an")
    pdf.drawString(72, 710, "AND gate can be implemented using two NAND gates in series.")
    pdf.drawImage(picture, 72, 500, width=150, height=110)
    pdf.showPage()
    # A mixed slide: the title only, with the body as a large image.
    pdf.drawString(72, 800, "2")
    pdf.drawString(72, 770, "Logical Gates and Combinatorial Circuits")
    pdf.drawImage(picture, 72, 300, width=400, height=300)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=True)
class MixedPageTests(TestCase):
    """A page whose title reads and whose body is an image (case b)."""

    def setUp(self):
        self.user = User.objects.create_user("nadia", password="quiet-precision-42")
        self.course = Course.objects.create(
            instructor=self.user, name="Discrete Maths", code="CS210"
        )

    def _ingest(self, provider=None, content=None):
        source_file = SourceFile.objects.create(
            course=self.course,
            file=SimpleUploadedFile("deck.pdf", content or make_mixed_pdf()),
            original_name="deck.pdf",
            kind=SourceFile.Kind.PDF,
        )
        return ingest_source_file(source_file, ocr_provider=provider or FakeOCRProvider())

    def test_the_body_image_is_read_and_the_title_page_is_not_touched(self):
        source_file = self._ingest()

        complete, mixed = source_file.pages.all()
        self.assertEqual(complete.source, ExtractedPage.Source.TEXT_LAYER)
        self.assertEqual(complete.ocr_reason, "")
        self.assertEqual(mixed.source, ExtractedPage.Source.OCR)
        self.assertEqual(mixed.ocr_reason, REASON.MIXED)

    def test_the_ocr_text_replaces_the_partial_text_layer_outright(self):
        # One source per page: no merging, so nothing can be duplicated.
        source_file = self._ingest(FakeOCRProvider(text="The full slide body, transcribed."))

        mixed = source_file.pages.get(number=2)
        self.assertEqual(mixed.text, "The full slide body, transcribed.")
        self.assertNotIn("Logical Gates", mixed.text)

    def test_a_file_whose_pages_were_all_recovered_is_ready(self):
        source_file = self._ingest()

        self.assertEqual(source_file.status, SourceFile.Status.READY)
        self.assertEqual(source_file.pages_without_text, 0)
        self.assertEqual(source_file.pages_from_ocr, 1)

    def test_the_file_says_how_many_pages_were_re_read_and_why(self):
        source_file = self._ingest()

        self.assertIn("body content in an image", source_file.status_detail)
        self.assertEqual(
            source_file.ocr_breakdown,
            [{"reason": REASON.MIXED, "label": REASON.MIXED.label, "read": 1,
              "unread": 0, "pages": [2]}],
        )

    def test_a_mixed_page_ocr_could_not_read_is_flagged_not_passed_off(self):
        source_file = self._ingest(FakeOCRProvider(text=None))

        mixed = source_file.pages.get(number=2)
        self.assertEqual(mixed.source, ExtractedPage.Source.TEXT_LAYER)
        self.assertTrue(mixed.is_partially_unread)
        self.assertEqual(source_file.status, SourceFile.Status.PARTIAL_TEXT)
        # The title it did extract is kept — nothing is thrown away.
        self.assertIn("Logical Gates", mixed.text)

    def test_the_reader_marks_a_page_that_is_only_partly_read(self):
        source_file = self._ingest(FakeOCRProvider(text=None))
        self.client.login(username="nadia", password="quiet-precision-42")

        response = self.client.get(source_file.get_absolute_url(), {"page": 2})

        self.assertContains(response, "Only part of this page could be read")
        self.assertContains(response, "Logical Gates")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(), OCR_ENABLED=True)
class DefectiveFontPageTests(TestCase):
    """A page with a full text layer of font junk (case c).

    The trigger itself is covered by `OCRTriggerTests` against the real file's
    measurements; what matters here is that such a page is re-read whole, and
    that a short transcription never silently replaces a page of real text.
    """

    def setUp(self):
        user = User.objects.create_user("nadia", password="quiet-precision-42")
        course = Course.objects.create(instructor=user, name="Policy", code="AR101")
        self.source_file = SourceFile.objects.create(
            course=course,
            file=SimpleUploadedFile("guidelines.pdf", make_pdf()),
            original_name="guidelines.pdf",
            kind=SourceFile.Kind.PDF,
        )
        ingest_source_file(self.source_file, run_ocr=False)
        self.page = self.source_file.pages.get(number=1)
        self.original = self.page.text
        self.page.ocr_reason = REASON.DEFECTIVE_FONT
        self.page.save(update_fields=["ocr_reason"])

    def _run(self, provider):
        from courses.services.ingest import ocr_pages_needing_it

        run = ocr_pages_needing_it(self.source_file, provider)
        self.page.refresh_from_db()
        return run

    def test_the_whole_page_is_re_read_and_marked_as_ocr(self):
        long_enough = " ".join(["Precision is the share of predicted positives"] * 3)
        run = self._run(FakeOCRProvider(text=long_enough))

        self.assertEqual(run.read, 1)
        self.assertEqual(run.reasons_read, {REASON.DEFECTIVE_FONT: 1})
        self.assertEqual(self.page.source, ExtractedPage.Source.OCR)
        self.assertEqual(self.page.text, long_enough)

    def test_a_truncated_transcription_never_replaces_a_page_of_real_text(self):
        # Losing a page of body text to fix a heading would be the worse bug.
        run = self._run(FakeOCRProvider(text="Chapter 1"))

        self.assertEqual(run.read, 0)
        self.assertEqual(run.kept_text_layer, 1)
        self.assertEqual(self.page.text, self.original)
        self.assertEqual(self.page.source, ExtractedPage.Source.TEXT_LAYER)
