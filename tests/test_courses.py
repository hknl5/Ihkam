"""M1 success check, as code: create a course, upload a PDF, read the text.

Sample PDFs are generated here rather than committed, so the fixtures stay
readable and the expected text lives next to the assertion.
"""

import io
import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from courses.forms import MAX_UPLOAD_BYTES
from courses.models import Course, ExtractedPage, SourceFile
from courses.services.ingest import (
    UnsupportedFormatError,
    extract_pages,
    extract_pdf,
    ingest_source_file,
    normalize_whitespace,
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


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
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
