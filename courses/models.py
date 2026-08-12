"""Course content: the material an exam is later drafted from (M1).

Topics, chunks and embeddings arrive in M2 — nothing here depends on the LLM.
"""

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse


class Course(models.Model):
    """A course an instructor teaches. The instructor is the owner."""

    class Level(models.TextChoices):
        INTRODUCTORY = "introductory", "Introductory"
        INTERMEDIATE = "intermediate", "Intermediate"
        ADVANCED = "advanced", "Advanced"
        POSTGRADUATE = "postgraduate", "Postgraduate"

    class ContentLanguage(models.TextChoices):
        ARABIC = "ar", "Arabic"
        ENGLISH = "en", "English"
        MIXED = "mixed", "Arabic + English"

    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="courses",
    )
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=32, help_text="e.g. CS310")
    level = models.CharField(max_length=20, choices=Level.choices, default=Level.INTRODUCTORY)
    content_language = models.CharField(
        max_length=10,
        choices=ContentLanguage.choices,
        default=ContentLanguage.ENGLISH,
        help_text="The language of the lecture material, not of the interface.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["instructor", "code"], name="unique_course_code_per_instructor"
            )
        ]

    def __str__(self) -> str:
        return f"{self.code} — {self.name}"

    def get_absolute_url(self) -> str:
        return reverse("courses:detail", args=[self.pk])

    @property
    def is_rtl_content(self) -> bool:
        return self.content_language == self.ContentLanguage.ARABIC

    @property
    def content_lang_attr(self) -> str:
        """A valid BCP-47 tag for `lang=`, or "" for mixed-language material
        (where per-run direction is left to the browser's `dir="auto"`)."""
        if self.content_language in {self.ContentLanguage.ARABIC, self.ContentLanguage.ENGLISH}:
            return self.content_language
        return ""


def source_file_upload_to(instance: "SourceFile", filename: str) -> str:
    return f"courses/{instance.course_id}/sources/{filename}"


class SourceFile(models.Model):
    """An uploaded piece of course material, plus the state of its extraction.

    PDF is the only format extracted in M1. The other kinds are recognised and
    stored so the upload path is already shaped for them — see
    `courses/services/ingest.py` for the stubs.
    """

    class Kind(models.TextChoices):
        PDF = "pdf", "PDF"
        POWERPOINT = "pptx", "PowerPoint"
        WORD = "docx", "Word"
        TEXT = "txt", "Plain text"

    class Status(models.TextChoices):
        PENDING = "pending", "Not extracted yet"
        READY = "ready", "Text extracted"
        PARTIAL_TEXT = "partial_text", "Some pages have no text layer"
        NO_TEXT = "no_text", "No text layer"
        UNSUPPORTED = "unsupported", "Format not supported yet"
        FAILED = "failed", "Extraction failed"

    #: Extension → kind. The upload form rejects anything not listed here.
    EXTENSION_KINDS = {
        ".pdf": Kind.PDF,
        ".pptx": Kind.POWERPOINT,
        ".ppt": Kind.POWERPOINT,
        ".docx": Kind.WORD,
        ".doc": Kind.WORD,
        ".txt": Kind.TEXT,
        ".md": Kind.TEXT,
    }

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="files")
    file = models.FileField(upload_to=source_file_upload_to)
    original_name = models.CharField(max_length=255)
    kind = models.CharField(max_length=10, choices=Kind.choices)
    size_bytes = models.PositiveBigIntegerField(default=0)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    #: Instructor-facing explanation when status is not `ready`.
    status_detail = models.TextField(blank=True)
    page_count = models.PositiveIntegerField(default=0)
    #: Pages that carry content as images or diagrams with no text to read —
    #: the OCR gap, counted so it is never mistaken for an empty page.
    pages_without_text = models.PositiveIntegerField(default=0)
    #: Characters the file's own embedded fonts could not map to real letters.
    unmappable_chars = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    extracted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return self.original_name

    def get_absolute_url(self) -> str:
        return reverse("courses:file_detail", args=[self.course_id, self.pk])

    @classmethod
    def kind_for_filename(cls, filename: str) -> str | None:
        return cls.EXTENSION_KINDS.get(Path(filename).suffix.lower())

    @property
    def is_ready(self) -> bool:
        return self.status == self.Status.READY

    @property
    def has_readable_text(self) -> bool:
        """Whether there is anything worth opening the reader for."""
        return self.status in {self.Status.READY, self.Status.PARTIAL_TEXT}

    @property
    def status_tone(self) -> str:
        """Maps to the design system's `.status--*` modifiers (§5)."""
        return {
            self.Status.READY: "ok",
            self.Status.PENDING: "candidate",
            self.Status.PARTIAL_TEXT: "warn",
            self.Status.NO_TEXT: "warn",
            self.Status.UNSUPPORTED: "warn",
            self.Status.FAILED: "danger",
        }.get(self.status, "candidate")


class ExtractedPage(models.Model):
    """One page of extracted text.

    Pages are kept separate rather than concatenated because every downstream
    citation ("source: page 14") depends on the page number surviving ingest.
    """

    source_file = models.ForeignKey(SourceFile, on_delete=models.CASCADE, related_name="pages")
    number = models.PositiveIntegerField(help_text="1-based page number in the source document.")
    text = models.TextField(blank=True)
    #: The page has content — a screenshot, a figure, a scan — but no text
    #: layer. Distinct from a genuinely blank page, and from a page we simply
    #: failed to read: this one needs OCR, which is deferred.
    is_image_only = models.BooleanField(default=False)

    class Meta:
        ordering = ["number"]
        constraints = [
            models.UniqueConstraint(
                fields=["source_file", "number"], name="unique_page_per_source_file"
            )
        ]

    def __str__(self) -> str:
        return f"{self.source_file.original_name} · page {self.number}"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()
