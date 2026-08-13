"""Course content: the material an exam is later drafted from (M1 + M2).

M1 is the file and its pages. M2 adds `Topic` — the editable syllabus the
instructor confirms before anything is generated — and `Chunk`, one embedded
passage per piece of readable page text, which Agent 1A retrieves from in M3.
"""

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse
from pgvector.django import VectorField


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
    #: Pages whose text came from OCR rather than a text layer.
    pages_from_ocr = models.PositiveIntegerField(default=0)
    #: Which engine read them, kept for traceability when OCR is re-run or
    #: the provider is switched (e.g. "gemini/gemini-3.6-flash").
    ocr_engine = models.CharField(max_length=100, blank=True)
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
    def ocr_breakdown(self) -> list[dict]:
        """Per-reason counts of the pages extraction could not read in full.

        Built from the pages rather than stored on the file, so it can never
        drift out of step with them. `read` is how many OCR recovered; `unread`
        is how many are still flagged, and is never quietly folded into `read`.
        """
        rows = []
        for reason, label in ExtractedPage.OCRReason.choices:
            pages = [p for p in self.pages.all() if p.ocr_reason == reason]
            if not pages:
                continue
            read = sum(1 for p in pages if p.is_from_ocr)
            rows.append(
                {
                    "reason": reason,
                    "label": label,
                    "read": read,
                    "unread": len(pages) - read,
                    "pages": [p.number for p in pages],
                }
            )
        return rows

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

    class Source(models.TextChoices):
        TEXT_LAYER = "text_layer", "Text layer"
        OCR = "ocr", "Read by OCR"

    class OCRReason(models.TextChoices):
        """Why extraction could not be trusted to have read the whole page.

        The one principle behind all three: a little extractable text is not
        proof the page is complete.
        """

        #: Nothing readable at all — the page is a screenshot or a scan.
        IMAGE_ONLY = "image_only", "No text layer"
        #: A text layer for the title, with the body content inside an image.
        MIXED = "mixed", "Text layer covers only part of the page"
        #: A full text layer, but the embedded fonts map letters to junk.
        DEFECTIVE_FONT = "defective_font", "Embedded fonts map letters to junk"

    source_file = models.ForeignKey(SourceFile, on_delete=models.CASCADE, related_name="pages")
    number = models.PositiveIntegerField(help_text="1-based page number in the source document.")
    text = models.TextField(blank=True)
    #: Where the text came from. A model transcription is not the same
    #: evidence as a text layer, and later milestones weigh it accordingly.
    source = models.CharField(
        max_length=12, choices=Source.choices, default=Source.TEXT_LAYER
    )
    #: The page has content — a screenshot, a figure, a scan — but no text
    #: that could be read, by extraction or by OCR. Distinct from a genuinely
    #: blank page: this one still holds something we cannot see.
    is_image_only = models.BooleanField(default=False)
    #: Why this page was sent for OCR, kept after the fact: it is the only
    #: record of *what* was wrong with the text layer, and the counts per file
    #: are built from it. Blank for a page extraction read in full.
    ocr_reason = models.CharField(
        max_length=16, choices=OCRReason.choices, blank=True, default=""
    )

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

    @property
    def is_from_ocr(self) -> bool:
        return self.source == self.Source.OCR

    @property
    def is_readable(self) -> bool:
        """Whether this page's text may be used as course content (M2).

        One definition, used by both topic extraction and chunking, and the
        same one `_finalize_extraction` counts with: there is text, and the
        page is not the image whose only readable characters are its slide
        number. A page OCR could not read keeps its flag and is excluded here —
        letting an unread page contribute is exactly what flagging it prevents.

        A partially-unread page *is* readable: what was extracted is real text,
        it is simply not all of the page. It is marked as such in the reader,
        and its chunks carry the page number, so the gap stays visible.
        """
        return bool(self.text.strip()) and not self.is_image_only

    @property
    def is_partially_unread(self) -> bool:
        """Text was extracted, but not all of the page's content.

        A mixed or defective-font page OCR could not recover. It reads like an
        ordinary page, which is exactly why it has to be marked: the missing
        part is invisible.
        """
        return bool(self.ocr_reason) and not self.is_from_ocr and not self.is_image_only

    @property
    def ocr_reason_explanation(self) -> str:
        """Why this page was re-read, in words for the instructor."""
        return {
            self.OCRReason.IMAGE_ONLY: "this page had no text layer",
            self.OCRReason.MIXED: (
                "only part of this page had a text layer — the rest of its content "
                "sits in an image"
            ),
            self.OCRReason.DEFECTIVE_FONT: (
                "this page's embedded fonts do not map to real letters"
            ),
        }.get(self.ocr_reason, "extraction could not read this page in full")


class TopicQuerySet(models.QuerySet):
    def included(self):
        """Topics that may flow downstream.

        The instructor's "not taught in lectures" is a hard exclusion, not a
        hint: an excluded topic never reaches retrieval, a blueprint row or a
        generated question. Every downstream caller goes through this.
        """
        return self.filter(excluded=False)

    def chapters(self):
        return self.filter(parent__isnull=True)


class Topic(models.Model):
    """One chapter or sub-topic of a course's syllabus (M2).

    Extracted by a model, then **confirmed by the instructor** — renamed,
    merged, deleted, added to, or marked "not taught". Nothing here is trusted
    until they have been through it, which is why `excluded` lives on the row
    rather than being inferred.

    A chapter is a topic with no parent; a sub-topic points at its chapter.
    """

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="topics")
    #: The parent chapter. Null for a chapter itself. Deleting a chapter
    #: promotes its sub-topics rather than deleting them — an instructor
    #: removing a heading is not asking to lose everything under it.
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="subtopics",
    )
    name = models.CharField(max_length=300)
    #: Where in the uploaded material this topic was found. Null for a topic
    #: the instructor added by hand — they know it is taught, and no page
    #: reference is invented for it.
    source_file = models.ForeignKey(
        SourceFile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="topics",
    )
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    #: "Not taught in lectures." Set by the instructor, honoured everywhere.
    excluded = models.BooleanField(
        default=False,
        help_text="Not taught in lectures — never used to generate questions.",
    )

    # --- Detail the extraction found, kept with the topic it belongs to ------
    # Stored rather than discarded so Agent 2A (M5) can ground a question in
    # the course's own wording instead of re-reading the whole file. All four
    # are lists of plain strings, except `definitions` which is
    # [{"term": ..., "text": ...}].
    key_terms = models.JSONField(default=list, blank=True)
    definitions = models.JSONField(default=list, blank=True)
    formulas = models.JSONField(default=list, blank=True)
    examples = models.JSONField(default=list, blank=True)

    #: Ordering within the course, so a merged or renamed list keeps the
    #: reading order of the material rather than jumping around.
    position = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = TopicQuerySet.as_manager()

    class Meta:
        ordering = ["position", "pk"]

    def __str__(self) -> str:
        return self.name

    @property
    def is_chapter(self) -> bool:
        return self.parent_id is None

    @property
    def page_span(self) -> str:
        """"pages 3–7", "page 12", or "" when there is no reference."""
        if self.page_start is None:
            return ""
        if self.page_end is None or self.page_end == self.page_start:
            return f"page {self.page_start}"
        return f"pages {self.page_start}–{self.page_end}"

    @property
    def status_tone(self) -> str:
        """Maps to the design system's `.status--*` modifiers (§5)."""
        return "warn" if self.excluded else "ok"

    @property
    def detail_counts(self) -> list[tuple[str, int]]:
        """Non-empty detail groups, for the review screen's summary line."""
        pairs = [
            ("terms", len(self.key_terms or [])),
            ("definitions", len(self.definitions or [])),
            ("formulas", len(self.formulas or [])),
            ("examples", len(self.examples or [])),
        ]
        return [(label, count) for label, count in pairs if count]


class ChunkQuerySet(models.QuerySet):
    def usable(self):
        """Chunks that may be retrieved (M3 onwards).

        A chunk attached to an excluded topic is out, the same way the topic
        is. A chunk with no topic stays in: it is course material the
        instructor never ruled out, only material no topic claimed.
        """
        return self.exclude(topic__excluded=True)


class Chunk(models.Model):
    """One embedded passage of readable page text (M2).

    Chunks never cross a page boundary. That is deliberate: every citation the
    system will later make ("source: page 14") is only as good as the page
    number on the passage it came from, and a passage spanning two pages has no
    honest answer to that question.
    """

    source_file = models.ForeignKey(SourceFile, on_delete=models.CASCADE, related_name="chunks")
    #: 1-based page number, copied from `ExtractedPage.number` so a citation
    #: survives even if the page row is later re-extracted.
    page = models.PositiveIntegerField()
    #: Position of this passage within its page, 0-based.
    position = models.PositiveIntegerField(default=0)
    text = models.TextField()
    embedding = VectorField(dimensions=settings.EMBEDDING_DIM)
    #: Where the underlying page text came from. An OCR transcription is a
    #: model's reading of a picture, not a text layer, and later milestones are
    #: entitled to know which one they are quoting.
    source = models.CharField(
        max_length=12,
        choices=ExtractedPage.Source.choices,
        default=ExtractedPage.Source.TEXT_LAYER,
    )
    #: The topic this passage falls under, matched deterministically by page
    #: span after the instructor confirms the topic list. Null when no topic
    #: claims the page — which is not an error, just an unclaimed passage.
    topic = models.ForeignKey(
        Topic,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chunks",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    objects = ChunkQuerySet.as_manager()

    class Meta:
        ordering = ["source_file_id", "page", "position"]
        constraints = [
            models.UniqueConstraint(
                fields=["source_file", "page", "position"], name="unique_chunk_position"
            )
        ]

    def __str__(self) -> str:
        return f"{self.source_file.original_name} · page {self.page} · #{self.position}"

    @property
    def is_from_ocr(self) -> bool:
        return self.source == ExtractedPage.Source.OCR
