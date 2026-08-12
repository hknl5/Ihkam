"""File → text, with page numbers preserved (M1).

Every extractor returns the same thing: an ordered list of `PageText`. Page
numbers survive ingest because every later citation ("source: page 14") is
built from them — losing them here cannot be repaired downstream.

PDF is implemented. PowerPoint / Word / plain text are stubbed behind the same
interface so adding them is one function, not a refactor. Chunking and
embeddings belong to M2 and are deliberately absent.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from django.db import transaction
from django.utils import timezone

from ..models import ExtractedPage, SourceFile

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """The file could not be read at all (corrupt, encrypted, unreadable)."""


class UnsupportedFormatError(Exception):
    """The format is recognised but no extractor exists for it yet."""


@dataclass(frozen=True)
class PageText:
    number: int  # 1-based, as the reader sees it
    text: str


# --- Normalisation ----------------------------------------------------------

_SOFT_HYPHEN = "­"
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+(\n)")
_SPACES = re.compile(r"[ \t ]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Tidy extractor output without changing what it says.

    NFKC keeps Arabic presentation forms and Latin ligatures comparable later;
    line structure is kept (paragraphs still read as paragraphs), only runs of
    whitespace collapse.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_SOFT_HYPHEN, "")
    text = _ZERO_WIDTH.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACE.sub(r"\1", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


# --- Extractors -------------------------------------------------------------


def extract_pdf(fileobj) -> list[PageText]:
    """One `PageText` per PDF page, in document order.

    Pages with no text layer come back empty rather than missing — an
    image-only page is still page 7, and the caller reports the gap.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(fileobj)
        if reader.is_encrypted:
            # An empty password unlocks most "protected" lecture PDFs.
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 — surfaced to the instructor
                raise ExtractionError("This PDF is password-protected.") from exc
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:  # noqa: BLE001 — one bad page must not lose the rest
                logger.warning("Could not extract page %s", index, exc_info=True)
                raw = ""
            pages.append(PageText(number=index, text=normalize_whitespace(raw)))
    except ExtractionError:
        raise
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError("This file could not be read as a PDF.") from exc

    if not pages:
        raise ExtractionError("This PDF has no pages.")
    return pages


def extract_pptx(fileobj) -> list[PageText]:
    """TODO (post-M1): PowerPoint. One `PageText` per slide, `number` = slide
    number, text from shape text frames + speaker notes. Add `python-pptx`."""
    raise UnsupportedFormatError("PowerPoint files are not extracted yet.")


def extract_docx(fileobj) -> list[PageText]:
    """TODO (post-M1): Word. Word has no fixed pages, so page numbers must be
    synthesised — either explicit page breaks or a fixed paragraph window —
    and the choice recorded, since citations depend on it. Add `python-docx`."""
    raise UnsupportedFormatError("Word files are not extracted yet.")


def extract_txt(fileobj) -> list[PageText]:
    """TODO (post-M1): plain text / Markdown. No inherent pages; split on form
    feeds, else on a fixed line count, and treat each block as one page."""
    raise UnsupportedFormatError("Plain text files are not extracted yet.")


EXTRACTORS = {
    SourceFile.Kind.PDF: extract_pdf,
    SourceFile.Kind.POWERPOINT: extract_pptx,
    SourceFile.Kind.WORD: extract_docx,
    SourceFile.Kind.TEXT: extract_txt,
}


def extract_pages(fileobj, kind: str) -> list[PageText]:
    """Dispatch to the extractor for `kind`. The one entry point for callers."""
    extractor = EXTRACTORS.get(kind)
    if extractor is None:
        raise UnsupportedFormatError(f"No extractor for '{kind}'.")
    return extractor(fileobj)


# --- Persistence ------------------------------------------------------------

NO_TEXT_MESSAGE = (
    "No text layer found — this looks like a scanned or image-only document. "
    "Text recognition (OCR) is not part of this version, so please upload a "
    "text-based PDF for now."
)


def ingest_source_file(source_file: SourceFile) -> SourceFile:
    """Extract `source_file` and store its pages. Never raises for bad input.

    Failure is recorded on the row (`status` + `status_detail`) and shown to
    the instructor; only a genuine bug propagates.
    """
    try:
        with source_file.file.open("rb") as fh:
            pages = extract_pages(fh, source_file.kind)
    except UnsupportedFormatError as exc:
        return _record_failure(source_file, SourceFile.Status.UNSUPPORTED, str(exc))
    except ExtractionError as exc:
        return _record_failure(source_file, SourceFile.Status.FAILED, str(exc))

    with transaction.atomic():
        source_file.pages.all().delete()
        ExtractedPage.objects.bulk_create(
            ExtractedPage(source_file=source_file, number=p.number, text=p.text) for p in pages
        )
        has_text = any(p.text.strip() for p in pages)
        source_file.page_count = len(pages)
        source_file.status = SourceFile.Status.READY if has_text else SourceFile.Status.NO_TEXT
        source_file.status_detail = "" if has_text else NO_TEXT_MESSAGE
        source_file.extracted_at = timezone.now()
        source_file.save(
            update_fields=["page_count", "status", "status_detail", "extracted_at"]
        )
    return source_file


def _record_failure(source_file: SourceFile, status: str, detail: str) -> SourceFile:
    source_file.status = status
    source_file.status_detail = detail
    source_file.page_count = 0
    source_file.extracted_at = timezone.now()
    source_file.save(update_fields=["status", "status_detail", "page_count", "extracted_at"])
    return source_file
