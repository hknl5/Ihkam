"""File → text, with page numbers preserved (M1).

Every extractor returns the same thing: an ordered list of `PageText`. Page
numbers survive ingest because every later citation ("source: page 14") is
built from them — losing them here cannot be repaired downstream.

PDF is implemented. PowerPoint / Word / plain text are stubbed behind the same
interface so adding them is one function, not a refactor. Chunking and
embeddings belong to M2 and are deliberately absent.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass

from django.conf import settings
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
    #: The page carries content (images, diagrams, vector art) but no text
    #: layer to read. This is the OCR gap, reported rather than hidden.
    is_image_only: bool = False
    #: Characters the PDF's own font tables could not map to real letters.
    #: A defective embedded ToUnicode CMap, not something extraction can fix.
    unmappable_chars: int = 0


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

    Uses PDFium (`pypdfium2`). It was chosen over pypdf on the real uploaded
    files: pypdf silently dropped every heading set in this document's
    subsetted Arabic fonts, and PDFium recovers them. PDFium also keeps the
    logical (not visual) character order for Arabic, which pdfminer/pdfplumber
    do not — they return Arabic reversed.

    Pages with no text layer come back empty and flagged rather than missing —
    an image-only page is still page 7, and the caller reports the gap.
    """
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(fileobj)
        page_total = len(document)
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise ExtractionError("This PDF is password-protected.") from exc
        raise ExtractionError("This file could not be read as a PDF.") from exc
    except (OSError, ValueError) as exc:
        raise ExtractionError("This file could not be read as a PDF.") from exc

    pages = []
    for index in range(page_total):
        page = document[index]
        try:
            raw = _page_text(page)
        except Exception:  # noqa: BLE001 — one bad page must not lose the rest
            logger.warning("Could not extract page %s", index + 1, exc_info=True)
            raw = ""
        text = normalize_whitespace(raw)
        pages.append(
            PageText(
                number=index + 1,
                text=text,
                is_image_only=_looks_image_only(page, text),
                unmappable_chars=len(UNMAPPABLE_GLYPHS.findall(text)),
            )
        )

    if not pages:
        raise ExtractionError("This PDF has no pages.")
    return pages


#: Characters outside any script this project handles. They appear when a PDF
#: embeds a subsetted font whose ToUnicode table maps glyphs to arbitrary
#: codepoints — the text is unrecoverable without OCR, so it is counted and
#: reported rather than passed off as real content.
UNMAPPABLE_GLYPHS = re.compile(r"[Ā-ʯͰ-Ͽ]")

#: A page with content objects but essentially no letters is a picture of a
#: page, not a page. Short real pages (a section title) clear this easily.
MIN_MEANINGFUL_LETTERS = 8
_NON_LETTERS = re.compile(r"[\W\d_]+", re.UNICODE)

_ALEF = {"ا", "أ", "إ", "آ"}  # ا أ إ آ
_LAM = "ل"  # ل


def _page_text(page) -> str:
    """Text of one page, with lam-alef ligatures put back in logical order.

    PDFium reports a lam-alef ligature as its two letters in *visual* order
    (alef then lam), both carrying the identical character box because they
    come from one glyph. That identical box is what makes the repair safe:
    a genuine "ال" (the definite article) is two glyphs with two boxes, and is
    left alone. Without this, every "الاصطناعي" reads "االصطناعي".
    """
    textpage = page.get_textpage()
    chars: list[list] = []
    for i in range(textpage.count_chars()):
        try:
            box = textpage.get_charbox(i, loose=False)
        except Exception:  # noqa: BLE001 — newlines and marks have no box
            box = None
        chars.append([textpage.get_text_range(i, 1), box])

    repair_lam_alef(chars)
    return "".join(char for char, _ in chars)


def repair_lam_alef(chars: list[list]) -> int:
    """Swap the two letters of every lam-alef ligature back into logical order.

    `chars` is a list of `[character, charbox]` pairs in extraction order;
    it is modified in place. Returns how many ligatures were repaired.

    Two adjacent characters sharing an identical box came from a single glyph.
    When that glyph is a lam-alef, the pair arrives as (alef, lam) and must be
    read (lam, alef). The shared box is the whole safety of this: the definite
    article "ال" is two glyphs with two different boxes and is never touched.
    """
    repaired = 0
    for i in range(len(chars) - 1):
        (char, box), (next_char, next_box) = chars[i], chars[i + 1]
        if char in _ALEF and next_char == _LAM and box is not None and box == next_box:
            chars[i][0], chars[i + 1][0] = next_char, char
            repaired += 1
    return repaired


def _looks_image_only(page, text: str) -> bool:
    """True when the page shows content but hands us no text worth reading.

    The failure this catches is a slide whose body is a pasted screenshot: the
    only extractable text is the slide number, so a naive extractor returns
    "8" and calls it a page.
    """
    if len(_NON_LETTERS.sub("", text)) >= MIN_MEANINGFUL_LETTERS:
        return False
    try:
        return any(obj.type in (_PDFIUM_IMAGE, _PDFIUM_PATH) for obj in page.get_objects())
    except Exception:  # noqa: BLE001 — object inspection is best-effort
        logger.warning("Could not inspect page objects", exc_info=True)
        return False


_PDFIUM_IMAGE = 3
_PDFIUM_PATH = 2


def render_page_png(fileobj, number: int, width: int | None = None) -> bytes:
    """Rasterise one 1-based PDF page to a PNG, for OCR.

    Rendering is kept here, next to extraction, because both answer the same
    question about the same file — and it keeps the OCR provider free of any
    knowledge of PDFs.
    """
    import pypdfium2 as pdfium

    target_width = width or getattr(settings, "OCR_RENDER_WIDTH", 1600)
    try:
        document = pdfium.PdfDocument(fileobj)
        page = document[number - 1]
    except (IndexError, ValueError) as exc:
        raise ExtractionError(f"This PDF has no page {number}.") from exc
    except pdfium.PdfiumError as exc:
        raise ExtractionError("This file could not be read as a PDF.") from exc

    scale = max(0.5, min(6.0, target_width / max(page.get_width(), 1)))
    image = page.render(scale=scale).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def _partial_text_message(image_only: int, total: int) -> str:
    return (
        f"{image_only} of {total} pages carry their content as images or diagrams "
        "with no text layer, so nothing could be read from them. Text recognition "
        "(OCR) is not part of this version. The remaining pages extracted normally."
    )


def _unmappable_message(count: int) -> str:
    return (
        f"{count} characters could not be mapped to real letters — this file embeds "
        "fonts whose internal character tables are incomplete, which usually affects "
        "decorative headings. Body text is unaffected."
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
            ExtractedPage(
                source_file=source_file,
                number=p.number,
                text=p.text,
                is_image_only=p.is_image_only,
            )
            for p in pages
        )

        readable = [p for p in pages if p.text.strip()]
        image_only = [p for p in pages if p.is_image_only]
        unmappable = sum(p.unmappable_chars for p in pages)

        if not readable:
            status, detail = SourceFile.Status.NO_TEXT, NO_TEXT_MESSAGE
        elif image_only:
            # Some pages read, some are pictures. Saying "ready" here is what
            # made a 43-page deck look complete when half of it was unread.
            status = SourceFile.Status.PARTIAL_TEXT
            detail = _partial_text_message(len(image_only), len(pages))
        else:
            status, detail = SourceFile.Status.READY, ""

        if unmappable:
            detail = f"{detail} {_unmappable_message(unmappable)}".strip()

        source_file.page_count = len(pages)
        source_file.pages_without_text = len(image_only)
        source_file.unmappable_chars = unmappable
        source_file.status = status
        source_file.status_detail = detail
        source_file.extracted_at = timezone.now()
        source_file.save(
            update_fields=[
                "page_count",
                "pages_without_text",
                "unmappable_chars",
                "status",
                "status_detail",
                "extracted_at",
            ]
        )
    return source_file


def _record_failure(source_file: SourceFile, status: str, detail: str) -> SourceFile:
    source_file.status = status
    source_file.status_detail = detail
    source_file.page_count = 0
    source_file.pages_without_text = 0
    source_file.unmappable_chars = 0
    source_file.extracted_at = timezone.now()
    source_file.save(
        update_fields=[
            "status",
            "status_detail",
            "page_count",
            "pages_without_text",
            "unmappable_chars",
            "extracted_at",
        ]
    )
    return source_file
