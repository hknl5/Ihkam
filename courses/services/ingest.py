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
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

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
    #: Why this page needs re-reading by OCR, or "" when extraction read it in
    #: full. One of `ExtractedPage.OCRReason`.
    ocr_reason: str = ""


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
        signals = measure_page(page, text)
        reason = classify_ocr_need(signals)
        pages.append(
            PageText(
                number=index + 1,
                text=text,
                is_image_only=(reason == ExtractedPage.OCRReason.IMAGE_ONLY),
                unmappable_chars=signals.unmappable,
                ocr_reason=reason,
            )
        )

    if not pages:
        raise ExtractionError("This PDF has no pages.")
    return pages


#: Characters outside any script this project handles. They appear when a PDF
#: embeds a subsetted font whose ToUnicode table maps glyphs to arbitrary
#: codepoints — the text is unrecoverable without OCR, so it is counted and
#: reported rather than passed off as real content.
#:
#: Latin Extended-A/B and IPA only. Greek was here too, and had to come out:
#: on ch10.3.pdf it matched 57 real characters — the α, β, γ, δ of ordinary
#: maths notation — which both misreported a clean file as having broken fonts
#: and, now that this count decides whether to re-read a page, would have sent
#: every maths page for needless OCR. The defective Arabic fonts this catches
#: map into Latin Extended exclusively (measured: 427 chars, none Greek).
UNMAPPABLE_GLYPHS = re.compile(r"[Ā-ʯ]")

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


_PDFIUM_IMAGE = 3
_PDFIUM_PATH = 2

#: Cells per axis when measuring how much of a page its images cover. A grid
#: is used rather than summing areas because slide images overlap, and summing
#: overlapping boxes reports coverage above 100% — which would re-read pages
#: that are perfectly readable.
_COVERAGE_GRID = 60


@dataclass(frozen=True)
class PageSignals:
    """What is measurable about one page, before deciding whether to trust it."""

    letters: int
    total_chars: int
    unmappable: int
    #: Fraction of the page covered by images, 0.0-1.0.
    image_coverage: float
    #: The page draws something — an image or vector art — as opposed to being
    #: genuinely blank.
    has_content: bool

    @property
    def unmappable_ratio(self) -> float:
        return self.unmappable / self.total_chars if self.total_chars else 0.0


def _image_coverage(boxes, width: float, height: float) -> float:
    """Fraction of the page the image boxes cover, counting overlaps once."""
    if not boxes:
        return 0.0
    covered = set()
    for left, bottom, right, top in boxes:
        first_col = int(_COVERAGE_GRID * max(left, 0.0) / width)
        last_col = int(_COVERAGE_GRID * min(right, width) / width)
        first_row = int(_COVERAGE_GRID * max(bottom, 0.0) / height)
        last_row = int(_COVERAGE_GRID * min(top, height) / height)
        for col in range(first_col, min(last_col + 1, _COVERAGE_GRID)):
            for row in range(first_row, min(last_row + 1, _COVERAGE_GRID)):
                covered.add((col, row))
    return len(covered) / (_COVERAGE_GRID * _COVERAGE_GRID)


def measure_page(page, text: str) -> PageSignals:
    """Measure one page's text against what it actually draws.

    Image geometry comes from the objects' own bounds, and nested objects are
    walked (`max_depth`): on the real slide decks every body image sits inside
    a form XObject, so a shallow scan finds no images at all.
    """
    width = max(page.get_width(), 1.0)
    height = max(page.get_height(), 1.0)
    boxes, has_content = [], False
    try:
        for obj in page.get_objects(max_depth=_OBJECT_DEPTH):
            if obj.type not in (_PDFIUM_IMAGE, _PDFIUM_PATH):
                continue
            has_content = True
            if obj.type != _PDFIUM_IMAGE:
                continue
            try:
                boxes.append(obj.get_bounds())
            except Exception:  # noqa: BLE001 — an unplaceable image still counts
                logger.debug("Could not measure an image's bounds", exc_info=True)
    except Exception:  # noqa: BLE001 — object inspection is best-effort
        logger.warning("Could not inspect page objects", exc_info=True)

    return PageSignals(
        letters=len(_NON_LETTERS.sub("", text)),
        total_chars=len(text),
        unmappable=len(UNMAPPABLE_GLYPHS.findall(text)),
        image_coverage=_image_coverage(boxes, width, height),
        has_content=has_content,
    )


#: How deep to walk form XObjects looking for images. Real decks nest body
#: images one or two levels down; deeper than this is pointless.
_OBJECT_DEPTH = 6


def classify_ocr_need(signals: PageSignals) -> str:
    """Which of the three incomplete-page shapes this is, or "" if none.

    All three exist because a little extractable text is not proof that a page
    was read. Thresholds and the pages that set them are in `config/settings.py`.

    The order is deliberate: a page with nothing readable is image-only even
    when its fonts are also broken, and the reason recorded is the one that
    describes the page best.
    """
    reason = ExtractedPage.OCRReason
    if signals.letters < MIN_MEANINGFUL_LETTERS:
        # Nothing worth reading. Only a page that draws something is a missed
        # page; a genuinely blank one is blank, and stays that way.
        return reason.IMAGE_ONLY if signals.has_content else ""
    if signals.letters < _setting("OCR_MIXED_MAX_LETTERS", 80) and signals.image_coverage >= _setting(
        "OCR_MIXED_MIN_IMAGE_COVERAGE", 0.20
    ):
        # A title reads; the body is a picture.
        return reason.MIXED
    if signals.unmappable >= _setting("OCR_DEFECTIVE_MIN_CHARS", 8) and (
        signals.unmappable_ratio >= _setting("OCR_DEFECTIVE_MIN_RATIO", 0.005)
    ):
        # The text layer is complete but partly unreadable junk.
        return reason.DEFECTIVE_FONT
    return ""


def _setting(name: str, default):
    return getattr(settings, name, default)


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
    "No text layer found — this looks like a scanned or image-only document, and "
    "text recognition (OCR) could not read it either. Nothing readable was stored "
    "rather than storing something that is not really there."
)


def _partial_text_message(unread: int, total: int) -> str:
    return (
        f"{unread} of {total} pages could not be read in full — their content sits "
        "in images, or their embedded fonts do not map to real letters — and text "
        "recognition (OCR) did not recover them either. The remaining pages read "
        "normally."
    )


def _unmappable_message(count: int) -> str:
    return (
        f"{count} characters could not be mapped to real letters — this file embeds "
        "fonts whose internal character tables are incomplete, which usually affects "
        "decorative headings. Body text is unaffected."
    )


def _over_cap_message(skipped: int, cap: int) -> str:
    return (
        f"{skipped} further pages with no text layer were left unread: this file is "
        f"over the {cap}-page OCR limit for a single upload. Raise "
        "OCR_MAX_PAGES_PER_FILE and upload again to read the rest."
    )


@dataclass
class OCRRun:
    """What one OCR pass over a file did."""

    attempted: int = 0
    read: int = 0
    still_unreadable: int = 0
    skipped_over_cap: int = 0
    #: Pages where the transcription looked truncated, so the text layer was
    #: kept instead. Counted separately: nothing was lost, but nothing was
    #: fixed either.
    kept_text_layer: int = 0
    #: `OCRReason` → pages read, so the file can say *why* it re-read pages.
    reasons_read: dict = field(default_factory=dict)
    engine: str = ""
    error: str = ""


def _looks_truncated(transcription: str, page: ExtractedPage) -> bool:
    """True when a transcription is too short to be replacing the page's text.

    Only a defective-font page can trip this, and only it should: its text
    layer is complete apart from the junk, so a half-length transcription means
    something went wrong and replacing it would lose real content.

    An image-only or mixed page is the opposite case — its text layer is a page
    number and maybe a title, known to be a fragment. A short transcription of
    one of those is not evidence of failure, and preferring the fragment would
    keep exactly the gap this all exists to close.
    """
    if page.ocr_reason != ExtractedPage.OCRReason.DEFECTIVE_FONT:
        return False
    had = len(_NON_LETTERS.sub("", page.text))
    if not had:
        return False
    got = len(_NON_LETTERS.sub("", transcription))
    return got < had * _setting("OCR_MIN_KEEP_RATIO", 0.6)


def ingest_source_file(source_file: SourceFile, *, ocr_provider=None, run_ocr=None) -> SourceFile:
    """Extract `source_file`, store its pages, and OCR the ones with no text.

    Never raises for bad input: failure is recorded on the row (`status` +
    `status_detail`) and shown to the instructor; only a genuine bug
    propagates. OCR runs inline — a slide deck costs about a minute, which is
    the price of the file being readable when the instructor next looks at it.

    Pass `run_ocr=False` (or set `OCR_ENABLED=false`) to skip the OCR pass;
    pages then simply stay flagged as needing it.
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
                ocr_reason=p.ocr_reason,
                source=ExtractedPage.Source.TEXT_LAYER,
            )
            for p in pages
        )
        source_file.page_count = len(pages)
        source_file.unmappable_chars = sum(p.unmappable_chars for p in pages)
        source_file.pages_from_ocr = 0
        source_file.ocr_engine = ""
        source_file.save(
            update_fields=["page_count", "unmappable_chars", "pages_from_ocr", "ocr_engine"]
        )

    if run_ocr is None:
        run_ocr = getattr(settings, "OCR_ENABLED", False)
    run = ocr_pages_needing_it(source_file, ocr_provider) if run_ocr else OCRRun()

    return _finalize_extraction(source_file, run)


def ocr_pages_needing_it(source_file: SourceFile, provider=None) -> OCRRun:
    """Re-read every page extraction could not be trusted to have read in full.

    That is all three shapes of incomplete page — no text layer at all, a text
    layer covering only the title, and a text layer of font junk — chosen by
    `classify_ocr_need` at extraction time and recorded on each page.

    One page per request — batching pages into a single call measurably
    degrades transcription quality, so concurrency is used only to shorten the
    total wait, never to change what is asked of the model.

    A page whose transcription comes back empty keeps whatever it had: a blank
    result is reported as unread, never stored as read.
    """
    from agents.ocr import OCRError, get_ocr_provider  # the seam; see agents/ocr.py

    pending = list(source_file.pages.exclude(ocr_reason="").order_by("number"))
    if not pending:
        return OCRRun()

    try:
        provider = provider or get_ocr_provider()
    except (OCRError, NotImplementedError) as exc:
        logger.warning("OCR unavailable: %s", exc)
        return OCRRun(still_unreadable=len(pending), error=str(exc))

    cap = getattr(settings, "OCR_MAX_PAGES_PER_FILE", 60)
    targets, skipped = pending[:cap], pending[cap:]
    engine = f"{provider.name}/{getattr(provider, 'model', '')}".rstrip("/")
    language_hint = source_file.course.content_language

    # Rendering is local and PDFium is not safe to drive from several threads,
    # so pages are rasterised here and only the network calls fan out.
    images: dict[int, bytes] = {}
    with source_file.file.open("rb") as fh:
        document = fh.read()
    for page in targets:
        try:
            images[page.number] = render_page_png(io.BytesIO(document), page.number)
        except ExtractionError:
            logger.warning("Could not render page %s for OCR", page.number, exc_info=True)

    workers = max(1, min(getattr(settings, "OCR_CONCURRENCY", 5), len(images) or 1))
    transcriptions: dict[int, str] = {}
    # Set when the provider's quota is spent: every remaining page would fail
    # the same way, so the pass stops instead of grinding through them.
    exhausted = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ocr_one_page, provider, image, language_hint, exhausted): number
            for number, image in images.items()
        }
        for future in as_completed(futures):
            number = futures[future]
            text = future.result()
            if text:
                transcriptions[number] = text

    read = 0
    kept_text_layer = 0
    reasons_read: dict[str, int] = {}
    with transaction.atomic():
        for page in targets:
            text = transcriptions.get(page.number, "")
            if not text:
                continue  # stays flagged — an unread page is not a blank page
            if _looks_truncated(text, page):
                # The page keeps its text layer. Replacing a full page of text
                # with half a transcription would lose content silently, which
                # is worse than a mangled heading.
                logger.warning(
                    "OCR of page %s came back much shorter than its text layer; "
                    "keeping the text layer",
                    page.number,
                )
                kept_text_layer += 1
                continue
            # One source per page: the transcription replaces the text layer
            # outright rather than being merged into it, so nothing can be
            # duplicated or half-dropped.
            page.text = text
            page.source = ExtractedPage.Source.OCR
            page.is_image_only = False
            page.save(update_fields=["text", "source", "is_image_only"])
            reasons_read[page.ocr_reason] = reasons_read.get(page.ocr_reason, 0) + 1
            read += 1

        source_file.pages_from_ocr = read
        source_file.ocr_engine = engine if read else ""
        source_file.save(update_fields=["pages_from_ocr", "ocr_engine"])

    return OCRRun(
        attempted=len(targets),
        read=read,
        still_unreadable=len(targets) - read,
        skipped_over_cap=len(skipped),
        kept_text_layer=kept_text_layer,
        reasons_read=reasons_read,
        engine=engine,
        error=(
            "the OCR provider's daily quota ran out part-way through"
            if exhausted.is_set()
            else ""
        ),
    )


def _ocr_one_page(provider, image: bytes, language_hint: str, exhausted=None) -> str:
    """One page, one call. Returns "" for anything not worth storing."""
    from agents.ocr import OCRQuotaExhausted

    if exhausted is not None and exhausted.is_set():
        return ""
    try:
        result = provider.ocr_page(image, language_hint=language_hint)
    except OCRQuotaExhausted as exc:
        if exhausted is not None:
            exhausted.set()
        logger.warning("OCR stopped: %s", exc)
        return ""
    except Exception:  # noqa: BLE001 — one unreadable page must not lose the rest
        logger.warning("OCR call failed", exc_info=True)
        return ""
    if not result.is_usable:
        return ""
    return normalize_whitespace(result.text)


def _finalize_extraction(source_file: SourceFile, run: OCRRun) -> SourceFile:
    """Set status and counts from the stored pages, after any OCR pass."""
    pages = list(source_file.pages.all())
    readable = [p for p in pages if p.text.strip() and not p.is_image_only]
    image_only = [p for p in pages if p.is_image_only]
    # Flagged at extraction and not recovered since: still not read in full.
    # A mixed or defective-font page belongs here too — it has *some* text, and
    # counting it as readable is the dishonesty this work set out to remove.
    unread = [p for p in pages if p.ocr_reason and not p.is_from_ocr]

    if not readable:
        status, detail = SourceFile.Status.NO_TEXT, NO_TEXT_MESSAGE
    elif unread:
        status = SourceFile.Status.PARTIAL_TEXT
        detail = _partial_text_message(len(unread), len(pages))
    else:
        status, detail = SourceFile.Status.READY, ""

    notes = [detail]
    if run.read:
        notes.append(
            f"{run.read} page{'s' if run.read != 1 else ''} "
            f"({_reasons_phrase(run.reasons_read)}) "
            f"were re-read by OCR ({run.engine}); that text is a model "
            "transcription, not a text layer."
        )
    if run.kept_text_layer:
        notes.append(
            f"{run.kept_text_layer} page(s) kept their original text because the "
            "transcription came back suspiciously short."
        )
    if run.skipped_over_cap:
        notes.append(_over_cap_message(run.skipped_over_cap, settings.OCR_MAX_PAGES_PER_FILE))
    if run.error:
        notes.append(
            f"Some pages were left unread because {run.error}."
            if run.read
            else f"OCR did not run: {run.error}"
        )

    # Recounted from what is stored now, not from what extraction first saw:
    # OCR replaces the junk with real letters, and the count has to show that.
    source_file.unmappable_chars = sum(len(UNMAPPABLE_GLYPHS.findall(p.text)) for p in pages)
    if source_file.unmappable_chars:
        notes.append(_unmappable_message(source_file.unmappable_chars))

    source_file.pages_without_text = len(unread)
    source_file.status = status
    source_file.status_detail = " ".join(n for n in notes if n).strip()
    source_file.extracted_at = timezone.now()
    source_file.save(
        update_fields=[
            "pages_without_text",
            "unmappable_chars",
            "status",
            "status_detail",
            "extracted_at",
        ]
    )
    return source_file


def _reasons_phrase(reasons_read: dict) -> str:
    """"3 image-only, 2 mixed" — why the OCR'd pages needed OCR."""
    words = {
        ExtractedPage.OCRReason.IMAGE_ONLY: "no text layer",
        ExtractedPage.OCRReason.MIXED: "body content in an image",
        ExtractedPage.OCRReason.DEFECTIVE_FONT: "defective embedded fonts",
    }
    parts = [
        f"{count} with {words.get(reason, reason)}"
        for reason, count in sorted(reasons_read.items())
        if count
    ]
    return ", ".join(parts)


def _record_failure(source_file: SourceFile, status: str, detail: str) -> SourceFile:
    source_file.status = status
    source_file.status_detail = detail
    source_file.page_count = 0
    source_file.pages_without_text = 0
    source_file.unmappable_chars = 0
    source_file.pages_from_ocr = 0
    source_file.ocr_engine = ""
    source_file.extracted_at = timezone.now()
    source_file.save(
        update_fields=[
            "status",
            "status_detail",
            "page_count",
            "pages_without_text",
            "unmappable_chars",
            "pages_from_ocr",
            "ocr_engine",
            "extracted_at",
        ]
    )
    return source_file
