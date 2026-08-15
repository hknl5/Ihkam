"""PDF rendering — the first implementation behind the export seam (M11).

ReportLab, because it is pure Python and already in the project: no headless
browser to install, no system libraries to match on a marker's laptop, and the
same output on every machine that runs the suite.

**Arabic is the hard part, and it is handled explicitly.** Open-source ReportLab
draws glyphs; it does not do Arabic shaping and it does not run the bidirectional
algorithm. Handed raw Arabic it produces disconnected letters in the wrong order
— readable by nobody. Two steps fix that, in this order:

1. **Wrap first, then shape.** The bidi algorithm returns text in *visual* order,
   and visual-order text cannot be line-wrapped: the wrap points are computed on
   a string whose direction has already been baked in, and every line after the
   first comes out scrambled. So a paragraph is measured and broken into lines
   while it is still in logical order, and only then is each line reshaped and
   reordered. This is why `flow_text` does its own line breaking instead of
   handing the string to a `Paragraph` and letting ReportLab wrap it.
2. **Shape, then reorder.** `arabic_reshaper` joins the letters into their
   initial/medial/final forms; `python-bidi` puts them in visual order and
   handles the numbers and Latin words embedded in them.

Latin text takes neither step and is wrapped by ReportLab as usual, which keeps
justification and hyphenation behaviour normal for English papers.

The font is IBM Plex Sans Arabic (SIL Open Font License, vendored in
`static/fonts/`): one family covering Arabic and Latin, which is what §5 already
specifies for the interface. Vendored rather than fetched, so an export works
offline and on a machine with no Arabic system font.

This module knows nothing about `Form`, `Question` or answer-key models. It
renders the documents `export.py` builds — which is what makes a Word or QTI
exporter an addition rather than a rewrite.
"""

from __future__ import annotations

import io
import os
import re
from decimal import Decimal

from django.conf import settings
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .export import AnswerKeyDocument, ExamDocument, Exporter, register_exporter

# --- Fonts -------------------------------------------------------------------

FONT_REGULAR = "IhkamSans"
FONT_BOLD = "IhkamSans-Bold"

#: Both weights of the one vendored family. §5's type roles collapse to two here
#: — a printed paper has no interface chrome to differentiate.
FONT_FILES = {
    FONT_REGULAR: "IBMPlexSansArabic-Regular.ttf",
    FONT_BOLD: "IBMPlexSansArabic-SemiBold.ttf",
}

_FONTS_REGISTERED = False


def font_dir() -> str:
    return os.path.join(settings.BASE_DIR, "static", "fonts")


def register_fonts() -> None:
    """Register the vendored family once per process.

    Falls back to Helvetica only if the files are missing, and says so in the
    log rather than silently producing a paper with no Arabic on it.
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    import logging

    logger = logging.getLogger(__name__)
    for name, filename in FONT_FILES.items():
        path = os.path.join(font_dir(), filename)
        if not os.path.exists(path):
            logger.error(
                "Font %s is missing from %s — Arabic will not render. "
                "Falling back to Helvetica.",
                filename,
                font_dir(),
            )
            return
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR, bold=FONT_BOLD)
    _FONTS_REGISTERED = True


def _font(bold: bool = False) -> str:
    register_fonts()
    if not _FONTS_REGISTERED:
        return "Helvetica-Bold" if bold else "Helvetica"
    return FONT_BOLD if bold else FONT_REGULAR


# --- Arabic: shaping, direction, and wrapping in that order ------------------

#: Arabic, Arabic Supplement, Extended-A, and the presentation forms.
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC.search(text or ""))


def shape(text: str) -> str:
    """Join the letters and put them in visual order. One line at a time.

    Never call this on a string you still intend to wrap — see the module
    docstring. It is applied to a line that has already been broken.
    """
    if not text:
        return ""
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def _escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def wrap_lines(text: str, *, font: str, size: float, width: float) -> list[str]:
    """Break logical-order text into lines that fit `width`. Greedy, by word.

    Deliberately simple: exam prose is short and this runs before any direction
    is applied, which is the only property that matters here.
    """
    lines: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        tokens = raw.split()
        if not tokens:
            lines.append("")
            continue
        current = tokens[0]
        for word in tokens[1:]:
            trial = f"{current} {word}"
            if pdfmetrics.stringWidth(trial, font, size) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def flow_text(text: str, style: ParagraphStyle, width: float) -> list:
    """Text as ReportLab flowables, correct in either direction.

    Latin goes straight to a `Paragraph` and is wrapped by ReportLab. Arabic is
    wrapped here first, then each line is shaped and reordered and becomes its
    own right-aligned `Paragraph` — the only order of operations that produces
    readable multi-line Arabic.
    """
    text = text or ""
    if not has_arabic(text):
        return [Paragraph(_escape(text).replace("\n", "<br/>"), style)]

    rtl = ParagraphStyle(
        f"{style.name}-rtl",
        parent=style,
        alignment=TA_RIGHT,
        spaceBefore=0,
        spaceAfter=0,
    )
    lines = wrap_lines(text, font=style.fontName, size=style.fontSize, width=width)
    flowables = [Paragraph(_escape(shape(line)) or "&nbsp;", rtl) for line in lines]
    if style.spaceAfter:
        flowables.append(Spacer(1, style.spaceAfter))
    return flowables


def _cell(text: str, style: ParagraphStyle) -> Paragraph:
    """One table cell — shaped whole, because a cell is a single short line."""
    text = str(text or "")
    if has_arabic(text):
        return Paragraph(
            _escape(shape(text)),
            ParagraphStyle(f"{style.name}-cell-rtl", parent=style, alignment=TA_RIGHT),
        )
    return Paragraph(_escape(text), style)


# --- Page furniture ----------------------------------------------------------

PAGE_MARGIN = 18 * mm
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN

#: The words the *paper* supplies, in the language the questions are written in.
#: Not the interface's language and not the course material's — a student reading
#: an Arabic paper should not meet the word "Instructions" on it, and a marker
#: reading an Arabic key should not meet "Final answer".
#:
#: This is a small fixed vocabulary rather than Django's i18n machinery on
#: purpose: it is eight labels on one artefact, the interface itself is not
#: translated yet, and a `.po` file would imply a localisation story this project
#: does not have.
CHROME = {
    "en": {
        "answer_key": "Answer key — not for distribution",
        "correct_answer": "Correct answer",
        "duration": "Duration: {minutes} minutes",
        "final_answer": "Final answer",
        "form": "Form {label}",
        "instructions": "Instructions",
        "marks": "({marks} marks)",
        "mark_sum_warning": (
            "Check: the steps in this key do not add up to the marks the question carries."
        ),
        "model_answer": "Model answer",
        "must_contain": "The answer must contain",
        "page": "Page {page}",
        "questions": "Questions: {count}",
        "score_distribution": "Score distribution",
        "source": "Source",
        "table_head": ("Topic", "Questions", "Marks", "Share"),
        "total_marks": "Total marks: {marks}",
        "worked_solution": "Worked solution",
    },
    "ar": {
        "answer_key": "نموذج الإجابة — غير مخصص للتوزيع",
        "correct_answer": "الإجابة الصحيحة",
        "duration": "المدة: {minutes} دقيقة",
        "final_answer": "الإجابة النهائية",
        "form": "النموذج {label}",
        "instructions": "التعليمات",
        "marks": "({marks} درجات)",
        "mark_sum_warning": "تنبيه: مجموع درجات خطوات الحل لا يساوي درجة السؤال.",
        "model_answer": "الإجابة النموذجية",
        "must_contain": "يجب أن تتضمن الإجابة",
        "page": "صفحة {page}",
        "questions": "عدد الأسئلة: {count}",
        "score_distribution": "توزيع الدرجات",
        "source": "المصدر",
        "table_head": ("الموضوع", "الأسئلة", "الدرجات", "النسبة"),
        "total_marks": "الدرجة الكلية: {marks}",
        "worked_solution": "خطوات الحل",
    },
}


def words(language: str) -> dict:
    """The paper's own vocabulary. Falls back to English for any other language."""
    return CHROME.get((language or "en").lower(), CHROME["en"])


#: Languages whose papers are laid out right to left. A list rather than a test
#: on the content, because an Arabic paper with one English question on it is
#: still an Arabic paper and its table columns still run right to left.
RTL_LANGUAGES = frozenset({"ar"})


def is_rtl(language: str) -> bool:
    return (language or "en").lower() in RTL_LANGUAGES


def _script_of(text: str) -> str:
    """Which vocabulary a single line should borrow, judged by the line itself."""
    return "ar" if has_arabic(text) else "en"

#: §5's ink, carried onto paper. The accent is used once, on the rules that
#: separate the cover from the questions — a printed paper has no decisions to
#: signal, so colour does almost nothing here.
INK = colors.HexColor("#1C1D1B")
INK_SOFT = colors.HexColor("#4B4D48")
HAIRLINE = colors.HexColor("#DEDBD3")
ACCENT = colors.HexColor("#0E5E5A")
SUNKEN = colors.HexColor("#EFEDE8")


def styles() -> dict:
    base = _font()
    bold = _font(bold=True)
    return {
        "title": ParagraphStyle(
            "title", fontName=bold, fontSize=18, leading=22, textColor=INK, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=base, fontSize=11, leading=15, textColor=INK_SOFT, spaceAfter=2
        ),
        "meta": ParagraphStyle(
            "meta", fontName=base, fontSize=9.5, leading=13, textColor=INK_SOFT
        ),
        "section": ParagraphStyle(
            "section",
            fontName=bold,
            fontSize=11,
            leading=15,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "stem": ParagraphStyle(
            "stem", fontName=base, fontSize=11, leading=16, textColor=INK, spaceAfter=3
        ),
        "option": ParagraphStyle(
            "option",
            fontName=base,
            fontSize=10.5,
            leading=15,
            textColor=INK,
            leftIndent=10 * mm,
        ),
        "answer": ParagraphStyle(
            "answer", fontName=bold, fontSize=10.5, leading=15, textColor=ACCENT
        ),
        "detail": ParagraphStyle(
            "detail",
            fontName=base,
            fontSize=9.5,
            leading=13,
            textColor=INK_SOFT,
            leftIndent=6 * mm,
        ),
        "caption": ParagraphStyle(
            "caption", fontName=base, fontSize=8.5, leading=11, textColor=INK_SOFT
        ),
    }


def _footer(kind: str, cover, language: str = "en"):
    """Every page says which paper it is and where it is in it.

    An answer key that loses its footer is a loose sheet of answers, and the
    page number is how a marker knows the set is complete.
    """

    vocabulary = words(language)

    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(_font(), 8)
        canvas.setFillColor(INK_SOFT)
        left = " · ".join(
            part
            for part in (
                cover.course_code,
                _titled(cover, vocabulary),
                vocabulary["answer_key"] if kind == "key" else "",
            )
            if part
        )
        if has_arabic(left):
            left = shape(left)
        canvas.drawString(PAGE_MARGIN, 12 * mm, left[:110])
        page = vocabulary["page"].format(page=doc.page)
        canvas.drawRightString(
            A4[0] - PAGE_MARGIN, 12 * mm, shape(page) if has_arabic(page) else page
        )
        canvas.setStrokeColor(HAIRLINE)
        canvas.line(PAGE_MARGIN, 15 * mm, A4[0] - PAGE_MARGIN, 15 * mm)
        canvas.restoreState()

    return draw


def _titled(cover, vocabulary: dict) -> str:
    """The paper's heading: its title, and the form label in the paper's language."""
    parts = [cover.title]
    if cover.form_label:
        parts.append(vocabulary["form"].format(label=cover.form_label))
    return " — ".join(part for part in parts if part)


def _document(buffer, cover, kind: str, language: str = "en") -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=22 * mm,
        title=_titled(cover, words(language))
        + (f" — {words(language)['answer_key']}" if kind == "key" else ""),
        author=cover.institution or "Ihkam",
        subject=cover.course_name,
    )
    frame = Frame(
        PAGE_MARGIN,
        22 * mm,
        CONTENT_WIDTH,
        A4[1] - PAGE_MARGIN - 22 * mm,
        id="body",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [PageTemplate(id="page", frames=[frame], onPage=_footer(kind, cover, language))]
    )
    return doc


def _rule(color=HAIRLINE, thickness=0.6):
    table = Table([[""]], colWidths=[CONTENT_WIDTH], rowHeights=[thickness])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)]))
    return table


def cover_flowables(cover, style: dict, *, kind: str, language: str = "en") -> list:
    """The head of the paper: logo, titles, the meta line, instructions, marks."""
    out: list = []

    if cover.logo_path and os.path.exists(cover.logo_path):
        try:
            image = Image(cover.logo_path)
            ratio = image.imageHeight / image.imageWidth if image.imageWidth else 1
            image.drawWidth = min(40 * mm, CONTENT_WIDTH)
            image.drawHeight = image.drawWidth * ratio
            image.hAlign = "LEFT"
            out += [image, Spacer(1, 6)]
        except Exception:  # noqa: BLE001 — a bad logo must not lose the paper
            pass

    vocabulary = words(language)

    if cover.institution:
        out += flow_text(cover.institution, style["subtitle"], CONTENT_WIDTH)
    out += flow_text(_titled(cover, vocabulary), style["title"], CONTENT_WIDTH)
    if kind == "key":
        out += flow_text(vocabulary["answer_key"], style["subtitle"], CONTENT_WIDTH)

    meta = []
    if cover.course_code or cover.course_name:
        meta.append(" — ".join(part for part in (cover.course_code, cover.course_name) if part))
    if cover.duration_minutes:
        meta.append(vocabulary["duration"].format(minutes=cover.duration_minutes))
    meta.append(vocabulary["questions"].format(count=cover.question_count))
    meta.append(vocabulary["total_marks"].format(marks=_number(cover.total_marks)))
    if meta:
        out += flow_text(" · ".join(meta), style["meta"], CONTENT_WIDTH)

    out += [Spacer(1, 8), _rule(ACCENT, 1.2), Spacer(1, 8)]

    if cover.instructions:
        out += flow_text(vocabulary["instructions"], style["section"], CONTENT_WIDTH)
        out += flow_text(cover.instructions, style["stem"], CONTENT_WIDTH)
        out += [Spacer(1, 4)]

    if cover.distribution:
        out += flow_text(vocabulary["score_distribution"], style["section"], CONTENT_WIDTH)
        head = list(vocabulary["table_head"])
        rows = [[_cell(part, style["caption"]) for part in head]]
        for topic, count, marks, percent in cover.distribution:
            rows.append(
                [
                    _cell(topic, style["caption"]),
                    _cell(str(count), style["caption"]),
                    _cell(_number(marks), style["caption"]),
                    _cell(f"{percent}%", style["caption"]),
                ]
            )
        widths = [CONTENT_WIDTH - 75 * mm, 25 * mm, 25 * mm, 25 * mm]
        if is_rtl(language):
            # Mirror the whole table, not only the text inside it. A right-to-left
            # paper whose first column is on the left reads back-to-front even
            # when every cell in it is shaped correctly.
            rows = [list(reversed(row)) for row in rows]
            widths = list(reversed(widths))
        table = Table(rows, colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), SUNKEN),
                    ("GRID", (0, 0), (-1, -1), 0.4, HAIRLINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        out += [table, Spacer(1, 10)]

    return out


def _number(value) -> str:
    """Marks without trailing zeros: 3, 4.5 — never 3.00."""
    value = Decimal(value or 0)
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    return str(value.normalize())


def _numbered(number: int, text: str, style: ParagraphStyle, *, suffix: str = "") -> list:
    """A question, numbered. The number stays Latin in both directions.

    Exam numbering stays "1." on an Arabic paper too: that is what students are
    told to write on the answer sheet, and mirroring it would make the answer
    sheet and the paper disagree.

    The number is prepended to the *logical* first line and shaped with it, so
    bidi puts it at the reading start of the line — the right-hand edge on an
    Arabic paper — instead of leaving it stranded on a line of its own.
    """
    body = f"{text}{suffix}"
    if has_arabic(body):
        rtl = ParagraphStyle(
            f"{style.name}-n",
            parent=style,
            alignment=TA_RIGHT,
            spaceBefore=0,
            spaceAfter=0,
        )
        marker = f"{number}. "
        width = CONTENT_WIDTH - pdfmetrics.stringWidth(
            marker, style.fontName, style.fontSize
        )
        lines = wrap_lines(body, font=style.fontName, size=style.fontSize, width=width)
        lines = [f"{marker}{lines[0]}" if lines else marker, *lines[1:]]
        out = [Paragraph(_escape(shape(line)) or "&nbsp;", rtl) for line in lines]
        return [*out, Spacer(1, style.spaceAfter or 3)]
    marker = ParagraphStyle(
        f"{style.name}-num",
        parent=style,
        alignment=TA_LEFT,
        leftIndent=8 * mm,
        firstLineIndent=-8 * mm,
    )
    # Built here rather than through `flow_text`, which escapes: the number is
    # the one piece of markup on the line and it must survive as markup.
    return [Paragraph(f"<b>{number}.</b> {_escape(body)}".replace("\n", "<br/>"), marker)]


class PdfExporter(Exporter):
    """The exam and the key as two PDFs."""

    name = "pdf"
    extension = "pdf"
    content_type = "application/pdf"
    label = "PDF"

    def render_exam(self, document: ExamDocument) -> bytes:
        style = styles()
        vocabulary = words(document.language)
        story = cover_flowables(
            document.cover, style, kind="exam", language=document.language
        )

        for question in document.questions:
            # The marks follow the *stem's* script, not the paper's. A bilingual
            # course puts English questions on an Arabic paper, and appending an
            # Arabic phrase to a Latin sentence makes one bidi-mixed line whose
            # brackets mirror the wrong way. Each question is internally
            # consistent instead.
            suffix = (
                "  " + words(_script_of(question.stem)).get(
                    "marks", vocabulary["marks"]
                ).format(marks=_number(question.marks))
                if question.marks
                else ""
            )
            block: list = _numbered(question.number, question.stem, style["stem"], suffix=suffix)
            for letter, option in question.option_labels:
                block += flow_text(f"{letter}. {option}", style["option"], CONTENT_WIDTH - 10 * mm)
            block.append(Spacer(1, 8))
            # A question and its options stay on one page: an MCQ split across a
            # page break is a question a student answers from half its options.
            story.append(KeepTogether(block))

        return _build(story, document.cover, "exam", document.language)

    def render_key(self, document: AnswerKeyDocument) -> bytes:
        style = styles()
        vocabulary = words(document.language)
        story = cover_flowables(
            document.cover, style, kind="key", language=document.language
        )

        for answer in document.answers:
            block: list = _numbered(answer.number, answer.stem, style["stem"])

            if answer.kind == "numeric" and answer.steps:
                block += flow_text(
                    vocabulary["worked_solution"], style["section"], CONTENT_WIDTH
                )
                for index, (text, marks) in enumerate(answer.steps, start=1):
                    block += flow_text(
                        f"{index}. [{_number(marks)} marks] {text}",
                        style["detail"],
                        CONTENT_WIDTH - 6 * mm,
                    )
                block += flow_text(
                    f"{vocabulary['final_answer']}: {answer.answer}",
                    style["answer"],
                    CONTENT_WIDTH,
                )
            elif answer.kind == "short_answer":
                block += flow_text(
                    f"{vocabulary['model_answer']}: {answer.answer}",
                    style["answer"],
                    CONTENT_WIDTH,
                )
                if answer.required_elements:
                    block += flow_text(
                        vocabulary["must_contain"], style["section"], CONTENT_WIDTH
                    )
                    for element in answer.required_elements:
                        block += flow_text(
                            f"— {element}", style["detail"], CONTENT_WIDTH - 6 * mm
                        )
            else:
                block += flow_text(
                    f"{vocabulary['correct_answer']}: {answer.answer}",
                    style["answer"],
                    CONTENT_WIDTH,
                )

            if answer.explanation:
                block += flow_text(answer.explanation, style["detail"], CONTENT_WIDTH - 6 * mm)
            if answer.needs_mark_review:
                block += flow_text(
                    vocabulary["mark_sum_warning"], style["detail"], CONTENT_WIDTH - 6 * mm
                )
            if answer.source_ref:
                block += flow_text(
                    f"{vocabulary['source']}: {answer.source_ref}",
                    style["caption"],
                    CONTENT_WIDTH,
                )
            block.append(Spacer(1, 10))
            story.append(KeepTogether(block))

        return _build(story, document.cover, "key", document.language)


def _build(story: list, cover, kind: str, language: str = "en") -> bytes:
    buffer = io.BytesIO()
    _document(buffer, cover, kind, language).build(story)
    return buffer.getvalue()


register_exporter(PdfExporter())


__all__ = [
    "FONT_BOLD",
    "FONT_REGULAR",
    "PdfExporter",
    "flow_text",
    "has_arabic",
    "register_fonts",
    "is_rtl",
    "shape",
    "words",
    "wrap_lines",
]
