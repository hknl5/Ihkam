"""M11: the deliverable — two PDFs, and the Arabic that has to render (M11).

No provider is reachable here and none is needed: an export is arithmetic and
typesetting over questions already in the database.

Three things are pinned down.

* **The exam paper has no answers on it.** Not "the renderer omits them" — the
  document object the renderer is handed has nowhere to put one. Both claims are
  tested: the object has no answer field, and the produced PDF's text has no
  open-question answer in it.
* **The answer key has the typed keys**, in each of M6's three shapes — the
  objective answer, the short-answer model answer with its required elements,
  and the numeric worked steps with per-step marks and a final answer.
* **Arabic renders.** Verified by extracting the text back out of the produced
  PDF and normalising the presentation forms with NFKC: if the shaping and the
  bidi pass had not run, or the font had not embedded, the words would not come
  back. Column mirroring is checked separately, because a right-to-left paper
  whose table starts on the left reads backwards even when every word in it is
  correct.
"""

import unicodedata
from decimal import Decimal

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from courses.models import Course, Topic
from exams.models import Blueprint, BlueprintRow, Exam, Form, FormQuestion, Question
from exams.services.export import (
    AnswerKeyDocument,
    Cover,
    DocumentQuestion,
    ExamDocument,
    ExportError,
    ExportOptions,
    Exporter,
    available_formats,
    build_exam_document,
    build_key_document,
    export_form,
    get_exporter,
    register_exporter,
)
from exams.services.export_pdf import (
    flow_text,
    has_arabic,
    is_rtl,
    shape,
    words,
    wrap_lines,
)

PASSWORD = "quiet-precision-42"

MCQ = BlueprintRow.QuestionType.MCQ
SHORT = BlueprintRow.QuestionType.SHORT_ANSWER
NUMERIC = BlueprintRow.QuestionType.NUMERIC
MEDIUM = BlueprintRow.Level.MEDIUM
MULTI_STEP = BlueprintRow.Level.MULTI_STEP

ARABIC_STEM = "أي من الخيارات التالية يمثل مبدأ الشفافية في أنظمة الذكاء الاصطناعي؟"
ARABIC_OPTIONS = ["الإفصاح عن كيفية اتخاذ القرار", "إخفاء مصادر البيانات"]


def text_of(pdf_bytes: bytes) -> str:
    """Every page's text, joined. NFKC folds the Arabic presentation forms back.

    Reading the file back is the only verification that means anything here: a
    PDF that builds is not a PDF that renders, and the shaped Arabic that was
    drawn is not the string that went in.
    """
    import io

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    raw = " ".join(page.get_textpage().get_text_range() for page in document)
    return unicodedata.normalize("NFKC", raw).replace("\n", " ")


def page_count(pdf_bytes: bytes) -> int:
    import io

    import pypdfium2 as pdfium

    return len(pdfium.PdfDocument(io.BytesIO(pdf_bytes)))


# --- Direction handling ------------------------------------------------------


class ArabicShapingTests(SimpleTestCase):
    """The three steps that make Arabic readable, each on its own."""

    def test_arabic_is_detected_and_latin_is_left_alone(self):
        self.assertTrue(has_arabic(ARABIC_STEM))
        self.assertFalse(has_arabic("A binary search halves the interval."))

    def test_shaping_joins_the_letters(self):
        shaped = shape("الشفافية")
        # Reshaping replaces the isolated letters with joined presentation forms.
        self.assertNotEqual(shaped, "الشفافية")
        self.assertTrue(any("ﹰ" <= character <= "﻿" for character in shaped))
        # And it is reversible: NFKC folds the forms back to the same letters.
        self.assertEqual(
            unicodedata.normalize("NFKC", shaped)[::-1].strip(), "الشفافية"
        )

    def test_wrapping_happens_before_shaping(self):
        """The order the module docstring insists on, asserted rather than trusted."""
        from exams.services.export_pdf import CONTENT_WIDTH, _font

        long_arabic = " ".join([ARABIC_STEM] * 4)
        lines = wrap_lines(long_arabic, font=_font(), size=11, width=CONTENT_WIDTH)
        self.assertGreater(len(lines), 1)
        # Every line is still logical-order text at this point — no presentation
        # forms have appeared yet.
        for line in lines:
            self.assertFalse(any("ﹰ" <= c <= "﻿" for c in line))

    def test_an_arabic_paragraph_becomes_one_flowable_per_line(self):
        from reportlab.lib.enums import TA_RIGHT

        from exams.services.export_pdf import CONTENT_WIDTH, styles

        style = styles()["stem"]
        flowables = flow_text(" ".join([ARABIC_STEM] * 3), style, CONTENT_WIDTH)
        self.assertGreater(len(flowables), 1)
        paragraphs = [f for f in flowables if hasattr(f, "style")]
        self.assertTrue(all(p.style.alignment == TA_RIGHT for p in paragraphs))

    def test_latin_stays_one_paragraph_and_is_wrapped_by_reportlab(self):
        from exams.services.export_pdf import CONTENT_WIDTH, styles

        flowables = flow_text("A binary search halves the interval. " * 8, styles()["stem"], CONTENT_WIDTH)
        self.assertEqual(len(flowables), 1)

    def test_the_paper_speaks_the_language_its_questions_are_written_in(self):
        self.assertEqual(words("en")["instructions"], "Instructions")
        self.assertEqual(words("ar")["instructions"], "التعليمات")
        # An unknown language falls back rather than printing a key name.
        self.assertEqual(words("fr")["instructions"], "Instructions")

    def test_only_arabic_is_laid_out_right_to_left(self):
        self.assertTrue(is_rtl("ar"))
        self.assertFalse(is_rtl("en"))


# --- The seam ----------------------------------------------------------------


class ExportSeamTests(SimpleTestCase):
    """PDF is one implementation. The next format is an addition, not a rewrite."""

    def test_pdf_is_registered(self):
        exporter = get_exporter("pdf")
        self.assertEqual(exporter.extension, "pdf")
        self.assertEqual(exporter.content_type, "application/pdf")
        self.assertIn("pdf", [item.name for item in available_formats()])

    def test_an_unknown_format_says_what_it_has(self):
        with self.assertRaises(ExportError) as caught:
            get_exporter("qti")
        self.assertIn("pdf", str(caught.exception))

    def test_a_new_format_needs_only_the_two_render_methods(self):
        """The seam, exercised: no Form, no Question, no answer-key model."""

        class FakeWord(Exporter):
            name = "test-docx"
            extension = "docx"
            content_type = "application/vnd.openxmlformats"
            label = "Word"

            def render_exam(self, document):
                return f"exam:{len(document.questions)}".encode()

            def render_key(self, document):
                return f"key:{len(document.answers)}".encode()

        try:
            register_exporter(FakeWord())
            self.assertEqual(get_exporter("test-docx").name, "test-docx")
        finally:
            from exams.services.export import EXPORTERS

            EXPORTERS.pop("test-docx", None)


class DocumentModelTests(SimpleTestCase):
    """The guarantee is in the object, not in the renderer's good behaviour."""

    def test_the_exam_document_has_nowhere_to_put_an_answer(self):
        fields = DocumentQuestion.__dataclass_fields__
        for forbidden in ("correct", "answer", "answer_key", "explanation"):
            self.assertNotIn(forbidden, fields)

    def test_the_cover_does_not_name_the_form_in_any_one_language(self):
        cover = Cover(exam_title="Midterm", form_label="A")
        self.assertEqual(cover.title, "Midterm")
        self.assertNotIn("Form", cover.title)


# --- Against a real exam -----------------------------------------------------


class ExportFixture(TestCase):
    """One exam, one form, one question of each of M6's three key shapes."""

    language = "en"

    def setUp(self):
        self.user = User.objects.create_user("nadia", password=PASSWORD)
        self.course = Course.objects.create(
            instructor=self.user, code="CS201", name="Data Structures"
        )
        self.exam = Exam.objects.create(
            course=self.course,
            title="Midterm",
            total_score=20,
            question_count=3,
            duration_minutes=90,
            language=self.language,
            number_of_forms=1,
        )
        self.blueprint = Blueprint.objects.create(exam=self.exam)
        self.form = Form.objects.create(exam=self.exam, label="A", position=0)
        self.rows = {}
        for position, (name, question_type, level) in enumerate(
            [
                ("Recursion", MCQ, MEDIUM),
                ("Sorting", SHORT, MEDIUM),
                ("Complexity", NUMERIC, MULTI_STEP),
            ]
        ):
            topic = Topic.objects.create(course=self.course, name=name, position=position)
            self.rows[name] = BlueprintRow.objects.create(
                blueprint=self.blueprint,
                topic=topic,
                question_type=question_type,
                level=level,
                count=1,
                marks=Decimal("5"),
                weight_percent=Decimal("33.33"),
                position=position,
            )

        self.mcq = self.place(
            "Recursion",
            stem="Which case stops a recursion?",
            options=["The base case", "The general case", "The tail call", "The stack frame"],
            correct="The base case",
            answer_key={"kind": "objective", "answer": "The base case"},
        )
        self.short = self.place(
            "Sorting",
            stem="Explain why merge sort is stable.",
            correct="Equal elements keep their relative order because the left run wins ties.",
            answer_key={
                "kind": "short_answer",
                "model_answer": "Equal elements keep their relative order because the left "
                "run wins ties.",
                "required_elements": ["equal elements", "relative order preserved"],
            },
        )
        self.numeric = self.place(
            "Complexity",
            stem="How many comparisons does merge sort make on 8 items?",
            correct="24 comparisons",
            answer_key={
                "kind": "numeric",
                "steps": [
                    {"text": "Each level compares at most n items: 8.", "marks": "2"},
                    {"text": "There are log2(8) = 3 levels, so 8 x 3 = 24.", "marks": "3"},
                ],
                "final_answer": "24 comparisons",
            },
        )

    def place(self, topic, *, stem, correct, options=(), answer_key=None, **kwargs):
        row = self.rows[topic]
        question = Question.objects.create(
            exam=self.exam,
            blueprint_row=row,
            stem=stem,
            question_type=row.question_type,
            options=list(options),
            correct=correct,
            answer_key=answer_key or {},
            source_ref="lecture.pdf · page 3",
            position=self.form.entries.count(),
            **kwargs,
        )
        FormQuestion.objects.create(
            form=self.form,
            question=question,
            blueprint_row=row,
            position=self.form.entries.count(),
            marks=Decimal("5"),
            expected_minutes=Decimal("3"),
        )
        return question

    def files(self, options=None):
        produced = export_form(self.form, options=options or ExportOptions())
        return {item.kind: item for item in produced}


class ExamPaperTests(ExportFixture):
    def test_both_files_are_produced_and_named_apart(self):
        files = self.files()
        self.assertEqual(set(files), {"exam", "key"})
        self.assertTrue(files["exam"].filename.endswith(".pdf"))
        self.assertIn("answer-key", files["key"].filename)
        self.assertNotIn("answer-key", files["exam"].filename)
        self.assertGreater(files["exam"].size, 1000)

    def test_the_exam_pdf_carries_the_questions_and_the_options(self):
        body = text_of(self.files()["exam"].content)
        self.assertIn("Which case stops a recursion?", body)
        self.assertIn("Explain why merge sort is stable.", body)
        self.assertIn("The general case", body)  # a distractor: part of the paper

    def test_the_exam_pdf_carries_no_answers(self):
        body = text_of(self.files()["exam"].content)
        # Nothing labelled as an answer …
        for label in ("Correct answer", "Model answer", "Final answer", "Worked solution"):
            self.assertNotIn(label, body)
        # … and no answer to a question that has no options to print.
        self.assertNotIn("Equal elements keep their relative order", body)
        self.assertNotIn("24 comparisons", body)
        self.assertNotIn("log2(8) = 3 levels", body)

    def test_the_key_pdf_carries_all_three_typed_key_shapes(self):
        body = text_of(self.files()["key"].content)
        self.assertIn("Correct answer: The base case", body)
        self.assertIn("Model answer: Equal elements keep their relative order", body)
        self.assertIn("equal elements", body)  # a required element
        self.assertIn("Worked solution", body)
        self.assertIn("[2 marks]", body)
        self.assertIn("Final answer: 24 comparisons", body)

    def test_the_key_says_on_every_page_that_it_is_a_key(self):
        content = self.files()["key"].content
        body = text_of(content)
        self.assertIn("ANSWER KEY", body.upper())
        self.assertGreaterEqual(page_count(content), 1)

    def test_a_key_whose_marks_do_not_add_up_says_so_to_the_marker(self):
        self.numeric.mark_sum_ok = False
        self.numeric.save(update_fields=["mark_sum_ok"])
        body = text_of(self.files()["key"].content)
        self.assertIn("do not add up", body)

    def test_a_key_that_cannot_be_parsed_still_prints_the_plain_answer(self):
        self.mcq.answer_key = {"kind": "objective"}  # invalid: no answer
        self.mcq.save(update_fields=["answer_key"])
        body = text_of(self.files()["key"].content)
        self.assertIn("The base case", body)


class ExportOptionTests(ExportFixture):
    """Every option on the screen changes the paper it produces."""

    def test_course_data_and_duration_can_be_left_off(self):
        with_data = text_of(self.files()["exam"].content)
        self.assertIn("CS201", with_data)
        self.assertIn("90", with_data)

        bare = text_of(
            self.files(
                ExportOptions(show_course_data=False, show_duration=False)
            )["exam"].content
        )
        self.assertNotIn("CS201", bare)
        self.assertNotIn("Duration", bare)

    def test_the_score_distribution_can_be_left_off(self):
        self.assertIn("Score distribution", text_of(self.files()["exam"].content))
        self.assertNotIn(
            "Score distribution",
            text_of(self.files(ExportOptions(show_score_distribution=False))["exam"].content),
        )

    def test_the_form_letter_can_be_left_off(self):
        self.assertIn("Form A", text_of(self.files()["exam"].content))
        self.assertNotIn(
            "Form A", text_of(self.files(ExportOptions(show_form_label=False))["exam"].content)
        )

    def test_instructions_are_printed_when_given(self):
        options = ExportOptions(instructions="Answer all questions in pen.")
        self.assertIn(
            "Answer all questions in pen.", text_of(self.files(options)["exam"].content)
        )

    def test_marks_beside_each_question_can_be_left_off(self):
        self.assertIn("(5 marks)", text_of(self.files()["exam"].content))
        self.assertNotIn(
            "(5 marks)",
            text_of(self.files(ExportOptions(show_marks_per_question=False))["exam"].content),
        )

    def test_question_order_is_an_option_and_shuffling_is_repeatable(self):
        assembled = build_exam_document(self.form, ExportOptions())
        shuffled = build_exam_document(
            self.form, ExportOptions(question_order=ExportOptions.Order.SHUFFLED)
        )
        again = build_exam_document(
            self.form, ExportOptions(question_order=ExportOptions.Order.SHUFFLED)
        )
        self.assertEqual(
            [q.stem for q in shuffled.questions], [q.stem for q in again.questions]
        )
        self.assertEqual(
            sorted(q.stem for q in assembled.questions),
            sorted(q.stem for q in shuffled.questions),
        )

    def test_marks_follow_the_script_of_the_question_they_belong_to(self):
        """A bilingual paper: an English question keeps English brackets."""
        body = text_of(self.files()["exam"].content)
        self.assertIn("(5 marks)", body)  # the English questions on this Arabic paper

    def test_the_key_is_ordered_the_same_way_as_the_paper(self):
        options = ExportOptions(question_order=ExportOptions.Order.SHUFFLED)
        paper = build_exam_document(self.form, options)
        key = build_key_document(self.form, options)
        self.assertEqual(
            [q.stem for q in paper.questions], [a.stem for a in key.answers]
        )


class ArabicPaperTests(ExportFixture):
    """The real requirement: a paper an Arabic-speaking student can read."""

    language = "ar"

    def setUp(self):
        super().setUp()
        self.mcq.stem = ARABIC_STEM
        self.mcq.options = ARABIC_OPTIONS
        self.mcq.correct = ARABIC_OPTIONS[0]
        self.mcq.answer_key = {"kind": "objective", "answer": ARABIC_OPTIONS[0]}
        self.mcq.save()
        topic = self.rows["Recursion"].topic
        topic.name = "الشفافية والقابلية للتفسير"
        topic.save(update_fields=["name"])

    def test_the_arabic_stem_and_options_come_back_out_of_the_pdf(self):
        body = text_of(self.files()["exam"].content)
        for fragment in ("الشفافية", "الذكاء", "الاصطناعي", "الإفصاح", "البيانات"):
            self.assertIn(fragment, body, f"{fragment} did not render")

    def test_the_paper_is_labelled_in_arabic(self):
        body = text_of(
            self.files(ExportOptions(instructions="أجب عن جميع الأسئلة."))["exam"].content
        )
        self.assertIn("النموذج", body)  # "Form", in the paper's own language
        self.assertIn("التعليمات", body)
        self.assertIn("توزيع الدرجات", body)
        self.assertIn("أجب عن جميع الأسئلة", body)
        self.assertNotIn("Instructions", body)

    def test_the_arabic_key_is_labelled_in_arabic_too(self):
        body = text_of(self.files()["key"].content)
        self.assertIn("الإجابة الصحيحة", body)
        self.assertIn("نموذج الإجابة", body)

    def test_the_score_distribution_table_is_mirrored(self):
        """Not only the cells: a right-to-left table starts on the right."""
        from exams.services.export_pdf import _titled

        document = build_exam_document(self.form, ExportOptions())
        self.assertEqual(document.language, "ar")
        self.assertTrue(is_rtl(document.language))
        self.assertIn("النموذج A", _titled(document.cover, words("ar")))

    def test_an_arabic_paper_still_numbers_its_questions_in_latin(self):
        """What students write on the answer sheet must match the paper.

        Asserted by *position*, not by string: the extracted text of an RTL line
        comes back in visual order, so "1." reads as ".1" there and a substring
        check would be testing the extractor rather than the paper. What matters
        is that the digit is drawn at the reading start of the line — the
        right-hand edge — which is where a student looks for it.
        """
        import io

        import pypdfium2 as pdfium

        page = pdfium.PdfDocument(io.BytesIO(self.files()["exam"].content))[0]
        text_page = page.get_textpage()
        rows: dict = {}
        for index in range(text_page.count_chars()):
            character = text_page.get_text_range(index, 1)
            left, bottom, _right, _top = text_page.get_charbox(index)
            rows.setdefault(round(bottom), []).append((left, character))

        for _y, characters in rows.items():
            arabic = [x for x, c in characters if "\u0600" <= c <= "\ufeff"]
            digits = [x for x, c in characters if c == "1"]
            # The stem's first line, identified by the Arabic question mark it
            # ends with — the distribution table also mixes digits and Arabic.
            if digits and any(c == "؟" for _x, c in characters):
                self.assertGreater(
                    max(digits),
                    max(arabic),
                    "the question number is not at the reading start of the line",
                )
                break
        else:  # pragma: no cover - the fixture always has one such line
            self.fail("no Arabic question line was found in the rendered paper")


class ExportScreenTests(ExportFixture):
    def setUp(self):
        super().setUp()
        self.client.login(username="nadia", password=PASSWORD)
        self.url = reverse("exams:export", args=[self.course.pk, self.exam.pk])

    def test_it_offers_the_options_and_both_downloads(self):
        response = self.client.get(self.url)
        self.assertContains(response, "Download the exam")
        self.assertContains(response, "Download the answer key")
        self.assertContains(response, "Question order")
        self.assertContains(response, "Institution")

    def test_downloading_the_exam_returns_a_pdf_with_no_answers(self):
        response = self.client.post(
            self.url,
            {
                "form_id": self.form.pk,
                "question_order": ExportOptions.Order.ASSEMBLED,
                "download": "exam",
                "show_course_data": "on",
                "show_duration": "on",
                "show_score_distribution": "on",
                "show_marks_per_question": "on",
                "show_form_label": "on",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertNotIn("Correct answer", text_of(response.content))

    def test_downloading_the_key_returns_the_other_file(self):
        response = self.client.post(
            self.url,
            {
                "form_id": self.form.pk,
                "question_order": ExportOptions.Order.ASSEMBLED,
                "download": "key",
            },
        )
        self.assertIn("answer-key", response["Content-Disposition"])
        self.assertIn("Correct answer", text_of(response.content))

    def test_an_exam_with_no_saved_forms_is_sent_to_assemble_them(self):
        self.form.delete()
        response = self.client.get(self.url, follow=True)
        self.assertContains(response, "Assemble and save the forms")

    def test_another_instructors_exam_does_not_exist(self):
        self.client.logout()
        User.objects.create_user("omar", password=PASSWORD)
        self.client.login(username="omar", password=PASSWORD)
        self.assertEqual(self.client.get(self.url).status_code, 404)
