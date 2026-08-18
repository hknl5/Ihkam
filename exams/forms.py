from decimal import Decimal, InvalidOperation

from django import forms
from django.forms import modelformset_factory

from .models import Blueprint, BlueprintRow, Exam, Question
from .services.blueprint import RowSpec, eligible_topics
from .services.export import ORDER_LABELS, ExportOptions
from .services.forms import MAX_FORMS


def _wants_multiple_forms(source, form=None) -> bool:
    """Does this data ask for more than one form?

    Reads whatever the caller has — posted strings, form initials, or an empty
    mapping — and falls back to the field's own default rather than assuming
    one. Unreadable input is treated as one form: an instructor mid-keystroke
    has not asked for a second paper.
    """
    raw = (source or {}).get("number_of_forms")
    if raw in (None, ""):
        if form is not None:
            raw = form.fields["number_of_forms"].initial
        if raw in (None, ""):
            raw = Exam._meta.get_field("number_of_forms").default
    try:
        return int(raw) > 1
    except (TypeError, ValueError):
        return False


class ExamForm(forms.ModelForm):
    """The exam specification — the numbers everything downstream is checked against."""

    class Meta:
        model = Exam
        fields = (
            "title",
            "kind",
            "total_score",
            "question_count",
            "duration_minutes",
            "language",
            "number_of_forms",
            "form_sharing",
            "sourcing",
            "bank_share_percent",
        )
        labels = {
            "title": "Name (optional)",
            "kind": "Type",
            "total_score": "Out of",
            "question_count": "Questions",
            "duration_minutes": "Duration (minutes)",
            "language": "Question language",
            "number_of_forms": "Forms",
            "form_sharing": "How the forms relate",
            "sourcing": "Where the questions come from",
            "bank_share_percent": "Share from the bank (%)",
        }
        help_texts = {
            "title": "For telling three quizzes apart. Left blank, the type is used.",
            "number_of_forms": f"Equivalent versions — A, B. At most {MAX_FORMS} for now.",
            "form_sharing": (
                "Fully separate needs twice the questions. If the pool cannot cover "
                "both papers, إحكام says so rather than quietly reusing a question."
            ),
            "sourcing": (
                "Questions you have approved and saved are in this course's bank. "
                "An exam can be written from scratch, drawn from the bank, or both."
            ),
            "bank_share_percent": (
                "One ratio for the whole paper. If the bank cannot cover that share, "
                "إحكام says by how much and writes the rest."
            ),
        }
        widgets = {
            "title": forms.TextInput(attrs={"placeholder": "Midterm — week 7"}),
            "form_sharing": forms.RadioSelect,
            "sourcing": forms.RadioSelect,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The field is asked for only when there is a second form to relate to.
        # It stays in `fields` either way, so a posted value is still validated
        # and stored — hiding a question is not the same as discarding it.
        self.fields["number_of_forms"].widget.attrs.update({"min": "1", "max": str(MAX_FORMS)})
        self.fields["form_sharing"].required = False
        # The share is only asked about in `mix`, and the two other modes
        # resolve it themselves (`Exam.bank_share`). Leaving it out of a posted
        # form is therefore not an error — see `clean_bank_share_percent`.
        self.fields["sourcing"].required = False
        self.fields["bank_share_percent"].required = False
        self.fields["bank_share_percent"].widget.attrs.update({"min": "0", "max": "100"})

    @property
    def show_sharing(self) -> bool:
        """Whether the sharing question applies to what is currently typed.

        Read by the template on first paint and by `exam_sharing_option` on
        every change of the form count, so the reveal is decided in one place
        rather than duplicated in JavaScript.
        """
        return _wants_multiple_forms(self.data if self.is_bound else self.initial, self)

    def clean_total_score(self):
        return self._positive("total_score", "An exam is out of at least one mark.")

    def clean_question_count(self):
        return self._positive("question_count", "An exam has at least one question.")

    def clean_duration_minutes(self):
        return self._positive("duration_minutes", "An exam lasts at least one minute.")

    def clean_number_of_forms(self):
        value = self._positive("number_of_forms", "There is at least one form.")
        if value is not None and value > MAX_FORMS:
            raise forms.ValidationError(
                f"إحكام builds at most {MAX_FORMS} forms from one blueprint. Two papers "
                f"already need twice the questions; a third has nothing to draw on yet."
            )
        return value

    def clean_form_sharing(self):
        """An unanswered sharing question means the stricter paper, not nothing.

        The field is hidden for a one-form exam and therefore posts empty. Empty
        is not a valid choice, so it becomes the default here rather than a
        validation error on a question the instructor was never asked.
        """
        return self.cleaned_data.get("form_sharing") or Exam.FormSharing.SEPARATE

    def clean_sourcing(self):
        """An unanswered sourcing question means a fully new exam.

        The same rule as `clean_form_sharing`: an instructor who was not asked
        has not chosen, and the safe reading of silence is the behaviour every
        milestone before M12 had.
        """
        return self.cleaned_data.get("sourcing") or Exam.Sourcing.NEW

    def clean_bank_share_percent(self):
        """A share is a percentage, and a blank one is the field's default.

        Kept in range here rather than trusted from the widget: the number
        decides how much of a paper is reused, and a 400% share would be a
        silent request for questions that do not exist.
        """
        value = self.cleaned_data.get("bank_share_percent")
        if value in (None, ""):
            return Exam._meta.get_field("bank_share_percent").default
        if value > 100:
            raise forms.ValidationError(
                "A share of the exam cannot be more than 100%."
            )
        return value

    def _positive(self, field, message):
        value = self.cleaned_data[field]
        if value is not None and value < 1:
            raise forms.ValidationError(message)
        return value


class BlueprintRowForm(forms.ModelForm):
    """One editable row of the blueprint table.

    The topic choices are the ones a blueprint may draw on — excluded topics and
    sub-topics of excluded chapters are not offered, so the instructor cannot
    build a row that the validator would then have to reject.
    """

    class Meta:
        model = BlueprintRow
        fields = ("topic", "question_type", "level", "count", "marks", "weight_percent")

    def __init__(self, *args, course=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.course = course
        if course is not None:
            choices = eligible_topics(course)
            # A row already sitting on a now-excluded topic keeps its option, so
            # the select still shows what the row says while the validator
            # explains why it has to change.
            current = self.instance.topic if self.instance and self.instance.pk else None
            if current and current not in choices:
                choices = [current, *choices]
            self.fields["topic"].queryset = (
                self.fields["topic"].queryset.filter(pk__in=[t.pk for t in choices])
            )
        for name in ("count", "marks", "weight_percent"):
            self.fields[name].widget.attrs.update({"min": "0", "inputmode": "decimal"})
        self.fields["count"].widget.attrs["step"] = "1"
        self.fields["marks"].widget.attrs["step"] = "0.5"
        self.fields["weight_percent"].widget.attrs["step"] = "0.01"


BlueprintRowFormSet = modelformset_factory(
    BlueprintRow,
    form=BlueprintRowForm,
    extra=1,
    can_delete=True,
)


def row_formset(course, blueprint: Blueprint | None, data=None):
    """The editor's formset, always scoped to one blueprint's rows."""
    queryset = (
        blueprint.rows.select_related("topic") if blueprint else BlueprintRow.objects.none()
    )
    return BlueprintRowFormSet(
        data,
        queryset=queryset,
        form_kwargs={"course": course},
        prefix="rows",
    )


# --- Reading the editor's table without saving it ----------------------------


def _decimal(raw, *, label, unreadable):
    """A posted number, or zero — with anything unreadable named, not ignored.

    A cell holding "12o" is not a zero. Counting it as one would make the totals
    disagree with what the instructor can see in the field, which is the one
    thing a live validator must never do.
    """
    raw = (raw or "").strip()
    if not raw:
        return Decimal("0")
    try:
        return Decimal(raw)
    except InvalidOperation:
        unreadable.append(label)
        return Decimal("0")


def specs_from_post(post, *, course):
    """The rows as they are being typed, ready for `validate`.

    Read straight from the posted table rather than from a bound formset,
    because live validation has to describe rows that are still half-filled —
    a formset would reject them before the arithmetic ever ran. Blank rows (the
    empty one always waiting at the bottom) are not rows; deleted ones are gone.

    Returns `(specs, unreadable)`.
    """
    topics = {topic.pk: topic for topic in course.topics.select_related("parent")}
    specs, unreadable = [], []
    try:
        total_forms = int(post.get("rows-TOTAL_FORMS") or 0)
    except ValueError:
        total_forms = 0

    for index in range(total_forms):
        prefix = f"rows-{index}-"
        if post.get(f"{prefix}DELETE"):
            continue
        raw_topic = (post.get(f"{prefix}topic") or "").strip()
        topic = topics.get(int(raw_topic)) if raw_topic.isdigit() else None
        if topic is None:
            continue

        human_row = index + 1
        count = _decimal(
            post.get(f"{prefix}count"), label=f"row {human_row}'s question count", unreadable=unreadable
        )
        specs.append(
            RowSpec(
                topic_id=topic.pk,
                topic_name=topic.name,
                count=int(count),
                marks=_decimal(
                    post.get(f"{prefix}marks"), label=f"row {human_row}'s marks", unreadable=unreadable
                ),
                weight_percent=_decimal(
                    post.get(f"{prefix}weight_percent"),
                    label=f"row {human_row}'s weight",
                    unreadable=unreadable,
                ),
                question_type=post.get(f"{prefix}question_type") or BlueprintRow.QuestionType.MCQ,
                level=post.get(f"{prefix}level") or BlueprintRow.Level.MEDIUM,
                topic_excluded=topic.excluded or bool(topic.parent and topic.parent.excluded),
            )
        )
    return specs, unreadable


# --- M11: the decision surface and the deliverable ---------------------------


class QuestionEditForm(forms.ModelForm):
    """The instructor editing a question by hand.

    Saving this sets the lock (`Question.mark_edited`) — the view does it rather
    than the form, so that the one place a lock is set is the one place a human
    pressed Save.

    `options` is edited as one option per line. A JSON textarea would be honest
    about the storage and useless to the person using it, and every stored shape
    here is a flat list of strings.
    """

    options_text = forms.CharField(
        label="Options",
        required=False,
        widget=forms.Textarea(attrs={"rows": 4, "dir": "auto"}),
        help_text="One option per line. Leave empty for short-answer and numeric questions.",
    )

    class Meta:
        model = Question
        fields = ("stem", "correct", "explanation")
        labels = {
            "stem": "Question",
            "correct": "Correct answer",
            "explanation": "Explanation",
        }
        widgets = {
            "stem": forms.Textarea(attrs={"rows": 4, "dir": "auto"}),
            "correct": forms.TextInput(attrs={"dir": "auto"}),
            "explanation": forms.Textarea(attrs={"rows": 2, "dir": "auto"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields["options_text"].initial = "\n".join(
                str(option) for option in (self.instance.options or [])
            )

    def clean(self):
        cleaned = super().clean()
        options = [
            line.strip()
            for line in (cleaned.get("options_text") or "").splitlines()
            if line.strip()
        ]
        correct = (cleaned.get("correct") or "").strip()
        # The one rule worth enforcing here: a multiple-choice question whose
        # correct answer is not one of its options cannot be marked. Everything
        # else about a question is the instructor's judgement, and this screen
        # does not second-guess it.
        if options and correct and correct not in options:
            self.add_error(
                "correct",
                "The correct answer has to be one of the options, word for word.",
            )
        cleaned["options"] = options
        return cleaned

    def save(self, commit=True):
        question = super().save(commit=False)
        question.options = self.cleaned_data.get("options", [])
        # A hand-edited key is out of scope for M11's card, but an edited stem
        # must not leave a stale objective key behind it pointing at an answer
        # that is no longer offered.
        key = question.answer_key if isinstance(question.answer_key, dict) else {}
        if key.get("kind") == "objective" and question.correct:
            key = {**key, "answer": question.correct, "options": question.options}
            question.answer_key = key
        if commit:
            question.save()
        return question


class ExportOptionsForm(forms.Form):
    """What goes on the paper. Every field is a choice the instructor makes."""

    form_id = forms.ChoiceField(label="Form", choices=())
    question_order = forms.ChoiceField(
        label="Question order",
        choices=(),
        initial=ExportOptions.Order.ASSEMBLED,
        widget=forms.RadioSelect,
    )
    institution = forms.CharField(
        label="Institution",
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "King Saud University", "dir": "auto"}),
    )
    logo = forms.ImageField(label="Institution logo", required=False)
    instructions = forms.CharField(
        label="Instructions to students",
        required=False,
        widget=forms.Textarea(attrs={"rows": 3, "dir": "auto"}),
    )
    show_course_data = forms.BooleanField(label="Course code and name", required=False, initial=True)
    show_duration = forms.BooleanField(label="Duration", required=False, initial=True)
    show_score_distribution = forms.BooleanField(
        label="Score distribution table", required=False, initial=True
    )
    show_marks_per_question = forms.BooleanField(
        label="Marks beside each question", required=False, initial=True
    )
    show_form_label = forms.BooleanField(label="Form letter", required=False, initial=True)

    def __init__(self, *args, forms_available=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["form_id"].choices = [
            (form.pk, f"Form {form.label} — {form.question_count} questions")
            for form in forms_available
        ]
        self.fields["question_order"].choices = list(ORDER_LABELS.items())

    def options(self, *, logo_path: str = "") -> ExportOptions:
        """The cleaned input as the frozen options object the exporters take."""
        data = self.cleaned_data
        return ExportOptions(
            show_form_label=data.get("show_form_label", True),
            question_order=data.get("question_order") or ExportOptions.Order.ASSEMBLED,
            institution=(data.get("institution") or "").strip(),
            logo_path=logo_path,
            show_course_data=data.get("show_course_data", True),
            show_duration=data.get("show_duration", True),
            show_score_distribution=data.get("show_score_distribution", True),
            instructions=(data.get("instructions") or "").strip(),
            show_marks_per_question=data.get("show_marks_per_question", True),
        )
