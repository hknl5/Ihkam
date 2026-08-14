from decimal import Decimal, InvalidOperation

from django import forms
from django.forms import modelformset_factory

from .models import Blueprint, BlueprintRow, Exam
from .services.blueprint import RowSpec, eligible_topics


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
        )
        labels = {
            "title": "Name (optional)",
            "kind": "Type",
            "total_score": "Out of",
            "question_count": "Questions",
            "duration_minutes": "Duration (minutes)",
            "language": "Question language",
            "number_of_forms": "Forms",
        }
        help_texts = {
            "title": "For telling three quizzes apart. Left blank, the type is used.",
            "number_of_forms": "Equivalent versions — A, B, … Balancing them comes later.",
        }
        widgets = {"title": forms.TextInput(attrs={"placeholder": "Midterm — week 7"})}

    def clean_total_score(self):
        return self._positive("total_score", "An exam is out of at least one mark.")

    def clean_question_count(self):
        return self._positive("question_count", "An exam has at least one question.")

    def clean_duration_minutes(self):
        return self._positive("duration_minutes", "An exam lasts at least one minute.")

    def clean_number_of_forms(self):
        return self._positive("number_of_forms", "There is at least one form.")

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
