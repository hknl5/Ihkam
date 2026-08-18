"""The two small forms the bank needs (M12).

`SourcingForm` is the sourcing choice on its own, so the Generate screen can
change it without re-posting the whole exam spec. It edits the same two fields
`ExamForm` does — one definition of the choice, two places to make it, because
an instructor decides where questions come from when they specify the exam and
again when they are standing in front of the button.

`BankFilterForm` is the browse screen's filter and search box. Its `query` field
is what gets embedded; everything else is a plain database filter.
"""

from django import forms

from exams.models import BlueprintRow, Exam


class SourcingForm(forms.ModelForm):
    """Where this exam's questions come from — asked again at Generate."""

    class Meta:
        model = Exam
        fields = ("sourcing", "bank_share_percent")
        labels = {
            "sourcing": "Where the questions come from",
            "bank_share_percent": "Share from the bank (%)",
        }
        widgets = {"sourcing": forms.RadioSelect}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["sourcing"].required = False
        self.fields["bank_share_percent"].required = False
        self.fields["bank_share_percent"].widget.attrs.update({"min": "0", "max": "100"})

    def clean_sourcing(self):
        return self.cleaned_data.get("sourcing") or Exam.Sourcing.NEW

    def clean_bank_share_percent(self):
        value = self.cleaned_data.get("bank_share_percent")
        if value in (None, ""):
            return Exam._meta.get_field("bank_share_percent").default
        if value > 100:
            raise forms.ValidationError("A share of the exam cannot be more than 100%.")
        return value


class BankFilterForm(forms.Form):
    """Search and filter the bank. Nothing here writes, and only `query` costs.

    Bound to `request.GET` for the filters and to the search POST for the query,
    so a filtered browse is a link an instructor can bookmark while the search —
    which spends one embedding call — is a press.
    """

    query = forms.CharField(
        required=False,
        label="Search the bank",
        help_text=(
            "Searched by meaning, with the same embeddings the rest of إحكام uses — "
            "so “halving the interval” finds the question about binary search."
        ),
        widget=forms.TextInput(attrs={"placeholder": "What is the question about?"}),
    )
    question_type = forms.ChoiceField(
        required=False,
        label="Type",
        choices=[("", "Any type"), *BlueprintRow.QuestionType.choices],
    )
    level = forms.ChoiceField(
        required=False,
        label="Level",
        choices=[("", "Any level"), *BlueprintRow.Level.choices],
    )

    def filters(self) -> dict:
        """The filter arguments, whatever the form was bound to."""
        data = self.cleaned_data if self.is_valid() else {}
        return {
            "question_type": data.get("question_type") or "",
            "level": data.get("level") or "",
        }

    @property
    def query_text(self) -> str:
        return (self.cleaned_data.get("query") or "").strip() if self.is_valid() else ""
