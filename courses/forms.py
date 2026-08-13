from django import forms

from .models import Course, SourceFile, Topic

#: Upload ceiling. Lecture decks are big; scanned books are not our problem yet.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ("name", "code", "level", "content_language")
        labels = {"name": "Course name", "code": "Course code"}
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Introduction to Machine Learning"}),
            "code": forms.TextInput(attrs={"placeholder": "CS310"}),
        }

    def __init__(self, *args, instructor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.instructor = instructor

    def clean_code(self):
        code = self.cleaned_data["code"].strip().upper()
        if self.instructor and Course.objects.filter(
            instructor=self.instructor, code__iexact=code
        ).exists():
            raise forms.ValidationError("You already have a course with this code.")
        return code

    def save(self, commit=True):
        course = super().save(commit=False)
        if self.instructor:
            course.instructor = self.instructor
        if commit:
            course.save()
        return course


class SourceFileUploadForm(forms.ModelForm):
    file = forms.FileField(
        label="Lecture file",
        help_text="PDF is extracted now. PowerPoint, Word and plain text are accepted "
        "but not extracted until a later milestone.",
    )

    class Meta:
        model = SourceFile
        fields = ("file",)

    def clean_file(self):
        upload = self.cleaned_data["file"]
        kind = SourceFile.kind_for_filename(upload.name)
        if kind is None:
            raise forms.ValidationError(
                "Unsupported file type. Upload a PDF, PowerPoint, Word or text file."
            )
        if upload.size > MAX_UPLOAD_BYTES:
            raise forms.ValidationError(
                f"This file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
        self.detected_kind = kind
        return upload

    def save(self, commit=True):
        source_file = super().save(commit=False)
        upload = self.cleaned_data["file"]
        source_file.original_name = upload.name
        source_file.kind = self.detected_kind
        source_file.size_bytes = upload.size
        if commit:
            source_file.save()
        return source_file


class TopicForm(forms.ModelForm):
    """Adding a topic by hand, on the review screen (M2).

    An instructor adding a topic is stating a fact about their own teaching, so
    only the name is required: they may well be adding something the material
    never covered, which is precisely why no page reference is demanded.
    """

    class Meta:
        model = Topic
        fields = ("name", "parent", "source_file", "page_start", "page_end")
        labels = {
            "name": "Topic name",
            "parent": "Chapter it belongs to",
            "source_file": "Found in",
            "page_start": "First page",
            "page_end": "Last page",
        }
        help_texts = {
            "parent": "Leave blank to add it as a chapter of its own.",
            "source_file": "Optional — leave blank for something you teach that "
            "the uploaded material does not cover.",
        }
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Bayes' theorem"}),
            "page_start": forms.NumberInput(attrs={"min": 1}),
            "page_end": forms.NumberInput(attrs={"min": 1}),
        }

    def __init__(self, *args, course=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.course = course
        if course is not None:
            self.fields["parent"].queryset = course.topics.chapters()
            self.fields["source_file"].queryset = course.files.all()
        self.fields["parent"].required = False
        self.fields["source_file"].required = False
        self.fields["parent"].empty_label = "— none, this is a chapter —"
        self.fields["source_file"].empty_label = "— not in the uploaded material —"

    def clean_name(self):
        name = self.cleaned_data["name"].strip()
        if not name:
            raise forms.ValidationError("Give the topic a name.")
        return name

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("page_start"), cleaned.get("page_end")
        if start and end and end < start:
            raise forms.ValidationError("The last page comes before the first page.")
        if end and not start:
            raise forms.ValidationError("Give a first page as well as a last page.")
        source_file = cleaned.get("source_file")
        if source_file and start and start > source_file.page_count:
            raise forms.ValidationError(
                f"{source_file.original_name} has only {source_file.page_count} pages."
            )
        return cleaned

    def save(self, commit=True):
        topic = super().save(commit=False)
        if self.course is not None:
            topic.course = self.course
            # New topics go to the end of the list, where the instructor left off.
            topic.position = (
                self.course.topics.order_by("-position")
                .values_list("position", flat=True)
                .first()
                or 0
            ) + 1
        if commit:
            topic.save()
        return topic


class TopicRenameForm(forms.ModelForm):
    """Just the name. Renaming is the most common edit on this screen and must
    never risk touching anything else on the row."""

    class Meta:
        model = Topic
        fields = ("name",)

    def clean_name(self):
        name = self.cleaned_data["name"].strip()
        if not name:
            raise forms.ValidationError("A topic needs a name.")
        return name
