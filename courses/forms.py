from django import forms

from .models import Course, SourceFile

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
