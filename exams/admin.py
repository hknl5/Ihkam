from django.contrib import admin

from .models import Blueprint, BlueprintRow, Exam, ItemRun, Question, QuestionAttempt


class BlueprintRowInline(admin.TabularInline):
    model = BlueprintRow
    extra = 0


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = ("__str__", "course", "kind", "total_score", "question_count")
    list_filter = ("kind",)


@admin.register(Blueprint)
class BlueprintAdmin(admin.ModelAdmin):
    list_display = ("__str__", "is_auto_built", "updated_at")
    inlines = [BlueprintRowInline]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "exam", "question_type", "status", "from_ocr", "source_ref")
    list_filter = ("status", "question_type", "from_ocr")
    search_fields = ("stem", "correct")


class QuestionAttemptInline(admin.TabularInline):
    """The audit trail, read where it makes sense: beside the item it belongs to."""

    model = QuestionAttempt
    extra = 0
    fields = ("round", "outcome", "stem", "failed_checks", "notes", "question")
    readonly_fields = fields  # an attempt is a fact about a past round


@admin.register(ItemRun)
class ItemRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "exam", "required", "approved_count", "rounds", "status")
    list_filter = ("status",)
    search_fields = ("topic_name",)
    inlines = [QuestionAttemptInline]
