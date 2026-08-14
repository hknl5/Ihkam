from django.contrib import admin

from .models import Blueprint, BlueprintRow, Exam, Question


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
