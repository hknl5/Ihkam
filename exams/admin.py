from django.contrib import admin

from .models import Blueprint, BlueprintRow, Exam


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
