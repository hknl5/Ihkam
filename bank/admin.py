from django.contrib import admin

from .models import BankQuestion, BankUsage


class BankUsageInline(admin.TabularInline):
    model = BankUsage
    extra = 0


@admin.register(BankQuestion)
class BankQuestionAdmin(admin.ModelAdmin):
    list_display = ("stem", "course", "topic_label", "question_type", "level", "usage_count")
    list_filter = ("course", "question_type", "level")
    search_fields = ("stem", "correct", "topic_name")
    inlines = [BankUsageInline]
