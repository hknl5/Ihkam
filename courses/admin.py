from django.contrib import admin

from .models import Course, ExtractedPage, SourceFile


class SourceFileInline(admin.TabularInline):
    model = SourceFile
    extra = 0
    fields = ("original_name", "kind", "status", "page_count", "uploaded_at")
    readonly_fields = ("original_name", "kind", "status", "page_count", "uploaded_at")
    show_change_link = True
    can_delete = True


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "instructor", "level", "content_language", "created_at")
    list_filter = ("level", "content_language")
    search_fields = ("name", "code", "instructor__username")
    autocomplete_fields = ()
    inlines = [SourceFileInline]


@admin.register(SourceFile)
class SourceFileAdmin(admin.ModelAdmin):
    list_display = (
        "original_name",
        "course",
        "kind",
        "status",
        "page_count",
        "pages_without_text",
        "pages_from_ocr",
        "uploaded_at",
    )
    list_filter = ("kind", "status")
    search_fields = ("original_name", "course__code", "course__name")
    readonly_fields = (
        "page_count",
        "pages_without_text",
        "unmappable_chars",
        "pages_from_ocr",
        "ocr_engine",
        "status_detail",
        "uploaded_at",
        "extracted_at",
    )


@admin.register(ExtractedPage)
class ExtractedPageAdmin(admin.ModelAdmin):
    list_display = ("source_file", "number", "source", "is_image_only")
    list_filter = ("source", "is_image_only")
    search_fields = ("source_file__original_name", "text")
