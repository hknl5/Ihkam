from django.contrib import admin

from .models import Chunk, Course, ExtractedPage, SourceFile, Topic


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
    list_display = ("source_file", "number", "source", "ocr_reason", "is_image_only")
    list_filter = ("source", "ocr_reason", "is_image_only")
    search_fields = ("source_file__original_name", "text")


class SubtopicInline(admin.TabularInline):
    model = Topic
    fk_name = "parent"
    extra = 0
    fields = ("name", "source_file", "page_start", "page_end", "excluded")
    show_change_link = True


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ("name", "course", "parent", "source_file", "page_span", "excluded")
    list_filter = ("excluded", "course")
    search_fields = ("name", "course__code", "course__name")
    list_select_related = ("course", "parent", "source_file")
    autocomplete_fields = ("parent", "source_file")
    readonly_fields = ("created_at", "updated_at")
    inlines = [SubtopicInline]

    @admin.display(description="Pages")
    def page_span(self, obj):
        return obj.page_span or "—"


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    """Read-only. Chunks are built by `courses/services/chunking.py`; editing
    a passage by hand would put its text out of step with its embedding."""

    list_display = ("source_file", "page", "position", "source", "topic", "preview")
    list_filter = ("source", "source_file__course")
    search_fields = ("text", "source_file__original_name")
    list_select_related = ("source_file", "topic")
    # `embedding` is 1536 floats — never worth rendering in a form.
    exclude = ("embedding",)
    readonly_fields = ("source_file", "page", "position", "text", "source", "topic", "created_at")

    @admin.display(description="Text")
    def preview(self, obj):
        return obj.text[:90] + ("…" if len(obj.text) > 90 else "")

    def has_add_permission(self, request):
        return False
