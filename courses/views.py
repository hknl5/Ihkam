from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render

from .forms import CourseForm, SourceFileUploadForm
from .models import Course, SourceFile
from .services.ingest import ingest_source_file


def _own_course(request, pk) -> Course:
    """A course is only ever reachable by the instructor who owns it."""
    return get_object_or_404(Course, pk=pk, instructor=request.user)


@login_required
def dashboard(request):
    """"My courses": the instructor's courses, plus the create form."""
    if request.method == "POST":
        form = CourseForm(request.POST, instructor=request.user)
        if form.is_valid():
            course = form.save()
            messages.success(request, f"Course {course.code} created.")
            return redirect(course)
    else:
        form = CourseForm(instructor=request.user)

    courses = (
        Course.objects.filter(instructor=request.user)
        .annotate(file_count=Count("files"))
        .order_by("-created_at")
    )
    return render(
        request,
        "courses/dashboard.html",
        {"courses": courses, "form": form, "show_form": request.method == "POST"},
    )


@login_required
def detail(request, pk):
    """Course detail: upload material, then read what was extracted."""
    course = _own_course(request, pk)

    if request.method == "POST":
        form = SourceFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            source_file = form.save(commit=False)
            source_file.course = course
            source_file.save()
            # Extraction is synchronous in M1: a lecture PDF takes well under a
            # second, and a visible result beats a background job to debug.
            ingest_source_file(source_file)
            if source_file.is_ready:
                messages.success(
                    request,
                    f"{source_file.original_name} uploaded — "
                    f"{source_file.page_count} pages extracted.",
                )
            else:
                messages.warning(
                    request,
                    f"{source_file.original_name} uploaded, but no text was extracted.",
                )
            return redirect(source_file)
    else:
        form = SourceFileUploadForm()

    return render(
        request,
        "courses/detail.html",
        {"course": course, "form": form, "files": course.files.all()},
    )


@login_required
def file_detail(request, pk, file_pk):
    """Extracted text, one page at a time, with the page number kept visible."""
    course = _own_course(request, pk)
    source_file = get_object_or_404(SourceFile, pk=file_pk, course=course)

    paginator = Paginator(source_file.pages.all(), 1)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "courses/file_detail.html",
        {
            "course": course,
            "source_file": source_file,
            "page": page,
            "extracted_page": page.object_list[0] if page.object_list else None,
        },
    )


@login_required
def file_delete(request, pk, file_pk):
    course = _own_course(request, pk)
    source_file = get_object_or_404(SourceFile, pk=file_pk, course=course)
    if request.method == "POST":
        name = source_file.original_name
        source_file.file.delete(save=False)
        source_file.delete()
        messages.success(request, f"{name} removed.")
    return redirect(course)
