from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .forms import CourseForm, SourceFileUploadForm, TopicForm, TopicRenameForm
from .models import Chunk, Course, SourceFile, Topic
from .services.ingest import ingest_source_file


def _own_course(request, pk) -> Course:
    """A course is only ever reachable by the instructor who owns it."""
    return get_object_or_404(Course, pk=pk, instructor=request.user)


def _own_topic(request, pk, topic_pk) -> Topic:
    return get_object_or_404(Topic, pk=topic_pk, course__pk=pk, course__instructor=request.user)


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
            ocr_note = (
                f" {source_file.pages_from_ocr} of them read by OCR."
                if source_file.pages_from_ocr
                else ""
            )
            if source_file.is_ready:
                messages.success(
                    request,
                    f"{source_file.original_name} uploaded — "
                    f"{source_file.page_count} pages extracted.{ocr_note}",
                )
            elif source_file.has_readable_text:
                # Partly readable. Say which part is missing rather than
                # letting a half-read file pass for a complete one.
                messages.warning(
                    request,
                    f"{source_file.original_name} uploaded — "
                    f"{source_file.page_count - source_file.pages_without_text} of "
                    f"{source_file.page_count} pages extracted.{ocr_note} "
                    f"{source_file.pages_without_text} still have no readable text.",
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
        {
            "course": course,
            "form": form,
            "files": course.files.all(),
            # So the course screen can offer the way back into an exam already
            # under way, rather than only the way forward into topics.
            "exam_count": course.exams.count(),
        },
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


# --- M2: topic review -------------------------------------------------------
#
# A required product step, not a convenience screen. Nothing the model
# extracted is trusted until the instructor has been through it, so every
# action here is theirs: rename, merge, delete, add, exclude.


def _topics_url(course) -> str:
    return reverse("courses:topics", args=[course.pk])


def _topics_context(course, add_form=None) -> dict:
    """Everything the review screen shows, built once for both its entry points."""
    from .services.topics import readable_page_count, unreadable_page_count

    chapters = list(
        course.topics.chapters()
        .select_related("source_file")
        .prefetch_related("subtopics__source_file")
    )
    total = course.topics.count()
    excluded = course.topics.filter(excluded=True).count()
    return {
        "course": course,
        "chapters": chapters,
        # A sub-topic whose chapter was deleted is still the instructor's, so
        # it is shown rather than silently missing from the screen.
        "orphans": list(
            course.topics.filter(parent__isnull=False)
            .exclude(parent__in=[c.pk for c in chapters])
            .select_related("source_file")
        ),
        "topic_total": total,
        "excluded_count": excluded,
        "included_count": total - excluded,
        "add_form": add_form if add_form is not None else TopicForm(course=course),
        "readable_pages": readable_page_count(course),
        "unreadable_pages": unreadable_page_count(course),
        "has_files": course.files.exists(),
        "chunk_count": course.files.aggregate(n=Count("chunks"))["n"] or 0,
    }


@login_required
def topics(request, pk):
    """The topic review screen: what was extracted, and every way to fix it."""
    course = _own_course(request, pk)
    return render(request, "courses/topics.html", _topics_context(course))


@login_required
@require_POST
def topics_extract(request, pk):
    """Run extraction over the course's readable pages.

    Replacing an existing list is a destructive act — it removes edits the
    instructor made — so it only happens when the form says `replace=yes`,
    which the screen only sends from a button that spells that out.
    """
    course = _own_course(request, pk)
    from .services.topics import TopicExtractionError, extract_topics
    from .services.chunking import link_chunks_to_topics

    if course.topics.exists() and request.POST.get("replace") != "yes":
        messages.warning(
            request,
            "This course already has topics. Use “Extract again” if you want to "
            "replace them — your edits would not survive it.",
        )
        return redirect(_topics_url(course))

    try:
        run = extract_topics(course)
    except TopicExtractionError as exc:
        messages.error(request, str(exc))
        return redirect(_topics_url(course))

    link_chunks_to_topics(course)

    note = (
        f"{run.chapters} chapter{'s' if run.chapters != 1 else ''} and "
        f"{run.subtopics} sub-topic{'s' if run.subtopics != 1 else ''} extracted from "
        f"{run.pages_included} readable page{'s' if run.pages_included != 1 else ''}."
    )
    if run.pages_from_ocr:
        note += (
            f" {run.pages_from_ocr} of those pages were read by OCR, so their wording "
            "is a transcription."
        )
    if run.pages_skipped:
        note += f" {run.pages_skipped} unreadable page(s) contributed nothing."
    if run.spans_dropped:
        note += (
            f" {run.spans_dropped} page reference(s) did not match a page in the "
            "material and were dropped rather than stored."
        )
    if run.pages_over_budget:
        note += (
            f" {run.pages_over_budget} page(s) did not fit in one extraction call — "
            "raise TOPIC_EXTRACTION_MAX_CHARS to include them."
        )
    messages.success(request, f"{note} Nothing here is confirmed until you say so.")
    return redirect(_topics_url(course))


@login_required
@require_POST
def topic_add(request, pk):
    course = _own_course(request, pk)
    form = TopicForm(request.POST, course=course)
    if form.is_valid():
        topic = form.save()
        messages.success(request, f"“{topic.name}” added.")
        return redirect(_topics_url(course))

    # Re-render with the errors rather than losing what they typed.
    return render(request, "courses/topics.html", _topics_context(course, form), status=400)


@login_required
@require_POST
def topic_rename(request, pk, topic_pk):
    course = _own_course(request, pk)
    topic = _own_topic(request, pk, topic_pk)
    was = topic.name
    form = TopicRenameForm(request.POST, instance=topic)
    if form.is_valid():
        form.save()
        messages.success(request, f"“{was}” renamed to “{topic.name}”.")
    else:
        messages.error(request, "A topic needs a name — nothing was changed.")
    return redirect(_topics_url(course))


@login_required
@require_POST
def topic_delete(request, pk, topic_pk):
    course = _own_course(request, pk)
    topic = _own_topic(request, pk, topic_pk)
    from .services.topics import delete_topic

    name = topic.name
    promoted = delete_topic(topic)
    note = f"“{name}” deleted."
    if promoted:
        note += (
            f" Its {promoted} sub-topic{'s' if promoted != 1 else ''} "
            f"{'were' if promoted != 1 else 'was'} kept, now listed as chapters."
        )
    messages.success(request, note)
    return redirect(_topics_url(course))


@login_required
@require_POST
def topic_exclude(request, pk, topic_pk):
    """Mark a topic "not taught in lectures", or put it back."""
    course = _own_course(request, pk)
    topic = _own_topic(request, pk, topic_pk)
    topic.excluded = not topic.excluded
    topic.save(update_fields=["excluded", "updated_at"])
    messages.success(
        request,
        f"“{topic.name}” marked as not taught — it will not be used to generate "
        "questions."
        if topic.excluded
        else f"“{topic.name}” is back in the syllabus.",
    )
    return redirect(_topics_url(course))


@login_required
def retrieval_debug(request, pk):
    """M3: the passages retrieval would hand the generator, and their scores.

    A diagnostic screen. It exists so retrieval can be judged on real material
    before anything is generated from it — if the wrong chapter shows up here,
    it would have shown up inside a question later, where it is far harder to
    see. GET-only: a query changes nothing, so it belongs in the URL and stays
    shareable and re-runnable.
    """
    course = _own_course(request, pk)
    from .services.retrieval import RetrievalError, retrieve

    raw_query = (request.GET.get("q") or "").strip()
    topic_pk = (request.GET.get("topic") or "").strip()

    topics_for_pick = list(
        course.topics.select_related("parent").order_by("position", "pk")
    )
    topic = next((t for t in topics_for_pick if str(t.pk) == topic_pk), None)

    # A picked topic wins over stale free text left in the URL, so what ran is
    # never ambiguous. Whichever it was, the composed query text is shown.
    query = topic or raw_query
    passages, error, ran = [], "", bool(topic or raw_query)
    if ran:
        try:
            passages = retrieve(course, query)
        except RetrievalError as exc:
            error = str(exc)

    from .services.retrieval import query_text_for

    course_chunks = Chunk.objects.filter(source_file__course=course)
    chunk_count = course_chunks.count()
    usable_count = course_chunks.usable().count()

    return render(
        request,
        "courses/retrieval.html",
        {
            "course": course,
            "topics": topics_for_pick,
            "q": raw_query,
            "selected_topic": topic,
            "query_text": query_text_for(query) if ran else "",
            "passages": passages,
            "error": error,
            "ran": ran,
            "top_k": settings.RETRIEVAL_TOP_K,
            "min_score": settings.RETRIEVAL_MIN_SCORE,
            "chunk_count": chunk_count,
            "usable_count": usable_count,
            "excluded_count": chunk_count - usable_count,
        },
    )


@login_required
@require_POST
def topics_merge(request, pk):
    """Merge exactly two selected topics into one."""
    course = _own_course(request, pk)
    from .services.topics import merge_topics

    selected = request.POST.getlist("topic")
    if len(selected) != 2:
        messages.warning(
            request, "Select exactly two topics to merge — merging is a pairwise decision."
        )
        return redirect(_topics_url(course))

    topics_to_merge = list(course.topics.filter(pk__in=selected))
    if len(topics_to_merge) != 2:
        messages.error(request, "One of those topics no longer exists.")
        return redirect(_topics_url(course))

    # The one earlier in the list survives, so the merged topic keeps the
    # reading order of the material.
    keep, absorb = sorted(topics_to_merge, key=lambda t: (t.position, t.pk))
    absorbed_name = absorb.name
    merge_topics(keep, absorb)
    messages.success(
        request,
        f"“{absorbed_name}” merged into “{keep.name}”. Rename it if the combined "
        "topic needs a different name.",
    )
    return redirect(_topics_url(course))
