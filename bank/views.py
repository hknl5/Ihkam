"""The bank's screens: browse it, search it, pull a question out of it (M12).

Two views and two rules.

* **Browsing is free; searching is a press.** A GET lists the course's bank with
  every question's metadata and costs nothing. The search box posts, because it
  spends one embedding call — the same separation M10's compare screen makes
  between the arithmetic half and the half that costs.
* **A pull needs a slot.** "Use this question" is meaningless without a
  blueprint row to put it in, so the browse screen is opened *with* an exam
  (`?exam=`) when it is being used for sourcing, and the pull form names the row.
  Opened without one, it is simply the instructor's bank, read.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from courses.models import Course
from exams.models import BlueprintRow, Exam

from .forms import BankFilterForm
from .models import BankQuestion
from .services.save import BankError, reuse_in_exam
from .services.search import browse, search_bank


def _own_course(request, pk) -> Course:
    return get_object_or_404(Course, pk=pk, instructor=request.user)


def _browse_url(course, exam=None) -> str:
    url = reverse("bank:browse", args=[course.pk])
    return f"{url}?exam={exam.pk}" if exam is not None else url


@login_required
def bank_browse(request, pk):
    """The course's bank: every approved question kept for reuse.

    GET lists it. POST runs the semantic search over the same list — and when
    the search cannot run, the list is still there, with the reason above it.
    """
    course = _own_course(request, pk)
    exam = None
    exam_pk = request.GET.get("exam") or request.POST.get("exam")
    if exam_pk:
        exam = Exam.objects.filter(
            pk=exam_pk, course=course, course__instructor=request.user
        ).first()

    form = BankFilterForm(request.POST or request.GET or None)
    form.is_valid()  # every field is optional; this only populates cleaned_data
    filters = form.filters()

    results, search = None, None
    if request.method == "POST" and form.query_text:
        search = search_bank(course, form.query_text, **filters)
        if search.error:
            messages.warning(
                request,
                "The search did not run, so this is the whole bank rather than a "
                "ranked list. " + search.error,
            )
        else:
            messages.success(request, search.summary)
            results = search.questions

    questions = list(results if results is not None else browse(course, **filters))

    # The rows a pulled question could go into. Offered only when an exam is in
    # hand — a question with nowhere to go is not a question you can "use".
    rows = []
    if exam is not None and exam.has_blueprint:
        rows = list(exam.blueprint.rows.select_related("topic"))

    return render(
        request,
        "bank/browse.html",
        {
            "course": course,
            "exam": exam,
            "rows": rows,
            "form": form,
            "questions": questions,
            "search": search,
            "total": BankQuestion.objects.for_course(course).count(),
            "exams": list(course.exams.all()[:20]),
        },
    )


@login_required
@require_POST
def bank_use(request, pk, bank_pk):
    """Pull one banked question into one blueprint slot of one exam."""
    course = _own_course(request, pk)
    banked = get_object_or_404(BankQuestion, pk=bank_pk, course=course)
    exam = get_object_or_404(
        Exam, pk=request.POST.get("exam") or 0, course=course
    )
    row = BlueprintRow.objects.filter(
        pk=request.POST.get("row") or 0, blueprint__exam=exam
    ).select_related("topic").first()

    if row is None:
        messages.warning(
            request, "Pick the blueprint row this question should fill."
        )
        return redirect(_browse_url(course, exam))

    if row.question_type != banked.question_type or row.level != banked.level:
        # Not refused — the instructor may know exactly what they are doing —
        # but not silent either: the row promised a type and a level, and this
        # question is about to make that promise untrue.
        messages.warning(
            request,
            f"That row asks for {row.get_question_type_display()} · "
            f"{row.get_level_display()} and this question is {banked.type_label} · "
            f"{banked.level_label}. It was added anyway — the plan is yours — but the "
            f"blueprint no longer describes the paper.",
        )

    try:
        reuse_in_exam(banked, exam=exam, row=row)
    except BankError as exc:
        messages.error(request, str(exc))
        return redirect(_browse_url(course, exam))

    messages.success(
        request,
        f"Added to {exam.display_title} under {row.topic.name}, approved as it was "
        f"when you banked it. It is on the review screen with everything else.",
    )
    return redirect(reverse("exams:review", args=[course.pk, exam.pk]))
