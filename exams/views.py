"""The exam spec and the blueprint editor — the first screens where the
instructor decides rather than reviews (M4).

The editor validates without saving. Every keystroke posts the table to
`blueprint_validate`, which runs the same checks against the same code as the
save path and re-renders one total row — nothing is written until the instructor
presses Save. That separation is the point: an instructor should be able to try
a distribution, see it fail, and walk away without having changed anything.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from courses.models import Course

from .forms import ExamForm, _wants_multiple_forms, row_formset, specs_from_post
from .models import Blueprint, Exam
from .services.blueprint import Issue, auto_build, eligible_topics, validate, validate_blueprint
from .services.forms import FormAssemblyError, assemble_forms, save_assembly


def _own_course(request, pk) -> Course:
    return get_object_or_404(Course, pk=pk, instructor=request.user)


def _own_exam(request, pk, exam_pk) -> Exam:
    return get_object_or_404(
        Exam, pk=exam_pk, course__pk=pk, course__instructor=request.user
    )


def _blueprint_url(exam) -> str:
    return reverse("exams:blueprint", args=[exam.course_id, exam.pk])


@login_required
def exam_list(request, pk):
    """The exam spec screen: what this exam is, before how it is spent."""
    course = _own_course(request, pk)

    if request.method == "POST":
        form = ExamForm(request.POST)
        if form.is_valid():
            exam = form.save(commit=False)
            exam.course = course
            exam.save()
            messages.success(
                request,
                f"{exam.display_title} defined — {exam.question_count} questions, "
                f"{exam.total_score} marks. Now plan how it is spread.",
            )
            return redirect(_blueprint_url(exam))
    else:
        form = ExamForm()

    exams = list(course.exams.all())
    return render(
        request,
        "exams/exam_list.html",
        {
            "course": course,
            "exams": exams,
            "form": form,
            "topic_count": len(eligible_topics(course)),
        },
    )


@login_required
@require_POST
def exam_sharing_option(request, pk):
    """Show or hide the sharing question as the form count is typed (M9, HTMX).

    A one-form exam has nothing to relate to anything, so the question is not
    asked — an option that cannot apply is clutter on the one screen where every
    field is a decision. Nothing is saved; this only re-renders one field.
    """
    _own_course(request, pk)
    show = _wants_multiple_forms(request.POST)
    form = ExamForm(initial={"form_sharing": request.POST.get("form_sharing") or None})
    return render(
        request,
        "exams/partials/sharing_option.html",
        {"field": form["form_sharing"], "show": show},
    )


def _editor_context(request, exam, *, formset=None, report=None) -> dict:
    blueprint = getattr(exam, "blueprint", None)
    return {
        "course": exam.course,
        "exam": exam,
        "blueprint": blueprint,
        "formset": formset if formset is not None else row_formset(exam.course, blueprint),
        # An exam with no blueprint is still validated — against no rows, which
        # says so in the same total row as every other failure rather than
        # leaving the instructor a blank table with no verdict on it.
        "report": report
        if report is not None
        else (validate_blueprint(blueprint) if blueprint else validate([], exam=exam)),
        "eligible_count": len(eligible_topics(exam.course)),
    }


@login_required
def blueprint(request, pk, exam_pk):
    """The blueprint editor. GET shows the table; POST saves it."""
    exam = _own_exam(request, pk, exam_pk)

    if request.method == "POST":
        board, _ = Blueprint.objects.get_or_create(exam=exam)
        formset = row_formset(exam.course, board, request.POST)
        if formset.is_valid():
            rows = formset.save(commit=False)
            for row in formset.deleted_objects:
                row.delete()
            for row in rows:
                row.blueprint = board
                row.save()
            # A saved blueprint is the instructor's, whatever built it first.
            if board.is_auto_built:
                board.is_auto_built = False
            board.save()
            _renumber(board)

            report = validate_blueprint(board)
            if report.is_valid:
                messages.success(request, "Blueprint saved. " + report.summary)
            else:
                messages.warning(request, "Blueprint saved as it stands. " + report.summary)
            return redirect(_blueprint_url(exam))

        messages.error(request, "Some rows could not be saved — see the table.")
        return render(
            request,
            "exams/blueprint.html",
            _editor_context(request, exam, formset=formset),
            status=400,
        )

    return render(request, "exams/blueprint.html", _editor_context(request, exam))


def _renumber(board) -> None:
    """Keep `position` in the order the rows are shown, after any add or delete."""
    for position, row in enumerate(board.rows.all()):
        if row.position != position:
            row.position = position
            row.save(update_fields=["position"])


@login_required
@require_POST
def blueprint_validate(request, pk, exam_pk):
    """Live validation: the same checks, against unsaved input (HTMX).

    Writes nothing. Returns only the total row, which is the whole surface the
    instructor is watching while they type.
    """
    exam = _own_exam(request, pk, exam_pk)
    specs, unreadable = specs_from_post(request.POST, course=exam.course)
    report = validate(specs, exam=exam)
    for label in unreadable:
        report.issues.insert(
            0,
            Issue("unreadable_value", f"{label.capitalize()} is not a number, so it counts as 0."),
        )
    return render(
        request,
        "exams/partials/blueprint_totals.html",
        {"report": report, "exam": exam, "live": True},
    )


@login_required
@require_POST
def blueprint_autobuild(request, pk, exam_pk):
    """Build the first draft: equal weight across the course's usable topics."""
    exam = _own_exam(request, pk, exam_pk)
    board = auto_build(exam)
    count = board.rows.count()

    if not count:
        messages.warning(
            request,
            "There are no topics to build from. Every topic in this course is either "
            "excluded or sits under an excluded chapter.",
        )
    else:
        report = validate_blueprint(board)
        note = (
            f"Blueprint built from {count} topic{'s' if count != 1 else ''}, weighted "
            f"equally. Adjust anything — إحكام does not know which chapter you spent "
            f"three weeks on."
        )
        if report.is_valid:
            messages.success(request, note)
        else:
            messages.warning(request, note + " " + report.summary)
    return redirect(_blueprint_url(exam))


@login_required
def blueprint_plan(request, pk, exam_pk):
    """Agent 1A's output: the passages each planned question will be written from.

    Runs retrieval — one embedding call per row — so it is a deliberate GET the
    instructor asks for, not something the editor does while they type.
    """
    exam = _own_exam(request, pk, exam_pk)
    board = getattr(exam, "blueprint", None)
    if board is None:
        messages.warning(request, "Build a blueprint before grounding it.")
        return redirect(_blueprint_url(exam))

    from agents.analyze import BlueprintNotReady, build_exam_plan
    from courses.services.retrieval import RetrievalError

    plan, error = None, ""
    try:
        plan = build_exam_plan(board)
    except BlueprintNotReady as exc:
        messages.warning(
            request, f"This blueprint is not ready to be grounded. {exc.report.summary}"
        )
        return redirect(_blueprint_url(exam))
    except RetrievalError as exc:
        error = str(exc)

    return render(
        request,
        "exams/plan.html",
        {"course": exam.course, "exam": exam, "blueprint": board, "plan": plan, "error": error},
    )


@login_required
def exam_forms(request, pk, exam_pk):
    """The assembled forms — a diagnostic screen, not the comparison screen (M9).

    GET assembles from the reviewed pool and shows what would be built, beside
    what is already saved. POST saves it, and refuses to save a form that is
    short of questions: the shortfalls are shown per row instead, each naming
    the topic, the count, and the two ways out.

    Assembly is plain arithmetic, so a GET here is cheap and makes no call. The
    side-by-side comparison an instructor actually works from is M10's.
    """
    exam = _own_exam(request, pk, exam_pk)

    assembly, error = None, ""
    try:
        assembly = assemble_forms(exam)
    except FormAssemblyError as exc:
        error = str(exc)

    if request.method == "POST" and assembly is not None:
        if assembly.is_complete:
            saved = save_assembly(exam, assembly)
            messages.success(
                request,
                f"{len(saved)} form{'s' if len(saved) != 1 else ''} assembled. "
                + assembly.summary,
            )
        else:
            messages.warning(
                request,
                "No form was saved — some rows cannot be filled. " + assembly.summary,
            )
        return redirect(reverse("exams:forms", args=[exam.course_id, exam.pk]))

    return render(
        request,
        "exams/forms.html",
        {
            "course": exam.course,
            "exam": exam,
            "assembly": assembly,
            "error": error,
            "saved_forms": list(exam.forms.prefetch_related("entries__question")),
        },
    )
