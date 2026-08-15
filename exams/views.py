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
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from courses.models import Course

from agents.prompts.revision import LABELS as REVISION_LABELS
from agents.prompts.revision import MODES as REVISION_MODES

from .forms import (
    ExamForm,
    ExportOptionsForm,
    QuestionEditForm,
    _wants_multiple_forms,
    row_formset,
    specs_from_post,
)
from .models import Blueprint, Exam, Question
from .services.blueprint import Issue, auto_build, eligible_topics, validate, validate_blueprint
from .services.convergence import report_for_exam
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

    # `written` is what decides where an exam's button goes: an exam with no
    # questions opens at Generate, one with questions opens at Review. Counting
    # it here keeps that decision out of the template.
    exams = list(course.exams.annotate(written=Count("questions")))
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
    saved_report = validate_blueprint(blueprint) if blueprint else None
    return {
        # Whether Generate can run is decided against what is *saved*, never
        # against the table currently on screen: the loop reads the database, so
        # an editor full of good numbers that has not been saved is not ready.
        "can_generate": bool(
            blueprint is not None and saved_report.is_valid and blueprint.rows.exists()
        ),
        "saved_report": saved_report,
        "existing": exam.questions.count(),
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


# --- M11.5: the generate step -------------------------------------------------


def _attention_lines(run) -> list[str]:
    """Why each unfinished item stopped, in the instructor's words.

    An item that produced nothing has a reason — no passages behind the topic, an
    unsupported question type, the retry cap — and that reason is the only useful
    thing to say on return. Coming back to an empty Review screen with a cheerful
    message is the failure this exists to prevent.
    """
    lines = []
    for item in run.items_needing_attention:
        why = item.error or (
            f"only {item.approved_count} of {item.required} passed review after "
            f"{item.rounds} round{'s' if item.rounds != 1 else ''}"
        )
        lines.append(f"{item.item.topic_name} — {why}")
    return lines


@login_required
def exam_generate(request, pk, exam_pk):
    """Run the correction loop over this exam's blueprint (M8), from a button.

    Until M11.5 the loop had no trigger outside `manage.py orchestrate_probe`,
    which meant a specced exam could never become questions from the UI.

    The POST blocks: one model call per row, plus a gap-fill round for anything
    short of its count. That is minutes, not seconds, so the button disables
    itself and says what it is doing — a screen that looks frozen is a screen the
    instructor reloads, and a reload here spends the calls twice.

    Nothing is generated from a blueprint that does not add up. The loop would
    refuse anyway (Agent 1A raises `BlueprintNotReady`), but refusing here sends
    the instructor to the editor with the arithmetic in front of them instead of
    to an error.
    """
    exam = _own_exam(request, pk, exam_pk)
    board = getattr(exam, "blueprint", None)
    report = validate_blueprint(board) if board is not None else None
    can_generate = bool(board is not None and report.is_valid and board.rows.exists())

    if request.method == "POST":
        if board is None:
            messages.warning(
                request,
                "This exam has no blueprint yet, so there is nothing to generate "
                "from. Plan it first.",
            )
            return redirect(_blueprint_url(exam))
        if not can_generate:
            messages.warning(
                request,
                "Nothing was generated: this blueprint does not add up yet. "
                + report.summary,
            )
            return redirect(_blueprint_url(exam))

        from agents.orchestrator import OrchestrationError, run_exam
        from courses.services.retrieval import RetrievalError

        try:
            run = run_exam(exam)
        except (OrchestrationError, RetrievalError) as exc:
            messages.error(request, f"The generation run stopped: {exc}")
            return redirect(_generate_url(exam))

        return _report_run(request, exam, run)

    return render(
        request,
        "exams/generate.html",
        {
            "course": exam.course,
            "exam": exam,
            "blueprint": board,
            "report": report,
            "can_generate": can_generate,
            "existing": exam.questions.count(),
        },
    )


def _generate_url(exam) -> str:
    return reverse("exams:generate", args=[exam.course_id, exam.pk])


def _report_run(request, exam, run):
    """Say what the run produced, then land where the instructor can act on it."""
    produced = len(run.questions)
    attention = _attention_lines(run)

    if run.aborted:
        messages.error(
            request,
            f"The run stopped early — the model could not be reached. {run.aborted} "
            f"{produced} question{'s' if produced != 1 else ''} written before it "
            f"stopped {'are' if produced != 1 else 'is'} saved; press Generate again "
            f"to carry on.",
        )
    elif attention:
        messages.warning(
            request,
            f"{produced} question{'s' if produced != 1 else ''} written, and "
            f"{len(attention)} row{'s' if len(attention) != 1 else ''} needs manual "
            f"attention: " + "; ".join(attention) + ".",
        )
    else:
        messages.success(
            request,
            f"{produced} question{'s' if produced != 1 else ''} written and checked. "
            f"None of them is an exam question until you say so.",
        )

    # Nothing to review means Review has nothing to show, so the instructor stays
    # here with the reasons rather than being sent to an empty screen.
    if not exam.questions.exists():
        return redirect(_generate_url(exam))
    return redirect(_review_url(exam))


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


@login_required
def exam_compare(request, pk, exam_pk):
    """The comparison screen: how close the two saved papers are (M10).

    GET compares on what can be counted — coverage, marks, per-chapter share,
    level and type spread, the four expected-difficulty proxies, expected time.
    That half is arithmetic over questions already in the database, so it is
    free and runs on every visit.

    POST runs the semantic half: one embedding call for the paper, then one
    leakage verdict per shortlisted pair. It is a separate press because it is
    the only thing on this screen that costs anything, and because an instructor
    should be able to read the indicators without paying for a judgement they
    did not ask for.

    Nothing here computes an equivalence figure, and there is nowhere to put
    one — see `services/convergence.py`. The screen shows the indicators and the
    instructor decides.
    """
    exam = _own_exam(request, pk, exam_pk)
    forms = list(exam.forms.all())

    check_semantics = request.method == "POST"
    report = report_for_exam(exam, check_semantics=check_semantics) if forms else None

    if report is not None and check_semantics:
        if report.semantic_error:
            messages.warning(
                request,
                "The leakage and similarity checks did not complete, so those pairs are "
                "unchecked rather than clear. " + report.semantic_error,
            )
        else:
            messages.success(
                request,
                f"{report.shortlisted_pairs} pair"
                f"{'s' if report.shortlisted_pairs != 1 else ''} read for leakage, "
                f"{len(report.leaks)} confirmed.",
            )

    return render(
        request,
        "exams/compare.html",
        {
            "course": exam.course,
            "exam": exam,
            "report": report,
            "forms": forms,
            "semantic_ran": bool(report and report.semantic_ran),
        },
    )


# --- M11: the decision surface ------------------------------------------------


def _question_notes(question) -> list[str]:
    """The system notes panel: what إحكام recorded about this question.

    Read back from what is already stored — the review findings M7 wrote, M6's
    mark-sum flag, and the OCR provenance — rather than recomputed. A review
    screen that re-ran the checks would show the instructor a different verdict
    from the one the loop acted on.
    """
    notes: list[str] = []
    if question.needs_mark_review:
        notes.append(
            "The steps in this question's answer key do not add up to the marks it "
            "carries. Nothing was auto-corrected — the split is yours to fix."
        )
    if question.from_ocr:
        notes.append(
            "The passage this question cites is an OCR transcription of a scanned "
            "page, so the wording is worth checking against the original."
        )
    for attempt in question.attempts.order_by("-pk")[:3]:
        for note in attempt.notes or []:
            if note not in notes:
                notes.append(note)
    return notes


@login_required
def question_review(request, pk, exam_pk):
    """The review screen: every question of this exam, as a decision.

    The screen the whole pipeline has been feeding. Nothing here calls a model
    on load — the notes are read back from what was recorded, so opening the
    screen costs nothing and shows exactly what the loop decided.
    """
    exam = _own_exam(request, pk, exam_pk)
    questions = list(
        exam.questions.select_related("blueprint_row", "blueprint_row__topic")
        .prefetch_related("attempts", "form_entries__form")
        .order_by("position", "pk")
    )

    status = request.GET.get("status") or ""
    if status in dict(Question.Status.choices):
        questions = [question for question in questions if question.status == status]

    confirm_pk = _int_or_none(request.GET.get("confirm"))
    confirm_mode = request.GET.get("mode") or ""

    cards = [
        {
            "question": question,
            "form": QuestionEditForm(instance=question, prefix=f"q{question.pk}"),
            "notes": _question_notes(question),
            "placements": [entry.form for entry in question.form_entries.all()],
            "confirm": question.pk == confirm_pk and confirm_mode in REVISION_MODES,
            "confirm_mode": confirm_mode if question.pk == confirm_pk else "",
            # The button's own words, resolved here: a template filter that
            # looked this up would be a filter written to avoid a dictionary.
            "confirm_label": REVISION_LABELS.get(confirm_mode, ("", ""))[0].lower(),
        }
        for question in questions
    ]

    return render(
        request,
        "exams/review.html",
        {
            "course": exam.course,
            "exam": exam,
            "cards": cards,
            "status": status,
            "statuses": Question.Status.choices,
            "counts": {
                value: exam.questions.filter(status=value).count()
                for value, _label in Question.Status.choices
            },
            "revision_labels": REVISION_LABELS,
            "forms_saved": list(exam.forms.all()),
        },
    )


def _int_or_none(raw):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _review_url(exam, **params) -> str:
    url = reverse("exams:review", args=[exam.course_id, exam.pk])
    if params:
        from urllib.parse import urlencode

        url = f"{url}?{urlencode(params)}"
    return url


@login_required
@require_POST
def question_action(request, pk, exam_pk, question_pk):
    """One decision about one question. Every card action posts here.

    Dispatching in one view rather than eight keeps the permission check, the
    ownership check and the redirect in one place; what each action *does* lives
    in the model or the service it belongs to.
    """
    exam = _own_exam(request, pk, exam_pk)
    question = get_object_or_404(Question, pk=question_pk, exam=exam)
    action = request.POST.get("action") or ""

    if action == "approve":
        question.status = Question.Status.APPROVED
        question.save(update_fields=["status", "updated_at"])
        messages.success(request, f"Approved. {question.stem[:60]}")

    elif action == "reject":
        question.status = Question.Status.REJECTED
        question.save(update_fields=["status", "updated_at"])
        messages.warning(
            request,
            "Rejected. It stays in the log and is out of every form and every pool.",
        )

    elif action == "delete":
        stem = question.stem[:60]
        question.delete()
        messages.success(request, f"Deleted. {stem}")

    elif action == "edit":
        form = QuestionEditForm(
            request.POST, instance=question, prefix=f"q{question.pk}"
        )
        if form.is_valid():
            saved = form.save()
            # The lock, set at the one moment a human changed the question.
            saved.mark_edited()
            messages.success(
                request,
                "Saved. This question is now yours — no generation cycle will "
                "overwrite it.",
            )
        else:
            messages.error(
                request,
                "That edit could not be saved: "
                + "; ".join(
                    f"{field}: {'; '.join(errors)}" for field, errors in form.errors.items()
                ),
            )

    elif action == "move":
        _move_question(request, exam, question)

    elif action in REVISION_MODES:
        return _revise_question(request, exam, question, action)

    else:
        messages.error(request, f"There is no “{action}” action.")

    return redirect(_review_url(exam))


def _move_question(request, exam, question) -> None:
    """Move this question's placement to the other form (M9's forms, M11's hand)."""
    from .models import FormQuestion

    target_pk = _int_or_none(request.POST.get("form_id"))
    entry = question.form_entries.select_related("form").first()
    if entry is None:
        messages.warning(
            request,
            "This question is not on a form yet, so there is nothing to move. "
            "Assemble the forms first.",
        )
        return
    target = exam.forms.filter(pk=target_pk).exclude(pk=entry.form_id).first()
    if target is None:
        messages.warning(request, "Pick the other form to move this question to.")
        return
    if FormQuestion.objects.filter(form=target, question=question).exists():
        messages.warning(
            request, f"Form {target.label} already carries this question."
        )
        return
    was = entry.form.label
    entry.form = target
    entry.position = target.entries.count()
    entry.save(update_fields=["form", "position"])
    messages.success(
        request,
        f"Moved from Form {was} to Form {target.label}. The comparison screen will "
        f"show what that did to the two papers.",
    )


def _revise_question(request, exam, question, mode: str):
    """Regenerate / make easier / make harder / clarify — through 2A and 3A.

    A question the instructor edited is not regenerated on the first press: the
    service raises, and this redirects back with the confirmation showing on
    that card. The second press carries `confirmed`, and then it obeys.
    """
    from .services.revision import RevisionError, RevisionNeedsConfirmation, revise

    confirmed = request.POST.get("confirmed") == "1"
    try:
        result = revise(question, mode, confirmed=confirmed)
    except RevisionNeedsConfirmation as exc:
        messages.warning(request, str(exc))
        return redirect(_review_url(exam, confirm=question.pk, mode=mode))
    except RevisionError as exc:
        messages.error(request, str(exc))
        return redirect(_review_url(exam))

    messages.success(request, result.message)
    return redirect(_review_url(exam))


# --- M11: the deliverable -----------------------------------------------------


@login_required
def form_export(request, pk, exam_pk):
    """Configure the paper, then download it — exam and answer key, separately.

    GET shows the options. POST produces the file the button asked for. Two
    buttons and two files, never one file with the answers at the back.
    """
    import os

    from django.conf import settings
    from django.core.files.storage import default_storage
    from django.http import HttpResponse

    from .services.export import ExportError, export_form

    exam = _own_exam(request, pk, exam_pk)
    saved_forms = list(exam.forms.prefetch_related("entries"))

    if not saved_forms:
        messages.warning(
            request, "Assemble and save the forms before exporting them."
        )
        return redirect(reverse("exams:forms", args=[exam.course_id, exam.pk]))

    if request.method == "POST":
        form = ExportOptionsForm(request.POST, request.FILES, forms_available=saved_forms)
        if form.is_valid():
            target = exam.forms.filter(pk=form.cleaned_data["form_id"]).first()
            logo_path = ""
            logo = form.cleaned_data.get("logo")
            if logo is not None:
                stored = default_storage.save(f"logos/{logo.name}", logo)
                logo_path = os.path.join(settings.MEDIA_ROOT, stored)
            try:
                files = export_form(target, options=form.options(logo_path=logo_path))
            except ExportError as exc:
                messages.error(request, str(exc))
                return redirect(reverse("exams:export", args=[exam.course_id, exam.pk]))

            wanted = "key" if request.POST.get("download") == "key" else "exam"
            produced = next(item for item in files if item.kind == wanted)
            response = HttpResponse(produced.content, content_type=produced.content_type)
            response["Content-Disposition"] = f'attachment; filename="{produced.filename}"'
            return response
    else:
        form = ExportOptionsForm(
            forms_available=saved_forms,
            initial={
                "form_id": saved_forms[0].pk,
                "institution": exam.course.instructor.get_full_name() and "" or "",
            },
        )

    return render(
        request,
        "exams/export.html",
        {"course": exam.course, "exam": exam, "form": form, "saved_forms": saved_forms},
    )
