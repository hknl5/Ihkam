from django.urls import path

from . import views

app_name = "exams"

# M4 — the exam spec and the blueprint editor. Validation is its own POST
# endpoint because it saves nothing: it is safe to call on every keystroke, and
# it can never be the thing that wrote a half-typed row.
urlpatterns = [
    path("courses/<int:pk>/exams/", views.exam_list, name="list"),
    path("courses/<int:pk>/exams/<int:exam_pk>/blueprint/", views.blueprint, name="blueprint"),
    path(
        "courses/<int:pk>/exams/<int:exam_pk>/blueprint/validate/",
        views.blueprint_validate,
        name="blueprint_validate",
    ),
    path(
        "courses/<int:pk>/exams/<int:exam_pk>/blueprint/build/",
        views.blueprint_autobuild,
        name="blueprint_autobuild",
    ),
    path("courses/<int:pk>/exams/<int:exam_pk>/plan/", views.blueprint_plan, name="plan"),
    # M11.5 — the Generate step. GET is the screen the stepper points at; POST is
    # the M8 loop itself, and it blocks until every row is written or given up on.
    path("courses/<int:pk>/exams/<int:exam_pk>/generate/", views.exam_generate, name="generate"),
    # M9 — the sharing question is revealed by the same code that decides
    # whether it applies, so the screen and the server never disagree about it.
    path(
        "courses/<int:pk>/exams/sharing-option/",
        views.exam_sharing_option,
        name="exam_sharing_option",
    ),
    path("courses/<int:pk>/exams/<int:exam_pk>/forms/", views.exam_forms, name="forms"),
    # M10 — the comparison screen. GET is free arithmetic over the saved forms;
    # the POST is the semantic half (embeddings + one leakage verdict per
    # shortlisted pair), asked for deliberately because it costs calls.
    path(
        "courses/<int:pk>/exams/<int:exam_pk>/forms/compare/",
        views.exam_compare,
        name="compare",
    ),
    # M11 — the decision surface, then the deliverable. Every card action is one
    # POST to one endpoint: the thing that changes is what the instructor
    # decided, not which URL it went to.
    path("courses/<int:pk>/exams/<int:exam_pk>/review/", views.question_review, name="review"),
    path(
        "courses/<int:pk>/exams/<int:exam_pk>/review/<int:question_pk>/",
        views.question_action,
        name="question_action",
    ),
    path("courses/<int:pk>/exams/<int:exam_pk>/export/", views.form_export, name="export"),
]
