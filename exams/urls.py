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
]
