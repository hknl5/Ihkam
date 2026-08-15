"""The stepper: one description of the instructor journey, rendered everywhere.

Before M11.5 this list was retyped in four templates, each a different truncated
version of it, and the steps past Blueprint were plain text — which is exactly
how "Generate" ended up being a word on a screen with no button behind it. The
journey is declared once here, and a step is a link whenever the screen it names
can actually be opened from what is in hand.

A step whose scope is `exam` has nowhere to point while the instructor is still
on the course screens: no exam exists yet, so it renders as text, and that is the
truth rather than a dead link.
"""

from django import template
from django.urls import reverse

register = template.Library()

#: slug, label, url name, what the url needs. In the plan's order — the order
#: the instructor walks, not the order the code was written in.
STEPS = (
    ("course", "Course", "courses:detail", "course"),
    ("upload", "Upload", "courses:detail", "course"),
    ("topics", "Topics", "courses:topics", "course"),
    ("spec", "Spec", "exams:list", "course"),
    ("blueprint", "Blueprint", "exams:blueprint", "exam"),
    ("generate", "Generate", "exams:generate", "exam"),
    ("review", "Review", "exams:review", "exam"),
    ("forms", "Forms", "exams:forms", "exam"),
    ("compare", "Compare", "exams:compare", "exam"),
    ("approve", "Approve", "exams:review", "exam"),
    ("export", "Export", "exams:export", "exam"),
)

SLUGS = [slug for slug, *_ in STEPS]

#: Approve is a decision taken on the review screen, so it shares that screen's
#: url — filtered to the questions that still have no decision on them.
QUERY = {"approve": "?status=candidate"}


def _href(url_name: str, scope: str, course, exam) -> str:
    if scope == "exam":
        if exam is None:
            return ""
        return reverse(url_name, args=[exam.course_id, exam.pk])
    if course is None:
        return ""
    return reverse(url_name, args=[course.pk])


@register.inclusion_tag("partials/stepper.html")
def stepper(current, course=None, exam=None):
    """Render the journey with `current` in accent and everything before it done.

    `current` is a slug from `STEPS`. Passing one that is not in the list is a
    template bug, not a user-facing state, so it raises rather than silently
    rendering a stepper with nothing marked.
    """
    if current not in SLUGS:
        raise ValueError(f"'{current}' is not a step of the journey: {', '.join(SLUGS)}")

    if course is None and exam is not None:
        course = exam.course

    here = SLUGS.index(current)
    items = []
    for index, (slug, label, url_name, scope) in enumerate(STEPS):
        href = _href(url_name, scope, course, exam)
        items.append(
            {
                "slug": slug,
                "label": label,
                "href": (href + QUERY.get(slug, "")) if href else "",
                "is_done": index < here,
                "is_current": index == here,
            }
        )
    return {"items": items}
