"""Prompt for the leakage judgement — the second stage of M10's check.

Versioned like the topics, generation and review prompts: ``VERSION`` is what
lets you say which wording produced a stored verdict.

One distinction does all the work here, and it is the reason this call exists at
all: **leakage is not similarity.** Two questions on the same definition are
similar and leak nothing — a student who cannot answer one cannot answer the
other. A worked numeric problem whose stem states the very constant a later
question asks the student to recall leaks completely, and embeds nowhere near
it. The vector stage upstream is a lenient net that decides which pairs are
*worth reading*; this call is the only thing that decides whether a pair leaks.

So the model is asked one narrow question — can a student who has this paper in
front of them get question B right *because* question A is on it — and is told
in as many words that "these are about the same topic" is not that.

The direction matters and is asked for explicitly. "Q8 may help answer Q17" and
"Q17 may help answer Q8" are different notes to an instructor: the first is
fixed by moving Q17 earlier or rewriting it, the second by rewriting Q8.
"""

from __future__ import annotations

VERSION = "leakage/v1"

#: The whole point of the second stage, stated in both halves of the call and
#: asserted by the suite.
NOT_SIMILARITY_RULE = (
    "Two questions covering the same topic, or worded alike, do not leak. "
    "Leakage is when the text of one question — its stem, its options, or the "
    "situation it sets up — hands the student information that answers the "
    "other. Judge the leak, not the resemblance."
)

#: Agent 3A's rule, restated: a judge that rewrites is a second author nobody
#: checked. This call returns a verdict and a sentence, never a fixed question.
JUDGE_ONLY_RULE = (
    "You judge the pair. You never rewrite either question and never suggest "
    "replacement wording."
)

SYSTEM = f"""\
You check one pair of questions from the same exam paper for answer leakage.

A student sits this paper with every question visible at once. Your only \
question is: could a student who does not know the answer to one of these \
questions work it out from reading the other?

{NOT_SIMILARITY_RULE}

Leakage looks like this:
- one question states a value, definition, formula or result that the other \
asks the student to supply;
- one question's options contain the answer to the other;
- one question's worked setup reveals the classification, outcome or count that \
the other asks for;
- one question presupposes as given exactly what the other asks the student to \
derive.

These are NOT leakage:
- the two questions are on the same topic, chapter or formula;
- the two questions are worded similarly;
- both questions are answerable by a student who studied the same material;
- one is simply easier than the other.

{JUDGE_ONLY_RULE}

Judge the pair in the language it is written in.

Answer with JSON only, in exactly this shape:

{{
  "leaks": false,
  "direction": "none",
  "reason": ""
}}

`direction` is one of "a_reveals_b", "b_reveals_a", "both", or "none". \
When `leaks` is false, `direction` is "none" and `reason` may be empty. \
When `leaks` is true, `direction` is not "none" and `reason` says in one or two \
concrete sentences what is revealed and where it is revealed — quote the \
giveaway.
"""


def _one(label: str, question) -> str:
    """One side of the pair, with its answer, laid out for reading.

    The answer is included deliberately: the judgement is about whether the
    *other* question hands this one's answer over, which cannot be assessed
    without knowing what that answer is.
    """
    options = (
        "\n".join(f"  - {option}" for option in question.options)
        if question.options
        else "  (an open question — no options)"
    )
    return (
        f"Question {label} ({question.ref.code}) — {question.topic_name}\n"
        f"{question.stem}\n"
        f"Options:\n{options}\n"
        f"Its correct answer: {question.answer_text or '(not recorded)'}"
    )


def build_user_prompt(*, course_name: str, form_label: str, first, second) -> str:
    """The user half: two questions from one paper, side by side, with answers."""
    return (
        f"Course: {course_name}\n"
        f"Both questions are printed on the same paper: Form {form_label}.\n\n"
        f"{_one('A', first)}\n\n"
        f"{_one('B', second)}\n\n"
        f"{NOT_SIMILARITY_RULE}\n"
        f"{JUDGE_ONLY_RULE}\n\n"
        "Does either question reveal the other's answer to a student reading "
        "both?"
    )


__all__ = ["JUDGE_ONLY_RULE", "NOT_SIMILARITY_RULE", "SYSTEM", "VERSION", "build_user_prompt"]
