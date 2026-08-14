"""Prompt for question review — Agent 3A (M7).

Versioned like the topics and generation prompts: ``VERSION`` is what lets you
say which wording produced a stored verdict. Bump it whenever the text below
changes in a way that could change a judgement.

Two instructions do the work here.

``JUDGE_ONLY_RULE`` is the one that keeps the architecture honest. A model asked
to review a question will happily return a better question — and the moment it
does, the reviewer becomes a second author, nobody has checked the rewrite, and
the loop that was supposed to catch bad questions has quietly started writing
them. Agent 3A judges. Agent 2A writes. The rule is stated in both halves of the
call and asserted by the suite.

``ACTIONABLE_RULE`` is the one that makes M8 possible. "Bad question" is not a
note a generator can act on. Every rejection has to say what the replacement
must *do* differently, because that sentence is the entire input to the next
generation attempt.

The deterministic checks — one correct option, duplicate options, the correct
option's length, the mark sum — are not in this prompt at all. They are
arithmetic and string comparison, they belong in Python, and asking a model to
re-do them would make a fixed answer probabilistic.
"""

from __future__ import annotations

VERSION = "review/v1"

#: Stated in both halves of the call, and asserted by the test suite.
JUDGE_ONLY_RULE = (
    "You judge questions. You never rewrite them. Do not supply a corrected "
    "stem, corrected options or a corrected answer — say what is wrong and what "
    "a replacement would have to do instead."
)

#: The sentence that makes a rejection usable by the generator (M8).
ACTIONABLE_RULE = (
    "For every check you fail, `reason` says what is wrong with this question "
    "in one or two concrete sentences, and `requirement` says what a replacement "
    "question must do differently. Write `requirement` as an instruction to the "
    "question writer, not as a description of the fault."
)

#: What each level means, in the blueprint's own vocabulary. The level check is
#: the brief's worked example — a request for a computation that came back as a
#: definition — so the reviewer is given the same definitions the generator was.
LEVEL_MEANINGS = {
    "direct": (
        "Direct recall — the answer is stated in the passage; the student has to "
        "have read it, not reasoned from it."
    ),
    "medium": (
        "Applied — the student applies something the passage teaches to a case "
        "the passage does not already work through. Recognising or restating a "
        "definition is below this level."
    ),
    "multi_step": (
        "Multi-step — reaching the answer takes two or more connected steps: a "
        "computation, a derivation, or an application that depends on a previous "
        "result. A question that asks what something *is*, or that can be "
        "answered by quoting one sentence of the passage, is below this level, "
        "however difficult its vocabulary."
    ),
}

SYSTEM = f"""\
You review exam questions written for one university course, against that \
course's own teaching material. You are the last check before a human \
instructor sees the question.

{JUDGE_ONLY_RULE}

Judge exactly these checks, each independently:

- `content_link`: is this question answerable from the passages supplied below, \
and only from them? Fail it if the question, its options or its answer depend on \
a fact, term, formula or example that is not in the passages — even if that fact \
is true and standard in the subject. Being correct about the world is not the \
test; being drawn from this course's material is.
- `clarity`: does the question have exactly one reasonable interpretation? Fail \
it for an ambiguous stem, a question that asks two things at once, a missing \
condition the answer depends on, or a question verb that does not match what the \
student is asked to produce.
- `answer_consistency`: is the given answer the right one, and is it supported by \
the passages? Fail it if the passage contradicts the answer, if more than one \
answer would be defensible, or if the explanation does not actually support the \
answer.
- `level_match`: does the question demand the cognitive level that was \
requested? Fail it in either direction — too easy (a definition where a \
computation was asked for) or too hard — and say which.
- `distractor_quality`: for multiple choice only, are the wrong options wrong, \
plausible to a student who has read the passage, and different from each other \
in meaning? Fail it for an option that is also correct, for two options that mean \
the same thing in different words, or for a distractor no student would consider. \
For any other question type, pass this check and say it does not apply.

{ACTIONABLE_RULE}

Judge the question in the language it is written in. Do not penalise a question \
for being in Arabic or English.

Answer with JSON only, in exactly this shape:

{{
  "content_link": {{"ok": true, "reason": "", "requirement": ""}},
  "clarity": {{"ok": true, "reason": "", "requirement": ""}},
  "answer_consistency": {{"ok": true, "reason": "", "requirement": ""}},
  "level_match": {{"ok": true, "reason": "", "requirement": ""}},
  "distractor_quality": {{"ok": true, "reason": "", "requirement": ""}}
}}

When a check passes, `reason` and `requirement` may be empty. When it fails, \
both must be filled in.
"""


def format_passages(passages) -> str:
    """The passages the question claims to come from, labelled as in generation."""
    blocks = []
    for position, passage in enumerate(passages, start=1):
        label = " (OCR transcription)" if passage.from_ocr else ""
        blocks.append(f"[P{position}] {passage.page_ref}{label}\n{passage.text.strip()}")
    return "\n\n".join(blocks) if blocks else "(none supplied)"


def format_answer_key(key) -> str:
    """The answer key, laid out so the reviewer judges the key and not only the stem.

    Written out in full rather than dumped as JSON: the reviewer is being asked
    whether a marker could use this, and a marker reads sentences.
    """
    from agents.answer_key import NumericKey, ObjectiveKey, ShortAnswerKey

    if key is None:
        return "(no answer key)"
    if isinstance(key, ObjectiveKey):
        return f"Correct answer: {key.answer}"
    if isinstance(key, ShortAnswerKey):
        elements = "\n".join(f"  - {e}" for e in key.required_elements)
        return f"Model answer: {key.model_answer}\nThe answer must contain:\n{elements}"
    if isinstance(key, NumericKey):
        steps = "\n".join(
            f"  {n}. [{step.marks} marks] {step.text}" for n, step in enumerate(key.steps, 1)
        )
        return f"Worked solution:\n{steps}\nFinal answer: {key.final_answer}"
    return str(key)


def build_user_prompt(
    *,
    course_name: str,
    topic_name: str,
    question_type: str,
    type_label: str,
    level: str,
    level_label: str,
    marks,
    stem: str,
    options,
    correct: str,
    explanation: str,
    answer_key=None,
    passages=(),
) -> str:
    """The user half of the call: what was asked for, what came back, what from."""
    option_lines = (
        "\n".join(
            f"  {'✔' if option == correct else ' '} {option}" for option in options
        )
        if options
        else "  (an open question — no options)"
    )
    return (
        f"Course: {course_name}\n"
        f"Topic: {topic_name}\n"
        f"Question type asked for: {type_label}\n"
        f"Level asked for: {level_label} — {LEVEL_MEANINGS.get(level, '')}\n"
        f"Marks: {marks}\n\n"
        "The question as written:\n\n"
        f"{stem}\n\n"
        f"Options:\n{option_lines}\n\n"
        f"{format_answer_key(answer_key)}\n\n"
        f"The writer's explanation: {explanation or '(none given)'}\n\n"
        f"{JUDGE_ONLY_RULE}\n"
        f"{ACTIONABLE_RULE}\n\n"
        "The passages this question was supposed to be written from — the only "
        "material it is allowed to depend on:\n\n"
        f"{format_passages(passages)}"
    )
