"""Prompt for question generation (M5).

Versioned like the topics prompt: when this wording changes what the generator
produces, ``VERSION`` is what lets you say which prompt a stored candidate came
from. Bump it whenever the text below changes in a way that could change output.

The whole prompt is built around one instruction — ``GROUNDING_RULE``. A model
asked to "write a question about recursion" writes a question about recursion as
it understands the subject; a model asked to write a question *from these five
passages* writes one the instructor can check against their own slides. The
difference is not style, it is whether إحكام is examining the course or the
internet, and it is the single thing M5 is measured on. The rule is therefore
stated twice, once in the system message and once beside the passages, and every
candidate has to name the passage it came from.
"""

from __future__ import annotations

VERSION = "generate/v2"  # M6: the answer key comes back with the question

#: Stated in both halves of the call, and asserted by the test suite. If this
#: sentence ever stops reaching the model, generation silently becomes general
#: knowledge with citations attached.
GROUNDING_RULE = (
    "Write only from the numbered passages below. Every question, every option, "
    "every answer and every explanation must be supported by the text of one of "
    "those passages. Do not use anything you know about the subject that is not "
    "in them."
)

#: Stated with the type rules and asserted by the suite. The key is not a
#: separate deliverable the model may skip when it is running out of room: it is
#: part of the question, and M6 exists because a key produced by a later call is
#: a key for a question the model has to re-read rather than one it wrote.
ANSWER_KEY_RULE = (
    "Every question carries its own `answer_key` in the same reply. Never leave "
    "it out and never leave it to be filled in later — a question without its "
    "key is not finished."
)

#: What `answer_key` must contain, per type. Objective questions repeat the
#: answer they already gave; the open types carry what a marker actually needs.
KEY_RULES = {
    "mcq": (
        "`answer_key`: {\"answer\": the correct option text, exactly as it "
        "appears in `options`}."
    ),
    "true_false": '`answer_key`: {"answer": "True" or "False"}.',
    "short_answer": (
        "`answer_key`: {\"model_answer\": the answer you would accept in full, "
        "\"required_elements\": a list of 2–4 short phrases naming the ideas a "
        "student's answer must contain to earn the marks}. The elements are "
        "ideas, not wording — a marker uses them on an answer phrased "
        "differently from yours. Every element must come from the passage."
    ),
    "numeric": (
        "`answer_key`: {\"steps\": a list of {\"text\": what is done in this "
        "step, with the arithmetic, \"marks\": what this step is worth}, "
        "\"final_answer\": the answer with its unit}. Split the question's "
        "marks across the steps however the work deserves — the marks are a "
        "guide for a human marker, not an answer to match — but **the step "
        "marks must add up to exactly the marks the question is worth**. Use "
        "only quantities, formulas and methods the passage gives."
    ),
}

TYPE_RULES = {
    "mcq": (
        "Multiple choice: exactly 4 options. One is correct and stated in the "
        "passage; the other three are wrong but plausible to a student who has "
        "read it — a near-miss from the same passage, not a joke answer. "
        "`correct` repeats the correct option text exactly. Never write 'all of "
        "the above' or 'none of the above'."
    ),
    "true_false": (
        "True / false: `options` is exactly [\"True\", \"False\"] and `correct` is "
        "one of those two words. The statement must be decidable from the "
        "passage alone. A false statement is a real claim the passage "
        "contradicts, not a sentence with 'not' inserted."
    ),
    "short_answer": (
        "Short answer: `options` is an empty list. `correct` is the model answer "
        "— one or two sentences in the passage's own terms, answerable in the "
        "time allowed."
    ),
    "numeric": (
        "Numeric problem: `options` is an empty list. The problem is stated in "
        "text only — no diagram, no figure, no symbols the passage does not use. "
        "`correct` is the final answer with its unit; `explanation` shows the "
        "steps that reach it. Use only quantities, formulas and methods the "
        "passage gives."
    ),
}

LEVEL_RULES = {
    "direct": (
        "Direct recall: the answer is stated in the passage. The student has to "
        "have read it, not reasoned from it."
    ),
    "medium": (
        "Applied: the student applies something the passage teaches to a case "
        "the passage does not already work through."
    ),
    "multi_step": (
        "Multi-step: reaching the answer takes two or more connected steps, all "
        "of them supported by the passages."
    ),
}

SYSTEM = f"""\
You write exam questions for one university course, from that course's own \
teaching material.

{GROUNDING_RULE}

Rules:
- Every question carries `source_ref`: the label (for example "P3") of the one \
passage it was written from. A question you cannot label this way is a question \
you should not have written — leave it out.
- Do not mention the passages, the page, or "the text" in the question itself. \
A student sits the exam without them. Write "In a binary search, ..." not \
"According to the passage, ...".
- Each question stands alone and is answerable without the others.
- Questions in one batch must be about different things the passages say. Do \
not rephrase one question four ways.
- Some passages are marked `(OCR transcription)` — a model's reading of a \
picture of the page, which may contain errors. You may use them, but do not \
build a question on a number, symbol or word that looks like a transcription \
slip, and prefer a text-layer passage when both say the same thing.
- Write the questions in the exam language you are given, whatever language the \
passages are in.
- `explanation` says why the answer is right, in one or two sentences, pointing \
at what the passage states.
- {ANSWER_KEY_RULE}

Answer with JSON only, in exactly this shape. `answer_key` holds the fields \
listed for the type you are asked for, and nothing else:

{{
  "questions": [
    {{
      "stem": "the question as the student reads it",
      "type": "mcq | true_false | short_answer | numeric",
      "options": ["string"],
      "correct": "string",
      "explanation": "string",
      "source_ref": "P1",
      "answer_key": {{}}
    }}
  ]
}}
"""

_LANGUAGES = {
    "ar": "Write the questions in Arabic.",
    "en": "Write the questions in English.",
}


def format_passages(passages) -> str:
    """The passages, numbered so a candidate can name the one it came from.

    The label is what `source_ref` has to match, so it is short and impossible
    to confuse with a page number: `P1`, `P2`, … The page reference is printed
    beside it because a citation the instructor reads has to say *slide 7 of the
    OOP deck*, not *passage 3*.
    """
    blocks = []
    for position, passage in enumerate(passages, start=1):
        label = " (OCR transcription)" if passage.from_ocr else ""
        blocks.append(
            f"[P{position}] {passage.page_ref}{label}\n{passage.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    *,
    course_name: str,
    topic_name: str,
    question_type: str,
    type_label: str,
    level: str,
    level_label: str,
    marks,
    count: int,
    passages,
    language: str = "en",
    expected_minutes: float | None = None,
) -> str:
    """The user half of the call: what to write, then what to write it from."""
    timing = (
        f"A student should need about {expected_minutes:.0f} minute(s) on it.\n"
        if expected_minutes
        else ""
    )
    return (
        f"Course: {course_name}\n"
        f"Topic: {topic_name}\n"
        f"Question type: {type_label}\n"
        f"Level: {level_label}\n"
        f"Marks per question: {marks}\n"
        f"{timing}"
        f"{_LANGUAGES.get(language, '')}\n\n"
        f"{TYPE_RULES.get(question_type, '')}\n"
        f"{LEVEL_RULES.get(level, '')}\n\n"
        f"{ANSWER_KEY_RULE}\n"
        f"{KEY_RULES.get(question_type, '')}\n\n"
        f"Write exactly {count} question(s), all of this type and level, each "
        "about a different thing the passages say.\n\n"
        f"{GROUNDING_RULE}\n\n"
        "Passages:\n\n"
        f"{format_passages(passages)}"
    )
