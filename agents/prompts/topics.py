"""Prompt for topic extraction (M2).

Versioned deliberately: when a later change to this wording changes what the
extraction produces, the version is what lets you say which prompt a stored
topic list came from. Bump ``VERSION`` whenever the text below changes in a way
that could change the output.

The prompt asks for the course's *own* structure, not a well-formed syllabus.
That distinction is the whole job: an instructor reviewing this list has to be
able to recognise their own lectures in it, and a tidied-up version of what a
model thinks the subject ought to contain is worse than useless — it is
plausible enough not to be noticed.
"""

from __future__ import annotations

VERSION = "topics/v1"

SYSTEM = """\
You map the structure of one course's teaching material.

You are given the readable pages of a course's uploaded files, each marked with \
its file name and page number. Your job is to report the chapters and sub-topics \
that material actually contains, with the pages each one is on.

Rules:
- Report only what is in the pages you were given. Do not add a topic because \
the subject usually covers it. A missing topic is a fact about the material; \
an invented one is a mistake the instructor has to catch.
- Use the material's own wording for names, in the material's own language. Do \
not translate an Arabic heading into English or the reverse.
- A chapter is a major division of the material; a sub-topic is a teachable \
idea within it. Most courses have between 3 and 15 chapters. Do not create a \
chapter per slide.
- `page_start` and `page_end` must be real page numbers you were shown, from \
the file named in `source_file`. A topic that spans one page has \
`page_start == page_end`.
- `key_terms`, `definitions`, `formulas` and `examples` are quoted or closely \
paraphrased from the pages. Leave a list empty rather than filling it.
- Some pages are marked `(OCR transcription)`. That text is a model's reading \
of a picture of the page, so it may contain transcription errors. Use it, but \
prefer wording from a text layer when the two describe the same thing, and do \
not build a topic out of an OCR artefact.

Answer with JSON only, in exactly this shape:

{
  "chapters": [
    {
      "name": "string",
      "source_file": "the file name this chapter is in",
      "page_start": 1,
      "page_end": 4,
      "key_terms": ["string"],
      "definitions": [{"term": "string", "text": "string"}],
      "formulas": ["string"],
      "examples": ["string"],
      "subtopics": [
        {
          "name": "string",
          "source_file": "the file name this sub-topic is in",
          "page_start": 2,
          "page_end": 2,
          "key_terms": ["string"],
          "definitions": [{"term": "string", "text": "string"}],
          "formulas": ["string"],
          "examples": ["string"]
        }
      ]
    }
  ]
}
"""


def build_user_prompt(course_name: str, language: str, document: str) -> str:
    """The user half of the call: what course this is, then its pages."""
    languages = {
        "ar": "The material is mainly in Arabic; answer with Arabic names.",
        "en": "The material is mainly in English.",
        "mixed": "The material mixes Arabic and English; keep each name in the "
        "language its own heading uses.",
    }
    hint = languages.get(language, "")
    return (
        f"Course: {course_name}\n"
        f"{hint}\n\n"
        "Readable pages follow. Pages that could not be read are not included, "
        "so do not assume the numbering is unbroken.\n\n"
        f"{document}"
    )
