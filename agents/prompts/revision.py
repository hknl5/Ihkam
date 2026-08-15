"""What the instructor's revision buttons actually say to Agent 2A (M11).

There is no new prompt here, and that is the point. M8 already established the
channel by which a question is asked for *again, differently*: Agent 3A's
rejection notes are handed to Agent 2A as the brief for the replacement
(`agents.prompts.generate.format_rejections`). "Make this easier" is the same
kind of instruction from a different author — the instructor rather than the
reviewer — so it travels the same way, as a note, into the same call.

Writing it as a second prompt would mean a second place where question rules
live, and the first time the type rules or the grounding rule changed, one of
the two would be forgotten.

Each brief is written the way M8 requires a rejection note to be written: an
instruction to the question writer about what the replacement must *do*, not a
description of what was wrong. `VERSION` is bumped when the wording changes in
a way that could change what comes back.
"""

from __future__ import annotations

VERSION = "revision/v1"

REGENERATE = "regenerate"
EASIER = "easier"
HARDER = "harder"
CLARIFY = "clarify"

#: The brief each button sends. Phrased as instructions, and deliberately silent
#: about the topic, the type, the level and the passages — those come from the
#: blueprint row, exactly as they do on a first generation. A revision that
#: could change the topic would be a different question, not a revision.
BRIEFS = {
    REGENERATE: (
        "Write a different question on this topic. It must not be a rewording of "
        "the question below — change what the student has to do, not only how it "
        "is phrased."
    ),
    EASIER: (
        "Write an easier question on the same topic, at the same marks and the "
        "same question type. Reduce the number of steps the student has to take "
        "and the amount they must hold in mind at once; keep the same cognitive "
        "level the blueprint asked for and stay within the supplied passages. Do "
        "not make it easier by making it vaguer."
    ),
    HARDER: (
        "Write a more demanding question on the same topic, at the same marks and "
        "the same question type. Require more of the student — an extra step, an "
        "application rather than a restatement, a distractor that a careless "
        "reader would take. Do not make it harder by making it longer, by adding "
        "trick wording, or by depending on anything outside the supplied passages."
    ),
    CLARIFY: (
        "Write the same question again with wording that has exactly one "
        "reasonable interpretation. Keep what the student is asked to do "
        "unchanged: same topic, same type, same level, same marks, same answer if "
        "the answer was right. Remove ambiguity, state any condition the answer "
        "depends on, and make the question verb match what the student must "
        "produce."
    ),
}

#: What each button is called on the screen, and what the toast says afterwards.
#: Kept beside the briefs so a button and the instruction it sends cannot drift
#: apart — §5's rule that a button saying *Approve* produces *Approved*.
LABELS = {
    REGENERATE: ("Regenerate", "Regenerated"),
    EASIER: ("Make easier", "Made easier"),
    HARDER: ("Make harder", "Made harder"),
    CLARIFY: ("Clarify wording", "Wording clarified"),
}

MODES = tuple(BRIEFS)


def brief_for(mode: str, *, stem: str = "") -> list[str]:
    """The note(s) this revision sends to Agent 2A, in M8's own note format.

    The current stem goes in as a second note rather than as prose inside the
    first: `format_rejections` already knows how to show the writer what not to
    repeat, and a replacement that has not seen the question it replaces will
    happily hand back the same one.
    """
    if mode not in BRIEFS:
        raise KeyError(f"unknown revision mode {mode!r}")
    notes = [BRIEFS[mode]]
    if stem:
        notes.append(
            "The question being replaced is quoted below. Do not return it, or a "
            "reworded copy of it."
        )
    return notes


__all__ = [
    "BRIEFS",
    "CLARIFY",
    "EASIER",
    "HARDER",
    "LABELS",
    "MODES",
    "REGENERATE",
    "VERSION",
    "brief_for",
]
