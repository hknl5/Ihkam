"""How close two assembled papers are — measured honestly (M10).

M9 built two forms out of one blueprint and could therefore *guarantee* seven
dimensions. This module measures the things that are not guaranteed, and it is
built around one refusal: **it never says how equivalent the forms are.**

There is no measurement basis for that claim. Equivalence is a property of what
students score, and no student has sat this paper. Every number here is an
indicator of something countable — minutes estimated, steps in a key, marks per
chapter, questions carrying a formula — shown as itself, beside the same number
for the other form, with the difference stated. An instructor reading two
columns can decide whether a difference matters. A single "97% equivalent" would
take that decision away and would be, in the strict sense, made up.

Two naming rules follow from that and are enforced by the suite:

* **"expected", never "actual".** Expected difficulty, expected time. What is
  computed from a question is what the question *is*, not what it does to a
  student. `estimate_minutes` (M9) already held this line; difficulty joins it.
* **no equivalence percentage, anywhere.** Percentages appear only where they
  are a share of something real — this chapter is 30% of the paper's marks —
  never as a score for how alike the forms are. `dishonest_claims` is the
  machine-readable form of this rule, and the tests run it over the rendered
  screen as well as over the data.

Expected difficulty is deliberately **four proxies rather than one number**:
worked steps, length, formula presence, option complexity. Combining them into
a single index would require weights nobody can justify, and would hide the one
thing the instructor needs — *which* proxy diverged. Four columns, four
tolerances, and a note that names the proxy that moved.

The leakage check is the only place a model is asked anything, and it is
two-stage on purpose:

1. **A lenient net.** Embeddings shortlist pairs, at a threshold low enough to
   be almost permissive, *plus* two channels that do not depend on embeddings at
   all — one question's stem echoing the other's answer, and rare terms shared
   between two stems. This matters because the worst leaks are not semantically
   close: a question that states a constant in passing and a later question that
   asks for that constant embed far apart and leak completely. A pre-filter that
   trusted cosine distance alone would miss exactly the pairs worth catching.
2. **The verdict.** For each shortlisted pair, the model is asked whether one
   question actually reveals the other's answer. Leakage is not similarity, so
   the vector distance never decides — it only decides what gets read. A pair
   the net caught and the model cleared is not reported.

Question similarity is reported separately for the same reason. Two questions
asking nearly the same thing is a real finding — an instructor may want the
variety back — but it is a different finding from a leak, and merging them would
make both unreadable.

The shortlist is ranked and read down to a cap (`MAX_JUDGED_PAIRS`), because a
real 18-question paper has 153 pairs on it and one press of a button should not
be 300 calls. What the cap leaves unread is *said*, not swallowed: those pairs
are reported as unchecked, and unchecked is not clean.

Nothing here writes. The deterministic half makes no call at all; the semantic
half is asked for deliberately, because it costs one embedding call and one
completion per pair actually read.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ValidationError, model_validator

from ..models import BlueprintRow
from .forms import Dimension, FormPlan, Item, compare

logger = logging.getLogger(__name__)

CENT = Decimal("0.01")
HUNDRED = Decimal("100")


def _q(value, exp=CENT) -> Decimal:
    return Decimal(value or 0).quantize(exp, rounding=ROUND_HALF_UP)


class ConvergenceError(RuntimeError):
    """A check could not be completed. No verdict is better than a guessed one."""


# --- Thresholds, stated ------------------------------------------------------
#
# Every number below is a decision, so each one says what it means. None of them
# is a quality bar the forms "pass" — they decide when a difference is worth an
# instructor's attention, and the raw values are shown either way.

#: Cosine similarity above which two questions are called near-duplicates. High
#: on purpose: this is a claim that two questions ask nearly the same thing, and
#: at 0.75 half a well-written paper would trip it, because every question in a
#: course shares that course's vocabulary.
SIMILARITY_HIGH = Decimal("0.88")

#: Cosine similarity above which a pair is merely *read* by the leakage stage.
#: Deliberately lenient — sending a handful of innocent pairs to the model costs
#: a few calls, and missing a real leak costs an exam. Nothing is reported as a
#: leak on the strength of this number; it only opens the door.
LEAKAGE_PREFILTER = Decimal("0.30")

#: The most pairs one report will send to the model. A real 18-question paper
#: has 153 pairs on it and a lenient net catches a large share of them, so a cap
#: is what keeps one press of the button from being 300 calls. It is a budget,
#: not a verdict. `select_pairs_to_judge` splits it between the two channels so
#: neither buries the other, and everything past it is reported as *unchecked*
#: rather than quietly counted as clean.
MAX_JUDGED_PAIRS = 16

#: An answer longer than this is not treated as something another question can
#: "state". Overlap with a long prose answer is a resemblance — which the
#: similarity channel already catches — and treating it as an echo puts every
#: pair in the course on the shortlist.
ECHO_MAX_ANSWER_TOKENS = 6

#: Shared rare vocabulary that makes a pair worth reading: this many terms, and
#: at least this share of the two stems' combined vocabulary.
SHARED_TERMS_MINIMUM = 5
SHARED_TERMS_RATIO = Decimal("0.2")

#: How far the expected-difficulty proxies may differ between forms before the
#: cell is called divergent. Absolute, in the proxy's own unit — a paper with
#: one more worked step on it is not a different paper; five more is.
STEP_TOLERANCE = Decimal("2")
FORMULA_TOLERANCE = Decimal("1")
OPTION_TOLERANCE = Decimal("2")
#: Reading length is compared as a share, because 40 words apart means one thing
#: on a 200-word paper and nothing on a 2,000-word one.
LENGTH_TOLERANCE_RATIO = Decimal("0.15")

#: Expected time is compared against the exam's limit as well as against the
#: other form. Over the limit is a note; far under it is also a note, because a
#: 60-minute paper that is expected to take 25 is not the paper that was
#: specified.
TIME_UNDERRUN_RATIO = Decimal("0.6")

#: Tokens too common to be evidence of anything when two stems share them.
STOPWORDS = frozenset(
    """
    the a an and or of to in on for with from by is are was were be been being that this these
    those it its as at which what who whom whose when where how why not no if then than
    following true false all none both each every any some question questions answer answers
    given below above value values using use used calculate find state explain describe
    consider assume suppose show determine
    في من على إلى عن أن إن ما هو هي هذا هذه ذلك التي الذي كان يكون مع كل بين عند أي هل
    """.split()
)

#: What counts as a formula for the length-and-symbols proxy. Not a parser — the
#: claim is only "this question has notation in it", which is a real signal about
#: what a student has to do with a pen.
FORMULA_PATTERN = re.compile(
    r"(\\\\[a-zA-Z]+|\$[^$]+\$|[=≠≤≥±∑∏∫√∞≈]|\b\d+\s*[-+*/^×÷]\s*\d+|\b[a-zA-Z]\s*[=]\s*\S)"
)

WORD_PATTERN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


# --- The rule, in code -------------------------------------------------------

#: Claims this milestone is not allowed to make, as patterns. Run over the data
#: *and* over the rendered screen by the suite. Kept here rather than in the
#: tests because it is a property of the product, not of the test: anyone adding
#: a template or a serializer field can check it against the same list.
DISHONEST_CLAIMS = (
    (re.compile(r"actual\s+difficult", re.IGNORECASE), 'says "actual difficulty"'),
    (re.compile(r"actual\s+(time|duration|minutes)", re.IGNORECASE), 'says "actual time"'),
    (re.compile(r"real\s+difficult", re.IGNORECASE), 'says "real difficulty"'),
    (
        re.compile(r"\d+(?:\.\d+)?\s*%\s*\w{0,12}\s*(equivalent|equivalence|identical|the\s+same)",
                   re.IGNORECASE),
        "puts a percentage on equivalence",
    ),
    (
        re.compile(r"(equivalen\w*|similarit\w*|match\w*|convergence)\s*(score|index|rating|"
                   r"percentage)", re.IGNORECASE),
        "scores equivalence",
    ),
    (
        re.compile(r"\d+(?:\.\d+)?\s*%\s*(similar|alike|match)", re.IGNORECASE),
        "puts a percentage on similarity",
    ),
)


def dishonest_claims(text: str) -> list[str]:
    """Every forbidden claim `text` makes, named. Empty means it is honest.

    The naming discipline is a hard rule, and a hard rule that is only written
    in a docstring is a hope. This is what makes it checkable.
    """
    return [description for pattern, description in DISHONEST_CLAIMS if pattern.search(text or "")]


# --- One question, on one paper ---------------------------------------------


@dataclass(frozen=True)
class QuestionRef:
    """How a question is named to an instructor: Q8A — question 8 on Form A."""

    form_label: str
    number: int
    question_id: int

    @property
    def code(self) -> str:
        return f"Q{self.number}{self.form_label}"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "form": self.form_label,
            "number": self.number,
            "question_id": self.question_id,
        }


@dataclass(frozen=True)
class Placed:
    """One question as it sits on one paper, with everything the checks read.

    Detached from the ORM for the same reason M9's `Item` is: the checks are then
    functions over plain values, and the two that cost money take a provider
    rather than a queryset.
    """

    ref: QuestionRef
    topic_name: str
    question_type: str
    level: str
    marks: Decimal
    minutes: Decimal
    stem: str = ""
    options: tuple = ()
    answer_text: str = ""
    explanation: str = ""
    steps: int = 0
    needs_mark_review: bool = False
    from_ocr: bool = False

    # -- The expected-difficulty proxies, one property each --------------------

    @property
    def words(self) -> int:
        """How much there is to read — stem and options together."""
        return len(WORD_PATTERN.findall(self.stem)) + sum(
            len(WORD_PATTERN.findall(str(option))) for option in self.options
        )

    @property
    def has_formula(self) -> bool:
        return bool(FORMULA_PATTERN.search(self.stem)) or any(
            FORMULA_PATTERN.search(str(option)) for option in self.options
        )

    @property
    def option_load(self) -> int:
        """Options past the usual four — the part of an MCQ that is extra work."""
        return max(len(self.options) - 4, 0)

    @property
    def worked_steps(self) -> int:
        """Steps the answer key takes past the first."""
        return max(self.steps - 1, 0)

    @property
    def embed_text(self) -> str:
        """What is embedded: what the question *asks*, options included.

        The answer is left out on purpose. Similarity is about what a student is
        asked to do; the answer belongs to the leakage stage, where it is what
        the other question might be handing over.
        """
        return "\n".join([self.stem, *(str(option) for option in self.options)]).strip()

    @property
    def search_text(self) -> str:
        """Stem and options, for the lexical channels of the leakage pre-filter."""
        return self.embed_text

    def as_dict(self) -> dict:
        return {
            "ref": self.ref.as_dict(),
            "topic": self.topic_name,
            "question_type": self.question_type,
            "level": self.level,
            "marks": str(self.marks),
            "expected_minutes": str(self.minutes),
            "words": self.words,
            "has_formula": self.has_formula,
            "worked_steps": self.worked_steps,
            "option_load": self.option_load,
        }


def _answer_text(question) -> str:
    """The question's answer as a string, however its key is shaped (M6).

    Read out of the stored dict rather than through `Question.key`, because a
    key that fails validation must not stop a comparison screen from rendering —
    the answer is then simply the `correct` column, which is always there.
    """
    key = question.answer_key if isinstance(question.answer_key, dict) else {}
    for field_name in ("final_answer", "answer", "model_answer"):
        value = key.get(field_name)
        if value:
            return str(value)
    return str(question.correct or "")


def _steps(question) -> int:
    key = question.answer_key if isinstance(question.answer_key, dict) else {}
    return len(key.get("steps") or [])


def placed_from_entry(entry, *, form_label: str, number: int, level: str) -> Placed:
    """One `FormQuestion` as a `Placed`. The level comes from the blueprint row."""
    question = entry.question
    return Placed(
        ref=QuestionRef(form_label=form_label, number=number, question_id=question.pk),
        topic_name=(
            entry.blueprint_row.topic.name
            if entry.blueprint_row and entry.blueprint_row.topic_id
            else (
                question.blueprint_row.topic.name
                if question.blueprint_row and question.blueprint_row.topic_id
                else ""
            )
        ),
        question_type=question.question_type,
        level=level,
        marks=_q(entry.marks),
        minutes=_q(entry.expected_minutes),
        stem=question.stem or "",
        options=tuple(question.options or ()),
        answer_text=_answer_text(question),
        explanation=question.explanation or "",
        steps=_steps(question),
        needs_mark_review=question.mark_sum_ok is False,
        from_ocr=bool(question.from_ocr),
    )


# --- One paper ---------------------------------------------------------------


@dataclass
class FormProfile:
    """One assembled paper, reduced to what the comparison reads."""

    label: str
    questions: list[Placed] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def total_marks(self) -> Decimal:
        return _q(sum((q.marks for q in self.questions), Decimal("0")))

    @property
    def expected_minutes(self) -> Decimal:
        return _q(sum((q.minutes for q in self.questions), Decimal("0")))

    @property
    def topics(self) -> list[str]:
        return sorted({q.topic_name for q in self.questions if q.topic_name})

    def count_where(self, predicate) -> int:
        return sum(1 for q in self.questions if predicate(q))

    def count_level(self, level: str) -> int:
        return self.count_where(lambda q: q.level == level)

    def count_type(self, question_type: str) -> int:
        return self.count_where(lambda q: q.question_type == question_type)

    def topic_marks(self, topic: str) -> Decimal:
        return _q(sum((q.marks for q in self.questions if q.topic_name == topic), Decimal("0")))

    def topic_count(self, topic: str) -> int:
        return self.count_where(lambda q: q.topic_name == topic)

    def topic_share(self, topic: str) -> Decimal:
        """This topic's share of the paper's marks, as a percentage.

        A percentage of something real: marks on a chapter over marks on the
        paper. Not a claim about the two forms — those never get a percentage.
        """
        total = self.total_marks
        if not total:
            return Decimal("0")
        return _q(self.topic_marks(topic) / total * HUNDRED)

    # -- The four expected-difficulty proxies, summed over the paper ----------

    @property
    def total_words(self) -> int:
        return sum(q.words for q in self.questions)

    @property
    def total_worked_steps(self) -> int:
        return sum(q.worked_steps for q in self.questions)

    @property
    def formula_count(self) -> int:
        return self.count_where(lambda q: q.has_formula)

    @property
    def option_load(self) -> int:
        return sum(q.option_load for q in self.questions)

    @property
    def demanding_questions(self) -> list[Placed]:
        """Questions that are heavy on more than one proxy at once.

        Not a difficulty score and not a ranking — a shortlist. A question that
        is long *and* multi-step *and* carries notation is the kind of item that
        makes one paper feel different from another, and naming those items is
        more use to an instructor than averaging them into an index.
        """
        if not self.questions:
            return []
        long_words = max(_median(q.words for q in self.questions) * 1.5, 40)
        heavy = []
        for question in self.questions:
            signals = sum(
                [
                    question.worked_steps >= 2,
                    question.words >= long_words,
                    question.has_formula and question.level == BlueprintRow.Level.MULTI_STEP,
                    question.option_load >= 1,
                ]
            )
            if signals >= 2:
                heavy.append(question)
        return heavy

    def as_plan(self) -> FormPlan:
        """This profile in M9's shape, so M9's `compare` runs on it unchanged."""
        return FormPlan(
            label=self.label,
            items=[
                Item(
                    question_id=q.ref.question_id,
                    topic_id=None,
                    topic_name=q.topic_name,
                    question_type=q.question_type,
                    level=q.level,
                    marks=q.marks,
                    minutes=q.minutes,
                    stem=q.stem,
                )
                for q in self.questions
            ],
        )


def _median(values) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2


def profiles_from_forms(forms) -> list[FormProfile]:
    """Read saved `Form` rows into profiles. One query per form, no calls."""
    profiles = []
    for form in forms:
        entries = form.entries.select_related(
            "question", "blueprint_row", "blueprint_row__topic", "question__blueprint_row"
        ).order_by("position", "pk")
        questions = []
        for number, entry in enumerate(entries, start=1):
            row = entry.blueprint_row or entry.question.blueprint_row
            questions.append(
                placed_from_entry(
                    entry,
                    form_label=form.label,
                    number=number,
                    level=row.level if row else BlueprintRow.Level.MEDIUM,
                )
            )
        profiles.append(FormProfile(label=form.label, questions=questions))
    return profiles


# --- The side-by-side table --------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One line of the comparison table: a label, a value per form, a verdict.

    `divergent` is what the screen colors. Nothing else on the table is colored,
    because color here means "these two differ by more than the tolerance beside
    them" and nothing else — §5's rule that color signals divergence rather than
    decorating a table.
    """

    key: str
    label: str
    values: dict
    difference: str = ""
    tolerance: str = ""
    divergent: bool = False
    unit: str = ""
    group: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "values": {k: str(v) for k, v in self.values.items()},
            "difference": self.difference,
            "tolerance": self.tolerance,
            "divergent": self.divergent,
            "unit": self.unit,
            "group": self.group,
            "note": self.note,
        }


def _numeric_row(
    key, label, profiles, getter, *, tolerance=Decimal("0"), unit="", group="", note=""
) -> Row:
    values = {p.label: getter(p) for p in profiles}
    numbers = [Decimal(str(v)) for v in values.values()]
    spread = _q(max(numbers) - min(numbers)) if numbers else Decimal("0")
    return Row(
        key=key,
        label=label,
        values=values,
        difference=str(_trim(spread)),
        tolerance=str(_trim(_q(tolerance))),
        divergent=spread > tolerance,
        unit=unit,
        group=group,
        note=note,
    )


def _trim(value: Decimal) -> Decimal:
    """Drop trailing zeros so a table of counts does not read as 3.00 questions."""
    value = Decimal(value)
    if value == value.to_integral_value():
        return value.to_integral_value()
    return value.normalize()


# --- Coverage ----------------------------------------------------------------


@dataclass
class Coverage:
    """Which topics each form covers, and where the two disagree.

    M9's row-based assembly makes this true by construction, which is exactly
    why it is checked rather than assumed: a check that can only pass is the one
    that catches the day the assembly changes.
    """

    topics: list[str] = field(default_factory=list)
    per_form: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)
    uneven: list[str] = field(default_factory=list)

    @property
    def is_even(self) -> bool:
        return not any(self.missing.values()) and not self.uneven

    def as_dict(self) -> dict:
        return {
            "topics": self.topics,
            "per_form": {k: dict(v) for k, v in self.per_form.items()},
            "missing": {k: list(v) for k, v in self.missing.items()},
            "uneven": list(self.uneven),
            "is_even": self.is_even,
        }


def check_coverage(profiles: list[FormProfile]) -> Coverage:
    topics = sorted({topic for profile in profiles for topic in profile.topics})
    per_form = {p.label: {topic: p.topic_count(topic) for topic in topics} for p in profiles}
    missing = {
        p.label: [topic for topic in topics if not p.topic_count(topic)] for p in profiles
    }
    uneven = [
        topic
        for topic in topics
        if len({p.topic_count(topic) for p in profiles}) > 1
        and not any(topic in missing[p.label] for p in profiles)
    ]
    return Coverage(topics=topics, per_form=per_form, missing=missing, uneven=uneven)


# --- Expected time -----------------------------------------------------------


@dataclass
class Timing:
    """Expected time per form against the limit the instructor set.

    "Expected", and only ever that. It is M9's `estimate_minutes` summed over the
    paper — built from type, level, reading length and steps in the key. Nobody
    has been timed.
    """

    limit_minutes: int
    per_form: dict = field(default_factory=dict)
    over: list = field(default_factory=list)
    under: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "limit_minutes": self.limit_minutes,
            "expected_minutes_per_form": {k: str(v) for k, v in self.per_form.items()},
            "over_limit": list(self.over),
            "well_under_limit": list(self.under),
        }


def check_time(profiles: list[FormProfile], *, limit_minutes: int) -> Timing:
    per_form = {p.label: p.expected_minutes for p in profiles}
    limit = Decimal(limit_minutes or 0)
    over = [label for label, minutes in per_form.items() if limit and minutes > limit]
    under = [
        label
        for label, minutes in per_form.items()
        if limit and minutes < limit * TIME_UNDERRUN_RATIO
    ]
    return Timing(limit_minutes=limit_minutes or 0, per_form=per_form, over=over, under=under)


# --- Semantic checks: the shared vector work ---------------------------------


def cosine(first, second) -> Decimal:
    """Cosine similarity of two vectors, as a Decimal in [-1, 1]."""
    if not first or not second or len(first) != len(second):
        return Decimal("0")
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    left = math.sqrt(sum(a * a for a in first))
    right = math.sqrt(sum(b * b for b in second))
    if not left or not right:
        return Decimal("0")
    return _q(Decimal(str(dot / (left * right))), Decimal("0.0001"))


def embed_questions(questions: list[Placed], *, provider) -> dict:
    """One embedding call for every question on both papers, keyed by question id."""
    if not questions:
        return {}
    texts = [q.embed_text for q in questions]
    try:
        vectors = provider.embed(texts)
    except Exception as exc:  # noqa: BLE001 — surfaced as itself, never as a pass
        raise ConvergenceError(f"The questions could not be embedded: {exc}") from exc
    if len(vectors) != len(questions):
        raise ConvergenceError(
            f"Embedding returned {len(vectors)} vectors for {len(questions)} questions."
        )
    return {q.ref.question_id: vector for q, vector in zip(questions, vectors, strict=True)}


@dataclass(frozen=True)
class Pair:
    """Two questions, and how they were noticed."""

    first: Placed
    second: Placed
    similarity: Decimal = Decimal("0")
    signals: tuple = ()

    @property
    def same_form(self) -> bool:
        return self.first.ref.form_label == self.second.ref.form_label

    @property
    def codes(self) -> str:
        return f"{self.first.ref.code} / {self.second.ref.code}"


# --- Question similarity -----------------------------------------------------


@dataclass(frozen=True)
class SimilarPair:
    """Two questions that ask nearly the same thing.

    Its own indicator, never merged into leakage: near-duplicates cost variety,
    leaks cost the answer, and an instructor fixes them differently.
    """

    first: QuestionRef
    second: QuestionRef
    similarity: Decimal
    same_form: bool
    first_stem: str = ""
    second_stem: str = ""

    @property
    def message(self) -> str:
        where = (
            f"on Form {self.first.form_label}"
            if self.same_form
            else "across the two forms"
        )
        return (
            f"{self.first.code} and {self.second.code} ask nearly the same thing "
            f"({where}). Similar wording is not leakage — but if both are meant to "
            f"test different things, one of them is not earning its place."
        )

    def as_dict(self) -> dict:
        return {
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "similarity": str(self.similarity),
            "same_form": self.same_form,
            "message": self.message,
        }


def find_similar_pairs(
    questions: list[Placed], vectors: dict, *, threshold: Decimal = SIMILARITY_HIGH
) -> list[SimilarPair]:
    """Every pair — within a form and across the forms — above the threshold."""
    found = []
    for index, first in enumerate(questions):
        for second in questions[index + 1 :]:
            if first.ref.question_id == second.ref.question_id:
                continue  # a shared question is one question, not two similar ones
            score = cosine(
                vectors.get(first.ref.question_id), vectors.get(second.ref.question_id)
            )
            if score >= threshold:
                found.append(
                    SimilarPair(
                        first=first.ref,
                        second=second.ref,
                        similarity=score,
                        same_form=first.ref.form_label == second.ref.form_label,
                        first_stem=first.stem,
                        second_stem=second.stem,
                    )
                )
    return sorted(found, key=lambda pair: -pair.similarity)


# --- Leakage, stage one: the lenient net -------------------------------------


def content_tokens(text: str) -> set:
    """The words in `text` worth comparing — long enough, and not furniture."""
    return {
        token.lower()
        for token in WORD_PATTERN.findall(text or "")
        if len(token) > 3 and token.lower() not in STOPWORDS
    }


def _answer_echo(source: Placed, target: Placed) -> bool:
    """Does `source`'s text hand over `target`'s answer, word for word?

    The channel that does not go through embeddings, and the reason the
    pre-filter can be called lenient rather than merely low-threshold. A stem
    that states "the maximum depth is 7" and a later question asking for that
    depth are nowhere near each other in vector space and leak completely.
    """
    answer = (target.answer_text or "").strip()
    if not answer:
        return False
    haystack = source.search_text.lower()

    numbers = set(NUMBER_PATTERN.findall(answer))
    tokens = content_tokens(answer)

    if len(tokens) > ECHO_MAX_ANSWER_TOKENS:
        # A long prose answer that overlaps another question is a resemblance,
        # not an echo — and the similarity channel already has that pair. Left
        # here, it fires on almost every pair in a course whose questions share
        # a vocabulary, and a channel that flags everything ranks nothing.
        return False
    if not numbers and not tokens:
        return False

    # The claim being made is that the other question *states this answer* — so
    # the answer has to appear essentially whole. Both halves are required when
    # both exist: "175 logic gates" is echoed by a stem that says 175 about
    # logic gates, not by every stem that happens to contain the digit 2.
    if numbers and not all(
        re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", haystack) for number in numbers
    ):
        return False
    if tokens:
        echoed = tokens & content_tokens(source.search_text)
        needed = len(tokens) if len(tokens) <= 3 else math.ceil(len(tokens) * 0.75)
        if len(echoed) < needed:
            return False
    return True


def _shared_terms(first: Placed, second: Placed) -> bool:
    """Whether two stems share enough rare vocabulary to be worth reading.

    Both a count and a share: every question in one course shares that course's
    terms, so an absolute overlap on its own puts the whole paper on the
    shortlist and leaves the ranking meaningless.
    """
    left, right = content_tokens(first.search_text), content_tokens(second.search_text)
    if not left or not right:
        return False
    shared = left & right
    if len(shared) < SHARED_TERMS_MINIMUM:
        return False
    return Decimal(len(shared)) / Decimal(len(left | right)) >= SHARED_TERMS_RATIO


def shortlist_leak_candidates(
    profiles: list[FormProfile],
    vectors: dict,
    *,
    threshold: Decimal = LEAKAGE_PREFILTER,
) -> list[Pair]:
    """Pairs on the *same paper* worth reading, by any of three channels.

    Leakage is a property of one paper: a student sits Form A, and only what is
    printed on Form A can help them. Cross-form resemblance is a similarity
    finding, not a leak, and is reported as one.

    The three channels are unioned, never intersected — a pair gets in on any
    one of them. Nothing is dropped here; the whole shortlist is returned,
    **ranked**, with the answer-echo pairs first. Capping is the caller's job
    (`MAX_JUDGED_PAIRS`) and the caller reports what it did not read, because a
    pair nobody looked at is unchecked rather than clean, and a net that quietly
    swallows its own catch is not a lenient net.
    """
    candidates: list[Pair] = []
    for profile in profiles:
        questions = profile.questions
        for index, first in enumerate(questions):
            for second in questions[index + 1 :]:
                if first.ref.question_id == second.ref.question_id:
                    continue
                signals = []
                score = cosine(
                    vectors.get(first.ref.question_id), vectors.get(second.ref.question_id)
                )
                if score >= threshold:
                    signals.append("similarity")
                if _answer_echo(first, second) or _answer_echo(second, first):
                    signals.append("answer_echo")
                if _shared_terms(first, second):
                    signals.append("shared_terms")
                if signals:
                    candidates.append(
                        Pair(
                            first=first,
                            second=second,
                            similarity=score,
                            signals=tuple(signals),
                        )
                    )

    return sorted(candidates, key=_rank)


def _rank(pair: Pair):
    """Echo first, then closest first, then paper order. Deterministic."""
    return (
        0 if "answer_echo" in pair.signals else 1,
        -pair.similarity,
        pair.first.ref.number,
        pair.second.ref.number,
    )


def select_pairs_to_judge(candidates: list[Pair], *, limit: int = MAX_JUDGED_PAIRS) -> tuple:
    """Split the budget between the two channels. Returns (read, not read).

    A single ranking cannot serve both channels, and the reason is worth stating
    because it took a real paper to find it. Rank echo-first and a paper with
    seventy weak echo pairs buries the closest pairs on it; rank by similarity
    alone and the far-apart pair that states another's answer — the case the
    echo channel exists for — never gets read at all.

    So the budget is halved: the strongest echo pairs, and the closest pairs,
    each get their own share, and whatever one channel does not use goes to the
    other. No weighting between the two is invented, because there is no honest
    exchange rate between "these look alike" and "this one says the other's
    answer".
    """
    ordered = sorted(candidates, key=_rank)
    if len(ordered) <= limit:
        return ordered, []

    echoes = [pair for pair in ordered if "answer_echo" in pair.signals]
    closest = sorted(
        (pair for pair in ordered if "answer_echo" not in pair.signals),
        key=lambda pair: (-pair.similarity, pair.first.ref.number, pair.second.ref.number),
    )
    share = limit // 2
    picked = echoes[:share] + closest[: limit - min(share, len(echoes))]
    chosen = {id(pair) for pair in picked[:limit]}
    read = [pair for pair in ordered if id(pair) in chosen]
    return read, [pair for pair in ordered if id(pair) not in chosen]


# --- Leakage, stage two: the verdict -----------------------------------------


class LeakOut(BaseModel):
    """The model's verdict on one pair, checked before it is used."""

    leaks: bool
    direction: str = "none"
    reason: str = ""

    @model_validator(mode="after")
    def _check(self):
        self.direction = (self.direction or "none").strip().lower()
        self.reason = (self.reason or "").strip()
        if self.direction not in {"a_reveals_b", "b_reveals_a", "both", "none"}:
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.leaks and self.direction == "none":
            raise ValueError("a leak with no direction")
        if self.leaks and not self.reason:
            # A leak with no reason cannot be shown to an instructor. Retrying is
            # cheaper than inventing one — the same rule M7 holds.
            raise ValueError("a leak with no reason")
        if not self.leaks:
            self.direction = "none"
        return self


@dataclass(frozen=True)
class Leak:
    """A confirmed leak: one question that helps answer another, and how.

    Only the model puts a record in this list. The vector distance that led here
    is carried for the debug view and is *not* the evidence — the same distance
    on a pair the model cleared produces nothing at all.
    """

    source: QuestionRef
    target: QuestionRef
    reason: str
    similarity: Decimal = Decimal("0")
    signals: tuple = ()

    @property
    def message(self) -> str:
        return f"{self.source.code} may help answer {self.target.code}."

    def as_dict(self) -> dict:
        return {
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "reason": self.reason,
            "similarity": str(self.similarity),
            "signals": list(self.signals),
            "message": self.message,
        }


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _loads(text: str) -> dict:
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def judge_pair(pair: Pair, *, course_name: str, provider) -> LeakOut:
    """Ask the model whether this pair leaks. One call, one retry on bad JSON."""
    from agents.prompts.leakage import SYSTEM, build_user_prompt

    user = build_user_prompt(
        course_name=course_name,
        form_label=pair.first.ref.form_label,
        first=pair.first,
        second=pair.second,
    )
    last_error = ""
    for attempt in (1, 2):
        try:
            response = provider.complete(SYSTEM, user, json_mode=True, temperature=0.0)
            return LeakOut.model_validate(_loads(response.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"the answer was not the expected JSON ({exc.__class__.__name__})"
            logger.warning("Leakage attempt %s returned bad JSON: %s", attempt, exc)
        except Exception as exc:  # noqa: BLE001 — a call that never landed is not a pass
            raise ConvergenceError(f"The leakage check did not complete: {exc}") from exc
    raise ConvergenceError(
        f"The leakage verdict on {pair.codes} could not be read after two attempts: "
        f"{last_error}. The pair is reported as unjudged, not as clean."
    )


def confirm_leaks(candidates: list[Pair], *, course_name: str, provider) -> tuple:
    """Run stage two over the shortlist. Returns (confirmed leaks, unjudged pairs).

    A pair the model could not be read on comes back in the second list rather
    than being dropped: "we could not tell" and "there is no leak" are different
    sentences, and the screen prints the one that is true.
    """
    leaks, unjudged = [], []
    for pair in candidates:
        try:
            verdict = judge_pair(pair, course_name=course_name, provider=provider)
        except ConvergenceError as exc:
            unjudged.append((pair, str(exc)))
            continue
        if not verdict.leaks:
            continue
        if verdict.direction in {"a_reveals_b", "both"}:
            leaks.append(
                Leak(
                    source=pair.first.ref,
                    target=pair.second.ref,
                    reason=verdict.reason,
                    similarity=pair.similarity,
                    signals=pair.signals,
                )
            )
        if verdict.direction in {"b_reveals_a", "both"}:
            leaks.append(
                Leak(
                    source=pair.second.ref,
                    target=pair.first.ref,
                    reason=verdict.reason,
                    similarity=pair.similarity,
                    signals=pair.signals,
                )
            )
    return leaks, unjudged


# --- إحكام notes -------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    """One sentence in إحكام's voice, pointing at one thing worth a look.

    Every note names something countable that was found — never a general
    observation, never filler to fill a panel. An empty list is a legitimate
    result and the screen says so plainly.
    """

    kind: str
    text: str
    tone: str = "info"  # info | attention | danger
    refs: tuple = ()

    #: What each kind is called on the screen. Written out rather than derived
    #: from the key so the wording is chosen, not generated — "Expected time",
    #: never "Expected_time".
    LABELS = {
        "leakage": "Answer leakage",
        "leakage_unjudged": "Leakage unchecked",
        "similarity": "Near-duplicate",
        "coverage": "Topic coverage",
        "level_spread": "Cognitive level",
        "type_distribution": "Question type",
        "expected_difficulty": "Expected difficulty",
        "expected_time": "Expected time",
        "clarity": "Worth a second read",
        "sharing": "Shared questions",
    }

    @property
    def label(self) -> str:
        return self.LABELS.get(self.kind, self.kind.replace("_", " ").capitalize())

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "tone": self.tone, "refs": list(self.refs)}


def _level_label(level: str) -> str:
    return dict(BlueprintRow.Level.choices).get(level, level).lower()


def _type_label(question_type: str) -> str:
    return dict(BlueprintRow.QuestionType.choices).get(question_type, question_type).lower()


def _count_note(profiles, counter, *, kind, singular, plural) -> list[Note]:
    """"Form B has one extra multi-step question" — for any countable thing.

    Both counts are stated after the sentence, because "one extra" is only
    alarming next to the number it is extra to: one extra out of two is half the
    paper, one extra out of twelve is a rounding.
    """
    if len(profiles) != 2:
        return []
    first, second = profiles
    left, right = counter(first), counter(second)
    if left == right:
        return []
    more, fewer = (first, second) if left > right else (second, first)
    difference = abs(left - right)
    noun = singular if difference == 1 else plural
    word = "one" if difference == 1 else str(difference)
    return [
        Note(
            kind=kind,
            text=(
                f"Form {more.label} has {word} extra {noun} — "
                f"{counter(more)} against {counter(fewer)} on Form {fewer.label}."
            ),
            tone="attention",
        )
    ]


def build_notes(
    profiles: list[FormProfile],
    *,
    coverage: Coverage,
    timing: Timing,
    rows: list[Row],
    leaks: list[Leak],
    similar: list[SimilarPair],
    unjudged: list = (),
    unread: list = (),
    shared_question_ids=frozenset(),
) -> list[Note]:
    """What إحكام has to say about these two papers, in the order it matters.

    Leaks first: a leak is the only finding here that can cost a student marks
    they earned. Then divergence between the papers, then items on one paper
    worth a second read, then the informational ones.
    """
    notes: list[Note] = []

    for leak in leaks:
        notes.append(
            Note(
                kind="leakage",
                text=f"{leak.message} {leak.reason}".strip(),
                tone="danger",
                refs=(leak.source.code, leak.target.code),
            )
        )

    for pair, error in unjudged:
        notes.append(
            Note(
                kind="leakage_unjudged",
                text=(
                    f"{pair.codes} was shortlisted for a leakage check and could not be "
                    f"judged, so it is unchecked rather than clear. {error}"
                ),
                tone="attention",
                refs=(pair.first.ref.code, pair.second.ref.code),
            )
        )

    if unread:
        n = len(unread)
        notes.append(
            Note(
                kind="leakage_unjudged",
                text=(
                    f"{n} more pair{'s' if n != 1 else ''} were shortlisted and not read: "
                    f"one comparison reads at most {MAX_JUDGED_PAIRS} pairs, most suspicious "
                    f"first. Those pairs are unchecked, which is not the same as clear."
                ),
                tone="attention",
            )
        )

    for pair in similar:
        notes.append(
            Note(
                kind="similarity",
                text=pair.message,
                tone="attention",
                refs=(pair.first.code, pair.second.code),
            )
        )

    # Coverage, then each divergent cell of the table, said as a sentence.
    for label, missing in coverage.missing.items():
        for topic in missing:
            notes.append(
                Note(
                    kind="coverage",
                    text=(
                        f"Form {label} covers nothing from “{topic}”, which the other "
                        f"form examines. Both papers should spend the blueprint the "
                        f"same way."
                    ),
                    tone="danger",
                )
            )

    if len(profiles) == 2:
        for level, _display in BlueprintRow.Level.choices:
            notes.extend(
                _count_note(
                    profiles,
                    lambda p, level=level: p.count_level(level),
                    kind="level_spread",
                    singular=f"{_level_label(level)} question",
                    plural=f"{_level_label(level)} questions",
                )
            )
        for question_type, _display in BlueprintRow.QuestionType.choices:
            notes.extend(
                _count_note(
                    profiles,
                    lambda p, question_type=question_type: p.count_type(question_type),
                    kind="type_distribution",
                    singular=f"{_type_label(question_type)} question",
                    plural=f"{_type_label(question_type)} questions",
                )
            )
        notes.extend(
            _count_note(
                profiles,
                lambda p: p.formula_count,
                kind="expected_difficulty",
                singular="question carrying a formula",
                plural="questions carrying formulas",
            )
        )

    for row in rows:
        if row.divergent and row.key == "expected_minutes":
            values = ", ".join(f"Form {k} about {v} min" for k, v in row.values.items())
            notes.append(
                Note(
                    kind="expected_time",
                    text=(
                        f"Expected time differs by more than the {row.tolerance}-minute "
                        f"tolerance: {values}. This is a planning estimate from what the "
                        f"questions are, not a measurement of how long students take."
                    ),
                    tone="attention",
                )
            )

    for label in timing.over:
        notes.append(
            Note(
                kind="expected_time",
                text=(
                    f"Form {label}'s expected time is about {timing.per_form[label]} minutes "
                    f"against a {timing.limit_minutes}-minute limit."
                ),
                tone="danger",
            )
        )
    for label in timing.under:
        notes.append(
            Note(
                kind="expected_time",
                text=(
                    f"Form {label} is expected to take about {timing.per_form[label]} minutes "
                    f"of the {timing.limit_minutes} allowed — well short of the paper that "
                    f"was specified."
                ),
                tone="attention",
            )
        )

    # Items worth a second read. Both flags are facts already recorded about the
    # question, not opinions formed here.
    for profile in profiles:
        for question in profile.questions:
            if question.needs_mark_review:
                notes.append(
                    Note(
                        kind="clarity",
                        text=(
                            f"{question.ref.code} needs a clarity check: its answer key's "
                            f"steps do not add up to the marks the question carries."
                        ),
                        tone="attention",
                        refs=(question.ref.code,),
                    )
                )
    # OCR-sourced questions are collapsed into one note rather than one each.
    # On a course whose material was scanned this is *every* question, and
    # twenty-seven identical sentences would bury the six findings above them —
    # a note that fires on everything points at nothing.
    from_ocr = [
        question.ref.code
        for profile in profiles
        for question in profile.questions
        if question.from_ocr
    ]
    if from_ocr:
        n = len(from_ocr)
        listed = ", ".join(from_ocr[:6]) + (" …" if n > 6 else "")
        notes.append(
            Note(
                kind="clarity",
                text=(
                    f"{n} question{'s' if n != 1 else ''} cite a scanned page read by OCR "
                    f"({listed}) — worth reading the wording against the original before "
                    f"the paper is printed."
                ),
                tone="info",
                refs=tuple(from_ocr),
            )
        )

    if shared_question_ids:
        n = len(shared_question_ids)
        notes.append(
            Note(
                kind="sharing",
                text=(
                    f"{n} question{'s' if n != 1 else ''} appear{'' if n != 1 else 's'} on "
                    f"both forms — the sharing you allowed on the exam spec."
                ),
                tone="info",
            )
        )

    return notes


# --- The report --------------------------------------------------------------


@dataclass
class ConvergenceReport:
    """Everything M10 measured about two papers, and nothing it did not.

    Read the docstring at the top of this module for what is deliberately
    absent: there is no equivalence figure on this object, and adding one would
    be the one change that makes the milestone dishonest.
    """

    exam_title: str = ""
    course_name: str = ""
    profiles: list[FormProfile] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    timing: Timing = field(default_factory=lambda: Timing(limit_minutes=0))
    rows: list[Row] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    leaks: list[Leak] = field(default_factory=list)
    similar_pairs: list[SimilarPair] = field(default_factory=list)
    unjudged_pairs: list = field(default_factory=list)
    unread_pairs: list = field(default_factory=list)
    shortlisted_pairs: int = 0
    judged_pairs: int = 0
    semantic_ran: bool = False
    semantic_error: str = ""
    shared_question_ids: frozenset = frozenset()

    @property
    def labels(self) -> list[str]:
        return [profile.label for profile in self.profiles]

    @property
    def divergent_rows(self) -> list[Row]:
        return [row for row in self.rows if row.divergent]

    @property
    def has_findings(self) -> bool:
        return bool(self.notes)

    @property
    def summary(self) -> str:
        """What the screen says at the top. An indicator count, never a score."""
        if len(self.profiles) < 2:
            return "There is only one form, so there is nothing to compare it with."
        divergent = len(self.divergent_rows)
        parts = [
            f"{len(self.rows)} indicators compared across {len(self.profiles)} forms",
            (
                f"{divergent} diverge beyond tolerance"
                if divergent
                else "none diverge beyond tolerance"
            ),
        ]
        if self.semantic_ran:
            parts.append(
                f"{len(self.leaks)} confirmed leak{'s' if len(self.leaks) != 1 else ''}"
                f" and {len(self.similar_pairs)} near-duplicate pair"
                f"{'s' if len(self.similar_pairs) != 1 else ''}"
            )
        else:
            parts.append("leakage and similarity not checked yet")
        return ". ".join(parts) + "."

    def as_dict(self) -> dict:
        return {
            "exam": self.exam_title,
            "forms": [
                {
                    "label": p.label,
                    "question_count": p.question_count,
                    "total_marks": str(p.total_marks),
                    "expected_minutes": str(p.expected_minutes),
                    "expected_difficulty": {
                        "worked_steps": p.total_worked_steps,
                        "reading_words": p.total_words,
                        "questions_with_formulas": p.formula_count,
                        "options_beyond_four": p.option_load,
                        "demanding_questions": [q.ref.code for q in p.demanding_questions],
                    },
                    "questions": [q.as_dict() for q in p.questions],
                }
                for p in self.profiles
            ],
            "coverage": self.coverage.as_dict(),
            "expected_time": self.timing.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
            "dimensions": [d.as_dict() for d in self.dimensions],
            "notes": [note.as_dict() for note in self.notes],
            "leakage": {
                "checked": self.semantic_ran,
                "shortlisted_pairs": self.shortlisted_pairs,
                "judged_pairs": self.judged_pairs,
                "confirmed": [leak.as_dict() for leak in self.leaks],
                "unjudged": [
                    {"pair": pair.codes, "error": error} for pair, error in self.unjudged_pairs
                ],
                "unread": [pair.codes for pair in self.unread_pairs],
            },
            "similarity": {
                "checked": self.semantic_ran,
                "pairs": [pair.as_dict() for pair in self.similar_pairs],
            },
            "summary": self.summary,
            "semantic_error": self.semantic_error,
        }


def build_rows(profiles: list[FormProfile], *, timing: Timing, dimensions) -> list[Row]:
    """The side-by-side table, one row at a time, each with its own tolerance.

    Grouped so the screen can head the sections: what the papers are, how they
    are spread, what they are expected to demand, and how long they should take.
    """
    if not profiles:
        return []

    time_tolerance = next(
        (d.tolerance for d in dimensions if d.name == "expected_minutes"), Decimal("2")
    )
    rows = [
        _numeric_row("question_count", "Questions", profiles, lambda p: p.question_count,
                     unit="questions", group="The paper"),
        _numeric_row("total_marks", "Total marks", profiles, lambda p: p.total_marks,
                     unit="marks", group="The paper"),
    ]

    for topic in sorted({topic for p in profiles for topic in p.topics}):
        rows.append(
            _numeric_row(
                f"topic:{topic}",
                topic,
                profiles,
                lambda p, topic=topic: p.topic_count(topic),
                unit="questions",
                group="Per chapter",
                note=" · ".join(
                    f"Form {p.label} {p.topic_share(topic)}% of marks" for p in profiles
                ),
            )
        )

    for level, display in BlueprintRow.Level.choices:
        rows.append(
            _numeric_row(
                f"level:{level}",
                display,
                profiles,
                lambda p, level=level: p.count_level(level),
                unit="questions",
                group="Cognitive level",
            )
        )

    for question_type, display in BlueprintRow.QuestionType.choices:
        if not any(p.count_type(question_type) for p in profiles):
            continue
        rows.append(
            _numeric_row(
                f"type:{question_type}",
                display,
                profiles,
                lambda p, question_type=question_type: p.count_type(question_type),
                unit="questions",
                group="Question type",
            )
        )

    # Expected difficulty — four proxies, never combined into one figure.
    lengths = [Decimal(p.total_words) for p in profiles]
    mean_words = sum(lengths, Decimal("0")) / Decimal(len(lengths)) if lengths else Decimal("0")
    rows.extend(
        [
            _numeric_row(
                "steps",
                "Solution steps (past the first)",
                profiles,
                lambda p: p.total_worked_steps,
                tolerance=STEP_TOLERANCE,
                unit="steps",
                group="Expected difficulty",
                note="Counted from the answer keys, not judged by a model.",
            ),
            _numeric_row(
                "words",
                "Words to read",
                profiles,
                lambda p: p.total_words,
                tolerance=_q(mean_words * LENGTH_TOLERANCE_RATIO, Decimal("1")),
                unit="words",
                group="Expected difficulty",
            ),
            _numeric_row(
                "formulas",
                "Questions carrying a formula",
                profiles,
                lambda p: p.formula_count,
                tolerance=FORMULA_TOLERANCE,
                unit="questions",
                group="Expected difficulty",
            ),
            _numeric_row(
                "option_load",
                "Options beyond four",
                profiles,
                lambda p: p.option_load,
                tolerance=OPTION_TOLERANCE,
                unit="options",
                group="Expected difficulty",
            ),
            _numeric_row(
                "demanding",
                "Items heavy on two proxies or more",
                profiles,
                lambda p: len(p.demanding_questions),
                tolerance=Decimal("1"),
                unit="questions",
                group="Expected difficulty",
                note=" · ".join(
                    f"Form {p.label}: "
                    + (", ".join(q.ref.code for q in p.demanding_questions) or "none")
                    for p in profiles
                ),
            ),
            _numeric_row(
                "expected_minutes",
                "Expected time",
                profiles,
                lambda p: p.expected_minutes,
                tolerance=time_tolerance,
                unit="minutes",
                group="Expected time",
                note=(
                    f"Against a {timing.limit_minutes}-minute limit."
                    if timing.limit_minutes
                    else ""
                ),
            ),
        ]
    )
    return rows


def report_for_forms(
    forms,
    *,
    exam=None,
    limit_minutes: int = 0,
    course_name: str = "",
    provider=None,
    check_semantics: bool = False,
    similarity_threshold: Decimal = SIMILARITY_HIGH,
    prefilter: Decimal = LEAKAGE_PREFILTER,
) -> ConvergenceReport:
    """Compare saved forms. Deterministic by default; semantic when asked.

    `check_semantics` is what costs money — one embedding call for the paper and
    one completion per shortlisted pair — so it is a parameter rather than a
    default, and the report says plainly whether it ran.
    """
    profiles = profiles_from_forms(forms)
    if exam is not None:
        limit_minutes = limit_minutes or exam.duration_minutes
        course_name = course_name or exam.course.name

    coverage = check_coverage(profiles)
    timing = check_time(profiles, limit_minutes=limit_minutes)
    dimensions = compare([p.as_plan() for p in profiles])
    rows = build_rows(profiles, timing=timing, dimensions=dimensions)

    counted: dict = {}
    for profile in profiles:
        for question in profile.questions:
            counted[question.ref.question_id] = counted.get(question.ref.question_id, 0) + 1
    shared = frozenset(qid for qid, n in counted.items() if n > 1)

    report = ConvergenceReport(
        exam_title=exam.display_title if exam is not None else "",
        course_name=course_name,
        profiles=profiles,
        coverage=coverage,
        timing=timing,
        rows=rows,
        dimensions=dimensions,
        shared_question_ids=shared,
    )

    leaks: list[Leak] = []
    similar: list[SimilarPair] = []
    unjudged: list = []
    if check_semantics and profiles:
        all_questions = [q for profile in profiles for q in profile.questions]
        try:
            if provider is None:
                from agents.provider import get_provider

                provider = get_provider()
            vectors = embed_questions(all_questions, provider=provider)
            similar = find_similar_pairs(
                all_questions, vectors, threshold=similarity_threshold
            )
            candidates = shortlist_leak_candidates(profiles, vectors, threshold=prefilter)
            report.shortlisted_pairs = len(candidates)
            # The cap bounds what one report costs. What it excludes is said out
            # loud rather than left looking clean.
            judged, deferred = select_pairs_to_judge(candidates)
            report.judged_pairs = len(judged)
            report.unread_pairs = deferred
            leaks, unjudged = confirm_leaks(
                judged, course_name=course_name, provider=provider
            )
            report.semantic_ran = True
        except ConvergenceError as exc:
            report.semantic_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — reported as itself, never as clean
            report.semantic_error = f"The semantic checks did not complete: {exc}"

    report.leaks = leaks
    report.similar_pairs = similar
    report.unjudged_pairs = unjudged
    report.notes = build_notes(
        profiles,
        coverage=coverage,
        timing=timing,
        rows=rows,
        leaks=leaks,
        similar=similar,
        unjudged=unjudged,
        unread=report.unread_pairs,
        shared_question_ids=shared,
    )
    return report


def report_for_exam(exam, *, provider=None, check_semantics: bool = False) -> ConvergenceReport:
    """The comparison for one exam's saved forms."""
    forms = list(exam.forms.all())
    return report_for_forms(
        forms, exam=exam, provider=provider, check_semantics=check_semantics
    )


__all__ = [
    "DISHONEST_CLAIMS",
    "LEAKAGE_PREFILTER",
    "MAX_JUDGED_PAIRS",
    "SIMILARITY_HIGH",
    "ConvergenceError",
    "ConvergenceReport",
    "Coverage",
    "Leak",
    "LeakOut",
    "Note",
    "Pair",
    "Placed",
    "QuestionRef",
    "Row",
    "SimilarPair",
    "Timing",
    "build_notes",
    "build_rows",
    "check_coverage",
    "check_time",
    "confirm_leaks",
    "content_tokens",
    "cosine",
    "dishonest_claims",
    "embed_questions",
    "find_similar_pairs",
    "judge_pair",
    "profiles_from_forms",
    "report_for_exam",
    "report_for_forms",
    "select_pairs_to_judge",
    "shortlist_leak_candidates",
]
