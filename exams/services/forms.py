"""Form assembly: two papers from one blueprint, similar by construction (M9).

No model is called here, and none is needed. Two forms are equivalent when they
spend the same blueprint the same way, and a blueprint already says exactly how
it is spent: one row per (topic, question type, level, count, marks). So the
assembly is done **per row** — every form takes `row.count` questions out of
that row's reviewed pool — and seven of the eight dimensions M9 must match on
fall out of that arithmetic rather than out of a search:

* question count — each form takes the same count from each row;
* score distribution — the questions of one row are worth the same marks;
* topic distribution — a row belongs to one topic;
* question-type distribution — a row is one type;
* cognitive-level spread — a row is one level;
* multi-step count — multi-step *is* a level, so it is counted by the above;
* numeric count — numeric *is* a type, likewise.

That is what "similar by construction, not by luck" means concretely: those
seven cannot drift, because nothing in this module is free to make them drift.

The eighth dimension is the one that is genuinely free. Two MCQs on the same
topic at the same level are worth the same marks and take different amounts of
time to answer, and *which* of a row's surplus questions goes to A and which to
B is a real choice. That choice — and only that choice — is made by scoring:
`estimate_minutes` puts a number on each question, and the allocator deals the
row's questions so the two forms' expected time comes out close.

Three things are deliberate:

* **A shortfall is reported, never absorbed.** Fully-separate forms need twice
  the questions, and the over-generation surplus from M8's loop is not
  guaranteed to cover a second paper. When a row cannot fill both forms, this
  module says which form, which topic, and how many are missing — and does not
  save a form with a hole in it. `save_assembly` refuses an incomplete assembly
  outright, because a paper that is quietly one question short is worse than no
  paper.
* **Sharing is the instructor's decision, not a fallback.** Nothing here
  silently reuses a question to rescue a thin pool. It reuses one only when the
  exam is set to allow it, which is a choice made on the spec screen before any
  of this runs.
* **The greedy allocator is a seam, not an answer.** `allocate` is a plain
  callable with a documented signature; passing a different one to `distribute`
  replaces the strategy entirely. A later optimizer (simulated annealing over
  the whole exam rather than greedy within a row, or a solver minimising several
  dimensions at once) drops in there without touching the reporting, the
  models, or the screens.

The pool this reads is M8's: questions that a recorded review passed. A question
the instructor has since rejected is out — approval is theirs, and so is
rejection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal

from ..models import BlueprintRow, Exam, Form, FormQuestion, ItemRun, Question

#: The MVP cap. Two forms is what a paper hall needs and what a single blueprint
#: can honestly fill; a third form is a third of the pool again, and there is no
#: evidence yet that the loop can produce it. Raising this is a decision with a
#: generation cost attached, so it is a constant rather than an argument.
MAX_FORMS = 2

QUARTER = Decimal("0.25")
CENT = Decimal("0.01")
HUNDRED_CENTS = Decimal("100")


def _q(value, exp=CENT) -> Decimal:
    return Decimal(value or 0).quantize(exp, rounding=ROUND_HALF_UP)


def _quarter(value) -> Decimal:
    """To the nearest quarter minute — the finest an estimate this rough earns.

    `Decimal.quantize(Decimal("0.25"))` does not do this: it matches the
    *exponent*, so it rounds to two places and leaves 1.51 alone. Rounding to a
    step means dividing by it, and the difference matters here — a per-question
    estimate carried to the cent would read as a precision this does not have.
    """
    return (Decimal(value or 0) / QUARTER).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * QUARTER


def mark_shares(total: Decimal, count: int) -> list[Decimal]:
    """Split a row's marks into `count` parts that add back to exactly `total`.

    The same largest-remainder rule `blueprint.auto_build` uses on the exam's
    score, applied to one row's: 14 marks over 3 questions is 4.67 / 4.67 / 4.66,
    never 4.67 three times — which would put a form on 14.01 and make the paper
    disagree with the blueprint it was built from.

    The shares depend only on the row, so both forms are priced identically
    however their questions differ.
    """
    if count <= 0:
        return []
    cents = int((_q(total) * HUNDRED_CENTS).to_integral_value(rounding=ROUND_HALF_UP))
    base, remainder = divmod(cents, count)
    return [
        (Decimal(base + (1 if index < remainder else 0)) / HUNDRED_CENTS).quantize(CENT)
        for index in range(count)
    ]



class FormAssemblyError(RuntimeError):
    """The forms could not be assembled at all — not "assembled with gaps"."""


# --- Expected time -----------------------------------------------------------
#
# Named "expected", and only ever that. This is a planning estimate built from
# what the question *is* — its type, its level, how much there is to read, how
# many steps its key takes — not a measurement of what students do. M10 holds
# the same naming rule for difficulty; it starts here.

#: Minutes a question of each type costs before anything about this particular
#: question is taken into account.
BASE_MINUTES = {
    BlueprintRow.QuestionType.TRUE_FALSE: Decimal("0.75"),
    BlueprintRow.QuestionType.MCQ: Decimal("1.5"),
    BlueprintRow.QuestionType.SHORT_ANSWER: Decimal("3"),
    BlueprintRow.QuestionType.NUMERIC: Decimal("4"),
}

#: How much the cognitive level multiplies that by. A direct-recall MCQ is read
#: and answered; a multi-step one is worked.
LEVEL_FACTOR = {
    BlueprintRow.Level.DIRECT: Decimal("0.8"),
    BlueprintRow.Level.MEDIUM: Decimal("1"),
    BlueprintRow.Level.MULTI_STEP: Decimal("1.3"),
}

#: Words a student reads per minute, for the stem and its options. Deliberately
#: conservative — this is exam prose, read carefully, in a hall.
READING_WORDS_PER_MINUTE = Decimal("180")

#: Each worked step past the first in a numeric key, and each MCQ option past
#: the usual four, adds this much.
PER_EXTRA_STEP = Decimal("0.5")
PER_EXTRA_OPTION = Decimal("0.25")


def estimate_minutes(
    *,
    question_type: str,
    level: str,
    stem: str = "",
    options=None,
    answer_key=None,
) -> Decimal:
    """The expected time for one question, in minutes, to the nearest quarter.

    Built from four things that can be counted rather than judged: the type, the
    level, the length of what has to be read, and the number of steps the key
    takes. Nothing here asks a model how hard a question is — that would be an
    opinion dressed as a number, and M10's honesty rule forbids exactly that.
    """
    base = BASE_MINUTES.get(question_type, Decimal("2"))
    minutes = base * LEVEL_FACTOR.get(level, Decimal("1"))

    words = len((stem or "").split())
    for option in options or []:
        words += len(str(option).split())
    minutes += Decimal(words) / READING_WORDS_PER_MINUTE

    steps = len((answer_key or {}).get("steps") or [])
    if steps > 1:
        minutes += PER_EXTRA_STEP * (steps - 1)

    extra_options = max(len(options or []) - 4, 0)
    minutes += PER_EXTRA_OPTION * extra_options

    return _quarter(minutes)


# --- What is being distributed ----------------------------------------------


@dataclass(frozen=True)
class Item:
    """One question in the pool, reduced to the dimensions assembly cares about.

    Detached from the database on purpose: the allocator is then a pure function
    over plain values, testable without fixtures and replaceable without knowing
    anything about Django. `question` is carried along only so the caller can
    save what was chosen.
    """

    question_id: int
    topic_id: int | None
    topic_name: str
    question_type: str
    level: str
    marks: Decimal
    minutes: Decimal
    stem: str = ""
    question: object | None = None

    @property
    def is_multi_step(self) -> bool:
        return self.level == BlueprintRow.Level.MULTI_STEP

    @property
    def is_numeric(self) -> bool:
        return self.question_type == BlueprintRow.QuestionType.NUMERIC

    @classmethod
    def from_question(cls, question: Question, *, marks: Decimal, level: str) -> "Item":
        """Build an item from a stored question and the row it was written for.

        Marks and level come from the blueprint row rather than the question,
        because the row is what the exam's arithmetic was validated against —
        a question does not carry its own price.
        """
        return cls(
            question_id=question.pk,
            topic_id=question.blueprint_row.topic_id if question.blueprint_row else None,
            topic_name=question.blueprint_row.topic.name if question.blueprint_row else "",
            question_type=question.question_type,
            level=level,
            marks=_q(marks),
            minutes=estimate_minutes(
                question_type=question.question_type,
                level=level,
                stem=question.stem,
                options=question.options,
                answer_key=question.answer_key,
            ),
            stem=question.stem,
            question=question,
        )


@dataclass(frozen=True)
class RowDemand:
    """What one blueprint row asks each form for.

    The unit of assembly. Every form gets `count` questions from this row — that
    is the whole guarantee about topic, type, level and marks, stated once.
    """

    row_id: int | None
    topic_id: int | None
    topic_name: str
    question_type: str
    level: str
    count: int
    marks_per_question: Decimal = Decimal("0")
    position: int = 0
    #: The row's marks in total. Kept alongside the per-question figure because
    #: the two are not interchangeable when the split is not exact: 14 over 3 is
    #: 4.67 a question, and three of those is 14.01. `mark_shares` prices the
    #: block from this, so a form's marks equal the blueprint's exactly.
    marks_total: Decimal | None = None

    @property
    def block_marks(self) -> Decimal:
        """What this row contributes to one form."""
        if self.marks_total is not None:
            return _q(self.marks_total)
        return _q(self.marks_per_question * self.count)

    def shares(self) -> list[Decimal]:
        return mark_shares(self.block_marks, self.count)

    @property
    def label(self) -> str:
        """How this row is named to an instructor in a shortfall message."""
        type_name = dict(BlueprintRow.QuestionType.choices).get(
            self.question_type, self.question_type
        )
        level_name = dict(BlueprintRow.Level.choices).get(self.level, self.level)
        return f"{self.topic_name} ({type_name.lower()}, {level_name.lower()})"

    @classmethod
    def from_row(cls, row: BlueprintRow) -> "RowDemand":
        return cls(
            row_id=row.pk,
            topic_id=row.topic_id,
            topic_name=row.topic.name,
            question_type=row.question_type,
            level=row.level,
            count=row.count,
            marks_per_question=row.marks_per_question,
            position=row.position,
            marks_total=row.marks,
        )


# --- The allocator seam ------------------------------------------------------


def greedy_balanced(
    items: list[Item], *, per_form: int, form_count: int, allow_sharing: bool
) -> list[list[Item]]:
    """Deal one row's pool across the forms, balancing expected time.

    **This is the seam.** Any callable with this signature can replace it —
    `distribute(..., allocate=my_optimizer)`. What a replacement must honour is
    only what the caller relies on: return one list per form, never more than
    `per_form` items in a list, never the same question twice in one list, and
    never reuse a question across lists unless `allow_sharing`. Returning short
    lists is allowed and is how a thin pool is reported rather than hidden.

    The strategy itself is longest-processing-time first: sort by expected
    minutes descending, and give each question to whichever form has room and
    the least time on it so far. Greedy, deterministic, and good enough at this
    size — a row is a handful of questions, and the difference between greedy
    and optimal on a handful is measured in seconds of expected time. A real
    optimizer becomes worth writing when it can balance *across* rows at once,
    which is why this seam takes the row's pool rather than the whole exam's.

    Ties break on marks then question id, so the same pool always assembles into
    the same two forms — an instructor who rebuilds must not get a new paper.

    When the row cannot fill every form, the shortfall is **concentrated on the
    later form** rather than spread evenly. Two papers each missing one question
    are two papers that cannot be sat; one complete paper and one that is short
    by two is a finished Form A and a specific, countable thing to generate for
    Form B. The message an instructor gets is the difference between "something
    is missing everywhere" and "Form B needs two more on Recursion".
    """
    ordered = sorted(items, key=lambda item: (-item.minutes, -item.marks, item.question_id))
    buckets: list[list[Item]] = [[] for _ in range(form_count)]
    totals = [Decimal("0") for _ in range(form_count)]

    capacity = [per_form] * form_count
    if len(ordered) < per_form * form_count:
        remaining = len(ordered)
        for index in range(form_count):
            capacity[index] = min(per_form, remaining)
            remaining -= capacity[index]

    def _open(index: int) -> bool:
        return len(buckets[index]) < capacity[index]

    # Pass one: distinct questions only, spread by time. This runs in both modes
    # — sharing is a permission to reuse when the pool runs out, not a licence
    # to put the same question on both papers while alternatives sit unused.
    for item in ordered:
        candidates = [i for i in range(form_count) if _open(i)]
        if not candidates:
            break
        pick = min(candidates, key=lambda i: (totals[i], len(buckets[i]), i))
        buckets[pick].append(item)
        totals[pick] += item.minutes

    # Pass two: fill what is left from questions already placed, if allowed. A
    # form still never carries the same question twice, so a pool smaller than
    # one form's demand is short even here — and is reported as such.
    if allow_sharing:
        for index in range(form_count):
            # Up to `per_form` here, not up to `capacity`: capacity is the share
            # of *distinct* questions this form was allotted, and sharing exists
            # precisely to go past it.
            while len(buckets[index]) < per_form:
                placed = {item.question_id for item in buckets[index]}
                available = [item for item in ordered if item.question_id not in placed]
                if not available:
                    break
                item = min(
                    available,
                    key=lambda it: (totals[index] + it.minutes, -it.marks, it.question_id),
                )
                buckets[index].append(item)
                totals[index] += item.minutes

    return buckets


# --- The result --------------------------------------------------------------


@dataclass
class FormPlan:
    """One assembled paper: its label and the questions on it, in row order."""

    label: str
    items: list[Item] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len(self.items)

    @property
    def total_marks(self) -> Decimal:
        return _q(sum((item.marks for item in self.items), Decimal("0")))

    @property
    def expected_minutes(self) -> Decimal:
        return _quarter(sum((item.minutes for item in self.items), Decimal("0")))

    @property
    def multi_step_count(self) -> int:
        return sum(1 for item in self.items if item.is_multi_step)

    @property
    def numeric_count(self) -> int:
        return sum(1 for item in self.items if item.is_numeric)

    @property
    def question_ids(self) -> set[int]:
        return {item.question_id for item in self.items}

    def counts_by(self, attribute: str) -> dict[str, int]:
        return dict(Counter(getattr(item, attribute) for item in self.items))

    def marks_by_topic(self) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = {}
        for item in self.items:
            totals[item.topic_name] = _q(totals.get(item.topic_name, Decimal("0")) + item.marks)
        return totals


@dataclass(frozen=True)
class Shortfall:
    """One form, one row, and the questions that are not there.

    Carries the numbers as well as the sentence, so the screen can show a count
    and the tests can assert on one without matching prose.
    """

    form_label: str
    topic_name: str
    question_type: str
    level: str
    required: int
    available_in_pool: int
    missing: int
    sharing_allowed: bool
    row_label: str = ""

    @property
    def fix(self) -> str:
        """What the instructor can actually do about it — both routes, named."""
        if self.sharing_allowed:
            return (
                "Sharing is already on, so this row is short for a single paper: "
                "run generation again for this topic."
            )
        return (
            "Run generation again for this topic, or let the forms share questions "
            "(Exam settings → How the forms relate)."
        )

    @property
    def message(self) -> str:
        plural = "s" if self.missing != 1 else ""
        return (
            f"Form {self.form_label} is short {self.missing} question{plural} in "
            f"“{self.topic_name}” — {self.available_in_pool} in the reviewed pool, "
            f"{self.required} needed. {self.fix}"
        )


@dataclass
class Assembly:
    """Two forms out of one blueprint, plus everything that is missing.

    Never partially true: if `shortfalls` is non-empty the forms in `forms` are
    what *could* be filled, and `is_complete` is False. Nothing downstream may
    treat those as papers — `save_assembly` will not write them.
    """

    forms: list[FormPlan] = field(default_factory=list)
    shortfalls: list[Shortfall] = field(default_factory=list)
    sharing_allowed: bool = False
    exam: object = None
    pool_size: int = 0

    @property
    def is_complete(self) -> bool:
        return bool(self.forms) and not self.shortfalls

    @property
    def form_count(self) -> int:
        return len(self.forms)

    @property
    def shared_question_ids(self) -> set[int]:
        """Questions that appear on more than one form."""
        counter: Counter = Counter()
        for form in self.forms:
            counter.update(form.question_ids)
        return {question_id for question_id, n in counter.items() if n > 1}

    @property
    def shared_count(self) -> int:
        return len(self.shared_question_ids)

    @property
    def summary(self) -> str:
        if not self.forms:
            return "No forms could be assembled."
        if self.shortfalls:
            n = len(self.shortfalls)
            return (
                f"{n} row{'s' if n != 1 else ''} cannot be filled, so no form was saved. "
                f"Each one says what is missing and where."
            )
        labels = " and ".join(f"Form {form.label}" for form in self.forms)
        shared = self.shared_count
        note = (
            f" {shared} question{'s' if shared != 1 else ''} appear on both."
            if shared
            else " No question appears on both."
        )
        return f"{labels} assembled from one blueprint.{note}"

    # -- The machine-readable summary M10 renders ----------------------------

    @property
    def distribution(self) -> dict:
        """Every dimension, per form, with the spread between them.

        This is M9's deliverable to M10: the comparison screen renders this, and
        the success check reads `matches`. It is a dict of plain values rather
        than objects so it can be serialised, logged, or diffed as it stands.
        """
        return {
            "sharing_allowed": self.sharing_allowed,
            "pool_size": self.pool_size,
            "shared_question_count": self.shared_count,
            "is_complete": self.is_complete,
            "forms": [
                {
                    "label": form.label,
                    "question_count": form.question_count,
                    "total_marks": str(form.total_marks),
                    "expected_minutes": str(form.expected_minutes),
                    "multi_step_count": form.multi_step_count,
                    "numeric_count": form.numeric_count,
                    "topics": form.counts_by("topic_name"),
                    "topic_marks": {k: str(v) for k, v in form.marks_by_topic().items()},
                    "types": form.counts_by("question_type"),
                    "levels": form.counts_by("level"),
                }
                for form in self.forms
            ],
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            "matches": self.matches,
            "shortfalls": [
                {
                    "form": s.form_label,
                    "topic": s.topic_name,
                    "question_type": s.question_type,
                    "level": s.level,
                    "required": s.required,
                    "available": s.available_in_pool,
                    "missing": s.missing,
                    "message": s.message,
                }
                for s in self.shortfalls
            ],
        }

    @property
    def dimensions(self) -> list["Dimension"]:
        return compare(self.forms)

    @property
    def matches(self) -> bool:
        """True when every dimension is inside its stated tolerance."""
        return bool(self.forms) and all(d.within_tolerance for d in self.dimensions)

    @property
    def failing_dimensions(self) -> list["Dimension"]:
        return [d for d in self.dimensions if not d.within_tolerance]


# --- Comparing the forms -----------------------------------------------------
#
# Tolerances are stated, not implied. Seven dimensions are exact because the
# construction makes them exact — a non-zero spread there is a bug in this
# module, not a close call — and expected time is the one that is allowed to
# differ, because it is an estimate over questions that are genuinely different.

#: Expected time may differ by this many minutes, or this share of the paper,
#: whichever is larger. Two minutes over a 60-minute paper is inside the noise
#: of the estimate itself; a tenth of the paper is not.
TIME_TOLERANCE_MINUTES = Decimal("2")
TIME_TOLERANCE_RATIO = Decimal("0.1")


@dataclass(frozen=True)
class Dimension:
    """One thing the forms are compared on, and whether they match on it."""

    name: str
    label: str
    values: dict[str, object]
    spread: Decimal
    tolerance: Decimal
    unit: str = ""

    @property
    def within_tolerance(self) -> bool:
        return self.spread <= self.tolerance

    @property
    def by_construction(self) -> bool:
        """A dimension the assembly cannot make drift. Zero tolerance, and
        earned — not a threshold chosen to make the check pass."""
        return self.tolerance == 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "values": {k: str(v) for k, v in self.values.items()},
            "spread": str(self.spread),
            "tolerance": str(self.tolerance),
            "unit": self.unit,
            "within_tolerance": self.within_tolerance,
            "by_construction": self.by_construction,
        }


def _spread(values) -> Decimal:
    values = [Decimal(v) for v in values]
    if not values:
        return Decimal("0")
    return _q(max(values) - min(values))


def _categorical_spread(forms: list[FormPlan], attribute: str) -> Decimal:
    """The largest per-category disagreement between the forms.

    Zero means every topic (or type, or level) carries the same number of
    questions on every form. A category present on one form and absent on
    another counts as its full size, which is what it is.
    """
    tables = [form.counts_by(attribute) for form in forms]
    keys = sorted({key for table in tables for key in table})
    if not keys:
        return Decimal("0")
    return max(_spread([table.get(key, 0) for table in tables]) for key in keys)


def compare(forms: list[FormPlan]) -> list[Dimension]:
    """Every dimension M9 must match on, measured across the assembled forms."""
    if not forms:
        return []
    labels = [form.label for form in forms]

    def values(getter):
        return {label: getter(form) for label, form in zip(labels, forms, strict=True)}

    minutes = [form.expected_minutes for form in forms]
    mean_minutes = sum(minutes, Decimal("0")) / Decimal(len(minutes))
    time_tolerance = max(TIME_TOLERANCE_MINUTES, _quarter(mean_minutes * TIME_TOLERANCE_RATIO))

    return [
        Dimension(
            "question_count",
            "Question count",
            values(lambda f: f.question_count),
            _spread([f.question_count for f in forms]),
            Decimal("0"),
            "questions",
        ),
        Dimension(
            "total_marks",
            "Score distribution",
            values(lambda f: f.total_marks),
            _spread([f.total_marks for f in forms]),
            Decimal("0"),
            "marks",
        ),
        Dimension(
            "topic_distribution",
            "Topic distribution",
            values(lambda f: f.counts_by("topic_name")),
            _categorical_spread(forms, "topic_name"),
            Decimal("0"),
            "questions per topic",
        ),
        Dimension(
            "type_distribution",
            "Question-type distribution",
            values(lambda f: f.counts_by("question_type")),
            _categorical_spread(forms, "question_type"),
            Decimal("0"),
            "questions per type",
        ),
        Dimension(
            "level_spread",
            "Cognitive-level spread",
            values(lambda f: f.counts_by("level")),
            _categorical_spread(forms, "level"),
            Decimal("0"),
            "questions per level",
        ),
        Dimension(
            "multi_step_count",
            "Multi-step items",
            values(lambda f: f.multi_step_count),
            _spread([f.multi_step_count for f in forms]),
            Decimal("0"),
            "questions",
        ),
        Dimension(
            "numeric_count",
            "Numeric items",
            values(lambda f: f.numeric_count),
            _spread([f.numeric_count for f in forms]),
            Decimal("0"),
            "questions",
        ),
        Dimension(
            "expected_minutes",
            "Expected time",
            values(lambda f: f.expected_minutes),
            _spread(minutes),
            time_tolerance,
            "minutes",
        ),
    ]


# --- Assembly ----------------------------------------------------------------


def form_labels(count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(count)]


def distribute(
    demands: list[RowDemand],
    pool: dict,
    *,
    sharing_allowed: bool = False,
    form_count: int = 2,
    allocate=greedy_balanced,
) -> Assembly:
    """Spread a reviewed pool over `form_count` forms, one blueprint row at a time.

    Pure: takes plain demands and plain items, returns a plain result, touches
    no database and makes no call. `pool` maps a row id to that row's reviewed
    questions as `Item`s.

    `allocate` is the seam described on `greedy_balanced` — pass an optimizer
    here and nothing else in M9 changes.
    """
    if form_count < 1:
        raise FormAssemblyError("An exam has at least one form.")
    if form_count > MAX_FORMS:
        raise FormAssemblyError(
            f"إحكام assembles at most {MAX_FORMS} forms from one blueprint, "
            f"not {form_count}."
        )

    labels = form_labels(form_count)
    assembly = Assembly(
        forms=[FormPlan(label=label) for label in labels],
        sharing_allowed=sharing_allowed,
        pool_size=sum(len(items) for items in pool.values()),
    )

    for demand in sorted(demands, key=lambda d: (d.position, d.row_id or 0)):
        if demand.count <= 0:
            continue
        items = list(pool.get(demand.row_id) or [])
        buckets = allocate(
            items,
            per_form=demand.count,
            form_count=form_count,
            allow_sharing=sharing_allowed,
        )
        # Price the block after allocation, not before: the marks belong to the
        # row's slots on the paper, not to the questions that happen to fill
        # them. Both forms therefore carry the same shares in the same order,
        # however differently they were filled.
        shares = demand.shares()
        for label, form, bucket in zip(labels, assembly.forms, buckets, strict=True):
            form.items.extend(
                replace(item, marks=shares[index])
                for index, item in enumerate(bucket)
            )
            missing = demand.count - len(bucket)
            if missing > 0:
                assembly.shortfalls.append(
                    Shortfall(
                        form_label=label,
                        topic_name=demand.topic_name,
                        question_type=demand.question_type,
                        level=demand.level,
                        required=demand.count,
                        available_in_pool=len(items),
                        missing=missing,
                        sharing_allowed=sharing_allowed,
                        row_label=demand.label,
                    )
                )
    return assembly


# --- Reading the pool out of the database ------------------------------------


def pool_for_exam(exam: Exam) -> dict:
    """The reviewed pool, per blueprint row, as `Item`s.

    Read through M8's `ItemRun` → passed attempts rather than through `Question`
    directly: a question is in the pool because a recorded review passed it, not
    because a row exists. Questions the instructor has since rejected are left
    out — إحكام's opinion put them here, and the instructor's overrides it.
    """
    rows = {}
    if exam.has_blueprint:
        rows = {
            row.pk: row for row in exam.blueprint.rows.select_related("topic")
        }

    pool: dict = {}
    runs = (
        ItemRun.objects.filter(exam=exam)
        .exclude(blueprint_row__isnull=True)
        .select_related("blueprint_row", "blueprint_row__topic")
    )
    for run in runs:
        row = rows.get(run.blueprint_row_id) or run.blueprint_row
        questions = (
            run.approved_questions.exclude(status=Question.Status.REJECTED)
            .select_related("blueprint_row", "blueprint_row__topic")
            .order_by("position", "pk")
        )
        pool[run.blueprint_row_id] = [
            Item.from_question(question, marks=row.marks_per_question, level=row.level)
            for question in questions
        ]
    return pool


def assemble_forms(exam: Exam, *, allocate=greedy_balanced, form_count: int | None = None):
    """Assemble this exam's forms from its reviewed pool. No writes, no calls.

    The mode is the instructor's setting on the exam spec, not an argument: how
    the forms relate is a decision they made before the pool existed, and
    overriding it here would make the screen lie about what was built.
    """
    if not exam.has_blueprint:
        raise FormAssemblyError(
            f"{exam.display_title} has no blueprint, so there is nothing to assemble."
        )
    count = exam.number_of_forms if form_count is None else form_count
    demands = [
        RowDemand.from_row(row) for row in exam.blueprint.rows.select_related("topic")
    ]
    return distribute(
        demands,
        pool_for_exam(exam),
        sharing_allowed=exam.forms_may_share,
        form_count=count,
        allocate=allocate,
    )


def save_assembly(exam: Exam, assembly: Assembly) -> list[Form]:
    """Write a *complete* assembly to the database, replacing any previous one.

    Refuses an incomplete one. This is the rule M9 exists to hold: a form with a
    hole in it must never reach a screen, an export, or a student, and the only
    way to be sure of that is to never let it be written down. The caller
    reports `assembly.shortfalls` instead.
    """
    if not assembly.is_complete:
        raise FormAssemblyError(
            "This assembly is short of questions, so no form was saved. "
            + assembly.summary
        )

    exam.forms.all().delete()
    saved = []
    for position, plan in enumerate(assembly.forms):
        form = Form.objects.create(
            exam=exam,
            label=plan.label,
            position=position,
            sharing_allowed=assembly.sharing_allowed,
            expected_minutes=plan.expected_minutes,
            total_marks=plan.total_marks,
        )
        FormQuestion.objects.bulk_create(
            [
                FormQuestion(
                    form=form,
                    question=item.question,
                    blueprint_row_id=(
                        item.question.blueprint_row_id if item.question is not None else None
                    ),
                    position=index,
                    expected_minutes=item.minutes,
                    marks=item.marks,
                )
                for index, item in enumerate(plan.items)
            ]
        )
        saved.append(form)
    return saved


__all__ = [
    "MAX_FORMS",
    "Assembly",
    "Dimension",
    "FormAssemblyError",
    "FormPlan",
    "Item",
    "RowDemand",
    "Shortfall",
    "assemble_forms",
    "compare",
    "distribute",
    "estimate_minutes",
    "greedy_balanced",
    "pool_for_exam",
    "save_assembly",
]
