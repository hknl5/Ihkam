"""Searching the bank — the same `embed()` as everything else (M12).

An instructor with two hundred banked questions cannot find "the one about
halving the interval" by scrolling, and cannot find it by keyword either: they
remember what the question was *about*, not the words it used. So the bank is
searched semantically.

The one design rule here is a negative: **there is no second search stack.**
This module embeds the query through `courses.services.retrieval.embed_query`,
which goes through `agents.provider.get_provider().embed()` — the same seam the
chunks, the topics and M10's leakage check go through. That means the bank
moves to a local model on the day the rest of إحكام does, with no separate index
to migrate, no separate service to run, and no second place where a provider
key has to be configured. A dedicated search engine here would be a second
system to keep alive for a table of a few hundred rows.

Ranking is cosine distance in the database, against the vector stored when the
question was banked. Below `min_score` a banked question is not a worse answer,
it is a different subject — the same cut M3's retrieval makes, for the same
reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from pgvector.django import CosineDistance

from ..models import BankQuestion

#: How many hits a search returns. The screen is a list an instructor reads, so
#: this is a page of results rather than a candidate set for something else.
DEFAULT_LIMIT = 20

#: Deliberately gentler than retrieval's floor. A stem and a query are both
#: short, so their similarity runs lower than a query against a paragraph, and
#: reusing `RETRIEVAL_MIN_SCORE` here would return nothing for perfectly good
#: matches.
DEFAULT_MIN_SCORE = 0.35


@dataclass(frozen=True)
class BankHit:
    """One banked question a search found, with the score that ranked it."""

    question: BankQuestion
    score: float

    @property
    def score_display(self) -> str:
        return f"{self.score:.3f}"


@dataclass(frozen=True)
class BankSearch:
    """What one search found, and what it could not see.

    `unindexed` is not a footnote: a bank where half the rows were saved during
    a provider outage would otherwise return four hits and look like a bank of
    four questions. The screen says how many rows the ranking could not reach.
    """

    query: str
    hits: tuple[BankHit, ...] = ()
    searched: int = 0
    unindexed: int = 0
    error: str = ""

    @property
    def found(self) -> int:
        return len(self.hits)

    @property
    def questions(self) -> list[BankQuestion]:
        return [hit.question for hit in self.hits]

    @property
    def summary(self) -> str:
        if self.error:
            return f"The search did not run: {self.error}"
        note = (
            f"{self.found} of {self.searched} banked question"
            f"{'s' if self.searched != 1 else ''} matched “{self.query}”."
        )
        if self.unindexed:
            note += (
                f" {self.unindexed} more {'are' if self.unindexed != 1 else 'is'} in the "
                f"bank but not embedded, so ranking could not see "
                f"{'them' if self.unindexed != 1 else 'it'}."
            )
        return note


def search_bank(
    course,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    topic=None,
    question_type: str = "",
    level: str = "",
    provider=None,
) -> BankSearch:
    """The banked questions of `course` that are about `query`, best first.

    Scoped to one course by construction, like retrieval: another course's bank
    is not a weaker match here, it is not a candidate at all.

    A failed embedding call is returned as an error on the result rather than
    raised. The browse screen still has a bank to show — an unsearchable list is
    a degraded screen, not a broken one.
    """
    from courses.services.retrieval import RetrievalError, embed_query

    text = (query or "").strip()
    pool = BankQuestion.objects.for_course(course).matching(
        topic=topic, question_type=question_type, level=level
    )
    searchable = pool.indexed()
    unindexed = pool.count() - searchable.count()

    if not text or limit <= 0:
        return BankSearch(query=text, searched=searchable.count(), unindexed=unindexed)

    try:
        vector = embed_query(text, provider=provider)
    except RetrievalError as exc:
        return BankSearch(
            query=text, searched=searchable.count(), unindexed=unindexed, error=str(exc)
        )

    rows = (
        searchable.select_related("topic")
        .annotate(distance=CosineDistance("embedding", vector))
        .order_by("distance", "pk")[:limit]
    )
    hits = tuple(
        BankHit(question=row, score=1.0 - float(row.distance))
        for row in rows
        if 1.0 - float(row.distance) >= min_score
    )
    return BankSearch(
        query=text, hits=hits, searched=searchable.count(), unindexed=unindexed
    )


def browse(course, *, topic=None, question_type: str = "", level: str = ""):
    """The whole bank of a course, newest first. No call, no ranking.

    What the screen shows before anything is typed — and what it falls back to
    when the search itself could not run.
    """
    return (
        BankQuestion.objects.for_course(course)
        .matching(topic=topic, question_type=question_type, level=level)
        .select_related("topic", "origin_exam")
        .prefetch_related("usages__exam")
    )
