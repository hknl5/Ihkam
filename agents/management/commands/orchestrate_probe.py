"""Run the closed correction loop over a real exam, and watch it work — M8's manual test.

    uv run python manage.py orchestrate_probe --exam 5 --provider gemini
    uv run python manage.py orchestrate_probe --row 12 --provider gemini --dry-run

It prints, per item: what each round asked for, every candidate and its verdict,
the note a rejection sent back to Agent 2A, and how the item ended. The point of
reading it is the third block of any item that needed a gap-fill round — the
note, then the alternative written against it.

Everything is stored unless `--dry-run` is passed: the approved pool as
`Question` rows (status `candidate`, as always — إحكام proposes), and the whole
attempt log as `ItemRun` + `QuestionAttempt`.
"""

from django.core.management.base import BaseCommand, CommandError

from agents.orchestrator import (
    MAX_GAP_FILL_ROUNDS,
    OrchestrationError,
    items_for_plan,
    run_item,
    run_plan,
)
from courses.services.retrieval import RetrievalError, retrieve
from exams.models import BlueprintRow, Exam, QuestionAttempt


class Command(BaseCommand):
    help = "Run 1A → 2A → 3A over a blueprint and print every attempt and note."

    def add_arguments(self, parser):
        parser.add_argument("--exam", type=int, help="Run the whole blueprint of this exam.")
        parser.add_argument("--row", type=int, help="Run one blueprint row only.")
        parser.add_argument("--provider", help="Override LLM_PROVIDER for this run.")
        parser.add_argument(
            "--max-rounds",
            type=int,
            default=MAX_GAP_FILL_ROUNDS,
            help=f"Gap-fill rounds before an item needs manual attention (default {MAX_GAP_FILL_ROUNDS}).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Store nothing.")
        parser.add_argument(
            "--python-only", action="store_true", help="Deterministic review only, no review call."
        )

    def handle(self, *args, **options):
        provider = self._provider(options.get("provider"))
        persist = not options["dry_run"]

        try:
            results = self._run(options, provider=provider, persist=persist)
        except (OrchestrationError, RetrievalError) as exc:
            raise CommandError(str(exc)) from exc

        for result in results:
            self._print_item(result)
        self._print_summary(results, persist=persist)

    # --- running ---------------------------------------------------------

    def _run(self, options, *, provider, persist):
        if options["row"]:
            row = (
                BlueprintRow.objects.filter(pk=options["row"])
                .select_related("topic", "blueprint__exam__course")
                .first()
            )
            if row is None:
                raise CommandError("No such blueprint row.")
            exam = row.blueprint.exam
            passages = retrieve(exam.course, row.topic, provider=provider)
            from agents.generate import item_for_row

            return [
                run_item(
                    item_for_row(row, passages),
                    provider=provider,
                    max_rounds=options["max_rounds"],
                    python_only=options["python_only"],
                    persist=persist,
                    exam=exam,
                    row=row,
                )
            ]

        if not options["exam"]:
            raise CommandError("Pass --exam or --row.")
        exam = Exam.objects.filter(pk=options["exam"]).select_related("course").first()
        if exam is None:
            raise CommandError("No such exam.")
        if not exam.has_blueprint:
            raise CommandError(f"{exam.display_title} has no blueprint yet.")

        from agents.analyze import BlueprintNotReady, build_exam_plan

        try:
            plan = build_exam_plan(exam.blueprint, retrieve=self._retriever(provider))
        except BlueprintNotReady as exc:
            raise CommandError(f"The blueprint does not add up: {exc}") from exc

        self.stdout.write("")
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"{exam.display_title} — {len(items_for_plan(plan))} item(s), "
                f"{plan.question_count} planned question(s)"
            )
        )
        for name in plan.topics_without_passages:
            self.stdout.write(self.style.WARNING(f"  no passages retrieved for {name!r}"))

        run = run_plan(
            plan,
            provider=provider,
            max_rounds=options["max_rounds"],
            python_only=options["python_only"],
            persist=persist,
        )
        if run.aborted:
            self.stdout.write(self.style.ERROR(f"\n  run stopped: {run.aborted}"))
        return run.items

    def _retriever(self, provider):
        def _retrieve(course, topic, **kwargs):
            return retrieve(course, topic, provider=provider, **kwargs)

        return _retrieve

    # --- printing --------------------------------------------------------

    def _print_item(self, result):
        item = result.item
        self.stdout.write("")
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"── {item.topic_name} · {item.type_label} · {item.level_label} "
                f"— asks {result.required}"
            )
        )
        if result.error:
            self.stdout.write(self.style.ERROR(f"   {result.error}"))

        for number in range(1, result.rounds + 1):
            attempts = [a for a in result.attempts if a.round == number]
            heading = "first batch" if number == 1 else f"gap-fill round {number - 1}"
            self.stdout.write(f"\n   {heading}: {len(attempts)} candidate(s)")
            if number > 1:
                self.stdout.write("   briefed with:")
                for note in result.notes_from_round(number - 1):
                    self.stdout.write(f"      ← {note}")
            for attempt in attempts:
                self._print_attempt(attempt)

        stored = " (not stored)" if result.record is None else ""
        style = self.style.SUCCESS if result.passed else self.style.ERROR
        label = "PASSED" if result.passed else "NEEDS MANUAL ATTENTION"
        self.stdout.write("")
        self.stdout.write(
            style(
                f"   {label}: {result.approved_count} of {result.required} in "
                f"{result.rounds} round(s), {result.gap_fill_rounds} gap-fill"
                f"{stored}"
            )
        )
        self.stdout.write(
            f"   attempts {len(result.attempts)} = approved {result.approved_count} "
            f"+ rejected {len(result.rejected)} + dropped {len(result.dropped)} "
            f"({'reconciles' if result.counts_reconcile else 'DOES NOT RECONCILE'})"
        )

    def _print_attempt(self, attempt):
        if attempt.outcome == QuestionAttempt.Outcome.PASSED:
            self.stdout.write(self.style.SUCCESS(f"      ✓ {attempt.stem[:100]}"))
        elif attempt.outcome == QuestionAttempt.Outcome.DROPPED:
            self.stdout.write(
                self.style.WARNING(f"      ⊘ dropped before review: {attempt.stem[:80]}")
            )
        else:
            self.stdout.write(self.style.ERROR(f"      ✗ {attempt.stem[:100]}"))
            for note in attempt.notes:
                self.stdout.write(f"          {note}")

    def _print_summary(self, results, *, persist):
        approved = sum(r.approved_count for r in results)
        required = sum(r.required for r in results)
        first_round = [r for r in results if r.passed and r.rounds <= 1]
        gap_filled = [r for r in results if r.passed and r.rounds > 1]
        capped = [r for r in results if r.needs_attention]
        attempts = sum(len(r.attempts) for r in results)
        rejected = sum(len(r.rejected) for r in results)
        dropped = sum(len(r.dropped) for r in results)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("── Run"))
        self.stdout.write(f"   {approved} question(s) passed review, for {required} asked for")
        self.stdout.write(f"   {len(first_round)} item(s) passed on the first batch")
        self.stdout.write(f"   {len(gap_filled)} item(s) needed a gap-fill round")
        for result in capped:
            self.stdout.write(
                self.style.ERROR(
                    f"   needs manual attention: {result.item.topic_name} "
                    f"({result.approved_count}/{result.required})"
                )
            )
        self.stdout.write(
            f"   {attempts} attempt(s) = {approved} approved + {rejected} rejected "
            f"+ {dropped} dropped"
        )
        self.stdout.write(
            f"   problems caught before approval: {rejected + dropped}"
        )
        if persist:
            self.stdout.write("   attempt log stored (ItemRun + QuestionAttempt).")

    def _provider(self, name):
        from agents.provider import LLMError, get_provider

        try:
            return get_provider(name) if name else None
        except LLMError as exc:
            raise CommandError(str(exc)) from exc
