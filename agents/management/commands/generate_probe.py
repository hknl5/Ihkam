"""Run Agent 2A on one real blueprint row and print what came back.

The generation counterpart of `ocr_probe`: it is how M5's manual test is run and
re-run — read the candidates, check each one is on-topic and cites a passage
that was actually supplied — without a screen existing yet.

    uv run python manage.py generate_probe --row 12
    uv run python manage.py generate_probe --exam 1 --provider gemini
    uv run python manage.py generate_probe --exam 1 --save

Nothing is stored unless `--save` is passed.
"""

from django.core.management.base import BaseCommand, CommandError

from agents.generate import (
    QuestionGenerationError,
    generate_candidates,
    item_for_row,
    save_candidates,
)
from courses.services.retrieval import RetrievalError, retrieve
from exams.models import BlueprintRow


class Command(BaseCommand):
    help = "Generate candidate questions for one blueprint row and print them."

    def add_arguments(self, parser):
        parser.add_argument("--row", type=int, help="BlueprintRow id.")
        parser.add_argument("--exam", type=int, help="Use the first row of this exam.")
        parser.add_argument("--provider", help="Override LLM_PROVIDER for this run.")
        parser.add_argument("--save", action="store_true", help="Store the candidates.")

    def handle(self, *args, **options):
        row = self._row(options)
        provider = self._provider(options.get("provider"))
        exam = row.blueprint.exam

        try:
            passages = retrieve(exam.course, row.topic, provider=provider)
        except RetrievalError as exc:
            raise CommandError(str(exc)) from exc

        item = item_for_row(row, passages)
        self.stdout.write("")
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"[row {row.pk}] {exam.course.code} · {row.topic.name} · "
                f"{item.type_label} · {item.level_label} — asks {item.count}, "
                f"generates {item.candidates_wanted}"
            )
        )
        for position, passage in enumerate(passages, start=1):
            mark = " (OCR)" if passage.from_ocr else ""
            self.stdout.write(
                f"  P{position}  {passage.page_ref}{mark}  {passage.score_display}  "
                f"{passage.text.strip()[:90]}…"
            )
        if not passages:
            raise CommandError("No passages retrieved — nothing to ground a question in.")

        try:
            run = generate_candidates(item, provider=provider)
        except QuestionGenerationError as exc:
            raise CommandError(str(exc)) from exc

        for position, candidate in enumerate(run.candidates, start=1):
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS(f"  {position}. {candidate.stem}"))
            for option in candidate.options:
                marker = "✓" if option == candidate.correct else " "
                self.stdout.write(f"       {marker} {option}")
            if not candidate.options:
                self.stdout.write(f"       ✓ {candidate.correct}")
            self.stdout.write(f"       why: {candidate.explanation}")
            self.stdout.write(
                f"       source: {candidate.source_ref}"
                + ("  (OCR transcription)" if candidate.from_ocr else "")
            )

        for stem in run.ungrounded:
            self.stdout.write("")
            self.stdout.write(self.style.ERROR(f"  dropped (cited nothing supplied): {stem}"))

        self.stdout.write("")
        self.stdout.write(
            f"  {len(run.candidates)}/{run.returned} candidates grounded "
            f"({run.grounded_rate:.0%}); wanted {run.wanted}."
        )
        if options["save"]:
            stored = save_candidates(run)
            self.stdout.write(self.style.SUCCESS(f"  stored {len(stored)} candidate questions."))

    def _row(self, options):
        if options["row"]:
            row = BlueprintRow.objects.filter(pk=options["row"]).first()
        elif options["exam"]:
            row = BlueprintRow.objects.filter(blueprint__exam=options["exam"]).first()
        else:
            raise CommandError("Pass --row or --exam.")
        if row is None:
            raise CommandError("No matching blueprint row.")
        return row

    def _provider(self, name):
        from agents.provider import LLMError, get_provider

        try:
            return get_provider(name) if name else None
        except LLMError as exc:
            raise CommandError(str(exc)) from exc
