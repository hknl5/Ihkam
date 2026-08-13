"""Run OCR on real uploaded pages and print what came back.

The OCR counterpart of `llm_ping`: it proves the provider works against real
material before anything is wired into ingest, and stays useful for judging a
provider swap later.

    uv run python manage.py ocr_probe --file 1 --pages 8,12
    uv run python manage.py ocr_probe --file 2 --pages 2 --provider gemini
    uv run python manage.py ocr_probe --image-only --limit 4
"""

from django.core.management.base import BaseCommand, CommandError

from agents.ocr import OCRError, get_ocr_provider
from courses.models import SourceFile
from courses.services.ingest import render_page_png


class Command(BaseCommand):
    help = "Transcribe uploaded PDF pages with the configured OCR provider."

    def add_arguments(self, parser):
        parser.add_argument("--file", type=int, help="SourceFile id.")
        parser.add_argument("--pages", help="Comma-separated 1-based page numbers.")
        parser.add_argument(
            "--image-only",
            action="store_true",
            help="Pick pages already flagged as having no text layer.",
        )
        parser.add_argument("--limit", type=int, default=4, help="Max pages with --image-only.")
        parser.add_argument("--provider", help="Override OCR_PROVIDER for this run.")
        parser.add_argument("--chars", type=int, default=0, help="Truncate output (0 = full).")

    def handle(self, *args, **options):
        try:
            provider = get_ocr_provider(options["provider"])
        except (OCRError, NotImplementedError) as exc:
            raise CommandError(str(exc)) from exc

        for source_file, numbers in self._targets(options):
            self.stdout.write("")
            self.stdout.write(
                self.style.MIGRATE_HEADING(
                    f"[{source_file.pk}] {source_file.original_name} "
                    f"— {provider.name}/{getattr(provider, 'model', '?')}"
                )
            )
            for number in numbers:
                self._probe(provider, source_file, number, options["chars"])

    def _targets(self, options):
        files = SourceFile.objects.all()
        if options["file"]:
            files = files.filter(pk=options["file"])
        if not files:
            raise CommandError("No matching source files.")

        for source_file in files:
            if options["pages"]:
                numbers = [int(n) for n in options["pages"].split(",") if n.strip()]
            elif options["image_only"]:
                # Every page extraction could not read in full, not only the
                # ones with no text layer at all.
                numbers = list(
                    source_file.pages.exclude(ocr_reason="")
                    .values_list("number", flat=True)[: options["limit"]]
                )
            else:
                numbers = [1]
            if numbers:
                yield source_file, numbers

    def _probe(self, provider, source_file, number, truncate):
        page = source_file.pages.filter(number=number).first()
        stored = (page.text if page else "") or ""
        self.stdout.write("")
        self.stdout.write(
            self.style.HTTP_INFO(
                f"── page {number} "
                f"(stored: {len(stored.strip())} chars"
                f"{', flagged ' + page.ocr_reason if page and page.ocr_reason else ''})"
            )
        )
        try:
            with source_file.file.open("rb") as fh:
                image = render_page_png(fh, number)
            result = provider.ocr_page(
                image, language_hint=source_file.course.content_language
            )
        except Exception as exc:  # noqa: BLE001 — a probe reports, never crashes
            self.stdout.write(self.style.ERROR(f"   failed: {exc}"))
            return

        if not result.is_usable:
            self.stdout.write(self.style.WARNING("   provider reports no readable text"))
            return

        text = result.text
        self.stdout.write(f"   {len(text)} chars transcribed")
        self.stdout.write("")
        self.stdout.write(text[:truncate] if truncate else text)
