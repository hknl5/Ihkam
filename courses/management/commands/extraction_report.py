"""Compare what we stored against what each page actually contains.

Written to diagnose a real bug — some uploaded PDFs stored only a page number
where the page was full of content — and kept because it is the only way to
tell "this page is empty" from "we failed to read this page".

    uv run python manage.py extraction_report
    uv run python manage.py extraction_report --file 1 --verbose
    uv run python manage.py extraction_report --reingest
"""

import re

from django.core.management.base import BaseCommand

from courses.models import SourceFile
# One definition of "unmappable", shared with the trigger that acts on it.
from courses.services.ingest import UNMAPPABLE_GLYPHS as UNMAPPABLE
from courses.services.ingest import ingest_source_file

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


class Command(BaseCommand):
    help = "Report extracted-text coverage per page for every uploaded source file."

    def add_arguments(self, parser):
        parser.add_argument("--file", type=int, help="Only this SourceFile id.")
        parser.add_argument(
            "--reingest",
            action="store_true",
            help="Re-run extraction on each file before reporting.",
        )
        parser.add_argument(
            "--verbose-pages",
            action="store_true",
            help="List every page, not only the suspicious ones.",
        )

    def handle(self, *args, **options):
        files = SourceFile.objects.all()
        if options["file"]:
            files = files.filter(pk=options["file"])
        if not files:
            self.stdout.write("No source files uploaded.")
            return

        for source_file in files:
            if options["reingest"]:
                ingest_source_file(source_file)
                source_file.refresh_from_db()
            self._report(source_file, verbose=options["verbose_pages"])

    def _report(self, source_file, *, verbose):
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"[{source_file.pk}] {source_file}"))
        self.stdout.write(
            f"  status={source_file.status}  pages={source_file.page_count}  "
            f"not-read-in-full={source_file.pages_without_text}  "
            f"from-ocr={source_file.pages_from_ocr}"
            f"{' (' + source_file.ocr_engine + ')' if source_file.ocr_engine else ''}  "
            f"unmappable-chars={source_file.unmappable_chars}"
        )
        for row in source_file.ocr_breakdown:
            self.stdout.write(
                f"  why: {row['reason']:<15} read={row['read']:<3} "
                f"still-unread={row['unread']:<3} pages={row['pages']}"
            )
        if source_file.status_detail:
            self.stdout.write(f"  note: {source_file.status_detail}")

        pages = list(source_file.pages.all())
        if not pages:
            self.stdout.write("  (no extracted pages)")
            return

        suspicious = []
        self.stdout.write(
            f"  {'page':>5} {'src':>10} {'chars':>7} {'arabic':>7} {'latin':>6} {'junk':>5}  note"
        )
        for page in pages:
            chars = len(page.text.strip())
            junk = len(UNMAPPABLE.findall(page.text))
            note = ""
            if page.is_image_only:
                note = "image/diagram only — no readable text"
            elif page.is_from_ocr:
                note = f"read by OCR ({page.ocr_reason})"
            elif page.is_partially_unread:
                note = f"partially unread ({page.ocr_reason}) — OCR did not recover it"
                suspicious.append(page.number)
            elif not chars:
                note = "blank page"
            elif chars < 20:
                # The original bug's signature: a page that gave up its number
                # and nothing else, while carrying real content.
                note = "very little text — check the source page"
                suspicious.append(page.number)
            if junk:
                note = f"{note}; {junk} unmappable chars".lstrip("; ")

            if verbose or note:
                self.stdout.write(
                    f"  {page.number:>5} {page.source:>10} {chars:>7} "
                    f"{len(ARABIC.findall(page.text)):>7} "
                    f"{len(LATIN.findall(page.text)):>6} {junk:>5}  {note}"
                )

        total = sum(len(p.text.strip()) for p in pages)
        # "Fully readable" is the honest count: text present, and no part of the
        # page known to be missing from it.
        full = [p for p in pages if p.text.strip() and not p.is_image_only and not p.is_partially_unread]
        partial = [p for p in pages if p.is_partially_unread]
        self.stdout.write(
            f"  total extracted: {total} chars across {len(full)} fully readable "
            f"of {len(pages)} pages ({len(partial)} partially unread, "
            f"{sum(1 for p in pages if p.is_image_only)} unreadable)"
        )
        if suspicious:
            self.stdout.write(
                self.style.WARNING(f"  pages worth eyeballing: {suspicious}")
            )
