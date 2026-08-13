"""Build (or rebuild) the embedded passages for a course's files (M2).

Chunking normally runs at the end of ingest. This command exists for the cases
where it did not: files uploaded before M2, an embedding provider that was
unreachable at upload time, or a change to the chunk sizes.

    uv run python manage.py build_chunks --course CS310
    uv run python manage.py build_chunks --all
"""

from django.core.management.base import BaseCommand, CommandError

from courses.models import Course
from courses.services.chunking import chunk_source_file


class Command(BaseCommand):
    help = "Split readable pages into passages and store one embedding each."

    def add_arguments(self, parser):
        parser.add_argument("--course", help="Course code, e.g. CS310.")
        parser.add_argument("--all", action="store_true", help="Every course.")

    def handle(self, *args, **options):
        if options["all"]:
            courses = list(Course.objects.all())
        elif options["course"]:
            courses = list(Course.objects.filter(code__iexact=options["course"]))
            if not courses:
                raise CommandError(f"No course with code {options['course']!r}.")
        else:
            raise CommandError("Pass --course CODE or --all.")

        failures = 0
        for course in courses:
            self.stdout.write(self.style.MIGRATE_HEADING(str(course)))
            for source_file in course.files.all():
                run = chunk_source_file(source_file)
                if run.error:
                    failures += 1
                    self.stdout.write(
                        self.style.ERROR(f"  {source_file.original_name}: {run.error}")
                    )
                    continue
                sources = ", ".join(
                    f"{count} {label}" for label, count in sorted(run.by_source.items())
                )
                self.stdout.write(
                    f"  {source_file.original_name}: {run.chunks} chunk(s) from "
                    f"{run.pages_chunked} readable page(s)"
                    + (f" ({sources})" if sources else "")
                    + (
                        f"; {run.pages_skipped} unreadable page(s) skipped"
                        if run.pages_skipped
                        else ""
                    )
                )

        if failures:
            raise CommandError(f"{failures} file(s) could not be embedded.")
        self.stdout.write(self.style.SUCCESS("Done."))
