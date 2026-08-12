"""`uv run python manage.py llm_ping` — the §3 smoke test.

Runs one completion (in JSON mode) and one embedding through whichever
provider `LLM_PROVIDER` selects, and prints what came back. Success means:
valid JSON from `complete()` and a vector of the expected dimension from
`embed()`, on both `openai` and `gemini`.
"""

from __future__ import annotations

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from agents.provider import LLMError, get_provider

SYSTEM = (
    "You are a health check for an exam-drafting system. "
    'Reply with JSON only, exactly this shape: {"ok": true, "provider": "<your model family>"}'
)
USER = "Return the health check object."


class Command(BaseCommand):
    help = "Run one completion and one embedding through the configured LLM provider."

    def add_arguments(self, parser):
        parser.add_argument(
            "--provider",
            default=None,
            help="Override LLM_PROVIDER for this run (openai | gemini | airllm).",
        )
        parser.add_argument(
            "--text",
            default="Precision is the share of predicted positives that are correct.",
            help="Text to embed.",
        )

    def handle(self, *args, **options):
        requested = options["provider"] or settings.LLM_PROVIDER
        try:
            provider = get_provider(options["provider"])
        except LLMError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.MIGRATE_HEADING(f"provider: {provider.name}"))
        if requested != provider.name:  # pragma: no cover - defensive
            self.stdout.write(f"(requested: {requested})")

        # 1. completion
        try:
            response = provider.complete(SYSTEM, USER, json_mode=True, temperature=0.0)
        except Exception as exc:
            raise CommandError(f"complete() failed: {exc}") from exc

        self.stdout.write("\ncomplete():")
        self.stdout.write(f"  raw text : {response.text}")
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise CommandError(f"complete() did not return valid JSON: {exc}") from exc
        self.stdout.write(f"  parsed   : {parsed!r}")
        self.stdout.write(self.style.SUCCESS("  ✓ valid JSON"))

        # 2. embedding
        try:
            vectors = provider.embed([options["text"]])
        except Exception as exc:
            raise CommandError(f"embed() failed: {exc}") from exc

        if not vectors:
            raise CommandError("embed() returned no vectors.")
        vector = vectors[0]
        expected = settings.EMBEDDING_DIM
        self.stdout.write("\nembed():")
        self.stdout.write(f"  dimension: {len(vector)} (expected {expected})")
        self.stdout.write(f"  first 5  : {[round(v, 6) for v in vector[:5]]}")
        if len(vector) != expected:
            raise CommandError(
                f"embed() returned {len(vector)} dimensions, expected {expected}."
            )
        self.stdout.write(self.style.SUCCESS("  ✓ expected dimension"))

        self.stdout.write(
            self.style.SUCCESS(f"\nllm_ping passed for provider '{provider.name}'.")
        )
