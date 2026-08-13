"""The OCR seam (agents/ocr.py). No network, no database.

Provider behaviour against real pages is proved by `manage.py ocr_probe`;
these cover the contract every provider must honour.
"""

import io

from django.test import SimpleTestCase, override_settings

from agents.ocr import (
    NO_TEXT_SENTINEL,
    OCRError,
    OCRProvider,
    OCRResult,
    PaddleVLProvider,
    get_ocr_provider,
)
from courses.services.ingest import render_page_png


class OCRResultTests(SimpleTestCase):
    def test_a_transcription_is_usable(self):
        self.assertTrue(OCRResult(text="AND gate | Input x").is_usable)

    def test_an_empty_page_is_not_usable(self):
        # The caller must keep its "unreadable" flag rather than store nothing.
        self.assertFalse(OCRResult(text="", is_empty=True).is_usable)
        self.assertFalse(OCRResult(text="   ").is_usable)

    def test_the_sentinel_is_a_single_agreed_token(self):
        self.assertEqual(NO_TEXT_SENTINEL, "[[NO_TEXT]]")


class OCRProviderSelectionTests(SimpleTestCase):
    @override_settings(OCR_PROVIDER="nonesuch")
    def test_an_unknown_provider_names_the_valid_ones(self):
        with self.assertRaises(OCRError) as caught:
            get_ocr_provider()
        self.assertIn("gemini", str(caught.exception))

    def test_the_local_provider_is_still_a_stub(self):
        with self.assertRaises(NotImplementedError):
            PaddleVLProvider()

    def test_every_provider_implements_the_one_interface(self):
        from agents.ocr import OCR_PROVIDERS

        for provider in OCR_PROVIDERS.values():
            self.assertTrue(issubclass(provider, OCRProvider))
            self.assertIn("ocr_page", dir(provider))


class RenderPageTests(SimpleTestCase):
    """Rasterisation lives with extraction so OCR providers stay PDF-agnostic."""

    def _pdf(self) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.drawString(72, 800, "AND gate")
        pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    def test_a_page_renders_to_a_png_of_the_configured_width(self):
        png = render_page_png(io.BytesIO(self._pdf()), 1, width=800)

        self.assertTrue(png.startswith(b"\x89PNG"))
        from PIL import Image

        self.assertAlmostEqual(Image.open(io.BytesIO(png)).width, 800, delta=2)

    def test_asking_for_a_page_that_is_not_there_says_so(self):
        from courses.services.ingest import ExtractionError

        with self.assertRaises(ExtractionError):
            render_page_png(io.BytesIO(self._pdf()), 9)


class RateLimitTests(SimpleTestCase):
    """A per-minute limit is worth waiting out; a per-day quota is not."""

    def test_a_per_minute_limit_uses_the_servers_own_retry_delay(self):
        from agents.ocr import _rate_limit_delay

        exc = Exception(
            "429 RESOURCE_EXHAUSTED. quotaId: "
            "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
            "'retryDelay': '52s'"
        )
        self.assertAlmostEqual(_rate_limit_delay(exc), 53.0)

    def test_a_per_day_quota_is_not_waited_out(self):
        from agents.ocr import _is_daily_quota, _rate_limit_delay

        exc = Exception(
            "429 RESOURCE_EXHAUSTED. quotaId: "
            "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
            "'retryDelay': '55s'"
        )
        # Waiting 55s for a quota that resets tomorrow just hangs the upload.
        self.assertIsNone(_rate_limit_delay(exc))
        self.assertTrue(_is_daily_quota(exc))

    def test_ordinary_errors_are_not_retried(self):
        from agents.ocr import _rate_limit_delay

        self.assertIsNone(_rate_limit_delay(Exception("400 INVALID_ARGUMENT")))

    def test_a_transient_server_error_is_retried(self):
        # Seen on a real file: one page of thirty came back 503 and was lost,
        # though it read fine moments later.
        from agents.ocr import _rate_limit_delay

        exc = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': "
            "'Deadline expired before operation could complete.'}}"
        )
        self.assertGreater(_rate_limit_delay(exc, attempt=1), 1.0)

    def test_a_transient_error_is_not_mistaken_for_a_spent_quota(self):
        from agents.ocr import _is_daily_quota

        self.assertFalse(_is_daily_quota(Exception("503 UNAVAILABLE")))

    def test_a_rate_limit_without_a_stated_delay_backs_off(self):
        from agents.ocr import MAX_RETRY_DELAY_SECONDS, _rate_limit_delay

        delay = _rate_limit_delay(Exception("429 too many requests"), attempt=3)
        self.assertGreater(delay, 8.0)
        self.assertLessEqual(delay, MAX_RETRY_DELAY_SECONDS)
