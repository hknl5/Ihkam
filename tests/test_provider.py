"""Provider-abstraction tests. No network, no database."""

from django.test import SimpleTestCase, override_settings

from agents.provider import (
    AirLLMProvider,
    GeminiProvider,
    LLMError,
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    get_provider,
)


class ProviderFactoryTests(SimpleTestCase):
    @override_settings(LLM_PROVIDER="openai", OPENAI_API_KEY="test-key")
    def test_openai_selected_by_settings(self):
        provider = get_provider()
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertIsInstance(provider, LLMProvider)

    @override_settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key")
    def test_gemini_selected_by_settings(self):
        self.assertIsInstance(get_provider(), GeminiProvider)

    @override_settings(LLM_PROVIDER="openai", GEMINI_API_KEY="test-key")
    def test_explicit_name_overrides_settings(self):
        self.assertIsInstance(get_provider("gemini"), GeminiProvider)

    @override_settings(LLM_PROVIDER="llama-in-a-trenchcoat")
    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(LLMError):
            get_provider()

    @override_settings(LLM_PROVIDER="openai", OPENAI_API_KEY="")
    def test_missing_key_is_a_clear_error(self):
        with self.assertRaises(LLMError):
            get_provider("openai")

    def test_airllm_is_still_a_stub(self):
        with self.assertRaises(NotImplementedError):
            AirLLMProvider()


class InterfaceTests(SimpleTestCase):
    def test_every_provider_implements_the_interface(self):
        for cls in (OpenAIProvider, GeminiProvider, AirLLMProvider):
            self.assertTrue(issubclass(cls, LLMProvider))
            self.assertFalse(getattr(cls.complete, "__isabstractmethod__", False))
            self.assertFalse(getattr(cls.embed, "__isabstractmethod__", False))

    def test_response_carries_text_and_raw(self):
        response = LLMResponse(text="{}")
        self.assertEqual(response.text, "{}")
        self.assertEqual(response.raw, {})
