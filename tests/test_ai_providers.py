"""Tests for AI provider implementations and registry."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from luxc.ai.config import AIConfig
from luxc.ai.providers import get_provider, _REGISTRY


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_complete(self):
        from luxc.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")

        mock_content = MagicMock()
        mock_content.text = "generated shader code"
        mock_message = MagicMock()
        mock_message.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        provider._client = mock_client

        result = provider.complete("system prompt", [{"role": "user", "content": "hello"}])
        assert result == "generated shader code"
        mock_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="system prompt",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_multimodal_translation(self):
        from luxc.ai.providers.anthropic import AnthropicProvider, _translate_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_base64", "data": "abc123", "media_type": "image/png"},
                    {"type": "text", "text": "Analyze this"},
                ],
            }
        ]
        translated = _translate_messages(messages)
        assert translated[0]["role"] == "user"
        parts = translated[0]["content"]
        assert parts[0]["type"] == "image"
        assert parts[0]["source"]["type"] == "base64"
        assert parts[0]["source"]["data"] == "abc123"
        assert parts[1]["type"] == "text"

    def test_supports_vision(self):
        from luxc.ai.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.supports_vision is True


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class TestOpenAICompatProvider:
    def test_complete(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(model="gpt-4o", api_key="test-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "openai response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.complete("system", [{"role": "user", "content": "hi"}])
        assert result == "openai response"

        # Verify system message was prepended
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "system"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_multimodal_translation(self):
        from luxc.ai.providers.openai_compat import _translate_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_base64", "data": "abc123", "media_type": "image/png"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        translated = _translate_messages(messages)
        parts = translated[0]["content"]
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,abc123"
        assert parts[1]["type"] == "text"

    def test_vision_check_gpt4o(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(model="gpt-4o")
        assert provider.supports_vision is True

    def test_vision_check_gpt35(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(model="gpt-3.5-turbo")
        assert provider.supports_vision is False

    def test_no_vision_raises(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(model="gpt-3.5-turbo")

        with pytest.raises(NotImplementedError, match="does not support vision"):
            provider.complete_multimodal("sys", [{"role": "user", "content": "hi"}])

    def test_ollama_preset_defaults(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(model="llama3.2", preset="ollama")
        assert provider.base_url == "http://localhost:11434/v1"
        assert provider.api_key == "ollama"

    def test_lm_studio_preset_defaults(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(model="default", preset="lm-studio")
        assert provider.base_url == "http://localhost:1234/v1"
        assert provider.api_key == "lm-studio"

    def test_explicit_url_overrides_preset(self):
        from luxc.ai.providers.openai_compat import OpenAICompatProvider
        provider = OpenAICompatProvider(
            model="llama3.2", preset="ollama",
            base_url="http://custom:9999/v1",
        )
        assert provider.base_url == "http://custom:9999/v1"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def test_role_mapping(self):
        from luxc.ai.providers.gemini import _translate_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        translated = _translate_messages(messages)
        assert translated[0]["role"] == "user"
        assert translated[1]["role"] == "model"

    def test_multimodal_translation(self):
        from luxc.ai.providers.gemini import _translate_multimodal_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_base64", "data": "xyz", "media_type": "image/jpeg"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        translated = _translate_multimodal_messages(messages)
        parts = translated[0]["parts"]
        assert parts[0] == {"inline_data": {"mime_type": "image/jpeg", "data": "xyz"}}
        assert parts[1] == "What is this?"

    def test_supports_vision(self):
        from luxc.ai.providers.gemini import GeminiProvider
        provider = GeminiProvider(model="gemini-2.0-flash")
        assert provider.supports_vision is True


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_get_provider_unknown(self):
        config = AIConfig(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_provider(config)

    def test_get_provider_missing_package(self):
        """ImportError should include pip install hint."""
        config = AIConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        with patch("importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="pip install"):
                get_provider(config)

    def test_get_anthropic_provider(self):
        config = AIConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        with patch("luxc.ai.providers.anthropic.AnthropicProvider") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance
            provider = get_provider(config)
            assert provider is mock_instance

    def test_get_ollama_provider(self):
        config = AIConfig(provider="ollama", model="llama3.2")
        with patch("luxc.ai.providers.openai_compat.OpenAICompatProvider") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance
            provider = get_provider(config)
            MockCls.assert_called_once_with(
                model="llama3.2", api_key="", base_url="", preset="ollama",
            )

    def test_all_registry_entries_have_correct_format(self):
        for name, (module_path, class_name, pip_pkg) in _REGISTRY.items():
            assert isinstance(name, str)
            assert module_path.startswith("luxc.ai.providers.")
            assert isinstance(class_name, str)
            assert isinstance(pip_pkg, str)
