"""Tests for AI config loading, saving, and overrides."""

import pytest
from pathlib import Path

from luxc.ai.config import AIConfig, load_config, save_config


class TestDefaultConfig:
    def test_default_values(self):
        config = AIConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_key == ""
        assert config.base_url == ""
        assert config.max_tokens == 4096

    def test_load_missing_file(self, tmp_path):
        """Loading from nonexistent file returns defaults."""
        config = load_config(tmp_path / "nonexistent.toml")
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "config.toml"
        original = AIConfig(
            provider="ollama",
            model="llama3.2",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            max_tokens=2048,
        )
        save_config(original, path)
        loaded = load_config(path)
        assert loaded.provider == original.provider
        assert loaded.model == original.model
        assert loaded.api_key == original.api_key
        assert loaded.base_url == original.base_url
        assert loaded.max_tokens == original.max_tokens

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "config.toml"
        save_config(AIConfig(), path)
        assert path.exists()

    def test_special_chars_in_values(self, tmp_path):
        path = tmp_path / "config.toml"
        original = AIConfig(
            provider="openai",
            model="gpt-4o",
            api_key='sk-abc"def\\ghi',
            base_url="http://example.com/v1",
        )
        save_config(original, path)
        loaded = load_config(path)
        assert loaded.api_key == original.api_key


class TestWithOverrides:
    def test_override_provider(self):
        config = AIConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        overridden = config.with_overrides(provider="openai")
        assert overridden.provider == "openai"
        assert overridden.model == "claude-sonnet-4-20250514"

    def test_override_model(self):
        config = AIConfig()
        overridden = config.with_overrides(model="gpt-4o")
        assert overridden.model == "gpt-4o"
        assert overridden.provider == "anthropic"

    def test_override_none_keeps_original(self):
        config = AIConfig(provider="gemini", model="gemini-2.0-flash")
        overridden = config.with_overrides(provider=None, model=None, base_url=None)
        assert overridden.provider == "gemini"
        assert overridden.model == "gemini-2.0-flash"

    def test_override_base_url(self):
        config = AIConfig()
        overridden = config.with_overrides(base_url="http://localhost:8080/v1")
        assert overridden.base_url == "http://localhost:8080/v1"


class TestMalformedConfig:
    def test_malformed_toml(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("this is not [valid toml !!!", encoding="utf-8")
        config = load_config(path)
        # Should fall back to defaults
        assert config.provider == "anthropic"

    def test_missing_ai_section(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[other]\nkey = "value"\n', encoding="utf-8")
        config = load_config(path)
        assert config.provider == "anthropic"

    def test_partial_config(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[ai]\nprovider = "openai"\n', encoding="utf-8")
        config = load_config(path)
        assert config.provider == "openai"
        assert config.model == "claude-sonnet-4-20250514"  # default for missing
        assert config.max_tokens == 4096
