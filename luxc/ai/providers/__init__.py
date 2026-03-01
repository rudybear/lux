"""AI provider registry and factory."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luxc.ai.config import AIConfig
    from luxc.ai.providers.base import AIProvider

# Registry: provider name -> (module_path, class_name, pip_package)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "anthropic": ("luxc.ai.providers.anthropic", "AnthropicProvider", "anthropic"),
    "openai": ("luxc.ai.providers.openai_compat", "OpenAICompatProvider", "openai"),
    "ollama": ("luxc.ai.providers.openai_compat", "OpenAICompatProvider", "openai"),
    "lm-studio": ("luxc.ai.providers.openai_compat", "OpenAICompatProvider", "openai"),
    "gemini": ("luxc.ai.providers.gemini", "GeminiProvider", "google-generativeai"),
}

# Providers that use OpenAI-compatible presets
_PRESET_PROVIDERS = {"openai", "ollama", "lm-studio"}


def get_provider(config: AIConfig) -> AIProvider:
    """Instantiate an AI provider from config.

    Raises ValueError for unknown provider names.
    Raises ImportError with pip install hint for missing SDK packages.
    """
    name = config.provider
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown AI provider '{name}'. Available: {available}"
        )

    module_path, class_name, pip_package = _REGISTRY[name]

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        raise ImportError(
            f"The '{pip_package}' package is required for the '{name}' provider.\n"
            f"Install it with: pip install {pip_package}"
        )

    cls = getattr(mod, class_name)

    if name in _PRESET_PROVIDERS:
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            preset=name,
        )
    return cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
