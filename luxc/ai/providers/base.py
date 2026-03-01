"""Abstract base class for AI providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Abstract interface for AI model providers.

    Each provider translates between a provider-neutral message format
    and its own SDK/API conventions.

    Multimodal message parts use a provider-neutral format:
        {"type": "image_base64", "data": "<b64>", "media_type": "image/png"}
        {"type": "text", "text": "Analyze..."}
    """

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def complete(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        """Send a text-only chat completion and return the response text."""

    @abstractmethod
    def complete_multimodal(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        """Send a multimodal (text + image) completion.

        Raises NotImplementedError if the model does not support vision.
        """

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether the current model supports image inputs."""

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify that the provider is reachable and credentials work.

        Returns True on success, False on failure.
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models from this provider.

        Returns an empty list if the provider has no list endpoint.
        """
