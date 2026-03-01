"""Anthropic (Claude) AI provider."""

from __future__ import annotations

from luxc.ai.providers.base import AIProvider


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider using the anthropic SDK."""

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        super().__init__(model, api_key, base_url)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return message.content[0].text

    def complete_multimodal(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        client = self._get_client()
        translated = _translate_messages(messages)
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=translated,
        )
        return message.content[0].text

    @property
    def supports_vision(self) -> bool:
        return True  # All Claude models support vision

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            client.messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        return []  # Anthropic has no public list endpoint


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Translate provider-neutral multimodal messages to Anthropic format.

    Converts image_base64 parts to Anthropic's image source format.
    Text-only messages pass through unchanged.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append(msg)
            continue
        # content is a list of parts
        translated_parts = []
        for part in content:
            if part.get("type") == "image_base64":
                translated_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part["media_type"],
                        "data": part["data"],
                    },
                })
            elif part.get("type") == "text":
                translated_parts.append({"type": "text", "text": part["text"]})
            else:
                translated_parts.append(part)
        result.append({"role": msg["role"], "content": translated_parts})
    return result
