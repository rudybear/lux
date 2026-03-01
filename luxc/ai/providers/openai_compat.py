"""OpenAI-compatible AI provider (OpenAI, Ollama, LM Studio, custom)."""

from __future__ import annotations

from luxc.ai.providers.base import AIProvider


# Known presets for OpenAI-compatible local servers
PRESETS: dict[str, dict[str, str | None]] = {
    "openai": {"base_url": None, "api_key": None},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    "lm-studio": {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"},
}

# Models known to support vision (substring match)
_VISION_MODELS = {
    "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
    "llava", "llava-llama3", "llava-phi3",
    "moondream", "bakllava",
}


class OpenAICompatProvider(AIProvider):
    """Provider for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        preset: str | None = None,
    ) -> None:
        # Apply preset defaults, then let explicit args override
        if preset and preset in PRESETS:
            defaults = PRESETS[preset]
            if not base_url and defaults["base_url"]:
                base_url = defaults["base_url"]
            if not api_key and defaults["api_key"]:
                api_key = defaults["api_key"]

        super().__init__(model, api_key, base_url)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        client = self._get_client()
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(messages)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=api_messages,
        )
        return response.choices[0].message.content

    def complete_multimodal(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model '{self.model}' does not support vision/image inputs. "
                f"Use a vision-capable model (e.g. gpt-4o, llava)."
            )
        client = self._get_client()
        translated = _translate_messages(messages)
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(translated)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=api_messages,
        )
        return response.choices[0].message.content

    @property
    def supports_vision(self) -> bool:
        model_lower = self.model.lower()
        return any(vm in model_lower for vm in _VISION_MODELS)

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            client.chat.completions.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models = client.models.list()
            return sorted(m.id for m in models.data)
        except Exception:
            return []


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Translate provider-neutral multimodal messages to OpenAI format.

    Converts image_base64 parts to data-URL image_url format.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append(msg)
            continue
        translated_parts = []
        for part in content:
            if part.get("type") == "image_base64":
                data_url = f"data:{part['media_type']};base64,{part['data']}"
                translated_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
            elif part.get("type") == "text":
                translated_parts.append({"type": "text", "text": part["text"]})
            else:
                translated_parts.append(part)
        result.append({"role": msg["role"], "content": translated_parts})
    return result
