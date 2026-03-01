"""Google Gemini AI provider."""

from __future__ import annotations

import base64

from luxc.ai.providers.base import AIProvider


class GeminiProvider(AIProvider):
    """Google Gemini provider using the google-generativeai SDK."""

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        super().__init__(model, api_key, base_url)
        self._genai = None

    def _get_genai(self):
        if self._genai is None:
            import google.generativeai as genai

            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._genai = genai
        return self._genai

    def complete(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        genai = self._get_genai()
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )
        contents = _translate_messages(messages)
        response = model.generate_content(contents)
        return response.text

    def complete_multimodal(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> str:
        genai = self._get_genai()
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )
        contents = _translate_multimodal_messages(messages)
        response = model.generate_content(contents)
        return response.text

    @property
    def supports_vision(self) -> bool:
        return True  # All Gemini models support vision

    def test_connection(self) -> bool:
        try:
            genai = self._get_genai()
            model = genai.GenerativeModel(self.model)
            model.generate_content(
                "Hi",
                generation_config=genai.GenerationConfig(max_output_tokens=16),
            )
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            genai = self._get_genai()
            models = genai.list_models()
            return sorted(
                m.name.removeprefix("models/")
                for m in models
                if "generateContent" in (m.supported_generation_methods or [])
            )
        except Exception:
            return []


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Translate provider-neutral text messages to Gemini format.

    Maps 'assistant' role to 'model' (Gemini convention).
    """
    result = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "parts": [content]})
        else:
            # list of parts — extract text only
            parts = [p["text"] for p in content if p.get("type") == "text"]
            result.append({"role": role, "parts": parts})
    return result


def _translate_multimodal_messages(messages: list[dict]) -> list[dict]:
    """Translate provider-neutral multimodal messages to Gemini format.

    Converts image_base64 parts to Gemini inline_data format.
    """
    result = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "parts": [content]})
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_base64":
                parts.append({
                    "inline_data": {
                        "mime_type": part["media_type"],
                        "data": part["data"],
                    }
                })
            elif part.get("type") == "text":
                parts.append(part["text"])
            else:
                parts.append(str(part))
        result.append({"role": role, "parts": parts})
    return result
