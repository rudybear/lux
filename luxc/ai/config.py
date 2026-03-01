"""AI provider configuration with TOML persistence."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


_CONFIG_DIR = Path.home() / ".luxc"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


@dataclass
class AIConfig:
    """AI provider settings, loadable from ~/.luxc/config.toml."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096

    def with_overrides(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> AIConfig:
        """Return a copy with selective field replacements (None = keep)."""
        return AIConfig(
            provider=provider if provider is not None else self.provider,
            model=model if model is not None else self.model,
            api_key=self.api_key,
            base_url=base_url if base_url is not None else self.base_url,
            max_tokens=self.max_tokens,
        )


def load_config(path: Path | None = None) -> AIConfig:
    """Read config from TOML file, falling back to defaults on any error."""
    path = path or _CONFIG_FILE
    if not path.exists():
        return AIConfig()
    try:
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
        ai = data.get("ai", {})
        return AIConfig(
            provider=str(ai.get("provider", "anthropic")),
            model=str(ai.get("model", "claude-sonnet-4-20250514")),
            api_key=str(ai.get("api_key", "")),
            base_url=str(ai.get("base_url", "")),
            max_tokens=int(ai.get("max_tokens", 4096)),
        )
    except Exception:
        return AIConfig()


def save_config(config: AIConfig, path: Path | None = None) -> None:
    """Write config to TOML file (manual serialisation, no tomli_w dep)."""
    path = path or _CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    def _quote(v: str) -> str:
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        "[ai]",
        f"provider = {_quote(config.provider)}",
        f"model = {_quote(config.model)}",
        f"api_key = {_quote(config.api_key)}",
        f"base_url = {_quote(config.base_url)}",
        f"max_tokens = {config.max_tokens}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
