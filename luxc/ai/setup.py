"""Interactive AI provider setup wizard."""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from luxc.ai.config import AIConfig, save_config
from luxc.ai.providers import get_provider


def _detect_local_server(url: str, timeout: float = 2.0) -> bool:
    """Check if a local server is responding."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _detect_ollama() -> bool:
    return _detect_local_server("http://localhost:11434/api/tags")


def _detect_lm_studio() -> bool:
    return _detect_local_server("http://localhost:1234/v1/models")


def run_setup() -> None:
    """Run the interactive AI provider setup wizard."""
    print("=== Lux AI Provider Setup ===\n")

    # Auto-detect local servers
    ollama_running = _detect_ollama()
    lm_studio_running = _detect_lm_studio()

    # Build provider menu
    providers = [
        ("anthropic", "Anthropic (Claude)"),
        ("openai", "OpenAI (GPT)"),
        ("gemini", "Google Gemini"),
    ]
    if ollama_running:
        providers.append(("ollama", "Ollama (detected running locally)"))
    else:
        providers.append(("ollama", "Ollama"))
    if lm_studio_running:
        providers.append(("lm-studio", "LM Studio (detected running locally)"))
    else:
        providers.append(("lm-studio", "LM Studio"))

    print("Available providers:")
    for i, (_, label) in enumerate(providers, 1):
        print(f"  {i}. {label}")

    # Provider selection
    while True:
        choice = input("\nSelect provider [1]: ").strip()
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                provider_name = providers[idx][0]
                break
        except ValueError:
            pass
        print(f"Please enter a number 1-{len(providers)}.")

    print(f"\nSelected: {provider_name}")

    # Credentials
    api_key = ""
    base_url = ""
    if provider_name in ("anthropic", "openai", "gemini"):
        api_key = input(f"API key for {provider_name} (or Enter to use env var): ").strip()
    elif provider_name == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        custom_url = input(f"Ollama URL [{base_url}]: ").strip()
        if custom_url:
            base_url = custom_url
    elif provider_name == "lm-studio":
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        custom_url = input(f"LM Studio URL [{base_url}]: ").strip()
        if custom_url:
            base_url = custom_url

    # Model selection
    config = AIConfig(
        provider=provider_name,
        model="",
        api_key=api_key,
        base_url=base_url,
    )

    # Default models per provider
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "ollama": "llama3.2",
        "lm-studio": "default",
    }
    default_model = default_models.get(provider_name, "")

    # Try to list available models
    try:
        config.model = default_model  # need a model set to create provider
        provider = get_provider(config)
        models = provider.list_models()
        if models:
            print(f"\nAvailable models (showing up to 20):")
            display = models[:20]
            for i, m in enumerate(display, 1):
                marker = " *" if m == default_model else ""
                print(f"  {i}. {m}{marker}")
            if len(models) > 20:
                print(f"  ... and {len(models) - 20} more")
            model_input = input(f"\nSelect model number or type name [{default_model}]: ").strip()
            if not model_input:
                config.model = default_model
            else:
                try:
                    midx = int(model_input) - 1
                    if 0 <= midx < len(display):
                        config.model = display[midx]
                    else:
                        config.model = model_input
                except ValueError:
                    config.model = model_input
        else:
            config.model = input(f"Model name [{default_model}]: ").strip() or default_model
    except (ImportError, Exception) as e:
        print(f"\nCould not list models: {e}")
        config.model = input(f"Model name [{default_model}]: ").strip() or default_model

    # Verify connection
    print(f"\nTesting connection to {provider_name} with model '{config.model}'...")
    try:
        provider = get_provider(config)
        if provider.test_connection():
            print("Connection successful!")
        else:
            print("Warning: Connection test failed.")
            if input("Save config anyway? [y/N]: ").strip().lower() != "y":
                print("Setup cancelled.")
                return
    except ImportError as e:
        print(f"Warning: {e}")
        if input("Save config anyway? [y/N]: ").strip().lower() != "y":
            print("Setup cancelled.")
            return
    except Exception as e:
        print(f"Warning: Connection test error: {e}")
        if input("Save config anyway? [y/N]: ").strip().lower() != "y":
            print("Setup cancelled.")
            return

    # Save
    save_config(config)
    print(f"\nConfig saved to ~/.luxc/config.toml")
    print(f"  provider: {config.provider}")
    print(f"  model: {config.model}")
    if config.base_url:
        print(f"  base_url: {config.base_url}")
    print("\nDone! You can now use 'luxc --ai \"description\"' with your configured provider.")
