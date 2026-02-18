"""AI shader generation via Claude API."""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from luxc.ai.system_prompt import build_system_prompt


@dataclass
class GenerateResult:
    lux_source: str
    compilation_success: bool = False
    errors: list[str] = field(default_factory=list)


def extract_code(text: str) -> str:
    """Extract Lux code from a response, stripping markdown fences if present."""
    # Try to find ```lux ... ``` or ``` ... ``` blocks
    patterns = [
        r"```lux\s*\n(.*?)```",
        r"```glsl\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    # No fences found — return as-is (strip leading/trailing whitespace)
    return text.strip()


def verify_lux_source(source: str) -> tuple[bool, list[str]]:
    """Verify that Lux source parses and type-checks.

    Returns (success, error_messages).
    """
    errors = []
    try:
        from luxc.parser.tree_builder import parse_lux
        module = parse_lux(source)

        # Resolve imports if any
        if module.imports:
            from luxc.compiler import _resolve_imports
            _resolve_imports(module)

        # Expand surfaces
        if module.surfaces or module.pipelines:
            from luxc.expansion.surface_expander import expand_surfaces
            expand_surfaces(module)

        # Autodiff
        from luxc.autodiff.forward_diff import autodiff_expand
        autodiff_expand(module)

        # Type check
        from luxc.analysis.type_checker import type_check
        type_check(module)

        return True, []
    except Exception as e:
        errors.append(str(e))
        return False, errors


def generate_lux_shader(
    description: str,
    verify: bool = True,
    model: str = "claude-sonnet-4-20250514",
) -> GenerateResult:
    """Generate a Lux shader from a natural language description.

    Requires the `anthropic` package and ANTHROPIC_API_KEY env var.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for AI generation.\n"
            "Install it with: pip install 'luxc[ai]' or pip install anthropic"
        )

    client = anthropic.Anthropic()
    system_prompt = build_system_prompt()

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {"role": "user", "content": description}
        ],
    )

    raw_text = message.content[0].text
    lux_source = extract_code(raw_text)

    result = GenerateResult(lux_source=lux_source)

    if verify:
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        success, errors = verify_lux_source(lux_source)
        result.compilation_success = success
        result.errors = errors
    else:
        result.compilation_success = True

    return result
