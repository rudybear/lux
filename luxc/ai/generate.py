"""AI shader generation with multi-provider support."""

from __future__ import annotations
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from luxc.ai.system_prompt import build_system_prompt


@dataclass
class CompileError:
    """Structured compilation error with phase and location info."""
    phase: str        # "parse", "resolve", "expand", "autodiff", "type_check"
    message: str
    line: int | None = None
    column: int | None = None


@dataclass
class GenerateResult:
    lux_source: str
    compilation_success: bool = False
    errors: list[CompileError] = field(default_factory=list)
    attempts: int = 1


@dataclass
class CritiqueIssue:
    """A single issue found during AI critique."""
    severity: str      # "error", "warning", "info"
    category: str      # "physics", "energy", "performance", "style", "correctness"
    message: str
    line: int | None = None
    suggestion: str = ""


@dataclass
class CritiqueResult:
    """Result of an AI critique / validation pass."""
    issues: list[CritiqueIssue] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


@dataclass
class BatchResult:
    """Result of batch material generation."""
    materials: list[GenerateResult] = field(default_factory=list)
    material_names: list[str] = field(default_factory=list)
    scene_description: str = ""


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


def extract_json(text: str) -> str:
    """Extract JSON from a response, stripping markdown fences if present."""
    # Try ```json ... ``` blocks first
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ``` blocks
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON (array or object)
    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text.strip()


def verify_lux_source_structured(source: str) -> tuple[bool, list[CompileError]]:
    """Verify that Lux source parses and type-checks.

    Returns (success, errors) where errors are structured CompileError objects
    with phase, message, and optional line/column information.
    """
    # Parse
    try:
        from luxc.parser.tree_builder import parse_lux
        module = parse_lux(source)
    except Exception as e:
        line = getattr(e, "line", None)
        column = getattr(e, "column", None)
        return False, [CompileError(phase="parse", message=str(e), line=line, column=column)]

    # Resolve imports
    if module.imports:
        try:
            from luxc.compiler import _resolve_imports
            _resolve_imports(module)
        except Exception as e:
            return False, [CompileError(phase="resolve", message=str(e))]

    # Expand surfaces
    if module.surfaces or module.pipelines:
        try:
            from luxc.expansion.surface_expander import expand_surfaces
            expand_surfaces(module)
        except Exception as e:
            return False, [CompileError(phase="expand", message=str(e))]

    # Autodiff
    try:
        from luxc.autodiff.forward_diff import autodiff_expand
        autodiff_expand(module)
    except Exception as e:
        return False, [CompileError(phase="autodiff", message=str(e))]

    # Type check
    try:
        from luxc.analysis.type_checker import type_check
        type_check(module)
    except Exception as e:
        return False, [CompileError(phase="type_check", message=str(e))]

    return True, []


def verify_lux_source(source: str) -> tuple[bool, list[str]]:
    """Verify that Lux source parses and type-checks.

    Returns (success, error_messages).

    This is a backward-compatible wrapper around verify_lux_source_structured().
    """
    success, errors = verify_lux_source_structured(source)
    return success, [e.message for e in errors]


def _build_error_feedback(errors: list[CompileError]) -> str:
    """Build a human-readable error feedback message for the retry loop."""
    lines = ["The generated Lux code failed to compile. Please fix the following errors:\n"]
    for err in errors:
        loc = ""
        if err.line is not None:
            loc = f" (line {err.line}"
            if err.column is not None:
                loc += f", column {err.column}"
            loc += ")"
        lines.append(f"- [{err.phase}]{loc}: {err.message}")
    lines.append("\nPlease output the corrected, complete Lux program.")
    return "\n".join(lines)


def _resolve_provider(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
):
    """Resolve an AI provider from config + CLI overrides.

    Returns (provider_instance, config).
    """
    from luxc.ai.config import load_config
    from luxc.ai.providers import get_provider

    config = load_config().with_overrides(
        provider=provider, model=model, base_url=base_url
    )
    return get_provider(config), config


def generate_lux_shader(
    description: str,
    verify: bool = True,
    model: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    base_url: str | None = None,
) -> GenerateResult:
    """Generate a Lux shader from a natural language description.

    Uses the configured AI provider (default: Anthropic Claude).
    Provider, model, and base_url can be overridden via arguments,
    otherwise reads from ~/.luxc/config.toml.
    """
    ai_provider, config = _resolve_provider(provider, model, base_url)
    system_prompt = build_system_prompt()

    messages: list[dict] = [{"role": "user", "content": description}]

    raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
    lux_source = extract_code(raw_text)
    attempts = 1

    if verify:
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        success, errors = verify_lux_source_structured(lux_source)

        retries_left = max_retries
        while not success and retries_left > 0:
            # Append assistant response and user error feedback to conversation
            messages.append({"role": "assistant", "content": raw_text})
            feedback = _build_error_feedback(errors)
            messages.append({"role": "user", "content": feedback})

            raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
            lux_source = extract_code(raw_text)
            attempts += 1
            retries_left -= 1

            clear_type_aliases()
            success, errors = verify_lux_source_structured(lux_source)

        result = GenerateResult(
            lux_source=lux_source,
            compilation_success=success,
            errors=errors,
            attempts=attempts,
        )
    else:
        result = GenerateResult(
            lux_source=lux_source,
            compilation_success=True,
            attempts=attempts,
        )

    return result


def _encode_image(path: str | Path) -> tuple[str, str]:
    """Read and base64-encode an image file.

    Returns (base64_data, media_type).

    Raises FileNotFoundError if the path doesn't exist.
    Raises ValueError for unsupported image extensions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    extension_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    ext = path.suffix.lower()
    if ext not in extension_map:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            f"Supported: {', '.join(sorted(extension_map.keys()))}"
        )

    media_type = extension_map[ext]
    raw_bytes = path.read_bytes()
    base64_data = base64.standard_b64encode(raw_bytes).decode("ascii")

    return base64_data, media_type


def generate_material_from_image(
    image_path: str | Path,
    description: str = "",
    verify: bool = True,
    model: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    base_url: str | None = None,
) -> GenerateResult:
    """Generate a Lux surface material from a reference image.

    Uses the configured AI provider's multimodal capabilities.
    Raises NotImplementedError if the provider/model does not support vision.
    """
    ai_provider, config = _resolve_provider(provider, model, base_url)

    if not ai_provider.supports_vision:
        raise NotImplementedError(
            f"The '{config.provider}' provider with model '{config.model}' "
            f"does not support image inputs. Use a vision-capable model."
        )

    base64_data, media_type = _encode_image(image_path)

    from luxc.ai.system_prompt import build_material_extraction_prompt
    system_prompt = build_material_extraction_prompt()

    text_instruction = "Analyze this image and generate a Lux surface material that matches it."
    if description:
        text_instruction += f" {description}"

    # Provider-neutral multimodal message format
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_base64",
                    "data": base64_data,
                    "media_type": media_type,
                },
                {
                    "type": "text",
                    "text": text_instruction,
                },
            ],
        }
    ]

    raw_text = ai_provider.complete_multimodal(system_prompt, messages, config.max_tokens)
    lux_source = extract_code(raw_text)
    attempts = 1

    if verify:
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        success, errors = verify_lux_source_structured(lux_source)

        retries_left = max_retries
        while not success and retries_left > 0:
            # Append assistant response and user error feedback to conversation
            messages.append({"role": "assistant", "content": raw_text})
            feedback = _build_error_feedback(errors)
            messages.append({"role": "user", "content": feedback})

            raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
            lux_source = extract_code(raw_text)
            attempts += 1
            retries_left -= 1

            clear_type_aliases()
            success, errors = verify_lux_source_structured(lux_source)

        result = GenerateResult(
            lux_source=lux_source,
            compilation_success=success,
            errors=errors,
            attempts=attempts,
        )
    else:
        result = GenerateResult(
            lux_source=lux_source,
            compilation_success=True,
            attempts=attempts,
        )

    return result


# ---------------------------------------------------------------------------
# P16.3c — Critique / Validation
# ---------------------------------------------------------------------------

def critique_lux_file(
    source: str,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> CritiqueResult:
    """Run AI critique/validation on a Lux source file.

    Returns structured issues with severity, category, and suggestions.
    """
    from luxc.ai.system_prompt import build_critique_prompt

    ai_provider, config = _resolve_provider(provider, model, base_url)
    system_prompt = build_critique_prompt()

    messages: list[dict] = [
        {"role": "user", "content": f"Review this Lux shader:\n\n```lux\n{source}\n```"}
    ]

    raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)

    # Parse JSON response
    issues: list[CritiqueIssue] = []
    summary = ""

    try:
        json_str = extract_json(raw_text)
        data = json.loads(json_str)
        summary = data.get("summary", "")
        for item in data.get("issues", []):
            issues.append(CritiqueIssue(
                severity=item.get("severity", "info"),
                category=item.get("category", "correctness"),
                message=item.get("message", ""),
                line=item.get("line"),
                suggestion=item.get("suggestion", ""),
            ))
    except (json.JSONDecodeError, KeyError, TypeError):
        # If JSON parsing fails, return raw text as summary
        summary = raw_text

    return CritiqueResult(issues=issues, summary=summary, raw_response=raw_text)


# ---------------------------------------------------------------------------
# P16.2a — Style Transfer / Material Modification
# ---------------------------------------------------------------------------

def modify_material(
    source: str,
    instruction: str,
    verify: bool = True,
    model: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    base_url: str | None = None,
) -> GenerateResult:
    """Modify an existing Lux material based on a style instruction.

    Examples: "make it more weathered", "add clear coat layer",
    "change to dark copper".
    """
    from luxc.ai.system_prompt import build_style_transfer_prompt

    ai_provider, config = _resolve_provider(provider, model, base_url)
    system_prompt = build_style_transfer_prompt()

    prompt = (
        f"Here is the existing Lux shader:\n\n```lux\n{source}\n```\n\n"
        f"Modification instruction: {instruction}\n\n"
        f"Output the complete modified Lux program."
    )

    messages: list[dict] = [{"role": "user", "content": prompt}]

    raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
    lux_source = extract_code(raw_text)
    attempts = 1

    if verify:
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        success, errors = verify_lux_source_structured(lux_source)

        retries_left = max_retries
        while not success and retries_left > 0:
            messages.append({"role": "assistant", "content": raw_text})
            feedback = _build_error_feedback(errors)
            messages.append({"role": "user", "content": feedback})

            raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
            lux_source = extract_code(raw_text)
            attempts += 1
            retries_left -= 1

            clear_type_aliases()
            success, errors = verify_lux_source_structured(lux_source)

        return GenerateResult(
            lux_source=lux_source,
            compilation_success=success,
            errors=errors,
            attempts=attempts,
        )
    else:
        return GenerateResult(
            lux_source=lux_source,
            compilation_success=True,
            attempts=attempts,
        )


# ---------------------------------------------------------------------------
# P16.2b — Scene-Aware Batch Generation
# ---------------------------------------------------------------------------

def generate_material_batch(
    scene_description: str,
    count: int | None = None,
    verify: bool = True,
    model: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    base_url: str | None = None,
) -> BatchResult:
    """Generate a batch of complementary materials for a scene.

    Two-phase approach:
    1. AI plans material names and descriptions (JSON list)
    2. Each material is generated individually with verification
    """
    from luxc.ai.system_prompt import build_batch_planning_prompt

    ai_provider, config = _resolve_provider(provider, model, base_url)

    # Phase 1: Plan materials
    planning_prompt = build_batch_planning_prompt()
    plan_instruction = f"Scene description: {scene_description}"
    if count is not None:
        plan_instruction += f"\nGenerate exactly {count} materials."

    messages: list[dict] = [{"role": "user", "content": plan_instruction}]
    raw_plan = ai_provider.complete(planning_prompt, messages, config.max_tokens)

    # Parse the material plan
    try:
        json_str = extract_json(raw_plan)
        plans = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        plans = [{"name": "material", "description": scene_description}]

    if not isinstance(plans, list):
        plans = [plans]

    # Phase 2: Generate each material
    materials: list[GenerateResult] = []
    material_names: list[str] = []

    for plan in plans:
        name = plan.get("name", f"material_{len(materials)}")
        desc = plan.get("description", name)

        result = generate_lux_shader(
            f"Create a PBR material for: {desc}",
            verify=verify,
            model=model,
            max_retries=max_retries,
            provider=provider,
            base_url=base_url,
        )

        materials.append(result)
        material_names.append(name)

    return BatchResult(
        materials=materials,
        material_names=material_names,
        scene_description=scene_description,
    )


# ---------------------------------------------------------------------------
# P16.1a — Video-to-Animation
# ---------------------------------------------------------------------------

def generate_animated_shader_from_video(
    video_path: Path,
    description: str = "",
    verify: bool = True,
    model: str | None = None,
    max_retries: int = 2,
    max_frames: int = 6,
    provider: str | None = None,
    base_url: str | None = None,
) -> GenerateResult:
    """Generate an animated Lux shader from a video file.

    Pipeline:
    1. Extract key frames from video (requires opencv-python)
    2. Vision AI describes the motion pattern
    3. Generate a shader with time-based noise/animation
    """
    from luxc.ai.video import extract_key_frames, describe_motion
    from luxc.ai.system_prompt import build_video_analysis_prompt

    ai_provider, config = _resolve_provider(provider, model, base_url)

    # Step 1: Extract frames
    key_frames = extract_key_frames(video_path, max_frames=max_frames)

    # Step 2: Describe motion
    motion_description = describe_motion(key_frames, ai_provider, config)

    # Step 3: Generate animated shader
    system_prompt = build_video_analysis_prompt()
    prompt = f"Motion analysis from video:\n{motion_description}"
    if description:
        prompt += f"\n\nAdditional context: {description}"

    messages: list[dict] = [{"role": "user", "content": prompt}]

    raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
    lux_source = extract_code(raw_text)
    attempts = 1

    if verify:
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        success, errors = verify_lux_source_structured(lux_source)

        retries_left = max_retries
        while not success and retries_left > 0:
            messages.append({"role": "assistant", "content": raw_text})
            feedback = _build_error_feedback(errors)
            messages.append({"role": "user", "content": feedback})

            raw_text = ai_provider.complete(system_prompt, messages, config.max_tokens)
            lux_source = extract_code(raw_text)
            attempts += 1
            retries_left -= 1

            clear_type_aliases()
            success, errors = verify_lux_source_structured(lux_source)

        return GenerateResult(
            lux_source=lux_source,
            compilation_success=success,
            errors=errors,
            attempts=attempts,
        )
    else:
        return GenerateResult(
            lux_source=lux_source,
            compilation_success=True,
            attempts=attempts,
        )
