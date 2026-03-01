"""AI-powered shader generation for Lux."""

from luxc.ai.generate import (
    generate_lux_shader,
    generate_material_from_image,
    GenerateResult,
    CompileError,
)
from luxc.ai.system_prompt import build_system_prompt, build_material_extraction_prompt
from luxc.ai.config import AIConfig, load_config, save_config

__all__ = [
    "generate_lux_shader",
    "generate_material_from_image",
    "GenerateResult",
    "CompileError",
    "build_system_prompt",
    "build_material_extraction_prompt",
    "AIConfig",
    "load_config",
    "save_config",
]
