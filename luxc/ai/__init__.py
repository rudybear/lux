"""AI-powered shader generation for Lux."""

from luxc.ai.generate import (
    generate_lux_shader,
    generate_material_from_image,
    generate_material_batch,
    generate_animated_shader_from_video,
    critique_lux_file,
    modify_material,
    GenerateResult,
    CompileError,
    CritiqueIssue,
    CritiqueResult,
    BatchResult,
)
from luxc.ai.system_prompt import (
    build_system_prompt,
    build_material_extraction_prompt,
    build_critique_prompt,
    build_style_transfer_prompt,
    build_batch_planning_prompt,
    build_video_analysis_prompt,
)
from luxc.ai.config import AIConfig, load_config, save_config
from luxc.ai.skills import Skill, discover_skills, load_skill, build_skill_context

__all__ = [
    "generate_lux_shader",
    "generate_material_from_image",
    "generate_material_batch",
    "generate_animated_shader_from_video",
    "critique_lux_file",
    "modify_material",
    "GenerateResult",
    "CompileError",
    "CritiqueIssue",
    "CritiqueResult",
    "BatchResult",
    "build_system_prompt",
    "build_material_extraction_prompt",
    "build_critique_prompt",
    "build_style_transfer_prompt",
    "build_batch_planning_prompt",
    "build_video_analysis_prompt",
    "AIConfig",
    "load_config",
    "save_config",
    "Skill",
    "discover_skills",
    "load_skill",
    "build_skill_context",
]
