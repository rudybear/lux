"""Tests for AI skill loading infrastructure (Phase 16)."""

import os
import tempfile
from pathlib import Path

import pytest

from luxc.ai.skills import Skill, discover_skills, load_skill, build_skill_context
from luxc.ai.system_prompt import build_system_prompt


# Path to the built-in skills directory
_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# The built-in skills shipped with the project
_BUILTIN_SKILL_NAMES = {"pbr-authoring", "layer-composition", "optimization", "debugging", "gpu-performance", "cpu-guidance", "shader-debugger"}


class TestLoadSkill:
    def test_load_skill(self):
        """Load a skill file, verify name, title, and content are populated."""
        path = _SKILLS_DIR / "debugging.md"
        skill = load_skill(path)

        assert isinstance(skill, Skill)
        assert skill.name == "debugging"
        assert len(skill.title) > 0
        assert len(skill.content) > 0
        assert skill.path == path


class TestDiscoverSkills:
    def test_discover_skills_built_in(self):
        """Discover skills from the project skills/ directory; all 4 exist."""
        skills = discover_skills(search_dirs=[_SKILLS_DIR])

        assert isinstance(skills, dict)
        assert set(skills.keys()) == _BUILTIN_SKILL_NAMES

        for name, skill in skills.items():
            assert isinstance(skill, Skill)
            assert skill.name == name

    def test_discover_skills_custom_dir(self):
        """Create a temp dir with a custom skill, discover from it."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_md = Path(tmp) / "my-custom-skill.md"
            custom_md.write_text("# My Custom Skill\n\nSome useful content.", encoding="utf-8")

            skills = discover_skills(search_dirs=[Path(tmp)])

        assert "my-custom-skill" in skills
        assert skills["my-custom-skill"].title == "My Custom Skill"
        assert "Some useful content" in skills["my-custom-skill"].content


class TestBuildSkillContext:
    def test_build_skill_context(self):
        """Build context with specific skills, verify output contains content."""
        ctx = build_skill_context(["debugging"])

        assert isinstance(ctx, str)
        assert len(ctx) > 0
        assert "Skill:" in ctx
        # The debugging skill title should appear
        assert "Diagnosing" in ctx or "debugging" in ctx.lower()

    def test_build_skill_context_empty(self):
        """Empty or None returns empty string."""
        assert build_skill_context(None) == ""
        assert build_skill_context([]) == ""

    def test_build_skill_context_invalid(self):
        """Requesting a non-existent skill raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_skill_context(["totally-nonexistent-skill-xyz"])


class TestSystemPromptWithSkills:
    def test_system_prompt_with_skills(self):
        """Call build_system_prompt(skills=["debugging"]) and verify skill content appears."""
        prompt = build_system_prompt(skills=["debugging"])

        # The debugging skill content should be injected
        assert "Diagnosing" in prompt or "NaN" in prompt or "artifact" in prompt.lower()

    def test_system_prompt_without_skills(self):
        """Default build_system_prompt() does NOT contain skill-specific content."""
        prompt = build_system_prompt()

        # The prompt should NOT contain skill section headers
        assert "## Skill:" not in prompt


class TestBuiltinSkillsExist:
    def test_builtin_skills_exist(self):
        """All 4 skill files exist on disk."""
        for name in _BUILTIN_SKILL_NAMES:
            path = _SKILLS_DIR / f"{name}.md"
            assert path.exists(), f"Built-in skill file missing: {path}"

    def test_builtin_skills_load(self):
        """All 4 skills load without error; each has title and content."""
        for name in _BUILTIN_SKILL_NAMES:
            path = _SKILLS_DIR / f"{name}.md"
            skill = load_skill(path)

            assert skill.name == name
            assert len(skill.title) > 0, f"Skill '{name}' has empty title"
            assert len(skill.content) > 0, f"Skill '{name}' has empty content"
