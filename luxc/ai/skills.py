"""AI skill loading infrastructure for Lux shader generation.

Skills are markdown files that provide domain-specific knowledge
(PBR authoring, debugging, optimization, etc.) to augment the
AI system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """A loaded AI skill."""
    name: str       # e.g. "pbr-authoring" (stem of the file)
    title: str      # First H1 heading from markdown
    content: str    # Full markdown content
    path: Path      # Source file path


# Default search directories: project skills/ and user home ~/.luxc/skills/
_PROJECT_SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"
_USER_SKILLS_DIR = Path.home() / ".luxc" / "skills"


def load_skill(path: Path) -> Skill:
    """Load a single skill from a markdown file.

    The skill name is derived from the file stem (e.g., "pbr-authoring.md" -> "pbr-authoring").
    The title is extracted from the first H1 heading (# Title).
    """
    content = path.read_text(encoding="utf-8")
    name = path.stem

    # Extract title from first H1
    title = name  # fallback
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            break

    return Skill(name=name, title=title, content=content, path=path)


def discover_skills(search_dirs: list[Path] | None = None) -> dict[str, Skill]:
    """Discover all available skills from search directories.

    Searches (in order):
    1. <project_root>/skills/
    2. ~/.luxc/skills/

    Custom search_dirs override the defaults.
    Later directories take priority (user skills override built-in).
    """
    if search_dirs is None:
        search_dirs = [_PROJECT_SKILLS_DIR, _USER_SKILLS_DIR]

    skills: dict[str, Skill] = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for md_file in sorted(d.glob("*.md")):
            skill = load_skill(md_file)
            skills[skill.name] = skill

    return skills


def build_skill_context(skill_names: list[str] | None = None) -> str:
    """Build a combined prompt section from selected skills.

    If skill_names is None, returns empty string (no skills loaded).
    If skill_names is provided, loads only those skills.
    Raises ValueError if a requested skill is not found.
    """
    if not skill_names:
        return ""

    all_skills = discover_skills()
    sections: list[str] = []

    for name in skill_names:
        if name not in all_skills:
            available = ", ".join(sorted(all_skills.keys()))
            raise ValueError(
                f"Skill '{name}' not found. Available skills: {available}"
            )
        skill = all_skills[name]
        sections.append(f"## Skill: {skill.title}\n\n{skill.content}")

    return "\n\n---\n\n".join(sections)
