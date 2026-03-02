"""Backpropagation tool: learns from experiment results and updates skill files.

Parses experiment_data.json results, extracts actionable findings (significant
instruction count reductions, quality-safe optimizations), formats them as
markdown pattern sections, and appends to skills/gpu-performance.md under the
``## Learned Patterns`` heading.

Deduplicates by checking existing pattern titles before appending.

Usage:
    python tools/skill_updater.py perf_results/experiment_data.json
    python tools/skill_updater.py --skill-file skills/gpu-performance.md results/*.json
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from datetime import datetime


# Minimum instruction count reduction to be considered actionable
MIN_INSTR_DELTA = -5
# Minimum ALU reduction percentage to report
MIN_ALU_REDUCTION_PCT = 5.0

SKILL_FILE_DEFAULT = Path("skills/gpu-performance.md")
LEARNED_HEADING = "## Learned Patterns"


def extract_patterns(experiment: dict) -> list[dict]:
    """Extract actionable patterns from experiment data.

    Returns list of pattern dicts with keys:
        title, description, flags, stages, metrics
    """
    patterns = []
    shader = experiment.get("shader", "unknown")
    comparisons = experiment.get("comparisons", {})
    variants = {v["name"]: v for v in experiment.get("variants", [])}

    for variant_name, stages in comparisons.items():
        variant_info = variants.get(variant_name, {})
        flags = variant_info.get("flags", [])

        for stage, diff in stages.items():
            instr_delta = diff.get("instruction_count", 0)
            alu_delta = diff.get("alu_ops", 0)
            tex_delta = diff.get("texture_samples", 0)

            # Only report reductions that meet the threshold
            if instr_delta >= MIN_INSTR_DELTA and alu_delta >= 0:
                continue

            # Build pattern
            flags_str = " ".join(flags) if flags else "(default)"
            improvements = []
            if instr_delta < 0:
                improvements.append(f"{abs(instr_delta)} fewer instructions")
            if alu_delta < 0:
                improvements.append(f"{abs(alu_delta)} fewer ALU ops")
            if tex_delta < 0:
                improvements.append(f"{abs(tex_delta)} fewer texture samples")

            if not improvements:
                continue

            title = f"{variant_name} on {stage} ({Path(shader).stem})"
            description = (
                f"Flags `{flags_str}` reduced {stage} stage: "
                + ", ".join(improvements) + "."
            )

            patterns.append({
                "title": title,
                "description": description,
                "flags": flags,
                "stages": [stage],
                "metrics": {
                    "instruction_delta": instr_delta,
                    "alu_delta": alu_delta,
                    "tex_delta": tex_delta,
                },
                "shader": shader,
                "timestamp": experiment.get("timestamp", datetime.now().isoformat()),
            })

    return patterns


def format_pattern_md(pattern: dict) -> str:
    """Format a single pattern as a markdown section."""
    lines = [
        f"### {pattern['title']}",
        f"",
        f"{pattern['description']}",
        f"",
        f"- **Shader**: `{pattern['shader']}`",
        f"- **Flags**: `{' '.join(pattern['flags']) if pattern['flags'] else '(default)'}`",
        f"- **Date**: {pattern['timestamp']}",
    ]
    metrics = pattern.get("metrics", {})
    if metrics:
        parts = []
        if metrics.get("instruction_delta"):
            parts.append(f"instr: {metrics['instruction_delta']:+d}")
        if metrics.get("alu_delta"):
            parts.append(f"ALU: {metrics['alu_delta']:+d}")
        if metrics.get("tex_delta"):
            parts.append(f"tex: {metrics['tex_delta']:+d}")
        if parts:
            lines.append(f"- **Deltas**: {', '.join(parts)}")
    lines.append("")
    return "\n".join(lines)


def get_existing_pattern_titles(skill_text: str) -> set[str]:
    """Extract existing ### pattern titles from the skill file."""
    titles = set()
    for match in re.finditer(r"^### (.+)$", skill_text, re.MULTILINE):
        titles.add(match.group(1).strip())
    return titles


def append_patterns(skill_path: Path, patterns: list[dict], dry_run: bool = False) -> int:
    """Append new patterns to the skill file under ## Learned Patterns.

    Returns count of patterns appended.
    """
    if not skill_path.exists():
        print(f"Warning: skill file not found: {skill_path}")
        return 0

    text = skill_path.read_text(encoding="utf-8")
    existing_titles = get_existing_pattern_titles(text)

    new_patterns = [
        p for p in patterns
        if p["title"] not in existing_titles
    ]

    if not new_patterns:
        print("No new patterns to add (all duplicates).")
        return 0

    # Find the insertion point: after the LEARNED_HEADING line
    heading_idx = text.find(LEARNED_HEADING)
    if heading_idx == -1:
        # Append heading at end
        text = text.rstrip() + "\n\n" + LEARNED_HEADING + "\n"
        heading_idx = text.find(LEARNED_HEADING)

    # Find end of the heading line
    heading_end = text.index("\n", heading_idx) + 1

    # Find existing content after the heading (skip blank lines and comments)
    rest = text[heading_end:]

    # Build new content to insert
    new_sections = []
    for p in new_patterns:
        new_sections.append(format_pattern_md(p))

    insert_text = "\n".join(new_sections)

    # Insert after any existing content under Learned Patterns
    # Find the next ## heading or end of file
    next_heading = re.search(r"^## ", rest, re.MULTILINE)
    if next_heading:
        insert_pos = heading_end + next_heading.start()
        updated = text[:insert_pos] + insert_text + "\n" + text[insert_pos:]
    else:
        updated = text.rstrip() + "\n\n" + insert_text

    if dry_run:
        print("--- DRY RUN ---")
        print(insert_text)
        print("--- END DRY RUN ---")
    else:
        skill_path.write_text(updated, encoding="utf-8")

    for p in new_patterns:
        print(f"  Added: {p['title']}")

    return len(new_patterns)


def process_experiment_file(experiment_path: Path, skill_path: Path,
                            dry_run: bool = False) -> int:
    """Process a single experiment_data.json file.

    Returns count of patterns appended.
    """
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    patterns = extract_patterns(experiment)

    if not patterns:
        print(f"No actionable patterns found in {experiment_path}")
        return 0

    print(f"Found {len(patterns)} actionable pattern(s) in {experiment_path}")
    return append_patterns(skill_path, patterns, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Learn from experiment results and update GPU performance skills")
    parser.add_argument("experiment_files", nargs="+", type=Path,
                        help="experiment_data.json files to process")
    parser.add_argument("--skill-file", type=Path, default=SKILL_FILE_DEFAULT,
                        help=f"Path to skill markdown file (default: {SKILL_FILE_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing")

    args = parser.parse_args()

    total_added = 0
    for path in args.experiment_files:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        total_added += process_experiment_file(path, args.skill_file, args.dry_run)

    print(f"\nTotal patterns added: {total_added}")


if __name__ == "__main__":
    main()
