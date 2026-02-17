"""Invoke spirv-as and spirv-val to assemble and validate SPIR-V."""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path


class AssemblyError(Exception):
    pass


class ValidationError(Exception):
    pass


def assemble_and_validate(
    asm_text: str,
    output_path: Path,
    validate: bool = True,
    target_env: str = "vulkan1.2",
) -> None:
    # Write assembly to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".spvasm", delete=False, encoding="utf-8"
    ) as f:
        f.write(asm_text)
        asm_path = Path(f.name)

    try:
        # Assemble
        result = subprocess.run(
            ["spirv-as", "--target-env", target_env, str(asm_path), "-o", str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AssemblyError(
                f"spirv-as failed:\n{result.stderr}\n\nAssembly:\n{_numbered(asm_text)}"
            )

        # Validate
        if validate:
            result = subprocess.run(
                ["spirv-val", "--target-env", target_env, str(output_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise ValidationError(
                    f"spirv-val failed:\n{result.stderr}\n\nAssembly:\n{_numbered(asm_text)}"
                )
    finally:
        asm_path.unlink(missing_ok=True)


def _numbered(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(f"{i+1:4d}: {line}" for i, line in enumerate(lines))
