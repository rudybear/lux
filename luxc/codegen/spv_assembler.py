"""Invoke spirv-as and spirv-val to assemble and validate SPIR-V."""

from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


class AssemblyError(Exception):
    pass


class ValidationError(Exception):
    pass


def run_spirv_opt(spv_path: Path) -> bool:
    """Run spirv-opt -O on a .spv file in-place.

    Returns True if optimization succeeded, False if spirv-opt is not available
    or the optimization failed (in which case the original file is preserved).
    """
    spirv_opt = shutil.which("spirv-opt")
    if spirv_opt is None:
        warnings.warn(
            "spirv-opt not found on PATH; skipping SPIR-V optimization. "
            "Install the Vulkan SDK or spirv-tools to enable optimization.",
            stacklevel=2,
        )
        return False

    # Use a temporary output file so the original is preserved on failure
    opt_path = spv_path.with_suffix(".opt.spv")
    try:
        result = subprocess.run(
            [spirv_opt, "-O", str(spv_path), "-o", str(opt_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            warnings.warn(
                f"spirv-opt failed (exit {result.returncode}): {result.stderr.strip()}",
                stacklevel=2,
            )
            opt_path.unlink(missing_ok=True)
            return False

        # Replace the original with the optimized version
        opt_path.replace(spv_path)
        return True
    except Exception as exc:
        warnings.warn(f"spirv-opt error: {exc}", stacklevel=2)
        opt_path.unlink(missing_ok=True)
        return False


def run_spirv_opt_perf(spv_path: Path) -> bool:
    """Run performance-oriented spirv-opt passes on a .spv file in-place.

    Uses a curated set of passes focused on runtime performance rather than
    code size.  Adds loop-unrolling and strength-reduction on top of the
    standard optimization set.

    Returns True if optimization succeeded, False otherwise.
    """
    spirv_opt = shutil.which("spirv-opt")
    if spirv_opt is None:
        warnings.warn(
            "spirv-opt not found on PATH; skipping SPIR-V performance optimization. "
            "Install the Vulkan SDK or spirv-tools to enable optimization.",
            stacklevel=2,
        )
        return False

    opt_path = spv_path.with_suffix(".opt.spv")
    passes = [
        "--merge-blocks",
        "--eliminate-dead-branches",
        "--eliminate-dead-code-aggressive",
        "--private-to-local",
        "--scalar-replacement=100",
        "--ssa-rewrite",
        "--ccp",
        "--simplify-instructions",
        "--redundancy-elimination",
        "--combine-access-chains",
        "--vector-dce",
        "--if-conversion",
        "--loop-unroll",
        "--strength-reduction",
    ]
    try:
        result = subprocess.run(
            [spirv_opt] + passes + [str(spv_path), "-o", str(opt_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            warnings.warn(
                f"spirv-opt (perf) failed (exit {result.returncode}): {result.stderr.strip()}",
                stacklevel=2,
            )
            opt_path.unlink(missing_ok=True)
            return False

        opt_path.replace(spv_path)
        return True
    except Exception as exc:
        warnings.warn(f"spirv-opt (perf) error: {exc}", stacklevel=2)
        opt_path.unlink(missing_ok=True)
        return False


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
