"""SPIR-V assembly text cost estimator.

Parses SPIR-V assembly text (Op-prefixed instructions) and produces
cost metrics useful for estimating shader performance characteristics.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Op-code categories
# ---------------------------------------------------------------------------

_ALU_OPS = frozenset({
    "OpFAdd", "OpFSub", "OpFMul", "OpFDiv", "OpFMod", "OpFNegate",
    "OpDot", "OpVectorTimesScalar", "OpVectorTimesMatrix",
    "OpMatrixTimesVector", "OpMatrixTimesMatrix",
    "OpIAdd", "OpISub", "OpIMul", "OpSDiv", "OpUDiv",
})

_TEXTURE_OPS = frozenset({
    "OpImageSampleImplicitLod", "OpImageSampleExplicitLod",
    "OpImageSampleDrefImplicitLod", "OpImageSampleDrefExplicitLod",
    "OpImageFetch",
})

_BRANCH_OPS = frozenset({
    "OpBranchConditional", "OpSwitch",
})

# Pattern to extract the first Op-prefixed token from a line.
# Lines may start with ``%id = OpXxx ...`` or just ``OpXxx ...``.
_OP_RE = re.compile(r"(Op[A-Za-z]+)")

# Pattern for %id references (named or numeric).
_ID_RE = re.compile(r"%[A-Za-z_0-9]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_cost(spv_asm: str) -> dict:
    """Parse SPIR-V assembly text and estimate shader cost.

    Returns:
        dict with keys: instruction_count, alu_ops, texture_samples,
        branches, function_calls, temp_ids, vgpr_estimate
    """

    instruction_count = 0
    alu_ops = 0
    texture_samples = 0
    branches = 0
    function_calls = 0
    function_count = 0
    all_ids: set[str] = set()

    for line in spv_asm.splitlines():
        stripped = line.strip()

        # Skip empty lines and comments.
        if not stripped or stripped.startswith(";"):
            continue

        # Collect every %id on the line for register-pressure estimate.
        for m in _ID_RE.finditer(stripped):
            all_ids.add(m.group(0))

        # Find the Op-prefixed token.
        op_match = _OP_RE.search(stripped)
        if op_match is None:
            continue

        opcode = op_match.group(1)
        instruction_count += 1

        if opcode in _ALU_OPS:
            alu_ops += 1
        elif opcode in _TEXTURE_OPS:
            texture_samples += 1
        elif opcode in _BRANCH_OPS:
            branches += 1
        elif opcode == "OpFunctionCall":
            function_calls += 1
        elif opcode == "OpFunction":
            function_count += 1

    # VGPR pressure estimate ------------------------------------------------
    total_temps = len(all_ids)
    if function_count > 0:
        ratio = total_temps / function_count
    else:
        ratio = float(total_temps)

    if ratio < 30:
        vgpr_estimate = "low"
    elif ratio <= 60:
        vgpr_estimate = "medium"
    else:
        vgpr_estimate = "high"

    return {
        "instruction_count": instruction_count,
        "alu_ops": alu_ops,
        "texture_samples": texture_samples,
        "branches": branches,
        "function_calls": function_calls,
        "temp_ids": total_temps,
        "vgpr_estimate": vgpr_estimate,
    }


def format_cost_summary(costs: dict, stage_name: str) -> str:
    """Format a one-line human-readable cost summary.

    Example output::

        fragment: 312 instr, 89 ALU, 4 tex, VGPR: medium
    """

    return (
        f"{stage_name}: "
        f"{costs['instruction_count']} instr, "
        f"{costs['alu_ops']} ALU, "
        f"{costs['texture_samples']} tex, "
        f"VGPR: {costs['vgpr_estimate']}"
    )
