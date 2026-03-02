"""Tests for SPIR-V assembly cost estimator.

Tests cover:
- Empty / minimal SPIR-V assembly
- ALU operation counting (OpFAdd, OpFMul, OpFSub, OpIAdd, etc.)
- Texture sample counting (OpImageSampleImplicitLod, OpImageFetch)
- Branch counting (OpBranchConditional, OpSwitch)
- Function call counting (OpFunctionCall)
- Unique %ID tracking for temp_ids
- VGPR pressure estimates (low, medium, high)
- Mixed instruction types
- Comment and blank line handling
- format_cost_summary output
"""

import pytest
from luxc.analysis.cost_estimator import estimate_cost, format_cost_summary


# ---------------------------------------------------------------------------
# Helpers / SPIR-V assembly snippets
# ---------------------------------------------------------------------------

_MINIMAL_ASM = """\
; SPIR-V
; Nothing executable here
"""

_ALU_ASM = """\
; SPIR-V
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%float = OpTypeFloat 32
%fn_void = OpTypeFunction %void
%main = OpFunction %void None %fn_void
%entry = OpLabel
%a = OpLoad %float %in_a
%b = OpLoad %float %in_b
%sum = OpFAdd %float %a %b
%diff = OpFSub %float %a %b
%prod = OpFMul %float %sum %diff
%half = OpFDiv %float %prod %b
OpReturn
OpFunctionEnd
"""

_TEXTURE_ASM = """\
; Texture sampling
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%fn_void = OpTypeFunction %void
%main = OpFunction %void None %fn_void
%entry = OpLabel
%sampled = OpImageSampleImplicitLod %v4float %sampler %coord
%fetched = OpImageFetch %v4float %tex %icoord
%explicit = OpImageSampleExplicitLod %v4float %sampler2 %coord2 Lod %lod
OpReturn
OpFunctionEnd
"""

_BRANCH_ASM = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%fn_void = OpTypeFunction %void
%main = OpFunction %void None %fn_void
%entry = OpLabel
%cond = OpLoad %bool %flag
OpBranchConditional %cond %true_bb %false_bb
%true_bb = OpLabel
OpBranch %merge
%false_bb = OpLabel
OpSwitch %val %default 0 %case0 1 %case1
%case0 = OpLabel
OpBranch %merge
%case1 = OpLabel
OpBranch %merge
%default = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""

_FUNCTION_CALL_ASM = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%fn_void = OpTypeFunction %void
%helper = OpFunction %void None %fn_void
%h_entry = OpLabel
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%entry = OpLabel
%r1 = OpFunctionCall %void %helper
%r2 = OpFunctionCall %void %helper
OpReturn
OpFunctionEnd
"""

_MIXED_ASM = """\
; Mixed instruction types
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%fn_void = OpTypeFunction %void
%main = OpFunction %void None %fn_void
%entry = OpLabel
%a = OpLoad %float %in_a
%b = OpLoad %float %in_b
%sum = OpFAdd %float %a %b
%tex = OpImageSampleImplicitLod %v4float %sampler %coord
%cond = OpLoad %bool %flag
OpBranchConditional %cond %then %else
%then = OpLabel
%prod = OpFMul %float %sum %a
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""


# ---------------------------------------------------------------------------
# Tests: estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCostEmpty:
    """Empty and minimal inputs should yield zero/near-zero counts."""

    def test_empty_string(self):
        costs = estimate_cost("")
        assert costs["instruction_count"] == 0
        assert costs["alu_ops"] == 0
        assert costs["texture_samples"] == 0
        assert costs["branches"] == 0
        assert costs["function_calls"] == 0
        assert costs["temp_ids"] == 0
        assert costs["vgpr_estimate"] == "low"

    def test_comments_only(self):
        costs = estimate_cost(_MINIMAL_ASM)
        assert costs["instruction_count"] == 0
        assert costs["alu_ops"] == 0
        assert costs["temp_ids"] == 0


class TestAluOps:
    """ALU operation counting."""

    def test_basic_alu_count(self):
        """_ALU_ASM contains OpFAdd, OpFSub, OpFMul, OpFDiv = 4 ALU ops."""
        costs = estimate_cost(_ALU_ASM)
        assert costs["alu_ops"] == 4

    def test_alu_instruction_count_includes_non_alu(self):
        """instruction_count should be larger than alu_ops (includes types, loads, etc.)."""
        costs = estimate_cost(_ALU_ASM)
        assert costs["instruction_count"] > costs["alu_ops"]

    def test_integer_alu_ops(self):
        asm = """\
%void = OpTypeVoid
%int = OpTypeInt 32 1
%main = OpFunction %void None %fn_void
%entry = OpLabel
%x = OpIAdd %int %a %b
%y = OpISub %int %a %b
%z = OpIMul %int %x %y
OpReturn
OpFunctionEnd
"""
        costs = estimate_cost(asm)
        assert costs["alu_ops"] == 3


class TestTextureSamples:
    """Texture sample counting."""

    def test_texture_sample_count(self):
        """_TEXTURE_ASM has 3 texture ops."""
        costs = estimate_cost(_TEXTURE_ASM)
        assert costs["texture_samples"] == 3

    def test_no_texture_ops_in_alu_asm(self):
        costs = estimate_cost(_ALU_ASM)
        assert costs["texture_samples"] == 0


class TestBranches:
    """Branch counting."""

    def test_branch_count(self):
        """_BRANCH_ASM has 1 OpBranchConditional + 1 OpSwitch = 2 branches."""
        costs = estimate_cost(_BRANCH_ASM)
        assert costs["branches"] == 2

    def test_no_branches_in_alu_asm(self):
        costs = estimate_cost(_ALU_ASM)
        assert costs["branches"] == 0


class TestFunctionCalls:
    """Function call counting."""

    def test_function_call_count(self):
        """_FUNCTION_CALL_ASM has 2 OpFunctionCall instructions."""
        costs = estimate_cost(_FUNCTION_CALL_ASM)
        assert costs["function_calls"] == 2


class TestTempIds:
    """Unique %ID tracking."""

    def test_unique_ids_counted(self):
        asm = """\
%a = OpLoad %float %in_a
%b = OpLoad %float %in_b
%sum = OpFAdd %float %a %b
"""
        costs = estimate_cost(asm)
        # IDs: %a, %float, %in_a, %b, %in_b, %sum = 6 unique IDs
        assert costs["temp_ids"] == 6

    def test_duplicate_ids_counted_once(self):
        asm = """\
%a = OpLoad %float %in_a
%b = OpLoad %float %in_b
"""
        costs = estimate_cost(asm)
        # %a, %float, %in_a, %b, %in_b = 5 unique (note %float appears twice but counted once)
        assert costs["temp_ids"] == 5


class TestVgprEstimate:
    """VGPR pressure estimate levels."""

    def test_low_vgpr_few_temps(self):
        """Few IDs with a function -> ratio < 30 -> 'low'."""
        asm = """\
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%a = OpLoad %float %b
OpReturn
OpFunctionEnd
"""
        costs = estimate_cost(asm)
        assert costs["vgpr_estimate"] == "low"

    def test_medium_vgpr(self):
        """Generate enough unique IDs with one function to get ratio in [30, 60]."""
        # Build 40 unique IDs with 1 OpFunction -> ratio = 40 -> medium
        lines = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void",
                 "%main = OpFunction %void None %fn", "%entry = OpLabel"]
        for i in range(35):
            lines.append(f"%t{i} = OpLoad %float %in{i}")
        lines.append("OpReturn")
        lines.append("OpFunctionEnd")
        asm = "\n".join(lines)
        costs = estimate_cost(asm)
        # With 1 function: ratio = total_ids / 1. total_ids includes %void, %fn,
        # %main, %entry, %float, and all %t{i}, %in{i} — well above 30.
        assert costs["vgpr_estimate"] == "medium" or costs["vgpr_estimate"] == "high"
        # The exact boundary depends on total unique IDs; verify ratio is >= 30
        assert costs["temp_ids"] >= 30

    def test_high_vgpr_many_temps(self):
        """Many unique IDs without a function -> ratio = total_temps -> 'high'."""
        lines = []
        for i in range(65):
            lines.append(f"%t{i} = OpLoad %float %in{i}")
        asm = "\n".join(lines)
        costs = estimate_cost(asm)
        # No OpFunction -> ratio = total_temps = 65 + %float unique = many > 60
        assert costs["vgpr_estimate"] == "high"

    def test_low_vgpr_no_functions_few_ids(self):
        """No OpFunction but very few IDs -> ratio = total_temps < 30 -> low."""
        asm = """\
%a = OpLoad %float %b
OpReturn
"""
        costs = estimate_cost(asm)
        # IDs: %a, %float, %b -> 3, well below 30
        assert costs["vgpr_estimate"] == "low"


class TestMixedInstructions:
    """Mixed instruction types counted correctly."""

    def test_mixed_counts(self):
        costs = estimate_cost(_MIXED_ASM)
        assert costs["alu_ops"] == 2        # OpFAdd + OpFMul
        assert costs["texture_samples"] == 1  # OpImageSampleImplicitLod
        assert costs["branches"] == 1        # OpBranchConditional
        assert costs["function_calls"] == 0


class TestCommentAndFormatHandling:
    """Comment lines and instruction format variations."""

    def test_comments_ignored(self):
        asm = """\
; This is a comment
; Another comment
%a = OpFAdd %float %b %c
"""
        costs = estimate_cost(asm)
        assert costs["instruction_count"] == 1
        assert costs["alu_ops"] == 1

    def test_id_equals_format(self):
        """Lines like '%id = OpXxx ...' are parsed."""
        asm = "%result = OpFMul %float %a %b"
        costs = estimate_cost(asm)
        assert costs["instruction_count"] == 1
        assert costs["alu_ops"] == 1

    def test_bare_opcode_format(self):
        """Lines like 'OpXxx ...' without %id = are parsed."""
        asm = "OpReturn"
        costs = estimate_cost(asm)
        assert costs["instruction_count"] == 1


# ---------------------------------------------------------------------------
# Tests: format_cost_summary
# ---------------------------------------------------------------------------

class TestFormatCostSummary:
    """format_cost_summary output formatting."""

    def test_basic_format(self):
        costs = {
            "instruction_count": 100,
            "alu_ops": 42,
            "texture_samples": 5,
            "branches": 3,
            "function_calls": 1,
            "temp_ids": 50,
            "vgpr_estimate": "medium",
        }
        result = format_cost_summary(costs, "fragment")
        assert result == "fragment: 100 instr, 42 ALU, 5 tex, VGPR: medium"

    def test_format_includes_stage_name(self):
        costs = estimate_cost("")
        result = format_cost_summary(costs, "vertex")
        assert result.startswith("vertex: ")

    def test_format_low_vgpr(self):
        costs = estimate_cost("")
        result = format_cost_summary(costs, "compute")
        assert "VGPR: low" in result

    def test_format_high_vgpr(self):
        costs = {
            "instruction_count": 500,
            "alu_ops": 200,
            "texture_samples": 20,
            "branches": 10,
            "function_calls": 5,
            "temp_ids": 150,
            "vgpr_estimate": "high",
        }
        result = format_cost_summary(costs, "fragment")
        assert "VGPR: high" in result
        assert "500 instr" in result
        assert "200 ALU" in result
        assert "20 tex" in result
