"""Tests for AI shader generation."""

import pytest

from luxc.ai.system_prompt import build_system_prompt
from luxc.ai.generate import extract_code, verify_lux_source
from luxc.builtins.types import clear_type_aliases


class TestSystemPrompt:
    def test_prompt_contains_grammar(self):
        prompt = build_system_prompt()
        assert "scalar" in prompt
        assert "vec3" in prompt
        assert "fn" in prompt

    def test_prompt_contains_constraints(self):
        prompt = build_system_prompt()
        assert "loop" in prompt.lower()
        assert "scalar" in prompt

    def test_prompt_contains_examples(self):
        prompt = build_system_prompt()
        assert "hello_triangle" in prompt or "Examples" in prompt

    def test_prompt_contains_builtins(self):
        prompt = build_system_prompt()
        assert "sin" in prompt
        assert "normalize" in prompt
        assert "dot" in prompt


class TestCodeExtraction:
    def test_extract_from_markdown(self):
        text = """Here's a shader:
```lux
fragment {
    out color: vec4;
    fn main() {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
```
"""
        code = extract_code(text)
        assert "fragment" in code
        assert "```" not in code

    def test_extract_plain(self):
        text = """fragment {
    out color: vec4;
    fn main() {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}"""
        code = extract_code(text)
        assert "fragment" in code
        assert code == text.strip()

    def test_extract_generic_fences(self):
        text = "```\nlet x: scalar = 1.0;\n```"
        code = extract_code(text)
        assert code == "let x: scalar = 1.0;"


class TestVerification:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_verify_valid_code(self):
        code = """
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        success, errors = verify_lux_source(code)
        assert success is True
        assert errors == []

    def test_verify_invalid_code(self):
        code = "this is not valid lux code at all!!!"
        success, errors = verify_lux_source(code)
        assert success is False
        assert len(errors) > 0
