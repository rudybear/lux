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

    def test_prompt_contains_surface_syntax(self):
        prompt = build_system_prompt()
        assert "surface" in prompt
        assert "layers" in prompt

    def test_prompt_contains_geometry(self):
        prompt = build_system_prompt()
        assert "geometry" in prompt
        assert "outputs" in prompt

    def test_prompt_contains_pipeline(self):
        prompt = build_system_prompt()
        assert "pipeline" in prompt

    def test_prompt_contains_new_stdlib(self):
        prompt = build_system_prompt()
        assert "compositing" in prompt
        assert "ibl" in prompt
        assert "shadow" in prompt
        assert "texture" in prompt
        assert "toon" in prompt
        assert "colorspace" in prompt

    def test_prompt_contains_features(self):
        prompt = build_system_prompt()
        assert "features" in prompt

    def test_prompt_contains_properties(self):
        prompt = build_system_prompt()
        assert "properties" in prompt

    def test_prompt_mode_parameter(self):
        # mode parameter should exist and work
        prompt = build_system_prompt(mode="general")
        assert len(prompt) > 0


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


class TestStructuredErrors:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_error_parse_phase(self):
        from luxc.ai.generate import verify_lux_source_structured
        code = "this is not valid lux code!!!"
        success, errors = verify_lux_source_structured(code)
        assert success is False
        assert len(errors) > 0
        assert errors[0].phase == "parse"

    def test_error_type_check_phase(self):
        from luxc.ai.generate import verify_lux_source_structured
        # Valid syntax but type error: assigning scalar to vec4
        code = """
        fragment {
            out color: vec4;
            fn main() {
                color = 1.0;
            }
        }
        """
        success, errors = verify_lux_source_structured(code)
        assert success is False
        assert len(errors) > 0
        assert errors[0].phase in ("type_check", "expand")

    def test_error_has_line_info(self):
        from luxc.ai.generate import verify_lux_source_structured
        code = "this is not valid lux code!!!"
        success, errors = verify_lux_source_structured(code)
        assert success is False
        # Parse errors from lark should have line info
        assert errors[0].line is not None or errors[0].message != ""


class TestRetryLoop:
    """Test retry loop with mocked AI provider."""

    def _make_mock_provider(self, responses):
        """Create a mock AI provider that returns responses in sequence."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.complete.side_effect = list(responses)
        mock.supports_vision = True
        mock.complete_multimodal.side_effect = list(responses)
        return mock

    def test_retry_fixes_code(self):
        """First response bad, second good -> success after 2 attempts."""
        from unittest.mock import MagicMock, patch
        from luxc.ai.generate import generate_lux_shader
        from luxc.ai.config import AIConfig

        bad_code = "```lux\nthis is bad code!!!\n```"
        good_code = """```lux
fragment {
    out color: vec4;
    fn main() {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
```"""
        mock_provider = self._make_mock_provider([bad_code, good_code])
        mock_config = AIConfig()

        with patch("luxc.ai.generate._resolve_provider", return_value=(mock_provider, mock_config)):
            clear_type_aliases()
            result = generate_lux_shader("red shader", max_retries=2)
            clear_type_aliases()

        assert result.compilation_success is True
        assert result.attempts == 2

    def test_no_retry_when_success(self):
        """Good code on first try -> attempts=1."""
        from unittest.mock import MagicMock, patch
        from luxc.ai.generate import generate_lux_shader
        from luxc.ai.config import AIConfig

        good_code = """```lux
fragment {
    out color: vec4;
    fn main() {
        color = vec4(0.0, 1.0, 0.0, 1.0);
    }
}
```"""
        mock_provider = self._make_mock_provider([good_code])
        mock_config = AIConfig()

        with patch("luxc.ai.generate._resolve_provider", return_value=(mock_provider, mock_config)):
            clear_type_aliases()
            result = generate_lux_shader("green shader", max_retries=2)
            clear_type_aliases()

        assert result.compilation_success is True
        assert result.attempts == 1
        assert mock_provider.complete.call_count == 1

    def test_max_retries_zero(self):
        """max_retries=0 disables retry even on failure."""
        from unittest.mock import MagicMock, patch
        from luxc.ai.generate import generate_lux_shader
        from luxc.ai.config import AIConfig

        bad_code = "```lux\nthis is bad code!!!\n```"
        mock_provider = self._make_mock_provider([bad_code])
        mock_config = AIConfig()

        with patch("luxc.ai.generate._resolve_provider", return_value=(mock_provider, mock_config)):
            clear_type_aliases()
            result = generate_lux_shader("bad shader", max_retries=0)
            clear_type_aliases()

        assert result.compilation_success is False
        assert result.attempts == 1
        assert mock_provider.complete.call_count == 1


class TestImageToMaterial:
    """Tests for image-to-material generation."""

    def test_encode_image_png(self):
        """Tiny 1x1 PNG, verify base64 encoding."""
        import base64
        import tempfile
        import os
        from luxc.ai.generate import _encode_image

        # Minimal valid 1x1 white PNG (67 bytes)
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mP8/58BAwAI/AL+hc2rNAAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            tmp_path = f.name

        try:
            data, media_type = _encode_image(tmp_path)
            assert media_type == "image/png"
            assert len(data) > 0
            # Verify it's valid base64
            decoded = base64.b64decode(data)
            assert decoded == png_bytes
        finally:
            os.unlink(tmp_path)

    def test_encode_image_invalid_path(self):
        """Non-existent file raises FileNotFoundError."""
        from luxc.ai.generate import _encode_image

        with pytest.raises(FileNotFoundError):
            _encode_image("/nonexistent/path/image.png")

    def test_material_prompt_contains_pbr(self):
        """Material extraction prompt mentions PBR terms."""
        from luxc.ai.system_prompt import build_material_extraction_prompt

        prompt = build_material_extraction_prompt()
        assert "roughness" in prompt.lower()
        assert "metallic" in prompt.lower()
        assert "albedo" in prompt.lower() or "base color" in prompt.lower() or "base_color" in prompt.lower()
        assert "surface" in prompt

    def test_generate_material_mock(self):
        """Mock provider, feed surface declaration response, verify compilation."""
        from unittest.mock import MagicMock, patch
        import base64
        import tempfile
        import os
        from luxc.ai.generate import generate_material_from_image
        from luxc.ai.config import AIConfig

        surface_code = """```lux
import brdf;

geometry StandardMesh {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP { model: mat4, view: mat4, projection: mat4 }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface PhotoMaterial {
    sampler2d albedo_tex,
    brdf: pbr(sample(albedo_tex, frag_uv).xyz, 0.5, 0.0),
}

pipeline MaterialForward {
    geometry: StandardMesh,
    surface: PhotoMaterial,
}
```"""

        # Create a tiny PNG for the test
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mP8/58BAwAI/AL+hc2rNAAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            tmp_path = f.name

        mock_provider = MagicMock()
        mock_provider.supports_vision = True
        mock_provider.complete_multimodal.return_value = surface_code
        mock_provider.complete.return_value = surface_code
        mock_config = AIConfig()

        try:
            with patch("luxc.ai.generate._resolve_provider", return_value=(mock_provider, mock_config)):
                clear_type_aliases()
                result = generate_material_from_image(tmp_path)
                clear_type_aliases()

            assert result.compilation_success is True
            assert "surface" in result.lux_source or "PhotoMaterial" in result.lux_source
        finally:
            os.unlink(tmp_path)

    def test_vision_not_supported_raises(self):
        """Provider without vision raises NotImplementedError."""
        from unittest.mock import MagicMock, patch
        import base64
        import tempfile
        import os
        from luxc.ai.generate import generate_material_from_image
        from luxc.ai.config import AIConfig

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mP8/58BAwAI/AL+hc2rNAAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            tmp_path = f.name

        mock_provider = MagicMock()
        mock_provider.supports_vision = False
        mock_config = AIConfig(provider="ollama", model="llama3.2")

        try:
            with patch("luxc.ai.generate._resolve_provider", return_value=(mock_provider, mock_config)):
                with pytest.raises(NotImplementedError, match="does not support image"):
                    generate_material_from_image(tmp_path)
        finally:
            os.unlink(tmp_path)
