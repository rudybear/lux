"""Tests for AI video-to-animation support (Phase 16)."""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from luxc.ai.video import extract_key_frames, describe_motion


class TestExtractKeyFrames:
    def test_extract_key_frames_missing_cv2(self):
        """When cv2 is not available, ImportError is raised."""
        # Temporarily hide cv2 if it exists, then force ImportError
        with patch.dict(sys.modules, {"cv2": None}):
            with pytest.raises(ImportError, match="opencv-python"):
                extract_key_frames(Path("dummy_video.mp4"))

    def test_extract_key_frames_file_not_found(self):
        """Non-existent video path raises FileNotFoundError."""
        # Create a mock cv2 that is importable but the file doesn't exist
        mock_cv2 = MagicMock()
        mock_cv2.__name__ = "cv2"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(FileNotFoundError, match="not found"):
                extract_key_frames(Path("/nonexistent/path/video.mp4"))

    def test_extract_key_frames_mocked_cv2(self):
        """Mock cv2.VideoCapture to return fake frames, verify output format."""
        # Build a mock cv2 module
        mock_cv2 = MagicMock()
        mock_cv2.__name__ = "cv2"

        # Constants used by the code
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            7: 30,    # CAP_PROP_FRAME_COUNT
            5: 10.0,  # CAP_PROP_FPS
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap.set.return_value = None
        mock_cap.release.return_value = None

        mock_cv2.VideoCapture.return_value = mock_cap

        # Mock cv2.imencode to return success + fake PNG data
        fake_png = b"\x89PNG\r\n\x1a\nfakedata"
        mock_cv2.imencode.return_value = (True, np.frombuffer(fake_png, dtype=np.uint8))

        import tempfile
        import os

        # Create a temporary file so the path existence check passes
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            tmp_path = f.name

        try:
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                frames = extract_key_frames(Path(tmp_path), max_frames=4)

            # Verify output structure
            assert isinstance(frames, list)
            assert len(frames) > 0
            assert len(frames) <= 4

            for timestamp, b64_data, media_type in frames:
                # Timestamp is a float
                assert isinstance(timestamp, float)
                assert timestamp >= 0.0

                # base64 data is a non-empty string
                assert isinstance(b64_data, str)
                assert len(b64_data) > 0
                # Verify it decodes as valid base64
                decoded = base64.b64decode(b64_data)
                assert len(decoded) > 0

                # Media type is image/png
                assert media_type == "image/png"
        finally:
            os.unlink(tmp_path)


class TestDescribeMotion:
    def test_describe_motion(self):
        """Mock provider's complete_multimodal, verify it returns description text."""
        # Create fake key frames
        fake_b64 = base64.b64encode(b"fake image data").decode("ascii")
        key_frames = [
            (0.0, fake_b64, "image/png"),
            (1.0, fake_b64, "image/png"),
            (2.0, fake_b64, "image/png"),
        ]

        # Create mock provider with vision support
        mock_provider = MagicMock()
        mock_provider.supports_vision = True
        mock_provider.complete_multimodal.return_value = (
            "Oscillating radial pulse at ~2Hz with warm-to-cool color shift."
        )

        # Create mock config
        mock_config = MagicMock()
        mock_config.max_tokens = 4096

        result = describe_motion(key_frames, mock_provider, mock_config)

        assert isinstance(result, str)
        assert "pulse" in result.lower() or "oscillat" in result.lower()
        assert len(result) > 0

        # Verify the provider was called with multimodal content
        mock_provider.complete_multimodal.assert_called_once()
        call_args = mock_provider.complete_multimodal.call_args
        system_prompt = call_args[0][0]
        messages = call_args[0][1]

        # System prompt should mention motion analysis
        assert "motion" in system_prompt.lower()

        # Messages should contain image entries
        assert len(messages) == 1
        content = messages[0]["content"]
        image_entries = [c for c in content if c.get("type") == "image_base64"]
        assert len(image_entries) == 3  # One per frame

    def test_describe_motion_no_vision(self):
        """Provider without vision support raises NotImplementedError."""
        fake_b64 = base64.b64encode(b"fake image data").decode("ascii")
        key_frames = [
            (0.0, fake_b64, "image/png"),
        ]

        mock_provider = MagicMock()
        mock_provider.supports_vision = False

        mock_config = MagicMock()
        mock_config.max_tokens = 4096

        with pytest.raises(NotImplementedError, match="Vision support required"):
            describe_motion(key_frames, mock_provider, mock_config)
