"""Video-to-animation shader generation support.

Extracts key frames from video files using OpenCV (optional dependency)
and uses vision AI to describe motion patterns for animated shader generation.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path


def extract_key_frames(
    video_path: Path,
    max_frames: int = 6,
    strategy: str = "uniform",
) -> list[tuple[float, str, str]]:
    """Extract key frames from a video file.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    max_frames : int
        Maximum number of frames to extract (default: 6).
    strategy : str
        Frame selection strategy. Only "uniform" is currently supported
        (evenly spaced across the video duration).

    Returns
    -------
    list of (timestamp, base64_data, media_type)
        Each entry is a frame with its timestamp in seconds,
        base64-encoded PNG data, and MIME type.

    Raises
    ------
    ImportError
        If opencv-python is not installed.
    FileNotFoundError
        If the video file doesn't exist.
    ValueError
        If the video cannot be opened or has no frames.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video processing. "
            "Install it with: pip install 'luxc[ai-video]'"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Compute frame indices based on strategy
    if strategy == "uniform":
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
    else:
        raise ValueError(f"Unknown frame extraction strategy: {strategy}")

    result: list[tuple[float, str, str]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = idx / fps

        # Encode frame as PNG in memory
        success, png_data = cv2.imencode(".png", frame)
        if not success:
            continue

        b64 = base64.standard_b64encode(png_data.tobytes()).decode("ascii")
        result.append((timestamp, b64, "image/png"))

    cap.release()
    return result


def describe_motion(
    key_frames: list[tuple[float, str, str]],
    provider_instance,
    config,
) -> str:
    """Use vision AI to describe the motion pattern from key frames.

    Sends a grid of key frames to the vision model and asks it to
    describe the motion, colour changes, and temporal patterns
    suitable for driving animated shader parameters.

    Parameters
    ----------
    key_frames : list of (timestamp, base64_data, media_type)
        Extracted key frames from extract_key_frames().
    provider_instance : AIProvider
        An instantiated AI provider with vision support.
    config : AIConfig
        Configuration for the AI provider.

    Returns
    -------
    str
        A textual description of the observed motion pattern.
    """
    if not provider_instance.supports_vision:
        raise NotImplementedError(
            f"Vision support required for video analysis. "
            f"The current model does not support image inputs."
        )

    system = (
        "You are a motion analysis expert. Analyse the provided video frames "
        "(ordered by timestamp) and describe:\n"
        "1. The dominant motion pattern (oscillation, rotation, drift, pulse, etc.)\n"
        "2. Colour or intensity changes over time\n"
        "3. The approximate cycle period or speed of change\n"
        "4. Any spatial patterns (radial, linear, turbulent)\n\n"
        "Your description will be used to generate an animated shader, so focus on "
        "parameters like frequency, amplitude, direction, and noise characteristics. "
        "Be concise and specific."
    )

    # Build multimodal message with all frames
    content: list[dict] = []
    for timestamp, b64, media_type in key_frames:
        content.append({
            "type": "image_base64",
            "data": b64,
            "media_type": media_type,
        })
        content.append({
            "type": "text",
            "text": f"[Frame at t={timestamp:.2f}s]",
        })

    content.append({
        "type": "text",
        "text": "Describe the motion pattern observed across these frames.",
    })

    messages = [{"role": "user", "content": content}]
    return provider_instance.complete_multimodal(system, messages, config.max_tokens)
