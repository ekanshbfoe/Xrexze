"""
Xrexze Video Compositor — Low-memory chunk renderer.

Builds 1920x1080 canvases from panel images with blurred backgrounds,
syncs duration to audio, and exports .mp4 chunks one at a time.

Memory strategy:
  - Process exactly ONE panel at a time
  - Explicitly close all PIL Image objects after use
  - Call gc.collect() after every chunk
  - Never hold more than one frame in memory
  - Use FFmpeg subprocess directly (no MoviePy in-memory video objects)
"""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance, ImageFilter

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

BLUR_RADIUS = 30
BG_BRIGHTNESS = 0.3
JPEG_QUALITY = 95


def build_composite_frame(
    panel_path: Path,
    canvas_size: Tuple[int, int] = (1920, 1080),
) -> Image.Image:
    """
    Build a single 16:9 composite frame:
    blurred+darkened background with centered sharp panel.

    Parameters
    ----------
    panel_path : Path
        Path to the cropped panel image.
    canvas_size : tuple
        Output resolution (width, height).

    Returns
    -------
    PIL.Image.Image
        The composite frame, mode RGB.
    """
    canvas_w, canvas_h = canvas_size

    panel = Image.open(panel_path).convert("RGB")
    panel_w, panel_h = panel.size

    # Resize panel to fit within canvas
    scale_h = canvas_h / panel_h
    scale_w = canvas_w / panel_w
    scale = min(scale_h, scale_w)

    new_w = int(panel_w * scale)
    new_h = int(panel_h * scale)
    resized_panel = panel.resize((new_w, new_h), Image.LANCZOS)

    # Build blurred, darkened background
    bg = panel.resize((canvas_w, canvas_h), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    bg = ImageEnhance.Brightness(bg).enhance(BG_BRIGHTNESS)

    # Center the sharp panel on the background
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    bg.paste(resized_panel, (x_offset, y_offset))

    panel.close()
    resized_panel.close()

    return bg


def render_video_chunk(
    panel_path: Path,
    audio_path: Path,
    output_path: Path,
    canvas_size: Tuple[int, int] = (1920, 1080),
    fps: int = 1,
) -> dict:
    """
    Render a single .mp4 video chunk: static composite frame + audio.

    Uses FFmpeg directly via subprocess to avoid MoviePy memory overhead.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building composite frame for {panel_path.name}")
    frame = build_composite_frame(panel_path, canvas_size)

    temp_frame_path = output_path.parent / f"_temp_frame_{output_path.stem}.jpg"
    frame.save(str(temp_frame_path), "JPEG", quality=JPEG_QUALITY)
    frame.close()
    del frame
    gc.collect()

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(temp_frame_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={canvas_size[0]}:{canvas_size[1]}",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"FFmpeg render: {output_path.name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if temp_frame_path.exists():
        temp_frame_path.unlink()

    if result.returncode != 0:
        logger.error(f"FFmpeg error:\n{result.stderr}")
        raise RuntimeError(
            f"FFmpeg failed for {output_path.name}: {result.stderr[-500:]}"
        )

    file_size = output_path.stat().st_size
    duration = _get_video_duration(output_path)

    logger.info(
        f"Chunk rendered: {output_path.name}, "
        f"duration={duration:.1f}s, size={file_size / 1024 / 1024:.1f}MB"
    )

    gc.collect()

    return {
        "video_path": output_path,
        "duration_sec": duration,
        "file_size_bytes": file_size,
    }


def _get_video_duration(video_path: Path) -> float:
    """Use ffprobe to get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
