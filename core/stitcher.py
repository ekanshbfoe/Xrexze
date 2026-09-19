"""
Xrexze Final Stitcher — FFmpeg stream-copy concatenation.

Merges all rendered .mp4 chunks into a single output video
using FFmpeg's concat demuxer with stream copy (no re-encoding).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def create_concat_file(
    chunk_paths: List[Path], concat_file_path: Path
) -> Path:
    """Write an FFmpeg concat demuxer file listing all chunks."""
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(concat_file_path, "w", encoding="utf-8") as f:
        for chunk_path in chunk_paths:
            safe_path = str(chunk_path.resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    logger.info(
        f"Concat file written: {concat_file_path} "
        f"({len(chunk_paths)} chunks)"
    )
    return concat_file_path


def stitch_final_video(
    chunk_paths: List[Path],
    output_path: Path,
    temp_dir: Path,
) -> Path:
    """
    Concatenate all .mp4 chunks into the final video using FFmpeg
    stream copy (no re-encoding).
    """
    if not chunk_paths:
        raise ValueError("No video chunks to stitch")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    concat_file = temp_dir / "concat.txt"
    create_concat_file(chunk_paths, concat_file)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"Stitching {len(chunk_paths)} chunks -> {output_path.name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        logger.error(f"FFmpeg stitch error:\n{result.stderr}")
        raise RuntimeError(
            f"FFmpeg concat failed: {result.stderr[-500:]}"
        )

    final_size = output_path.stat().st_size
    logger.info(
        f"Final video stitched: {output_path.name}, "
        f"size={final_size / 1024 / 1024:.1f} MB"
    )

    return output_path
