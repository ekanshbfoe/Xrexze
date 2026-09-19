# Xrexze — ManhwaExplainerStudio: Implementation Specification

> **Project:** Xrexze  
> **Maintainer:** ACL Community  
> **Repository:** [github.com/ekanshbfoe/Xrexze](https://github.com/ekanshbfoe/Xrexze.git)  
> **License:** MIT  
> **Target Hardware:** Intel Core i3 / 8 GB RAM / No GPU / Windows 10+ & Linux  
> **Python:** 3.10 – 3.12  

---

## Table of Contents

1. [Project Directory Layout](#1-project-directory-layout)
2. [Exact Dependencies](#2-exact-dependencies-requirementstxt)
3. [Environment Configuration](#3-environment-configuration)
4. [Data Contracts & Typed Interfaces](#4-data-contracts--typed-interfaces)
5. [Phase 1 — Gutter Detection & Auto-Slicing](#5-phase-1--gutter-detection--auto-slicing)
6. [Phase 2 — Vision Prompt Engineering & VLM Wrapper](#6-phase-2--vision-prompt-engineering--vlm-wrapper)
7. [Phase 3 — OmniVoice Integration via Gradio Client](#7-phase-3--omnivoice-integration-via-gradio-client)
8. [Phase 4 — Low-Memory Video Compositor](#8-phase-4--low-memory-video-compositor)
9. [Phase 5 — PyQt6 GUI Architecture & Concurrency Model](#9-phase-5--pyqt6-gui-architecture--concurrency-model)
10. [Edge Cases & Failure Recovery](#10-edge-cases--failure-recovery)
11. [Testing & Verification Matrix](#11-testing--verification-matrix)

---

## 1. Project Directory Layout

```text
Xrexze/
├── assets/
│   ├── icons/
│   │   ├── app_icon.png
│   │   └── status_icons/
│   │       ├── pending.svg
│   │       ├── scripting.svg
│   │       ├── audio.svg
│   │       ├── rendering.svg
│   │       └── done.svg
│   └── fonts/
│       └── NotoSansDevanagari-Regular.ttf
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic-based settings loader from .env
├── core/
│   ├── __init__.py
│   ├── models.py                # Typed data contracts (PanelMetadata, etc.)
│   ├── slicer.py                # OpenCV panel detection & cropping
│   ├── vision_client.py         # OpenAI-compatible VLM interface with retry/proxy logic
│   ├── voice_client.py          # Gradio Client bridge to Hugging Face OmniVoice
│   ├── compositor.py            # Pillow layout & FFmpeg chunk renderer
│   ├── stitcher.py              # FFmpeg stream-copy concatenation
│   └── pipeline.py              # Orchestrator: chunk-by-chunk sequential pipeline
├── gui/
│   ├── __init__.py
│   ├── main_window.py           # PyQt6 primary layout (3-panel + header)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── telemetry_bar.py     # CPU / RAM / Stage header widget
│   │   ├── ingestion_panel.py   # Drag-drop + settings (left panel)
│   │   ├── state_table.py       # Kanban / panel state tracker (center)
│   │   └── qa_deck.py           # Preview + script editor + audio player (right)
│   └── workers.py               # QThread workers for non-blocking pipeline execution
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Rotating file + console logger
│   └── system_monitor.py        # psutil wrappers for telemetry
├── tests/
│   ├── test_slicer.py
│   ├── test_vision_client.py
│   ├── test_voice_client.py
│   ├── test_compositor.py
│   ├── test_stitcher.py
│   └── conftest.py
├── temp/                        # Runtime temp dir (gitignored)
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── implementation.md            # This file
└── main.py                      # Entry point
```

### File Roles

| File | Responsibility |
|------|---------------|
| `config/settings.py` | Loads `.env`, validates keys, exposes a singleton `AppSettings` Pydantic model |
| `core/models.py` | Pydantic/dataclass contracts: `PanelMetadata`, `ScriptChunk`, `AudioChunk`, `VideoChunk`, `ProjectManifest` |
| `core/slicer.py` | Scans vertical webtoon strips for gutter rows, outputs numbered panels |
| `core/vision_client.py` | Sends panel images to a VLM endpoint, returns narration scripts |
| `core/voice_client.py` | Sends Hindi text to OmniVoice Gradio Space, downloads `.wav` |
| `core/compositor.py` | Builds 1920×1080 canvases, syncs to audio duration, exports `.mp4` chunks |
| `core/stitcher.py` | Concatenates `.mp4` chunks via FFmpeg stream-copy |
| `core/pipeline.py` | Orchestrates A→B→C→D per panel, then E for final stitch |
| `gui/workers.py` | QThread subclasses emitting `pyqtSignal` for each pipeline stage |
| `utils/system_monitor.py` | Polls `psutil` for CPU %, RAM GB, disk I/O |

---

## 2. Exact Dependencies (`requirements.txt`)

```text
# Core GUI
PyQt6==6.7.1
PyQt6-Qt6==6.7.3
PyQt6-sip==13.8.0

# Computer Vision (headless — no Qt conflict)
opencv-python-headless==4.10.0.84

# Image Processing
Pillow==10.4.0

# Video Compositing
moviepy==1.0.3
ffmpeg-python==0.2.0

# VLM API Client
requests==2.32.3
openai==1.40.0

# Gradio Client (for Hugging Face Spaces)
gradio_client==1.3.0

# System Telemetry
psutil==6.0.0

# Configuration
python-dotenv==1.0.1
pydantic==2.8.2
pydantic-settings==2.4.0

# Utilities
numpy==1.26.4
```

### System-Level Requirements (Not pip-installable)

| Dependency | Version | Install |
|-----------|---------|---------|
| FFmpeg | >= 5.0 | `winget install FFmpeg` (Windows) / `sudo apt install ffmpeg` (Linux) |
| Python | 3.10 - 3.12 | https://python.org |

---

## 3. Environment Configuration

### `.env.example`

```env
# ──────────────────────────────────────────────
# Xrexze — Environment Configuration
# Copy this file to .env and fill in your values
# ──────────────────────────────────────────────

# ── VLM API (OpenAI-Compatible Endpoint) ─────
# Base URL for the vision-language model API.
# Examples:
#   AIHubMix:       https://aihubmix.com/v1
#   OpenRouter:     https://openrouter.ai/api/v1
#   Cloudflare:     https://your-worker.your-subdomain.workers.dev/v1
#   Local (Ollama): http://localhost:11434/v1
VLM_API_BASE_URL=https://aihubmix.com/v1

# Bearer token(s) for the VLM endpoint.
# For key rotation, provide comma-separated keys:
#   VLM_API_KEYS=sk-key1,sk-key2,sk-key3
VLM_API_KEYS=sk-your-key-here

# Model identifier as listed by the provider.
VLM_MODEL_NAME=nvidia/Llama-3.1-Nemotron-Nano-12B-V2-VL

# ── OmniVoice (Hugging Face Gradio Space) ────
# The Gradio Space ID (format: username/space-name)
HF_OMNIVOICE_SPACE_ID=your-username/omnivoice-space

# Hugging Face token (optional; required if Space is private)
HF_TOKEN=hf_your_token_here

# ── Narration Language ───────────────────────
# Options: "hindi_devanagari", "hinglish", "english"
NARRATION_LANGUAGE=hindi_devanagari

# ── Output Settings ──────────────────────────
OUTPUT_DIR=./output
TEMP_DIR=./temp

# Video resolution
VIDEO_WIDTH=1920
VIDEO_HEIGHT=1080

# ── Performance Tuning ───────────────────────
MAX_RAM_USAGE_GB=6.0
API_RETRY_DELAY_SEC=5
API_MAX_RETRIES=3
VLM_TIMEOUT_SEC=120
VOICE_TIMEOUT_SEC=300
```

### `config/settings.py`

```python
"""
Xrexze Configuration — Pydantic Settings loader.

Reads from .env file at project root and validates all required
configuration values at startup. Exposes a singleton `settings` object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # ── VLM API ──────────────────────────────
    vlm_api_base_url: str = Field(
        ..., description="Base URL for OpenAI-compatible VLM endpoint"
    )
    vlm_api_keys: str = Field(
        ..., description="Comma-separated API keys for rotation"
    )
    vlm_model_name: str = Field(
        default="nvidia/Llama-3.1-Nemotron-Nano-12B-V2-VL"
    )
    vlm_timeout_sec: int = Field(default=120)

    # ── OmniVoice ────────────────────────────
    hf_omnivoice_space_id: str = Field(
        ..., description="Gradio Space ID: username/space-name"
    )
    hf_token: str = Field(default="", description="HF token (optional)")
    voice_timeout_sec: int = Field(default=300)

    # ── Language ─────────────────────────────
    narration_language: Literal[
        "hindi_devanagari", "hinglish", "english"
    ] = Field(default="hindi_devanagari")

    # ── Paths ────────────────────────────────
    output_dir: Path = Field(default=Path("./output"))
    temp_dir: Path = Field(default=Path("./temp"))

    # ── Video ────────────────────────────────
    video_width: int = Field(default=1920)
    video_height: int = Field(default=1080)

    # ── Performance ──────────────────────────
    max_ram_usage_gb: float = Field(default=6.0)
    api_retry_delay_sec: int = Field(default=5)
    api_max_retries: int = Field(default=3)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def api_keys_list(self) -> List[str]:
        """Split comma-separated keys into a rotation list."""
        return [k.strip() for k in self.vlm_api_keys.split(",") if k.strip()]

    @field_validator("output_dir", "temp_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# ── Singleton ─────────────────────────────────
_settings_instance: AppSettings | None = None


def get_settings() -> AppSettings:
    """Return the global settings singleton, initializing on first call."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance
```

---

## 4. Data Contracts & Typed Interfaces

All inter-module data is passed through strictly typed dataclasses. No raw dicts cross module boundaries.

### `core/models.py`

```python
"""
Typed data contracts for the Xrexze pipeline.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class PanelState(enum.Enum):
    """Discrete states for the pipeline Kanban board."""
    PENDING = "Pending"
    SCRIPTING = "Scripting"
    AUDIO_SYNTHESIS = "Audio Synthesis"
    RENDERING = "Rendering"
    DONE = "Done"
    FAILED = "Failed"


@dataclass
class PanelMetadata:
    """
    Represents a single cropped panel extracted from a webtoon strip.

    Attributes:
        panel_id:       Sequential index (e.g. 1, 2, 3...).
        source_page:    Path to the original full-page image file.
        panel_path:     Path to the cropped panel image on disk.
        y_start:        Top pixel row in the source page (for debugging).
        y_end:          Bottom pixel row in the source page.
        width:          Panel width in pixels.
        height:         Panel height in pixels.
        state:          Current pipeline state.
    """
    panel_id: int
    source_page: Path
    panel_path: Path
    y_start: int
    y_end: int
    width: int
    height: int
    state: PanelState = PanelState.PENDING


@dataclass
class ScriptChunk:
    """
    Narration script generated by the VLM for a single panel.

    Attributes:
        panel_id:       References back to PanelMetadata.panel_id.
        narration_text: The Hindi/Hinglish narration string.
        raw_vlm_response: Full JSON response from the VLM for debugging.
        language:       Language variant used.
        estimated_duration_sec: Rough duration estimate from word count.
    """
    panel_id: int
    narration_text: str
    raw_vlm_response: str = ""
    language: str = "hindi_devanagari"
    estimated_duration_sec: float = 0.0


@dataclass
class AudioChunk:
    """
    Audio file generated by OmniVoice for a single panel's narration.

    Attributes:
        panel_id:       References PanelMetadata.panel_id.
        audio_path:     Path to the saved .wav file on disk.
        duration_sec:   Actual audio duration in seconds.
        sample_rate:    Sample rate of the wav (typically 24000 or 22050).
        file_size_bytes: Size on disk for memory budgeting.
    """
    panel_id: int
    audio_path: Path
    duration_sec: float
    sample_rate: int = 24000
    file_size_bytes: int = 0


@dataclass
class VideoChunk:
    """
    Rendered .mp4 chunk for a single panel (static image + audio).

    Attributes:
        panel_id:       References PanelMetadata.panel_id.
        video_path:     Path to the rendered .mp4 chunk on disk.
        duration_sec:   Duration matching the audio.
        resolution:     Tuple of (width, height).
        file_size_bytes: Size on disk.
    """
    panel_id: int
    video_path: Path
    duration_sec: float
    resolution: tuple[int, int] = (1920, 1080)
    file_size_bytes: int = 0


@dataclass
class ProjectManifest:
    """
    Persistent state for an entire project, enabling crash recovery.
    Serialized to JSON after every panel completes.

    Attributes:
        project_name:   User-defined project name.
        source_dir:     Directory containing raw webtoon pages.
        temp_dir:       Working directory for intermediate files.
        output_path:    Final output video path.
        panels:         Ordered list of all panel metadata.
        scripts:        Dict mapping panel_id -> ScriptChunk.
        audios:         Dict mapping panel_id -> AudioChunk.
        videos:         Dict mapping panel_id -> VideoChunk.
        last_completed_panel_id: For crash recovery; resume from here + 1.
    """
    project_name: str
    source_dir: Path
    temp_dir: Path
    output_path: Path
    panels: list[PanelMetadata] = field(default_factory=list)
    scripts: dict[int, ScriptChunk] = field(default_factory=dict)
    audios: dict[int, AudioChunk] = field(default_factory=dict)
    videos: dict[int, VideoChunk] = field(default_factory=dict)
    last_completed_panel_id: int = -1
```

---

## 5. Phase 1 — Gutter Detection & Auto-Slicing

### Algorithm Overview

Manhwa/webtoon pages are long vertical strips (typically 800-1200 px wide, 5000-20000 px tall). Panels are separated by horizontal "gutter" rows — solid bands of white, black, or near-uniform color spanning the full width.

**Detection strategy:**
1. Convert the page to grayscale.
2. For each row, compute the pixel intensity variance across the row.
3. Rows with variance below a threshold (<= 50) are candidate gutter rows.
4. Group consecutive gutter rows into "gutter bands."
5. Split the page at the midpoint of each gutter band.
6. Discard panels shorter than `MIN_PANEL_HEIGHT` (100 px) to filter artifacts.

### `core/slicer.py`

```python
"""
Xrexze Panel Slicer — OpenCV gutter detection & panel extraction.

Processes a single webtoon page image at a time to respect
the 2.5 GB RAM headroom constraint. Each page is loaded, sliced,
and immediately freed before the next page is loaded.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Tunable Constants ─────────────────────────
GUTTER_VARIANCE_THRESHOLD = 50.0   # Max per-row pixel variance to be a gutter
MIN_GUTTER_BAND_HEIGHT = 5         # Min consecutive gutter rows to split
MIN_PANEL_HEIGHT = 100             # Discard panels shorter than this (pixels)
EDGE_MARGIN_RATIO = 0.05           # Ignore 5% of left/right edges for variance


def compute_row_variances(
    gray_image: np.ndarray, margin_ratio: float = EDGE_MARGIN_RATIO
) -> np.ndarray:
    """
    Compute per-row pixel variance, ignoring edge margins.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image, shape (H, W), dtype uint8.
    margin_ratio : float
        Fraction of width to ignore on each side (handles page borders).

    Returns
    -------
    np.ndarray
        1D array of shape (H,) with the variance of each row.
    """
    h, w = gray_image.shape
    margin = int(w * margin_ratio)
    center_band = gray_image[:, margin : w - margin].astype(np.float32)
    return np.var(center_band, axis=1)


def find_gutter_bands(
    variances: np.ndarray,
    threshold: float = GUTTER_VARIANCE_THRESHOLD,
    min_band_height: int = MIN_GUTTER_BAND_HEIGHT,
) -> List[Tuple[int, int]]:
    """
    Identify contiguous bands of low-variance rows (gutters).

    Returns
    -------
    List of (band_start_row, band_end_row) tuples.
    """
    is_gutter = variances <= threshold
    bands: List[Tuple[int, int]] = []
    band_start: int | None = None

    for row_idx in range(len(is_gutter)):
        if is_gutter[row_idx]:
            if band_start is None:
                band_start = row_idx
        else:
            if band_start is not None:
                band_length = row_idx - band_start
                if band_length >= min_band_height:
                    bands.append((band_start, row_idx - 1))
                band_start = None

    # Handle gutter at the very bottom of the image
    if band_start is not None:
        band_length = len(is_gutter) - band_start
        if band_length >= min_band_height:
            bands.append((band_start, len(is_gutter) - 1))

    return bands


def compute_split_points(
    bands: List[Tuple[int, int]], image_height: int
) -> List[int]:
    """
    Convert gutter bands to split points (midpoints of each band).
    Prepends 0 and appends image_height as implicit boundaries.
    """
    splits = [0]
    for start, end in bands:
        midpoint = (start + end) // 2
        splits.append(midpoint)
    splits.append(image_height)
    return splits


def slice_page(
    page_path: Path,
    output_dir: Path,
    panel_id_offset: int = 0,
    min_panel_height: int = MIN_PANEL_HEIGHT,
) -> List[dict]:
    """
    Slice a single webtoon page into discrete panel images.

    Parameters
    ----------
    page_path : Path
        Absolute path to the source page image.
    output_dir : Path
        Directory to write cropped panel PNGs.
    panel_id_offset : int
        Starting panel ID (for multi-page chapters).
    min_panel_height : int
        Minimum height in pixels; shorter crops are discarded.

    Returns
    -------
    List of dicts with keys: panel_id, panel_path, y_start, y_end,
    width, height — ready to construct PanelMetadata objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img_color = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img_color is None:
        logger.error(f"Failed to read image: {page_path}")
        return []

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    logger.info(f"Processing page: {page_path.name} ({w}x{h})")

    # Verify minimum dimensions
    if h < 200 or w < 200:
        logger.warning(f"Image too small ({w}x{h}), skipping: {page_path}")
        del img_color, img_gray
        gc.collect()
        return []

    variances = compute_row_variances(img_gray)
    bands = find_gutter_bands(variances)
    logger.info(f"  Found {len(bands)} gutter band(s)")

    splits = compute_split_points(bands, h)

    panels_info: List[dict] = []
    current_id = panel_id_offset

    for i in range(len(splits) - 1):
        y_start = splits[i]
        y_end = splits[i + 1]
        panel_height = y_end - y_start

        if panel_height < min_panel_height:
            logger.debug(
                f"  Skipping thin strip: rows {y_start}-{y_end} "
                f"(height={panel_height})"
            )
            continue

        current_id += 1
        panel_crop = img_color[y_start:y_end, :]
        panel_filename = f"panel_{current_id:04d}.png"
        panel_path = output_dir / panel_filename

        cv2.imwrite(str(panel_path), panel_crop)

        panels_info.append({
            "panel_id": current_id,
            "source_page": page_path,
            "panel_path": panel_path,
            "y_start": y_start,
            "y_end": y_end,
            "width": w,
            "height": panel_height,
        })

        logger.info(
            f"  Panel {current_id}: rows {y_start}-{y_end} "
            f"({w}x{panel_height}) -> {panel_filename}"
        )

    # Explicit memory release
    del img_color, img_gray, variances
    gc.collect()

    return panels_info


def slice_chapter(
    page_paths: List[Path], output_dir: Path
) -> List[dict]:
    """
    Slice an entire chapter (multiple pages) into panels, processing
    one page at a time to stay within RAM budget.

    Parameters
    ----------
    page_paths : List[Path]
        Ordered list of page image paths.
    output_dir : Path
        Directory to write all panel PNGs.

    Returns
    -------
    List of all panel info dicts, sequentially numbered.
    """
    all_panels: List[dict] = []
    panel_id_offset = 0

    for page_path in sorted(page_paths):
        page_panels = slice_page(
            page_path=page_path,
            output_dir=output_dir,
            panel_id_offset=panel_id_offset,
        )
        if page_panels:
            panel_id_offset = page_panels[-1]["panel_id"]
            all_panels.extend(page_panels)

    logger.info(f"Chapter slicing complete: {len(all_panels)} panels extracted")
    return all_panels
```

---

## 6. Phase 2 — Vision Prompt Engineering & VLM Wrapper

### System Prompt Design

The system prompt instructs the VLM to:
1. Act as a professional manhwa narrator and visual analyst.
2. Read any visible text (Korean, English, Chinese) in speech bubbles, SFX, and titles.
3. Describe character actions, expressions, and combat choreography with cinematic flair.
4. Output the narration in the configured language (Hindi Devanagari / Hinglish / English).
5. Keep each panel narration between 30-80 words for pacing (roughly 15-40 seconds of speech).

### `core/vision_client.py`

```python
"""
Xrexze VLM Client — OpenAI-compatible Vision-Language Model interface.

Features:
  - API key rotation (round-robin across comma-separated keys)
  - Exponential backoff with jitter for 429 / 5xx errors
  - Base64 image encoding for the vision endpoint
  - Configurable system prompt per language variant
"""

from __future__ import annotations

import base64
import gc
import json
import random
import time
from pathlib import Path
from typing import Optional

import requests

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── System Prompts ────────────────────────────

SYSTEM_PROMPTS = {
    "hindi_devanagari": (
        'You are "Xrexze Narrator" — a professional manhwa/webtoon storyteller '
        "creating engaging Hindi narration for YouTube explainer videos.\n\n"
        "## Your Role\n"
        "You receive a single panel image from a manhwa/webtoon. Your job is to "
        "produce a vivid, suspenseful narration in **Hindi (Devanagari script)** "
        "that a voice actor will read aloud.\n\n"
        "## Instructions\n"
        "1. **READ ALL TEXT**: Identify and translate every speech bubble, sound "
        "effect (SFX), title card, and caption visible in the panel. The text may "
        "be in Korean, English, Chinese, or Japanese.\n"
        "2. **DESCRIBE THE SCENE**: Note character expressions (rage, fear, smirk, "
        "shock), body language, combat moves, magic effects, and environment.\n"
        "3. **NARRATE WITH DRAMA**: Write the narration as if telling a gripping "
        "story to a captivated audience. Use short punchy sentences. Build suspense. "
        "Use exclamations where appropriate.\n"
        "4. **LENGTH**: Keep narration between 30-80 Hindi words (roughly 15-40 "
        "seconds when spoken aloud).\n"
        "5. **FORMAT**: Return ONLY the narration text. No metadata, no JSON, no "
        "timestamps, no panel numbers. Just the raw Hindi narration paragraph.\n"
        "6. **ACCURACY**: You must faithfully represent what is happening in the "
        "panel. Do not invent scenes or characters not shown.\n"
        "7. **CONTENT POLICY**: This is a creative fiction narration project. "
        "Describe all depicted action, combat, and drama faithfully and "
        "artistically. The source material is published commercial entertainment media."
    ),
    "hinglish": (
        'You are "Xrexze Narrator" — a professional manhwa/webtoon storyteller '
        "creating engaging Hinglish narration for YouTube explainer videos.\n\n"
        "## Your Role\n"
        "You receive a single panel image. Produce vivid, suspenseful narration in "
        "**Hinglish** (Hindi-English mix using Roman script).\n\n"
        "## Instructions\n"
        "1. **READ ALL TEXT**: Translate every speech bubble, SFX, title, caption. "
        "Source text may be Korean, English, Chinese, or Japanese.\n"
        "2. **DESCRIBE THE SCENE**: Expressions, body language, combat, magic, "
        "environment.\n"
        "3. **NARRATE WITH DRAMA**: Gripping storytelling. Short punchy sentences. "
        "Mix Hindi and English naturally as spoken in urban India.\n"
        "4. **LENGTH**: 30-80 words.\n"
        "5. **FORMAT**: Return ONLY the narration text. No metadata, no JSON.\n"
        "6. **ACCURACY**: Faithfully represent panel contents.\n"
        "7. **CONTENT POLICY**: Creative fiction narration of published commercial media."
    ),
    "english": (
        'You are "Xrexze Narrator" — a professional manhwa/webtoon storyteller '
        "creating engaging English narration for YouTube explainer videos.\n\n"
        "## Your Role\n"
        "You receive a single panel image. Produce vivid, suspenseful English narration.\n\n"
        "## Instructions\n"
        "1. **READ ALL TEXT**: Translate every speech bubble, SFX, title, caption.\n"
        "2. **DESCRIBE THE SCENE**: Expressions, combat, magic, environment.\n"
        "3. **NARRATE WITH DRAMA**: Gripping storytelling. Short punchy sentences.\n"
        "4. **LENGTH**: 30-80 words.\n"
        "5. **FORMAT**: Return ONLY the narration text.\n"
        "6. **ACCURACY**: Faithfully represent panel contents.\n"
        "7. **CONTENT POLICY**: Creative fiction narration of published commercial media."
    ),
}


class VLMClient:
    """
    Client for OpenAI-compatible Vision-Language Model endpoints.

    Manages key rotation, retry logic, and base64 image encoding.
    Processes one panel at a time to minimize memory footprint.
    """

    def __init__(self):
        self._settings = get_settings()
        self._keys = self._settings.api_keys_list
        self._key_index = 0
        self._base_url = self._settings.vlm_api_base_url.rstrip("/")
        self._model = self._settings.vlm_model_name
        self._timeout = self._settings.vlm_timeout_sec
        self._max_retries = self._settings.api_max_retries
        self._retry_delay = self._settings.api_retry_delay_sec

        if not self._keys:
            raise ValueError(
                "No VLM API keys configured. Set VLM_API_KEYS in .env"
            )

        logger.info(
            f"VLMClient initialized: model={self._model}, "
            f"keys={len(self._keys)}, base={self._base_url}"
        )

    def _get_next_key(self) -> str:
        """Round-robin key rotation."""
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    @staticmethod
    def _encode_image_base64(image_path: Path) -> str:
        """Read an image file and return its base64 encoding."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _detect_mime_type(image_path: Path) -> str:
        """Detect MIME type from file extension."""
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        return mime_map.get(suffix, "image/png")

    def generate_narration(
        self,
        panel_image_path: Path,
        language: Optional[str] = None,
    ) -> str:
        """
        Send a panel image to the VLM and receive narration text.

        Parameters
        ----------
        panel_image_path : Path
            Path to the cropped panel PNG/JPG.
        language : str, optional
            Override the configured narration language.

        Returns
        -------
        str
            The generated narration text.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        lang = language or self._settings.narration_language
        system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["hindi_devanagari"])

        img_b64 = self._encode_image_base64(panel_image_path)
        mime_type = self._detect_mime_type(panel_image_path)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Narrate this manhwa panel. Follow your system "
                                "instructions exactly. Return ONLY the narration."
                            ),
                        },
                    ],
                },
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            api_key = self._get_next_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            try:
                logger.info(
                    f"VLM request: panel={panel_image_path.name}, "
                    f"attempt={attempt}/{self._max_retries}"
                )

                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )

                if response.status_code == 429:
                    wait = self._retry_delay * attempt + random.uniform(0, 2)
                    logger.warning(
                        f"Rate limited (429). Rotating key, waiting {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = self._retry_delay * attempt
                    logger.warning(
                        f"Server error ({response.status_code}). "
                        f"Retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                narration = (
                    data["choices"][0]["message"]["content"].strip()
                )

                logger.info(
                    f"VLM success: {len(narration)} chars, "
                    f"panel={panel_image_path.name}"
                )

                del img_b64, payload
                gc.collect()

                return narration

            except requests.exceptions.Timeout:
                logger.warning(
                    f"VLM timeout after {self._timeout}s (attempt {attempt})"
                )
                last_error = TimeoutError(
                    f"VLM request timed out after {self._timeout}s"
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"VLM request error: {e}")
                last_error = e

        raise RuntimeError(
            f"VLM narration failed after {self._max_retries} attempts "
            f"for panel {panel_image_path.name}: {last_error}"
        )
```

---

## 7. Phase 3 — OmniVoice Integration via Gradio Client

### Architecture Notes

OmniVoice runs on a free Hugging Face CPU Space. Key considerations:
- **Cold starts:** The Space may be asleep; the first request can take 60-120s.
- **Audio format:** OmniVoice returns WAV (PCM 16-bit, typically 24 kHz mono).
- **Request pattern:** Send text -> receive audio file path -> download to local disk.

### `core/voice_client.py`

```python
"""
Xrexze Voice Client — Gradio bridge to Hugging Face OmniVoice Space.

Handles:
  - Gradio Client connection with HF token auth
  - Cold-start resilience (long timeout + retry)
  - Audio download and local caching
  - Duration measurement via wave module
"""

from __future__ import annotations

import gc
import shutil
import time
import wave
from pathlib import Path
from typing import Optional

from gradio_client import Client

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VoiceClient:
    """
    Client for OmniVoice TTS via a Hugging Face Gradio Space.

    Connects lazily on first use to avoid blocking app startup.
    Processes one text chunk at a time.
    """

    def __init__(self):
        self._settings = get_settings()
        self._space_id = self._settings.hf_omnivoice_space_id
        self._hf_token = self._settings.hf_token or None
        self._timeout = self._settings.voice_timeout_sec
        self._max_retries = self._settings.api_max_retries
        self._retry_delay = self._settings.api_retry_delay_sec
        self._client: Optional[Client] = None

    def _ensure_connected(self) -> Client:
        """
        Lazily initialize the Gradio Client connection.
        Handles cold-start by retrying with exponential backoff.
        """
        if self._client is not None:
            return self._client

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    f"Connecting to OmniVoice Space: {self._space_id} "
                    f"(attempt {attempt}/{self._max_retries})"
                )
                self._client = Client(
                    self._space_id,
                    hf_token=self._hf_token,
                )
                logger.info("OmniVoice connection established")
                return self._client

            except Exception as e:
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"OmniVoice connection failed: {e}. "
                    f"Retrying in {wait}s (cold start?)"
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Failed to connect to OmniVoice Space '{self._space_id}' "
            f"after {self._max_retries} attempts"
        )

    @staticmethod
    def _get_wav_duration(wav_path: Path) -> float:
        """Read WAV file header to determine duration in seconds."""
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0

    @staticmethod
    def _get_wav_sample_rate(wav_path: Path) -> int:
        """Read WAV sample rate from file header."""
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getframerate()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        panel_id: int = 0,
    ) -> dict:
        """
        Convert narration text to speech via OmniVoice.

        Parameters
        ----------
        text : str
            The narration text to synthesize (Hindi/Hinglish/English).
        output_path : Path
            Where to save the resulting .wav file.
        panel_id : int
            Panel ID for logging context.

        Returns
        -------
        dict with keys: audio_path, duration_sec, sample_rate, file_size_bytes

        Raises
        ------
        RuntimeError
            If synthesis fails after all retries.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._ensure_connected()

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    f"OmniVoice synthesis: panel={panel_id}, "
                    f"chars={len(text)}, attempt={attempt}"
                )

                result = client.predict(
                    text,
                    "hindi",
                    api_name="/predict",
                )

                if isinstance(result, dict):
                    remote_path = Path(result.get("value", result.get("name", "")))
                elif isinstance(result, (str, Path)):
                    remote_path = Path(result)
                else:
                    raise ValueError(
                        f"Unexpected OmniVoice result type: {type(result)}"
                    )

                if remote_path.exists():
                    shutil.copy2(str(remote_path), str(output_path))
                else:
                    shutil.copy2(str(result), str(output_path))

                duration = self._get_wav_duration(output_path)
                sample_rate = self._get_wav_sample_rate(output_path)
                file_size = output_path.stat().st_size

                logger.info(
                    f"OmniVoice success: panel={panel_id}, "
                    f"duration={duration:.1f}s, rate={sample_rate}Hz, "
                    f"size={file_size / 1024:.0f}KB"
                )

                return {
                    "audio_path": output_path,
                    "duration_sec": duration,
                    "sample_rate": sample_rate,
                    "file_size_bytes": file_size,
                }

            except Exception as e:
                wait = self._retry_delay * attempt
                logger.error(
                    f"OmniVoice error (attempt {attempt}): {e}. "
                    f"Retrying in {wait}s"
                )
                last_error = e
                time.sleep(wait)
                self._client = None

        raise RuntimeError(
            f"OmniVoice synthesis failed for panel {panel_id} "
            f"after {self._max_retries} attempts: {last_error}"
        )

    def close(self):
        """Release the Gradio client connection."""
        self._client = None
        gc.collect()
```

---

## 8. Phase 4 — Low-Memory Video Compositor

### Canvas Construction Mathematics

For a 1920x1080 output frame from a panel of arbitrary dimensions:

```
Given:
  panel_w, panel_h  = original panel dimensions
  canvas_w, canvas_h = 1920, 1080

Step 1 — Resize panel to fit height:
  scale_h = canvas_h / panel_h
  scale_w = canvas_w / panel_w
  scale = min(scale_h, scale_w)    # Fit inside without cropping
  new_w = int(panel_w * scale)
  new_h = int(panel_h * scale)

Step 2 — Build blurred background:
  bg = panel.resize((canvas_w, canvas_h))  # stretch to fill
  bg = bg.filter(GaussianBlur(radius=30))  # heavy blur
  bg = ImageEnhance.Brightness(bg).enhance(0.3)  # darken to 30%

Step 3 — Center the resized panel:
  x_offset = (canvas_w - new_w) // 2
  y_offset = (canvas_h - new_h) // 2
  bg.paste(resized_panel, (x_offset, y_offset))
```

### `core/compositor.py`

```python
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
```

### `core/stitcher.py`

```python
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
```

---

## 9. Phase 5 — PyQt6 GUI Architecture & Concurrency Model

### Thread Architecture

```
+------------------------------------------------------------------+
|                        MAIN THREAD (GUI)                          |
|  +-------------+  +--------------+  +-------------------------+  |
|  | Telemetry   |  | State Table  |  | QA Deck                 |  |
|  | Header      |  | (Kanban)     |  | (Preview/Edit/Audio)    |  |
|  +-------------+  +--------------+  +-------------------------+  |
|  +--------------------------------------------------------------+ |
|  | Ingestion Panel (Left): Drag-Drop, Settings, Controls        | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Receives signals from worker threads -> updates UI widgets        |
|  NEVER performs I/O, network calls, or heavy computation           |
+---------------------+--------------------------------------------+
                      | pyqtSignal connections
     +----------------+--------------------+
     v                v                    v
+-----------+  +---------------+  +------------------+
| Telemetry |  | Pipeline      |  | Audio Playback   |
| Worker    |  | Worker        |  | (QMediaPlayer)   |
| (QThread) |  | (QThread)     |  | Main thread OK   |
|           |  |               |  | for small .wav   |
| psutil    |  | slicer ->     |  +------------------+
| polling   |  | vlm_client -> |
| every 1s  |  | voice_client->|
|           |  | compositor -> |
| Emits:    |  | stitcher      |
| cpu_pct   |  |               |
| ram_gb    |  | Emits:        |
| stage     |  | panel_state   |
+-----------+  | script_ready  |
               | audio_ready   |
               | chunk_done    |
               | all_done      |
               | error         |
               +---------------+
```

### `gui/workers.py`

```python
"""
Xrexze GUI Workers — QThread-based non-blocking pipeline execution.

Two worker threads:
1. TelemetryWorker: polls psutil every 1s, emits system metrics.
2. PipelineWorker: runs the full A->B->C->D->E pipeline, emitting
   signals at each state transition so the GUI updates in real-time.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class TelemetryWorker(QThread):
    """Polls system metrics every second on a background thread."""

    telemetry_updated = pyqtSignal(float, float, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._current_stage = "Idle"

    def set_stage(self, stage: str):
        self._current_stage = stage

    def run(self):
        import psutil

        while self._running:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            ram_used = mem.used / (1024 ** 3)
            ram_total = mem.total / (1024 ** 3)
            self.telemetry_updated.emit(
                cpu, ram_used, ram_total, self._current_stage
            )
            time.sleep(1)

    def stop(self):
        self._running = False
        self.wait()


class PipelineWorker(QThread):
    """
    Executes the full pipeline on a background thread.
    Emits granular signals so the GUI can update in real-time.
    """

    panel_state_changed = pyqtSignal(int, str)
    script_ready = pyqtSignal(int, str, str)
    audio_ready = pyqtSignal(int, str, float)
    chunk_rendered = pyqtSignal(int, str)
    pipeline_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(int, str)
    progress_updated = pyqtSignal(int, int)

    def __init__(
        self,
        page_paths: list,
        mode: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self._page_paths = page_paths
        self._mode = mode
        self._running = True
        self._qa_approved = False
        self._edited_script: str | None = None

    def approve_panel(self, edited_script: str | None = None):
        """Called from GUI when user approves a panel in manual QA mode."""
        self._edited_script = edited_script
        self._qa_approved = True

    def stop(self):
        self._running = False

    def run(self):
        """Execute the full pipeline: Slice -> Script -> Audio -> Render -> Stitch."""
        from core.slicer import slice_chapter
        from core.vision_client import VLMClient
        from core.voice_client import VoiceClient
        from core.compositor import render_video_chunk
        from core.stitcher import stitch_final_video
        from config.settings import get_settings

        settings = get_settings()
        temp_dir = settings.temp_dir
        panels_dir = temp_dir / "panels"
        audio_dir = temp_dir / "audio"
        chunks_dir = temp_dir / "chunks"
        manifest_path = temp_dir / "manifest.json"

        # Check for existing manifest (crash recovery)
        resume_from = -1
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                resume_from = manifest.get("last_completed_panel_id", -1)
            except Exception:
                resume_from = -1

        # Phase 1: Slice
        self.panel_state_changed.emit(0, "Slicing pages...")
        panels = slice_chapter(
            [Path(p) for p in self._page_paths], panels_dir
        )

        if not panels:
            self.error_occurred.emit(
                0, "No panels detected in the provided pages"
            )
            return

        total = len(panels)
        vlm = VLMClient()
        voice = VoiceClient()
        chunk_paths: list[Path] = []

        for idx, panel_info in enumerate(panels):
            if not self._running:
                return

            pid = panel_info["panel_id"]
            panel_path = Path(panel_info["panel_path"])

            # Skip already-completed panels (crash recovery)
            if pid <= resume_from:
                chunk_path = chunks_dir / f"chunk_{pid:04d}.mp4"
                if chunk_path.exists():
                    chunk_paths.append(chunk_path)
                    self.panel_state_changed.emit(pid, "Done")
                    continue

            self.progress_updated.emit(idx + 1, total)

            # Phase 2: Script
            self.panel_state_changed.emit(pid, "Scripting")
            try:
                narration = vlm.generate_narration(panel_path)
            except RuntimeError as e:
                self.error_occurred.emit(pid, str(e))
                self.panel_state_changed.emit(pid, "Failed")
                continue

            # Validate narration quality
            if not narration or len(narration.strip()) < 10:
                try:
                    narration = vlm.generate_narration(panel_path)
                except RuntimeError:
                    pass
                if not narration or len(narration.strip()) < 10:
                    narration = (
                        f"[Panel {pid}: narration generation failed"
                        " - manual edit required]"
                    )

            # Manual QA mode: pause for user approval
            if self._mode == "manual_qa":
                self.script_ready.emit(pid, narration, str(panel_path))
                self._qa_approved = False
                while not self._qa_approved and self._running:
                    self.msleep(200)
                if not self._running:
                    return
                if self._edited_script:
                    narration = self._edited_script
                    self._edited_script = None

            # Phase 3: Audio
            self.panel_state_changed.emit(pid, "Audio Synthesis")
            audio_path = audio_dir / f"audio_{pid:04d}.wav"
            try:
                audio_result = voice.synthesize(
                    text=narration,
                    output_path=audio_path,
                    panel_id=pid,
                )
            except RuntimeError as e:
                self.error_occurred.emit(pid, str(e))
                self.panel_state_changed.emit(pid, "Failed")
                continue

            if self._mode == "manual_qa":
                self.audio_ready.emit(
                    pid, str(audio_path), audio_result["duration_sec"]
                )
                self._qa_approved = False
                while not self._qa_approved and self._running:
                    self.msleep(200)
                if not self._running:
                    return

            # Phase 4: Render
            self.panel_state_changed.emit(pid, "Rendering")
            chunk_path = chunks_dir / f"chunk_{pid:04d}.mp4"
            try:
                render_result = render_video_chunk(
                    panel_path=panel_path,
                    audio_path=audio_path,
                    output_path=chunk_path,
                )
                chunk_paths.append(chunk_path)
            except RuntimeError as e:
                self.error_occurred.emit(pid, str(e))
                self.panel_state_changed.emit(pid, "Failed")
                continue

            self.panel_state_changed.emit(pid, "Done")
            self.chunk_rendered.emit(pid, str(chunk_path))

            # Save manifest for crash recovery
            manifest = {
                "last_completed_panel_id": pid,
                "total_panels": total,
                "chunk_paths": [str(p) for p in chunk_paths],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

            gc.collect()

        # Phase 5: Stitch
        if chunk_paths:
            output_path = settings.output_dir / "final_output.mp4"
            try:
                stitch_final_video(chunk_paths, output_path, temp_dir)
                self.pipeline_complete.emit(str(output_path))
            except RuntimeError as e:
                self.error_occurred.emit(0, f"Stitching failed: {e}")

        voice.close()
        gc.collect()
```

### `gui/main_window.py`

```python
"""
Xrexze Main Window — PyQt6 primary application layout.

Layout:
  +------------------------------------------------------+
  |              TELEMETRY HEADER BAR                      |
  |  CPU: 34%  |  RAM: 4.2/8.0 GB  |  Stage: Scripting   |
  +------------+------------------+-----------------------+
  |  LEFT      |  CENTER          |  RIGHT                |
  |  Ingestion |  Kanban/State    |  QA & Preview Deck    |
  |  & Settings|  Table           |                       |
  |            |                  |  +-----------------+   |
  | [Drop Zone]|  Panel | State   |  | 16:9 Preview    |   |
  |            |  -----+--------  |  |                 |   |
  | API Base:  |  001  | Done     |  +-----------------+   |
  | [________] |  002  | Script   |                       |
  |            |  003  | Pending  |  Script:              |
  | HF Space:  |  ...  | ...      |  [Editable TextBox]   |
  | [________] |                  |                       |
  |            |                  |  [> Play] [Stop]      |
  | Mode:      |                  |  [Approve & Render]   |
  | (*) Auto   |                  |                       |
  | ( ) Manual |                  |                       |
  |            |                  |                       |
  | [> START]  |                  |                       |
  +------------+------------------+-----------------------+
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.workers import TelemetryWorker, PipelineWorker
from gui.components.telemetry_bar import TelemetryBar
from gui.components.ingestion_panel import IngestionPanel
from gui.components.state_table import StateTable
from gui.components.qa_deck import QADeck


class MainWindow(QMainWindow):
    """Primary application window for Xrexze ManhwaExplainerStudio."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xrexze - ManhwaExplainerStudio")
        self.setMinimumSize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Telemetry Header
        self.telemetry_bar = TelemetryBar()
        main_layout.addWidget(self.telemetry_bar)

        # Three-Panel Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.ingestion_panel = IngestionPanel()
        self.state_table = StateTable()
        self.qa_deck = QADeck()

        splitter.addWidget(self.ingestion_panel)
        splitter.addWidget(self.state_table)
        splitter.addWidget(self.qa_deck)
        splitter.setSizes([300, 400, 500])

        main_layout.addWidget(splitter)

        # Workers
        self.telemetry_worker = TelemetryWorker()
        self.telemetry_worker.telemetry_updated.connect(
            self.telemetry_bar.update_metrics
        )
        self.telemetry_worker.start()

        self.pipeline_worker: PipelineWorker | None = None
        self.ingestion_panel.start_requested.connect(self._start_pipeline)

    def _start_pipeline(self, page_paths: list, mode: str):
        """Launch the pipeline worker."""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            return

        self.pipeline_worker = PipelineWorker(page_paths, mode)
        self.pipeline_worker.panel_state_changed.connect(
            self.state_table.update_panel_state
        )
        self.pipeline_worker.script_ready.connect(
            self.qa_deck.show_script_for_review
        )
        self.pipeline_worker.audio_ready.connect(
            self.qa_deck.show_audio_for_review
        )
        self.pipeline_worker.chunk_rendered.connect(
            self.state_table.mark_chunk_done
        )
        self.pipeline_worker.progress_updated.connect(
            self.telemetry_bar.update_progress
        )
        self.pipeline_worker.error_occurred.connect(self._handle_error)
        self.pipeline_worker.pipeline_complete.connect(
            self._on_pipeline_complete
        )
        self.qa_deck.panel_approved.connect(
            self.pipeline_worker.approve_panel
        )
        self.pipeline_worker.start()

    def _handle_error(self, panel_id: int, message: str):
        self.state_table.update_panel_state(panel_id, "Failed")

    def _on_pipeline_complete(self, output_path: str):
        self.telemetry_bar.set_stage("Complete")

    def closeEvent(self, event):
        self.telemetry_worker.stop()
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.stop()
            self.pipeline_worker.wait(5000)
        event.accept()
```

### GUI Components

#### `gui/components/telemetry_bar.py`

```python
"""Telemetry header bar showing CPU, RAM, and pipeline stage."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel


class TelemetryBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet(
            "background-color: #1a1a2e; color: #e0e0e0; "
            "font-family: 'Consolas', monospace; font-size: 13px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)

        self.cpu_label = QLabel("CPU: --%")
        self.ram_label = QLabel("RAM: --/-- GB")
        self.stage_label = QLabel("Stage: Idle")
        self.progress_label = QLabel("Progress: 0/0")

        for lbl in [self.cpu_label, self.ram_label,
                     self.stage_label, self.progress_label]:
            lbl.setStyleSheet("padding: 0 12px;")
            layout.addWidget(lbl)
        layout.addStretch()

    def update_metrics(self, cpu: float, ram_used: float,
                       ram_total: float, stage: str):
        self.cpu_label.setText(f"CPU: {cpu:.0f}%")
        self.ram_label.setText(f"RAM: {ram_used:.1f}/{ram_total:.1f} GB")
        self.stage_label.setText(f"Stage: {stage}")

    def update_progress(self, current: int, total: int):
        self.progress_label.setText(f"Progress: {current}/{total}")

    def set_stage(self, stage: str):
        self.stage_label.setText(f"Stage: {stage}")
```

#### `gui/components/ingestion_panel.py`

```python
"""Left panel: drag-drop zone, API settings, and pipeline controls."""

from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QRadioButton, QButtonGroup, QFileDialog, QGroupBox,
)


class IngestionPanel(QWidget):
    start_requested = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page_paths: list[str] = []

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.drop_label = QLabel(
            "Drop manhwa pages folder here\nor click Browse"
        )
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #555; padding: 30px; "
            "border-radius: 8px; color: #aaa; font-size: 13px;"
        )
        layout.addWidget(self.drop_label)

        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.clicked.connect(self._browse_folder)
        layout.addWidget(self.browse_btn)

        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        api_layout.addWidget(QLabel("VLM Base URL:"))
        self.vlm_url_input = QLineEdit()
        self.vlm_url_input.setPlaceholderText("https://aihubmix.com/v1")
        api_layout.addWidget(self.vlm_url_input)
        api_layout.addWidget(QLabel("VLM API Key(s):"))
        self.vlm_key_input = QLineEdit()
        self.vlm_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.vlm_key_input.setPlaceholderText("sk-key1,sk-key2")
        api_layout.addWidget(self.vlm_key_input)
        api_layout.addWidget(QLabel("HF Space ID:"))
        self.hf_space_input = QLineEdit()
        self.hf_space_input.setPlaceholderText("username/omnivoice")
        api_layout.addWidget(self.hf_space_input)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        mode_group = QGroupBox("Pipeline Mode")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()
        self.auto_radio = QRadioButton("Full Auto-Pilot")
        self.manual_radio = QRadioButton("Manual QA Mode")
        self.auto_radio.setChecked(True)
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.manual_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self.start_btn = QPushButton("START PIPELINE")
        self.start_btn.setStyleSheet(
            "background-color: #16a085; color: white; "
            "font-size: 16px; font-weight: bold; padding: 12px; "
            "border-radius: 6px;"
        )
        self.start_btn.clicked.connect(self._start_clicked)
        layout.addWidget(self.start_btn)
        layout.addStretch()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Manhwa Pages Folder"
        )
        if folder:
            folder_path = Path(folder)
            self._page_paths = sorted([
                str(p) for p in folder_path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            self.drop_label.setText(
                f"{folder_path.name}\n{len(self._page_paths)} pages loaded"
            )

    def _start_clicked(self):
        if not self._page_paths:
            self.drop_label.setText(
                "No pages loaded! Browse a folder first."
            )
            return
        mode = "auto" if self.auto_radio.isChecked() else "manual_qa"
        self.start_requested.emit(self._page_paths, mode)
```

#### `gui/components/state_table.py`

```python
"""Center panel: Kanban-style panel state tracker."""

from pathlib import Path

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView,
)


STATE_COLORS = {
    "Pending": "#7f8c8d",
    "Scripting": "#f39c12",
    "Audio Synthesis": "#3498db",
    "Rendering": "#9b59b6",
    "Done": "#27ae60",
    "Failed": "#e74c3c",
}


class StateTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        header = QLabel("Panel Pipeline Status")
        header.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(header)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Panel", "State", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self._panel_rows: dict[int, int] = {}

    def update_panel_state(self, panel_id: int, state: str):
        if panel_id not in self._panel_rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._panel_rows[panel_id] = row
            self.table.setItem(
                row, 0, QTableWidgetItem(f"Panel {panel_id:04d}")
            )

        row = self._panel_rows[panel_id]
        state_item = QTableWidgetItem(state)
        color = STATE_COLORS.get(state, "#ffffff")
        state_item.setForeground(QColor(color))
        self.table.setItem(row, 1, state_item)
        self.table.scrollToBottom()

    def mark_chunk_done(self, panel_id: int, video_path: str):
        if panel_id in self._panel_rows:
            row = self._panel_rows[panel_id]
            self.table.setItem(
                row, 2,
                QTableWidgetItem(f"Done: {Path(video_path).name}"),
            )
```

#### `gui/components/qa_deck.py`

```python
"""Right panel: QA preview deck with image, script editor, audio player."""

from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QHBoxLayout,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput


class QADeck(QWidget):
    panel_approved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        header = QLabel("QA & Preview Deck")
        header.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(header)

        self.preview_label = QLabel("No panel selected")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet(
            "background-color: #111; border-radius: 8px; color: #555;"
        )
        layout.addWidget(self.preview_label)

        layout.addWidget(QLabel("Narration Script:"))
        self.script_editor = QPlainTextEdit()
        self.script_editor.setPlaceholderText(
            "Generated narration will appear here for editing..."
        )
        self.script_editor.setMaximumHeight(150)
        layout.addWidget(self.script_editor)

        audio_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play Audio")
        self.stop_btn = QPushButton("Stop")
        self.play_btn.clicked.connect(self._play_audio)
        self.stop_btn.clicked.connect(self._stop_audio)
        audio_layout.addWidget(self.play_btn)
        audio_layout.addWidget(self.stop_btn)
        layout.addLayout(audio_layout)

        self.audio_status = QLabel("No audio loaded")
        self.audio_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.audio_status)

        self.approve_btn = QPushButton("Approve & Render")
        self.approve_btn.setStyleSheet(
            "background-color: #2980b9; color: white; "
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "border-radius: 6px;"
        )
        self.approve_btn.clicked.connect(self._approve_clicked)
        self.approve_btn.setEnabled(False)
        layout.addWidget(self.approve_btn)
        layout.addStretch()

        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._current_audio_path: str | None = None

    def show_script_for_review(
        self, panel_id: int, narration: str, panel_image_path: str
    ):
        pixmap = QPixmap(panel_image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
        self.script_editor.setPlainText(narration)
        self.approve_btn.setEnabled(False)

    def show_audio_for_review(
        self, panel_id: int, audio_path: str, duration: float
    ):
        self._current_audio_path = audio_path
        self._player.setSource(QUrl.fromLocalFile(audio_path))
        self.audio_status.setText(
            f"Audio: {Path(audio_path).name} ({duration:.1f}s)"
        )
        self.approve_btn.setEnabled(True)

    def _play_audio(self):
        if self._current_audio_path:
            self._player.play()

    def _stop_audio(self):
        self._player.stop()

    def _approve_clicked(self):
        edited_text = self.script_editor.toPlainText().strip()
        self.panel_approved.emit(edited_text if edited_text else "")
        self.approve_btn.setEnabled(False)
```

---

## 10. Edge Cases & Failure Recovery

### 10.1 Rate Limiting (HTTP 429)

Implemented in `vision_client.py`:
1. Round-robin key rotation on every request.
2. On 429: rotate to next key + exponential backoff with jitter.
3. Configurable max retries (default: 3).
4. After exhausting retries: mark panel as FAILED, continue to next panel.

### 10.2 Crash Recovery (Manifest-Based Resume)

The pipeline writes a JSON manifest after every successfully completed panel:

```json
{
  "last_completed_panel_id": 42,
  "total_panels": 150,
  "chunk_paths": [
    "temp/chunks/chunk_0001.mp4",
    "temp/chunks/chunk_0002.mp4"
  ]
}
```

**Resume logic:**
1. On startup, check if `temp/manifest.json` exists.
2. If yes, read `last_completed_panel_id`.
3. Skip all panels with `id <= last_completed_panel_id` (verify chunk files exist).
4. Continue processing from `last_completed_panel_id + 1`.
5. No re-slicing needed — panel PNGs persist in `temp/panels/`.

### 10.3 Memory Cleanup Protocol

Every pipeline stage follows this memory discipline:

```python
# After every panel completes ALL stages:

# 1. Delete base64 strings from vision_client
del img_b64, payload

# 2. Close PIL Image objects from compositor
frame.close()
del frame

# 3. Delete numpy arrays from slicer
del img_color, img_gray, variances

# 4. Force garbage collection
import gc
gc.collect()

# 5. Check RAM headroom (safety valve)
import psutil
mem = psutil.virtual_memory()
if mem.percent > 85:
    logger.warning(f"RAM at {mem.percent}% - forcing aggressive cleanup")
    gc.collect()
```

### 10.4 FFmpeg Not Found

Checked at application startup in `main.py`:

```python
import shutil

def verify_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg not found in PATH. Install FFmpeg:\n"
            "  Windows: winget install FFmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )
    if shutil.which("ffprobe") is None:
        raise EnvironmentError("ffprobe not found. Install FFmpeg.")
```

### 10.5 Hugging Face Space Cold Starts

- Free CPU Spaces sleep after ~15 min of inactivity.
- First request after sleep takes 60-180 seconds.
- `VoiceClient._ensure_connected()` retries with exponential backoff.
- GUI shows "Waiting for OmniVoice Space to wake up..." in stage label.

### 10.6 Corrupted/Tiny Panel Detection

In `slicer.py`, images smaller than 200x200 pixels are discarded. Failed `cv2.imread` calls return an empty list and log the error.

### 10.7 Empty/Garbage VLM Narration

In the pipeline worker, narrations shorter than 10 characters trigger one automatic retry. If the retry also fails, a placeholder string is inserted that the user can manually edit in QA mode.

---

## 11. Testing & Verification Matrix

### 11.1 Unit Test Structure

```text
tests/
├── test_slicer.py           # Test gutter detection with synthetic images
├── test_vision_client.py    # Mock API responses, test retry logic
├── test_voice_client.py     # Mock Gradio client, test WAV handling
├── test_compositor.py       # Test canvas math, verify 1920x1080 output
├── test_stitcher.py         # Test concat file generation
└── conftest.py              # Shared fixtures
```

### 11.2 Three-Panel Smoke Test

```bash
# Create a test directory with 1 manhwa page (containing ~3 panels)
mkdir test_input
# Copy a single manhwa page image into test_input/

# Run the pipeline
python main.py

# Verify:
# 1. temp/panels/ contains 2-4 panel PNGs
# 2. temp/audio/ contains matching .wav files
# 3. temp/chunks/ contains matching .mp4 files
# 4. output/final_output.mp4 exists and is playable
# 5. Peak RAM during execution stayed under 6 GB
```

### 11.3 Memory Verification Script

```python
# utils/memcheck.py
import psutil
import time
import csv
from pathlib import Path


def monitor_memory(
    output_csv: Path, duration_sec: int = 300, interval_sec: int = 2
):
    """Log memory usage to CSV for analysis."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "ram_used_gb", "ram_percent", "cpu_percent"
        ])

        start = time.time()
        while time.time() - start < duration_sec:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            writer.writerow([
                f"{time.time() - start:.1f}",
                f"{mem.used / (1024**3):.2f}",
                f"{mem.percent:.1f}",
                f"{cpu:.1f}",
            ])
            time.sleep(interval_sec)


if __name__ == "__main__":
    monitor_memory(Path("memory_log.csv"))
```

---

## `main.py` — Application Entry Point

```python
"""
Xrexze — ManhwaExplainerStudio
Automated Manhwa/Manga narrative explainer video generator.

Made by ACL Community
https://github.com/ekanshbfoe/Xrexze
"""

import sys
import shutil
from pathlib import Path

from dotenv import load_dotenv


def verify_dependencies():
    """Verify all required system dependencies are available."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: FFmpeg not found in PATH.")
        print(
            "Install: winget install FFmpeg (Windows) "
            "/ sudo apt install ffmpeg (Linux)"
        )
        sys.exit(1)
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not found. Install FFmpeg.")
        sys.exit(1)
    print("[OK] FFmpeg found")


def main():
    """Application entry point."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded .env from {env_path}")
    else:
        print(
            "WARNING: No .env file found. "
            "Copy .env.example to .env and configure it."
        )

    verify_dependencies()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Xrexze - ManhwaExplainerStudio")
    app.setOrganizationName("ACL Community")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    app.setStyleSheet("""
        QWidget {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        QGroupBox {
            border: 1px solid #30363d;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 16px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QLineEdit {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px;
            color: #c9d1d9;
        }
        QLineEdit:focus {
            border-color: #58a6ff;
        }
        QPushButton {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px 12px;
            color: #c9d1d9;
        }
        QPushButton:hover {
            background-color: #30363d;
        }
        QPlainTextEdit {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px;
            color: #c9d1d9;
        }
        QTableWidget {
            background-color: #0d1117;
            gridline-color: #21262d;
        }
        QHeaderView::section {
            background-color: #161b22;
            border: 1px solid #21262d;
            padding: 4px;
            font-weight: bold;
        }
        QRadioButton {
            spacing: 8px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

---

## Utility Modules

### `utils/logger.py`

```python
"""Xrexze Logger — Rotating file + console logger."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        _LOG_DIR / "xrexze.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s",
    ))
    logger.addHandler(file_handler)

    return logger
```

### `utils/system_monitor.py`

```python
"""Xrexze System Monitor — psutil wrappers for telemetry."""

import psutil


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=0)


def get_ram_usage() -> tuple[float, float, float]:
    mem = psutil.virtual_memory()
    return (
        mem.used / (1024 ** 3),
        mem.total / (1024 ** 3),
        mem.percent,
    )


def is_ram_safe(threshold_gb: float = 6.0) -> bool:
    used_gb, _, _ = get_ram_usage()
    return used_gb < threshold_gb


def get_system_summary() -> dict:
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    return {
        "cpu_cores": cpu_count,
        "cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "ram_available_gb": round(mem.available / (1024 ** 3), 1),
        "ram_percent": mem.percent,
    }
```

---

*End of Implementation Specification.*
*Xrexze — Made by ACL Community — [github.com/ekanshbfoe/Xrexze](https://github.com/ekanshbfoe/Xrexze)*
