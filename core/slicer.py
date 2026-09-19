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
