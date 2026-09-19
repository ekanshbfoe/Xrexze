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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{system_prompt}\n\n"
                                "---\n"
                                "Narrate this manhwa panel. Follow your system "
                                "instructions exactly. Return ONLY the narration."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_b64}",
                            },
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
                        f"Server error ({response.status_code}). Response: {response.text}\n"
                        f"Retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    print(f"\n[VLM ERROR] API returned status {response.status_code}")
                    print(f"Raw Response: {response.text}\n")
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
