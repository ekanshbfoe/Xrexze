"""
Xrexze Colab Client — Unified bridge to the Colab GPU backend.

Replaces all legacy API clients (Groq, AIHubMix, OpenRouter, HF Serverless).
Communicates with a single FastAPI server running on Google Colab via
a Cloudflare tunnel URL.

Methods:
  - generate_script(image_path, context) -> str
  - generate_voice(text, output_path) -> Path
"""

from __future__ import annotations

import base64
import gc
import time
from pathlib import Path
from typing import Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


class ColabClient:
    """
    Client for the Xrexze Colab GPU backend.

    Sends panel images for VLM narration and text for TTS synthesis
    to a FastAPI server running on Google Colab with a T4 GPU.
    """

    def __init__(self, base_url: str, timeout: int = 120, max_retries: int = 3):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = 5

        logger.info(
            f"ColabClient initialized: base={self._base_url}, "
            f"timeout={self._timeout}s, retries={self._max_retries}"
        )

    def health_check(self) -> dict:
        """Check if the Colab backend is alive and ready."""
        try:
            resp = requests.get(
                f"{self._base_url}/health", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                f"Colab health: status={data.get('status')}, "
                f"gpu={data.get('gpu')}, "
                f"vram_free={data.get('vram_free_gb')}GB"
            )
            return data
        except Exception as e:
            logger.error(f"Colab health check failed: {e}")
            return {"status": "unreachable", "error": str(e)}

    def generate_script(
        self,
        image_path: Path,
        context: str = "",
        prompt: Optional[str] = None,
    ) -> str:
        """
        Send a panel image to the Colab VLM and receive narration text.

        Parameters
        ----------
        image_path : Path
            Path to the cropped panel PNG/JPG.
        context : str
            Previous panel summary for continuity.
        prompt : str, optional
            Override the default system prompt.

        Returns
        -------
        str
            The generated Hindi narration text.
        """
        # Encode image to base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image_base64": img_b64,
            "context": context,
        }
        if prompt:
            payload["prompt"] = prompt

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    f"Script request: panel={image_path.name}, "
                    f"attempt={attempt}/{self._max_retries}"
                )

                resp = requests.post(
                    f"{self._base_url}/api/script",
                    json=payload,
                    timeout=self._timeout,
                )

                if resp.status_code != 200:
                    error_detail = resp.text[:500]
                    logger.error(
                        f"Script API error ({resp.status_code}): {error_detail}"
                    )
                    print(f"\n[COLAB ERROR] Status {resp.status_code}: {error_detail}\n")

                    if resp.status_code == 503:
                        wait = self._retry_delay * attempt
                        logger.warning(f"Server busy, retrying in {wait}s")
                        time.sleep(wait)
                        continue

                    resp.raise_for_status()

                data = resp.json()
                narration = data.get("script", "").strip()

                logger.info(
                    f"Script success: {len(narration)} chars, "
                    f"panel={image_path.name}"
                )

                del img_b64
                gc.collect()
                return narration

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Script timeout after {self._timeout}s (attempt {attempt})"
                )
                last_error = TimeoutError(f"Timed out after {self._timeout}s")
            except requests.exceptions.ConnectionError as e:
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Connection error: {e}. Retrying in {wait}s"
                )
                last_error = e
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                logger.error(f"Script request error: {e}")
                last_error = e

        raise RuntimeError(
            f"Script generation failed after {self._max_retries} attempts "
            f"for panel {image_path.name}: {last_error}"
        )

    def generate_voice(
        self,
        text: str,
        output_path: Path,
        voice_id: str = "hi-IN-SwaraNeural",
        language: str = "hi",
    ) -> Path:
        """
        Send narration text to the Colab TTS and save the audio file.

        Parameters
        ----------
        text : str
            The Hindi narration text to synthesize.
        output_path : Path
            Where to save the resulting audio file.
        voice_id : str
            Voice identifier for edge-tts.
        language : str
            Language code.

        Returns
        -------
        Path
            The saved audio file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "text": text,
            "language": language,
            "voice_id": voice_id,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    f"Voice request: chars={len(text)}, "
                    f"attempt={attempt}/{self._max_retries}"
                )

                resp = requests.post(
                    f"{self._base_url}/api/voice",
                    json=payload,
                    timeout=self._timeout,
                )

                if resp.status_code != 200:
                    error_detail = resp.text[:500]
                    logger.error(
                        f"Voice API error ({resp.status_code}): {error_detail}"
                    )
                    print(f"\n[COLAB ERROR] Status {resp.status_code}: {error_detail}\n")

                    if resp.status_code == 503:
                        wait = self._retry_delay * attempt
                        time.sleep(wait)
                        continue

                    resp.raise_for_status()

                # Save audio response to disk
                with open(output_path, "wb") as f:
                    f.write(resp.content)

                file_size = output_path.stat().st_size
                logger.info(
                    f"Voice success: size={file_size / 1024:.0f}KB, "
                    f"file={output_path.name}"
                )

                return output_path

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Voice timeout after {self._timeout}s (attempt {attempt})"
                )
                last_error = TimeoutError(f"Timed out after {self._timeout}s")
            except requests.exceptions.ConnectionError as e:
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Connection error: {e}. Retrying in {wait}s"
                )
                last_error = e
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                logger.error(f"Voice request error: {e}")
                last_error = e

        raise RuntimeError(
            f"Voice synthesis failed after {self._max_retries} attempts: "
            f"{last_error}"
        )
