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
