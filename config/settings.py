"""
Xrexze Configuration — Pydantic Settings loader.

Reads from .env file at project root and validates all required
configuration values at startup. Exposes a singleton `settings` object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # ── Colab Backend (Primary) ──────────────
    colab_base_url: str = Field(
        default="", description="Public URL of the Colab FastAPI backend"
    )

    # ── Legacy VLM API (kept for backward compat) ─
    vlm_api_base_url: str = Field(
        default="", description="Base URL for OpenAI-compatible VLM endpoint"
    )
    vlm_api_keys: str = Field(
        default="", description="Comma-separated API keys for rotation"
    )
    vlm_model_name: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    vlm_timeout_sec: int = Field(default=120)

    # ── Legacy OmniVoice (kept for backward compat) ─
    hf_omnivoice_space_id: str = Field(
        default="", description="Gradio Space ID: username/space-name"
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
        if not self.vlm_api_keys:
            return []
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
