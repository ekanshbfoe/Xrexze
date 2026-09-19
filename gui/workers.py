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
