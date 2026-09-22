"""
Xrexze GUI Workers — QThread-based non-blocking pipeline execution.

Two worker threads:
1. TelemetryWorker: polls psutil every 1s, emits system metrics.
2. PipelineWorker: runs the full A->B->C->D->E pipeline, emitting
   signals at each state transition so the GUI updates in real-time.
"""

from __future__ import annotations

import threading
import gc
import json
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


def _parse_entity_block(narration: str) -> tuple[str, list[dict]]:
    """Extract <<<JSON>>>...<<<END>>> block, return (clean_narration, entities)."""
    import json, re
    match = re.search(r"<<<JSON>>>(.*?)<<<END>>>", narration, re.DOTALL)
    if not match:
        return narration, []
    try:
        data = json.loads(match.group(1).strip())
        entities = data.get("characters", [])
    except (json.JSONDecodeError, KeyError):
        entities = []
    clean = narration[:match.start()].strip()
    return clean, entities


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
        project_root: Path | None = None,
        chapter_id: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self._page_paths = page_paths
        self._mode = mode
        self._project_root = project_root
        self._chapter_id = chapter_id
        
        if self._project_root:
            from core.workspace import load_project_config
            config = load_project_config(self._project_root)
            self._enable_memory = config.get("enable_memory", True)
        else:
            self._enable_memory = False
        self._running = True
        self._script_approved = threading.Event()
        self._render_approved = threading.Event()
        self._edited_script: str | None = None

    def approve_script(self, edited_script: str | None = None):
        """Called from GUI when user approves the script."""
        self._edited_script = edited_script
        self._script_approved.set()

    def approve_render(self):
        """Called from GUI when user approves the render."""
        self._render_approved.set()

    def stop(self):
        self._running = False
        self._script_approved.set()
        self._render_approved.set()

    def run(self):
        """Execute the full pipeline: Slice -> Script -> Audio -> Render -> Stitch."""
        from core.slicer import slice_chapter
        from core.colab_client import ColabClient
        from core.compositor import render_video_chunk
        from core.stitcher import stitch_final_video
        from config.settings import get_settings

        settings = get_settings()
        
        if self._project_root:
            workspace_dir = self._project_root / "2_Workspace"
            panels_dir = workspace_dir / "panels"
            audio_dir = workspace_dir / "audio"
            chunks_dir = workspace_dir / "chunks"
            manifest_path = workspace_dir / "manifest.json"
            temp_dir = workspace_dir
        else:
            temp_dir = settings.temp_dir
            panels_dir = temp_dir / "panels"
            audio_dir = temp_dir / "audio"
            chunks_dir = temp_dir / "chunks"
            manifest_path = temp_dir / "manifest.json"

        # Initialize Memory Engine
        rag, kg, summarizer = None, None, None
        if self._enable_memory and self._project_root:
            from core.memory.rag_engine import RAGEngine
            from core.memory.knowledge_graph import KnowledgeGraph
            from core.memory.summarizer import RollingSummarizer
            memory_dir = self._project_root / "memory"
            rag = RAGEngine(memory_dir)
            kg = KnowledgeGraph(memory_dir)
            summarizer = RollingSummarizer(memory_dir)

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

        # Initialize unified Colab client
        colab_url = settings.colab_base_url
        if not colab_url:
            self.error_occurred.emit(
                0, "No Colab URL configured. Set COLAB_BASE_URL in .env "
                "or paste it in the GUI."
            )
            return

        colab = ColabClient(base_url=colab_url)

        # Health check
        health = colab.health_check()
        if health.get("status") != "ready":
            self.error_occurred.emit(
                0, f"Colab backend not ready: {health}"
            )
            return

        chunk_paths: list[Path] = []
        prev_context = ""
        all_narrations: list[str] = []

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

            # Phase 2: Script (via Colab VLM)
            self.panel_state_changed.emit(pid, "Scripting")
            
            memory_block = ""
            if self._enable_memory and self._project_root and rag and kg and summarizer:
                rag_context = ""
                if prev_context:
                    try:
                        query_vec = colab.embed(prev_context)
                        similar = rag.query(query_vec, top_k=3)
                        rag_context = "\n".join(similar)
                    except Exception:
                        pass

                active_facts = kg.get_active_facts(self._chapter_id)
                core_lore = summarizer.get_core_lore()

                if core_lore:
                    memory_block += f"\n## Story So Far\n{core_lore}\n"
                if active_facts:
                    memory_block += f"\n## Active Character States\n{active_facts}\n"
                if rag_context:
                    memory_block += f"\n## Similar Past Narrations\n{rag_context}\n"
            
            full_context = f"{prev_context}\n{memory_block}".strip()

            try:
                narration = colab.generate_script(
                    image_path=panel_path,
                    context=full_context,
                )
            except RuntimeError as e:
                self.error_occurred.emit(pid, str(e))
                self.panel_state_changed.emit(pid, "Failed")
                continue

            if not narration or len(narration.strip()) < 10:
                try:
                    narration = colab.generate_script(
                        image_path=panel_path, context=full_context
                    )
                except RuntimeError:
                    pass
                if not narration or len(narration.strip()) < 10:
                    narration = (
                        f"[Panel {pid}: narration generation failed"
                        " - manual edit required]"
                    )

            if self._enable_memory and self._project_root and rag and kg:
                narration, entities = _parse_entity_block(narration)
                if entities:
                    kg.update(self._chapter_id, pid, entities)
                try:
                    emb = colab.embed(narration)
                    rag.add(narration, emb)
                except Exception:
                    pass

            prev_context = narration[:100]
            all_narrations.append(narration)

            # Manual QA mode: pause for user approval
            if self._mode == "manual_qa":
                self.script_ready.emit(pid, narration, str(panel_path))
                self._script_approved.clear()
                self._script_approved.wait()
                if not self._running:
                    return
                if self._edited_script:
                    narration = self._edited_script
                    self._edited_script = None

            # Phase 3: Audio (via Colab TTS)
            self.panel_state_changed.emit(pid, "Audio Synthesis")
            audio_path = audio_dir / f"audio_{pid:04d}.mp3"
            try:
                colab.generate_voice(
                    text=narration,
                    output_path=audio_path,
                )
            except RuntimeError as e:
                self.error_occurred.emit(pid, str(e))
                self.panel_state_changed.emit(pid, "Failed")
                continue

            if self._mode == "manual_qa":
                self.audio_ready.emit(pid, str(audio_path), 0.0)
                self._render_approved.clear()
                self._render_approved.wait()
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

        # Phase 5: Memory Summarizer & Stitch
        if self._enable_memory and self._project_root and summarizer:
            chapter_text = "\n".join(all_narrations)
            summarizer.add_chapter(self._chapter_id, chapter_text)
            if summarizer.should_compress():
                summarizer.compress(colab)

        if chunk_paths:
            if self._project_root:
                output_path = self._project_root / "3_Final_Exports" / f"chapter_{self._chapter_id:04d}_output.mp4"
            else:
                output_path = settings.output_dir / "final_output.mp4"
                
            try:
                stitch_final_video(chunk_paths, output_path, temp_dir)
                self.pipeline_complete.emit(str(output_path))
            except RuntimeError as e:
                self.error_occurred.emit(0, f"Stitching failed: {e}")

        gc.collect()

