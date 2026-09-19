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
