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
