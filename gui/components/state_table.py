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
