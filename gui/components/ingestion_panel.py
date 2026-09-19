"""Left panel: drag-drop zone, API settings, and pipeline controls."""

from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QRadioButton, QButtonGroup, QFileDialog, QGroupBox,
)


class IngestionPanel(QWidget):
    start_requested = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page_paths: list[str] = []

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.drop_label = QLabel(
            "Drop manhwa pages folder here\nor click Browse"
        )
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #555; padding: 30px; "
            "border-radius: 8px; color: #aaa; font-size: 13px;"
        )
        layout.addWidget(self.drop_label)

        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.clicked.connect(self._browse_folder)
        layout.addWidget(self.browse_btn)

        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        api_layout.addWidget(QLabel("VLM Base URL:"))
        self.vlm_url_input = QLineEdit()
        self.vlm_url_input.setPlaceholderText("https://aihubmix.com/v1")
        api_layout.addWidget(self.vlm_url_input)
        api_layout.addWidget(QLabel("VLM API Key(s):"))
        self.vlm_key_input = QLineEdit()
        self.vlm_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.vlm_key_input.setPlaceholderText("sk-key1,sk-key2")
        api_layout.addWidget(self.vlm_key_input)
        api_layout.addWidget(QLabel("HF Space ID:"))
        self.hf_space_input = QLineEdit()
        self.hf_space_input.setPlaceholderText("username/omnivoice")
        api_layout.addWidget(self.hf_space_input)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        mode_group = QGroupBox("Pipeline Mode")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()
        self.auto_radio = QRadioButton("Full Auto-Pilot")
        self.manual_radio = QRadioButton("Manual QA Mode")
        self.auto_radio.setChecked(True)
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.manual_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self.start_btn = QPushButton("START PIPELINE")
        self.start_btn.setStyleSheet(
            "background-color: #16a085; color: white; "
            "font-size: 16px; font-weight: bold; padding: 12px; "
            "border-radius: 6px;"
        )
        self.start_btn.clicked.connect(self._start_clicked)
        layout.addWidget(self.start_btn)
        layout.addStretch()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Manhwa Pages Folder"
        )
        if folder:
            folder_path = Path(folder)
            self._page_paths = sorted([
                str(p) for p in folder_path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            self.drop_label.setText(
                f"{folder_path.name}\n{len(self._page_paths)} pages loaded"
            )

    def _start_clicked(self):
        if not self._page_paths:
            self.drop_label.setText(
                "No pages loaded! Browse a folder first."
            )
            return
        mode = "auto" if self.auto_radio.isChecked() else "manual_qa"
        self.start_requested.emit(self._page_paths, mode)
