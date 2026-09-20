"""Left panel: folder browser, Colab URL, and pipeline controls."""

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

        # ── Folder Browser ───────────────────
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

        # ── Colab Backend URL ────────────────
        colab_group = QGroupBox("Colab Backend")
        colab_layout = QVBoxLayout()
        colab_layout.addWidget(QLabel("Colab Tunnel URL:"))
        self.colab_url_input = QLineEdit()
        self.colab_url_input.setPlaceholderText(
            "https://abc-123.trycloudflare.com"
        )
        colab_layout.addWidget(self.colab_url_input)

        self.health_btn = QPushButton("Test Connection")
        self.health_btn.clicked.connect(self._test_connection)
        colab_layout.addWidget(self.health_btn)

        self.health_status = QLabel("Not connected")
        self.health_status.setStyleSheet("color: #888; font-size: 11px;")
        colab_layout.addWidget(self.health_status)

        colab_group.setLayout(colab_layout)
        layout.addWidget(colab_group)

        # ── Pipeline Mode ────────────────────
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

        # ── Start Button ─────────────────────
        self.start_btn = QPushButton("▶ START PIPELINE")
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

    def _test_connection(self):
        """Test the Colab backend connection."""
        url = self.colab_url_input.text().strip()
        if not url:
            self.health_status.setText("Enter a Colab URL first")
            self.health_status.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return

        self.health_status.setText("Testing...")
        self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")

        try:
            import requests
            resp = requests.get(f"{url.rstrip('/')}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.health_status.setText(
                    f"✓ Connected — GPU: {data.get('gpu', '?')}, "
                    f"VRAM free: {data.get('vram_free_gb', '?')} GB"
                )
                self.health_status.setStyleSheet(
                    "color: #27ae60; font-size: 11px;"
                )

                # Save URL to settings
                from config.settings import get_settings
                settings = get_settings()
                settings.colab_base_url = url.rstrip("/")
            else:
                self.health_status.setText(
                    f"✗ Error: HTTP {resp.status_code}"
                )
                self.health_status.setStyleSheet(
                    "color: #e74c3c; font-size: 11px;"
                )
        except Exception as e:
            self.health_status.setText(f"✗ {str(e)[:60]}")
            self.health_status.setStyleSheet(
                "color: #e74c3c; font-size: 11px;"
            )

    def _start_clicked(self):
        if not self._page_paths:
            self.drop_label.setText(
                "No pages loaded! Browse a folder first."
            )
            return

        # Save Colab URL from GUI input
        url = self.colab_url_input.text().strip()
        if url:
            from config.settings import get_settings
            get_settings().colab_base_url = url.rstrip("/")

        mode = "auto" if self.auto_radio.isChecked() else "manual_qa"
        self.start_requested.emit(self._page_paths, mode)
