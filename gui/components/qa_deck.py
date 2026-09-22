"""Right panel: QA preview deck with image, script editor, audio player."""

from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QHBoxLayout,
)
import pygame


class QADeck(QWidget):
    script_approved = pyqtSignal(str)
    render_approved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        header = QLabel("QA & Preview Deck")
        header.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(header)

        self.preview_label = QLabel("No panel selected")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet(
            "background-color: #111; border-radius: 8px; color: #555;"
        )
        layout.addWidget(self.preview_label)

        layout.addWidget(QLabel("Narration Script:"))
        self.script_editor = QPlainTextEdit()
        self.script_editor.setPlaceholderText(
            "Generated narration will appear here for editing..."
        )
        self.script_editor.setMaximumHeight(150)
        layout.addWidget(self.script_editor)

        # Script Action
        self.accept_script_btn = QPushButton("Accept Script → Generate Audio")
        self.accept_script_btn.setStyleSheet(
            "background-color: #f39c12; color: white; font-weight: bold;"
        )
        self.accept_script_btn.clicked.connect(self._accept_script_clicked)
        self.accept_script_btn.setEnabled(False)
        layout.addWidget(self.accept_script_btn)

        # Audio Actions
        audio_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play Audio")
        self.stop_btn = QPushButton("Stop")
        self.play_btn.clicked.connect(self._play_audio)
        self.stop_btn.clicked.connect(self._stop_audio)
        audio_layout.addWidget(self.play_btn)
        audio_layout.addWidget(self.stop_btn)
        layout.addLayout(audio_layout)

        self.audio_status = QLabel("No audio loaded")
        self.audio_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.audio_status)

        # Render Action
        self.approve_render_btn = QPushButton("Approve & Render")
        self.approve_render_btn.setStyleSheet(
            "background-color: #2980b9; color: white; "
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "border-radius: 6px;"
        )
        self.approve_render_btn.clicked.connect(self._approve_render_clicked)
        self.approve_render_btn.setEnabled(False)
        layout.addWidget(self.approve_render_btn)
        layout.addStretch()

        pygame.mixer.init()
        self._current_audio_path: str | None = None

    def show_script_for_review(
        self, panel_id: int, narration: str, panel_image_path: str
    ):
        pixmap = QPixmap(panel_image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
        self.script_editor.setPlainText(narration)
        self.accept_script_btn.setEnabled(True)
        self.approve_render_btn.setEnabled(False)

    def show_audio_for_review(
        self, panel_id: int, audio_path: str, duration: float
    ):
        self._current_audio_path = audio_path
        self.audio_status.setText(
            f"Audio: {Path(audio_path).name} ({duration:.1f}s)"
        )
        self.approve_render_btn.setEnabled(True)

    def _play_audio(self):
        if self._current_audio_path:
            try:
                sound = pygame.mixer.Sound(self._current_audio_path)
                sound.play()
            except Exception as e:
                self.audio_status.setText(f"Error playing audio: {e}")

    def _stop_audio(self):
        pygame.mixer.stop()

    def _accept_script_clicked(self):
        edited_text = self.script_editor.toPlainText().strip()
        self.script_approved.emit(edited_text if edited_text else "")
        self.accept_script_btn.setEnabled(False)

    def _approve_render_clicked(self):
        self.render_approved.emit()
        self.approve_render_btn.setEnabled(False)
