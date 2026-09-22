"""Left panel: folder browser, Colab URL, and pipeline controls."""

from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QRadioButton, QButtonGroup, QFileDialog, QGroupBox,
    QCheckBox,
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import shutil

from core.workspace import load_project_config, save_project_config
from core.pdf_ingester import PdfIngesterWorker


class IngestionPanel(QWidget):
    start_requested = pyqtSignal(list, str)

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self._project_root = project_root
        self._page_paths: list[str] = []
        self._is_server_ready: bool = False
        
        self.setAcceptDrops(True)

        self._health_poll_timer = QTimer(self)
        self._health_poll_timer.setInterval(3000)
        self._health_poll_timer.timeout.connect(self._poll_health)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Folder Browser ───────────────────
        self.drop_label = QLabel(
            "Drop manhwa pages folder or PDF here\nor click Browse"
        )
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #555; padding: 30px; "
            "border-radius: 8px; color: #aaa; font-size: 13px;"
        )
        layout.addWidget(self.drop_label)

        btn_layout = QVBoxLayout()
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.clicked.connect(self._browse_folder)
        btn_layout.addWidget(self.browse_btn)
        
        self.browse_pdf_btn = QPushButton("📄 Browse PDF")
        self.browse_pdf_btn.clicked.connect(self._browse_pdf)
        btn_layout.addWidget(self.browse_pdf_btn)
        layout.addLayout(btn_layout)

        # ── Colab Backend URL ────────────────
        colab_group = QGroupBox("Colab Backend")
        colab_layout = QVBoxLayout()
        
        # Notebook URL for auto-boot
        colab_layout.addWidget(QLabel("Colab Notebook URL:"))
        self.notebook_url_input = QLineEdit()
        self.notebook_url_input.setPlaceholderText(
            "https://colab.research.google.com/drive/..."
        )
        colab_layout.addWidget(self.notebook_url_input)

        # Start Server button
        self.start_server_btn = QPushButton("🚀 Start Server")
        self.start_server_btn.clicked.connect(self._start_server_clicked)
        colab_layout.addWidget(self.start_server_btn)

        # Tunnel URL (auto-filled or manual)
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
        mode_group = QGroupBox("Pipeline Settings")
        mode_layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.auto_radio = QRadioButton("Full Auto-Pilot")
        self.manual_radio = QRadioButton("Manual QA Mode")
        self.auto_radio.setChecked(True)
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.manual_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        
        # Memory Toggle
        self.memory_checkbox = QCheckBox("🧠 Enable 3-Pillar Memory Engine")
        config = load_project_config(self._project_root)
        self.memory_checkbox.setChecked(config.get("enable_memory", True))
        self.memory_checkbox.stateChanged.connect(self._on_memory_toggled)
        mode_layout.addWidget(self.memory_checkbox)
        
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

    def _browse_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Manhwa PDF", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self._process_pdf(Path(file_path))

    def _on_memory_toggled(self, state):
        config = load_project_config(self._project_root)
        config["enable_memory"] = bool(state)
        save_project_config(self._project_root, config)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".pdf"):
                event.acceptProposedAction()
                return
            elif urls and Path(urls[0].toLocalFile()).is_dir():
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
            
        path = Path(urls[0].toLocalFile())
        if path.is_file() and path.suffix.lower() == ".pdf":
            self._process_pdf(path)
        elif path.is_dir():
            self._page_paths = sorted([
                str(p) for p in path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            self.drop_label.setText(
                f"{path.name}\n{len(self._page_paths)} pages loaded"
            )

    def _process_pdf(self, pdf_path: Path):
        self.start_btn.setEnabled(False)
        self.drop_label.setText(f"Copying {pdf_path.name}...")
        
        dest = self._project_root / "1_Source_Files" / pdf_path.name
        shutil.copy2(pdf_path, dest)
        
        output_dir = self._project_root / "2_Workspace" / "pages"
        
        self._pdf_worker = PdfIngesterWorker(dest, output_dir)
        self._pdf_worker.progress.connect(
            lambda c, t: self.drop_label.setText(f"Converting PDF: page {c}/{t}...")
        )
        self._pdf_worker.finished.connect(self._on_pdf_finished)
        self._pdf_worker.error.connect(
            lambda e: self.drop_label.setText(f"Error: {e}")
        )
        self._pdf_worker.start()

    def _on_pdf_finished(self, paths: list[str]):
        self._page_paths = sorted(paths)
        self.drop_label.setText(f"PDF processed!\n{len(self._page_paths)} pages extracted.")
        self.start_btn.setEnabled(True)

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

    def _start_server_clicked(self):
        if hasattr(self, "bridge_worker") and self.bridge_worker.isRunning():
            self.bridge_worker.stop()
            self.stop_health_polling()
            self.start_server_btn.setText("🚀 Start Server")
            self.start_server_btn.setStyleSheet("")
            self.health_status.setText("Server stopped.")
            self.colab_url_input.clear()
            self._is_server_ready = False
            return

        notebook_url = self.notebook_url_input.text().strip()
        if not notebook_url:
            self.health_status.setText("Enter a Colab Notebook URL first")
            self.health_status.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return

        from core.browser_bridge import ColabBridgeWorker
        
        self.start_server_btn.setText("⏳ Booting...")
        self.start_server_btn.setStyleSheet("background-color: #888; color: white;")
        self.start_server_btn.setEnabled(False)
        
        self.bridge_worker = ColabBridgeWorker(notebook_url)
        self.bridge_worker.status_changed.connect(self._on_bridge_status)
        self.bridge_worker.url_extracted.connect(self._on_bridge_url)
        self.bridge_worker.error_occurred.connect(self._on_bridge_error)
        self.bridge_worker.start()

    def _on_bridge_status(self, msg: str):
        self.health_status.setText(msg)
        self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")

    def _on_bridge_url(self, url: str):
        self.colab_url_input.setText(url)
        self.health_status.setText("⏳ Connected to Colab... polling for model load status")
        self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")
        
        self.start_server_btn.setText("⬛ Stop Server")
        self.start_server_btn.setStyleSheet("background-color: #c0392b; color: white;")
        self.start_server_btn.setEnabled(True)
        
        from config.settings import get_settings
        get_settings().colab_base_url = url.rstrip("/")
        
        self._is_server_ready = False
        self._health_poll_timer.start()

    def _poll_health(self):
        url = self.colab_url_input.text().strip()
        if not url:
            return
            
        try:
            import requests
            resp = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                progress = data.get("progress", "Unknown status")
                
                if status == "ready":
                    self.health_status.setText(
                        f"✓ Server Ready — GPU: {data.get('gpu', '?')}, "
                        f"VRAM free: {data.get('vram_free_gb', '?')} GB"
                    )
                    self.health_status.setStyleSheet("color: #27ae60; font-size: 11px;")
                    
                    self._is_server_ready = True
                    self._health_poll_timer.stop()
                else:
                    self.health_status.setText(f"⏳ {progress}")
                    self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")
            elif resp.status_code == 502 or resp.status_code == 503:
                self.health_status.setText(f"⏳ Server booting (HTTP {resp.status_code})...")
                self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")
            else:
                 self.health_status.setText(f"✗ Error: HTTP {resp.status_code}")
                 self.health_status.setStyleSheet("color: #e74c3c; font-size: 11px;")
        except Exception as e:
            self.health_status.setText(f"⏳ Waiting for server to come online... ({str(e)[:30]})")
            self.health_status.setStyleSheet("color: #f39c12; font-size: 11px;")

    def stop_health_polling(self):
        self._health_poll_timer.stop()
        
    def _on_bridge_error(self, err: str):
        self.health_status.setText(f"✗ {err}")
        self.health_status.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self.start_server_btn.setText("🚀 Start Server")
        self.start_server_btn.setStyleSheet("")
        self.start_server_btn.setEnabled(True)

    def _start_clicked(self):
        if not self._page_paths:
            self.drop_label.setText(
                "No pages loaded! Browse a folder first."
            )
            return

        if not self._is_server_ready:
            self.health_status.setText("✗ Wait for server to become ready!")
            self.health_status.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return

        # Save Colab URL from GUI input
        url = self.colab_url_input.text().strip()
        if url:
            from config.settings import get_settings
            get_settings().colab_base_url = url.rstrip("/")

        mode = "auto" if self.auto_radio.isChecked() else "manual_qa"
        self.start_requested.emit(self._page_paths, mode)
