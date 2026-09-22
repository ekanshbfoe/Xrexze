"""Pre-Dashboard Workspace Launcher."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QInputDialog, QMessageBox
)

from core.workspace import create_workspace, list_projects


class ProjectLauncherWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xrexze — Select Project")
        self.resize(600, 400)
        
        # Set dark theme to match app
        self.setStyleSheet("""
            QDialog {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            QLabel {
                color: #c9d1d9;
                font-size: 14px;
            }
            QTableWidget {
                background-color: #161b22;
                color: #c9d1d9;
                gridline-color: #30363d;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #21262d;
                color: #c9d1d9;
                padding: 4px;
                border: 1px solid #30363d;
            }
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton#newBtn {
                background-color: #238636;
                color: white;
                border: 1px solid #2ea043;
            }
            QPushButton#newBtn:hover {
                background-color: #2ea043;
            }
        """)

        self.selected_project: Path | None = None
        # Default projects dir in the app root
        self._projects_dir = Path("./projects").resolve()
        self._projects_dir.mkdir(parents=True, exist_ok=True)

        layout = QVBoxLayout(self)

        header = QLabel("Recent Projects")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Projects Table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Project Name", "Last Modified"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

        # Buttons Layout
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("Open Folder...")
        self.open_btn.clicked.connect(self._on_open_clicked)
        btn_layout.addWidget(self.open_btn)
        
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.new_btn = QPushButton("＋ New Project")
        self.new_btn.setObjectName("newBtn")
        self.new_btn.clicked.connect(self._on_new_clicked)
        btn_layout.addWidget(self.new_btn)
        
        layout.addLayout(btn_layout)

        self._load_projects()

    def _load_projects(self):
        self.table.setRowCount(0)
        self._project_data = list_projects(self._projects_dir)
        
        for row, proj in enumerate(self._project_data):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(proj["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(proj["last_modified"]))

    def _on_double_click(self, row, col):
        if 0 <= row < len(self._project_data):
            self.selected_project = self._project_data[row]["path"]
            self.accept()

    def _on_open_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Xrexze Project Folder", str(self._projects_dir)
        )
        if folder:
            folder_path = Path(folder)
            if not (folder_path / "config.json").exists():
                QMessageBox.warning(
                    self, "Invalid Project", 
                    "The selected folder does not contain a config.json file."
                )
                return
            self.selected_project = folder_path
            self.accept()

    def _on_new_clicked(self):
        name, ok = QInputDialog.getText(
            self, "New Project", "Enter project name:"
        )
        if ok and name.strip():
            name = name.strip()
            # Basic sanitization
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-")
            target = self._projects_dir / safe_name
            
            if target.exists():
                QMessageBox.warning(self, "Error", f"Project '{safe_name}' already exists.")
                return
                
            self.selected_project = create_workspace(safe_name, self._projects_dir)
            self.accept()
