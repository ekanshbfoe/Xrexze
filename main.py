"""
Xrexze — ManhwaExplainerStudio
Automated Manhwa/Manga narrative explainer video generator.

Made by ACL Community
https://github.com/ekanshbfoe/Xrexze
"""

import sys
import shutil
from pathlib import Path

from dotenv import load_dotenv


def verify_dependencies():
    """Verify all required system dependencies are available."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: FFmpeg not found in PATH.")
        print(
            "Install: winget install FFmpeg (Windows) "
            "/ sudo apt install ffmpeg (Linux)"
        )
        sys.exit(1)
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not found. Install FFmpeg.")
        sys.exit(1)
    print("[OK] FFmpeg found")


def main():
    """Application entry point."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded .env from {env_path}")
    else:
        print(
            "WARNING: No .env file found. "
            "Copy .env.example to .env and configure it."
        )

    verify_dependencies()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Xrexze - ManhwaExplainerStudio")
    app.setOrganizationName("ACL Community")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    app.setStyleSheet("""
        QWidget {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        QGroupBox {
            border: 1px solid #30363d;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 16px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QLineEdit {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px;
            color: #c9d1d9;
        }
        QLineEdit:focus {
            border-color: #58a6ff;
        }
        QPushButton {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px 12px;
            color: #c9d1d9;
        }
        QPushButton:hover {
            background-color: #30363d;
        }
        QPlainTextEdit {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 6px;
            color: #c9d1d9;
        }
        QTableWidget {
            background-color: #0d1117;
            gridline-color: #21262d;
        }
        QHeaderView::section {
            background-color: #161b22;
            border: 1px solid #21262d;
            padding: 4px;
            font-weight: bold;
        }
        QRadioButton {
            spacing: 8px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
