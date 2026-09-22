"""
Xrexze PDF Ingester.

Uses PyMuPDF (fitz) in a background QThread to convert PDF pages
into high-resolution PNGs for the workspace.
"""

from pathlib import Path
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal

from utils.logger import get_logger

logger = get_logger(__name__)


class PdfIngesterWorker(QThread):
    """
    Background worker that extracts pages from a PDF and saves them as PNGs.
    """
    progress = pyqtSignal(int, int)   # (current_page, total_pages)
    finished = pyqtSignal(list)       # list of output PNG paths
    error = pyqtSignal(str)

    def __init__(self, pdf_path: Path, output_dir: Path):
        super().__init__()
        self._pdf_path = pdf_path
        self._output_dir = output_dir

    def run(self):
        logger.info(f"Starting PDF ingestion for {self._pdf_path.name}")
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(self._pdf_path))
            total = len(doc)
            paths: List[str] = []
            
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, page in enumerate(doc):
                # Use 2x zoom for high-quality rendering
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                out_path = self._output_dir / f"page_{i+1:04d}.png"
                pix.save(str(out_path))
                paths.append(str(out_path))
                
                self.progress.emit(i + 1, total)
                
            doc.close()
            logger.info(f"Successfully extracted {total} pages from PDF.")
            self.finished.emit(paths)
            
        except Exception as e:
            err_msg = f"Failed to ingest PDF: {e}"
            logger.error(err_msg)
            self.error.emit(err_msg)
