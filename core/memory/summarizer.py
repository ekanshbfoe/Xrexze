"""
Pillar C: Recursive Summarizer.
Compresses old chapters to maintain a concise core lore block.
"""

import json
from pathlib import Path

class RollingSummarizer:
    def __init__(self, memory_dir: Path, window: int = 3):
        self._memory_dir = memory_dir
        self._window = window
        self._chapters_file = memory_dir / "chapter_texts.json"
        self._lore_file = memory_dir / "core_lore.txt"
        self._chapters: dict[int, str] = {}
        if self._chapters_file.exists():
            self._chapters = json.loads(self._chapters_file.read_text())

    def add_chapter(self, chapter_id: int, text: str):
        self._chapters[str(chapter_id)] = text
        self._chapters_file.write_text(
            json.dumps(self._chapters, ensure_ascii=False, indent=2)
        )

    def should_compress(self) -> bool:
        return len(self._chapters) > self._window

    def compress(self, colab_client) -> str:
        """Send oldest chapters to /api/summarize, store result, prune."""
        sorted_ids = sorted(self._chapters.keys(), key=int)
        oldest = sorted_ids[:len(sorted_ids) - self._window]
        text_to_compress = "\n\n---\n\n".join(
            f"Chapter {cid}:\n{self._chapters[cid]}" for cid in oldest
        )
        summary = colab_client.summarize(text_to_compress)
        # Prepend to existing lore
        existing = self._lore_file.read_text() if self._lore_file.exists() else ""
        self._lore_file.write_text(f"{summary}\n\n{existing}".strip())
        # Prune compressed chapters
        for cid in oldest:
            del self._chapters[cid]
        self._chapters_file.write_text(
            json.dumps(self._chapters, ensure_ascii=False, indent=2)
        )
        return summary

    def get_core_lore(self) -> str:
        return self._lore_file.read_text() if self._lore_file.exists() else ""
