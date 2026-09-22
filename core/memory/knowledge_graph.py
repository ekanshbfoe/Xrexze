"""
Pillar B: Knowledge Graph Engine.
Uses SQLite to store character state, locations, and skills.
"""

import sqlite3
import json
from pathlib import Path

class KnowledgeGraph:
    def __init__(self, memory_dir: Path):
        self._db_path = memory_dir / "entities.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("""CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter INTEGER, panel_id INTEGER,
            name TEXT, status TEXT, location TEXT, skills TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self._conn.commit()

    def update(self, chapter: int, panel_id: int, entities: list[dict]):
        for ent in entities:
            self._conn.execute(
                "INSERT INTO entities (chapter, panel_id, name, status, location, skills) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chapter, panel_id, ent.get("name", ""),
                 ent.get("status", ""), ent.get("location", ""),
                 json.dumps(ent.get("skills", []), ensure_ascii=False))
            )
        self._conn.commit()

    def get_active_facts(self, chapter: int) -> str:
        """Returns formatted string of the latest state per character."""
        cursor = self._conn.execute("""
            SELECT name, status, location, skills FROM entities
            WHERE id IN (
                SELECT MAX(id) FROM entities WHERE chapter <= ? GROUP BY name
            )
        """, (chapter,))
        lines = []
        for name, status, location, skills in cursor:
            lines.append(f"- {name}: {status}, at {location}, skills: {skills}")
        return "\n".join(lines) if lines else ""

    def close(self):
        self._conn.close()
