"""
Pillar A: Retrieval-Augmented Generation (RAG) Engine.
Uses local FAISS index with BAAI/bge-m3 1024-dim embeddings.
"""

import faiss
import numpy as np
import json
from pathlib import Path

class RAGEngine:
    DIMENSION = 1024  # BAAI/bge-m3 dense vector dimension

    def __init__(self, memory_dir: Path):
        self._memory_dir = memory_dir
        self._index_path = memory_dir / "faiss.index"
        self._texts_path = memory_dir / "rag_texts.json"
        self._texts: list[str] = []

        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._texts = json.loads(self._texts_path.read_text())
        else:
            self._index = faiss.IndexFlatIP(self.DIMENSION)

    def add(self, text: str, embedding: list[float]):
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._texts.append(text)
        self.save()

    def query(self, embedding: list[float], top_k: int = 3) -> list[str]:
        if self._index.ntotal == 0:
            return []
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        return [self._texts[i] for i in indices[0] if i < len(self._texts)]

    def save(self):
        faiss.write_index(self._index, str(self._index_path))
        self._texts_path.write_text(json.dumps(self._texts, ensure_ascii=False))
