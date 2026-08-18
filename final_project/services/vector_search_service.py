from __future__ import annotations

import numpy as np


class VectorSearchService:
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def search(self, query: np.ndarray, vectors: list[tuple[str, np.ndarray]], top_k: int) -> list[tuple[str, float]]:
        scored = [
            (person_id, self.cosine_similarity(query, vector))
            for person_id, vector in vectors
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
