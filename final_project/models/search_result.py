from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    person: dict[str, Any]
    score: float
    face_similarity: float | None = None
    text_similarity: float | None = None
    method: str = "vector"
