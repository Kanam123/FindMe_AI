from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from config import settings
from utils.cv_loader import load_cv2


cv2 = load_cv2()


class FaceEmbeddingService:
    dimensions = 128

    def available(self) -> bool:
        return (
            settings.FACE_DETECTOR_MODEL.exists()
            and settings.FACE_RECOGNIZER_MODEL.exists()
            and hasattr(cv2, "FaceDetectorYN_create")
            and hasattr(cv2, "FaceRecognizerSF_create")
        )

    def model_status(self) -> dict[str, str | bool | int]:
        return {
            "available": self.available(),
            "model": "OpenCV SFace",
            "embedding_dimensions": self.dimensions,
            "detector_model": str(settings.FACE_DETECTOR_MODEL),
            "recognizer_model": str(settings.FACE_RECOGNIZER_MODEL),
            "version": settings.FACE_EMBEDDING_VERSION,
        }

    def embed_image(self, image_path: Path) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Deep face embedding model is not available. Using baseline recognition.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("The uploaded image could not be read.")

        detector = cv2.FaceDetectorYN_create(str(settings.FACE_DETECTOR_MODEL), "", (320, 320))
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        recognizer = cv2.FaceRecognizerSF_create(str(settings.FACE_RECOGNIZER_MODEL), "")

        if faces is not None and len(faces) > 0:
            face = max(faces, key=lambda item: item[2] * item[3])
            aligned = recognizer.alignCrop(image, face)
        else:
            aligned = cv2.resize(image, (112, 112))

        feature = recognizer.feature(aligned).flatten().astype("float32")
        return self.normalize(feature)

    def embedding_path(self, person_id: str) -> Path:
        return settings.VECTOR_INDEX_DIR / "face" / f"{person_id}.npy"

    def save_embedding(self, person_id: str, embedding: np.ndarray) -> Path:
        path = self.embedding_path(person_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embedding.astype("float32"))
        return path

    def load_embedding(self, path: str | Path) -> np.ndarray | None:
        target = Path(path)
        if not target.is_absolute():
            target = settings.BASE_DIR / target
        if not target.exists():
            return None
        return self.normalize(np.load(target).astype("float32"))

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class TextEmbeddingService:
    dimensions = 384

    def profile_text(self, person: dict) -> str:
        fields = [
            person.get("Name", ""),
            person.get("Profession", ""),
            person.get("Skills", ""),
            person.get("Education", ""),
            person.get("Experience", ""),
            person.get("Projects", ""),
            person.get("Certifications", ""),
            person.get("Address", ""),
            person.get("Bio", ""),
        ]
        return " ".join(str(field) for field in fields if field)

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        tokens = self.expand_tokens(self.tokenize(text))
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return self.normalize(vector)

    def embedding_path(self, person_id: str) -> Path:
        return settings.VECTOR_INDEX_DIR / "text" / f"{person_id}.npy"

    def save_embedding(self, person_id: str, embedding: np.ndarray) -> Path:
        path = self.embedding_path(person_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embedding.astype("float32"))
        return path

    def load_embedding(self, path: str | Path) -> np.ndarray | None:
        target = Path(path)
        if not target.is_absolute():
            target = settings.BASE_DIR / target
        if not target.exists():
            return None
        return self.normalize(np.load(target).astype("float32"))

    def tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9+#.]+", text.lower()) if len(token) > 1]

    def expand_tokens(self, tokens: Iterable[str]) -> list[str]:
        synonyms = {
            "developer": ["software", "programming", "engineer"],
            "engineer": ["developer", "software"],
            "backend": ["api", "server", "database"],
            "frontend": ["ui", "javascript", "web"],
            "ml": ["machine", "learning", "ai"],
            "ai": ["machine", "learning", "artificial", "intelligence"],
            "python": ["django", "flask", "automation"],
            "java": ["spring", "backend"],
            "bangalore": ["bengaluru"],
            "bengaluru": ["bangalore"],
        }
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            expanded.extend(synonyms.get(token, []))
        return expanded

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm == 0:
            return vector
        return vector / norm
