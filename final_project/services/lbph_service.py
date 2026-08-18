from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from repositories.person_repository import PersonRepository
from utils.cv_loader import load_cv2


cv2 = load_cv2()


class LBPHBaselineService:
    def __init__(self, repository: PersonRepository):
        self.repository = repository

    def detect_gray_faces(self, image_path: Path) -> list[np.ndarray]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("The uploaded image could not be read.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        return [gray[y : y + h, x : x + w] for (x, y, w, h) in faces]

    def create_face_samples(self, person_id: str, image_path: Path) -> int:
        faces = self.detect_gray_faces(image_path)
        if not faces:
            raise ValueError("No clear face was detected in the uploaded image.")

        settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        for old_sample in settings.DATASET_DIR.glob(f"User.{person_id}.*.jpg"):
            old_sample.unlink()

        sample_count = 0
        for face in faces:
            resized = cv2.resize(face, (220, 220))
            for _ in range(12):
                sample_count += 1
                cv2.imwrite(str(settings.DATASET_DIR / f"User.{person_id}.{sample_count}.jpg"), resized)
        return sample_count

    def train(self) -> int:
        image_paths = sorted(settings.DATASET_DIR.glob("User.*.*.jpg"))
        faces: list[np.ndarray] = []
        ids: list[int] = []

        for image_path in image_paths:
            parts = image_path.name.split(".")
            if len(parts) < 4 or not parts[1].isdigit():
                continue
            face_img = Image.open(image_path).convert("L")
            faces.append(np.array(face_img, dtype="uint8"))
            ids.append(int(parts[1]))

        if not faces:
            raise ValueError("No LBPH training images are available yet.")

        settings.RECOGNIZER_DIR.mkdir(parents=True, exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(str(settings.RECOGNIZER_PATH))
        return len(set(ids))

    def search(self, image_path: Path) -> tuple[dict | None, float | None]:
        if not settings.RECOGNIZER_PATH.exists():
            self.train()
        faces = self.detect_gray_faces(image_path)
        if not faces:
            raise ValueError("No clear face was detected in the search image.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(settings.RECOGNIZER_PATH))
        best_id: int | None = None
        best_confidence: float | None = None
        for face in faces:
            resized = cv2.resize(face, (220, 220))
            predicted_id, confidence = recognizer.predict(resized)
            if best_confidence is None or confidence < best_confidence:
                best_id = predicted_id
                best_confidence = float(confidence)

        if best_id is None or best_confidence is None or best_confidence > 90:
            return None, best_confidence
        return self.repository.get_person(str(best_id)), best_confidence
