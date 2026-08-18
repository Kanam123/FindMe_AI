from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
DB_DIR = PROJECT_DIR / "db"
DB_PATH = DB_DIR / "UserDetails.db"
DATASET_DIR = BASE_DIR / "dataSet"
RECOGNIZER_DIR = BASE_DIR / "recognizer"
RECOGNIZER_PATH = RECOGNIZER_DIR / "trainningData.yml"
UPLOAD_DIR = BASE_DIR / "upfiles"
MODEL_DIR = BASE_DIR / "models"
VECTOR_INDEX_DIR = Path(os.environ.get("VECTOR_INDEX_PATH", BASE_DIR / "ai_index"))
FACE_DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "findme-ai-local-dev-secret")
FACE_SIMILARITY_THRESHOLD = float(os.environ.get("FACE_SIMILARITY_THRESHOLD", "0.42"))
FACE_WEIGHT = float(os.environ.get("FACE_WEIGHT", "0.65"))
TEXT_WEIGHT = float(os.environ.get("TEXT_WEIGHT", "0.35"))
TOP_K = int(os.environ.get("TOP_K", "5"))

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

FACE_EMBEDDING_VERSION = "opencv-sface-2021dec"
TEXT_EMBEDDING_VERSION = "hashing-tfidf-v1"
