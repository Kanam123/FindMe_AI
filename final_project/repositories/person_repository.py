from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings


PROFILE_COLUMNS = {
    "PersonType": "TEXT DEFAULT 'Indian'",
    "ImagePath": "TEXT",
    "Profession": "TEXT",
    "Skills": "TEXT",
    "Education": "TEXT",
    "Experience": "TEXT",
    "Projects": "TEXT",
    "Certifications": "TEXT",
    "Bio": "TEXT",
    "FaceEmbeddingPath": "TEXT",
    "TextEmbeddingPath": "TEXT",
    "EmbeddingVersion": "TEXT",
    "TextEmbeddingVersion": "TEXT",
    "CreatedAt": "TEXT",
    "UpdatedAt": "TEXT",
}


class PersonRepository:
    def __init__(self, db_path: Path = settings.DB_PATH):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        settings.DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        for path in [
            settings.DATASET_DIR,
            settings.RECOGNIZER_DIR,
            settings.UPLOAD_DIR,
            settings.MODEL_DIR,
            settings.VECTOR_INDEX_DIR,
            settings.VECTOR_INDEX_DIR / "face",
            settings.VECTOR_INDEX_DIR / "text",
        ]:
            path.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS People (
                    ID TEXT PRIMARY KEY,
                    Name TEXT NOT NULL,
                    Age TEXT NOT NULL,
                    Gender TEXT NOT NULL,
                    CN TEXT NOT NULL,
                    Address TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS SearchLogs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_type TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    candidates INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(People)")}
            for column, column_type in PROFILE_COLUMNS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE People ADD COLUMN {column} {column_type}")
            conn.commit()

    def all_people(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM People ORDER BY Name COLLATE NOCASE").fetchall()
        return [dict(row) for row in rows]

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM People WHERE ID = ?", (person_id,)).fetchone()
        return dict(row) if row else None

    def upsert_person(self, data: dict[str, Any]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        payload = {
            "ID": data.get("ID", "").strip(),
            "Name": data.get("Name", "").strip(),
            "Age": str(data.get("Age", "")).strip(),
            "Gender": data.get("Gender", "").strip(),
            "CN": data.get("CN", "").strip(),
            "Address": data.get("Address", "").strip(),
            "PersonType": data.get("PersonType", "Authorized").strip() or "Authorized",
            "ImagePath": data.get("ImagePath"),
            "Profession": data.get("Profession", "").strip(),
            "Skills": data.get("Skills", "").strip(),
            "Education": data.get("Education", "").strip(),
            "Experience": data.get("Experience", "").strip(),
            "Projects": data.get("Projects", "").strip(),
            "Certifications": data.get("Certifications", "").strip(),
            "Bio": data.get("Bio", "").strip(),
            "CreatedAt": data.get("CreatedAt") or now,
            "UpdatedAt": now,
        }
        if not payload["ID"] or not payload["Name"]:
            raise ValueError("ID and name are required.")

        columns = list(payload.keys())
        placeholders = ", ".join(["?"] * len(columns))
        updates = ", ".join([f"{column}=excluded.{column}" for column in columns if column not in {"ID", "CreatedAt"}])
        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO People ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(ID) DO UPDATE SET {updates}
                """,
                [payload[column] for column in columns],
            )
            conn.commit()

    def update_embedding_paths(
        self,
        person_id: str,
        face_path: str | None = None,
        text_path: str | None = None,
        face_version: str | None = None,
        text_version: str | None = None,
    ) -> None:
        updates: list[str] = []
        values: list[Any] = []
        if face_path is not None:
            updates.append("FaceEmbeddingPath = ?")
            values.append(face_path)
        if text_path is not None:
            updates.append("TextEmbeddingPath = ?")
            values.append(text_path)
        if face_version is not None:
            updates.append("EmbeddingVersion = ?")
            values.append(face_version)
        if text_version is not None:
            updates.append("TextEmbeddingVersion = ?")
            values.append(text_version)
        if not updates:
            return
        updates.append("UpdatedAt = ?")
        values.append(datetime.now().isoformat(timespec="seconds"))
        values.append(person_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE People SET {', '.join(updates)} WHERE ID = ?", values)
            conn.commit()

    def delete_person(self, person_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM People WHERE ID = ?", (person_id,))
            conn.commit()
            return cursor.rowcount > 0

    def verify_user_password(self, username: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT password FROM Users WHERE username = ?", (username,)).fetchone()
        return row["password"] if row else None

    def create_user(self, username: str, password_hash: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO Users (username, password) VALUES (?, ?)",
                (username, password_hash),
            )
            conn.commit()

    def log_search(self, search_type: str, duration_ms: float, candidates: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO SearchLogs (search_type, duration_ms, candidates, status)
                VALUES (?, ?, ?, ?)
                """,
                (search_type, duration_ms, candidates, status),
            )
            conn.commit()

    def dashboard_stats(self) -> dict[str, Any]:
        people = self.all_people()
        face_count = sum(1 for person in people if person.get("FaceEmbeddingPath"))
        text_count = sum(1 for person in people if person.get("TextEmbeddingPath"))
        with self.connect() as conn:
            searches = conn.execute("SELECT COUNT(*) FROM SearchLogs").fetchone()[0]
        return {
            "profiles": len(people),
            "face_embeddings": face_count,
            "text_embeddings": text_count,
            "missing_embeddings": max(len(people) - min(face_count, text_count), 0),
            "searches": searches,
            "index_ready": bool(people) and face_count > 0 and text_count > 0,
        }
