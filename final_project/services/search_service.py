from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from config import settings
from models.search_result import SearchResult
from repositories.person_repository import PersonRepository
from services.embedding_service import FaceEmbeddingService, TextEmbeddingService
from services.lbph_service import LBPHBaselineService
from services.vector_search_service import VectorSearchService


class SearchService:
    def __init__(
        self,
        repository: PersonRepository,
        face_embedding_service: FaceEmbeddingService,
        text_embedding_service: TextEmbeddingService,
        lbph_service: LBPHBaselineService,
    ):
        self.repository = repository
        self.face_embeddings = face_embedding_service
        self.text_embeddings = text_embedding_service
        self.lbph = lbph_service
        self.vector_search = VectorSearchService()

    def rebuild_index(self) -> dict[str, Any]:
        people = self.repository.all_people()
        current_ids = {str(person["ID"]) for person in people}
        self.remove_stale_embeddings(current_ids)
        face_built = 0
        text_built = 0
        face_errors: list[str] = []

        for person in people:
            person_id = str(person["ID"])
            image_path = self.resolve_profile_image(person)
            if image_path and self.face_embeddings.available():
                try:
                    embedding = self.face_embeddings.embed_image(image_path)
                    path = self.face_embeddings.save_embedding(person_id, embedding)
                    self.repository.update_embedding_paths(
                        person_id,
                        face_path=self.persisted_index_path(path),
                        face_version=settings.FACE_EMBEDDING_VERSION,
                    )
                    face_built += 1
                except Exception as exc:
                    face_errors.append(f"{person_id}: {exc}")

            profile_text = self.text_embeddings.profile_text(person)
            if profile_text.strip():
                embedding = self.text_embeddings.embed_text(profile_text)
                path = self.text_embeddings.save_embedding(person_id, embedding)
                self.repository.update_embedding_paths(
                    person_id,
                    text_path=self.persisted_index_path(path),
                    text_version=settings.TEXT_EMBEDDING_VERSION,
                )
                text_built += 1

        if face_built and text_built:
            status = "ready"
        elif face_built or text_built:
            status = "partial"
        else:
            status = "needs_data"

        return {
            "profiles": len(people),
            "face_embeddings_built": face_built,
            "text_embeddings_built": text_built,
            "face_model_available": self.face_embeddings.available(),
            "face_errors": face_errors[:10],
            "status": status,
        }

    def remove_stale_embeddings(self, current_ids: set[str]) -> None:
        for folder in [settings.VECTOR_INDEX_DIR / "face", settings.VECTOR_INDEX_DIR / "text"]:
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.glob("*.npy"):
                if path.stem not in current_ids:
                    path.unlink(missing_ok=True)

    def persisted_index_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(settings.BASE_DIR))
        except ValueError:
            return str(path)

    def face_search(self, image_path: Path, top_k: int = settings.TOP_K) -> tuple[list[SearchResult], dict[str, Any]]:
        started = time.perf_counter()
        method = "deep-face-vector"
        try:
            if not self.face_embeddings.available():
                person, confidence = self.lbph.search(image_path)
                duration_ms = (time.perf_counter() - started) * 1000
                results = []
                if person:
                    score = max(0.0, 1.0 - min(confidence or 90, 90) / 90)
                    results = [SearchResult(person=person, score=score, face_similarity=score, method="baseline-lbph")]
                self.repository.log_search("face", duration_ms, len(results), "baseline")
                return results, {
                    "method": "baseline-lbph",
                    "message": "Using baseline recognition because deep face model files are not installed.",
                    "duration_ms": round(duration_ms, 2),
                }

            query = self.face_embeddings.embed_image(image_path)
            vectors = []
            for person in self.repository.all_people():
                embedding_path = person.get("FaceEmbeddingPath")
                if not embedding_path:
                    continue
                embedding = self.face_embeddings.load_embedding(embedding_path)
                if embedding is not None:
                    vectors.append((str(person["ID"]), embedding))
            matches = self.vector_search.search(query, vectors, top_k)
            results = [
                SearchResult(
                    person=self.repository.get_person(person_id) or {},
                    score=similarity,
                    face_similarity=similarity,
                    method=method,
                )
                for person_id, similarity in matches
                if similarity >= settings.FACE_SIMILARITY_THRESHOLD
            ]
            duration_ms = (time.perf_counter() - started) * 1000
            self.repository.log_search("face", duration_ms, len(results), "ok")
            return results, {
                "method": method,
                "threshold": settings.FACE_SIMILARITY_THRESHOLD,
                "duration_ms": round(duration_ms, 2),
            }
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            self.repository.log_search("face", duration_ms, 0, "error")
            raise

    def text_search(self, query: str, top_k: int = settings.TOP_K) -> tuple[list[SearchResult], dict[str, Any]]:
        started = time.perf_counter()
        query_embedding = self.text_embeddings.embed_text(query)
        vectors = []
        for person in self.repository.all_people():
            embedding_path = person.get("TextEmbeddingPath")
            if not embedding_path:
                profile_text = self.text_embeddings.profile_text(person)
                if not profile_text.strip():
                    continue
                embedding = self.text_embeddings.embed_text(profile_text)
            else:
                embedding = self.text_embeddings.load_embedding(embedding_path)
            if embedding is not None:
                vectors.append((str(person["ID"]), embedding))
        matches = self.vector_search.search(query_embedding, vectors, top_k)
        results = [
            SearchResult(
                person=self.repository.get_person(person_id) or {},
                score=similarity,
                text_similarity=similarity,
                method="local-text-vector",
            )
            for person_id, similarity in matches
            if similarity > 0
        ]
        duration_ms = (time.perf_counter() - started) * 1000
        self.repository.log_search("text", duration_ms, len(results), "ok")
        return results, {
            "method": "local-text-vector",
            "duration_ms": round(duration_ms, 2),
        }

    def hybrid_search(
        self,
        image_path: Path | None,
        query: str,
        top_k: int = settings.TOP_K,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        started = time.perf_counter()
        face_results, face_meta = ([], {"method": "none"})
        text_results, text_meta = ([], {"method": "none"})
        if image_path is not None:
            face_results, face_meta = self.face_search(image_path, top_k=top_k * 2)
        if query.strip():
            text_results, text_meta = self.text_search(query, top_k=top_k * 2)

        combined: dict[str, SearchResult] = {}
        for result in face_results:
            person_id = str(result.person.get("ID"))
            combined[person_id] = SearchResult(
                person=result.person,
                score=settings.FACE_WEIGHT * (result.face_similarity or 0.0),
                face_similarity=result.face_similarity,
                method="hybrid",
            )
        for result in text_results:
            person_id = str(result.person.get("ID"))
            if person_id not in combined:
                combined[person_id] = SearchResult(person=result.person, score=0.0, method="hybrid")
            combined[person_id].text_similarity = result.text_similarity
            combined[person_id].score += settings.TEXT_WEIGHT * (result.text_similarity or 0.0)

        results = sorted(combined.values(), key=lambda item: item.score, reverse=True)[:top_k]
        duration_ms = (time.perf_counter() - started) * 1000
        self.repository.log_search("hybrid", duration_ms, len(results), "ok")
        return results, {
            "method": "hybrid-ranking",
            "face_weight": settings.FACE_WEIGHT,
            "text_weight": settings.TEXT_WEIGHT,
            "face_method": face_meta.get("method"),
            "text_method": text_meta.get("method"),
            "duration_ms": round(duration_ms, 2),
        }

    def index_status(self) -> dict[str, Any]:
        stats = self.repository.dashboard_stats()
        stats.update(self.face_embeddings.model_status())
        stats["text_model"] = "Local hashed profile vectors"
        stats["text_embedding_dimensions"] = self.text_embeddings.dimensions
        stats["face_threshold"] = settings.FACE_SIMILARITY_THRESHOLD
        stats["face_weight"] = settings.FACE_WEIGHT
        stats["text_weight"] = settings.TEXT_WEIGHT
        return stats

    def resolve_profile_image(self, person: dict[str, Any]) -> Path | None:
        image_path = person.get("ImagePath")
        if image_path:
            target = Path(image_path)
            if not target.is_absolute():
                target = settings.BASE_DIR / target
            if target.exists():
                return target
        person_id = str(person["ID"])
        first_sample = next(iter(sorted(settings.DATASET_DIR.glob(f"User.{person_id}.*.jpg"))), None)
        return first_sample
