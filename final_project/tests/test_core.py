from __future__ import annotations

import gc
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

import app
from config import settings
from repositories.person_repository import PersonRepository
from services.embedding_service import TextEmbeddingService
from services.search_service import SearchService
from services.upload_service import UploadService
from services.vector_search_service import VectorSearchService


class FindMeCoreTests(unittest.TestCase):
    def test_cosine_similarity_ranks_top_k(self) -> None:
        service = VectorSearchService()
        query = np.array([1.0, 0.0], dtype="float32")
        matches = service.search(
            query,
            [
                ("far", np.array([0.0, 1.0], dtype="float32")),
                ("near", np.array([0.9, 0.1], dtype="float32")),
            ],
            top_k=1,
        )
        self.assertEqual(matches[0][0], "near")
        self.assertGreater(matches[0][1], 0.9)

    def test_text_embedding_handles_semantic_expansion(self) -> None:
        service = TextEmbeddingService()
        query = service.embed_text("Python developer")
        profile = service.embed_text("Flask software engineer")
        self.assertGreater(VectorSearchService.cosine_similarity(query, profile), 0.0)

    def test_upload_rejects_fake_image_content(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            original_upload_dir = settings.UPLOAD_DIR
            settings.UPLOAD_DIR = Path(temp_dir)
            try:
                fake = FileStorage(stream=io.BytesIO(b"not an image"), filename="face.jpg")
                with self.assertRaises(ValueError):
                    UploadService().save_image(fake)
            finally:
                settings.UPLOAD_DIR = original_upload_dir

    def test_upload_accepts_real_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_upload_dir = settings.UPLOAD_DIR
            settings.UPLOAD_DIR = Path(temp_dir)
            try:
                buffer = io.BytesIO()
                Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
                buffer.seek(0)
                upload = FileStorage(stream=buffer, filename="face.png")
                saved = UploadService().save_image(upload)
                self.assertTrue(saved.exists())
            finally:
                settings.UPLOAD_DIR = original_upload_dir

    def test_profile_crud_and_text_search_with_temp_database(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            original_index_dir = settings.VECTOR_INDEX_DIR
            settings.VECTOR_INDEX_DIR = Path(temp_dir) / "ai_index"
            try:
                repository = PersonRepository(Path(temp_dir) / "people.db")
                repository.init_db()
                repository.upsert_person(
                    {
                        "ID": "1001",
                        "Name": "Test Person",
                        "Age": "25",
                        "Gender": "Other",
                        "CN": "0000000000",
                        "Address": "Bangalore",
                        "Profession": "Software Developer",
                        "Skills": "Python, Flask, Machine Learning",
                    }
                )
                search = SearchService(repository, app.face_embedding_service, TextEmbeddingService(), app.lbph_service)
                report = search.rebuild_index()
                results, meta = search.text_search("Python developer in Bangalore")
                self.assertEqual(report["text_embeddings_built"], 1)
                self.assertEqual(meta["method"], "local-text-vector")
                self.assertEqual(results[0].person["ID"], "1001")
                gc.collect()
            finally:
                settings.VECTOR_INDEX_DIR = original_index_dir

    def test_api_validation_errors_are_json(self) -> None:
        client = app.app.test_client()
        empty_text = client.post("/api/search/text", json={"query": ""})
        empty_hybrid = client.post("/api/search/hybrid", data={})
        self.assertEqual(empty_text.status_code, 400)
        self.assertEqual(empty_hybrid.status_code, 400)
        self.assertIn("error", empty_text.get_json())
        self.assertIn("error", empty_hybrid.get_json())

    def test_media_route_does_not_expose_source_files(self) -> None:
        client = app.app.test_client()
        response = client.get("/media/app.py")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
