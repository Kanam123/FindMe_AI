# FINDME AI Interview Notes

## 1. Project Overview
- FindMe AI is a Flask application for searching an authorized local people dataset by face image, natural-language profile query, or both.
- It solves the problem of retrieving people profiles when the user may have an image, a description, or mixed evidence.
- 30-second interview explanation: "FindMe AI started as an OpenCV/LBPH face-recognition project and was upgraded into a multimodal people-search prototype. It stores consented profiles in SQLite, generates face and text embeddings, compares vectors with cosine similarity, ranks top matches, and shows profile intelligence through a Flask web UI."

## 2. Technology Stack
- Python: backend language.
- Flask: web app, routes, templates, JSON APIs, sessions.
- SQLite: local database at `../db/UserDetails.db`.
- OpenCV contrib: Haar Cascade, LBPH, YuNet, SFace.
- ONNX models: YuNet detector and SFace recognizer stored in `models/`.
- NumPy: embeddings, vector storage, cosine math.
- Pillow: image validation and LBPH training image loading.
- Werkzeug: password hashing and secure filenames.
- Jinja templates/CSS/JS: server-rendered UI.
- `unittest`: current test suite.

## 3. Current Architecture
Frontend templates and forms  
-> Flask routes in `app.py`  
-> route handlers validate input and call services  
-> services perform upload, embedding, LBPH, vector search, LLM-summary fallback  
-> `PersonRepository` reads/writes SQLite  
-> `.npy` embedding files are stored in `ai_index/`

AI components actually present:
- `FaceEmbeddingService`: OpenCV YuNet + SFace deep face embeddings.
- `TextEmbeddingService`: local hashed text embeddings with small synonym expansion.
- `VectorSearchService`: in-memory cosine similarity top-K search.
- `LBPHBaselineService`: Haar Cascade + LBPH fallback/baseline.
- `LLMService`: local grounded summary only; live LLM calls are disabled.

## 4. Project Structure
- `app.py`: creates Flask app, registers routes, wires repository/services, formats public API results.
- `config/settings.py`: paths, upload limits, thresholds, weights, model names, env vars.
- `models/search_result.py`: `SearchResult` dataclass for ranked results.
- `models/*.onnx`: YuNet face detector and SFace face recognizer.
- `repositories/person_repository.py`: SQLite schema creation/migration and profile/user/search-log queries.
- `services/embedding_service.py`: face and text embedding generation/load/save.
- `services/lbph_service.py`: Haar face crop creation, LBPH training, LBPH prediction.
- `services/search_service.py`: index rebuild, face search, text search, hybrid search, index status.
- `services/upload_service.py`: file type, size, secure filename, and image validation.
- `services/vector_search_service.py`: cosine similarity and sorted top-K search.
- `services/llm_service.py`: local summary fallback using stored profile fields.
- `utils/cv_loader.py`: imports `cv2`, with fallback to system site packages.
- `Templates/`: Jinja pages for home, search, people, profile, dashboard, auth, AI index.
- `static/css/`: app styling plus legacy CSS.
- `static/javascript/engine.js`: legacy mobile-menu helper.
- `tests/test_core.py`: focused tests for vector search, text vectors, upload validation, CRUD/search, API validation, media protection.

## 5. Face Recognition Pipeline
Current face search:
Image upload -> `UploadService.save_image()` -> `SearchService.face_search()` -> deep model if available, otherwise LBPH -> ranked profile results.

Original LBPH implementation:
- File: `services/lbph_service.py`.
- Uses Haar Cascade: `haarcascade_frontalface_default.xml`.
- Converts image to grayscale.
- Detects faces with `detectMultiScale`.
- Saves 12 resized `220x220` grayscale samples per person in `dataSet/User.<id>.<n>.jpg`.
- Trains `cv2.face.LBPHFaceRecognizer_create()` and saves `recognizer/trainningData.yml`.
- During search, LBPH predicts a numeric person ID and confidence distance.
- If confidence is over `90`, the match is rejected.

New deep-learning implementation:
- File: `services/embedding_service.py`.
- Model files:
  - `models/face_detection_yunet_2023mar.onnx`
  - `models/face_recognition_sface_2021dec.onnx`
- `FaceEmbeddingService.available()` checks model files and OpenCV APIs.
- `embed_image()` reads the image, detects faces with `cv2.FaceDetectorYN_create`, picks the largest face, aligns/crops with `FaceRecognizerSF.alignCrop`, extracts a 128-dimensional SFace embedding, and L2-normalizes it.
- If YuNet detects no face, code resizes the whole image to `112x112` and still tries SFace. This is implemented but can produce weaker embeddings.
- `SearchService.face_search()` loads stored face vectors and ranks them with cosine similarity. Results below `FACE_SIMILARITY_THRESHOLD` default `0.42` are filtered out.

## 6. Semantic Search
IMPLEMENTED.
- Text input comes from `/search` or `/api/search/text`.
- `TextEmbeddingService.profile_text()` combines stored fields: name, profession, skills, education, experience, projects, certifications, address, bio.
- `embed_text()` tokenizes text, expands a small hardcoded synonym map, hashes tokens into a 384-dimensional NumPy vector, and normalizes it.
- `SearchService.text_search()` embeds the query, loads or computes profile vectors, calls cosine top-K search, and returns positive-scoring profiles.
- This is local vector search, not a transformer model.

## 7. Hybrid Search
IMPLEMENTED.
- File: `services/search_service.py`, function `hybrid_search()`.
- If an image exists, it runs `face_search(top_k * 2)`.
- If query text exists, it runs `text_search(top_k * 2)`.
- Formula:
  - initial face candidate score = `FACE_WEIGHT * face_similarity`
  - text score added = `TEXT_WEIGHT * text_similarity`
  - defaults: `FACE_WEIGHT=0.65`, `TEXT_WEIGHT=0.35`
- Results are sorted descending and truncated to `TOP_K`.

## 8. Vector Search
- Technology: custom in-memory NumPy cosine similarity, not FAISS/Chroma/pgvector.
- Face vectors: 128D SFace `.npy` files in `ai_index/face/<person_id>.npy`.
- Text vectors: 384D hashed `.npy` files in `ai_index/text/<person_id>.npy`.
- Search loads vectors from paths stored in SQLite, computes cosine similarity, sorts, returns top-K.
- `SearchService.rebuild_index()` removes stale `.npy` files, rebuilds embeddings for every profile, stores file paths and version fields in SQLite.

## 9. LLM / RAG
PARTIALLY IMPLEMENTED.
- `LLMService` reads `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` from settings.
- `available()` returns true if `LLM_API_KEY` exists.
- However, `profile_summary()` never calls an external LLM. Even when credentials exist, it returns `available: False` and a local summary.
- The summary is grounded only in stored profile fields.
- RAG over retrieved search results is NOT IMPLEMENTED.
- AI agent/tool calling is NOT IMPLEMENTED.

## 10. Database
- Technology: SQLite.
- Path: `../db/UserDetails.db`.
- Tables observed/created:
  - `People`
  - `Users`
  - `SearchLogs`
  - legacy `login` table also exists in the DB.
- `People` important fields: `ID` primary key, `Name`, `Age`, `Gender`, `CN`, `Address`, `PersonType`, `ImagePath`, profile fields, `FaceEmbeddingPath`, `TextEmbeddingPath`, embedding versions, timestamps.
- `Users`: `username` primary key, hashed `password`, `created_at`.
- `SearchLogs`: search type, duration, candidate count, status, timestamp.
- There are no foreign keys in the current implementation.
- Embeddings are not stored as BLOBs; the DB stores file paths to `.npy` vectors.

## 11. Important APIs/Routes
- `GET /`: home page.
- `GET,POST /search` and `/verify`: main UI search. Inputs: mode, image file, query. Output: rendered search page.
- `POST /search_img`: legacy face search form. Input: image. Output: result page.
- `GET /collect`: add-profile form.
- `POST /submit_form`: creates/updates person, optionally saves image, creates LBPH samples, trains LBPH, rebuilds index, renders profile.
- `GET /people`: directory page.
- `GET /people/<person_id>`: profile page with local summary.
- `GET /dashboard`: real counts from repository.
- `GET /ai-index`: index status page.
- `POST /admin/rebuild-index`: rebuild index and render status.
- `GET,POST /signup`: local user creation.
- `GET,POST /login`: session login.
- `GET /logout`: clears session.
- `GET /media/<path>`: serves only upload, dataset, or static files.
- `POST /api/search/face`: image -> JSON results/meta.
- `POST /api/search/text`: query -> JSON results/meta.
- `POST /api/search/hybrid`: optional image + query -> JSON results/meta.
- `GET,POST /api/people`: list/create people as JSON.
- `GET,PUT,DELETE /api/people/<id>`: read/update/delete one person.
- `POST /api/ai/profile-summary`: person_id -> local summary JSON.

## 12. Authentication & Security
- Signup uses `generate_password_hash()`.
- Login uses `check_password_hash()` and stores `session["username"]`.
- Logout clears the session.
- IMPORTANT limitation: most routes are not protected by a login-required decorator.
- Uploads use extension allowlist, `secure_filename`, timestamped filenames, max size `8 MB`, and Pillow image verification.
- `/media` is restricted to `upfiles`, `dataSet`, and `static`.
- `FLASK_SECRET_KEY` can come from env, but defaults to a development secret.
- API keys are not hardcoded; `LLM_API_KEY` is env-based.
- Raw embeddings are not exposed by public API responses.

## 13. Complete User Flow
A. Login: browser posts `/login` -> `app.login()` -> `repository.verify_user_password()` -> Werkzeug hash check -> session set -> redirect dashboard.

B. Signup: browser posts `/signup` -> `app.signup()` -> password confirmation -> `repository.create_user()` with hash -> redirect login.

C. Add person: browser posts `/submit_form` -> `person_payload()` -> optional `UploadService.save_image()` -> `LBPHBaselineService.create_face_samples()` -> `repository.upsert_person()` -> `lbph_service.train()` -> `search_service.rebuild_index()` -> profile page.

D. Face search: upload -> `UploadService.save_image()` -> `SearchService.face_search()` -> SFace vector search if available, else LBPH -> `SearchResult` list -> UI/JSON.

E. Text search: query -> `TextEmbeddingService.embed_text()` -> stored/live profile text vectors -> cosine top-K -> ranked results.

F. Hybrid search: optional image and query -> run face and/or text searches -> weighted merge -> sorted top-K.

G. Open profile: `/people/<id>` -> `repository.get_person()` -> `llm_service.profile_summary()` local grounded summary -> profile template.

## 14. Important Classes & Functions
- `app.person_payload(source)`: normalizes form/API fields into DB profile payload.
- `app.result_to_dict(result)`: formats API search results without raw embeddings.
- `PersonRepository.init_db()`: creates folders/tables and migrates `People` columns.
- `PersonRepository.upsert_person(data)`: insert/update profile.
- `FaceEmbeddingService.embed_image(path)`: image -> 128D normalized SFace vector.
- `TextEmbeddingService.embed_text(text)`: text -> 384D normalized hashed vector.
- `LBPHBaselineService.create_face_samples(id, path)`: creates grayscale training crops.
- `LBPHBaselineService.train()`: trains and saves LBPH recognizer.
- `LBPHBaselineService.search(path)`: predicts person ID from LBPH recognizer.
- `VectorSearchService.cosine_similarity(a,b)`: similarity score.
- `SearchService.rebuild_index()`: rebuilds face/text `.npy` index.
- `SearchService.face_search(path)`: face retrieval.
- `SearchService.text_search(query)`: semantic retrieval.
- `SearchService.hybrid_search(path, query)`: weighted multimodal ranking.
- `UploadService.save_image(file)`: validates and stores image.
- `LLMService.profile_summary(person)`: local grounded summary fallback.

## 15. Design Decisions
- Flask fits a small server-rendered prototype with simple APIs.
- SQLite keeps setup simple and local for a mini project.
- OpenCV was already aligned with the original face-recognition app and supports both LBPH and deep ONNX models.
- LBPH remains as a baseline/fallback.
- SFace embeddings improve comparison beyond hand-crafted LBPH features.
- `.npy` vectors are simple to persist locally without adding a vector database.
- Cosine similarity is used because embeddings are normalized and angle-based similarity is standard for vector retrieval.
- Services/repositories separate route code from AI/database logic.

## 16. Limitations
- No route-level authorization on most pages/APIs.
- Text embeddings are hashed local vectors, not transformer embeddings.
- Vector search is linear scan; no FAISS/ANN/vector DB.
- LLM API variables exist, but live LLM calls are disabled.
- RAG and AI agent/tool calling are not implemented.
- Face threshold is configurable but not scientifically calibrated.
- No benchmark or accuracy evaluation is present.
- SQLite has no production-grade concurrency/scaling setup.
- Some legacy templates/assets still exist.

## 17. Future Improvements
FUTURE WORK:
- Add login-required authorization to protected routes.
- Replace hashed text vectors with sentence-transformer embeddings.
- Use FAISS or pgvector for larger datasets.
- Move SQLite to PostgreSQL for production.
- Add background jobs for index rebuilds.
- Add rate limiting and audit logs.
- Add face-search evaluation data and threshold tuning.
- Implement real optional LLM/RAG summaries using retrieved profile context.
- Add Docker, CI, production WSGI server, monitoring, and backups.

## 18. Interview Questions
1. What is FindMe AI? A Flask-based multimodal people-search prototype for authorized datasets.
2. What makes it more than CRUD? It uses face embeddings, text vectors, vector search, and hybrid ranking.
3. What database is used? SQLite at `../db/UserDetails.db`.
4. Where are embeddings stored? `.npy` files in `ai_index/face` and `ai_index/text`.
5. Are embeddings stored in SQLite? No, SQLite stores paths and version strings.
6. What is the original face-recognition method? Haar Cascade plus LBPH.
7. What is the upgraded face method? OpenCV YuNet face detection plus SFace embeddings.
8. Which model files are used? `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`.
9. What is the face embedding size? 128 dimensions.
10. What is the text embedding size? 384 dimensions.
11. Is text search transformer-based? No, it uses local hashed vectors with synonym expansion.
12. What similarity metric is used? Cosine similarity.
13. What is top-K? Returning the highest-ranked K matches, default `TOP_K=5`.
14. How is hybrid score calculated? `0.65 * face_similarity + 0.35 * text_similarity` by default.
15. Are scores probabilities? No, they are similarity/relevance match scores.
16. What happens if deep models are missing? Face search falls back to LBPH.
17. What happens if no LBPH recognizer exists? `LBPHBaselineService.search()` trains it if dataset images exist.
18. How are uploads secured? Extension allowlist, secure filename, timestamp, size limit, Pillow verification.
19. How are passwords stored? Werkzeug password hashes.
20. Are routes protected after login? Mostly no; this is a limitation.
21. Is a real LLM call implemented? No, live calls are disabled.
22. Is RAG implemented? No.
23. Is an AI agent implemented? No.
24. How is the index rebuilt? Iterate profiles, compute embeddings, save `.npy`, update DB paths, remove stale vectors.
25. How are old profiles handled? Missing embeddings can be rebuilt from image paths or dataset samples; text can be computed from profile fields.
26. What table stores search metrics? `SearchLogs`.
27. Why use SQLite? Simple local setup for a mini project.
28. How would it scale to production? PostgreSQL plus pgvector/FAISS, auth, background jobs, monitoring.
29. What tests exist? Unit tests for vector ranking, text vectors, upload validation, profile CRUD/search, API validation, media protection.
30. What should you not claim? Do not claim transformer semantic search, production auth, calibrated probabilities, live LLM/RAG, or vector database support.

## Final Honesty Matrix
| Feature | Status | Actual implementation |
|---|---|---|
| Profile CRUD | IMPLEMENTED | Flask routes/API + SQLite repository |
| Password hashing | IMPLEMENTED | Werkzeug hashes |
| Session login | IMPLEMENTED | Flask session |
| Route authorization | PARTIALLY IMPLEMENTED | Login exists, most routes not protected |
| Secure upload checks | IMPLEMENTED | Extension, size, secure filename, Pillow verify |
| LBPH face recognition | IMPLEMENTED | Haar + LBPH recognizer |
| Deep face embeddings | IMPLEMENTED | OpenCV YuNet/SFace ONNX |
| Vector search | IMPLEMENTED | NumPy cosine linear scan |
| Semantic search | IMPLEMENTED | Local hashed text vectors |
| Hybrid search | IMPLEMENTED | Weighted face/text ranking |
| AI index management | IMPLEMENTED | `/ai-index`, rebuild routes |
| LLM summaries | PARTIALLY IMPLEMENTED | Local grounded fallback only |
| RAG | NOT IMPLEMENTED | No retrieval-to-LLM generation |
| AI assistant/agent | NOT IMPLEMENTED | No tool-calling system |
| Benchmarks | NOT IMPLEMENTED | No performance/accuracy benchmark |
