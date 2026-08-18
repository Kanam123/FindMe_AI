# FINDME AI

AI-powered multimodal people search for an authorized, consented private dataset.

FINDME AI combines a traditional OpenCV LBPH baseline with OpenCV YuNet/SFace deep face embeddings, local vector similarity search, semantic profile search, hybrid face plus text ranking, and grounded profile summaries.

## Current Architecture

- `app.py`: Flask routes, API endpoints, session login, template rendering.
- `repositories/person_repository.py`: SQLite schema migration, users, profiles, search logs.
- `services/embedding_service.py`: SFace face embeddings and local hashed text embeddings.
- `services/vector_search_service.py`: cosine similarity top-K retrieval.
- `services/search_service.py`: index rebuilds, face search, text search, hybrid ranking.
- `services/lbph_service.py`: Haar cascade plus LBPH baseline/fallback.
- `services/upload_service.py`: secure filenames, extension checks, size checks, image verification.
- `Templates/` and `static/css/app.css`: polished FINDME AI interface.

## AI Pipeline

Face search:

1. Upload an image.
2. Validate and store the image in `upfiles/`.
3. Detect/align the face with OpenCV YuNet/SFace.
4. Generate a 128-dimensional normalized face embedding.
5. Compare against stored `.npy` embeddings with cosine similarity.
6. Return top-K candidates above `FACE_SIMILARITY_THRESHOLD`.

Fallback face search uses Haar cascade detection plus LBPH when deep model files are unavailable.

Semantic search:

1. Build a text representation from stored profile fields only.
2. Generate local hashed text vectors.
3. Retrieve top-K profile matches by cosine similarity.

Hybrid search combines face similarity and profile relevance:

- `FACE_WEIGHT`, default `0.65`
- `TEXT_WEIGHT`, default `0.35`

Scores are similarity/relevance scores, not calibrated probabilities.

## Setup

```bat
cd "C:\Users\aayus\OneDrive\Desktop\Final mini project\final_project"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:5000`.

The helper `run_project.bat` creates `.venv`, installs dependencies, and starts Flask.

## Environment Variables

- `FLASK_SECRET_KEY`: Flask session secret.
- `VECTOR_INDEX_PATH`: optional custom index directory.
- `FACE_SIMILARITY_THRESHOLD`: default `0.42`.
- `FACE_WEIGHT`: default `0.65`.
- `TEXT_WEIGHT`: default `0.35`.
- `TOP_K`: default `5`.
- `LLM_API_KEY`: optional. The current app remains local-only if unset.
- `LLM_MODEL`: default `gpt-4o-mini`.
- `LLM_BASE_URL`: default `https://api.openai.com/v1`.

## Privacy And Safety

This project is for authorized private datasets only. It does not scrape the internet, social networks, or public people data. Do not use it for covert identification, mass surveillance, or identifying strangers without consent.

Raw embeddings are stored locally in `ai_index/` and are not exposed through public API responses.

## Tests

```bat
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests cover vector ranking, semantic text matching, upload validation, profile CRUD/indexing, API validation, and media route protection.

## Limitations

- Text embeddings are local hashed vectors, useful for a CPU-friendly prototype but not as strong as a transformer embedding model.
- LLM/RAG summaries currently use grounded local profile facts and do not make live API calls.
- Face recognition quality depends on image quality and the private dataset.
- Existing old profiles may have sparse profile intelligence fields until enriched manually.
