from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from config import settings
from repositories.person_repository import PersonRepository
from services.embedding_service import FaceEmbeddingService, TextEmbeddingService
from services.lbph_service import LBPHBaselineService
from services.llm_service import LLMService
from services.search_service import SearchService
from services.upload_service import UploadService


app = Flask(__name__, template_folder="Templates")
app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_BYTES
app.secret_key = settings.FLASK_SECRET_KEY

repository = PersonRepository()
upload_service = UploadService()
face_embedding_service = FaceEmbeddingService()
text_embedding_service = TextEmbeddingService()
lbph_service = LBPHBaselineService(repository)
search_service = SearchService(repository, face_embedding_service, text_embedding_service, lbph_service)
llm_service = LLMService()


def init_db() -> None:
    repository.init_db()


def current_user() -> str | None:
    return session.get("username")


def person_payload(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "ID": source.get("ID", "").strip(),
        "Name": source.get("name", source.get("Name", "")).strip(),
        "Age": source.get("age", source.get("Age", "")).strip(),
        "Gender": source.get("gender", source.get("Gender", "")).strip(),
        "CN": source.get("cno", source.get("CN", "")).strip(),
        "Address": source.get("add", source.get("Address", "")).strip(),
        "PersonType": source.get("person", source.get("PersonType", "Authorized")).strip() or "Authorized",
        "Profession": source.get("profession", source.get("Profession", "")).strip(),
        "Skills": source.get("skills", source.get("Skills", "")).strip(),
        "Education": source.get("education", source.get("Education", "")).strip(),
        "Experience": source.get("experience", source.get("Experience", "")).strip(),
        "Projects": source.get("projects", source.get("Projects", "")).strip(),
        "Certifications": source.get("certifications", source.get("Certifications", "")).strip(),
        "Bio": source.get("bio", source.get("Bio", "")).strip(),
    }


def result_to_dict(result) -> dict[str, Any]:
    return {
        "person": public_person(result.person),
        "match_score": round(result.score * 100, 2),
        "face_similarity": round(result.face_similarity * 100, 2) if result.face_similarity is not None else None,
        "profile_relevance": round(result.text_similarity * 100, 2) if result.text_similarity is not None else None,
        "method": result.method,
    }


def public_person(person: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "ID",
        "Name",
        "Age",
        "Gender",
        "Address",
        "PersonType",
        "ImagePath",
        "Profession",
        "Skills",
        "Education",
        "Experience",
        "Projects",
        "Certifications",
        "Bio",
    ]
    return {key: person.get(key) for key in allowed if key in person}


def api_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {
        "current_user": current_user(),
        "app_name": "FINDME AI",
    }


@app.before_request
def ensure_database() -> None:
    init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/services")
def services():
    return redirect(url_for("search"))


@app.route("/search", methods=["GET", "POST"])
@app.route("/verify", methods=["GET", "POST"])
def search():
    results = []
    meta: dict[str, Any] = {}
    mode = request.form.get("mode", request.args.get("mode", "face"))
    error = None
    query = request.form.get("query", "")
    if request.method == "POST":
        try:
            image_path: Path | None = None
            if request.files.get("path") and request.files["path"].filename:
                image_path = upload_service.save_image(request.files.get("path"))
            if mode == "face":
                if image_path is None:
                    raise ValueError("Please upload an image for face search.")
                results, meta = search_service.face_search(image_path)
            elif mode == "text":
                if not query.strip():
                    raise ValueError("Please enter a text search query.")
                results, meta = search_service.text_search(query)
            else:
                if image_path is None and not query.strip():
                    raise ValueError("Please provide an image, text query, or both.")
                results, meta = search_service.hybrid_search(image_path, query)
        except Exception as exc:
            error = str(exc)
    return render_template(
        "search.html",
        mode=mode,
        query=query,
        results=results,
        meta=meta,
        error=error,
        index_status=search_service.index_status(),
    )


@app.route("/search_img", methods=["POST"])
def search_img():
    try:
        image_path = upload_service.save_image(request.files.get("path"))
        results, meta = search_service.face_search(image_path)
        return render_template("result.html", mode="found" if results else "not_found", results=results, meta=meta)
    except Exception as exc:
        return render_template("result.html", mode="error", error=str(exc), results=[], meta={}), 400


@app.route("/collect")
def collect():
    return render_template("form.html")


@app.route("/submit_form", methods=["POST"])
def submit_form():
    try:
        payload = person_payload(request.form)
        if not payload["ID"].isdigit():
            raise ValueError("ID number must contain digits only.")
        image_file = request.files.get("path")
        if image_file and image_file.filename:
            image_path = upload_service.save_image(image_file)
            payload["ImagePath"] = str(image_path.relative_to(settings.BASE_DIR))
            lbph_service.create_face_samples(payload["ID"], image_path)
        repository.upsert_person(payload)
        lbph_people = lbph_service.train()
        index_report = search_service.rebuild_index()
        person = repository.get_person(payload["ID"])
        return render_template("profile.html", person=person, summary=llm_service.profile_summary(person or {}), created=True, lbph_people=lbph_people, index_report=index_report)
    except Exception as exc:
        return render_template("result.html", mode="error", error=str(exc), results=[], meta={}), 400


@app.route("/people")
def people():
    return render_template("people.html", people=repository.all_people())


@app.route("/people/<person_id>")
def profile(person_id: str):
    person = repository.get_person(person_id)
    if not person:
        return render_template("result.html", mode="error", error="Profile not found.", results=[], meta={}), 404
    return render_template("profile.html", person=person, summary=llm_service.profile_summary(person), created=False)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=repository.dashboard_stats())


@app.route("/ai-index")
def ai_index():
    return render_template("ai_index.html", status=search_service.index_status(), report=None)


@app.route("/admin/rebuild-index", methods=["POST"])
@app.route("/api/admin/rebuild-index", methods=["POST"])
def rebuild_index():
    report = search_service.rebuild_index()
    if request.path.startswith("/api/"):
        return jsonify(report)
    return render_template("ai_index.html", status=search_service.index_status(), report=report)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not username or not password:
            flash("Please enter a username and password.")
        elif password != confirm_password:
            flash("Passwords do not match.")
        else:
            try:
                repository.create_user(username, generate_password_hash(password))
                flash("Account created. Please log in.")
                return redirect(url_for("login"))
            except Exception:
                flash("An account already exists for this username.")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        stored_hash = repository.verify_user_password(username)
        if stored_hash and check_password_hash(stored_hash, password):
            session["username"] = username
            flash("Logged in successfully.")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("index"))


@app.route("/media/<path:filename>")
def media(filename: str):
    target = (settings.BASE_DIR / filename).resolve()
    allowed_roots = [settings.UPLOAD_DIR.resolve(), settings.DATASET_DIR.resolve(), (settings.BASE_DIR / "static").resolve()]
    if not any(target == root or root in target.parents for root in allowed_roots):
        return "Not found", 404
    return send_from_directory(target.parent, target.name)


@app.route("/api/search/face", methods=["POST"])
def api_search_face():
    try:
        image_path = upload_service.save_image(request.files.get("path"))
        results, meta = search_service.face_search(image_path)
        return jsonify({"results": [result_to_dict(result) for result in results], "meta": meta})
    except ValueError as exc:
        return api_error(str(exc))


@app.route("/api/search/text", methods=["POST"])
def api_search_text():
    data = request.get_json(silent=True) or request.form
    query = data.get("query", "")
    if not query.strip():
        return api_error("Please enter a text search query.")
    results, meta = search_service.text_search(query)
    return jsonify({"results": [result_to_dict(result) for result in results], "meta": meta})


@app.route("/api/search/hybrid", methods=["POST"])
def api_search_hybrid():
    try:
        query = request.form.get("query", "")
        image_path = None
        if request.files.get("path") and request.files["path"].filename:
            image_path = upload_service.save_image(request.files.get("path"))
        if image_path is None and not query.strip():
            return api_error("Please provide an image, text query, or both.")
        results, meta = search_service.hybrid_search(image_path, query)
        return jsonify({"results": [result_to_dict(result) for result in results], "meta": meta})
    except ValueError as exc:
        return api_error(str(exc))


@app.route("/api/people", methods=["GET", "POST"])
def api_people():
    if request.method == "GET":
        return jsonify({"people": [public_person(person) for person in repository.all_people()]})
    payload = person_payload(request.get_json(silent=True) or request.form)
    try:
        repository.upsert_person(payload)
    except ValueError as exc:
        return api_error(str(exc))
    return jsonify({"person": public_person(repository.get_person(payload["ID"]) or {})}), 201


@app.route("/api/people/<person_id>", methods=["GET", "PUT", "DELETE"])
def api_person(person_id: str):
    if request.method == "GET":
        person = repository.get_person(person_id)
        if not person:
            return jsonify({"error": "Profile not found."}), 404
        return jsonify({"person": public_person(person)})
    if request.method == "DELETE":
        deleted = repository.delete_person(person_id)
        return jsonify({"deleted": deleted})
    payload = person_payload(request.get_json(silent=True) or request.form)
    payload["ID"] = person_id
    try:
        repository.upsert_person(payload)
    except ValueError as exc:
        return api_error(str(exc))
    return jsonify({"person": public_person(repository.get_person(person_id) or {})})


@app.route("/api/ai/profile-summary", methods=["POST"])
def api_profile_summary():
    data = request.get_json(silent=True) or request.form
    person_id = data.get("person_id", "")
    person = repository.get_person(person_id)
    if not person:
        return jsonify({"error": "Profile not found."}), 404
    return jsonify(llm_service.profile_summary(person))


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
