from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import cv2
except ModuleNotFoundError:
    system_site_packages = Path(sys.base_prefix) / "Lib" / "site-packages"
    if system_site_packages.exists():
        sys.path.append(str(system_site_packages))
        import cv2
    else:
        raise


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DB_DIR = PROJECT_DIR / "db"
DB_PATH = DB_DIR / "UserDetails.db"
DATASET_DIR = BASE_DIR / "dataSet"
RECOGNIZER_DIR = BASE_DIR / "recognizer"
RECOGNIZER_PATH = RECOGNIZER_DIR / "trainningData.yml"
UPLOAD_DIR = BASE_DIR / "upfiles"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


app = Flask(__name__, template_folder="Templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.secret_key = os.environ.get("FIND_ME_SECRET_KEY", "find-me-local-dev-secret")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def connect_db() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RECOGNIZER_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS People (
                ID TEXT PRIMARY KEY,
                Name TEXT NOT NULL,
                Age INTEGER NOT NULL,
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
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(People)").fetchall()
        }
        migrations = {
            "PersonType": "ALTER TABLE People ADD COLUMN PersonType TEXT DEFAULT 'Indian'",
            "ImagePath": "ALTER TABLE People ADD COLUMN ImagePath TEXT",
            "CreatedAt": "ALTER TABLE People ADD COLUMN CreatedAt TEXT",
            "UpdatedAt": "ALTER TABLE People ADD COLUMN UpdatedAt TEXT",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                conn.execute(statement)
        conn.commit()


def save_upload(field_name: str) -> Path:
    uploaded_file = request.files.get(field_name)
    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("Please upload an image.")
    if not allowed_file(uploaded_file.filename):
        raise ValueError("Only JPG, JPEG, PNG, and WEBP images are supported.")

    safe_name = secure_filename(uploaded_file.filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    target = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    uploaded_file.save(target)
    return target


def detect_faces(image_path: Path) -> list[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded image could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return [gray[y : y + h, x : x + w] for (x, y, w, h) in faces]


def create_face_samples(person_id: str, image_path: Path) -> int:
    faces = detect_faces(image_path)
    if not faces:
        raise ValueError("No clear face was detected in the uploaded image.")

    for old_sample in DATASET_DIR.glob(f"User.{person_id}.*.jpg"):
        old_sample.unlink()

    sample_count = 0
    for face in faces:
        resized = cv2.resize(face, (220, 220))
        for _ in range(12):
            sample_count += 1
            cv2.imwrite(str(DATASET_DIR / f"User.{person_id}.{sample_count}.jpg"), resized)
    return sample_count


def train_recognizer() -> int:
    image_paths = sorted(DATASET_DIR.glob("User.*.*.jpg"))
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
        raise ValueError("No training images are available yet.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(str(RECOGNIZER_PATH))
    return len(set(ids))


def get_person(person_id: str) -> sqlite3.Row | None:
    with connect_db() as conn:
        return conn.execute("SELECT * FROM People WHERE ID = ?", (person_id,)).fetchone()


def upsert_person(data: dict[str, Any]) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO People (ID, Name, Age, Gender, CN, Address, PersonType, ImagePath, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ID) DO UPDATE SET
                Name = excluded.Name,
                Age = excluded.Age,
                Gender = excluded.Gender,
                CN = excluded.CN,
                Address = excluded.Address,
                PersonType = excluded.PersonType,
                ImagePath = excluded.ImagePath,
                UpdatedAt = excluded.UpdatedAt
            """,
            (
                data["ID"],
                data["Name"],
                data["Age"],
                data["Gender"],
                data["CN"],
                data["Address"],
                data["PersonType"],
                data["ImagePath"],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def find_matching_person(image_path: Path) -> tuple[sqlite3.Row | None, float | None]:
    if not RECOGNIZER_PATH.exists():
        train_recognizer()

    faces = detect_faces(image_path)
    if not faces:
        raise ValueError("No clear face was detected in the search image.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(RECOGNIZER_PATH))

    best_id: int | None = None
    best_confidence: float | None = None
    for face in faces:
        resized = cv2.resize(face, (220, 220))
        predicted_id, confidence = recognizer.predict(resized)
        if best_confidence is None or confidence < best_confidence:
            best_id = predicted_id
            best_confidence = float(confidence)

    if best_id is None or best_confidence is None:
        return None, None

    # Lower LBPH confidence is better. This threshold keeps weak matches out.
    if best_confidence > 90:
        return None, best_confidence

    return get_person(str(best_id)), best_confidence


@app.context_processor
def inject_year() -> dict[str, int]:
    return {"current_year": datetime.now().year}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")


@app.route("/about")
def about():
    return render_template("about.html")


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
                with connect_db() as conn:
                    conn.execute(
                        "INSERT INTO Users (username, password) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                    )
                    conn.commit()
                flash("Account created. Please log in.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("An account already exists for this username.")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with connect_db() as conn:
            user = conn.execute(
                "SELECT * FROM Users WHERE username = ?", (username,)
            ).fetchone()
        if user and check_password_hash(user["password"], password):
            flash("Logged in successfully.")
            return redirect(url_for("services"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/verify")
def verify():
    return render_template("verify.html")


@app.route("/collect")
def collect():
    return render_template("form.html")


@app.route("/submit_form", methods=["POST"])
def submit_form():
    try:
        person_id = request.form.get("ID", "").strip()
        if not person_id.isdigit():
            raise ValueError("ID number must contain digits only.")

        image_path = save_upload("path")
        person = {
            "ID": person_id,
            "Name": request.form.get("name", "").strip(),
            "Age": int(request.form.get("age", "0")),
            "Gender": request.form.get("gender", "").strip(),
            "CN": request.form.get("cno", "").strip(),
            "Address": request.form.get("add", "").strip(),
            "PersonType": request.form.get("person", "Indian").strip() or "Indian",
            "ImagePath": str(image_path.relative_to(BASE_DIR)),
        }

        missing_fields = [key for key, value in person.items() if key != "ImagePath" and not value]
        if missing_fields:
            raise ValueError("Please fill all required fields.")

        samples = create_face_samples(person_id, image_path)
        upsert_person(person)
        trained_people = train_recognizer()
        person_row = get_person(person_id)
        return render_template(
            "result.html",
            mode="created",
            person=person_row,
            samples=samples,
            trained_people=trained_people,
            confidence=None,
        )
    except Exception as exc:
        return render_template("result.html", mode="error", error=str(exc)), 400


@app.route("/search_img", methods=["POST"])
def search_img():
    try:
        image_path = save_upload("path")
        person, confidence = find_matching_person(image_path)
        if person is None:
            return render_template(
                "result.html",
                mode="not_found",
                confidence=confidence,
            )
        return render_template(
            "result.html",
            mode="found",
            person=person,
            confidence=round(confidence, 2) if confidence is not None else None,
        )
    except Exception as exc:
        return render_template("result.html", mode="error", error=str(exc)), 400


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
