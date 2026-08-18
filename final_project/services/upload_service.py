from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import settings


class UploadService:
    def allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.ALLOWED_EXTENSIONS

    def save_image(self, uploaded_file: FileStorage | None) -> Path:
        if uploaded_file is None or not uploaded_file.filename:
            raise ValueError("Please upload an image.")
        if not self.allowed_file(uploaded_file.filename):
            raise ValueError("Only JPG, JPEG, PNG, and WEBP images are supported.")

        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = secure_filename(uploaded_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        target = settings.UPLOAD_DIR / f"{timestamp}_{safe_name}"
        uploaded_file.save(target)
        if target.stat().st_size > settings.MAX_UPLOAD_BYTES:
            target.unlink(missing_ok=True)
            raise ValueError("Image is too large. Maximum size is 8 MB.")
        try:
            with Image.open(target) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            target.unlink(missing_ok=True)
            raise ValueError("The uploaded file is not a valid image.") from exc
        return target
