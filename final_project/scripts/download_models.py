from __future__ import annotations

import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}")
            continue
        print(f"downloading: {url}")
        urllib.request.urlretrieve(url, target)
        print(f"saved: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
