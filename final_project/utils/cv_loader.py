from __future__ import annotations

import sys
from pathlib import Path


def load_cv2():
    try:
        import cv2

        return cv2
    except ModuleNotFoundError:
        system_site_packages = Path(sys.base_prefix) / "Lib" / "site-packages"
        if system_site_packages.exists():
            sys.path.append(str(system_site_packages))
            import cv2

            return cv2
        raise
