"""Image and source-pool I/O for CDD-11 synthesis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_images(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {root}")
    indexed: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(f"Duplicate image stem in {root}: {path.stem}")
        indexed[path.stem] = path.resolve()
    if not indexed:
        raise ValueError(f"No supported images found in {root}")
    return dict(sorted(indexed.items()))


def _read(path: str | Path, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    image = cv2.imread(str(resolved), flags)
    if image is None:
        raise ValueError(f"Unable to decode image: {resolved}")
    return image


def load_clean(path: str | Path) -> np.ndarray:
    image = _read(path, flags=cv2.IMREAD_COLOR)
    return image.astype(np.float64) / 255.0


def load_light_map(path: str | Path) -> np.ndarray:
    image = _read(path, flags=cv2.IMREAD_COLOR)
    # Preserve the official syn_data.py conversion path.
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0


def load_depth_map(path: str | Path) -> np.ndarray:
    return _read(path, flags=cv2.IMREAD_COLOR).astype(np.float64) / 255.0


def load_mask(path: str | Path) -> np.ndarray:
    return _read(path, flags=cv2.IMREAD_COLOR).astype(np.float64) / 255.0


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(destination), encoded):
        raise IOError(f"Failed to write image: {destination}")
    return destination
