"""Paths are independent of the directory from which the server is started."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def model_path() -> Path:
    path = Path(os.getenv("MODEL_PATH", "models/best_yolo26.pt"))
    return path if path.is_absolute() else ROOT / path


def classifier_path() -> Path:
    path = Path(os.getenv("CLASSIFIER_PATH", "models/bestClassifier.pt"))
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=model_path)
    classifier_path: Path = field(default_factory=classifier_path)
    device: str = field(default_factory=lambda: os.getenv("MODEL_DEVICE", "auto"))
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    max_detections: int = 1000
    default_image_size: int = 1024
    confidence_threshold: float = 0.25
    classifier_confidence_threshold: float = 0.70
    classifier_batch_size: int = 16
    classifier_image_size: int = 320
    classifier_crop_padding: float = 0.1
    classifier_normalization: str = field(default_factory=lambda: os.getenv("CLASSIFIER_NORMALIZATION", "imagenet"))
