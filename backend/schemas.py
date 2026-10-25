from typing import Literal

from pydantic import BaseModel


class Classification(BaseModel):
    coating_class: Literal["coated", "insufficient_coating"]
    p_coated: float
    p_insufficient: float
    classifier_confidence: float
    coating_status: Literal["coated", "uncoated", "uncertain"]


class Detection(Classification):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    width_px: float
    height_px: float


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int


class Statistics(BaseModel):
    count: int
    coated_count: int
    uncoated_count: int
    uncertain_count: int
    coverage_percent: float | None
    mean_classifier_confidence: float | None
    mean_confidence: float | None
    mean_width_px: float | None
    mean_height_px: float | None
    detection_limit_reached: bool


class Prediction(BaseModel):
    model: str
    classifier: str
    classifier_confidence_threshold: float
    classifier_normalization: str
    device: str
    confidence_threshold: float
    image_size: int
    max_detections: int
    inference_ms: float
    detector_inference_ms: float
    classifier_inference_ms: float
    total_ms: float
    image: ImageInfo
    statistics: Statistics
    detections: list[Detection]
    original_image: str
    annotated_image: str


class Health(BaseModel):
    status: str
    model: str
    classifier: str
    classifier_ready: bool
    classifier_confidence_threshold: float
    classifier_normalization: str
    device: str
    classes: dict[int, str]
    error: str | None
    max_upload_bytes: int
    max_image_pixels: int
    max_detections: int
    default_image_size: int
