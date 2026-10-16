from pydantic import BaseModel


class Detection(BaseModel):
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
    mean_confidence: float | None
    mean_width_px: float | None
    mean_height_px: float | None
    detection_limit_reached: bool


class Prediction(BaseModel):
    model: str
    device: str
    confidence_threshold: float
    image_size: int
    max_detections: int
    inference_ms: float
    total_ms: float
    image: ImageInfo
    statistics: Statistics
    detections: list[Detection]
    original_image: str
    annotated_image: str


class Health(BaseModel):
    status: str
    model: str
    device: str
    classes: dict[int, str]
    error: str | None
    max_upload_bytes: int
    max_image_pixels: int
    max_detections: int
    default_image_size: int
