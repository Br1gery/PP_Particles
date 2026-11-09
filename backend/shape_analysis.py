"""Particle segmentation followed by geometric measurements, in source pixels."""

import logging
from hashlib import sha256
from math import pi, sqrt
from statistics import median
from threading import Lock
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
from PIL import ImageDraw
from pydantic import BaseModel, Field

from backend.config import Settings, CONTOUR_PROCESSING_VERSION
from backend.inference import ModelBusy, ModelUnavailable, decode_image, image_data_url
from backend.postprocessing import suppress_duplicate_boxes

logger = logging.getLogger(__name__)
SHAPE_COLORS = {"rounded": "#22a06b", "irregular": "#e9a2a2", "edge": "#e5b62f"}


class ShapeParticle(BaseModel):
    index: int
    form: Literal["rounded", "irregular", "edge"]
    confidence: float | None
    class_id: int | None
    class_name: str | None
    classification_source: Literal["model", "circularity", "manual"]
    source: Literal["model", "manual"] = "model"
    manual_form: Literal["rounded", "irregular"] | None = None
    manual_contour: bool = False
    circularity: float
    area_px2: float
    perimeter_px: float
    equivalent_diameter_px: float
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]  # OpenCV pixel bounding rectangle: x, y, width, height
    touches_border: bool
    polygon_px: list[tuple[float, float]]
    contour_refined: bool = False
    removed_area_percent: float = 0.0
    raw_polygon_px: list[tuple[float, float]] | None = None
    contour_refinement: list[Literal["overlap", "neck"]] = Field(default_factory=list)


class ShapeSummary(BaseModel):
    total: int
    rounded: int
    irregular: int
    edge: int
    shape_denominator: int
    rounded_percent: float | None
    irregular_percent: float | None
    median_circularity: float | None
    median_equivalent_diameter_px: float | None


class ShapePrediction(BaseModel):
    model: str
    contour_processing_version: str = CONTOUR_PROCESSING_VERSION
    model_sha256: str | None
    classes: dict[int, str]
    classification_source: Literal["model", "circularity"]
    inference_mode: Literal["nms"] = "nms"
    mask_size: tuple[int, int] | None  # height, width of decoded model masks
    mask_quality_warning: str | None = None
    device: str
    confidence_threshold: float
    circularity_threshold: float
    duplicate_iou_threshold: float
    image_size: int
    max_detections: int
    detection_limit_reached: bool
    total_ms: float
    image: dict
    summary: ShapeSummary
    particles: list[ShapeParticle]
    original_image: str
    annotated_image: str


def measure_particle(polygon, index: int, confidence: float | None, width: int, height: int,
                     threshold: float = .80, *, class_id: int | None = 0, class_name: str | None = "particle",
                     source: Literal["model", "manual"] = "model", manual_form=None, manual_contour=False) -> ShapeParticle | None:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    area = cv2.contourArea(points)
    perimeter = cv2.arcLength(points, True)
    if area <= 0 or perimeter <= 0:
        return None
    circularity = min(1., 4 * pi * area / perimeter ** 2)
    moments = cv2.moments(points)
    xmin, ymin = points.min(axis=0).tolist()
    xmax, ymax = points.max(axis=0).tolist()
    edge = xmin <= 1 or ymin <= 1 or xmax >= width - 2 or ymax >= height - 2
    model_form = class_name if class_name in {"rounded", "irregular"} else None
    form = manual_form or model_form or ("rounded" if circularity >= threshold else "irregular")
    return ShapeParticle(
        index=index, form="edge" if edge else form,
        class_id=class_id, class_name=class_name,
        classification_source="manual" if manual_form or source == "manual" else "model" if model_form is not None else "circularity",
        source=source, manual_form=manual_form, manual_contour=manual_contour,
        confidence=round(float(confidence), 6) if confidence is not None else None, circularity=round(circularity, 6),
        area_px2=round(area, 3), perimeter_px=round(perimeter, 3),
        equivalent_diameter_px=round(sqrt(4 * area / pi), 3),
        centroid_x=round(moments["m10"] / moments["m00"], 3),
        centroid_y=round(moments["m01"] / moments["m00"], 3),
        bbox=cv2.boundingRect(points), touches_border=edge,
        polygon_px=points.tolist(),
    )


def summarize_shapes(particles: list[ShapeParticle]) -> ShapeSummary:
    interior = [p for p in particles if not p.touches_border]
    rounded = sum(p.form == "rounded" for p in interior)
    count = len(interior)
    return ShapeSummary(
        total=len(particles), rounded=rounded, irregular=count - rounded, edge=len(particles) - count,
        shape_denominator=count, rounded_percent=round(100 * rounded / count, 3) if count else None,
        irregular_percent=round(100 * (count - rounded) / count, 3) if count else None,
        median_circularity=round(median(p.circularity for p in interior), 6) if count else None,
        median_equivalent_diameter_px=round(median(p.equivalent_diameter_px for p in interior), 3) if count else None,
    )


def annotate_shapes(image, particles):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for p in particles:
        points = [tuple(point) for point in p.polygon_px]
        draw.line(points + points[:1], fill=SHAPE_COLORS[p.form], width=max(2, round(min(image.size) / 500)))
    return annotated


class ShapeAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = settings.device
        self.error = None
        self.lock = Lock()
        self.classes: dict[int, str] = {}
        self.model_sha256: str | None = None
        self.weights_signature = None

    @property
    def classification_source(self):
        return "model" if set(self.classes.values()) == {"rounded", "irregular"} else "circularity"

    @property
    def weights_changed(self):
        if self.weights_signature is None:
            return False
        try:
            stat = self.settings.shape_model_path.stat()
            return (stat.st_mtime_ns, stat.st_size) != self.weights_signature
        except OSError:
            return True

    @property
    def unavailable_reason(self):
        return "Файл particle_yolo26.pt изменён после загрузки. Перезапустите backend, чтобы применить новые веса." if self.weights_changed else self.error

    @property
    def ready(self):
        return self.model is not None and not self.weights_changed

    def load(self):
        try:
            import torch
            from ultralytics import YOLO

            if not self.settings.shape_model_path.is_file():
                raise FileNotFoundError(self.settings.shape_model_path)
            stat = self.settings.shape_model_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            fingerprint = sha256(self.settings.shape_model_path.read_bytes()).hexdigest()
            self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if self.settings.device == "auto" else self.settings.device
            model = YOLO(str(self.settings.shape_model_path))
            if model.task != "segment":
                raise ValueError("Roundness analysis requires segmentation weights")
            classes = {int(key): str(value) for key, value in model.names.items()}
            if not (len(classes) == 1 and set(classes.values()) == {"particle"}
                    or len(classes) == 2 and set(classes.values()) == {"rounded", "irregular"}):
                raise ValueError(f"Unsupported shape classes: {classes}")
            model.to(self.device)
            stat = self.settings.shape_model_path.stat()
            if (stat.st_mtime_ns, stat.st_size) != signature:
                raise RuntimeError("Weights changed while loading")
            self.model, self.error = model, None
            self.classes, self.model_sha256, self.weights_signature = classes, fingerprint, signature
            logger.info("Loaded shape model %s on %s; SHA256=%s; classes=%s; classification=%s",
                        self.settings.shape_model_path.resolve(), self.device, fingerprint, classes, self.classification_source)
        except Exception:
            logger.exception("Could not load shape segmentation model")
            self.model = None
            self.classes, self.model_sha256, self.weights_signature = {}, None, None
            self.error = "Не удалось загрузить модель округлости. Проверьте models/particle_yolo26.pt; подробности в журнале сервера."

    def predict(self, data: bytes, filename: str) -> ShapePrediction:
        if not self.ready:
            raise ModelUnavailable(self.unavailable_reason or "Модель округлости недоступна.")
        if not self.lock.acquire(blocking=False):
            raise ModelBusy("Модель округлости обрабатывает другое изображение. Повторите запрос позже.")
        try:
            started = perf_counter()
            from backend.segmentation import NativeSegmentationPredictor

            # No footer-cropping heuristic was supplied; keep the whole image and its coordinates.
            image = decode_image(data, self.settings.max_image_pixels).convert("L").convert("RGB")
            # In Ultralytics 8.4.24 end2end=False selects the trained one-to-many head
            # with NMS. Applying NMS later to one-to-one predictions is not equivalent.
            result = self.model.predict(source=image, conf=self.settings.shape_confidence_threshold,
                                        imgsz=self.settings.shape_image_size, iou=self.settings.shape_duplicate_iou_threshold,
                                        max_det=self.settings.max_detections, device=self.device,
                                        end2end=False, retina_masks=True, predictor=NativeSegmentationPredictor,
                                        save=False, verbose=False)[0]
            boxes = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            mask_data = getattr(result.masks, "data", None)
            from backend.contours import particle_contours

            mask_size = tuple(mask_data.shape[-2:]) if mask_data is not None else None
            if len(boxes) != (len(mask_data) if mask_data is not None else 0):
                raise RuntimeError("Segmentation boxes and contours are not aligned")
            candidates = []
            for i, row in enumerate(boxes):
                x1, y1, x2, y2, score, class_id = row[:6]
                x1, x2 = (max(0., min(float(x), image.width)) for x in (x1, x2))
                y1, y2 = (max(0., min(float(y), image.height)) for y in (y1, y2))
                if x2 > x1 and y2 > y1:
                    # Carry the mask's index through NMS; never pair a retained box with another mask.
                    candidates.append([x1, y1, x2, y2, score, class_id, i])
            retained = suppress_duplicate_boxes(candidates, self.settings.shape_duplicate_iou_threshold)
            # Resolve mask conflicts only after duplicate detections are suppressed.
            selected_masks = mask_data[[int(row[6]) for row in retained]] if retained else None
            contours = particle_contours(selected_masks, (image.height, image.width),
                                        class_names=[self.classes.get(int(row[5]), "") for row in retained],
                                        confidences=[row[4] for row in retained], resolve_overlaps=True) if retained else []
            particles = []
            for detection_index, row in enumerate(retained):
                class_id = int(row[5])
                if class_id not in self.classes:
                    raise RuntimeError(f"Unknown segmentation class {class_id}")
                contour = contours[detection_index]
                particle = measure_particle(contour.polygon, len(particles) + 1, row[4], image.width,
                                            image.height, self.settings.circularity_threshold,
                                            class_id=class_id, class_name=self.classes[class_id])
                if particle is not None:
                    if contour.refinement_methods:
                        particle.contour_refined = True
                        particle.contour_refinement = list(contour.refinement_methods)
                        particle.removed_area_percent = contour.removed_area_percent
                        particle.raw_polygon_px = [tuple(point) for point in contour.original_polygon.tolist()]
                    particles.append(particle)
            return ShapePrediction(
                model=self.settings.shape_model_path.name, device=self.device,
                model_sha256=self.model_sha256, classes=self.classes, classification_source=self.classification_source,
                mask_size=mask_size, mask_quality_warning=getattr(result, "mask_quality_warning", None),
                confidence_threshold=self.settings.shape_confidence_threshold,
                circularity_threshold=self.settings.circularity_threshold,
                duplicate_iou_threshold=self.settings.shape_duplicate_iou_threshold,
                image_size=self.settings.shape_image_size, max_detections=self.settings.max_detections,
                detection_limit_reached=len(boxes) >= self.settings.max_detections,
                image={"filename": filename, "width": image.width, "height": image.height, "footer_cropped": False},
                summary=summarize_shapes(particles), particles=particles,
                original_image=image_data_url(image), annotated_image=image_data_url(annotate_shapes(image, particles)),
                total_ms=round((perf_counter() - started) * 1000, 2),
            )
        finally:
            self.lock.release()
