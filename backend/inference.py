import base64
import logging
import warnings
from io import BytesIO
from threading import Lock
from time import perf_counter

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from backend.config import Settings
from backend.classifier import CoverageClassifier, COATING_COLORS
from backend.postprocessing import suppress_duplicate_boxes
from backend.schemas import Detection, ImageInfo, Prediction, Statistics

logger = logging.getLogger(__name__)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


class InvalidImage(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class ModelBusy(RuntimeError):
    pass


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Поддерживаются JPEG, PNG, WEBP, BMP и TIFF.")
                if source.width * source.height > max_pixels:
                    raise InvalidImage(f"Изображение превышает лимит {max_pixels:,} пикселей.")
                if getattr(source, "n_frames", 1) != 1:
                    raise InvalidImage("Загрузите однокадровое изображение.")
                # Coordinates and preview use the same orientation, including phone photos.
                return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Не удалось прочитать изображение. Проверьте формат и целостность файла.") from exc


def image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(image.size) / 500))
    for detection in detections:
        draw.rectangle(detection.bbox, outline=COATING_COLORS[detection.coating_status], width=line_width)
    return annotated


class Detector:
    """One model per process; prevent concurrent use of Ultralytics' predictor."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.classifier = CoverageClassifier(settings)
        self.device = settings.device
        self.classes: dict[int, str] = {}
        self.error: str | None = None
        self.lock = Lock()

    def load(self) -> None:
        try:
            if not self.settings.model_path.is_file():
                raise FileNotFoundError(self.settings.model_path)
            import torch
            from ultralytics import YOLO

            self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if self.settings.device == "auto" else self.settings.device
            model = YOLO(str(self.settings.model_path), task="detect")
            model.to(self.device)
            self.classifier.load(self.device)
            self.classes = dict(model.names)
            self.model = model
            self.error = None
            logger.info("Loaded %s on %s", self.settings.model_path.name, self.device)
        except Exception:
            logger.exception("Could not load detection/classification pipeline")
            self.model = None
            self.error = "Не удалось загрузить модели. Проверьте best_yolo26.pt, bestClassifier.pt и настройки; подробности в журнале сервера."

    @property
    def ready(self) -> bool:
        return self.model is not None and self.classifier.ready

    def predict(self, data: bytes, filename: str) -> Prediction:
        if not self.ready:
            raise ModelUnavailable(self.error or "Модель недоступна.")
        if not self.lock.acquire(blocking=False):
            raise ModelBusy("Модель обрабатывает другое изображение. Повторите запрос после завершения.")
        try:
            started = perf_counter()
            image = decode_image(data, self.settings.max_image_pixels)
            inference_started = perf_counter()
            result = self.model.predict(
                source=image,
                conf=self.settings.confidence_threshold,
                imgsz=self.settings.default_image_size,
                max_det=self.settings.max_detections,
                device=self.device,
                save=False,
                verbose=False,
            )[0]
            # Transfer all detections to CPU once, instead of synchronizing per box.
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            # Keep the model's candidate-limit warning even if NMS removes some rows.
            detection_limit_reached = len(rows) >= self.settings.max_detections
            candidates = []
            for row in rows:
                x1, y1, x2, y2, score, class_id = row[:6]
                x1, x2 = (max(0.0, min(float(x), image.width)) for x in (x1, x2))
                y1, y2 = (max(0.0, min(float(y), image.height)) for y in (y1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append([x1, y1, x2, y2, score, class_id])
            rows = suppress_duplicate_boxes(candidates, self.settings.duplicate_iou_threshold)
            detector_inference_ms = (perf_counter() - inference_started) * 1000
            particles = []
            for index, (x1, y1, x2, y2, score, class_id) in enumerate(rows, 1):
                particles.append(dict(
                    id=index, class_id=int(class_id), class_name=self.classes.get(int(class_id), str(int(class_id))),
                    confidence=round(score, 6), bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                    width_px=round(x2 - x1, 2), height_px=round(y2 - y1, 2),
                ))
            classification_started = perf_counter()
            classifications = self.classifier.classify(image, [p["bbox"] for p in particles])
            classifier_inference_ms = (perf_counter() - classification_started) * 1000
            if len(classifications) != len(particles):
                raise RuntimeError("Classifier output does not match the detected particles")
            detections = [Detection(**particle, **classification.model_dump())
                          for particle, classification in zip(particles, classifications)]
            count = len(detections)
            coated_count = sum(d.coating_status == "coated" for d in detections)
            uncoated_count = sum(d.coating_status == "uncoated" for d in detections)
            statistics = Statistics(
                count=count,
                coated_count=coated_count, uncoated_count=uncoated_count,
                uncertain_count=count - coated_count - uncoated_count,
                coverage_percent=round(coated_count / count * 100, 2) if count else None,
                mean_classifier_confidence=sum(d.classifier_confidence for d in detections) / count if count else None,
                mean_confidence=round(sum(d.confidence for d in detections) / count, 6) if count else None,
                mean_width_px=round(sum(d.width_px for d in detections) / count, 2) if count else None,
                mean_height_px=round(sum(d.height_px for d in detections) / count, 2) if count else None,
                detection_limit_reached=detection_limit_reached,
            )
            original = image_data_url(image)
            annotated = image_data_url(annotate(image, detections))
            return Prediction(
                model=self.settings.model_path.name, device=self.device,
                classifier=self.settings.classifier_path.name,
                classifier_confidence_threshold=self.settings.classifier_confidence_threshold,
                classifier_normalization=self.settings.classifier_normalization,
                confidence_threshold=self.settings.confidence_threshold,
                image_size=self.settings.default_image_size, max_detections=self.settings.max_detections,
                inference_ms=round(detector_inference_ms + classifier_inference_ms, 2),
                detector_inference_ms=round(detector_inference_ms, 2),
                classifier_inference_ms=round(classifier_inference_ms, 2),
                total_ms=round((perf_counter() - started) * 1000, 2),
                image=ImageInfo(filename=filename, width=image.width, height=image.height),
                statistics=statistics, detections=detections,
                original_image=original, annotated_image=annotated,
            )
        finally:
            self.lock.release()
