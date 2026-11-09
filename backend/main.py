import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from backend.config import ROOT, Settings, CONTOUR_PROCESSING_VERSION
from backend.frontend import FrontendFiles
from backend.inference import Detector, InvalidImage, ModelBusy, ModelUnavailable
from backend.schemas import Health, Prediction
from backend.report_schemas import ReportRequest
from backend.shape_analysis import ShapeAnalyzer, ShapePrediction
from backend.shape_reporting import ShapeReportRequest

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, detector: Detector | None = None,
               shape_analyzer: ShapeAnalyzer | None = None) -> FastAPI:
    settings = settings or Settings()
    detector = detector or Detector(settings)
    shape_analyzer = shape_analyzer or ShapeAnalyzer(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(detector.load)
        await run_in_threadpool(shape_analyzer.load)
        yield

    app = FastAPI(
        title="PowderScope API", version="1.1.0",
        description="Анализ покрытия (YOLO26s + ResNet18) и округлости (YOLO26 segmentation). Все координаты в пикселях после коррекции EXIF; API округлости возвращает bbox в формате xywh, API покрытия — xyxy.",
        lifespan=lifespan,
    )

    @app.get("/api/health", response_model=Health, tags=["Service"])
    def health():
        payload = Health(
            status="ready" if detector.ready else "unavailable",
            model=settings.model_path.name, device=detector.device, classes=detector.classes,
            classifier=settings.classifier_path.name, classifier_ready=detector.classifier.ready,
            classifier_confidence_threshold=settings.classifier_confidence_threshold,
            classifier_normalization=settings.classifier_normalization,
            error=detector.error, max_upload_bytes=settings.max_upload_bytes,
            max_image_pixels=settings.max_image_pixels, max_detections=settings.max_detections,
            default_image_size=settings.default_image_size,
        )
        return JSONResponse(payload.model_dump(mode="json"), status_code=200 if detector.ready else 503)

    @app.get("/api/shape/health", tags=["Shape"])
    def shape_health():
        return JSONResponse({"status": "ready" if shape_analyzer.ready else "unavailable",
                             "model": settings.shape_model_path.name, "device": shape_analyzer.device,
                             "model_sha256": getattr(shape_analyzer, "model_sha256", None),
                             "classes": getattr(shape_analyzer, "classes", {}),
                             "classification_source": getattr(shape_analyzer, "classification_source", None),
                             "inference_mode": "nms",
                             "contour_processing_version": CONTOUR_PROCESSING_VERSION,
                             "manual_shape_annotations": True,
                             "weights_changed": getattr(shape_analyzer, "weights_changed", False),
                             "circularity_threshold": settings.circularity_threshold,
                             "error": getattr(shape_analyzer, "unavailable_reason", shape_analyzer.error)}, status_code=200 if shape_analyzer.ready else 503)

    @app.post("/api/predict", response_model=Prediction, tags=["Detection"], responses={
        400: {"description": "Поврежденное изображение или неподдерживаемый формат"},
        413: {"description": "Файл превышает лимит"},
        429: {"description": "Модель занята"},
        503: {"description": "Модель недоступна"},
    })
    def predict(
        file: Annotated[UploadFile, File(description="JPEG, PNG, WEBP, BMP или однокадровый TIFF, до 20 МБ / 25 Мп")],
    ):
        return process_upload(file, detector)

    @app.post("/api/shape/predict", response_model=ShapePrediction, tags=["Shape"])
    def predict_shape(file: Annotated[UploadFile, File(description="Изображение частиц для анализа округлости")]):
        return process_upload(file, shape_analyzer)

    def process_upload(file, analyzer):
        try:
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "Размер файла превышает 20 МБ.")
            if not data:
                raise HTTPException(400, "Загружен пустой файл.")
            filename = Path((file.filename or "image").replace("\\", "/")).name
            return analyzer.predict(data, filename)
        except InvalidImage as exc:
            raise HTTPException(400, str(exc)) from exc
        except ModelUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ModelBusy as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(500, "Не удалось распознать изображение. Попробуйте снова; подробности в журнале сервера.") from exc
        finally:
            file.file.close()

    @app.post("/api/report", tags=["Report"], response_class=Response,
              responses={200: {"content": {"application/pdf": {}}}, 400: {"description": "Некорректные данные отчёта"}})
    def report(payload: ReportRequest):
        from backend.reporting import build_report

        try:
            pdf = build_report(payload, settings)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("PDF report failed")
            raise HTTPException(500, "Не удалось сформировать PDF-отчёт. Подробности в журнале сервера.") from exc
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="powderscope-report.pdf"'})

    @app.post("/api/shape/report", tags=["Shape"], response_class=Response,
              responses={200: {"content": {"application/pdf": {}}}})
    def shape_report(payload: ShapeReportRequest):
        from backend.shape_reporting import build_shape_report

        try:
            pdf = build_shape_report(payload, settings)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Shape PDF report failed")
            raise HTTPException(500, "Не удалось сформировать отчёт округлости.") from exc
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="powderscope-shape-report.pdf"'})

    # Mount last, so API endpoints and documentation take precedence over static files.
    app.mount("/", FrontendFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()
