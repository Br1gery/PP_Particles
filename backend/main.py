import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.config import ROOT, Settings
from backend.inference import Detector, InvalidImage, ModelBusy, ModelUnavailable
from backend.schemas import Health, Prediction
from backend.report_schemas import ReportRequest

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    settings = settings or Settings()
    detector = detector or Detector(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(detector.load)
        yield

    app = FastAPI(
        title="PowderScope API", version="1.0.0",
        description="YOLO26s находит частицы, ResNet18 определяет покрытие. Координаты рамок — в пикселях исходного изображения после коррекции EXIF.",
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

    @app.post("/api/predict", response_model=Prediction, tags=["Detection"], responses={
        400: {"description": "Поврежденное изображение или неподдерживаемый формат"},
        413: {"description": "Файл превышает лимит"},
        429: {"description": "Модель занята"},
        503: {"description": "Модель недоступна"},
    })
    def predict(
        file: Annotated[UploadFile, File(description="JPEG, PNG, WEBP, BMP или однокадровый TIFF, до 20 МБ / 25 Мп")],
    ):
        try:
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "Размер файла превышает 20 МБ.")
            if not data:
                raise HTTPException(400, "Загружен пустой файл.")
            filename = Path((file.filename or "image").replace("\\", "/")).name
            return detector.predict(data, filename)
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

    # Mount last, so API endpoints and documentation take precedence over static files.
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()
