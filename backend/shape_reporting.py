"""PDF snapshot; shape measurements are recomputed from submitted contours."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, FiniteFloat, model_validator

from backend.config import Settings
from backend.report_schemas import ReportImage
from backend.reporting import FONT_LOCK, decode_report_source
from backend.shape_analysis import SHAPE_COLORS, annotate_shapes, measure_particle, summarize_shapes
from backend.manual_geometry import validate_manual_polygon

LABELS = {"rounded": "Округлая", "irregular": "Неокруглая", "edge": "На границе"}


class ShapeReportParticle(BaseModel):
    index: int = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    class_id: int | None = Field(default=0, ge=0)
    class_name: Literal["particle", "rounded", "irregular"] | None = "particle"
    source: Literal["model", "manual"] = "model"
    manual_form: Literal["rounded", "irregular"] | None = None
    manual_contour: bool = False
    polygon_px: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=3, max_length=10000)
    contour_refined: bool = False
    removed_area_percent: float = Field(default=0, ge=0, le=12, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_refinement(self):
        if self.source == "manual":
            if self.confidence is not None or self.class_id is not None or self.class_name is not None:
                raise ValueError("Ручной частице нельзя присваивать уверенность или класс модели.")
            if self.manual_form is None or not self.manual_contour:
                raise ValueError("Укажите ручной контур и форму частицы.")
            if self.contour_refined or self.removed_area_percent:
                raise ValueError("Ручной контур не проходит автоматическую очистку маски.")
        elif self.confidence is None or self.class_id is None or self.class_name is None:
            raise ValueError("У частицы модели должны сохраняться исходные класс и уверенность.")
        if self.source == "manual" or self.manual_contour:
            self.polygon_px = validate_manual_polygon(self.polygon_px)
        if not self.contour_refined and self.removed_area_percent > 0:
            raise ValueError("Признак очистки контура не соответствует удалённой площади.")
        return self


class ShapeReportRequest(BaseModel):
    contour_processing_version: str | None = Field(default=None, max_length=50)
    model: str = Field(default="particle_yolo26.pt", max_length=255)
    model_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    classification_source: Literal["model", "circularity"] = "circularity"
    image: ReportImage
    original_image: str = Field(max_length=48 * 1024 * 1024)
    particles: list[ShapeReportParticle] = Field(max_length=5000)
    timezone_offset_minutes: int = Field(default=0, ge=-840, le=840)

    @model_validator(mode="after")
    def validate_contours(self):
        if len({p.index for p in self.particles}) != len(self.particles):
            raise ValueError("Повторяются номера частиц.")
        if sum(len(p.polygon_px) for p in self.particles) > 250000:
            raise ValueError("Слишком много точек контуров.")
        if any(x < 0 or y < 0 or x > self.image.width - 1 or y > self.image.height - 1
               for p in self.particles for x, y in p.polygon_px):
            raise ValueError("Контур выходит за границы изображения.")
        return self


def build_shape_report(payload: ShapeReportRequest, settings: Settings) -> bytes:
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    image = decode_report_source(payload.image, payload.original_image, settings)
    particles = []
    for p in payload.particles:
        measured = measure_particle(p.polygon_px, p.index, p.confidence, image.width, image.height, settings.circularity_threshold,
                                    class_id=p.class_id, class_name=p.class_name, source=p.source,
                                    manual_form=p.manual_form, manual_contour=p.manual_contour)
        if measured is None:
            raise ValueError("Один из контуров имеет нулевую площадь или периметр.")
        particles.append(measured)
    summary = summarize_shapes(particles)
    with FONT_LOCK:
        if "PowderSans" not in pdfmetrics.getRegisteredFontNames():
            fonts = Path(get_data_path()) / "fonts" / "ttf"
            pdfmetrics.registerFont(TTFont("PowderSans", str(fonts / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("PowderSansBold", str(fonts / "DejaVuSans-Bold.ttf")))
    normal = ParagraphStyle("ShapeBody", fontName="PowderSans", fontSize=9, leading=14, spaceAfter=8, textColor=colors.HexColor("#20352f"))
    title = ParagraphStyle("ShapeTitle", parent=normal, fontName="PowderSansBold", fontSize=20, leading=26, spaceAfter=14)
    cell = ParagraphStyle("ShapeCell", parent=normal, fontSize=7, leading=10, spaceAfter=0)
    text = lambda value, style=normal: Paragraph(escape(str(value)), style)
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
                            title="Анализ округлости частиц", author="PowderScope")
    width = A4[0] - 72
    timestamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(minutes=-payload.timezone_offset_minutes)))
    rounded_percent = f"{summary.rounded_percent:.1f}%" if summary.rounded_percent is not None else "-"
    circularity = f"{summary.median_circularity:.3f}" if summary.median_circularity is not None else "-"
    diameter = f"{summary.median_equivalent_diameter_px:.2f} px" if summary.median_equivalent_diameter_px is not None else "-"
    story = [text("Анализ округлости частиц", title), text(f"Файл: {payload.image.filename}"),
             text(f"Сформирован: {timestamp:%d.%m.%Y %H:%M} (UTC{timestamp:%z})"),
             text(f"Модель: {payload.model}. Изображение: {image.width} × {image.height} px."),
             text(f"Обработка контуров: {payload.contour_processing_version or 'версия не указана'}."),
             text(f"Всего: {summary.total}. Округлые: {summary.rounded}. Неокруглые: {summary.irregular}. На границе: {summary.edge}."),
             text(f"Доля округлых: {rounded_percent} среди {summary.shape_denominator} частиц вне границы."),
             text(f"Ручных выделений: {sum(p.source == 'manual' for p in payload.particles)}. "
                  f"Ручных классов: {sum(p.source == 'model' and p.manual_form is not None for p in payload.particles)}. "
                  f"Исправленных контуров модели: {sum(p.source == 'model' and p.manual_contour for p in payload.particles)}."),
             text(f"Медиана круглости: {circularity}. Медиана эквивалентного диаметра: {diameter}."),
             text(("Форма определяется обученной YOLO по классам rounded / irregular. "
                   "Круглость C = 4πA / P² измеряется отдельно и не меняет класс модели."
                   if payload.classification_source == "model" or any(p.classification_source == "model" for p in particles)
                   else f"Форма определяется по контуру: C = 4πA / P²; округлые при C ≥ {settings.circularity_threshold:.2f}. "
                        "Круглость не является уверенностью обнаружения.")),
             text("Зеленые контуры - округлые; бледно-красные - неокруглые; желтые - на границе. "
                  "Частицы у границы исключены из долей и медиан. Отчет включает все оставшиеся контуры независимо от фильтров."),
             text(f"Очищено контуров: {sum(p.contour_refined for p in payload.particles)}. "
                  "Уточняется пересекающаяся периферия соседних масок; у rounded также удаляются малые выступы. Предел удаления - 12% площади. "
                  "Геометрические измерения относятся к отображаемому контуру. Ручной класс имеет приоритет. "
                  "Уверенность YOLO сохраняется исходной; у ручных выделений она отсутствует.")]
    if payload.model_sha256:
        story.append(text(f"Версия весов (SHA256): {payload.model_sha256[:32]} {payload.model_sha256[32:]}", cell))
    annotated = annotate_shapes(image, particles)
    annotated.thumbnail((1800, 1800))
    buffer = BytesIO()
    annotated.save(buffer, "JPEG", quality=95)
    buffer.seek(0)
    scale = min(width / annotated.width, 320 / annotated.height)
    story += [Spacer(1, 6), Image(buffer, width=annotated.width * scale, height=annotated.height * scale)]
    if particles:
        story += [PageBreak(), text("Измерения частиц", title)]
        rows = [[text(label, cell) for label in ("№", "Форма", "Источник", "Круглость C", "Площадь, px²", "Периметр, px", "Диаметр, px", "Уверенность YOLO")]]
        for p in particles:
            origin = "Вручную" if p.source == "manual" else "Контур и класс вручную" if p.manual_contour and p.manual_form else "Ручной контур" if p.manual_contour else "Ручной класс" if p.manual_form else "YOLO"
            rows.append([text(value, cell) for value in (p.index, LABELS[p.form], origin, f"{p.circularity:.4f}", f"{p.area_px2:.2f}",
                                                        f"{p.perimeter_px:.2f}", f"{p.equivalent_diameter_px:.2f}", f"{p.confidence:.1%}" if p.confidence is not None else "-")])
        table = LongTable(rows, repeatRows=1, colWidths=[24, 66, 65, 56, 78, 78, 72, width - 439], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7efea")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9f8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(table)

    def footer(canvas, document):
        canvas.setFont("PowderSans", 8)
        canvas.drawString(36, 18, "PowderScope / Округлость частиц")
        canvas.drawRightString(A4[0] - 36, 18, f"Страница {document.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
