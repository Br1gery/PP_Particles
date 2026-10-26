"""Generate a self-contained PDF of the current, manually reviewed analysis."""

import base64
import binascii
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from xml.sax.saxutils import escape

from PIL import ImageDraw

from backend.classifier import COATING_COLORS
from backend.config import Settings
from backend.inference import InvalidImage, decode_image
from backend.report_schemas import ReportRequest

FONT_LOCK = Lock()
LABELS = {"coated": "Покрыта", "uncoated": "Непокрыта", "uncertain": "Спорная"}


def decode_report_image(payload: ReportRequest, settings: Settings):
    try:
        prefix, content = payload.original_image.split(",", 1)
        if prefix not in ("data:image/jpeg;base64", "data:image/png;base64"):
            raise ValueError("Unsupported report image")
        data = base64.b64decode(content, validate=True)
        if len(data) > settings.max_upload_bytes * 2:
            raise ValueError("Report image is too large")
        image = decode_image(data, settings.max_image_pixels)
    except (ValueError, binascii.Error, InvalidImage) as exc:
        raise ValueError("Не удалось прочитать изображение для отчёта.") from exc
    if image.size != (payload.image.width, payload.image.height):
        raise ValueError("Размер изображения не соответствует данным отчёта.")
    if any(p.bbox[2] > image.width or p.bbox[3] > image.height for p in payload.detections):
        raise ValueError("Одна из рамок выходит за границы изображения.")
    return image


def build_report(payload: ReportRequest, settings: Settings) -> bytes:
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    image = decode_report_image(payload, settings)
    with FONT_LOCK:
        if "PowderSans" not in pdfmetrics.getRegisteredFontNames():
            font_dir = Path(get_data_path()) / "fonts" / "ttf"
            pdfmetrics.registerFont(TTFont("PowderSans", str(font_dir / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("PowderSansBold", str(font_dir / "DejaVuSans-Bold.ttf")))

    normal = ParagraphStyle("Body", fontName="PowderSans", fontSize=9, leading=13, textColor=colors.HexColor("#20352f"), spaceAfter=6)
    title = ParagraphStyle("Title", parent=normal, fontName="PowderSansBold", fontSize=21, leading=27, spaceAfter=12)
    heading = ParagraphStyle("Heading", parent=normal, fontName="PowderSansBold", fontSize=14, leading=19, spaceAfter=12)
    small = ParagraphStyle("Small", parent=normal, fontSize=8, leading=12, textColor=colors.HexColor("#52645b"))
    cell = ParagraphStyle("Cell", parent=normal, fontSize=8, leading=11, spaceAfter=0)
    centered = ParagraphStyle("Centered", parent=cell, alignment=TA_CENTER)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=38, bottomMargin=38,
                            title="Отчет об анализе частиц", author="PowderScope")
    width = A4[0] - 80
    particles = payload.detections
    count = len(particles)
    coated = sum(p.coating_status == "coated" for p in particles)
    uncoated = sum(p.coating_status == "uncoated" for p in particles)
    uncertain = count - coated - uncoated
    coverage = f"{coated / count * 100:.1f}%" if count else "-"
    tz = timezone(timedelta(minutes=-payload.timezone_offset_minutes))
    timestamp = datetime.now(timezone.utc).astimezone(tz)
    text = lambda value, style=normal: Paragraph(escape(str(value)), style)
    story = [text("Отчет об анализе частиц", title), text(f"Файл: {payload.image.filename}"),
             text(f"Сформирован: {timestamp:%d.%m.%Y %H:%M} (UTC{timestamp:%z})", small),
             text(f"Изображение: {image.width} × {image.height} px", small),
             text(f"Детектор: {payload.model} / классификатор: {payload.classifier}", small), Spacer(1, 10)]
    summary = Table([
        [text("Покрытые", cell), text("Непокрытые", cell), text("Спорные", cell), text("Покрытие", cell)],
        [text(coated, heading), text(uncoated, heading), text(uncertain, heading), text(coverage, heading)],
    ], colWidths=[width / 4] * 4)
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f6f3")),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#d9e3dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("LINEABOVE", (0, 0), (0, 0), 2, colors.HexColor(COATING_COLORS["coated"])),
        ("LINEABOVE", (1, 0), (1, 0), 2, colors.HexColor(COATING_COLORS["uncoated"])),
        ("LINEABOVE", (2, 0), (2, 0), 2, colors.HexColor(COATING_COLORS["uncertain"])),
    ]))
    story += [summary, Spacer(1, 12), text(f"Всего оставлено: {count}. Найдено моделью: {payload.initial_detected_count}. "
               f"Добавлено вручную: {payload.manual_added_count}. Удалено: {payload.deleted_count}.", small)]
    reviewed = sum(p.source == "model" and p.manual_review is not None for p in particles)
    story.append(text(f"Ручных уточнений класса для частиц модели: {reviewed}. "
                      f"Порог спорных: {payload.classifier_confidence_threshold:.0%}.", small))
    image.thumbnail((1800, 1800))
    sx, sy = image.width / payload.image.width, image.height / payload.image.height
    draw = ImageDraw.Draw(image)
    for particle in particles:
        x1, y1, x2, y2 = particle.bbox
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=COATING_COLORS[particle.coating_status],
                       width=max(2, round(min(image.size) / 400)))
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=95)
    image_buffer.seek(0)
    ratio = min(width / image.width, 330 / image.height)
    story += [Spacer(1, 8), Image(image_buffer, width=image.width * ratio, height=image.height * ratio), Spacer(1, 12),
              text("Зеленые рамки - покрытые; желтые - спорные; бледно-красные - непокрытые или с недостаточным покрытием.", small),
              text("Покрытие - доля покрытых частиц среди всех оставшихся, с учетом ручных решений. Это не процент площади белого слоя.", small),
              text("Отчет включает все оставшиеся выделения. Фильтры видимости на странице на состав отчета не влияют. "
                   "Ручным рамкам уверенность модели не присваивается.", small)]
    if particles:
        story += [PageBreak(), text("Список частиц", heading)]
        table_rows = [[text(label, cell) for label in ("№", "Класс", "Источник", "Уверенность модели", "Рамка и размеры, px")]]
        for particle in particles:
            x1, y1, x2, y2 = particle.bbox
            origin = "Ручная рамка" if particle.source == "manual" else "Ручной класс" if particle.manual_review else "Модель"
            confidence = f"{particle.classifier_confidence:.1%}" if particle.classifier_confidence is not None else "-"
            coordinates = f"({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})\n{x2-x1:.2f} × {y2-y1:.2f}"
            table_rows.append([text(particle.id, centered), text(LABELS[particle.coating_status], cell), text(origin, cell),
                               text(confidence, centered), Paragraph(escape(coordinates).replace("\n", "<br/>"), cell)])
        table = LongTable(table_rows, colWidths=[28, 76, 79, 83, width - 266], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7efea")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9f8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#bfd1c4")),
        ]))
        story.append(table)
    else:
        story += [Spacer(1, 12), text("Активных выделений нет.", normal)]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("PowderSans", 8)
        canvas.setFillColor(colors.HexColor("#68786f"))
        canvas.drawString(40, 20, "PowderScope / Анализ порошковых частиц")
        canvas.drawRightString(A4[0] - 40, 20, f"Страница {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
