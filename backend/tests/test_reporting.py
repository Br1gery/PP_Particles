import base64
import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.config import Settings
from backend.main import create_app
from backend.tests.test_api import MemoryDetector, image_bytes


def report_payload():
    return {
        "image": {"filename": "порошок_образец.png", "width": 120, "height": 80},
        "original_image": "data:image/png;base64," + base64.b64encode(image_bytes()).decode(),
        "initial_detected_count": 3, "manual_added_count": 1, "deleted_count": 1,
        "model": "best_yolo26.pt", "classifier": "bestClassifier.pt",
        "classifier_confidence_threshold": .7, "timezone_offset_minutes": -300,
        "detections": [
            {"id": 1, "bbox": [10, 10, 35, 50], "coating_status": "coated", "source": "model", "manual_review": "coated", "classifier_confidence": .6},
            {"id": 2, "bbox": [40, 10, 65, 50], "coating_status": "uncoated", "source": "model", "classifier_confidence": .9},
            {"id": 4, "bbox": [75, 10, 100, 50], "coating_status": "coated", "source": "manual"},
        ],
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.detector = MemoryDetector(self.settings)
        self.client = TestClient(create_app(self.settings, self.detector))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_pdf_contains_reviewed_counts_cyrillic_and_annotation_sources(self):
        response = self.client.post("/api/report", json=report_payload())
        self.assertEqual(response.status_code, 200, response.text[:200] if response.status_code != 200 else "")
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn("attachment", response.headers["content-disposition"])
        reader = PdfReader(BytesIO(response.content))
        self.assertEqual(len(reader.pages), 2)
        text = " ".join(" ".join(page.extract_text().split()) for page in reader.pages)
        for value in ("порошок_образец.png", "Всего оставлено: 3", "Добавлено вручную: 1", "Удалено: 1", "66.7%", "Ручная рамка", "Ручной класс", "60.0%", "+0500"):
            self.assertIn(value, text)

    def test_all_deleted_report_has_one_page_and_no_nan_percentage(self):
        payload = report_payload()
        payload.update(detections=[], deleted_count=4)
        response = self.client.post("/api/report", json=payload)
        self.assertEqual(response.status_code, 200)
        reader = PdfReader(BytesIO(response.content))
        self.assertEqual(len(reader.pages), 1)
        self.assertIn("Активных выделений нет", reader.pages[0].extract_text())

    def test_large_table_repeats_headers_and_keeps_last_particle(self):
        payload = report_payload()
        payload.update(initial_detected_count=100, manual_added_count=0, deleted_count=0)
        payload["detections"] = [{"id": i, "bbox": [10, 10, 35, 50], "coating_status": "coated", "source": "model", "classifier_confidence": .9} for i in range(1, 101)]
        response = self.client.post("/api/report", json=payload)
        self.assertEqual(response.status_code, 200)
        reader = PdfReader(BytesIO(response.content))
        self.assertGreater(len(reader.pages), 2)
        for page in reader.pages[1:]:
            self.assertIn("Класс", page.extract_text())
        self.assertIn("100", reader.pages[-1].extract_text())

    def test_report_works_without_loaded_detection_models(self):
        self.detector.classifier.ready = False
        self.assertEqual(self.client.get("/api/health").status_code, 503)
        self.assertEqual(self.client.post("/api/report", json=report_payload()).status_code, 200)

    def test_reject_invalid_report_image_or_out_of_bounds_box(self):
        for mutate in (
            lambda p: p.update(original_image="data:image/png;base64,broken"),
            lambda p: p["image"].update(width=999),
            lambda p: p["detections"][0].update(bbox=[10, 10, 500, 50]),
        ):
            payload = report_payload()
            mutate(payload)
            self.assertEqual(self.client.post("/api/report", json=payload).status_code, 400)

    def test_reject_duplicate_ids_inconsistent_counts_and_fake_manual_confidence(self):
        for mutate in (
            lambda p: p["detections"][2].update(id=1),
            lambda p: p.update(deleted_count=0),
            lambda p: p["detections"][2].update(classifier_confidence=1),
            lambda p: p["detections"][0].update(bbox=[10, 10, 10, 50]),
        ):
            payload = report_payload()
            mutate(payload)
            self.assertEqual(self.client.post("/api/report", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
