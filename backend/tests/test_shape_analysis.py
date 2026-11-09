import base64
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import cv2
from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.config import Settings, CONTOUR_PROCESSING_VERSION
from backend.main import create_app
from backend.shape_analysis import ShapeAnalyzer, measure_particle, summarize_shapes
from backend.tests.test_api import BoxData, MemoryDetector, image_bytes


def circle(cx=30, cy=30, radius=15):
    angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    return np.column_stack((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))


class MemoryShapeAnalyzer(ShapeAnalyzer):
    def __init__(self, settings):
        super().__init__(settings)
        self.loads = 0
        self.rows = [[10, 10, 50, 50, .8, 0], [11, 11, 51, 51, .9, 0], [70, 10, 110, 25, .6, 0]]
        self.polygons = [circle(radius=10), circle(), np.array([[70, 10], [110, 10], [110, 25], [70, 25]])]

    def load(self):
        self.loads += 1
        self.device = "cpu"
        self.classes = {0: "particle"}
        self.model = SimpleNamespace(predict=self.run)

    def run(self, **kwargs):
        self.arguments = kwargs
        image = kwargs["source"]
        masks = np.zeros((len(self.polygons), image.height, image.width), dtype=np.uint8)
        for mask, polygon in zip(masks, self.polygons):
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        return [SimpleNamespace(boxes=SimpleNamespace(data=BoxData(self.rows)),
                                masks=SimpleNamespace(data=masks) if self.rows else None)]


class GeometryTests(unittest.TestCase):
    def test_trained_class_is_preserved_independently_of_circularity(self):
        rounded = measure_particle([[10, 10], [50, 10], [50, 20], [10, 20]], 1, .9, 120, 80,
                                   class_id=0, class_name="rounded")
        irregular = measure_particle(circle(), 2, .8, 120, 80, class_id=1, class_name="irregular")
        self.assertLess(rounded.circularity, .8)
        self.assertEqual(rounded.form, "rounded")
        self.assertEqual(rounded.classification_source, "model")
        self.assertGreater(irregular.circularity, .8)
        self.assertEqual(irregular.form, "irregular")
        self.assertEqual(irregular.class_id, 1)
        edge = measure_particle(circle(cx=10), 3, .9, 120, 80, class_id=0, class_name="rounded")
        self.assertEqual(edge.form, "edge")
        self.assertEqual(edge.class_name, "rounded")

    def test_circle_measures_geometry_and_square_is_not_rounded(self):
        p = measure_particle(circle(), 1, .6, 120, 80)
        self.assertEqual(p.form, "rounded")
        self.assertAlmostEqual(p.circularity, 1, delta=.001)
        self.assertAlmostEqual(p.area_px2, np.pi * 15**2, delta=1)
        self.assertAlmostEqual(p.equivalent_diameter_px, 30, delta=.02)
        self.assertAlmostEqual(p.centroid_x, 30, places=3)
        square = measure_particle([[10, 10], [30, 10], [30, 30], [10, 30]], 2, .99, 120, 80)
        self.assertEqual(square.form, "irregular")
        self.assertAlmostEqual(square.circularity, np.pi / 4, places=5)

    def test_edges_excluded_from_percent_and_medians(self):
        rounded = measure_particle(circle(), 1, .9, 120, 80)
        edge = measure_particle(circle(cx=10), 2, .9, 120, 80)
        summary = summarize_shapes([rounded, edge])
        self.assertEqual(edge.form, "edge")
        self.assertEqual(summary.rounded_percent, 100)
        self.assertEqual(summary.shape_denominator, 1)
        self.assertIsNone(summarize_shapes([edge]).median_circularity)
        self.assertIsNone(summarize_shapes([]).rounded_percent)

    def test_degenerate_nonfinite_and_tiny_contours(self):
        for points in ([], [[1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]], [[1, 1], [float("nan"), 2], [3, 3]]):
            self.assertIsNone(measure_particle(points, 1, .9, 120, 80))


class ShapeApiTests(unittest.TestCase):
    def test_two_class_weights_use_yolo_classes_through_nms(self):
        self.shape.classes = {0: "rounded", 1: "irregular"}
        # Highest-confidence duplicate belongs to irregular; its circularity is near 1.
        self.shape.rows[1][5] = 1
        result = self.post().json()
        self.assertEqual(result["classification_source"], "model")
        self.assertEqual(result["classes"], {"0": "rounded", "1": "irregular"})
        first, second = result["particles"]
        self.assertEqual((first["class_id"], first["class_name"], first["form"]), (1, "irregular", "irregular"))
        self.assertEqual(first["confidence"], .9)
        self.assertGreater(first["circularity"], .8)
        self.assertEqual(second["form"], "rounded")
        self.assertLess(second["circularity"], .8)
        self.assertEqual(result["summary"]["rounded_percent"], 50)

    def setUp(self):
        self.settings = Settings()
        self.shape = MemoryShapeAnalyzer(self.settings)
        self.detector = MemoryDetector(self.settings)
        self.client = TestClient(create_app(self.settings, self.detector, self.shape))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, content=None):
        return self.client.post("/api/shape/predict", files={"file": ("sample.png", image_bytes() if content is None else content)})

    def test_fixed_defaults_nms_keeps_mask_confidence_pair_and_load_once(self):
        for _ in range(2):
            response = self.post()
            self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(self.shape.loads, 1)
        self.assertEqual(self.shape.arguments["imgsz"], 640)
        self.assertEqual(self.shape.arguments["conf"], .06)
        self.assertIs(self.shape.arguments["end2end"], False)
        self.assertIs(self.shape.arguments["retina_masks"], True)
        self.assertEqual(result["inference_mode"], "nms")
        self.assertEqual(result["summary"]["total"], 2)
        p = result["particles"][0]
        self.assertEqual(p["confidence"], .9)
        self.assertEqual(p["bbox"], [15, 15, 31, 31])
        # Raster boundaries differ slightly from the analytic training fixture.
        self.assertAlmostEqual(p["area_px2"], np.pi * 225, delta=25)
        self.assertEqual(result["summary"]["rounded_percent"], 50)
        self.assertFalse(result["image"]["footer_cropped"])
        self.assertEqual(result["contour_processing_version"], CONTOUR_PROCESSING_VERSION)
        self.assertEqual(self.client.get("/api/shape/health").json()["contour_processing_version"], CONTOUR_PROCESSING_VERSION)

    def test_mask_cleanup_receives_only_nms_survivors_with_their_own_classes(self):
        from backend.contours import particle_contours
        with patch("backend.contours.particle_contours", wraps=particle_contours) as cleanup:
            response = self.post()
        self.assertEqual(response.status_code, 200)
        args, kwargs = cleanup.call_args
        self.assertEqual(len(args[0]), 2)
        self.assertEqual(kwargs["class_names"], ["particle", "particle"])
        self.assertEqual(kwargs["confidences"], [.9, .6])
        self.assertTrue(kwargs["resolve_overlaps"])

    def test_independent_health_unavailable_busy_and_invalid_uploads(self):
        self.detector.classifier.ready = False
        self.assertEqual(self.client.get("/api/health").status_code, 503)
        self.assertEqual(self.client.get("/api/shape/health").status_code, 200)
        self.assertEqual(self.post().status_code, 200)
        for data in (b"", b"bad image"):
            self.assertEqual(self.post(data).status_code, 400)
        self.assertEqual(self.client.post("/api/shape/predict").status_code, 422)
        self.shape.lock.acquire()
        try:
            self.assertEqual(self.post().status_code, 429)
        finally:
            self.shape.lock.release()
        self.shape.model = None
        self.assertEqual(self.client.get("/api/shape/health").status_code, 503)
        self.assertEqual(self.post().status_code, 503)

    def test_zero_detections_and_candidate_limit(self):
        self.shape.settings = replace(self.settings, max_detections=3)
        self.assertTrue(self.post().json()["detection_limit_reached"])
        self.shape.rows = self.shape.polygons = []
        result = self.post().json()
        self.assertEqual(result["particles"], [])
        self.assertIsNone(result["summary"]["rounded_percent"])

    def test_disconnected_mask_never_draws_a_bridge_to_its_noise_island(self):
        mask = np.zeros((1, 80, 120), dtype=np.uint8)
        cv2.circle(mask[0], (30, 30), 12, 1, -1)
        cv2.circle(mask[0], (100, 60), 3, 1, -1)
        # The old xy field would draw a diagonal; the API must use the binary mask.
        result = SimpleNamespace(boxes=SimpleNamespace(data=BoxData([[10, 10, 110, 70, .9, 0]])),
                                 masks=SimpleNamespace(data=mask, xy=[[[20, 20], [100, 60], [40, 40]]]))
        with patch.object(self.shape, "run", return_value=[result]):
            self.shape.model = SimpleNamespace(predict=self.shape.run)
            response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        particle = response.json()["particles"][0]
        self.assertEqual(particle["confidence"], .9)
        self.assertEqual(particle["class_name"], "particle")
        self.assertEqual(particle["bbox"], [18, 18, 25, 25])
        self.assertLess(max(x for x, _ in particle["polygon_px"]), 50)
        self.assertEqual(response.json()["summary"]["total"], 1)

    def test_attached_lobe_cleanup_keeps_yolo_class_and_original_contour(self):
        mask = np.zeros((1, 80, 120), dtype=np.uint8)
        cv2.circle(mask[0], (35, 35), 20, 1, -1)
        cv2.line(mask[0], (53, 35), (78, 35), 1, 1)
        cv2.rectangle(mask[0], (76, 31), (84, 39), 1, -1)
        self.shape.classes = {0: "rounded", 1: "irregular"}
        result = SimpleNamespace(boxes=SimpleNamespace(data=BoxData([[10, 10, 90, 60, .9, 0]])), masks=SimpleNamespace(data=mask))
        with patch.object(self.shape, "run", return_value=[result]):
            self.shape.model = SimpleNamespace(predict=self.shape.run)
            response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        particle = response.json()["particles"][0]
        self.assertEqual((particle["class_name"], particle["form"], particle["confidence"]), ("rounded", "rounded", .9))
        self.assertTrue(particle["contour_refined"])
        self.assertGreater(particle["removed_area_percent"], 0)
        self.assertLessEqual(particle["removed_area_percent"], 12)
        self.assertGreaterEqual(max(x for x, _ in particle["raw_polygon_px"]), 84)
        self.assertLess(max(x for x, _ in particle["polygon_px"]), 60)
        self.assertEqual(response.json()["summary"]["total"], 1)

    def report_payload(self):
        result = self.post().json()
        return {"image": result["image"], "original_image": result["original_image"], "particles": result["particles"], "timezone_offset_minutes": -300}

    def test_report_recomputes_geometry_and_respects_deletions(self):
        payload = self.report_payload()
        payload["particles"] = payload["particles"][:1]
        payload["particles"][0].update(circularity=.1, form="irregular")
        response = self.client.post("/api/shape/report", json=payload)
        self.assertEqual(response.status_code, 200, response.text[:100] if response.status_code != 200 else "")
        reader = PdfReader(BytesIO(response.content))
        text = " ".join(" ".join(p.extract_text().split()) for p in reader.pages)
        self.assertIn("Доля округлых: 100.0%", text)
        self.assertIn("Всего: 1", text)
        self.assertIn("Уверенность YOLO", text)
        self.assertIn("90.0%", text)

    def test_report_rejects_duplicate_outside_and_invalid_contours(self):
        payload = self.report_payload()
        payload["particles"].append(payload["particles"][0])
        self.assertEqual(self.client.post("/api/shape/report", json=payload).status_code, 422)
        payload = self.report_payload()
        payload["particles"][0]["polygon_px"][0] = [-1, 0]
        self.assertEqual(self.client.post("/api/shape/report", json=payload).status_code, 422)
        payload["particles"][0]["polygon_px"] = [[5, 5], [6, 6], [7, 7]]
        self.assertEqual(self.client.post("/api/shape/report", json=payload).status_code, 400)
        payload = self.report_payload()
        payload["particles"] = []
        self.assertEqual(self.client.post("/api/shape/report", json=payload).status_code, 200)

    def test_report_preserves_yolo_class_instead_of_reclassifying_by_geometry(self):
        self.shape.classes = {0: "rounded", 1: "irregular"}
        self.shape.rows[1][5] = 1
        payload = self.report_payload()
        payload["particles"] = payload["particles"][:1]
        payload.update(classification_source="model", model_sha256="a" * 64)
        response = self.client.post("/api/shape/report", json=payload)
        self.assertEqual(response.status_code, 200)
        text = " ".join(" ".join(p.extract_text().split()) for p in PdfReader(BytesIO(response.content)).pages)
        self.assertIn("Доля округлых: 0.0%", text)
        self.assertIn("Форма определяется обученной YOLO", text)
        self.assertIn("Неокруглая", text)
        self.assertIn("Версия весов (SHA256)", text)

    def test_report_identifies_contour_refinements(self):
        payload = self.report_payload()
        payload["particles"][0].update(contour_refined=True, removed_area_percent=3)
        response = self.client.post("/api/shape/report", json=payload)
        self.assertEqual(response.status_code, 200)
        text = " ".join(" ".join(p.extract_text().split()) for p in PdfReader(BytesIO(response.content)).pages)
        self.assertIn("Очищено контуров: 1", text)
        self.assertIn("отображаемому контуру", text)
        payload["particles"][0]["removed_area_percent"] = 25
        self.assertEqual(self.client.post("/api/shape/report", json=payload).status_code, 422)

    def manual_report_payload(self):
        payload=self.report_payload()
        payload["particles"]=payload["particles"][:1]
        payload["particles"][0].update(source="model",manual_form="irregular")
        payload["particles"].append(dict(index=3,source="manual",confidence=None,class_id=None,class_name=None,
                                         manual_form="rounded",manual_contour=True,
                                         polygon_px=[[75,45],[105,45],[105,65],[75,65]]))
        return payload

    def test_report_includes_manual_shape_and_class_override_without_fake_confidence(self):
        self.assertTrue(self.client.get("/api/shape/health").json()["manual_shape_annotations"])
        response=self.client.post("/api/shape/report",json=self.manual_report_payload())
        self.assertEqual(response.status_code,200)
        text=" ".join(" ".join(p.extract_text().split()) for p in PdfReader(BytesIO(response.content)).pages)
        for expected in ("Всего: 2", "Доля округлых: 50.0%", "Ручных выделений: 1", "Ручных классов: 1", "Вручную", "Ручной класс", "90.0%"):
            self.assertIn(expected,text)

    def test_manual_shape_report_rejects_fabricated_predictions_and_invalid_outlines(self):
        for update in (
            dict(confidence=1), dict(class_name="rounded"), dict(manual_form=None),
            dict(polygon_px=[[75,45],[105,65],[75,65],[105,45]]),
            dict(polygon_px=[[110,40],[74,52],[96,21],[96,59],[74,28]]),
            dict(polygon_px=[[5,5],[6,6],[7,7]]), dict(polygon_px=[[75,45],[160,45],[105,65]]),
        ):
            payload=self.manual_report_payload();payload["particles"][-1].update(update)
            self.assertEqual(self.client.post("/api/shape/report",json=payload).status_code,422,update)

    def test_report_uses_edited_geometry_but_preserves_original_yolo_confidence(self):
        payload=self.report_payload();payload["particles"]=payload["particles"][:1]
        payload["particles"][0].update(source="model",manual_contour=True,manual_form="rounded",contour_refined=False,
                                        removed_area_percent=0,polygon_px=[[10,10],[50,10],[50,30],[10,30]])
        response=self.client.post("/api/shape/report",json=payload)
        self.assertEqual(response.status_code,200)
        text=" ".join(" ".join(p.extract_text().split()) for p in PdfReader(BytesIO(response.content)).pages)
        self.assertIn("800.00",text);self.assertIn("90.0%",text)
        self.assertIn("Исправленных контуров модели: 1",text)


class WeightLoadingTests(unittest.TestCase):
    def test_configured_weights_loaded_with_their_classes_and_fingerprint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "particle_yolo26.pt"
            path.write_bytes(b"test-only mock weights")
            settings = replace(Settings(), shape_model_path=path, device="cpu")
            model = SimpleNamespace(task="segment", names={0: "rounded", 1: "irregular"}, to=lambda device: None)
            with patch("ultralytics.YOLO", return_value=model) as loader:
                analyzer = ShapeAnalyzer(settings)
                analyzer.load()
            loader.assert_called_once_with(str(path))
            self.assertTrue(analyzer.ready)
            self.assertEqual(analyzer.model_sha256, sha256(path.read_bytes()).hexdigest())
            self.assertEqual(analyzer.classification_source, "model")
            path.write_bytes(b"new mock weights after server startup")
            self.assertTrue(analyzer.weights_changed)
            self.assertFalse(analyzer.ready)
            self.assertIn("Перезапустите", analyzer.unavailable_reason)

    def test_single_particle_checkpoint_retains_geometric_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "one-class.pt"
            path.write_bytes(b"mock")
            analyzer = ShapeAnalyzer(replace(Settings(), shape_model_path=path, device="cpu"))
            model = SimpleNamespace(task="segment", names={0: "particle"}, to=lambda device: None)
            with patch("ultralytics.YOLO", return_value=model):
                analyzer.load()
            self.assertTrue(analyzer.ready)
            self.assertEqual(analyzer.classification_source, "circularity")


if __name__ == "__main__":
    unittest.main()
