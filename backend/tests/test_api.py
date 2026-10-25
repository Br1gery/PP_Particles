"""API/serialization tests use an in-memory predictor, never download weights."""

import base64
import unittest
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.classifier import classify_probabilities
from backend.inference import Detector, InvalidImage, decode_image
from backend.main import create_app


def image_bytes(format="PNG", size=(120, 80)):
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=format)
    return buffer.getvalue()


class BoxData:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class MemoryDetector(Detector):
    def __init__(self, settings):
        super().__init__(settings)
        self.rows = [[10, 12, 40, 52, .9, 0], [50, 10, 70, 30, .6, 0]]
        self.arguments = None
        self.probabilities = [(.9, .1), (.1, .9)]
        self.classifier = SimpleNamespace(ready=True, classify=self.classify_crops)

    def classify_crops(self, image, boxes):
        return [classify_probabilities(*self.probabilities[i], self.settings.classifier_confidence_threshold)
                for i in range(len(boxes))]

    def load(self):
        self.device = "cpu"
        self.classes = {0: "particle"}
        self.model = self

    def predict_model(self, **kwargs):
        self.arguments = kwargs
        return [SimpleNamespace(boxes=SimpleNamespace(data=BoxData(self.rows)))]

    def predict(self, *args, **kwargs):
        # Real Detector handles preprocessing, coordinate conversion and response rendering.
        self.model = SimpleNamespace(predict=self.predict_model)
        return super().predict(*args, **kwargs)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.detector = MemoryDetector(self.settings)
        self.client = TestClient(create_app(self.settings, self.detector))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, data=None, **parameters):
        return self.client.post("/api/predict", files={"file": ("sample.png", image_bytes() if data is None else data, "image/png")}, data=parameters)

    def test_frontend_and_api_schema(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertIn("/api/predict", self.client.get("/openapi.json").json()["paths"])

    def test_health_reports_loaded_model(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["classes"], {"0": "particle"})
        self.assertTrue(response.json()["classifier_ready"])
        self.assertEqual(response.json()["classifier"], "bestClassifier.pt")

    def test_prediction_uses_fixed_defaults_and_original_coordinates(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["statistics"]["count"], 2)
        self.assertEqual(result["statistics"]["mean_confidence"], .75)
        self.assertEqual(result["statistics"]["coated_count"], 1)
        self.assertEqual(result["statistics"]["uncoated_count"], 1)
        self.assertEqual(result["statistics"]["coverage_percent"], 50)
        self.assertEqual(result["detections"][0]["coating_class"], "coated")
        self.assertEqual(result["detections"][0]["p_coated"], .9)
        self.assertEqual(result["detections"][0]["bbox"], [10, 12, 40, 52])
        self.assertEqual(result["detections"][0]["height_px"], 40)
        self.assertEqual(self.detector.arguments["imgsz"], 1024)
        self.assertEqual(self.detector.arguments["conf"], .25)
        self.assertFalse(self.detector.arguments["save"])
        for key in ("original_image", "annotated_image"):
            image = Image.open(BytesIO(base64.b64decode(result[key].split(",", 1)[1])))
            self.assertEqual(image.size, (120, 80))

    def test_zero_detections_are_valid(self):
        self.detector.rows = []
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"], [])
        self.assertIsNone(response.json()["statistics"]["mean_confidence"])
        self.assertIsNone(response.json()["statistics"]["coverage_percent"])
        self.assertIsNone(response.json()["statistics"]["mean_classifier_confidence"])
        self.assertEqual(response.json()["statistics"]["uncertain_count"], 0)

    def test_uncertain_particles_are_separate_from_covered_and_uncovered(self):
        self.detector.rows.append([75, 10, 95, 30, .8, 0])
        self.detector.probabilities = [(.9, .1), (.55, .45), (.1, .9)]
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        statistics = response.json()["statistics"]
        self.assertEqual(statistics["count"], 3)
        self.assertEqual(statistics["coated_count"], 1)
        self.assertEqual(statistics["uncoated_count"], 1)
        self.assertEqual(statistics["uncertain_count"], 1)
        self.assertEqual(statistics["coverage_percent"], 33.33)

    def test_classifier_failure_does_not_return_incomplete_results(self):
        self.detector.classifier.ready = False
        self.assertEqual(self.client.get("/api/health").status_code, 503)
        self.assertEqual(self.post().status_code, 503)

    def test_corrupt_and_empty_files(self):
        for data in (b"not an image", b""):
            with self.subTest(data=data):
                self.assertEqual(self.post(data).status_code, 400)

    def test_prediction_input_is_only_an_image(self):
        schema = self.client.get("/openapi.json").json()
        body = schema["paths"]["/api/predict"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        properties = schema["components"]["schemas"][body["$ref"].split("/")[-1]]["properties"]
        self.assertEqual(set(properties), {"file"})
        # Old clients cannot override the server's inference settings.
        response = self.post(confidence=.9, image_size=640)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.detector.arguments["conf"], .25)
        self.assertEqual(self.detector.arguments["imgsz"], 1024)

    def test_missing_file(self):
        self.assertEqual(self.client.post("/api/predict").status_code, 422)

    def test_file_limit(self):
        settings = replace(self.settings, max_upload_bytes=10)
        with TestClient(create_app(settings, self.detector)) as client:
            response = client.post("/api/predict", files={"file": ("large.png", image_bytes())})
            self.assertEqual(response.status_code, 413)

    def test_busy_model(self):
        with self.detector.lock:
            self.assertEqual(self.post().status_code, 429)

    def test_unavailable_model(self):
        unavailable = Detector(replace(self.settings, model_path=self.settings.model_path.with_name("missing.pt")))
        with self.assertLogs("backend.inference", level="ERROR"):
            with TestClient(create_app(self.settings, unavailable)) as client:
                self.assertEqual(client.get("/api/health").status_code, 503)
                self.assertEqual(client.post("/api/predict", files={"file": ("test.png", image_bytes())}).status_code, 503)

    def test_detection_limit_flag(self):
        self.detector.settings = replace(self.settings, max_detections=2)
        self.assertTrue(self.post().json()["statistics"]["detection_limit_reached"])


class ImageTests(unittest.TestCase):
    def test_pixel_limit(self):
        with self.assertRaises(InvalidImage):
            decode_image(image_bytes(), max_pixels=100)

    def test_tiff_is_normalized(self):
        image = decode_image(image_bytes("TIFF"), max_pixels=10000)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (120, 80))

    def test_exif_orientation(self):
        image = Image.new("RGB", (120, 80))
        exif = image.getexif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        self.assertEqual(decode_image(buffer.getvalue(), 10000).size, (80, 120))

    def test_animated_image_rejected(self):
        buffer = BytesIO()
        Image.new("RGB", (20, 20), "white").save(buffer, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (20, 20), "black")], duration=100, loop=0)
        with self.assertRaises(InvalidImage):
            decode_image(buffer.getvalue(), 10000)


if __name__ == "__main__":
    unittest.main()
