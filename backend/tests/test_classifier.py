import unittest
from dataclasses import replace

from PIL import Image, ImageDraw

from backend.classifier import CoverageClassifier, COATING_COLORS, classify_probabilities, crop_particle, pad_crop
from backend.config import Settings
from backend.inference import annotate
from backend.schemas import Detection


class ClassificationTests(unittest.TestCase):
    def test_argmax_and_uncertainty_threshold(self):
        self.assertEqual(classify_probabilities(.9, .1, .7).coating_status, "coated")
        uncovered = classify_probabilities(.1, .9, .7)
        self.assertEqual(uncovered.coating_class, "insufficient_coating")
        self.assertEqual(uncovered.coating_status, "uncoated")
        uncertain = classify_probabilities(.6, .4, .7)
        self.assertEqual(uncertain.coating_class, "coated")
        self.assertEqual(uncertain.coating_status, "uncertain")
        self.assertEqual(classify_probabilities(.7, .3, .7).coating_status, "coated")
        self.assertEqual(classify_probabilities(.5, .5, .7).coating_status, "uncertain")

    def test_crop_adds_context_and_stays_inside_image(self):
        image = Image.new("RGB", (100, 80))
        self.assertEqual(crop_particle(image, (10, 20, 60, 50), .1).size, (60, 36))
        self.assertEqual(crop_particle(image, (0, 0, 100, 80), .1).size, image.size)
        with self.assertRaises(ValueError):
            crop_particle(image, (20, 20, 20, 30), .1)

    def test_pad_preserves_aspect_ratio_and_uses_median_border_color(self):
        image = Image.new("RGB", (100, 50), (40, 50, 60))
        ImageDraw.Draw(image).rectangle((1, 1, 98, 48), fill="blue")
        padded = pad_crop(image)
        self.assertEqual(padded.size, (320, 320))
        self.assertEqual(padded.mode, "RGB")
        self.assertEqual(padded.getpixel((160, 0)), (40, 50, 60))
        self.assertEqual(padded.getpixel((160, 70)), (40, 50, 60))
        self.assertEqual(padded.getpixel((160, 160)), (0, 0, 255))
        self.assertEqual(padded.getpixel((160, 250)), (40, 50, 60))

    def test_annotation_colors_match_the_classification_without_labels(self):
        original = Image.new("RGB", (150, 100), "black")
        detections = []
        for index, probabilities in enumerate(((.9, .1), (.55, .45), (.1, .9))):
            detections.append(Detection(
                id=index + 1, class_id=0, class_name="particle", confidence=.95,
                bbox=(10 + index * 45, 20, 40 + index * 45, 70), width_px=30, height_px=50,
                **classify_probabilities(*probabilities, .7).model_dump(),
            ))
        result = annotate(original, detections)
        for detection in detections:
            color = COATING_COLORS[detection.coating_status]
            rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            self.assertEqual(result.getpixel((int(detection.bbox[0]), 20)), rgb)
            self.assertEqual(result.getpixel((int(detection.bbox[0]) + 5, 25)), (0, 0, 0))
        self.assertEqual(original.getpixel((10, 20)), (0, 0, 0))

    def test_classifier_batches_logits_and_preserves_particle_order(self):
        import torch

        settings = replace(Settings(), classifier_batch_size=2)
        classifier = CoverageClassifier(settings)
        batch_sizes = []

        def model(inputs):
            batch_sizes.append(tuple(inputs.shape))
            return torch.tensor([[3., 0.]] * len(inputs)) if len(batch_sizes) == 1 else torch.tensor([[0., 3.]])

        classifier.model = model
        classifier.transform = lambda image: torch.zeros(3, image.height, image.width)
        boxes = [(0, 0, 20, 20), (20, 20, 40, 40), (40, 40, 60, 60)]
        output = classifier.classify(Image.new("RGB", (80, 80)), boxes)
        self.assertEqual(batch_sizes, [(2, 3, 320, 320), (1, 3, 320, 320)])
        self.assertEqual([result.coating_status for result in output], ["coated", "coated", "uncoated"])
        self.assertAlmostEqual(output[0].p_coated + output[0].p_insufficient, 1., places=6)
        self.assertEqual(classifier.classify(Image.new("RGB", (80, 80)), []), [])

    def test_missing_classifier_weights(self):
        settings = Settings()
        classifier = CoverageClassifier(replace(settings, classifier_path=settings.classifier_path.with_name("missing.pt")))
        with self.assertRaises(FileNotFoundError):
            classifier.load("cpu")


if __name__ == "__main__":
    unittest.main()
