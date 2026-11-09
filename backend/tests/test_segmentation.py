import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

from backend.segmentation import NativeSegmentationPredictor, native_masks_chunked


class NativeMaskTests(unittest.TestCase):
    def test_chunked_decoder_matches_ultralytics_native_decoder(self):
        generator = torch.Generator().manual_seed(19)
        proto = torch.randn(4, 12, 16, generator=generator)
        coefficients = torch.randn(7, 4, generator=generator)
        boxes = torch.tensor([[4, 3, 54, 42]] * 7, dtype=torch.float32)
        expected = ops.process_mask_native(proto, coefficients, boxes, (48, 64))
        actual = native_masks_chunked(proto, coefficients, boxes, (48, 64), chunk_pixels=48 * 64 * 2)
        self.assertEqual(tuple(actual.shape), (7, 48, 64))
        self.assertTrue(torch.equal(actual, expected))

    def test_oversized_masks_keep_detections_and_report_fallback(self):
        predictor = object.__new__(NativeSegmentationPredictor)
        predictor.args = SimpleNamespace(retina_masks=True)
        predictor.native_pixel_budget = 1
        pred = torch.zeros((2, 10))
        sentinel = SimpleNamespace()

        def regular(*args):
            self.assertFalse(predictor.args.retina_masks)
            self.assertEqual(len(args[0]), 2)
            return sentinel

        with patch.object(SegmentationPredictor, "construct_result", side_effect=regular):
            result = predictor.construct_result(pred, None, np.zeros((48, 64, 3)), "test.png", None)
        self.assertIs(result, sentinel)
        self.assertIn("уменьшенном разрешении", result.mask_quality_warning)
        self.assertTrue(predictor.args.retina_masks)

    def test_large_cpu_batch_keeps_fractional_box_crop_semantics(self):
        proto = torch.ones((1, 8, 8))
        coefficients = torch.ones((51, 1))
        boxes = torch.tensor([[3.3, 4.2, 29.4, 27.6]] * 51)
        expected = ops.process_mask_native(proto, coefficients, boxes, (32, 32))
        actual = native_masks_chunked(proto, coefficients, boxes, (32, 32), chunk_pixels=32 * 32 * 3)
        self.assertTrue(torch.equal(actual, expected))

    def test_native_result_has_original_size_and_preserves_classes_and_scores(self):
        predictor = object.__new__(NativeSegmentationPredictor)
        predictor.args = SimpleNamespace(retina_masks=True)
        predictor.model = SimpleNamespace(names={0: "rounded", 1: "irregular"})
        # Inference image 32x32 corresponds to source image 64x64.
        pred = torch.tensor([[2, 3, 25, 27, .9, 1, 1], [4, 5, 20, 22, .8, 0, 1]], dtype=torch.float32)
        result = predictor.construct_result(pred, torch.zeros((1, 3, 32, 32)), np.zeros((64, 64, 3), dtype=np.uint8),
                                            "test.png", torch.ones((1, 8, 8)))
        self.assertEqual(tuple(result.masks.data.shape), (2, 64, 64))
        self.assertEqual(result.boxes.cls.tolist(), [1, 0])
        self.assertTrue(torch.allclose(result.boxes.conf, torch.tensor([.9, .8])))
        self.assertEqual(result.boxes.xyxy[0].tolist(), [4, 6, 50, 54])


if __name__ == "__main__":
    unittest.main()
