import unittest

import cv2
import numpy as np
import torch

from backend.contours import particle_polygons, particle_contours, trim_attached_fragments


class ParticleContourTests(unittest.TestCase):
    def test_neighbor_body_wins_over_a_thin_overlapping_contact_strip(self):
        masks = np.zeros((2, 90, 140), dtype=np.uint8)
        cv2.circle(masks[0], (38, 42), 22, 1, -1)
        cv2.line(masks[0], (58, 42), (87, 42), 1, 2)
        cv2.rectangle(masks[0], (72, 31), (76, 53), 1, -1)
        cv2.circle(masks[1], (94, 42), 24, 1, -1)
        before = masks.copy()
        result = particle_contours(masks, masks.shape[1:], class_names=["rounded", "rounded"],
                                   confidences=[.9, .8], resolve_overlaps=True)
        np.testing.assert_array_equal(masks, before)
        self.assertIn("overlap", result[0].refinement_methods)
        self.assertLess(result[0].polygon[:, 0].max(), 70)
        self.assertLessEqual(result[0].removed_area_percent, 12)
        self.assertGreater(cv2.pointPolygonTest(result[0].polygon, (38, 42), False), 0)
        self.assertGreater(cv2.pointPolygonTest(result[1].polygon, (94, 42), False), 0)

    def test_nonoverlapping_irregular_masks_keep_their_boundaries(self):
        masks = np.zeros((2, 100, 180), dtype=np.uint8)
        cv2.fillPoly(masks[0], [np.array([[10, 10], [70, 15], [35, 70], [25, 30]])], 1)
        cv2.circle(masks[1], (135, 45), 25, 1, -1)
        result = particle_contours(masks, masks.shape[1:], class_names=["irregular", "rounded"],
                                   confidences=[.8, .7], resolve_overlaps=True)
        for contour in result:
            np.testing.assert_array_equal(contour.polygon, contour.original_polygon)
            self.assertEqual(contour.refinement_methods, ())

    def test_overlap_refinement_has_a_combined_area_limit(self):
        masks = np.zeros((2, 100, 120), dtype=np.uint8)
        cv2.circle(masks[0], (48, 45), 28, 1, -1)
        cv2.circle(masks[1], (55, 45), 28, 1, -1)
        result = particle_contours(masks, masks.shape[1:], class_names=["rounded", "rounded"],
                                   confidences=[.8, .9], resolve_overlaps=True)
        for contour in result:
            area = cv2.contourArea(contour.original_polygon)
            loss = (area - cv2.contourArea(contour.polygon)) / area * 100
            self.assertGreaterEqual(loss, 0)
            self.assertLessEqual(loss, 12)

    def attached_lobe_mask(self):
        mask = np.zeros((110, 140), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 30, 1, -1)
        cv2.line(mask, (75, 49), (96, 49), 1, 2)
        cv2.rectangle(mask, (94, 43), (105, 55), 1, -1)
        return mask

    def test_small_attached_lobe_is_removed_without_growing_mask(self):
        mask = self.attached_lobe_mask()
        result = particle_contours(mask[None], mask.shape, class_names=["rounded"])[0]
        self.assertGreater(result.removed_area_percent, 0)
        self.assertLessEqual(result.removed_area_percent, 12)
        self.assertGreater(result.original_polygon[:, 0].max(), 100)
        self.assertLess(result.polygon[:, 0].max(), 85)
        cleaned = np.zeros_like(mask)
        cv2.fillPoly(cleaned, [result.polygon.astype(np.int32)], 1)
        self.assertTrue(np.all(cleaned <= mask))

    def test_irregular_class_keeps_its_real_protrusions(self):
        mask = self.attached_lobe_mask()
        result = particle_contours(mask[None], mask.shape, class_names=["irregular"])[0]
        np.testing.assert_array_equal(result.polygon, result.original_polygon)
        self.assertEqual(result.removed_area_percent, 0)

    def test_ordinary_round_boundary_is_not_modified(self):
        mask = np.zeros((110, 140), dtype=np.uint8)
        cv2.ellipse(mask, (60, 50), (35, 23), 15, 0, 360, 1, -1)
        result = particle_contours(mask[None], mask.shape, class_names=["rounded"])[0]
        np.testing.assert_array_equal(result.polygon, result.original_polygon)
        self.assertEqual(result.removed_area_percent, 0)

    def test_large_second_body_is_not_silently_deleted(self):
        mask = np.zeros((100, 160), dtype=np.uint8)
        cv2.circle(mask, (42, 45), 28, 1, -1)
        cv2.circle(mask, (115, 45), 25, 1, -1)
        cv2.line(mask, (65, 45), (93, 45), 1, 2)
        result = particle_contours(mask[None], mask.shape, class_names=["rounded"])[0]
        np.testing.assert_array_equal(result.polygon, result.original_polygon)
        self.assertEqual(result.removed_area_percent, 0)

    def test_partial_border_particle_is_not_refined(self):
        mask = self.attached_lobe_mask()
        mask = mask[:, 22:]
        result = particle_contours(mask[None], mask.shape, class_names=["rounded"])[0]
        np.testing.assert_array_equal(result.polygon, result.original_polygon)
        self.assertEqual(result.removed_area_percent, 0)

    def test_disconnected_islands_are_not_joined_and_largest_area_wins(self):
        masks = np.zeros((1, 100, 140), dtype=np.uint8)
        cv2.circle(masks[0], (35, 40), 20, 1, -1)
        # A thin noisy component has many vertices, but much smaller area.
        noisy = np.array([[80 + i, 65 + i % 2 * 4] for i in range(40)], dtype=np.int32)
        cv2.polylines(masks[0], [noisy], False, 1, 1)
        cv2.circle(masks[0], (120, 15), 2, 1, -1)
        polygon = particle_polygons(torch.from_numpy(masks), (100, 140))[0]
        self.assertTrue(np.all(polygon[:, 0] <= 55))
        self.assertTrue(np.all(polygon[:, 0] >= 15))
        self.assertAlmostEqual(cv2.contourArea(polygon), 1200, delta=20)
        segments = np.diff(np.vstack([polygon, polygon[:1]]), axis=0)
        self.assertLess(np.linalg.norm(segments, axis=1).max(), 10)

    def test_single_mask_boundary_is_kept_without_smoothing(self):
        masks = np.zeros((1, 80, 120), dtype=np.uint8)
        cv2.fillPoly(masks[0], [np.array([[10, 10], [60, 20], [30, 60], [20, 35]])], 1)
        expected = cv2.findContours(masks[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1, 2)
        actual = particle_polygons(masks, (80, 120))[0]
        np.testing.assert_array_equal(actual, expected)

    def test_empty_and_degenerate_masks_keep_detection_alignment(self):
        masks = np.zeros((3, 40, 40), dtype=np.uint8)
        masks[1, 4, 4] = 1
        cv2.circle(masks[2], (20, 20), 8, 1, -1)
        result = particle_polygons(masks, (40, 40))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (0, 2))
        self.assertEqual(result[1].shape, (0, 2))
        self.assertGreater(len(result[2]), 3)

    def test_letterbox_coordinates_are_restored_for_fallback_masks(self):
        # Original 200x100 is fitted into 100x100 with 25px top/bottom padding.
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 35:56, 10:31] = 1
        polygon = particle_polygons(masks, (100, 200))[0]
        np.testing.assert_array_equal(polygon.min(axis=0), [20, 20])
        np.testing.assert_array_equal(polygon.max(axis=0), [60, 60])


if __name__ == "__main__":
    unittest.main()
