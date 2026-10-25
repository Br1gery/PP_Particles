import unittest

from backend.postprocessing import suppress_duplicate_boxes


class DuplicateSuppressionTests(unittest.TestCase):
    def test_empty_and_single_detection(self):
        self.assertEqual(suppress_duplicate_boxes([], .7), [])
        row = [10, 10, 50, 50, .9, 0]
        self.assertEqual(suppress_duplicate_boxes([row], .7), [row])

    def test_identical_and_nearly_identical_boxes_keep_the_highest_score(self):
        lower = [14, 12, 155, 93, .65, 0]
        higher = [16, 15, 154, 90, .95, 0]
        identical = [16, 15, 154, 90, .75, 0]
        self.assertEqual(suppress_duplicate_boxes([lower, higher, identical], .7), [higher])

    def test_similar_sizes_do_not_remove_spatially_separate_particles(self):
        first = [0, 0, 30, 30, .6, 0]
        second = [200, 200, 230, 230, .9, 0]
        self.assertEqual(suppress_duplicate_boxes([first, second], .7), [second, first])

    def test_partially_overlapping_neighboring_particles_are_kept(self):
        upper = [14, 30, 108, 138, .9, 0]
        lower = [18, 96, 144, 198, .8, 0]
        self.assertEqual(suppress_duplicate_boxes([upper, lower], .7), [upper, lower])

    def test_small_box_inside_a_larger_particle_is_not_automatically_a_duplicate(self):
        outer = [0, 0, 100, 100, .9, 0]
        inner = [30, 30, 45, 45, .8, 0]
        self.assertEqual(suppress_duplicate_boxes([outer, inner], .7), [outer, inner])

    def test_suppressed_boxes_cannot_suppress_additional_neighbors(self):
        first = [0, 0, 100, 100, .95, 0]
        middle = [10, 0, 110, 100, .9, 0]
        last = [20, 0, 120, 100, .85, 0]
        self.assertEqual(suppress_duplicate_boxes([middle, last, first], .7), [first, last])

    def test_original_scores_are_compared_before_response_rounding(self):
        lower = [10, 10, 50, 50, .9000001, 0]
        higher = [10.01, 10, 50, 50, .9000002, 0]
        self.assertEqual(suppress_duplicate_boxes([lower, higher], .7), [higher])

    def test_threshold_is_configurable(self):
        first = [0, 0, 100, 100, .9, 0]
        shifted = [10, 0, 110, 100, .8, 0]
        self.assertEqual(suppress_duplicate_boxes([first, shifted], .7), [first])
        self.assertEqual(suppress_duplicate_boxes([first, shifted], .9), [first, shifted])
        for invalid in (-.1, 1.1, float("nan")):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                suppress_duplicate_boxes([first, shifted], invalid)


if __name__ == "__main__":
    unittest.main()
