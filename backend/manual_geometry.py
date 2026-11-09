"""Validation for user-drawn outlines; model contours retain their native topology."""

import cv2
import numpy as np


def validate_manual_polygon(points):
    normalized = []
    for point in points:
        if not normalized or np.linalg.norm(np.subtract(point, normalized[-1])) > 1e-6:
            normalized.append(tuple(point))
    if len(normalized) > 1 and np.linalg.norm(np.subtract(normalized[0], normalized[-1])) < 1e-6:
        normalized.pop()
    if not 3 <= len(normalized) <= 256:
        raise ValueError("Для ручного контура нужны от 3 до 256 точек.")
    contour = np.asarray(normalized, dtype=np.float32)
    if cv2.contourArea(contour) < 4:
        raise ValueError("Площадь ручного контура должна быть не меньше 4 px².")
    # Consistent turning alone does not exclude stars. Check all non-adjacent
    # edges, including endpoint touches; vectorization keeps 256-point edits fast.
    starts = np.asarray(normalized, dtype=np.float64)
    ends = np.roll(starts, -1, axis=0)
    a, b = starts[:, None, :], ends[:, None, :]
    c, d = starts[None, :, :], ends[None, :, :]

    def orientation(start, end, point):
        delta, offset = end - start, point - start
        return delta[..., 0] * offset[..., 1] - delta[..., 1] * offset[..., 0]

    def on_segment(point, cross):
        return (np.abs(cross) < 1e-8) & np.all(
            (point >= np.minimum(a, b) - 1e-8) & (point <= np.maximum(a, b) + 1e-8), axis=-1)

    first, second = orientation(a, b, c), orientation(a, b, d)
    touches_start, touches_end = on_segment(c, first), on_segment(d, second)
    intersections = ((first * second < 0) & (first.T * second.T < 0)
                     | touches_start | touches_end | touches_start.T | touches_end.T)
    intersections = np.triu(intersections, k=2)
    intersections[0, -1] = False
    if intersections.any():
        raise ValueError("Ручной контур пересекает сам себя.")
    return normalized
