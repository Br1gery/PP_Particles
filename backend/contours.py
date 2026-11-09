"""Extract one particle silhouette per detection without joining mask islands."""

import cv2
import numpy as np
from dataclasses import dataclass
from math import pi, sqrt
from ultralytics.utils import ops


@dataclass
class ParticleContour:
    polygon: np.ndarray
    original_polygon: np.ndarray
    removed_area_percent: float = 0.0
    refinement_methods: tuple[str, ...] = ()


def mask_distance_region(mask):
    """Normalized interior thickness, computed only inside the occupied ROI."""
    binary = np.asarray(mask > 0, dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(binary)
    if not width or not height:
        return None
    local = binary[y:y + height, x:x + width]
    distance = cv2.distanceTransform(np.pad(local, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    distance /= max(float(distance.max()), 1)
    return x, y, width, height, local, distance


def mask_owners(masks, confidences):
    """Prefer the neighboring mask's body over another mask's thin fringe."""
    owner = np.full(masks.shape[1:], -1, dtype=np.int32)
    thickness = np.zeros(masks.shape[1:], dtype=np.float32)
    # Confidence breaks equal-thickness ties; original order breaks exact ties.
    for index in sorted(range(len(masks)), key=lambda i: (-confidences[i], i)):
        region = mask_distance_region(masks[index])
        if region is None:
            continue
        x, y, width, height, local, distance = region
        best = thickness[y:y + height, x:x + width]
        wins = (local > 0) & (distance > best + 1e-7)
        owner[y:y + height, x:x + width][wins] = index
        best[wins] = distance[wins]
    return owner


def remove_overlapping_fringe(mask, owner, index, class_name):
    region = mask_distance_region(mask)
    if region is None:
        return mask
    x, y, width, height, local, distance = region
    competing = owner[y:y + height, x:x + width]
    remove = (local > 0) & (competing >= 0) & (competing != index) & (distance < .15)
    if not np.any(remove):
        return mask
    cleaned = local.copy()
    cleaned[remove] = 0
    if class_name == "rounded":
        # After separating neighbors, remove the remaining thin contact loops.
        # This is never applied to non-overlapping or irregular particles.
        radius = max(1, min(4, round(max(width, height) * .025)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        padded = np.pad(cleaned, radius)
        opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cleaned &= opened[radius:-radius, radius:-radius]
    result = np.asarray(mask > 0, dtype=np.uint8)
    result[y:y + height, x:x + width] = cleaned
    return result


def trim_attached_fragments(contour, mask_shape):
    """Conservatively remove small lobes attached through a narrow neck.

    Try the smallest erosion radius first. Reconstruct the largest core inside
    the original silhouette, accepting only distant protrusions and <=12% loss.
    Normal boundaries are returned verbatim when no such protrusion is found.
    """
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    if area < 64 or x <= 1 or y <= 1 or x + width >= mask_shape[1] - 1 or y + height >= mask_shape[0] - 1:
        return contour, 0.0
    max_radius = min(12, max(1, round(sqrt(area / pi) * .06)))
    padding = max_radius + 2
    offset = np.array([x - padding, y - padding], dtype=np.int32)
    silhouette = np.zeros((height + 2 * padding, width + 2 * padding), dtype=np.uint8)
    cv2.drawContours(silhouette, [contour - offset], -1, 1, -1)
    original_pixels = int(silhouette.sum())
    for radius in range(1, max_radius + 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        eroded = cv2.erode(silhouette, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        if count <= 1:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        core = np.asarray(labels == largest, dtype=np.uint8)
        restored = cv2.dilate(core, kernel) & silhouette
        removed_fraction = 1 - int(restored.sum()) / original_pixels
        if not .001 <= removed_fraction <= .12:
            continue
        distance = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
        distant_pixels = np.count_nonzero((distance > 3 * radius) & (silhouette > 0))
        if distant_pixels < max(4, radius * radius):
            continue
        candidates = cv2.findContours(restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not candidates:
            continue
        cleaned = max(candidates, key=cv2.contourArea) + offset
        removed_area = (area - cv2.contourArea(cleaned)) / area * 100
        if 0 < removed_area <= 12:
            return cleaned, round(removed_area, 3)
    return contour, 0.0


def particle_contours(mask_data, image_shape, *, class_names=None, confidences=None, resolve_overlaps=False):
    """Keep the largest external contour by area, in original image coordinates.

    Ultralytics masks.xy concatenates disconnected components using synthetic
    bridges. A particle detection must not acquire those lines or noise islands.
    Empty/degenerate masks keep their position so box/class indices stay aligned.
    Optional neck cleanup applies only to the trained rounded class; the primary
    original contour and the removed area are retained for inspection.
    """
    masks = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
    if masks.ndim != 3:
        raise ValueError("Expected masks with shape [N, H, W]")
    if class_names is not None and len(class_names) != len(masks):
        raise ValueError("Mask and class counts do not match")
    if confidences is None:
        confidences = [0.0] * len(masks)
    if len(confidences) != len(masks):
        raise ValueError("Mask and confidence counts do not match")
    owners = mask_owners(masks, confidences) if resolve_overlaps and len(masks) > 1 else None
    results = []
    for index, mask in enumerate(masks):
        contours, _ = cv2.findContours(np.asarray(mask > 0, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        if largest is None or cv2.contourArea(largest) <= 0:
            empty = np.empty((0, 2), dtype=np.float32)
            results.append(ParticleContour(empty, empty))
            continue
        original = largest
        original_area = cv2.contourArea(original)
        methods = []
        name = class_names[index] if class_names is not None else ""
        if owners is not None:
            separated = remove_overlapping_fringe(mask, owners, index, name)
            candidates = cv2.findContours(np.asarray(separated > 0, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if candidates:
                candidate = max(candidates, key=cv2.contourArea)
                loss = (original_area - cv2.contourArea(candidate)) / original_area * 100
                if 0 <= loss <= 12 and not np.array_equal(candidate, original):
                    largest = candidate
                    methods.append("overlap")
        if class_names is not None and class_names[index] == "rounded":
            candidate, neck_loss = trim_attached_fragments(largest, mask.shape)
            total_loss = (original_area - cv2.contourArea(candidate)) / original_area * 100
            if neck_loss > 0 and total_loss <= 12:
                largest = candidate
                methods.append("neck")
        removed_area = round(max(0.0, (original_area - cv2.contourArea(largest)) / original_area * 100), 3)
        polygon = ops.scale_coords(mask.shape, largest.reshape(-1, 2).astype(np.float32), image_shape)
        original_polygon = ops.scale_coords(mask.shape, original.reshape(-1, 2).astype(np.float32), image_shape)
        results.append(ParticleContour(polygon, original_polygon, removed_area, tuple(methods)))
    return results


def particle_polygons(mask_data, image_shape):
    return [item.polygon for item in particle_contours(mask_data, image_shape)]
