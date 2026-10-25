"""Remove duplicate particle detections before crop classification."""


def suppress_duplicate_boxes(rows: list[list[float]], iou_threshold: float) -> list[list[float]]:
    """Greedy NMS for the single particle class; keep original, unrounded scores.

    Input rows are [x1, y1, x2, y2, detector_confidence, class_id] with valid,
    image-clipped boxes. A lower-confidence box is removed only when its IoU
    with a retained higher-confidence box exceeds the threshold.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("Duplicate IoU threshold must be in [0, 1]")
    if len(rows) < 2:
        return list(rows)

    import torch
    from torchvision.ops import nms

    # Rows have already been copied to CPU; no extra CUDA synchronization or
    # confidence rounding should influence which detection wins.
    values = torch.tensor(rows, dtype=torch.float64, device="cpu")
    kept = nms(values[:, :4], values[:, 4], iou_threshold).tolist()
    return [rows[index] for index in kept]
