"""Native-resolution YOLO masks with bounded intermediate memory."""

import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

MAX_NATIVE_MASK_PIXELS = 128_000_000
NATIVE_CHUNK_PIXELS = 16_000_000


def native_masks_chunked(proto, coefficients, boxes, shape, chunk_pixels=NATIVE_CHUNK_PIXELS):
    """Same native decoder, with small batches rather than a full N×H×W float buffer."""
    pixels = shape[0] * shape[1]
    chunk_size = max(1, chunk_pixels // pixels)
    channels, height, width = proto.shape
    # Decode all low-resolution logits together, as Ultralytics does. Changing
    # the GEMM batch size can otherwise flip pixels whose logits are near zero.
    logits = (coefficients @ proto.float().view(channels, -1)).view(-1, height, width)
    masks = torch.empty((len(coefficients), *shape), dtype=torch.uint8, device=proto.device)
    for start in range(0, len(coefficients), chunk_size):
        stop = start + chunk_size
        scaled = ops.scale_masks(logits[start:stop][None], shape)[0]
        if not scaled.is_cuda and len(coefficients) >= 50:
            # Preserve native crop semantics: Ultralytics rounds CPU boxes for
            # N<50, but uses a pixel comparison for N>=50, independent of chunks.
            x1, y1, x2, y2 = torch.chunk(boxes[start:stop, :, None], 4, 1)
            columns = torch.arange(shape[1], device=scaled.device, dtype=x1.dtype)[None, None, :]
            rows = torch.arange(shape[0], device=scaled.device, dtype=x1.dtype)[None, :, None]
            cropped = scaled * ((columns >= x1) * (columns < x2) * (rows >= y1) * (rows < y2))
        else:
            cropped = ops.crop_mask(scaled, boxes[start:stop])
        masks[start:stop] = cropped.gt_(0.0).byte()
    return masks


class NativeSegmentationPredictor(SegmentationPredictor):
    """Ultralytics' native decoder; fall back explicitly for oversized mask batches."""

    native_pixel_budget = MAX_NATIVE_MASK_PIXELS

    def construct_result(self, pred, img, orig_img, img_path, proto):
        if not self.args.retina_masks or not len(pred):
            return super().construct_result(pred, img, orig_img, img_path, proto)
        shape = orig_img.shape[:2]
        if len(pred) * shape[0] * shape[1] > self.native_pixel_budget:
            # Keep all detections rather than reducing max_det to fit native masks.
            self.args.retina_masks = False
            try:
                result = super().construct_result(pred, img, orig_img, img_path, proto)
            finally:
                self.args.retina_masks = True
            result.mask_quality_warning = ("Для этого снимка маски рассчитаны в уменьшенном разрешении из-за их объёма. "
                                           "Для детальных контуров загрузите меньшую область снимка.")
            return result
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        masks = native_masks_chunked(proto, pred[:, 6:], pred[:, :4], shape)
        keep = masks.amax((-2, -1)) > 0
        if not all(keep):
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
