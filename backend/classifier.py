"""Classify detector crops with the locally trained ResNet18 checkpoint."""

import logging
from math import ceil, floor

from PIL import Image, ImageOps

from backend.config import Settings
from backend.schemas import Classification

logger = logging.getLogger(__name__)
CLASSES = ("coated", "insufficient_coating")
COATING_COLORS = {"coated": "#22a06b", "uncertain": "#e5b62f", "uncoated": "#e9a2a2"}


def classify_probabilities(p_coated: float, p_insufficient: float, threshold: float) -> Classification:
    index = 0 if p_coated >= p_insufficient else 1
    confidence = max(p_coated, p_insufficient)
    return Classification(
        coating_class=CLASSES[index], p_coated=p_coated, p_insufficient=p_insufficient,
        classifier_confidence=confidence,
        coating_status="uncertain" if confidence < threshold else "coated" if index == 0 else "uncoated",
    )


def crop_particle(image: Image.Image, bbox: tuple[float, float, float, float], padding: float) -> Image.Image:
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    left, top = max(0, floor(x1 - dx)), max(0, floor(y1 - dy))
    right, bottom = min(image.width, ceil(x2 + dx)), min(image.height, ceil(y2 + dy))
    if right <= left or bottom <= top:
        raise ValueError("Detector produced an empty particle crop")
    return image.crop((left, top, right, bottom))


def pad_crop(crop: Image.Image, size: int = 320) -> Image.Image:
    """Match training: preserve aspect ratio, use median border RGB as padding."""
    import numpy as np

    crop = crop.convert("RGB")
    pixels = np.asarray(crop)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0)
    fill = tuple(int(value) for value in np.median(border, axis=0))
    return ImageOps.pad(crop, (size, size), method=Image.Resampling.BICUBIC, color=fill)


class CoverageClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.image_size = settings.classifier_image_size
        self.padding = settings.classifier_crop_padding

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self, device: str) -> None:
        if not self.settings.classifier_path.is_file():
            raise FileNotFoundError(self.settings.classifier_path)
        import torch
        from torchvision import models, transforms

        checkpoint = torch.load(self.settings.classifier_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture", "resnet18") != "resnet18":
            raise ValueError("Expected a resnet18 classifier")
        if tuple(checkpoint.get("classes", CLASSES)) != CLASSES:
            raise ValueError(f"Expected classifier class order {CLASSES}")
        if checkpoint.get("preprocessing", "raw") != "raw":
            raise ValueError("Unsupported classifier preprocessing; expected raw RGB")
        self.image_size = int(checkpoint.get("size", self.settings.classifier_image_size))
        self.padding = float(checkpoint.get("padding", self.settings.classifier_crop_padding))
        if self.image_size != 320 or not 0 <= self.padding <= 1:
            raise ValueError("Expected size=320 and crop padding in [0, 1]")
        if self.settings.classifier_normalization not in ("imagenet", "none"):
            raise ValueError("CLASSIFIER_NORMALIZATION must be imagenet or none")
        if not .5 < self.settings.classifier_confidence_threshold <= 1:
            raise ValueError("Classifier confidence threshold must be in (0.5, 1]")
        if self.settings.classifier_batch_size < 1:
            raise ValueError("Classifier batch size must be positive")

        # The checkpoint supplies all weights, so never download ImageNet weights.
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        self.device = device
        self.model = model.to(device).eval()
        steps = [transforms.ToTensor()]
        if self.settings.classifier_normalization == "imagenet":
            steps.append(transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]))
        self.transform = transforms.Compose(steps)
        logger.info("Loaded %s on %s (size=%s, padding=%s, normalization=%s)",
                    self.settings.classifier_path.name, device, self.image_size, self.padding,
                    self.settings.classifier_normalization)

    def classify(self, image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> list[Classification]:
        if not boxes:
            return []
        if not self.ready or self.transform is None:
            raise RuntimeError("Coverage classifier is not loaded")
        import torch

        predictions = []
        # Only keep one batch of crops/tensors in memory, even with dense micrographs.
        with torch.inference_mode():
            for offset in range(0, len(boxes), self.settings.classifier_batch_size):
                batch_boxes = boxes[offset:offset + self.settings.classifier_batch_size]
                inputs = torch.stack([
                    self.transform(pad_crop(crop_particle(image, bbox, self.padding), self.image_size))
                    for bbox in batch_boxes
                ]).to(self.device)
                logits = self.model(inputs)
                if logits.shape != (len(batch_boxes), 2):
                    raise ValueError("Classifier must return logits [B, 2]")
                probabilities = logits.softmax(dim=1).cpu().tolist()
                predictions.extend(classify_probabilities(p[0], p[1], self.settings.classifier_confidence_threshold)
                                   for p in probabilities)
        return predictions
