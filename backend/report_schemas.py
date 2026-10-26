from typing import Annotated, Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator

Status = Literal["coated", "uncoated", "uncertain"]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]


class ReportImage(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ReportParticle(BaseModel):
    id: int = Field(gt=0)
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    coating_status: Status
    source: Literal["model", "manual"] = "model"
    manual_review: Status | None = None
    model_coating_status: Status | None = None
    confidence: Probability | None = None
    classifier_confidence: Probability | None = None
    p_coated: Probability | None = None
    p_insufficient: Probability | None = None

    @model_validator(mode="after")
    def valid_particle(self):
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid particle bounding box")
        if self.source == "manual" and any(value is not None for value in
                (self.confidence, self.classifier_confidence, self.p_coated, self.p_insufficient)):
            raise ValueError("Manual particles must not have invented model probabilities")
        return self


class ReportRequest(BaseModel):
    image: ReportImage
    original_image: str = Field(max_length=48 * 1024 * 1024)
    detections: list[ReportParticle] = Field(max_length=5000)
    initial_detected_count: int = Field(ge=0, le=5000)
    manual_added_count: int = Field(ge=0, le=5000)
    deleted_count: int = Field(ge=0, le=10000)
    model: str = Field(max_length=255)
    classifier: str = Field(max_length=255)
    classifier_confidence_threshold: Probability
    timezone_offset_minutes: int = Field(default=0, ge=-840, le=840)

    @model_validator(mode="after")
    def consistent_counts(self):
        if len({particle.id for particle in self.detections}) != len(self.detections):
            raise ValueError("Particle IDs must be unique")
        if self.initial_detected_count + self.manual_added_count - self.deleted_count != len(self.detections):
            raise ValueError("The annotation history does not match the remaining particles")
        if sum(p.source == "manual" for p in self.detections) > self.manual_added_count:
            raise ValueError("Invalid manual particle count")
        if sum(p.source == "model" for p in self.detections) > self.initial_detected_count:
            raise ValueError("Invalid model particle count")
        return self
