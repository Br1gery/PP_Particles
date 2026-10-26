export const COATING_COLORS = { coated: "#22a06b", uncertain: "#e5b62f", uncoated: "#e9a2a2" };
export const COATING_LABELS = { coated: "Покрыта", uncertain: "Спорная", uncoated: "Непокрыта" };

export function reviewedDetections(detections, overrides, additions = [], deletedIds = new Set()) {
  return [...detections, ...additions].filter((detection) => !deletedIds.has(detection.id)).map((detection) => ({
    ...detection,
    source: detection.source || "model",
    model_coating_status: detection.source === "manual" ? null : detection.coating_status,
    manual_review: overrides.get(detection.id) || (detection.source === "manual" ? detection.coating_status : null),
    coating_status: overrides.get(detection.id) || detection.coating_status,
  }));
}

export function setManualReview(detections, overrides, id, status) {
  const particle = detections.find((detection) => detection.id === id);
  if (!particle || (particle.source !== "manual" && particle.coating_status !== "uncertain")) throw new Error("Ручная оценка доступна для спорных и добавленных вручную частиц.");
  if (status !== null && status !== "coated" && status !== "uncoated" && !(particle.source === "manual" && status === "uncertain")) throw new Error("Некорректная оценка покрытия.");
  const next = new Map(overrides);
  if (status === null) next.delete(id);
  else next.set(id, status);
  return next;
}

export function createManualParticle(id, bbox, imageWidth, imageHeight) {
  if (!Number.isInteger(id) || id < 1 || bbox.length !== 4 || !bbox.every(Number.isFinite)) throw new Error("Некорректная рамка.");
  const [xa, ya, xb, yb] = bbox;
  const x1 = Math.max(0, Math.min(imageWidth, Math.min(xa, xb)));
  const y1 = Math.max(0, Math.min(imageHeight, Math.min(ya, yb)));
  const x2 = Math.max(0, Math.min(imageWidth, Math.max(xa, xb)));
  const y2 = Math.max(0, Math.min(imageHeight, Math.max(ya, yb)));
  if (x2 - x1 < 2 || y2 - y1 < 2) throw new Error("Выделите область размером не менее 2 × 2 пикселей.");
  return {
    id, bbox: [x1, y1, x2, y2].map((value) => Math.round(value * 100) / 100),
    width_px: x2 - x1, height_px: y2 - y1, source: "manual", coating_status: "uncertain",
    class_id: 0, class_name: "particle", confidence: null, classifier_confidence: null,
    coating_class: null, p_coated: null, p_insufficient: null,
  };
}

export function visibleDetections(detections, statuses) {
  return detections.filter((detection) => statuses.has(detection.coating_status));
}

export function summarizeCoverage(detections) {
  const summary = { count: detections.length, coated_count: 0, uncoated_count: 0, uncertain_count: 0, manual_count: 0 };
  for (const detection of detections) {
    summary[`${detection.coating_status}_count`] += 1;
    if (detection.manual_review) summary.manual_count += 1;
  }
  summary.coverage_percent = summary.count ? summary.coated_count / summary.count * 100 : null;
  return summary;
}

export function cropBounds(bbox, imageWidth, imageHeight, padding = .1) {
  const [x1, y1, x2, y2] = bbox;
  const dx = (x2 - x1) * padding;
  const dy = (y2 - y1) * padding;
  const left = Math.max(0, Math.floor(x1 - dx));
  const top = Math.max(0, Math.floor(y1 - dy));
  const right = Math.min(imageWidth, Math.ceil(x2 + dx));
  const bottom = Math.min(imageHeight, Math.ceil(y2 + dy));
  return { left, top, width: right - left, height: bottom - top };
}

export function drawFrames(context, detections, { x = 0, y = 0, scaleX = 1, scaleY = 1, lineWidth = 2 } = {}) {
  context.lineWidth = lineWidth;
  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox;
    context.strokeStyle = COATING_COLORS[detection.coating_status];
    context.strokeRect(x + x1 * scaleX, y + y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
  }
}
