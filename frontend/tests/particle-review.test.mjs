import test from "node:test";
import assert from "node:assert/strict";
import { reviewedDetections, setManualReview, createManualParticle, visibleDetections, summarizeCoverage, cropBounds, drawFrames, COATING_COLORS } from "../particle-review.mjs";

const particles = [
  { id: 1, coating_status: "coated", classifier_confidence: .9, bbox: [10, 10, 30, 30] },
  { id: 2, coating_status: "uncertain", classifier_confidence: .6, bbox: [40, 10, 60, 30] },
  { id: 3, coating_status: "uncoated", classifier_confidence: .9, bbox: [70, 10, 90, 30] },
];

test("filters can show one group, several groups, or no frames", () => {
  assert.deepEqual(visibleDetections(particles, new Set(["uncertain"])).map((p) => p.id), [2]);
  assert.deepEqual(visibleDetections(particles, new Set(["coated", "uncoated"])).map((p) => p.id), [1, 3]);
  assert.deepEqual(visibleDetections(particles, new Set()), []);
});

test("manual decisions update counts and coverage without changing model output", () => {
  const overrides = setManualReview(particles, new Map(), 2, "coated");
  const reviewed = reviewedDetections(particles, overrides);
  const summary = summarizeCoverage(reviewed);
  assert.equal(summary.coated_count, 2);
  assert.equal(summary.uncertain_count, 0);
  assert.equal(summary.uncoated_count, 1);
  assert.equal(summary.manual_count, 1);
  assert.ok(Math.abs(summary.coverage_percent - 200 / 3) < 1e-10);
  assert.equal(reviewed[1].model_coating_status, "uncertain");
  assert.equal(reviewed[1].classifier_confidence, .6);
  assert.equal(particles[1].coating_status, "uncertain");
  assert.deepEqual(visibleDetections(reviewed, new Set(["uncertain"])), []);
});

test("manual choices can be changed and undone", () => {
  let overrides = setManualReview(particles, new Map(), 2, "coated");
  overrides = setManualReview(particles, overrides, 2, "uncoated");
  assert.equal(summarizeCoverage(reviewedDetections(particles, overrides)).uncoated_count, 2);
  overrides = setManualReview(particles, overrides, 2, null);
  assert.equal(overrides.size, 0);
  assert.equal(summarizeCoverage(reviewedDetections(particles, overrides)).uncertain_count, 1);
});

test("only originally uncertain particles may be reviewed", () => {
  assert.throws(() => setManualReview(particles, new Map(), 1, "uncoated"));
  assert.throws(() => setManualReview(particles, new Map(), 999, "coated"));
  assert.throws(() => setManualReview(particles, new Map(), 2, "invalid"));
});

test("empty results have no coverage percentage", () => {
  const summary = summarizeCoverage([]);
  assert.equal(summary.count, 0);
  assert.equal(summary.coverage_percent, null);
});

test("detail crops add context and clip to the source image", () => {
  assert.deepEqual(cropBounds([10, 20, 60, 50], 100, 80), { left: 5, top: 17, width: 60, height: 36 });
  assert.deepEqual(cropBounds([0, 0, 100, 80], 100, 80), { left: 0, top: 0, width: 100, height: 80 });
});

test("drawing uses only selected particles and the current review color", () => {
  const records = [];
  const context = { strokeRect(...bbox) { records.push({ color: this.strokeStyle, bbox }); } };
  const reviewed = reviewedDetections(particles, setManualReview(particles, new Map(), 2, "uncoated"));
  drawFrames(context, visibleDetections(reviewed, new Set(["uncoated"])), { x: -20, y: -5, scaleX: 2, scaleY: 2 });
  assert.equal(records.length, 2);
  assert.equal(records[0].color, COATING_COLORS.uncoated);
  assert.deepEqual(records[0].bbox, [60, 15, 40, 40]);
});

test("manual boxes normalize drag direction, clip to image, and never invent confidence", () => {
  const particle = createManualParticle(4, [125, 90, 20, 10], 100, 80);
  assert.deepEqual(particle.bbox, [20, 10, 100, 80]);
  assert.equal(particle.width_px, 80);
  assert.equal(particle.height_px, 70);
  assert.equal(particle.source, "manual");
  assert.equal(particle.confidence, null);
  assert.equal(particle.classifier_confidence, null);
  assert.throws(() => createManualParticle(5, [0, 0, 1, 20], 100, 80));
});

test("new manual particles can take any class and contribute to coverage", () => {
  const addition = createManualParticle(4, [10, 40, 30, 60], 100, 80);
  let overrides = setManualReview([...particles, addition], new Map(), 4, "coated");
  let current = reviewedDetections(particles, overrides, [addition]);
  assert.equal(summarizeCoverage(current).coated_count, 2);
  assert.equal(summarizeCoverage(current).coverage_percent, 50);
  overrides = setManualReview([...particles, addition], overrides, 4, "uncertain");
  current = reviewedDetections(particles, overrides, [addition]);
  assert.equal(summarizeCoverage(current).uncertain_count, 2);
  assert.equal(current.find((p) => p.id === 4).model_coating_status, null);
});

test("deleting any model or manual box updates counts and undo restores its decision", () => {
  const addition = createManualParticle(4, [10, 40, 30, 60], 100, 80);
  const overrides = setManualReview([...particles, addition], new Map(), 4, "coated");
  const deleted = new Set([1, 4]);
  let current = reviewedDetections(particles, overrides, [addition], deleted);
  assert.deepEqual(current.map((p) => p.id), [2, 3]);
  assert.equal(summarizeCoverage(current).coverage_percent, 0);
  deleted.delete(4);
  current = reviewedDetections(particles, overrides, [addition], deleted);
  assert.equal(current.find((p) => p.id === 4).coating_status, "coated");
  assert.equal(summarizeCoverage(current).count, 3);
});
