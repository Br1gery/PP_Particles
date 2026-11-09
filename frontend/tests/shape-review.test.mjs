import test from "node:test";
import assert from "node:assert/strict";
import { shapeDetections, summarizeShapes, pointInPolygon, drawContours, currentContourProcessing,
  contourProcessingSummary, CONTOUR_PROCESSING_VERSION } from "../shape-review.mjs";
import { polygonGeometry, ellipsePolygon, createManualShape, setShapeReview, shapeReportParticles } from "../shape-review.mjs";
import { detectionAt } from "../image-viewer.mjs";

const particles = [
  { index: 1, form: "rounded", bbox: [10, 10, 20, 20], circularity: .9, equivalent_diameter_px: 20, touches_border: false, polygon_px: [[10, 20], [20, 10], [30, 20], [20, 30]] },
  { index: 2, form: "irregular", bbox: [40, 10, 20, 20], circularity: .5, equivalent_diameter_px: 10, touches_border: false },
  { index: 3, form: "edge", bbox: [0, 0, 10, 10], circularity: .99, equivalent_diameter_px: 5, touches_border: true },
];

test("ellipse annotations normalize reversed dragging, preserve shape and have no model confidence", () => {
  const outline=ellipsePolygon([70,60,10,20],100,80);
  const manual=createManualShape(4,outline,100,80);
  assert.equal(manual.source,"manual");
  assert.equal(manual.confidence,null);
  assert.equal(manual.class_id,null);
  assert.equal(manual.class_name,null);
  assert.equal(manual.form,"rounded");
  assert.ok(Math.abs(manual.area_px2-Math.PI*30*20)<4);
  assert.ok(Math.abs(manual.centroid_x-40)<1e-6);
  assert.ok(Math.abs(manual.centroid_y-40)<1e-6);
  assert.throws(()=>ellipsePolygon([1,1,2,2],100,80));
});

test("manual geometry validates polygon topology and works in both winding directions", () => {
  const polygon=[[10,10],[40,10],[40,30],[10,30]];
  const a=polygonGeometry(polygon,100,80),b=polygonGeometry([...polygon].reverse(),100,80);
  assert.equal(a.area_px2,600);assert.equal(b.area_px2,600);
  assert.equal(a.centroid_x,25);assert.equal(b.centroid_x,25);
  assert.equal(a.centroid_y,20);assert.equal(a.perimeter_px,100);
  assert.throws(()=>polygonGeometry([[10,10],[40,40],[10,40],[40,10]],100,80));
  assert.throws(()=>polygonGeometry([[1,1],[2,2],[3,3]],100,80));
  assert.throws(()=>polygonGeometry([[1,1],[NaN,3],[4,5]],100,80));
});

test("class and contour reviews keep original model data, update summary, and can be undone", () => {
  const source={...particles[0],class_id:0,class_name:"rounded",confidence:.83,classification_source:"model"};
  const addition=createManualShape(4,ellipsePolygon([45,40,75,70],100,80),100,80);
  let reviews=setShapeReview([source,addition],new Map(),1,{form:"irregular"});
  reviews=setShapeReview([source,addition],reviews,1,{polygon_px:[[10,10],[30,10],[30,30],[10,30]],image_width:100,image_height:80});
  let current=shapeDetections([source],new Set(),[addition],reviews);
  assert.equal(current[0].class_name,"rounded");assert.equal(current[0].confidence,.83);
  assert.equal(current[0].form,"irregular");assert.equal(current[0].manual_contour,true);
  assert.equal(current[0].area_px2,400);
  assert.equal(summarizeShapes(current).rounded_percent,50);
  assert.deepEqual(source.polygon_px,particles[0].polygon_px);
  reviews=setShapeReview([source,addition],reviews,1,null);
  current=shapeDetections([source],new Set(),[addition],reviews);
  assert.equal(current[0].form,"rounded");assert.equal(summarizeShapes(current).rounded_percent,100);
  assert.deepEqual(shapeDetections([source],new Set([1,4]),[addition],reviews),[]);
});

test("border manual annotations retain their semantic class and are excluded from shape percent", () => {
  const addition=createManualShape(4,ellipsePolygon([-10,-10,30,30],100,80),100,80);
  const current=shapeDetections([],new Set(),[addition]);
  assert.equal(current[0].form,"edge");assert.equal(current[0].manual_form,"rounded");
  assert.equal(summarizeShapes(current).rounded_percent,null);
  const report=shapeReportParticles(current);
  assert.equal(report[0].source,"manual");assert.equal(report[0].confidence,null);
  assert.equal(report[0].manual_contour,true);
  assert.equal(report[0].manual_form,"rounded");
});

test("editing a contour updates geometric classification for single-class checkpoints", () => {
  const source={...particles[1],class_id:0,class_name:"particle",confidence:.7,classification_source:"circularity"};
  const reviews=setShapeReview([source],new Map(),source.index,{polygon_px:ellipsePolygon([10,10,40,40],100,80),image_width:100,image_height:80});
  const current=shapeDetections([source],new Set(),[],reviews)[0];
  assert.equal(current.form,"rounded");
  assert.equal(current.confidence,.7);
  assert.equal(current.manual_form,null);
  assert.equal(current.manual_contour,true);
});

test("old backend processing is identified and current refinement count respects deletions", () => {
  assert.equal(currentContourProcessing({}), false);
  assert.equal(currentContourProcessing({ contour_processing_version: "older" }), false);
  assert.match(contourProcessingSummary({}, []), /Перезапустите backend/);
  const result = { contour_processing_version: CONTOUR_PROCESSING_VERSION };
  const detections = [{ id: 1, contour_refined: true }, { id: 2, contour_refined: false }];
  assert.equal(currentContourProcessing(result), true);
  assert.match(contourProcessingSummary(result, detections), /Изменено контуров: 1 из 2/);
  assert.match(contourProcessingSummary(result, detections.slice(1)), /Изменено контуров: 0 из 1/);
});

test("shape boxes convert xywh to xyxy without mutating API data; deletion preserves IDs", () => {
  const result = shapeDetections(particles, new Set([2]));
  assert.deepEqual(result.map((p) => p.id), [1, 3]);
  assert.deepEqual(result[0].bbox, [10, 10, 30, 30]);
  assert.deepEqual(particles[0].bbox, [10, 10, 20, 20]);
});

test("summary excludes border particles; deletion updates percent and both medians", () => {
  const result = summarizeShapes(particles);
  assert.equal(result.rounded_percent, 50);
  assert.equal(result.median_circularity, .7);
  assert.equal(result.median_equivalent_diameter_px, 15);
  assert.equal(summarizeShapes(shapeDetections(particles, new Set([2]))).rounded_percent, 100);
  assert.equal(summarizeShapes([particles[2]]).rounded_percent, null);
  assert.equal(summarizeShapes([]).median_circularity, null);
});

test("review and summary respect trained classes even when geometric circularity disagrees", () => {
  const modelParticles = [
    { ...particles[0], form: "irregular", class_id: 1, class_name: "irregular", classification_source: "model", circularity: .99 },
    { ...particles[1], form: "rounded", class_id: 0, class_name: "rounded", classification_source: "model", circularity: .4 },
  ];
  const reviewed = shapeDetections(modelParticles);
  assert.equal(reviewed[0].form, "irregular");
  assert.equal(reviewed[1].form, "rounded");
  assert.equal(summarizeShapes(reviewed).rounded_percent, 50);
  assert.equal(summarizeShapes(shapeDetections(modelParticles, new Set([1]))).rounded_percent, 100);
});

test("hit testing uses actual polygon interior and boundary, after zoom and pan", () => {
  const polygon = particles[0].polygon_px;
  assert.equal(pointInPolygon(11, 11, polygon), false);
  assert.equal(pointInPolygon(20, 20, polygon), true);
  assert.equal(pointInPolygon(10, 20, polygon), true);
  const detections = shapeDetections(particles);
  const view = { x: -10, y: 5, scale: 2 };
  const size = { width: 100, height: 100 };
  assert.equal(detectionAt({ x: 30, y: 45 }, detections, view, size, size)?.id, 1);
  assert.equal(detectionAt({ x: 12, y: 27 }, detections, view, size, size), null);
});

test("contour drawing scales all vertices and closes paths with category color", () => {
  const points = [];
  const ctx = { beginPath() {}, moveTo: (...p) => points.push(p), lineTo: (...p) => points.push(p), closePath() { this.closed = true; }, stroke() {} };
  drawContours(ctx, [particles[0]], { x: 5, y: -5, scaleX: 2, scaleY: 3 });
  assert.deepEqual(points[0], [25, 55]);
  assert.equal(ctx.closed, true);
  assert.equal(ctx.strokeStyle, "#22a06b");
});
