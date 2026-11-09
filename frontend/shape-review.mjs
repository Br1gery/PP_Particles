export const SHAPE_COLORS = { rounded: "#22a06b", irregular: "#e9a2a2", edge: "#e5b62f" };
export const SHAPE_LABELS = { rounded: "Округлая", irregular: "Неокруглая", edge: "На границе" };
export const CONTOUR_PROCESSING_VERSION = "2026-11-06.2";

export function currentContourProcessing(info) {
  return info?.contour_processing_version === CONTOUR_PROCESSING_VERSION;
}

export function contourProcessingSummary(result, detections) {
  if (!currentContourProcessing(result)) return "Версия обработки устарела. Перезапустите backend и повторите анализ.";
  const changed = detections.filter((p) => p.contour_refined).length;
  return `Обработка ${result.contour_processing_version} · Изменено контуров: ${changed} из ${detections.length}`;
}

// The shape API uses xywh; the shared viewer uses xyxy.
export function shapeDetections(particles, deletedIds = new Set(), additions = [], overrides = new Map()) {
  return [...particles, ...additions].filter((p) => !deletedIds.has(p.index)).map((source) => {
    const review = overrides.get(source.index) || {};
    let p = { ...source, source: source.source || "model", manual_form: review.form || source.manual_form || null };
    if (review.polygon_px) p = { ...p, ...polygonGeometry(review.polygon_px, review.image_width, review.image_height),
      manual_contour: true, contour_refined: false, removed_area_percent: 0, contour_refinement: [], raw_polygon_px: null };
    const semantic = p.manual_form || (["rounded","irregular"].includes(source.class_name) ? source.class_name
      : p.circularity >= .8 ? "rounded" : "irregular");
    p.form = p.touches_border ? "edge" : semantic;
    p.manual_review = Boolean(p.manual_form || p.manual_contour || p.source === "manual");
    const [x, y, width, height] = p.bbox;
    return { ...p, id: p.index, bbox: [x, y, x + width, y + height], width_px: width, height_px: height };
  });
}

function orientation(a, b, c) { return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); }
function between(a, b, p) {
  return Math.abs(orientation(a,b,p)) < 1e-8 && p[0] >= Math.min(a[0],b[0])-1e-8 && p[0] <= Math.max(a[0],b[0])+1e-8
    && p[1] >= Math.min(a[1],b[1])-1e-8 && p[1] <= Math.max(a[1],b[1])+1e-8;
}
function crosses(a,b,c,d) {
  return orientation(a,b,c)*orientation(a,b,d) < 0 && orientation(c,d,a)*orientation(c,d,b) < 0
    || between(a,b,c) || between(a,b,d) || between(c,d,a) || between(c,d,b);
}

export function polygonGeometry(points, imageWidth, imageHeight) {
  if (!Array.isArray(points) || points.length > 256 || !points.every((p) => p.length === 2 && p.every(Number.isFinite))) throw new Error("Некорректный контур (до 256 точек).");
  const polygon = points.map(([x,y]) => [Math.max(0,Math.min(imageWidth-1,x)),Math.max(0,Math.min(imageHeight-1,y))])
    .filter((p,i,a) => !i || Math.hypot(p[0]-a[i-1][0],p[1]-a[i-1][1]) > 1e-6);
  if (polygon.length > 1 && Math.hypot(polygon[0][0]-polygon.at(-1)[0],polygon[0][1]-polygon.at(-1)[1]) < 1e-6) polygon.pop();
  if (polygon.length < 3) throw new Error("Для контура нужны хотя бы три точки.");
  for (let i=0;i<polygon.length;i++) for (let j=i+1;j<polygon.length;j++) {
    if (j === i+1 || i===0 && j===polygon.length-1) continue;
    if (crosses(polygon[i],polygon[(i+1)%polygon.length],polygon[j],polygon[(j+1)%polygon.length])) throw new Error("Контур пересекает сам себя. Исправьте точки.");
  }
  let twiceArea=0,cx=0,cy=0,perimeter=0;
  for (let i=0;i<polygon.length;i++) {
    const [x,y]=polygon[i], [nx,ny]=polygon[(i+1)%polygon.length];
    const cross=x*ny-nx*y; twiceArea+=cross;cx+=(x+nx)*cross;cy+=(y+ny)*cross;perimeter+=Math.hypot(nx-x,ny-y);
  }
  const area=Math.abs(twiceArea)/2;
  if (area < 4) throw new Error("Выделите частицу площадью не менее 4 px².");
  const xs=polygon.map(p=>p[0]),ys=polygon.map(p=>p[1]);
  const left=Math.min(...xs),top=Math.min(...ys),right=Math.max(...xs),bottom=Math.max(...ys);
  return { polygon_px:polygon, bbox:[left,top,right-left,bottom-top], area_px2:area, perimeter_px:perimeter,
    circularity:Math.min(1,4*Math.PI*area/perimeter**2), equivalent_diameter_px:Math.sqrt(4*area/Math.PI),
    centroid_x:cx/(3*twiceArea),centroid_y:cy/(3*twiceArea),
    touches_border:left<=1 || top<=1 || right>=imageWidth-2 || bottom>=imageHeight-2 };
}

export function ellipsePolygon(bbox, imageWidth, imageHeight) {
  const [xa,ya,xb,yb]=bbox;
  const left=Math.max(0,Math.min(imageWidth-1,Math.min(xa,xb))),right=Math.max(0,Math.min(imageWidth-1,Math.max(xa,xb)));
  const top=Math.max(0,Math.min(imageHeight-1,Math.min(ya,yb))),bottom=Math.max(0,Math.min(imageHeight-1,Math.max(ya,yb)));
  if (right-left < 2 || bottom-top < 2) throw new Error("Выделите область не менее 2 × 2 пикселей.");
  return Array.from({length:64},(_,i)=>[(left+right)/2+(right-left)/2*Math.cos(i*Math.PI/32), (top+bottom)/2+(bottom-top)/2*Math.sin(i*Math.PI/32)]);
}

export function createManualShape(index, polygon, width, height) {
  if (!Number.isInteger(index) || index<1) throw new Error("Некорректный номер частицы.");
  const geometry=polygonGeometry(polygon,width,height);
  return { ...geometry, index, form:geometry.touches_border ? "edge" : "rounded", manual_form:"rounded",
    source:"manual", manual_contour:true, class_id:null,class_name:null,confidence:null,classification_source:"manual" };
}

export function setShapeReview(sources, overrides, index, patch) {
  if (!sources.some(p=>p.index===index)) throw new Error("Частица не найдена.");
  const next=new Map(overrides);
  if (patch===null) next.delete(index);
  else {
    if (patch.form && !["rounded","irregular"].includes(patch.form)) throw new Error("Некорректный класс формы.");
    if (patch.polygon_px) polygonGeometry(patch.polygon_px,patch.image_width,patch.image_height);
    next.set(index,{...next.get(index),...patch});
  }
  return next;
}

export function shapeReportParticles(detections) {
  return detections.map(({ index,polygon_px,source,manual_form,manual_contour,class_id,class_name,confidence,contour_refined,removed_area_percent }) =>
    ({ index,polygon_px,source,manual_form,manual_contour:Boolean(manual_contour),class_id,class_name,confidence,
      contour_refined:Boolean(contour_refined),removed_area_percent:removed_area_percent || 0 }));
}

export function summarizeShapes(particles) {
  const interior = particles.filter((p) => !p.touches_border);
  const rounded = interior.filter((p) => p.form === "rounded").length;
  const median = (values) => {
    if (!values.length) return null;
    values.sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
  };
  return { total: particles.length, rounded, irregular: interior.length - rounded,
    edge: particles.length - interior.length, shape_denominator: interior.length,
    rounded_percent: interior.length ? rounded / interior.length * 100 : null,
    median_circularity: median(interior.map((p) => p.circularity)),
    median_equivalent_diameter_px: median(interior.map((p) => p.equivalent_diameter_px)),
  };
}

export function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    const cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi);
    if (Math.abs(cross) < 1e-6 && x >= Math.min(xi, xj) && x <= Math.max(xi, xj) && y >= Math.min(yi, yj) && y <= Math.max(yi, yj)) return true;
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function drawContours(context, detections, { x = 0, y = 0, scaleX = 1, scaleY = 1, lineWidth = 2 } = {}) {
  context.lineWidth = lineWidth;
  for (const p of detections) {
    if (!p.polygon_px?.length) continue;
    context.strokeStyle = SHAPE_COLORS[p.form];
    context.beginPath();
    p.polygon_px.forEach(([px, py], i) => context[i ? "lineTo" : "moveTo"](x + px * scaleX, y + py * scaleY));
    context.closePath();
    context.stroke();
  }
}
