"use strict";

import { ImageViewer } from "./image-viewer.mjs?v=20261109-1";
import { COATING_LABELS, reviewedDetections, setManualReview, createManualParticle, visibleDetections, summarizeCoverage, cropBounds, drawFrames } from "./particle-review.mjs?v=20261109-1";
import { SHAPE_LABELS, shapeDetections, summarizeShapes, currentContourProcessing, contourProcessingSummary,
  ellipsePolygon, createManualShape, setShapeReview, shapeReportParticles } from "./shape-review.mjs?v=20261109-1";

const $ = (id) => document.getElementById(id);
const statuses = ["coated", "uncertain", "uncoated"];
const shapeStatuses = ["rounded", "irregular", "edge"];
const isShape = () => state.mode === "shape";
const labelFor = (particle) => isShape() ? SHAPE_LABELS[particle.form] : COATING_LABELS[particle.coating_status];
const state = {
  mode: "coverage",
  file: null, previewUrl: null, result: null, originalImage: null,
  ready: false, busy: false, exporting: false, reportBusy: false, connectionMessage: "Подключение…",
  view: "original", overrides: new Map(), filters: new Set(statuses), detections: [], selectedId: null,
  additions: [], deletedIds: new Set(), deletedHistory: [], editingId: null, manualShapeApiReady: false,
};
const supported = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
const circularityNumber = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 4 });
const percent = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 1 });
let healthTimer;
const imageViewer = new ImageViewer({
  stage: $("image-stage"), image: $("preview-image"), controls: $("zoom-controls"),
  zoomIn: $("zoom-in"), zoomOut: $("zoom-out"), reset: $("zoom-reset"), value: $("zoom-value"),
  tooltip: $("particle-tooltip"), overlay: $("annotation-layer"), onSelect: openParticle,
  onDraw: addParticle, onDrawCancel: () => setDrawing(false),
  onPolygon: commitShapePolygon,
  onDraftChange: (count) => { $("finish-contour").disabled=count<3; $("undo-contour-point").disabled=!count; },
});

function showError(message = "") {
  $("error-message").textContent = message;
  $("error-message").hidden = !message;
}

function updateControls() {
  imageViewer.setEnabled(!state.busy);
  $("mode-coverage").disabled = state.busy;
  $("mode-shape").disabled = state.busy;
  $("analyze-button").disabled = state.busy || !state.file || !state.ready;
  $("analyze-button").textContent = state.busy ? "Распознавание…" : "Распознать частицы";
  $("file-input").disabled = state.busy;
  for (const status of [...statuses, ...shapeStatuses]) $(`filter-${status}`).disabled = state.busy;
  $("particle-picker").disabled = state.busy || !getVisibleDetections().length;
  $("draw-particle").disabled = state.busy || !state.result;
  $("shape-draw-tool").hidden = !isShape();
  $("shape-draw-tool").disabled = state.busy || !state.result;
  $("undo-delete").disabled = state.busy;
  $("generate-report").disabled = state.busy || state.reportBusy || !state.result;
  $("generate-report").textContent = state.reportBusy ? "Формирование PDF…" : "Сформировать отчёт";
  $("run-status").textContent = state.busy ? "Подождите, изображение обрабатывается."
    : !state.ready ? state.connectionMessage
    : state.result ? "Готово. Кликните по частице для подробного просмотра."
    : state.file ? "Изображение готово к распознаванию."
    : "Выберите изображение для начала.";
}

async function checkHealth() {
  clearTimeout(healthTimer);
  const mode = state.mode;
  try {
    const response = await fetch(mode === "shape" ? "/api/shape/health" : "/api/health", { cache: "no-store", signal: AbortSignal.timeout(10000) });
    if (mode !== state.mode) return;
    if (response.status === 404) {
      state.ready = false;
      state.connectionMessage = mode === "shape"
        ? "Сервер ещё не поддерживает округлость. Перезапустите backend и обновите страницу."
        : "API не найден. Откройте сайт через запущенный backend: http://127.0.0.1:8000/.";
      return;
    }
    const health = await response.json();
    if (mode !== state.mode) return;
    state.ready = response.ok && health.status === "ready";
    if (mode === "shape") state.manualShapeApiReady=health.manual_shape_annotations===true;
    state.connectionMessage = health.error || "Модель временно недоступна.";
    if (mode === "shape" && state.ready && !currentContourProcessing(health)) {
      state.ready = false;
      state.connectionMessage = "Запущена старая обработка контуров. Перезапустите backend и обновите страницу.";
    }
  } catch {
    if (mode !== state.mode) return;
    state.ready = false;
    state.connectionMessage = "Нет соединения с сервером. Проверьте, запущено ли приложение.";
  } finally {
    if (mode === state.mode) {
      updateControls();
      clearTimeout(healthTimer);
      healthTimer = setTimeout(checkHealth, state.ready ? 30000 : 10000);
    }
  }
}

function getVisibleDetections() {
  return isShape() ? state.detections.filter((p) => state.filters.has(p.form)) : visibleDetections(state.detections, state.filters);
}
function allSources() { return [...(isShape() ? state.result?.particles || [] : state.result?.detections || []), ...state.additions]; }

function setDrawing(enabled) {
  const tool=isShape() ? $("shape-draw-tool").value : "rectangle";
  imageViewer.setDrawing(enabled,tool);
  if (!enabled) state.editingId=null;
  $("draw-particle").setAttribute("aria-pressed", String(enabled));
  $("draw-particle").textContent = enabled ? "Отменить выделение" : "Выделить частицу";
  $("draw-help").hidden = !enabled;
  $("finish-contour").hidden = !enabled || tool!=="polygon";
  $("undo-contour-point").hidden = !enabled || tool!=="polygon";
  $("draw-help").textContent = tool === "polygon"
    ? "Кликните по границе. Enter или первая точка — завершить; Backspace — убрать точку; Escape — отменить. Shift + перетаскивание — перемещение."
    : "Обведите частицу перетаскиванием. Escape — отмена.";
  if (enabled && state.editingId!==null) $("draw-help").textContent = `Новый контур частицы № ${state.editingId}. ` + $("draw-help").textContent;
}

function addParticle(bbox) {
  if (state.busy || !state.result) return;
  if (isShape()) {
    try { commitShapePolygon(ellipsePolygon(bbox,state.originalImage.naturalWidth,state.originalImage.naturalHeight)); }
    catch (error) { showError(error.message); }
    return;
  }
  try {
    if (state.detections.length >= 5000 || state.additions.length >= 5000) throw new Error("Достигнут лимит 5000 выделений.");
    const id = Math.max(0, ...allSources().map((particle) => particle.id)) + 1;
    const particle = createManualParticle(id, bbox, state.originalImage.naturalWidth, state.originalImage.naturalHeight);
    state.additions.push(particle);
    setDrawing(false);
    renderReview();
    openParticle(id);
  } catch (error) { showError(error.message); }
}

function commitShapePolygon(polygon) {
  if (state.busy || !state.result || !isShape()) return;
  try {
    const width=state.originalImage.naturalWidth,height=state.originalImage.naturalHeight;
    let id=state.editingId;
    if (id!==null) state.overrides=setShapeReview(allSources(),state.overrides,id,{polygon_px:polygon,image_width:width,image_height:height});
    else {
      if (state.detections.length>=5000 || state.additions.length>=5000) throw new Error("Достигнут лимит 5000 выделений.");
      id=Math.max(0,...allSources().map(p=>p.index))+1;
      state.additions.push(createManualShape(id,polygon,width,height));
    }
    setDrawing(false);showError();renderReview();openParticle(id);
  } catch (error) { showError(error.message); }
}

function updateVisibleParticles() {
  const visible = getVisibleDetections();
  const isResult = state.view === "result" && Boolean(state.result);
  imageViewer.setDetections(isResult ? visible : []);
  $("particle-filters").hidden = !isResult;
  $("visible-count").textContent = `Показано: ${number.format(visible.length)} из ${number.format(state.detections.length)}`;
  const options = [new Option("Открыть частицу…", "")];
  for (const particle of visible) options.push(new Option(`№ ${particle.id} · ${labelFor(particle)}`, String(particle.id)));
  $("particle-picker").replaceChildren(...options);
  $("particle-picker").disabled = state.busy || !visible.length;
}

function setView(view) {
  if (view !== "result") setDrawing(false);
  state.view = view;
  const isResult = view === "result" && Boolean(state.result);
  $("view-result").setAttribute("aria-pressed", String(isResult));
  $("view-original").setAttribute("aria-pressed", String(!isResult));
  $("preview-image").alt = isResult ? "Изображение с рамками найденных частиц" : "Исходное изображение порошка";
  // All frames are a separate layer: filtering never leaves baked-in boxes behind.
  const source = state.result ? state.result.original_image : state.previewUrl;
  if ($("preview-image").getAttribute("src") !== source) $("preview-image").src = source;
  updateVisibleParticles();
}

function resetReview() {
  if ($("particle-dialog").open) $("particle-dialog").close();
  state.selectedId = null;
  state.overrides = new Map();
  state.additions = [];
  state.deletedIds = new Set();
  state.deletedHistory = [];
  setDrawing(false);
  state.filters = new Set(isShape() ? shapeStatuses : statuses);
  state.detections = [];
  for (const status of [...statuses, ...shapeStatuses]) $(`filter-${status}`).checked = true;
  $("manual-review-note").hidden = true;
  $("undo-delete").hidden = true;
}

function selectFile(file) {
  if (state.busy) return;
  if (!supported.test(file.name)) return showError("Выберите изображение JPG, PNG, WEBP, BMP или TIFF.");
  if (!file.size) return showError("Выбран пустой файл.");
  if (file.size > 20 * 1024 * 1024) return showError("Размер файла превышает 20 МБ.");

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  state.result = null;
  state.originalImage = null;
  resetReview();
  imageViewer.clear();
  showError();
  $("upload-title").textContent = file.name;
  $("drop-zone").title = "Нажмите, чтобы выбрать другое изображение";
  $("viewer").hidden = false;
  $("viewer-title").textContent = "Изображение";
  $("view-tabs").hidden = true;
  $("result-summary").hidden = true;
  $("coverage-summary").hidden = true;
  $("shape-summary").hidden = true;
  $("shape-explanation").hidden = true;
  $("contour-processing").hidden = true;
  $("limit-message").hidden = true;
  $("download-image").removeAttribute("href");
  $("preview-image").hidden = true;
  $("preview-placeholder").textContent = "Загрузка предпросмотра…";
  $("preview-placeholder").hidden = false;
  setView("original");
  updateControls();
}

$("file-input").addEventListener("change", (event) => {
  if (event.target.files[0]) selectFile(event.target.files[0]);
  event.target.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
  $("drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!state.busy) $("drop-zone").classList.add("drag-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  $("drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("drop-zone").classList.remove("drag-over");
    if (eventName === "drop" && event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]);
  });
}
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

$("view-result").addEventListener("click", () => setView("result"));
$("view-original").addEventListener("click", () => setView("original"));
$("preview-image").addEventListener("load", () => {
  $("preview-image").hidden = false;
  $("preview-placeholder").hidden = true;
});
$("preview-image").addEventListener("error", () => {
  $("preview-image").hidden = true;
  $("preview-placeholder").hidden = false;
  $("preview-placeholder").textContent = state.result
    ? "Не удалось показать изображение. Попробуйте распознать его снова."
    : "Предпросмотр появится после распознавания.";
});

for (const status of [...statuses, ...shapeStatuses]) {
  $(`filter-${status}`).addEventListener("change", (event) => {
    if (event.target.checked) state.filters.add(status);
    else state.filters.delete(status);
    updateVisibleParticles();
  });
}
$("particle-picker").addEventListener("change", (event) => {
  if (event.target.value) openParticle(Number(event.target.value));
  event.target.value = "";
});

function renderReview() {
  if (isShape()) {
    state.detections = shapeDetections(state.result.particles, state.deletedIds, state.additions, state.overrides);
    const summary = summarizeShapes(state.detections);
    const hasManualClasses = state.detections.some(particle => particle.manual_form);
    const modelClasses = state.result.classification_source === "model";
    $("rounded-rule").textContent = hasManualClasses ? "С учётом ручной разметки" : modelClasses ? "Класс YOLO: rounded" : `C ≥ ${circularityNumber.format(state.result.circularity_threshold)}`;
    $("irregular-rule").textContent = hasManualClasses ? "С учётом ручной разметки" : modelClasses ? "Класс YOLO: irregular" : `C < ${circularityNumber.format(state.result.circularity_threshold)}`;
    $("contour-processing").textContent = contourProcessingSummary(state.result, state.detections);
    $("rounded-count").textContent = number.format(summary.rounded);
    $("irregular-count").textContent = number.format(summary.irregular);
    $("edge-count").textContent = number.format(summary.edge);
    $("rounded-percent").textContent = summary.rounded_percent === null ? "—" : percent.format(summary.rounded_percent / 100);
    $("shape-denominator").textContent = `Среди ${number.format(summary.shape_denominator)} частиц вне границы`;
    $("particle-count").textContent = `Всего частиц: ${number.format(summary.total)}`;
    $("confidence-summary").textContent = `Медиана круглости: ${summary.median_circularity === null ? "—" : circularityNumber.format(summary.median_circularity)} · Медиана диаметра: ${summary.median_equivalent_diameter_px === null ? "—" : number.format(summary.median_equivalent_diameter_px) + " px"}`;
    $("manual-review-note").hidden = !state.deletedIds.size && !state.additions.length && !state.overrides.size;
    $("manual-review-note").textContent = `Добавлено вручную: ${state.additions.length}; ручных правок: ${state.overrides.size}; удалено: ${state.deletedIds.size}. Учтено в сводке и отчёте; действует до обновления страницы или нового анализа.`;
    $("undo-delete").hidden = !state.deletedHistory.length;
    updateVisibleParticles();
    if ($("particle-dialog").open) renderParticleDetail();
    return;
  }
  state.detections = reviewedDetections(state.result.detections, state.overrides, state.additions, state.deletedIds);
  const summary = summarizeCoverage(state.detections);
  $("coated-count").textContent = number.format(summary.coated_count);
  $("uncoated-count").textContent = number.format(summary.uncoated_count);
  $("uncertain-count").textContent = number.format(summary.uncertain_count);
  $("coverage-percent").textContent = summary.coverage_percent === null ? "—" : percent.format(summary.coverage_percent / 100);
  $("particle-count").textContent = summary.count ? `Всего частиц: ${number.format(summary.count)}` : "Активных выделений нет.";
  const scores = state.detections.map((particle) => particle.classifier_confidence).filter(Number.isFinite);
  $("confidence-summary").textContent = scores.length ? `Уверенность классификации: ${percent.format(scores.reduce((sum, score) => sum + score, 0) / scores.length)}` : "Уверенность классификации: —";
  $("manual-review-note").hidden = !summary.manual_count && !state.deletedIds.size;
  $("manual-review-note").textContent = `Ручных решений: ${summary.manual_count}; добавлено рамок: ${state.additions.length}; удалено: ${state.deletedIds.size}. Учтены в сводке и отчёте; действуют до обновления страницы или нового анализа.`;
  $("undo-delete").hidden = !state.deletedHistory.length;
  updateVisibleParticles();
  if ($("particle-dialog").open && state.selectedId !== null) renderParticleDetail();
}

async function renderResult(result) {
  const original = new Image();
  original.src = result.original_image;
  await original.decode();
  resetReview();
  state.result = result;
  state.originalImage = original;
  $("viewer-title").textContent = "Результат";
  $("view-tabs").hidden = false;
  $("result-summary").hidden = false;
  $("coverage-summary").hidden = isShape();
  $("shape-summary").hidden = !isShape();
  $("shape-explanation").hidden = !isShape();
  $("contour-processing").hidden = !isShape();
  if (isShape()) {
    const modelClasses = result.classification_source === "model";
    $("rounded-rule").textContent = modelClasses ? "Класс YOLO: rounded" : `C ≥ ${circularityNumber.format(result.circularity_threshold)}`;
    $("irregular-rule").textContent = modelClasses ? "Класс YOLO: irregular" : `C < ${circularityNumber.format(result.circularity_threshold)}`;
    $("shape-explanation").textContent = (modelClasses
      ? "Форму определяет обученная YOLO. Круглость C = 4πA / P² измеряется отдельно и не меняет класс модели."
      : "YOLO находит частицы; форма определяется по круглости C = 4πA / P². Чем ближе C к 1, тем круглее контур.")
      + ` Ручной класс имеет приоритет. Модель: ${result.model}${result.model_sha256 ? " · " + result.model_sha256.slice(0, 12) : ""}. Анализируется весь снимок, включая подписи.`;
    if (result.mask_quality_warning) $("shape-explanation").textContent += " " + result.mask_quality_warning;
  }
  if (!isShape()) $("uncertain-description").textContent = `Уверенность ниже ${percent.format(result.classifier_confidence_threshold)}`;
  $("limit-message").hidden = !(isShape() ? result.detection_limit_reached : result.statistics.detection_limit_reached);
  $("limit-message").textContent = `Детектор достиг лимита ${number.format(result.max_detections)} рамок до удаления дубликатов. На изображении могут быть другие частицы.`;
  $("download-image").href = "#";
  $("download-image").title = "Скачать видимые рамки с учётом фильтров и ручных решений";
  renderReview();
  setView("result");
}

function openParticle(id) {
  if (state.busy || !state.originalImage || !state.detections.some((particle) => particle.id === id)) return;
  state.selectedId = id;
  imageViewer.hideTooltip();
  renderParticleDetail();
  if (!$("particle-dialog").open) $("particle-dialog").showModal();
}

function renderParticleDetail() {
  const particle = state.detections.find((item) => item.id === state.selectedId);
  const source = isShape() ? particle : allSources().find((item) => item.id === state.selectedId);
  if (!particle || !source) return;
  $("particle-title").textContent = `Частица № ${particle.id}`;
  $("particle-status").textContent = `${labelFor(particle)}${particle.manual_review ? " · вручную" : ""}`;
  $("particle-status").dataset.status = isShape() ? particle.form : particle.coating_status;
  const modelClass = source.coating_class === "coated" ? "покрыта" : "непокрыта / недостаточное покрытие";
  $("particle-model-info").textContent = source.source === "manual" ? "Частица выделена вручную. Уверенность модели не определялась."
    : `Модель: ${modelClass} · уверенность ${percent.format(source.classifier_confidence)}`;
  $("particle-dimensions").textContent = `Рамка: ${number.format(particle.width_px)} × ${number.format(particle.height_px)} px`;
  if (isShape()) {
    const sourceInfo = particle.source === "manual" ? "Ручная разметка. Уверенность YOLO не определялась."
      : (particle.class_name === "rounded" || particle.class_name === "irregular" ? `Исходный класс YOLO: ${SHAPE_LABELS[particle.class_name]}` : "Исходная форма определена по круглости контура") + ` · Уверенность YOLO: ${percent.format(particle.confidence)}`;
    const semanticForm = particle.manual_form || (["rounded", "irregular"].includes(particle.class_name) ? particle.class_name : particle.circularity >= .8 ? "rounded" : "irregular");
    $("particle-model-info").textContent = `${sourceInfo} · Круглость: ${circularityNumber.format(particle.circularity)} из 1${particle.touches_border ? " · Неполный контур у границы; класс: " + SHAPE_LABELS[semanticForm] : ""}${particle.manual_contour ? " · Контур исправлен вручную" : ""}`;
    $("particle-dimensions").textContent = `Площадь: ${number.format(particle.area_px2)} px² · Периметр: ${number.format(particle.perimeter_px)} px · Эквивалентный диаметр: ${number.format(particle.equivalent_diameter_px)} px`;
    if (particle.contour_refined) $("particle-model-info").textContent += ` · Контур уточнён: удалено ${number.format(particle.removed_area_percent)}% площади маски`;
  }
  const crop = cropBounds(particle.bbox, state.originalImage.naturalWidth, state.originalImage.naturalHeight);
  const canvas = $("particle-crop");
  canvas.width = crop.width;
  canvas.height = crop.height;
  canvas.setAttribute("aria-label", `Крупный план частицы № ${particle.id}`);
  const context = canvas.getContext("2d");
  context.drawImage(state.originalImage, crop.left, crop.top, crop.width, crop.height, 0, 0, crop.width, crop.height);
  drawFrames(context, [particle], { x: -crop.left, y: -crop.top, lineWidth: Math.max(1, Math.min(crop.width, crop.height) / 200) });
  $("manual-review-actions").hidden = isShape() || (source.coating_status !== "uncertain" && source.source !== "manual");
  $("shape-review-actions").hidden = !isShape();
  $("mark-rounded").setAttribute("aria-pressed",String(particle.manual_form==="rounded"));
  $("mark-irregular").setAttribute("aria-pressed",String(particle.manual_form==="irregular"));
  $("reset-shape-review").hidden = !state.overrides.has(particle.id);
  $("manual-review-actions").querySelector("p").textContent = source.source === "manual" ? "Определите класс выделенной частицы:" : "Определите покрытие спорной частицы:";
  $("mark-coated").setAttribute("aria-pressed", String(particle.manual_review === "coated"));
  $("mark-uncoated").setAttribute("aria-pressed", String(particle.manual_review === "uncoated"));
  $("mark-uncertain").hidden = source.source !== "manual";
  $("mark-uncertain").setAttribute("aria-pressed", String(particle.manual_review === "uncertain"));
  $("reset-review").hidden = source.source === "manual" || !particle.manual_review;
  const visible = getVisibleDetections();
  const index = visible.findIndex((item) => item.id === particle.id);
  $("previous-particle").disabled = !visible.length || index === 0;
  $("next-particle").disabled = !visible.length || index === visible.length - 1;
}

function reviewParticle(status) {
  if (state.selectedId === null) return;
  state.overrides = setManualReview(allSources(), state.overrides, state.selectedId, status);
  renderReview();
}
$("mark-coated").addEventListener("click", () => reviewParticle("coated"));
$("mark-uncoated").addEventListener("click", () => reviewParticle("uncoated"));
$("mark-uncertain").addEventListener("click", () => reviewParticle("uncertain"));
$("reset-review").addEventListener("click", () => reviewParticle(null));
for (const form of ["rounded","irregular"]) $("mark-"+form).addEventListener("click",()=>{
  state.overrides=setShapeReview(allSources(),state.overrides,state.selectedId,{form});renderReview();
});
$("reset-shape-review").addEventListener("click",()=>{
  state.overrides=setShapeReview(allSources(),state.overrides,state.selectedId,null);renderReview();
});
$("redraw-shape").addEventListener("click",()=>{
  state.editingId=state.selectedId;$("particle-dialog").close();setView("result");setDrawing(true);$("image-stage").focus({preventScroll:true});
});
$("shape-draw-tool").addEventListener("change",()=>{if(imageViewer.drawing)setDrawing(true);});
$("finish-contour").addEventListener("click",()=>imageViewer.finishPolygon());
$("undo-contour-point").addEventListener("click",()=>{imageViewer.undoPolygonPoint();$("image-stage").focus({preventScroll:true});});
document.addEventListener("keydown",event=>{if(event.key==="Escape" && imageViewer.drawing){event.preventDefault();setDrawing(false);}});
$("draw-particle").addEventListener("click", () => {
  if (state.busy || !state.result) return;
  setDrawing(!imageViewer.drawing);
  $("image-stage").focus({ preventScroll: true });
});
$("delete-particle").addEventListener("click", () => {
  if (state.selectedId === null) return;
  const id = state.selectedId;
  state.deletedIds.add(id);
  state.deletedHistory.push(id);
  $("particle-dialog").close();
  state.selectedId = null;
  renderReview();
});
$("undo-delete").addEventListener("click", () => {
  const id = state.deletedHistory.pop();
  if (id === undefined) return;
  state.deletedIds.delete(id);
  renderReview();
});
$("close-particle").addEventListener("click", () => $("particle-dialog").close());
$("particle-dialog").addEventListener("close", () => {
  state.selectedId = null;
  $("image-stage").focus({ preventScroll: true });
});
$("particle-dialog").addEventListener("click", (event) => {
  if (event.target !== $("particle-dialog")) return;
  const box = $("particle-dialog").getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) $("particle-dialog").close();
});
for (const [id, step] of [["previous-particle", -1], ["next-particle", 1]]) {
  $(id).addEventListener("click", () => {
    const visible = getVisibleDetections();
    const index = visible.findIndex((particle) => particle.id === state.selectedId);
    const target = index < 0 ? (step > 0 ? 0 : visible.length - 1) : index + step;
    if (visible[target]) openParticle(visible[target].id);
  });
}

$("download-image").addEventListener("click", async (event) => {
  event.preventDefault();
  if (!state.originalImage || state.exporting || state.busy) return;
  state.exporting = true;
  const filename = `${state.file.name.replace(/\.[^.]+$/, "")}_detected.jpg`;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = state.originalImage.naturalWidth;
    canvas.height = state.originalImage.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(state.originalImage, 0, 0);
    drawFrames(context, getVisibleDetections(), { lineWidth: Math.max(2, Math.min(canvas.width, canvas.height) / 500) });
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", .95));
    if (!blob) throw new Error("Не удалось подготовить изображение для скачивания.");
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (error) {
    showError(error.message);
  } finally {
    state.exporting = false;
  }
});

$("generate-report").addEventListener("click", async () => {
  if (!state.result || state.busy || state.reportBusy) return;
  if (isShape() && (state.additions.length || state.overrides.size) && !state.manualShapeApiReady) {
    showError("Перезапустите backend: запущенная версия ещё не поддерживает отчёты с ручной разметкой формы."); return;
  }
  const filename = `${state.file.name.replace(/\.[^.]+$/, "")}_report.pdf`;
  const payload = isShape() ? {
    image: state.result.image, original_image: state.result.original_image,
    model: state.result.model, model_sha256: state.result.model_sha256, classification_source: state.result.classification_source,
    contour_processing_version: state.result.contour_processing_version,
    particles: shapeReportParticles(state.detections),
    timezone_offset_minutes: new Date().getTimezoneOffset(),
  } : {
    image: state.result.image, original_image: state.result.original_image, detections: state.detections,
    initial_detected_count: state.result.detections.length, manual_added_count: state.additions.length,
    deleted_count: state.deletedIds.size, model: state.result.model, classifier: state.result.classifier,
    classifier_confidence_threshold: state.result.classifier_confidence_threshold, timezone_offset_minutes: new Date().getTimezoneOffset(),
  };
  state.reportBusy = true;
  showError();
  updateControls();
  try {
    const response = await fetch(isShape() ? "/api/shape/report" : "/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(typeof result.detail === "string" ? result.detail : "Не удалось сформировать отчёт. Проверьте данные выделений.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (error) { showError(error.message); }
  finally { state.reportBusy = false; updateControls(); }
});

$("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file || state.busy || !state.ready) return;
  state.busy = true;
  setDrawing(false);
  showError();
  $("processing-overlay").hidden = false;
  $("image-stage").setAttribute("aria-busy", "true");
  updateControls();
  const data = new FormData();
  data.append("file", state.file);
  try {
    const response = await fetch(isShape() ? "/api/shape/predict" : "/api/predict", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) {
      const message = Array.isArray(result.detail) ? result.detail.map((item) => item.msg).join("; ") : result.detail;
      throw new Error(message || "Не удалось распознать изображение. Попробуйте снова.");
    }
    if (isShape() && !currentContourProcessing(result)) throw new Error("Получен результат старой обработки контуров. Перезапустите backend и повторите анализ.");
    await renderResult(result);
  } catch (error) {
    showError(error instanceof TypeError ? "Нет соединения с сервером. Проверьте подключение и повторите попытку." : error.message);
  } finally {
    state.busy = false;
    $("processing-overlay").hidden = true;
    $("image-stage").setAttribute("aria-busy", "false");
    updateControls();
  }
});

for (const mode of ["coverage", "shape"]) {
  $(`mode-${mode}`).addEventListener("click", () => {
    if (state.busy || mode === state.mode) return;
    state.mode = mode;
    state.ready = false;
    state.connectionMessage = "Подключение…";
    $("mode-coverage").setAttribute("aria-pressed", String(!isShape()));
    $("mode-shape").setAttribute("aria-pressed", String(isShape()));
    $("analysis-description").textContent = isShape()
      ? "Загрузите изображение порошка, чтобы найти контуры частиц и оценить их округлость."
      : "Загрузите изображение порошка, чтобы найти частицы и оценить их покрытие.";
    $("coverage-filters").hidden = isShape();
    $("shape-filters").hidden = !isShape();
    $("confidence-summary").title = isShape() ? "Геометрические показатели частиц вне границы изображения" : "Средняя уверенность классификатора покрытия";
    if (state.file) selectFile(state.file);
    else resetReview();
    showError();
    updateControls();
    checkHealth();
  });
}
checkHealth();
