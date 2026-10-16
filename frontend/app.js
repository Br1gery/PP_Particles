"use strict";

import { ImageViewer } from "./image-viewer.mjs";

const $ = (id) => document.getElementById(id);
const state = { file: null, previewUrl: null, result: null, ready: false, busy: false, connectionMessage: "Подключение…" };
const supported = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
const number = new Intl.NumberFormat("ru-RU");
const percent = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 1 });
let healthTimer;
const imageViewer = new ImageViewer({
  stage: $("image-stage"), image: $("preview-image"), controls: $("zoom-controls"),
  zoomIn: $("zoom-in"), zoomOut: $("zoom-out"), reset: $("zoom-reset"), value: $("zoom-value"),
  tooltip: $("particle-tooltip"),
});

function showError(message = "") {
  $("error-message").textContent = message;
  $("error-message").hidden = !message;
}

function updateControls() {
  imageViewer.setEnabled(!state.busy);
  $("analyze-button").disabled = state.busy || !state.file || !state.ready;
  $("analyze-button").textContent = state.busy ? "Распознавание…" : "Распознать частицы";
  $("file-input").disabled = state.busy;
  $("run-status").textContent = state.busy ? "Подождите, изображение обрабатывается."
    : !state.ready ? state.connectionMessage
    : state.result ? "Готово. Можно загрузить другое изображение."
    : state.file ? "Изображение готово к распознаванию."
    : "Выберите изображение для начала.";
}

async function checkHealth() {
  clearTimeout(healthTimer);
  try {
    const response = await fetch("/api/health", { cache: "no-store", signal: AbortSignal.timeout(10000) });
    const health = await response.json();
    state.ready = response.ok && health.status === "ready";
    state.connectionMessage = health.error || "Модель временно недоступна.";
  } catch {
    state.ready = false;
    state.connectionMessage = "Нет соединения с сервером. Проверьте, запущено ли приложение.";
  } finally {
    updateControls();
    healthTimer = setTimeout(checkHealth, state.ready ? 30000 : 10000);
  }
}

function setView(view) {
  const isResult = view === "result" && Boolean(state.result);
  $("view-result").setAttribute("aria-pressed", String(isResult));
  $("view-original").setAttribute("aria-pressed", String(!isResult));
  imageViewer.setDetections(isResult ? state.result.detections : []);
  $("preview-image").alt = isResult ? "Изображение с рамками найденных частиц" : "Исходное изображение порошка";
  $("preview-image").src = state.result
    ? (isResult ? state.result.annotated_image : state.result.original_image)
    : state.previewUrl;
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
  imageViewer.clear();
  showError();
  $("upload-title").textContent = file.name;
  $("drop-zone").title = "Нажмите, чтобы выбрать другое изображение";
  $("viewer").hidden = false;
  $("viewer-title").textContent = "Изображение";
  $("view-tabs").hidden = true;
  $("result-summary").hidden = true;
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

function renderResult(result) {
  state.result = result;
  $("viewer-title").textContent = "Результат";
  $("view-tabs").hidden = false;
  $("result-summary").hidden = false;
  $("particle-count").textContent = result.statistics.count
    ? `Найдено частиц: ${number.format(result.statistics.count)}`
    : "Частицы не обнаружены.";
  $("confidence-summary").textContent = result.statistics.mean_confidence === null
    ? "Средняя уверенность модели: —"
    : `Средняя уверенность модели: ${percent.format(result.statistics.mean_confidence)}`;
  $("limit-message").hidden = !result.statistics.detection_limit_reached;
  $("limit-message").textContent = `Достигнут лимит ${number.format(result.max_detections)} обнаружений. На изображении могут быть другие частицы.`;
  $("download-image").href = result.annotated_image;
  $("download-image").download = `${state.file.name.replace(/\.[^.]+$/, "")}_detected.jpg`;
  setView("result");
}

$("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file || state.busy || !state.ready) return;
  state.busy = true;
  showError();
  $("processing-overlay").hidden = false;
  $("image-stage").setAttribute("aria-busy", "true");
  updateControls();

  const data = new FormData();
  data.append("file", state.file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) {
      const message = Array.isArray(result.detail) ? result.detail.map((item) => item.msg).join("; ") : result.detail;
      throw new Error(message || "Не удалось распознать изображение. Попробуйте снова.");
    }
    renderResult(result);
  } catch (error) {
    showError(error instanceof TypeError ? "Нет соединения с сервером. Проверьте подключение и повторите попытку." : error.message);
  } finally {
    state.busy = false;
    $("processing-overlay").hidden = true;
    $("image-stage").setAttribute("aria-busy", "false");
    updateControls();
  }
});

checkHealth();
