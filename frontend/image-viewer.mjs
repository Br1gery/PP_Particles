const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.25;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function fitSize(imageWidth, imageHeight, viewportWidth, viewportHeight) {
  if (Math.min(imageWidth, imageHeight, viewportWidth, viewportHeight) <= 0) {
    return { width: 0, height: 0 };
  }
  const ratio = Math.min(1, viewportWidth / imageWidth, viewportHeight / imageHeight);
  return { width: imageWidth * ratio, height: imageHeight * ratio };
}

export function constrainOffset(offset, imageSize, viewportSize) {
  return imageSize <= viewportSize ? (viewportSize - imageSize) / 2
    : clamp(offset, viewportSize - imageSize, 0);
}

export function zoomAt(view, requestedScale, anchor, image, viewport) {
  const scale = clamp(requestedScale, MIN_ZOOM, MAX_ZOOM);
  // Keep the image point under the cursor in place while changing its scale.
  const x = anchor.x - (anchor.x - view.x) * scale / view.scale;
  const y = anchor.y - (anchor.y - view.y) * scale / view.scale;
  return {
    scale,
    x: constrainOffset(x, image.width * scale, viewport.width),
    y: constrainOffset(y, image.height * scale, viewport.height),
  };
}

export function detectionAt(point, detections, view, displayedSize, originalSize) {
  const width = displayedSize.width * view.scale;
  const height = displayedSize.height * view.scale;
  if (width <= 0 || height <= 0) return null;
  const x = (point.x - view.x) * originalSize.width / width;
  const y = (point.y - view.y) * originalSize.height / height;
  if (x < 0 || y < 0 || x > originalSize.width || y > originalSize.height) return null;
  let match = null;
  let smallestArea = Infinity;
  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox;
    const area = (x2 - x1) * (y2 - y1);
    // Prefer the smaller particle when boxes overlap, so it remains reachable.
    if (area > 0 && area < smallestArea && x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      match = detection;
      smallestArea = area;
    }
  }
  return match;
}

export class ImageViewer {
  constructor({ stage, image, controls, zoomIn, zoomOut, reset, value, tooltip }) {
    Object.assign(this, { stage, image, controls, zoomIn, zoomOut, resetButton: reset, value, tooltip });
    this.view = { scale: 1, x: 0, y: 0 };
    this.baseSize = { width: 0, height: 0 };
    this.loaded = false;
    this.enabled = true;
    this.drag = null;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.detections = [];
    this.hoverPoint = null;

    image.addEventListener("load", () => {
      const changed = this.imageWidth !== image.naturalWidth || this.imageHeight !== image.naturalHeight;
      this.imageWidth = image.naturalWidth;
      this.imageHeight = image.naturalHeight;
      this.loaded = true;
      this.layout(changed);
    });
    image.addEventListener("error", () => this.clear());
    zoomIn.addEventListener("click", () => this.zoom(this.view.scale * ZOOM_STEP));
    zoomOut.addEventListener("click", () => this.zoom(this.view.scale / ZOOM_STEP));
    reset.addEventListener("click", () => this.reset());

    stage.addEventListener("wheel", (event) => {
      if (!this.canInteract()) return;
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? stage.clientHeight : 1);
      this.zoom(this.view.scale * Math.exp(-delta * .002), this.point(event));
    }, { passive: false });
    stage.addEventListener("dblclick", (event) => {
      if (!this.canInteract()) return;
      event.preventDefault();
      this.zoom(this.view.scale > 1 ? 1 : 2, this.point(event));
    });
    stage.addEventListener("pointerdown", (event) => {
      if (!this.canInteract() || this.view.scale <= 1 || event.button !== 0 || this.drag) return;
      event.preventDefault();
      this.hideTooltip();
      this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, start: { ...this.view } };
      stage.setPointerCapture(event.pointerId);
      stage.classList.add("panning");
    });
    stage.addEventListener("pointermove", (event) => {
      if (!this.drag) {
        if (event.pointerType !== "touch") {
          this.hoverPoint = this.point(event);
          this.updateTooltip();
        }
        return;
      }
      if (this.drag.id !== event.pointerId) return;
      this.view.x = this.drag.start.x + event.clientX - this.drag.x;
      this.view.y = this.drag.start.y + event.clientY - this.drag.y;
      this.render();
    });
    stage.addEventListener("pointerleave", () => this.hideTooltip());
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      stage.addEventListener(name, (event) => {
        if (this.drag?.id === event.pointerId) this.stopDrag();
      });
    }
    stage.addEventListener("keydown", (event) => {
      if (!this.canInteract()) return;
      if (["+", "=", "-", "0", "Escape"].includes(event.key)) {
        event.preventDefault();
        if (event.key === "+" || event.key === "=") this.zoom(this.view.scale * ZOOM_STEP);
        else if (event.key === "-") this.zoom(this.view.scale / ZOOM_STEP);
        else this.reset();
      } else if (this.view.scale > 1 && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
        event.preventDefault();
        this.view.x += event.key === "ArrowLeft" ? 50 : event.key === "ArrowRight" ? -50 : 0;
        this.view.y += event.key === "ArrowUp" ? 50 : event.key === "ArrowDown" ? -50 : 0;
        this.render();
      }
    });
    this.resizeObserver = new ResizeObserver(() => this.layout());
    this.resizeObserver.observe(stage);
    this.updateControls();
  }

  viewport() { return { width: this.stage.clientWidth, height: this.stage.clientHeight }; }
  canInteract() { return this.loaded && this.enabled; }
  point(event) {
    const rect = this.stage.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  layout(reset = false) {
    if (!this.loaded) return;
    const viewport = this.viewport();
    if (!viewport.width || !viewport.height) return;
    // Preserve the center of the viewed area when resizing or switching images.
    const centerX = this.baseSize.width ? (this.lastWidth / 2 - this.view.x) / (this.baseSize.width * this.view.scale) : .5;
    const centerY = this.baseSize.height ? (this.lastHeight / 2 - this.view.y) / (this.baseSize.height * this.view.scale) : .5;
    this.baseSize = fitSize(this.imageWidth, this.imageHeight, viewport.width, viewport.height);
    this.lastWidth = viewport.width;
    this.lastHeight = viewport.height;
    this.image.style.width = `${this.baseSize.width}px`;
    this.image.style.height = `${this.baseSize.height}px`;
    if (reset) this.view.scale = 1;
    this.view.x = viewport.width / 2 - centerX * this.baseSize.width * this.view.scale;
    this.view.y = viewport.height / 2 - centerY * this.baseSize.height * this.view.scale;
    this.render();
  }

  zoom(scale, anchor) {
    if (!this.canInteract()) return;
    this.stopDrag();
    const viewport = this.viewport();
    this.view = zoomAt(this.view, scale, anchor || { x: viewport.width / 2, y: viewport.height / 2 }, this.baseSize, viewport);
    this.render();
  }

  reset() {
    this.stopDrag();
    this.view = { scale: 1, x: 0, y: 0 };
    this.render();
  }

  clear() {
    this.loaded = false;
    this.setDetections([]);
    this.reset();
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    if (!enabled) this.stopDrag();
    this.updateControls();
    this.updateTooltip();
  }

  setDetections(detections) {
    this.detections = detections;
    this.hideTooltip();
  }

  hideTooltip() {
    this.hoverPoint = null;
    this.tooltip.hidden = true;
  }

  updateTooltip() {
    const match = this.canInteract() && !this.drag && this.hoverPoint
      ? detectionAt(this.hoverPoint, this.detections, this.view, this.baseSize,
        { width: this.imageWidth, height: this.imageHeight })
      : null;
    this.tooltip.hidden = !match;
    if (!match) return;
    this.tooltip.textContent = `№ ${match.id}`;
    const viewport = this.viewport();
    const left = clamp(this.hoverPoint.x + 12, 4, Math.max(4, viewport.width - this.tooltip.offsetWidth - 4));
    const top = clamp(this.hoverPoint.y + 12, 4, Math.max(4, viewport.height - this.tooltip.offsetHeight - 4));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
  }

  stopDrag() {
    const drag = this.drag;
    this.drag = null;
    this.stage.classList.remove("panning");
    if (drag && this.stage.hasPointerCapture(drag.id)) this.stage.releasePointerCapture(drag.id);
  }

  render() {
    const viewport = this.viewport();
    this.view.x = constrainOffset(this.view.x, this.baseSize.width * this.view.scale, viewport.width);
    this.view.y = constrainOffset(this.view.y, this.baseSize.height * this.view.scale, viewport.height);
    this.image.style.transform = `translate(${this.view.x}px, ${this.view.y}px) scale(${this.view.scale})`;
    this.stage.classList.toggle("zoomed", this.loaded && this.view.scale > 1);
    this.updateControls();
    this.updateTooltip();
  }

  updateControls() {
    this.controls.hidden = !this.loaded;
    this.zoomIn.disabled = !this.canInteract() || this.view.scale >= MAX_ZOOM;
    this.zoomOut.disabled = !this.canInteract() || this.view.scale <= MIN_ZOOM;
    this.resetButton.disabled = !this.canInteract() || this.view.scale <= MIN_ZOOM;
    this.value.textContent = `${Math.round(this.view.scale * 100)}%`;
  }
}
