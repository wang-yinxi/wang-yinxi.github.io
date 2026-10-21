import { renderMotionBEV } from './motion-bev.mjs';
import { historyWindow, signedTime, stateTime } from './case-readouts.mjs';
const SVG_NS = 'http://www.w3.org/2000/svg';
const MODALITIES = {
  semantic: { title: 'Semantic segmentation', detail: '64 × 64 · categorical readout', icon: '▦' },
  box: { title: 'Target localization', detail: 'Normalized 2D bounding box', icon: '⌖' },
  ego: { title: 'Past ego motion', detail: 'Preceding 0.6 s · selected robot frame', icon: '↗' },
  target: { title: 'Target motion', detail: 'Robot-anchor coordinates · meters', icon: '↗' },
  depth: { title: 'Depth reconstruction', detail: '32 × 32 · inverse-depth readout', icon: '▥' },
};
const CLASS_NAMES = ['Empty', 'Floor', 'Wall', 'Table', 'Chair', 'Cabinet', 'Plant', 'Person', 'Door', 'Sofa'];

export function normalizeBox(box) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return null;
  const clipped = box.map((value) => Math.max(0, Math.min(1, value)));
  return clipped[2] > clipped[0] && clipped[3] > clipped[1] ? clipped : null;
}

/** Project a normalized detection onto the content rect of an object-fit: contain image. */
export function projectBox(box, imageWidth, imageHeight, viewWidth, viewHeight) {
  const normalized = normalizeBox(box);
  if (!normalized || ![imageWidth, imageHeight, viewWidth, viewHeight].every((n) => Number.isFinite(n) && n > 0)) return null;
  const scale = Math.min(viewWidth / imageWidth, viewHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (viewWidth - width) / 2 + normalized[0] * width,
    y: (viewHeight - height) / 2 + normalized[1] * height,
    width: (normalized[2] - normalized[0]) * width,
    height: (normalized[3] - normalized[1]) * height,
  };
}

export function frameIndexAtTime(frames, time) {
  if (!frames?.length || !Number.isFinite(time)) return 0;
  return frames.reduce((best, frame, index) => {
    const distance = Number.isFinite(frame.time) ? Math.abs(frame.time - time) : Infinity;
    const bestDistance = Number.isFinite(frames[best]?.time) ? Math.abs(frames[best].time - time) : Infinity;
    return distance < bestDistance ? index : best;
  }, 0);
}

function validPoint(point) {
  return Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]);
}

function trajectory(frame, field) {
  const value = frame?.[field];
  return Array.isArray(value) ? value.filter(validPoint) : [];
}

/** Keep each source point paired with its actual timestamp, without extrapolation. */
export function motionSamples(caseData, frame, field) {
  const points = frame?.[field];
  if (!Array.isArray(points)) return [];
  const times = Array.isArray(frame.motionTimes) ? frame.motionTimes : caseData?.frames?.map(item => item.time) || [];
  return points.map((point, index) => ({ point, time: times[index] }))
    .filter(sample => validPoint(sample.point) && Number.isFinite(sample.time));
}

/** One metric scale for the complete time span, optionally scoped to one modality. */
export function motionBounds(caseData, width = 440, height = 280, field) {
  const points = [[0, 0]];
  for (const frame of caseData?.frames || []) {
    const fields = field ? [field] : ['egoMotion', 'targetMotion'];
    for (const name of fields) points.push(...trajectory(frame, name));
  }
  const forward = points.map((point) => point[0]);
  const left = points.map((point) => point[1]);
  const forwardMin = Math.min(...forward);
  const forwardMax = Math.max(...forward);
  const leftMin = Math.min(...left);
  const leftMax = Math.max(...left);
  const centerForward = (forwardMin + forwardMax) / 2;
  const centerLeft = (leftMin + leftMax) / 2;
  const ratio = width > 0 && height > 0 ? width / height : 1;
  let forwardSpan = Math.max(1, forwardMax - forwardMin) * 1.3;
  let leftSpan = Math.max(1, leftMax - leftMin) * 1.3;
  if (leftSpan / forwardSpan < ratio) leftSpan = forwardSpan * ratio;
  else forwardSpan = leftSpan / ratio;
  return {
    leftMin: centerLeft - leftSpan / 2,
    leftMax: centerLeft + leftSpan / 2,
    forwardMin: centerForward - forwardSpan / 2,
    forwardMax: centerForward + forwardSpan / 2,
  };
}

export function projectMotion(point, bounds, width, height) {
  return {
    x: ((point[1] - bounds.leftMin) / (bounds.leftMax - bounds.leftMin)) * width,
    y: height - ((point[0] - bounds.forwardMin) / (bounds.forwardMax - bounds.forwardMin)) * height,
  };
}

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function svgEl(tag, attrs = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, String(value));
  return element;
}

function svgText(parent, x, y, text, className, extra = {}) {
  const element = svgEl('text', { x, y, class: className, ...extra });
  element.textContent = text;
  parent.append(element);
  return element;
}

function timeLabel(time) {
  return `${signedTime(Number.isFinite(time) ? time : 0)} s`;
}

function ticks(min, max) {
  const rough = (max - min) / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].find((value) => value * magnitude >= rough) * magnitude;
  const result = [];
  for (let tick = Math.ceil(min / step) * step; tick <= max + step / 100; tick += step) {
    result.push(Math.abs(tick) < step / 100 ? 0 : tick);
  }
  return result;
}

function metricLabel(value) {
  return Number(value.toFixed(2)).toString();
}

function paletteEntries(palette) {
  if (!Array.isArray(palette)) return [];
  return palette.map((entry, index) => {
    const id = Number.isFinite(entry?.id) ? entry.id : index;
    const color = entry?.color ?? entry?.rgb ?? entry;
    return {
      id,
      name: entry?.name || entry?.label || CLASS_NAMES[id] || `Class ${id}`,
      color: Array.isArray(color) ? `rgb(${color.slice(0, 3).join(',')})` : String(color),
    };
  }).filter((entry) => entry.id !== 0);
}

export class ReadoutPanel {
  constructor(container, { onError } = {}) {
    if (!container?.replaceChildren) throw new TypeError('ReadoutPanel requires a container element.');
    this.container = container;
    this.onError = onError;
    this.version = 0;
    this.pendingImage = null;
    this.destroyed = false;
  }

  render(caseData, frame, modality) {
    if (this.destroyed) return;
    this.cancelImage();
    const version = ++this.version;
    this.lastRender = [caseData, frame, modality];
    const info = MODALITIES[modality];
    const panel = el('section', 'readout-panel');
    panel.setAttribute('aria-label', `${info?.title || 'Scientific readout'} at ${timeLabel(frame?.time)}`);
    this.container.replaceChildren(panel);
    if (!info || !frame) {
      this.showError(panel, new Error('This readout is unavailable.'), version);
      return;
    }
    const header = el('div', 'readout-header');
    const heading = el('div', 'readout-heading');
    const icon = el('span', 'readout-icon', info.icon);
    icon.setAttribute('aria-hidden', 'true');
    const title = el('div', 'readout-title-group');
    title.append(el('h3', 'readout-title', info.title), el('p', 'readout-subtitle', info.detail));
    heading.append(icon, title);
    header.append(heading, el('span', 'readout-time', modality === 'ego' ? historyWindow(frame, caseData.timeBasis) : stateTime(caseData, frame)));
    panel.append(header);

    if ((modality === 'box' || modality === 'target') && caseData?.hasTarget === false) {
      this.renderNotApplicable(panel);
    } else if (modality === 'ego' || modality === 'target') {
      this.renderMotion(panel, caseData, frame, modality);
    } else {
      this.renderImage(panel, caseData, frame, modality, version);
    }
  }

  cancelImage() {
    if (this.pendingImage) {
      this.pendingImage.onload = null;
      this.pendingImage.onerror = null;
      this.pendingImage = null;
    }
  }

  clear() {
    this.version += 1;
    this.cancelImage();
    this.lastRender = null;
    this.container.replaceChildren();
  }

  renderNotApplicable(panel) {
    const empty = el('div', 'readout-empty');
    const icon = el('div', 'readout-empty-icon', '—');
    icon.setAttribute('aria-hidden', 'true');
    empty.append(icon, el('h4', 'readout-empty-title', 'Not applicable'));
    empty.append(el('p', 'readout-empty-copy', 'This scene has no designated target.'));
    empty.append(el('span', 'readout-empty-tag', 'Target-free navigation'));
    panel.append(empty);
  }

  renderImage(panel, caseData, frame, modality, version) {
    const stage = el('div', `readout-image-stage readout-image-stage--${modality}`);
    const loading = el('div', 'readout-loading', 'Loading readout…');
    loading.setAttribute('role', 'status');
    stage.append(loading);
    panel.append(stage);
    const path = modality === 'box' ? frame.scene : frame[modality];
    if (typeof path !== 'string' || !path) {
      this.showError(stage, new Error('The image for this frame is unavailable.'), version);
      return;
    }
    const image = el('img', `readout-image${modality === 'box' ? '' : ' readout-image--pixelated'}`);
    image.alt = `${MODALITIES[modality].title}, ${caseData?.title || 'scene'}, ${timeLabel(frame.time)}`;
    image.decoding = 'async';
    image.draggable = false;
    this.pendingImage = image;
    let settled = false;
    const loaded = () => {
      if (settled || this.destroyed || version !== this.version) return;
      settled = true;
      this.pendingImage = null;
      image.onload = null;
      image.onerror = null;
      loading.remove();
      stage.append(image);
      if (modality === 'box') this.renderBox(stage, frame, image.naturalWidth, image.naturalHeight);
    };
    const decoded = () => {
      if (typeof image.decode === 'function') image.decode().then(loaded, loaded);
      else loaded();
    };
    image.onload = decoded;
    image.onerror = () => {
      if (settled || this.destroyed || version !== this.version) return;
      settled = true;
      this.pendingImage = null;
      image.onload = null;
      image.onerror = null;
      this.showError(stage, new Error('The readout image could not be loaded.'), version);
    };
    image.src = path;
    if (image.complete && image.naturalWidth > 0) decoded();
    if (modality === 'semantic') this.renderSemanticLegend(panel, caseData);
    if (modality === 'depth') this.renderDepthLegend(panel, caseData, frame);
    if (modality === 'box') {
      const box = normalizeBox(frame.box);
      panel.append(el('p', 'readout-footnote', box
        ? 'Target box on the scene at the selected time.'
        : 'No target box is available at this time.'));
    }
  }

  renderBox(stage, frame, width, height) {
    // Source figures already carry the target annotation. Never draw a second,
    // independently scaled box over those immutable RGB pixels.
    if (frame.boxEmbedded || frame.boxProvenance?.startsWith('Source ')) return;
    const box = normalizeBox(frame.box);
    if (!box || !width || !height) return;
    // The image and SVG share the same contain operation and intrinsic aspect ratio.
    const overlay = svgEl('svg', {
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: 'xMidYMid meet',
      class: 'readout-box-overlay',
      role: 'img',
      'aria-label': `Target bounding box at ${timeLabel(frame.time)}`,
    });
    const projected = projectBox(box, width, height, width, height);
    const { x, y } = projected;
    overlay.append(svgEl('rect', {
      ...projected,
      rx: 2,
      class: 'readout-box-rect',
      'vector-effect': 'non-scaling-stroke',
    }));
    const labelWidth = Math.min(width * 0.22, 138);
    const labelHeight = Math.max(18, height * 0.05);
    const labelY = y > labelHeight + 5 ? y - labelHeight - 3 : y + 3;
    overlay.append(svgEl('rect', {
      x, y: labelY, width: labelWidth, height: labelHeight, rx: 3, class: 'readout-box-label-bg',
    }));
    svgText(overlay, x + labelWidth / 2, labelY + labelHeight * 0.68, 'TARGET', 'readout-box-label', {
      'text-anchor': 'middle',
      'font-size': labelHeight * 0.55,
    });
    stage.append(overlay);
  }

  renderSemanticLegend(panel, caseData) {
    const entries = paletteEntries(caseData?.palette);
    const legend = el('div', 'readout-semantic-legend');
    legend.setAttribute('aria-label', 'Semantic classes');
    for (const entry of entries) {
      const item = el('span', 'readout-class');
      const swatch = el('span', 'readout-swatch');
      swatch.style.backgroundColor = entry.color;
      item.append(swatch, document.createTextNode(entry.name));
      legend.append(item);
    }
    if (entries.length) panel.append(legend);
    panel.append(el('p', 'readout-footnote', '64 × 64 semantic readout · native grid preserved.'));
  }

  renderDepthLegend(panel, caseData, frame) {
    const legend = el('div', 'readout-depth-legend');
    const range = frame.depthRange || caseData?.depthRange;
    const labels = el('div', 'readout-depth-labels');
    const hasRange = Array.isArray(range) && range.length === 2 && range.every(Number.isFinite);
    labels.append(el('span', '', hasRange ? `${metricLabel(range[1])} m · Far` : 'Far'));
    labels.append(el('span', '', hasRange ? `Near · ${metricLabel(range[0])} m` : 'Near'));
    legend.append(el('div', 'readout-depth-ramp'), labels);
    panel.append(legend, el('p', 'readout-footnote',
      'Brighter colors indicate nearer surfaces. Native 32 × 32 grid preserved.'));
  }

  renderMotion(panel, caseData, frame, modality) {
    const version = this.version;
    renderMotionBEV(panel, caseData, frame, modality, () => !this.destroyed && this.version === version)
      .catch(error => this.showError(panel, error, version));
  }

  showError(container, error, version) {
    if (this.destroyed || version !== this.version) return;
    const message = el('div', 'readout-error');
    message.setAttribute('role', 'status');
    message.append(el('strong', '', 'Readout unavailable'), el('p', '', error.message));
    const retry = el('button', 'readout-retry', 'Retry');
    retry.type = 'button';
    retry.addEventListener('click', () => {
      if (!this.destroyed && this.lastRender) this.render(...this.lastRender);
    });
    message.append(retry);
    container.replaceChildren(message);
    if (typeof this.onError === 'function') this.onError(error);
  }

  destroy() {
    this.destroyed = true;
    this.clear();
  }
}
