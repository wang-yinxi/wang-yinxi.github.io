import { historyWindow, signedTime } from './case-readouts.mjs';
/** Orthographic projection shared by the occupancy map and motion annotations.
 * Robot frame: +x forward (up), +y left (left), +z up. No per-trajectory rescaling.
 */
export function projectBEV(point, grid, rect) {
  const [xmin, ymin] = grid.min, [xmax, ymax] = grid.max;
  return {
    x: rect.x + (ymax - point[1]) / (ymax - ymin) * rect.width,
    y: rect.y + (xmax - point[0]) / (xmax - xmin) * rect.height,
  };
}

/** Binary obstacle projection, omitting the lowest 15 cm so a floor does not fill the map. */
export function occupancyBEV(voxels, grid) {
  const [nx, ny, nz] = grid.dims;
  if (voxels.length !== nx * ny * nz) throw new Error('Occupancy dimensions do not match the voxel data.');
  const pixels = new Uint8ClampedArray(nx * ny * 4);
  const floor = grid.min[2] + .15;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) {
    let occupied = false;
    for (let z = 0; z < nz; z++) {
      const height = grid.min[2] + (z + .5) * (grid.max[2] - grid.min[2]) / nz;
      if (height > floor && voxels[x + nx * (y + ny * z)] !== 0) { occupied = true; break; }
    }
    const offset = ((nx - 1 - x) * ny + ny - 1 - y) * 4;
    pixels.set(occupied ? [43, 45, 51, 255] : [255, 248, 205, 255], offset);
  }
  return { width: ny, height: nx, pixels };
}

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}, text) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
const volumes = new Map();
async function getVolume(url) {
  if (!volumes.has(url)) {
    const request = fetch(url).then(response => {
      if (!response.ok) throw new Error(`Occupancy returned ${response.status}.`);
      return response.arrayBuffer();
    }).then(buffer => new Uint8Array(buffer)).catch(error => { volumes.delete(url); throw error; });
    volumes.set(url, request);
    if (volumes.size > 8) volumes.delete(volumes.keys().next().value);
  }
  return volumes.get(url);
}

export async function renderMotionBEV(panel, caseData, frame, modality, isCurrent) {
  const field = modality === 'ego' ? 'egoMotion' : 'targetMotion';
  const points = frame[field] || [];
  const times = frame.motionTimes || caseData.frames.map(item => item.time);
  const samples = points.map((point, i) => ({ point, time: times[i] })).filter(item =>
    Array.isArray(item.point) && item.point.length >= 2 && item.point.slice(0, 2).every(Number.isFinite) && Number.isFinite(item.time));
  if (!samples.length || frame[`${field}Available`] === false || frame.time > samples.at(-1).time + 1e-6) {
    const empty = document.createElement('div'); empty.className = 'readout-empty';
    const title = document.createElement('h4'); title.textContent = 'Trajectory unavailable';
    const copy = document.createElement('p'); copy.className = 'readout-empty-copy';
    copy.textContent = 'There is no aligned motion sample at this time.';
    empty.append(title, copy); panel.append(empty); return;
  }
  const loading = document.createElement('p'); loading.className = 'readout-loading'; loading.textContent = 'Loading the occupancy map…'; panel.append(loading);
  const volume = await getVolume(frame.voxels);
  if (!isCurrent()) return;
  loading.remove();
  const grid = caseData.grid;
  const map = occupancyBEV(volume, grid);
  const canvas = document.createElement('canvas'); canvas.width = map.width; canvas.height = map.height;
  canvas.getContext('2d').putImageData(new ImageData(map.pixels, map.width, map.height), 0, 0);
  const height = 460, width = height * (grid.max[1] - grid.min[1]) / (grid.max[0] - grid.min[0]);
  const rect = { x: (720 - width) / 2, y: 38, width, height };
  const svg = svgEl('svg', { viewBox: '0 0 720 560', class: 'motion-bev', role: 'img', 'aria-label': `${modality === 'ego' ? 'Robot' : 'Target'} motion on the same scene's bird’s-eye occupancy map` });
  svg.append(svgEl('desc', {}, 'The 3D occupancy is projected vertically. Obstacles are dark; free space is pale yellow. Motion uses the selected state’s robot coordinates.'));
  svg.append(svgEl('image', { ...rect, href: canvas.toDataURL(), preserveAspectRatio: 'none', style: 'image-rendering:pixelated' }));
  svg.append(svgEl('rect', { ...rect, fill: 'none', stroke: '#c9c9b2', 'stroke-width': 1 }));
  const style = { fill: '#647783', 'font-family': 'ui-monospace,monospace', 'font-size': 12 };
  for (let x = Math.ceil(grid.min[0]); x <= grid.max[0]; x++) {
    const p = projectBEV([x, grid.max[1]], grid, rect);
    svg.append(svgEl('text', { x: rect.x - 12, y: p.y + 4, 'text-anchor': 'end', ...style }, String(x)));
    svg.append(svgEl('line', { x1: rect.x - 4, y1: p.y, x2: rect.x, y2: p.y, stroke: '#8b9b9d' }));
  }
  for (let y = Math.ceil(grid.min[1]); y <= grid.max[1]; y++) {
    const p = projectBEV([grid.min[0], y], grid, rect);
    svg.append(svgEl('text', { x: p.x, y: rect.y + height + 20, 'text-anchor': 'middle', ...style }, String(y)));
  }
  svg.append(svgEl('text', { x: 360, y: 24, 'text-anchor': 'middle', fill: '#416071', 'font-size': 14 }, 'Forward ↑'));
  svg.append(svgEl('text', { x: 360, y: 544, 'text-anchor': 'middle', ...style }, '← Left (m) · robot frame'));
  const origin = projectBEV([0, 0], grid, rect);
  svg.append(svgEl('path', { d: `M${origin.x} ${origin.y - 8}l-6 13h12z`, fill: '#e44336', stroke: '#fff', 'stroke-width': 1.5 }));
  svg.append(svgEl('text', { x: origin.x + 13, y: origin.y + 15, fill: '#a83830', 'font-size': 12, stroke: '#fff', 'stroke-width': 3, 'paint-order': 'stroke' }, 'Robot'));
  const decodedField = modality === 'ego' ? 'decodedEgoMotion' : 'decodedTargetMotion';
  const decoded = frame[decodedField]?.filter(point => Array.isArray(point) && point.slice(0, 2).every(Number.isFinite));
  const paths = [{ points: samples.map(item => item.point), color: '#00a5b5', label: decoded?.length ? 'Reference trajectory' : modality === 'ego' ? `Past ego motion · ${historyWindow(frame, caseData.timeBasis)}` : 'Target motion' }];
  if (decoded?.length) paths.push({ points: decoded, color: '#ed701d', label: 'Latent readout' });
  let selected = 0;
  samples.forEach((sample, i) => { if (Math.abs(sample.time - frame.time) < Math.abs(samples[selected].time - frame.time)) selected = i; });
  for (const path of paths) {
    const projected = path.points.map(point => projectBEV(point, grid, rect));
    const line = projected.map(point => `${point.x},${point.y}`).join(' ');
    svg.append(svgEl('polyline', { points: line, fill: 'none', stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round' }));
    svg.append(svgEl('polyline', { points: line, fill: 'none', stroke: path.color, 'stroke-width': 3, 'stroke-linejoin': 'round', 'stroke-dasharray': '7 4' }));
    projected.forEach((point, i) => svg.append(svgEl('circle', { cx: point.x, cy: point.y, r: i === selected ? 6 : 3.5, fill: i === selected ? path.color : '#fff', stroke: path.color, 'stroke-width': 2 })));
    const end = projected.at(-1), before = projected.at(-2);
    if (before && Math.hypot(end.x-before.x,end.y-before.y)>2) {
      const angle = Math.atan2(end.y-before.y,end.x-before.x);
      const arms = [-.6,.6].map(offset => [end.x-Math.cos(angle+offset)*11,end.y-Math.sin(angle+offset)*11]);
      svg.append(svgEl('path',{d:`M${arms[0]} L${end.x},${end.y} L${arms[1]}`,fill:'none',stroke:path.color,'stroke-width':3}));
    }
  }
  const selectedPoint = projectBEV(paths.at(-1).points[Math.min(selected, paths.at(-1).points.length-1)], grid, rect);
  const timeText = caseData.timeBasis === 'video-frame' ? `Frame ${samples[selected].time}` : `${signedTime(samples[selected].time)} s`;
  svg.append(svgEl('text', { x: selectedPoint.x + 15, y: selectedPoint.y - 12, fill: '#385a6e', 'font-size': 14, 'font-weight': 600, stroke: '#fff', 'stroke-width': 4, 'paint-order': 'stroke' }, timeText));
  const stage = document.createElement('div'); stage.className = 'motion-bev-stage'; stage.append(svg);
  const legend = document.createElement('div'); legend.className = 'motion-key';
  for (const path of paths) { const key = document.createElement('span'); key.className = 'dashed'; key.style.setProperty('--motion-color', path.color); key.textContent = path.label; legend.append(key); }
  panel.append(stage, legend);
}
