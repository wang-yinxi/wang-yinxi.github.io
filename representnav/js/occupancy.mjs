// Native WebGL 2 occupancy rendering. World axes: x forward, y left, z up.
// Matrices are column-major and camera angles are in radians.

function validateGrid(grid) {
  if (!grid?.dims || grid.dims.length !== 3 ||
      !grid.dims.every(value => Number.isInteger(value) && value > 0)) {
    throw new Error('Occupancy grid dimensions must be three positive integers.');
  }
  if (!grid.min || !grid.max || grid.min.length !== 3 || grid.max.length !== 3 ||
      !grid.min.every((value, axis) =>
        Number.isFinite(value) && Number.isFinite(grid.max[axis]) && grid.max[axis] > value)) {
    throw new Error('Occupancy grid bounds must be finite and increasing.');
  }
}

/** Keep occupied cells bordering empty space or a volume boundary. */
export function extractSurfaceVoxels(voxels, grid) {
  validateGrid(grid);
  const [nx, ny, nz] = grid.dims;
  const slice = nx * ny;
  if (!(voxels instanceof Uint8Array) || voxels.length !== slice * nz) {
    throw new Error(`Occupancy data length must be ${slice * nz} bytes.`);
  }
  const voxelSize = grid.dims.map((value, axis) => (grid.max[axis] - grid.min[axis]) / value);
  const positions = new Float32Array(voxels.length * 3);
  let count = 0;
  let occupiedCount = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const index = x + nx * y + slice * z;
        if (voxels[index] === 0) continue;
        occupiedCount++;
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1 &&
            voxels[index - 1] && voxels[index + 1] &&
            voxels[index - nx] && voxels[index + nx] &&
            voxels[index - slice] && voxels[index + slice]) continue;
        positions[count * 3] = grid.min[0] + (x + 0.5) * voxelSize[0];
        positions[count * 3 + 1] = grid.min[1] + (y + 0.5) * voxelSize[1];
        positions[count * 3 + 2] = grid.min[2] + (z + 0.5) * voxelSize[2];
        count++;
      }
    }
  }
  return { positions: positions.slice(0, count * 3), voxelSize, count, occupiedCount };
}

export function multiplyMatrices(a, b) {
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++) result[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return result;
}

function normalize(vector) {
  const length = Math.hypot(...vector) || 1;
  return vector.map(value => value / length);
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a, b) {
  return a.reduce((sum, value, axis) => sum + value * b[axis], 0);
}

export function lookAtMatrix(eye, target, up = [0, 0, 1]) {
  const z = normalize(eye.map((value, axis) => value - target[axis]));
  let x = cross(up, z);
  // A pole view needs a different up vector to avoid a degenerate basis.
  if (Math.hypot(...x) < 1e-6) x = cross([0, 1, 0], z);
  x = normalize(x);
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

export function perspectiveMatrix(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * range, -1,
    0, 0, 2 * far * near * range, 0,
  ]);
}

export function orthographicMatrix(left, right, bottom, top, near, far) {
  return new Float32Array([
    2 / (right - left), 0, 0, 0,
    0, 2 / (top - bottom), 0, 0,
    0, 0, -2 / (far - near), 0,
    -(right + left) / (right - left), -(top + bottom) / (top - bottom),
    -(far + near) / (far - near), 1,
  ]);
}

function cubeVertices() {
  const vertices = [];
  const faces = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
  ];
  for (const [normal, u, v] of faces) {
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]]) {
      vertices.push(...normal.map((value, i) => (value + a * u[i] + b * v[i]) * 0.5), ...normal);
    }
  }
  return new Float32Array(vertices);
}

const VOXEL_VERTEX = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_center;
uniform mat4 u_matrix;
uniform vec3 u_voxelSize;
uniform vec2 u_zBounds;
out vec3 v_normal;
out float v_height;
void main() {
  vec3 world = a_center + a_position * u_voxelSize * 0.98;
  gl_Position = u_matrix * vec4(world, 1.0);
  v_normal = a_normal;
  v_height = clamp((a_center.z - u_zBounds.x) / (u_zBounds.y - u_zBounds.x), 0.0, 1.0);
}`;

const VOXEL_FRAGMENT = `#version 300 es
precision highp float;
in vec3 v_normal;
in float v_height;
uniform bool u_bev;
out vec4 color;
vec3 viridis(float t) {
  vec3 a = vec3(0.267, 0.005, 0.329);
  vec3 b = vec3(0.230, 0.322, 0.546);
  vec3 c = vec3(0.128, 0.567, 0.551);
  vec3 d = vec3(0.369, 0.789, 0.383);
  vec3 e = vec3(0.993, 0.906, 0.144);
  if (t < 0.25) return mix(a, b, t * 4.0);
  if (t < 0.50) return mix(b, c, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c, d, (t - 0.50) * 4.0);
  return mix(d, e, (t - 0.75) * 4.0);
}
void main() {
  if (u_bev) { color = vec4(43.0 / 255.0, 45.0 / 255.0, 51.0 / 255.0, 1.0); return; }
  float light = 0.65 + 0.35 * max(dot(v_normal, normalize(vec3(-1.0, -1.8, 3.5))), 0.0);
  color = vec4(mix(viridis(v_height) * light, vec3(0.96, 0.97, 0.97), 0.045), 1.0);
}`;

const LINE_VERTEX = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
uniform mat4 u_matrix;
out vec3 v_color;
void main() {
  gl_Position = u_matrix * vec4(a_position, 1.0);
  v_color = a_color;
}`;

const LINE_FRAGMENT = `#version 300 es
precision highp float;
in vec3 v_color;
out vec4 color;
void main() { color = vec4(v_color, 1.0); }`;

function makeProgram(gl, vertexSource, fragmentSource) {
  const shaders = [];
  const program = gl.createProgram();
  try {
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader = gl.createShader(type);
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(`Could not compile occupancy renderer: ${gl.getShaderInfoLog(shader)}`);
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Could not initialize occupancy renderer: ${gl.getProgramInfoLog(program)}`);
    }
    return program;
  } catch (error) {
    gl.deleteProgram(program);
    throw error;
  } finally {
    shaders.forEach(shader => gl.deleteShader(shader));
  }
}

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const DEFAULT_AZIMUTH = -2.35;
const DEFAULT_ELEVATION = 0.94;

export class OccupancyViewer {
  constructor(canvas, { onError = () => {}, onViewChange = () => {}, showMotion = false } = {}) {
    this.canvas = canvas;
    this.showMotion = showMotion;
    this.onError = onError;
    this.onViewChange = onViewChange;
    this.view = 'orbit';
    this.azimuth = DEFAULT_AZIMUTH;
    this.elevation = DEFAULT_ELEVATION;
    this.distance = 18;
    this.voxelCount = 0;
    this.active = true;
    this.inView = true;
    this.destroyed = false;
    this.dirty = false;
    this.raf = 0;
    this.requestId = 0;
    this.listeners = [];
    this.resources = [];
    this.gl = null;
    this.ready = false;
    this.grid = null;
    this.caseKey = null;
    this.controller = null;
    this.drag = null;
    this.canvas.style.touchAction = 'none';
    this.canvas.style.cursor = 'grab';
    if (canvas.tabIndex < 0) canvas.tabIndex = 0;
    if (!canvas.getAttribute('aria-label')) {
      canvas.setAttribute('aria-label', 'Interactive 3D occupancy. Drag to rotate, scroll to zoom, or use arrow and plus/minus keys.');
    }
    this._bindEvents();
    try {
      this.gl = canvas.getContext('webgl2', { alpha: false, antialias: true, depth: true, powerPreference: 'low-power' });
      if (!this.gl) throw new Error('WebGL 2 is unavailable. Showing the recorded occupancy image.');
      this._initializeGL();
    } catch (error) {
      this._fail(error);
    }
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this._invalidate());
      this.resizeObserver.observe(canvas);
    } else {
      this._listen(globalThis, 'resize', () => this._invalidate());
    }
    if (typeof IntersectionObserver !== 'undefined') {
      this.intersectionObserver = new IntersectionObserver(entries => {
        this.inView = entries[0].isIntersecting;
        if (this.inView) this._invalidate();
        else this._cancelDraw();
      });
      this.intersectionObserver.observe(canvas);
    }
    this._invalidate();
  }

  _listen(target, name, callback, options) {
    target.addEventListener(name, callback, options);
    this.listeners.push(() => target.removeEventListener(name, callback, options));
  }

  _bindEvents() {
    this._listen(this.canvas, 'pointerdown', event => {
      if (!this.active || event.button !== 0) return;
      this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.focus({ preventScroll: true });
      this.canvas.style.cursor = 'grabbing';
    });
    this._listen(this.canvas, 'pointermove', event => {
      if (!this.drag || this.drag.id !== event.pointerId) return;
      this.azimuth -= (event.clientX - this.drag.x) * 0.008;
      this.elevation = clamp(this.elevation + (event.clientY - this.drag.y) * 0.007, 0.06, Math.PI / 2 - 0.015);
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      this.view = 'orbit';
      this._cameraChanged();
    });
    const endDrag = event => {
      if (!this.drag || this.drag.id !== event.pointerId) return;
      this.drag = null;
      this.canvas.style.cursor = 'grab';
      if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    };
    this._listen(this.canvas, 'pointerup', endDrag);
    this._listen(this.canvas, 'pointercancel', endDrag);
    this._listen(this.canvas, 'lostpointercapture', endDrag);
    this._listen(this.canvas, 'wheel', event => {
      if (!this.active || !this.grid) return;
      event.preventDefault();
      const scale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.canvas.clientHeight : 1;
      this._zoom(Math.exp(clamp(event.deltaY * scale, -250, 250) * 0.0017));
    }, { passive: false });
    this._listen(this.canvas, 'keydown', event => {
      if (!this.active || !this.grid) return;
      const key = event.key;
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', '_', 'Home'].includes(key)) return;
      event.preventDefault();
      if (key === 'Home') return this.reset();
      if (key === '+' || key === '=') return this._zoom(0.9);
      if (key === '-' || key === '_') return this._zoom(1.1);
      if (key === 'ArrowLeft') this.azimuth += 0.09;
      if (key === 'ArrowRight') this.azimuth -= 0.09;
      if (key === 'ArrowUp') this.elevation += 0.075;
      if (key === 'ArrowDown') this.elevation -= 0.075;
      this.elevation = clamp(this.elevation, 0.06, Math.PI / 2 - 0.015);
      this.view = 'orbit';
      this._cameraChanged();
    });
    this._listen(this.canvas, 'webglcontextlost', event => {
      event.preventDefault();
      this._cancelDraw();
      this.ready = false;
      this._fail(new Error('The 3D graphics context was lost. Showing the recorded occupancy image.'));
    });
    this._listen(this.canvas, 'webglcontextrestored', () => {
      if (this.destroyed) return;
      try {
        this.resources = [];
        this._initializeGL();
        if (this.surface) this._uploadScene();
        this._invalidate();
      } catch (error) {
        this._fail(error);
      }
    });
  }

  _initializeGL() {
    const gl = this.gl;
    const track = (kind, resource) => {
      this.resources.push([kind, resource]);
      return resource;
    };
    this.voxelProgram = track('Program', makeProgram(gl, VOXEL_VERTEX, VOXEL_FRAGMENT));
    this.lineProgram = track('Program', makeProgram(gl, LINE_VERTEX, LINE_FRAGMENT));
    this.voxelUniforms = Object.fromEntries(['matrix', 'voxelSize', 'zBounds', 'bev'].map(name =>
      [name, gl.getUniformLocation(this.voxelProgram, `u_${name}`)]));
    this.lineMatrix = gl.getUniformLocation(this.lineProgram, 'u_matrix');
    this.voxelVAO = track('VertexArray', gl.createVertexArray());
    gl.bindVertexArray(this.voxelVAO);
    const cubeBuffer = track('Buffer', gl.createBuffer());
    gl.bindBuffer(gl.ARRAY_BUFFER, cubeBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, cubeVertices(), gl.STATIC_DRAW);
    for (let location = 0; location < 2; location++) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 24, location * 12);
    }
    this.instanceBuffer = track('Buffer', gl.createBuffer());
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 12, 0);
    gl.vertexAttribDivisor(2, 1);
    this.lineVAO = track('VertexArray', gl.createVertexArray());
    gl.bindVertexArray(this.lineVAO);
    this.lineBuffer = track('Buffer', gl.createBuffer());
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    for (let location = 0; location < 2; location++) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 24, location * 12);
    }
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.frontFace(gl.CCW);
    gl.clearColor(1, 1, 1, 1);
    this.ready = true;
  }

  /** Load one frame; superseded requests never replace the current scene. */
  async setFrame(caseData, frame) {
    if (this.destroyed) return false;
    const requestId = ++this.requestId;
    this.controller?.abort();
    this.controller = new AbortController();
    if (!this.ready) {
      this._fail(new Error('The interactive occupancy renderer is unavailable.'));
      return false;
    }
    try {
      validateGrid(caseData.grid);
      const key = caseData.id ?? caseData;
      if (key !== this.caseKey) {
        this.caseKey = key;
        this.grid = caseData.grid;
        this.surface = null;
        this.voxelCount = 0;
        this.reset();
      }
      const documentBase = this.canvas.ownerDocument.baseURI;
      const base = caseData.baseUrl ? new URL(caseData.baseUrl, documentBase) : documentBase;
      const response = await fetch(new URL(frame.voxels, base), { signal: this.controller.signal });
      if (!response.ok) throw new Error(`Occupancy data could not be loaded (${response.status}).`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (this.destroyed || requestId !== this.requestId) return false;
      const surface = extractSurfaceVoxels(bytes, caseData.grid);
      this.grid = caseData.grid;
      this.frame = frame;
      this.surface = surface;
      this.voxelCount = surface.count;
      this._uploadScene();
      this._invalidate();
      return true;
    } catch (error) {
      if (this.destroyed || requestId !== this.requestId || error.name === 'AbortError') return false;
      this._fail(error);
      return false;
    }
  }

  _uploadScene() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.surface.positions, gl.DYNAMIC_DRAW);
    const lines = this._sceneLines();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
    this.lineCount = lines.length / 6;
  }

  _sceneLines() {
    if (!this.showMotion) return [];
    const lines = [];
    const { min, max } = this.grid;
    const z = min[2] - this.surface.voxelSize[2] * 0.12;
    const addLine = (a, b, color) => lines.push(...a, ...color, ...b, ...color);
    const motionZ = Math.max(min[2] + this.surface.voxelSize[2] * 1.5, 0.08);
    for (const [motion, color] of [
      [this.frame?.egoMotion, [0.10, 0.39, 0.90]],
      [this.frame?.targetMotion, [0.95, 0.40, 0.13]],
    ]) {
      if (!motion) continue;
      const points = Array.isArray(motion[0]) ? motion :
        [[0, 0], Array.isArray(motion) ? motion : [motion.x ?? motion.dx, motion.y ?? motion.dy]];
      if (points.length < 2 || !points.every(point => Number.isFinite(point[0]) && Number.isFinite(point[1]))) continue;
      for (let i = 1; i < points.length; i++) {
        addLine([points[i - 1][0], points[i - 1][1], motionZ], [points[i][0], points[i][1], motionZ], color);
      }
      const [endX, endY] = points.at(-1);
      const previous = points.at(-2);
      const dx = endX - previous[0];
      const dy = endY - previous[1];
      const length = Math.hypot(dx, dy);
      if (length < 0.001) continue;
      const head = Math.min(0.3, length * 0.25);
      const ux = dx / length;
      const uy = dy / length;
      addLine([endX, endY, motionZ], [endX - ux * head - uy * head * 0.5, endY - uy * head + ux * head * 0.5, motionZ], color);
      addLine([endX, endY, motionZ], [endX - ux * head + uy * head * 0.5, endY - uy * head - ux * head * 0.5, motionZ], color);
    }
    return lines;
  }

  setView(view) {
    if (!['orbit', 'top', 'side'].includes(view) || this.destroyed) return;
    this.view = view;
    if (view === 'orbit') {
      this.azimuth = DEFAULT_AZIMUTH;
      this.elevation = DEFAULT_ELEVATION;
    } else if (view === 'top') {
      this.azimuth = -Math.PI / 2;
      this.elevation = Math.PI / 2;
    } else {
      this.azimuth = -Math.PI / 2;
      this.elevation = 0.08;
    }
    this._cameraChanged();
  }

  reset() {
    this.view = 'orbit';
    this.azimuth = DEFAULT_AZIMUTH;
    this.elevation = DEFAULT_ELEVATION;
    this.span = this.grid ? Math.hypot(...this.grid.max.map((value, axis) => value - this.grid.min[axis])) : 14;
    this.distance = this.span * 1.4;
    this._cameraChanged();
  }

  _zoom(factor) {
    this.distance = clamp(this.distance * factor, this.span * 0.24, this.span * 3.5);
    this._cameraChanged();
  }

  _cameraChanged() {
    this._invalidate();
    this.onViewChange(this.getState());
  }

  setActive(active) {
    this.active = Boolean(active);
    if (this.active) this._invalidate();
    else {
      this.drag = null;
      this.canvas.style.cursor = 'grab';
      this._cancelDraw();
    }
  }

  getState() {
    return {
      view: this.view,
      azimuth: this.azimuth,
      elevation: this.elevation,
      distance: this.distance,
      voxelCount: this.voxelCount,
      renderer: this.ready && !this.destroyed ? 'webgl2' : 'unavailable',
    };
  }

  _invalidate() {
    this.dirty = true;
    if (this.destroyed || !this.ready || !this.active || !this.inView || this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (this.destroyed || !this.ready || !this.active || !this.inView || !this.dirty) return;
      this._draw();
    });
  }

  _cancelDraw() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  _draw() {
    this.dirty = false;
    const gl = this.gl;
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    if (this.view === 'top') gl.clearColor(1, 248 / 255, 205 / 255, 1);
    else gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.grid || !this.surface) return;
    const center = this.grid.min.map((value, axis) => (value + this.grid.max[axis]) / 2);
    const horizontal = Math.cos(this.elevation) * this.distance;
    const eye = [
      center[0] + Math.cos(this.azimuth) * horizontal,
      center[1] + Math.sin(this.azimuth) * horizontal,
      center[2] + Math.sin(this.elevation) * this.distance,
    ];
    const aspect = width / height;
    const near = Math.max(0.01, this.span * 0.002);
    const far = this.span * 8;
    let projection;
    if (this.view === 'orbit') {
      const fov = 2 * Math.atan(Math.tan(Math.PI / 8) / Math.min(1, aspect));
      projection = perspectiveMatrix(fov, aspect, near, far);
    } else {
      const halfHeight = this.distance * 0.33 / Math.min(1, aspect);
      projection = orthographicMatrix(-halfHeight * aspect, halfHeight * aspect, -halfHeight, halfHeight, near, far);
    }
    const matrix = multiplyMatrices(projection, lookAtMatrix(eye, center));
    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(this.lineMatrix, false, matrix);
    gl.bindVertexArray(this.lineVAO);
    gl.drawArrays(gl.LINES, 0, this.lineCount);
    gl.useProgram(this.voxelProgram);
    gl.uniformMatrix4fv(this.voxelUniforms.matrix, false, matrix);
    gl.uniform3fv(this.voxelUniforms.voxelSize, this.surface.voxelSize);
    gl.uniform2f(this.voxelUniforms.zBounds, this.grid.min[2], this.grid.max[2]);
    gl.uniform1i(this.voxelUniforms.bev, this.view === 'top' ? 1 : 0);
    gl.bindVertexArray(this.voxelVAO);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.voxelCount);
    gl.bindVertexArray(null);
  }

  _fail(error) {
    this.onError(error instanceof Error ? error : new Error(String(error)));
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestId++;
    this.controller?.abort();
    this._cancelDraw();
    this.resizeObserver?.disconnect();
    this.intersectionObserver?.disconnect();
    this.listeners.forEach(remove => remove());
    if (this.gl && !this.gl.isContextLost()) {
      this.resources.forEach(([kind, resource]) => this.gl[`delete${kind}`](resource));
    }
    this.resources = [];
    this.surface = null;
    this.ready = false;
  }
}
