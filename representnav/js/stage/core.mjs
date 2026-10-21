// Shared three.js toolkit for the two method stages.
// Colours are taken from the paper's pipeline figure so the 3D scenes read as the same diagram.
import * as THREE from '../../vendor/three/three.module.min.js';
import { RoundedBoxGeometry } from '../../vendor/three/RoundedBoxGeometry.js';
import { RoomEnvironment } from '../../vendor/three/RoomEnvironment.js';

export { THREE, RoundedBoxGeometry };

export const C = {
  navy: '#2f5597', navyDeep: '#1f3d73', navyNight: '#0d1a33', ink: '#333333', slate: '#56698f',
  blue50: '#eef3fb', blue100: '#e3ecf9', blue150: '#dbe6f7', blue200: '#cfe0f5', blue300: '#b9c8e4',
  blue400: '#7fa6d8', blue500: '#3f68b0',
  red: '#a44a3f', red50: '#f6e5e2', redGlow: '#e0796c',
  teal: '#1f6b60', teal2: '#2f8a7c', teal50: '#dff0ec',
  amber: '#8a5a1a', amber2: '#c08a3e', amber50: '#f8ecda',
  semantic: '#3f68b0', geometric: '#2f8a7c', dynamic: '#c08a3e',
  boundary: '#333333', white: '#ffffff', box: '#ed721f',
};

export const FONT = '"Source Sans 3", "Source Sans Pro", "Helvetica Neue", Arial, sans-serif';
export const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
export const SERIF = '"Source Serif 4", Georgia, "Times New Roman", serif';

export const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = t => t * t * (3 - 2 * t);
export const easeOut = t => 1 - Math.pow(1 - t, 3);
export const easeInOut = t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const backOut = t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
/** Normalised progress of t inside [a, b]. */
export const seg = (t, a, b) => clamp((t - a) / Math.max(1e-6, b - a));

export async function fontsReady() {
  if (!document.fonts?.load) return;
  try {
    await Promise.race([
      Promise.all([
        document.fonts.load(`700 64px ${FONT}`), document.fonts.load(`600 64px ${FONT}`),
        document.fonts.load(`400 64px ${FONT}`), document.fonts.load(`italic 600 64px ${SERIF}`),
        document.fonts.load(`500 64px ${MONO}`),
      ]),
      new Promise(resolve => setTimeout(resolve, 2500)),
    ]);
  } catch { /* fall back to system fonts */ }
}

// ---------------------------------------------------------------- canvas drawing

export function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function drawDrop(ctx, x, y, s, color = C.red) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s / 20, s / 20);
  ctx.beginPath();
  ctx.moveTo(0, -10);
  ctx.bezierCurveTo(-2, -4, -8, 0, -8, 4.5);
  ctx.bezierCurveTo(-8, 9.5, -4, 12, 0, 12);
  ctx.bezierCurveTo(4, 12, 8, 9.5, 8, 4.5);
  ctx.bezierCurveTo(8, 0, 2, -4, 0, -10);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

export function drawSnowflake(ctx, x, y, s, color = '#7f93b3') {
  ctx.save();
  ctx.translate(x, y);
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.5, s * .09);
  ctx.lineCap = 'round';
  for (let i = 0; i < 6; i++) {
    ctx.save();
    ctx.rotate(i * Math.PI / 3);
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(0, -s);
    ctx.moveTo(0, -s * .55); ctx.lineTo(-s * .25, -s * .78);
    ctx.moveTo(0, -s * .55); ctx.lineTo(s * .25, -s * .78);
    ctx.stroke();
    ctx.restore();
  }
  ctx.restore();
}

export function makeCanvas(w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(2, Math.round(w));
  canvas.height = Math.max(2, Math.round(h));
  return canvas;
}

export function toTexture(canvas, { nearest = false } = {}) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  if (nearest) {
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
  } else {
    texture.minFilter = THREE.LinearMipmapLinearFilter;
  }
  return texture;
}

export function canvasTexture(w, h, draw, options) {
  const canvas = makeCanvas(w, h);
  const ctx = canvas.getContext('2d');
  draw(ctx, canvas.width, canvas.height);
  return toTexture(canvas, options);
}

/** A flat, camera-agnostic plane that carries a canvas drawing. */
export function texturedPlane(texture, width, height, { opacity = 1, depthWrite = false, side = THREE.FrontSide, toneMapped = false, fog = true } = {}) {
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity, depthWrite, side, toneMapped, fog });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
  return mesh;
}

/**
 * Text as a plane. `height` is the world height of one text line.
 * Returns a mesh whose geometry is sized to the measured text.
 */
export function textPlane(text, { height = .4, weight = 600, family = FONT, italic = false, color = C.ink,
  align = 'center', px = 96, letterSpacing = 0, background = null, padX = 0, padY = 0, radius = 0,
  stroke = null, opacity = 1, fog = true } = {}) {
  const font = `${italic ? 'italic ' : ''}${weight} ${px}px ${family}`;
  const probe = makeCanvas(4, 4).getContext('2d');
  probe.font = font;
  if ('letterSpacing' in probe) probe.letterSpacing = `${letterSpacing}px`;
  const lines = String(text).split('\n');
  const textWidth = Math.max(...lines.map(line => probe.measureText(line).width));
  const lineHeight = px * 1.22;
  const w = Math.ceil(textWidth + px * .3 + padX * 2);
  const h = Math.ceil(lineHeight * lines.length + px * .18 + padY * 2);
  const canvas = makeCanvas(w, h);
  const ctx = canvas.getContext('2d');
  if (background) {
    roundRect(ctx, 1, 1, w - 2, h - 2, radius);
    ctx.fillStyle = background;
    ctx.fill();
    if (stroke) { ctx.lineWidth = 3; ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  ctx.font = font;
  if ('letterSpacing' in ctx) ctx.letterSpacing = `${letterSpacing}px`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.textAlign = align;
  const x = align === 'center' ? w / 2 : align === 'left' ? padX + px * .15 : w - padX - px * .15;
  lines.forEach((line, i) => ctx.fillText(line, x, padY + px * .09 + lineHeight * (i + .5)));
  const worldH = height * (h / (lineHeight + px * .18));
  const mesh = texturedPlane(toTexture(canvas), worldH * w / h, worldH, { opacity, fog });
  mesh.userData.aspect = w / h;
  return mesh;
}

// ---------------------------------------------------------------- materials

export function solidMaterial(color, { roughness = .48, clearcoat = .5, opacity = 1, emissive = '#000000', emissiveIntensity = 0, metalness = 0 } = {}) {
  return new THREE.MeshPhysicalMaterial({
    color, roughness, metalness, clearcoat, clearcoatRoughness: .32,
    transparent: opacity < 1, opacity, emissive, emissiveIntensity,
  });
}

export function glassMaterial(color, { opacity = .35, roughness = .15 } = {}) {
  return new THREE.MeshPhysicalMaterial({
    color, roughness, metalness: 0, clearcoat: 1, clearcoatRoughness: .08,
    transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide,
  });
}

const roundedCache = new Map();
export function roundedBox(w, h, d, r = .1, segments = 4) {
  const key = [w, h, d, r, segments].map(v => v.toFixed(3)).join('/');
  if (!roundedCache.has(key)) roundedCache.set(key, new RoundedBoxGeometry(w, h, d, segments, Math.min(r, w / 2, h / 2, d / 2)));
  return roundedCache.get(key);
}

/** An inverted-hull outline gives the 3D blocks the figure's stroked look. */
export function addOutline(mesh, color, thickness = .035) {
  const params = mesh.geometry.parameters || {};
  const box = new THREE.Box3().setFromBufferAttribute(mesh.geometry.attributes.position);
  const size = box.getSize(new THREE.Vector3());
  const material = new THREE.MeshBasicMaterial({ color, side: THREE.BackSide, transparent: true, toneMapped: false });
  const shell = new THREE.Mesh(mesh.geometry, material);
  shell.scale.set(1 + 2 * thickness / Math.max(size.x, 1e-3), 1 + 2 * thickness / Math.max(size.y, 1e-3), 1 + 2 * thickness / Math.max(size.z, 1e-3));
  shell.renderOrder = mesh.renderOrder - 1;
  shell.userData.isOutline = true;
  mesh.add(shell);
  mesh.userData.outline = shell;
  return shell;
}

let glowTexture;
export function glowSprite(color, size = 1, { opacity = 1, additive = true } = {}) {
  glowTexture ??= canvasTexture(128, 128, (ctx, w) => {
    const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(.25, 'rgba(255,255,255,.55)');
    g.addColorStop(.6, 'rgba(255,255,255,.14)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, w);
  });
  const material = new THREE.SpriteMaterial({ map: glowTexture, color, transparent: true, opacity,
    depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending, toneMapped: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.setScalar(size);
  sprite.userData.baseOpacity = opacity;
  return sprite;
}

// ---------------------------------------------------------------- wires

const WIRE_VERTEX = /* glsl */`
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vView;
#include <fog_pars_vertex>
void main() {
  vUv = uv;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-mvPosition.xyz);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;

const WIRE_FRAGMENT = /* glsl */`
uniform vec3 uColor;
uniform float uOpacity;
uniform float uProgress;
uniform float uStart;
uniform float uDash;
uniform float uLength;
uniform float uTime;
uniform float uSpeed;
uniform float uFlow;
uniform float uFlowGap;
uniform float uHead;
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vView;
#include <fog_pars_fragment>
void main() {
  float s = vUv.x;
  if (s > uProgress || s < uStart) discard;
  float along = s * uLength;
  if (uDash > 0.0) {
    float d = fract((along - uTime * uSpeed) / uDash);
    if (d > 0.56) discard;
  }
  float facing = clamp(dot(normalize(vNormal), normalize(vView)), 0.0, 1.0);
  vec3 col = uColor * (0.78 + 0.32 * facing);
  float pulse = 0.0;
  if (uFlow > 0.0) {
    float phase = fract((along - uTime * 2.4) / uFlowGap);
    pulse = smoothstep(0.0, 0.08, phase) * (1.0 - smoothstep(0.08, 0.26, phase)) * uFlow;
  }
  float head = (1.0 - smoothstep(0.0, 0.12 / max(uLength, 0.5), uProgress - s)) * uHead * step(uProgress, 0.999);
  vec3 bright = mix(uColor, vec3(1.0), 0.55);
  col = mix(col, bright * 1.15, clamp(pulse, 0.0, 1.0));
  col = mix(col, vec3(1.0), clamp(head * 0.6, 0.0, 1.0));
  gl_FragColor = vec4(col, uOpacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`;

export function wireMaterial({ color = C.slate, opacity = 1, dash = 0, speed = 0, flow = 0, flowGap = 1.6, head = 1 } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uColor: { value: new THREE.Color(color) }, uOpacity: { value: opacity }, uProgress: { value: 1 }, uStart: { value: 0 },
      uDash: { value: dash }, uLength: { value: 1 }, uTime: { value: 0 }, uSpeed: { value: speed },
      uFlow: { value: flow }, uFlowGap: { value: flowGap }, uHead: { value: head },
    }]),
    vertexShader: WIRE_VERTEX, fragmentShader: WIRE_FRAGMENT,
    transparent: true, fog: true, depthWrite: true,
  });
}

/** A polyline with rounded corners as a CurvePath. */
export function roundedPath(points, corner = .22) {
  const pts = points.map(p => p.isVector3 ? p.clone() : new THREE.Vector3(...p));
  const path = new THREE.CurvePath();
  let cursor = pts[0].clone();
  for (let i = 1; i < pts.length; i++) {
    const current = pts[i];
    if (i === pts.length - 1) { path.add(new THREE.LineCurve3(cursor, current.clone())); break; }
    const next = pts[i + 1];
    const inDir = current.clone().sub(pts[i - 1]);
    const outDir = next.clone().sub(current);
    const r = Math.min(corner, inDir.length() / 2, outDir.length() / 2);
    const a = current.clone().sub(inDir.normalize().multiplyScalar(r));
    const b = current.clone().add(outDir.normalize().multiplyScalar(r));
    if (cursor.distanceTo(a) > 1e-4) path.add(new THREE.LineCurve3(cursor, a));
    if (r > 1e-4) path.add(new THREE.QuadraticBezierCurve3(a, current.clone(), b));
    cursor = b;
  }
  return path;
}

/** Arc-length sampled curve wrapper so TubeGeometry uv.x is proportional to length. */
export class Wire extends THREE.Group {
  constructor(curveOrPoints, { color = C.slate, radius = .028, corner = .22, arrow = 'end', dash = 0, speed = 0,
    flow = 0, flowGap = 1.6, opacity = 1, arrowColor, head = 1, radialSegments = 8, arrowScale = 1 } = {}) {
    super();
    const curve = Array.isArray(curveOrPoints) ? roundedPath(curveOrPoints, corner) : curveOrPoints;
    const length = curve.getLength();
    const coneLength = arrow ? radius * 7.5 * arrowScale : 0;
    const tubeLength = Math.max(.001, length - (arrow === 'end' || arrow === 'both' ? coneLength * .8 : 0));
    this.curve = curve;
    this.length = length;
    this.tubeFraction = tubeLength / length;
    const segments = Math.max(12, Math.ceil(length * 22));
    const startTrim = arrow === 'both' ? coneLength * .8 / length : 0;
    this.startTrim = startTrim;
    const trimmed = new TrimmedCurve(curve, startTrim, this.tubeFraction);
    this.material = wireMaterial({ color, opacity, dash, speed, flow, flowGap, head });
    this.material.uniforms.uLength.value = tubeLength;
    this.tube = new THREE.Mesh(new THREE.TubeGeometry(trimmed, segments, radius, radialSegments, false), this.material);
    this.add(this.tube);
    this.arrows = [];
    const coneGeometry = new THREE.ConeGeometry(radius * 3.1 * arrowScale, coneLength, 16);
    coneGeometry.translate(0, -coneLength / 2, 0); // tip at the origin
    const coneMaterial = new THREE.MeshBasicMaterial({ color: arrowColor || color, transparent: true, opacity, toneMapped: false });
    this.coneMaterial = coneMaterial;
    if (arrow === 'end' || arrow === 'both') this.arrows.push({ mesh: new THREE.Mesh(coneGeometry, coneMaterial), at: 1, dir: 1 });
    if (arrow === 'both') this.arrows.push({ mesh: new THREE.Mesh(coneGeometry, coneMaterial), at: 0, dir: -1 });
    for (const a of this.arrows) {
      this.add(a.mesh);
      const p = curve.getPointAt(a.at);
      const t = curve.getTangentAt(a.at).multiplyScalar(a.dir);
      a.mesh.position.copy(p);
      a.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), t.normalize());
    }
    this.headCone = arrow === 'end' ? this.arrows[0] : null;
    this._progress = 1;
    this.setProgress(1);
  }
  setProgress(p) {
    p = clamp(p);
    this._progress = p;
    const span = Math.max(1e-3, this.tubeFraction - this.startTrim);
    this.material.uniforms.uProgress.value = p <= 0 ? -1 : (p - this.startTrim) / span + .002;
    this.visible = p > 0.0005;
    for (const a of this.arrows) {
      if (a === this.headCone) {
        // The arrowhead leads the line while it draws.
        const u = clamp(p);
        const point = this.curve.getPointAt(Math.max(.001, u));
        const tangent = this.curve.getTangentAt(Math.max(.001, u)).normalize();
        a.mesh.position.copy(point);
        a.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
        a.mesh.visible = p > .02;
        a.mesh.scale.setScalar(clamp(p * 6));
      } else a.mesh.visible = p > .98;
    }
  }
  setOpacity(o) {
    this.material.uniforms.uOpacity.value = o;
    this.coneMaterial.opacity = o;
  }
  tick(time) { this.material.uniforms.uTime.value = time; }
  pointAt(u) { return this.curve.getPointAt(clamp(u)); }
}

class TrimmedCurve extends THREE.Curve {
  constructor(curve, a, b) { super(); this.base = curve; this.a = a; this.b = b; }
  getPoint(t, target = new THREE.Vector3()) { return target.copy(this.base.getPointAt(this.a + (this.b - this.a) * t)); }
}

// ---------------------------------------------------------------- tokens and blocks

let dashedFaceTexture, starFaceTexture;
function dashedTexture(withStar = false) {
  const draw = (ctx, w) => {
    ctx.clearRect(0, 0, w, w);
    roundRect(ctx, 10, 10, w - 20, w - 20, 26);
    ctx.setLineDash([22, 14]);
    ctx.lineWidth = 9;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    if (withStar) {
      ctx.setLineDash([]);
      ctx.fillStyle = '#ffffff';
      ctx.font = `700 120px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('*', w / 2, w / 2 + 22);
    }
  };
  if (withStar) return (starFaceTexture ??= canvasTexture(256, 256, draw));
  return (dashedFaceTexture ??= canvasTexture(256, 256, draw));
}

export const TOKEN_STYLE = {
  rep: { fill: C.blue150, stroke: C.navy },
  cls: { fill: C.navy, stroke: C.navyDeep },
  text: { fill: '#ffffff', stroke: '#5b5b5b' },
  boundary: { fill: '#3a3a3a', stroke: '#1f1f1f' },
  latent: { fill: C.blue100, stroke: C.slate, dashed: true },
  learnable: { fill: '#f4f7fc', stroke: C.navy, dashed: true, star: true },
};

/** A rounded token cube. kind ∈ rep | cls | text | boundary | latent | learnable */
export function makeToken(kind = 'rep', size = .78, { dark = false } = {}) {
  const style = TOKEN_STYLE[kind];
  const group = new THREE.Group();
  group.userData.kind = kind;
  const geometry = roundedBox(size, size, size, size * .16, 4);
  if (style.dashed) {
    const body = new THREE.Mesh(geometry, glassMaterial(style.fill, { opacity: dark ? .2 : .55 }));
    body.renderOrder = 2;
    const shellMaterial = map => new THREE.MeshBasicMaterial({ map, color: dark ? '#cfe0ff' : style.stroke, transparent: true,
      depthWrite: false, toneMapped: false, side: THREE.DoubleSide });
    const plain = shellMaterial(dashedTexture(false));
    const front = style.star ? shellMaterial(dashedTexture(true)) : plain;
    const shell = new THREE.Mesh(roundedBox(size * 1.01, size * 1.01, size * 1.01, size * .16, 4),
      [plain, plain, plain, plain, front, plain]);
    shell.renderOrder = 3;
    group.add(body, shell);
    group.userData.body = body;
    group.userData.shell = shell;
  } else {
    const body = new THREE.Mesh(geometry, solidMaterial(style.fill, { roughness: kind === 'cls' ? .32 : .45, clearcoat: .7 }));
    addOutline(body, dark && kind === 'boundary' ? '#8ea6cf' : style.stroke, size * .045);
    group.add(body);
    group.userData.body = body;
  }
  return group;
}

/** Rounded slab with a front face drawn on a canvas. */
export function makeSlab({ w, h, d, r = .14, fill = C.blue150, stroke = C.navy, strokeWidth = .04, texture, texW = 1024,
  draw, material, outline = true, faceInset = 0 }) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(roundedBox(w, h, d, r, 5), material || solidMaterial(fill));
  if (outline && stroke) addOutline(body, stroke, strokeWidth);
  group.add(body);
  group.userData.body = body;
  if (draw || texture) {
    const texH = Math.round(texW * h / w);
    const tex = texture || canvasTexture(texW, texH, (ctx, cw, ch) => draw(ctx, cw, ch));
    const face = texturedPlane(tex, w - faceInset, h - faceInset);
    face.position.z = d / 2 + .004;
    face.renderOrder = 4;
    group.add(face);
    group.userData.face = face;
  }
  return group;
}

/** Fade every material under a node; materials are made transparent once and cached. */
export function setGroupOpacity(object, opacity) {
  object.traverse(node => {
    const materials = node.material ? (Array.isArray(node.material) ? node.material : [node.material]) : [];
    for (const m of materials) {
      if (m.userData.baseOpacity === undefined) m.userData.baseOpacity = m.uniforms?.uOpacity ? m.uniforms.uOpacity.value : m.opacity;
      const value = m.userData.baseOpacity * opacity;
      if (m.uniforms?.uOpacity) m.uniforms.uOpacity.value = value;
      else {
        if (!m.transparent && value < .999) { m.transparent = true; m.needsUpdate = true; }
        m.opacity = value;
      }
    }
  });
}

// ---------------------------------------------------------------- colour maps

const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const MAGMA = [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]];
function ramp(stops, t) {
  t = clamp(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  return stops[i].map((v, k) => (v + (stops[i + 1][k] - v) * f) / 255);
}
export const viridis = t => ramp(VIRIDIS, t);
export const magma = t => ramp(MAGMA, t);

// ---------------------------------------------------------------- image helpers

const imageCache = new Map();
export function loadImage(url) {
  if (!url) return Promise.reject(new Error('Missing image URL'));
  if (!imageCache.has(url)) {
    imageCache.set(url, new Promise((resolve, reject) => {
      const image = new Image();
      image.decoding = 'async';
      image.onload = () => resolve(image);
      image.onerror = () => { imageCache.delete(url); reject(new Error(`Image failed: ${url}`)); };
      image.src = url;
    }));
  }
  return imageCache.get(url);
}

export async function imageTexture(url, { nearest = false } = {}) {
  const image = await loadImage(url);
  const texture = new THREE.Texture(image);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  if (nearest) { texture.magFilter = THREE.NearestFilter; texture.minFilter = THREE.NearestFilter; texture.generateMipmaps = false; }
  texture.needsUpdate = true;
  return texture;
}

export function imagePixels(image) {
  const canvas = makeCanvas(image.naturalWidth || image.width, image.naturalHeight || image.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/** A photo card: the image with a thin rounded frame and soft shadow edge, as one texture. */
export function photoCardTexture(image, { w = 512, h, radius = 26, border = '#ffffff', stroke = '#b9c8e4', borderPx = 10, fit = 'cover' } = {}) {
  const ratio = (image.naturalHeight || image.height) / (image.naturalWidth || image.width);
  h ??= Math.round(w * ratio);
  return canvasTexture(w, h, ctx => {
    roundRect(ctx, 2, 2, w - 4, h - 4, radius);
    ctx.fillStyle = border;
    ctx.fill();
    ctx.save();
    roundRect(ctx, borderPx, borderPx, w - borderPx * 2, h - borderPx * 2, radius * .6);
    ctx.clip();
    const iw = image.naturalWidth || image.width, ih = image.naturalHeight || image.height;
    const scale = fit === 'cover' ? Math.max((w - borderPx * 2) / iw, (h - borderPx * 2) / ih) : Math.min((w - borderPx * 2) / iw, (h - borderPx * 2) / ih);
    const dw = iw * scale, dh = ih * scale;
    ctx.drawImage(image, (w - dw) / 2, (h - dh) / 2, dw, dh);
    ctx.restore();
    roundRect(ctx, 2, 2, w - 4, h - 4, radius);
    ctx.lineWidth = 3;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  });
}

// ---------------------------------------------------------------- voxels

const VOXEL_VERTEX = /* glsl */`
attribute vec3 aPos;
attribute vec3 aFrom;
attribute float aDelay;
attribute float aH;
uniform float uAssemble;
uniform float uScale;
uniform float uTime;
varying float vH;
varying vec3 vN;
varying float vFresh;
#include <fog_pars_vertex>
void main() {
  float k = clamp((uAssemble - aDelay) / 0.42, 0.0, 1.0);
  float e = 1.0 - pow(1.0 - k, 3.0);
  vec3 center = mix(aFrom, aPos, e);
  float s = uScale * mix(0.25, 1.0, e);
  vec3 p = center + position * s;
  vH = aH;
  vN = normalize(normalMatrix * normal);
  vFresh = 1.0 - e;
  vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;

const VOXEL_FRAGMENT = /* glsl */`
uniform float uOpacity;
uniform vec3 uTint;
uniform float uTintMix;
varying float vH;
varying vec3 vN;
varying float vFresh;
#include <fog_pars_fragment>
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
  vec3 L = normalize(vec3(0.35, 0.85, 0.55));
  float light = 0.62 + 0.42 * max(dot(normalize(vN), L), 0.0);
  vec3 col = pow(viridis(vH), vec3(2.2)) * light;
  col = mix(col, uTint, uTintMix);
  col = mix(col, vec3(1.0), vFresh * 0.65);
  gl_FragColor = vec4(col, uOpacity);
  #include <colorspace_fragment>
  #include <fog_fragment>
}`;

const voxelCache = new Map();
export function loadVoxels(url) {
  if (!voxelCache.has(url)) {
    voxelCache.set(url, fetch(url).then(r => {
      if (!r.ok) throw new Error(`Voxels ${r.status}`);
      return r.arrayBuffer();
    }).then(b => new Uint8Array(b)).catch(error => { voxelCache.delete(url); throw error; }));
  }
  return voxelCache.get(url);
}

/**
 * Surface voxels as one instanced mesh. Robot frame (x forward, y left, z up) maps to
 * three.js as (-y, z, -x): the robot looks into the screen.
 */
export function makeVoxelMesh(voxels, grid, { fit = 4, origin = new THREE.Vector3(), spread = 1.2, fromPoint = null } = {}) {
  const [nx, ny, nz] = grid.dims;
  const slice = nx * ny;
  const size = grid.dims.map((n, i) => (grid.max[i] - grid.min[i]) / n);
  const extent = Math.max(grid.max[0] - grid.min[0], grid.max[1] - grid.min[1]);
  const scale = fit / extent;
  const cx = (grid.min[0] + grid.max[0]) / 2, cy = (grid.min[1] + grid.max[1]) / 2;
  const pos = [], from = [], delay = [], hs = [];
  const rand = mulberry32(7);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = x + nx * y + slice * z;
    if (!voxels[i]) continue;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1 && voxels[i - 1] && voxels[i + 1] &&
        voxels[i - nx] && voxels[i + nx] && voxels[i - slice] && voxels[i + slice]) continue;
    const wx = grid.min[0] + (x + .5) * size[0] - cx;
    const wy = grid.min[1] + (y + .5) * size[1] - cy;
    const wz = grid.min[2] + (z + .5) * size[2];
    const px = -wy * scale, py = wz * scale, pz = -wx * scale;
    pos.push(px, py, pz);
    if (fromPoint) {
      const a = rand() * Math.PI * 2, r = rand() * .35;
      from.push(fromPoint.x + Math.cos(a) * r, fromPoint.y + (rand() - .5) * .35, fromPoint.z + Math.sin(a) * r);
    } else {
      const a = rand() * Math.PI * 2, r = (.4 + rand()) * spread;
      from.push(px + Math.cos(a) * r, py + 1.2 + rand() * spread, pz + Math.sin(a) * r);
    }
    // Build from the ground up, with a little jitter.
    delay.push(clamp(z / nz * .45 + rand() * .13));
    hs.push(clamp((wz - grid.min[2]) / (grid.max[2] - grid.min[2])));
  }
  const base = new THREE.BoxGeometry(1, 1, 1);
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = base.index;
  geometry.setAttribute('position', base.attributes.position);
  geometry.setAttribute('normal', base.attributes.normal);
  geometry.setAttribute('aPos', new THREE.InstancedBufferAttribute(new Float32Array(pos), 3));
  geometry.setAttribute('aFrom', new THREE.InstancedBufferAttribute(new Float32Array(from), 3));
  geometry.setAttribute('aDelay', new THREE.InstancedBufferAttribute(new Float32Array(delay), 1));
  geometry.setAttribute('aH', new THREE.InstancedBufferAttribute(new Float32Array(hs), 1));
  geometry.instanceCount = hs.length;
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 1, 0), fit * 2);
  const material = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uAssemble: { value: 1 }, uScale: { value: size[0] * scale * .96 }, uTime: { value: 0 }, uOpacity: { value: 1 },
      uTint: { value: new THREE.Color('#ffffff') }, uTintMix: { value: 0 },
    }]),
    vertexShader: VOXEL_VERTEX, fragmentShader: VOXEL_FRAGMENT, fog: true, transparent: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(origin);
  mesh.frustumCulled = false;
  mesh.userData.scale = scale;
  mesh.userData.count = hs.length;
  return mesh;
}

export function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- readout objects

/** Depth readout as a relief: height and colour from the readout's own magma ramp. */
export function makeDepthRelief(image, { width = 3, height = 3, amp = .9 } = {}) {
  const data = imagePixels(image);
  const W = data.width, H = data.height;
  const sample = (u, v) => {
    const x = clamp(u * (W - 1), 0, W - 1), y = clamp(v * (H - 1), 0, H - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
    const fx = x - x0, fy = y - y0;
    const lum = (xx, yy) => {
      const k = (yy * W + xx) * 4;
      return [data.data[k], data.data[k + 1], data.data[k + 2]];
    };
    const mix3 = (a, b, f) => a.map((v, i) => v + (b[i] - v) * f);
    return mix3(mix3(lum(x0, y0), lum(x1, y0), fx), mix3(lum(x0, y1), lum(x1, y1), fx), fy);
  };
  const seg = 72;
  const geometry = new THREE.PlaneGeometry(width, height, seg, seg);
  const position = geometry.attributes.position;
  const colors = new Float32Array(position.count * 3);
  const heights = new Float32Array(position.count);
  for (let i = 0; i < position.count; i++) {
    const u = position.getX(i) / width + .5, v = .5 - position.getY(i) / height;
    const rgb = sample(u, v);
    const value = (rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722) / 255;
    heights[i] = value;
    colors[i * 3] = Math.pow(rgb[0] / 255, 2.2); colors[i * 3 + 1] = Math.pow(rgb[1] / 255, 2.2); colors[i * 3 + 2] = Math.pow(rgb[2] / 255, 2.2);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.userData.heights = heights;
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .55, metalness: 0, side: THREE.DoubleSide, transparent: true });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.setRise = rise => {
    for (let i = 0; i < position.count; i++) position.setZ(i, heights[i] * amp * rise);
    position.needsUpdate = true;
    geometry.computeVertexNormals();
  };
  mesh.userData.setRise(1);
  return mesh;
}

/** Semantic readout split into one pixel-exact plane per class, for an exploded view. */
export function makeSemanticLayers(image, { width = 3, height = 3 } = {}) {
  const data = imagePixels(image);
  const W = data.width, H = data.height;
  const classes = new Map();
  for (let i = 0; i < W * H; i++) {
    const k = i * 4;
    const key = (data.data[k] << 16) | (data.data[k + 1] << 8) | data.data[k + 2];
    classes.set(key, (classes.get(key) || 0) + 1);
  }
  const ordered = [...classes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 7);
  const group = new THREE.Group();
  const layers = [];
  ordered.forEach(([key], index) => {
    const canvas = makeCanvas(W, H);
    const ctx = canvas.getContext('2d');
    const out = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const k = i * 4;
      const match = ((data.data[k] << 16) | (data.data[k + 1] << 8) | data.data[k + 2]) === key;
      out.data[k] = data.data[k]; out.data[k + 1] = data.data[k + 1]; out.data[k + 2] = data.data[k + 2];
      out.data[k + 3] = match ? 255 : 0;
    }
    ctx.putImageData(out, 0, 0);
    const plane = texturedPlane(toTexture(canvas, { nearest: true }), width, height, { depthWrite: false, side: THREE.DoubleSide });
    plane.renderOrder = 5 + index;
    plane.userData.order = index;
    group.add(plane);
    layers.push(plane);
  });
  group.userData.layers = layers;
  group.userData.setExplode = (amount) => {
    const n = layers.length;
    layers.forEach((plane, i) => { plane.position.z = (i - (n - 1) / 2) * .34 * amount; });
  };
  return group;
}

/** A rounded label card used to caption floating objects. */
export function captionCard(title, subtitle = '', { width = 2.4, dark = false, accent = C.navy } = {}) {
  const texW = 640, texH = subtitle ? 150 : 104;
  const tex = canvasTexture(texW, texH, ctx => {
    roundRect(ctx, 3, 3, texW - 6, texH - 6, 24);
    ctx.fillStyle = dark ? 'rgba(14,30,60,.82)' : 'rgba(255,255,255,.94)';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = dark ? 'rgba(160,190,240,.55)' : C.blue300;
    ctx.stroke();
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(40, 52, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = `700 46px ${FONT}`;
    ctx.fillStyle = dark ? '#f2f6ff' : C.navyDeep;
    ctx.textBaseline = 'middle';
    ctx.fillText(title, 64, 53);
    if (subtitle) {
      ctx.font = `500 32px ${FONT}`;
      ctx.fillStyle = dark ? 'rgba(220,232,255,.75)' : C.slate;
      ctx.fillText(subtitle, 64, 108);
    }
  });
  const mesh = texturedPlane(tex, width, width * texH / texW);
  mesh.renderOrder = 20;
  return mesh;
}

// ---------------------------------------------------------------- stage

export class Stage {
  constructor(host, { dark = false, fov = 30, exposure = 1, shadows = false } = {}) {
    this.host = host;
    this.dark = dark;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'stage-canvas';
    host.prepend(this.canvas);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = exposure;
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(fov, 16 / 9, .1, 400);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), .04).texture;
    this.scene.environmentIntensity = dark ? .55 : .85;
    pmrem.dispose();
    const hemi = new THREE.HemisphereLight(dark ? '#bcd0ff' : '#ffffff', dark ? '#0b1730' : '#d9e2f0', dark ? .9 : 1.15);
    const key = new THREE.DirectionalLight('#ffffff', dark ? 1.3 : 1.6);
    key.position.set(6, 14, 10);
    this.key = key;
    if (shadows) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      key.shadow.bias = -.0004;
      key.shadow.normalBias = .02;
      key.shadow.radius = 4;
      Object.assign(key.shadow.camera, { left: -20, right: 20, top: 20, bottom: -20, near: 1, far: 90 });
      this.scene.add(key.target);
    }
    const rim = new THREE.DirectionalLight(dark ? '#7fa6ff' : '#cfe0f5', dark ? 1.1 : .6);
    rim.position.set(-10, 4, -6);
    this.scene.add(hemi, key, rim);
    this.width = 1;
    this.height = 1;
    this.visible = false;
    this.running = false;
    this.last = 0;
    this.frameCallbacks = new Set();
    this.raf = 0;
    this.resize();
    this.resizeObserver = new ResizeObserver(() => { this.resize(); this.render(); });
    this.resizeObserver.observe(host);
    this.loop = this.loop.bind(this);
    this.intersection = new IntersectionObserver(entries => {
      for (const entry of entries) this.setVisible(entry.isIntersecting);
    }, { threshold: .02 });
    this.intersection.observe(host);
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.stop(); else if (this.visible) this.start(); });
  }
  resize() {
    const rect = this.host.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.renderer.setSize(this.width, this.height, false);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
  }
  setVisible(visible) {
    this.visible = visible;
    if (visible) this.start(); else this.stop();
    this.onVisibility?.(visible);
  }
  start() {
    if (this.raf || document.hidden) return;
    this.last = 0;
    this.raf = requestAnimationFrame(this.loop);
  }
  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }
  loop(now) {
    this.raf = 0;
    const dt = this.last ? Math.min(.05, (now - this.last) / 1000) : 0;
    this.last = now;
    for (const callback of this.frameCallbacks) callback(dt, now / 1000);
    this.render();
    if (this.visible && !document.hidden) this.raf = requestAnimationFrame(this.loop);
  }
  render() { this.renderer.render(this.scene, this.camera); }
}

/** Keyframed camera: shots are {t, pos, target, fov?}; positions are eased between shots. */
export class CameraRig {
  constructor(camera, shots) {
    this.camera = camera;
    this.shots = shots;
    this.offset = { yaw: 0, pitch: 0, zoom: 1 };
    this.aspectFit = 1;
    this._pos = new THREE.Vector3();
    this._target = new THREE.Vector3();
  }
  sample(t) {
    const shots = this.shots;
    let i = 0;
    while (i < shots.length - 1 && t > shots[i + 1].t) i++;
    const a = shots[i], b = shots[Math.min(shots.length - 1, i + 1)];
    const f = a === b ? 0 : easeInOut(clamp((t - a.t) / Math.max(1e-6, b.t - a.t)));
    const pos = new THREE.Vector3(...a.pos).lerp(new THREE.Vector3(...b.pos), f);
    const target = new THREE.Vector3(...a.target).lerp(new THREE.Vector3(...b.target), f);
    const fov = lerp(a.fov ?? 30, b.fov ?? 30, f);
    return { pos, target, fov };
  }
  apply(t, time = 0, sway = 1) {
    const { pos, target, fov } = this.sample(t);
    const dir = pos.clone().sub(target);
    // Narrow viewports back the camera away so the subject keeps fitting.
    dir.multiplyScalar(this.aspectFit * this.offset.zoom);
    const spherical = new THREE.Spherical().setFromVector3(dir);
    spherical.theta += this.offset.yaw + Math.sin(time * .23) * .025 * sway;
    spherical.phi = clamp(spherical.phi - this.offset.pitch + Math.sin(time * .17) * .012 * sway, .25, Math.PI - .25);
    dir.setFromSpherical(spherical);
    this.camera.position.copy(target).add(dir);
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(target);
    this._target.copy(target);
  }
}

/** Drag-to-orbit offsets for the final, interactive state (no wheel capture, so the page still scrolls). */
export function attachOrbitDrag(element, rig, { maxYaw = .7, maxPitch = .35, onStart } = {}) {
  let dragging = false, x0 = 0, y0 = 0, yaw0 = 0, pitch0 = 0, enabled = false;
  element.addEventListener('pointerdown', event => {
    if (!enabled || event.button !== 0) return;
    dragging = true; x0 = event.clientX; y0 = event.clientY; yaw0 = rig.offset.yaw; pitch0 = rig.offset.pitch;
    element.setPointerCapture?.(event.pointerId);
    element.classList.add('is-dragging');
    onStart?.();
  });
  element.addEventListener('pointermove', event => {
    if (!dragging) return;
    rig.offset.yaw = clamp(yaw0 - (event.clientX - x0) * .005, -maxYaw, maxYaw);
    rig.offset.pitch = clamp(pitch0 + (event.clientY - y0) * .004, -maxPitch, maxPitch);
  });
  const end = () => { dragging = false; element.classList.remove('is-dragging'); };
  element.addEventListener('pointerup', end);
  element.addEventListener('pointercancel', end);
  return { setEnabled(value) { enabled = value; element.classList.toggle('can-drag', value); }, get dragging() { return dragging; } };
}

/**
 * A camera shot that fits a w × h region centred at `center`, seen from a little to the side and above.
 * side/up are fractions of the viewing distance; aspect is the stage aspect the shot is designed for.
 */
export function fitShot(t, center, w, h, { aspect = 1.75, fov = 30, side = .18, up = .12, margin = 1.08 } = {}) {
  const v = Math.tan(THREE.MathUtils.degToRad(fov) / 2);
  const d = Math.max(h / (2 * v), w / (2 * v * aspect)) * margin;
  const dir = new THREE.Vector3(side, up, 1).normalize();
  const pos = center.clone().add(dir.multiplyScalar(d * Math.hypot(side, up, 1)));
  return { t, pos: pos.toArray(), target: center.toArray(), fov };
}

/** A radial particle burst driven by a single progress uniform (0 → 1). */
export function makeBurst({ count = 260, color = '#3f68b0', radius = 4, size = 9, additive = false, seed = 3, spread = 1 } = {}) {
  const rand = mulberry32(seed);
  const dirs = new Float32Array(count * 3), speed = new Float32Array(count), delay = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const u = rand() * 2 - 1, a = rand() * Math.PI * 2, r = Math.sqrt(1 - u * u);
    dirs[i * 3] = r * Math.cos(a); dirs[i * 3 + 1] = u * .75 + .15; dirs[i * 3 + 2] = r * Math.sin(a) * spread;
    speed[i] = .45 + rand() * .55;
    delay[i] = rand() * .25;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geometry.setAttribute('aDir', new THREE.BufferAttribute(dirs, 3));
  geometry.setAttribute('aSpeed', new THREE.BufferAttribute(speed, 1));
  geometry.setAttribute('aDelay', new THREE.BufferAttribute(delay, 1));
  const material = new THREE.ShaderMaterial({
    uniforms: { uP: { value: 0 }, uRadius: { value: radius }, uSize: { value: size * Math.min(2, window.devicePixelRatio || 1) }, uColor: { value: new THREE.Color(color) } },
    vertexShader: `attribute vec3 aDir; attribute float aSpeed; attribute float aDelay; uniform float uP; uniform float uRadius; uniform float uSize; varying float vA;
      void main(){ float k = clamp((uP - aDelay) / (1.0 - aDelay), 0.0, 1.0); float e = 1.0 - pow(1.0 - k, 3.0);
      vec3 p = aDir * uRadius * aSpeed * e; p.y -= k * k * .35 * uRadius;
      vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;
      vA = (k > 0.0 ? 1.0 : 0.0) * (1.0 - k); gl_PointSize = uSize * (0.35 + 0.65 * (1.0 - k)) * (12.0 / -mv.z); }`,
    fragmentShader: `uniform vec3 uColor; varying float vA; void main(){ float d = length(gl_PointCoord - .5); float a = smoothstep(.5, .1, d) * vA; if (a < .01) discard; gl_FragColor = vec4(uColor, a);
      #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.userData.setProgress = p => { material.uniforms.uP.value = p; points.visible = p > 0 && p < 1; };
  points.userData.setProgress(0);
  return points;
}
