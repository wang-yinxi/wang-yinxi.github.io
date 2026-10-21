// Section 01 · Representation Residuals as a 3D stage.
// Layout and arrow directions follow the left panel of the paper's pipeline figure:
// o → three frozen encoders → per-layer features on a bus → α (attention residual, query w)
// → learnable [CLS] + patch tokens → RR layers 1–3 (each layer's output also joins the bus)
// → [CLS] + self-constructed representation tokens. The camera then moves to the [CLS],
// which is decoded post hoc into the readouts shown on the page.
import {
  THREE, C, FONT, SERIF, MONO, Stage, CameraRig, Wire, attachOrbitDrag, fontsReady,
  clamp, lerp, seg, smooth, easeOut, easeInOut, backOut,
  canvasTexture, texturedPlane, textPlane, roundRect, drawDrop, drawSnowflake, makeToken, makeSlab,
  solidMaterial, glassMaterial, glowSprite, setGroupOpacity, roundedBox, addOutline,
  loadImage, imageTexture, photoCardTexture, loadVoxels, makeVoxelMesh, makeDepthRelief, makeSemanticLayers,
  captionCard, fitShot, makeBurst,
} from './core.mjs';
import { REP_STEPS, REP_END } from './meta.mjs';

export { REP_STEPS, REP_END };

const S = 46;
const F = (x, y, z = 0) => new THREE.Vector3((x - 326) / S, (620 - y) / S, z);
const TOKEN = .78;


const P = {
  img: [.15, 1.3], input: [1.0, 2.1], enc: [1.7, 3.1], tap: [2.6, 4.0],
  a0: [3.9, 5.7], tok: [5.6, 6.5], l1: [6.4, 8.4],
  a1: [8.3, 10.1], l2: [10.0, 11.7], a2: [11.6, 13.3], l3: [13.2, 14.9], out: [14.8, 15.6],
  focus: [15.5, 17.0], burst: [16.9, 19.5],
};
// When each α row attends over depth, and when each layer runs self-attention.
const ROW_PHASE = { a0: [P.a0[0], P.a0[1]], a1: [P.a1[0] + .5, P.a1[1]], a2: [P.a2[0] + .5, P.a2[1]] };
const SA_PHASE = [[P.l1[0] + .2, P.l1[1]], [P.l2[0] + .15, P.l2[1]], [P.l3[0] + .15, P.l3[1]]];
// Illustrative softmax weights over depth (uneven on purpose; not trained values).
// Order: semantic ×3, geometric ×3, dynamic ×3, then RR layer outputs below the row.
const DEPTH_WEIGHTS = {
  a0: [.05, .13, .2, .04, .16, .1, .03, .07, .22],
  a1: [.03, .08, .15, .06, .2, .07, .02, .12, .09, .18],
  a2: [.04, .06, .1, .03, .12, .05, .05, .16, .07, .14, .18],
};
const TOKEN_ALL_X = [210, 257, 303, 349, 395, 441];

const ENCODERS = [
  { key: 'semantic', title: 'Semantic', model: 'DINOv3', y: 641, layers: [603, 641, 679], color: C.semantic, title2: C.navy,
    tint: ['#e7ecfa', '#c9c8ec'], feature: new URL('../../assets/paper/features/semantic.png', import.meta.url).href, bus: [497, 504, 511] },
  { key: 'geometric', title: 'Geometric', model: 'Depth Anything 3', y: 782, layers: [744, 782, 820], color: C.geometric, title2: C.teal,
    tint: ['#e2f3ee', '#f3d9df'], feature: new URL('../../assets/paper/features/geometric.png', import.meta.url).href, bus: [519, 526, 533] },
  { key: 'dynamic', title: 'Dynamic', model: 'OpenD4RT', y: 923, layers: [885, 923, 961], color: C.dynamic, title2: C.amber,
    tint: ['#f8ecd6', '#e3e2c9'], feature: new URL('../../assets/paper/features/dynamic.png', import.meta.url).href, bus: [541, 548, 555] },
];
const ROWS = { a0: 553, a1: 365, a2: 250 };
const LAYERS = [{ y: 423, name: 'Layer 1' }, { y: 307, name: 'Layer 2' }, { y: 192, name: 'Layer 3' }];
const TOKEN_X = [257, 303, 349, 395, 441];
const RR_BUS = { 1: 489, 2: 482 };

function layerFace(name) {
  return (ctx, w, h) => {
    roundRect(ctx, 4, 4, w - 8, h - 8, 26);
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#e6eefb');
    g.addColorStop(1, '#d4e2f6');
    ctx.fillStyle = g;
    ctx.fill();
    ctx.lineWidth = 7;
    ctx.strokeStyle = C.navy;
    ctx.stroke();
    drawDrop(ctx, 92, h / 2 + 2, Math.round(h * .3));
    ctx.font = `700 ${Math.round(h * .3)}px ${FONT}`;
    ctx.fillStyle = C.navy;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.fillText(`Representation Residuals ${name}`, w / 2 + 30, h / 2 + 3);
  };
}

function encoderFace(enc, feature) {
  return (ctx, w, h) => {
    roundRect(ctx, 4, 4, w - 8, h - 8, 30);
    ctx.save();
    ctx.clip();
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, enc.tint[0]);
    g.addColorStop(1, enc.tint[1]);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    if (feature) {
      ctx.globalAlpha = .33;
      ctx.drawImage(feature, 0, 0, w, h);
      ctx.globalAlpha = 1;
      const veil = ctx.createLinearGradient(0, 0, w, 0);
      veil.addColorStop(0, 'rgba(255,255,255,.72)');
      veil.addColorStop(.42, 'rgba(255,255,255,.42)');
      veil.addColorStop(1, 'rgba(255,255,255,.08)');
      ctx.fillStyle = veil;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.restore();
    roundRect(ctx, 4, 4, w - 8, h - 8, 30);
    ctx.lineWidth = 6;
    ctx.strokeStyle = enc.color;
    ctx.stroke();
    drawSnowflake(ctx, 52, 52, 24, '#7d8fb0');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = enc.title2;
    ctx.font = `700 ${Math.round(h * .15)}px ${FONT}`;
    const cx = w * .19;
    ctx.fillText(enc.title, cx, h * .4);
    ctx.fillText('Encoder', cx, h * .56);
    ctx.font = `500 ${Math.round(h * .085)}px ${FONT}`;
    ctx.fillStyle = 'rgba(51,65,90,.72)';
    ctx.fillText(enc.model, cx, h * .73);
    // the vertical ellipsis between layers, as in the figure
    ctx.fillStyle = 'rgba(80,96,128,.6)';
    for (const fy of [.335, .645]) for (let k = -1; k <= 1; k++) {
      ctx.beginPath();
      ctx.arc(w * .662, h * fy + k * 9, 3.4, 0, Math.PI * 2);
      ctx.fill();
    }
  };
}

function encoderLayerFace(color) {
  return (ctx, w, h) => {
    roundRect(ctx, 4, 4, w - 8, h - 8, 18);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = 6;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.font = `500 ${Math.round(h * .5)}px ${FONT}`;
    ctx.fillStyle = '#2f3a4c';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('encoder layer', w / 2, h / 2 + 2);
  };
}

function insetFace(ctx, w, h) {
  // AttnRes operation inset, redrawn from the figure: Q = w; K, V = stacked features.
  roundRect(ctx, 6, 6, w - 12, h - 12, 26);
  ctx.fillStyle = 'rgba(255,255,255,.93)';
  ctx.fill();
  ctx.setLineDash([14, 10]);
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#9aa7bd';
  ctx.stroke();
  ctx.setLineDash([]);
  const col = { q: w * .2, k: w * .5, v: w * .78 };
  ctx.font = `700 34px ${FONT}`;
  ctx.fillStyle = '#2f3a4c';
  ctx.textAlign = 'center';
  ctx.fillText('Q', col.q, 70);
  ctx.fillText('K', col.k, 70);
  ctx.fillText('V', col.v, 70);
  const box = (x, y, fill, stroke, bw = 64, bh = 40) => {
    roundRect(ctx, x - bw / 2, y - bh / 2, bw, bh, 9);
    ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 4; ctx.strokeStyle = stroke; ctx.stroke();
  };
  box(col.q, 125, C.red50, C.red, 64, 52);
  ctx.font = `italic 600 36px ${SERIF}`;
  ctx.fillStyle = C.red;
  ctx.fillText('w', col.q, 136);
  const stack = [[C.blue150, C.semantic], ['#d9efe9', C.geometric], ['#f7e6c9', C.dynamic]];
  for (const x of [col.k, col.v]) stack.forEach(([fill, stroke], i) => box(x, 118 + i * 46, fill, stroke));
  const arrow = (pts, color = '#4a4a4a') => {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(...pts[0]); for (const p of pts.slice(1)) ctx.lineTo(...p); ctx.stroke();
    const [x1, y1] = pts.at(-1), [x0, y0] = pts.at(-2);
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 16 * Math.cos(a - .45), y1 - 16 * Math.sin(a - .45));
    ctx.lineTo(x1 - 16 * Math.cos(a + .45), y1 - 16 * Math.sin(a + .45));
    ctx.closePath(); ctx.fill();
  };
  const times = (x, y) => {
    ctx.beginPath(); ctx.arc(x, y, 17, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 3.5; ctx.strokeStyle = '#555'; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); ctx.stroke();
  };
  const m1 = [col.k - 28, 290], m2 = [col.v - 28, 352];
  arrow([[col.q, 152], [col.q, m1[1]], [m1[0] - 19, m1[1]]]);
  arrow([[col.k, 210], [col.k, m1[1] - 19]]);
  times(...m1);
  arrow([[m1[0], m1[1] + 18], [m1[0], m2[1]], [m2[0] - 19, m2[1]]]);
  ctx.font = `500 26px ${FONT}`;
  ctx.fillStyle = '#4a4a4a';
  ctx.textAlign = 'left';
  ctx.fillText('softmax', m1[0] + 14, m1[1] + 44);
  arrow([[col.v, 210], [col.v, m2[1] - 19]]);
  times(...m2);
  arrow([[m2[0], m2[1] + 18], [m2[0], h - 40]]);
  arrow([[w - 20, 164], [col.v + 40, 164]], '#4a4a4a');
  ctx.textAlign = 'center';
}

function insetCaption(ctx, w, h) {
  ctx.font = `600 ${Math.round(h * .62)}px ${FONT}`;
  ctx.fillStyle = C.red;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('AttnRes Op (α)', w / 2, h / 2);
}

/** Small italic glyph plane, e.g. α or w. */
function glyph(text, { color = C.red, height = .34, family = SERIF, italic = true, weight = 600 } = {}) {
  const mesh = textPlane(text, { height, family, italic, weight, color, px: 120 });
  mesh.renderOrder = 12;
  return mesh;
}

export class RepresentScene {
  constructor(host, { onStep, onSelectModality, onReady, onFocusChange } = {}) {
    this.host = host;
    this.onStep = onStep;
    this.onSelectModality = onSelectModality;
    this.onReady = onReady;
    this.onFocusChange = onFocusChange;
    this.t = 0;
    this.idle = 0;
    this.playing = true;
    this.speed = 1;
    this.panels = [];
    this.focus = -1;
    this.focusBlend = 0;
    this.stepIndex = -1;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.ready = this.init();
  }

  async init() {
    await fontsReady();
    const stage = this.stage = new Stage(this.host, { dark: false, fov: 34, shadows: true });
    stage.key.position.set(7, 15, 30);
    stage.key.target.position.set(0, 3, 0);
    this.scene = stage.scene;
    this.fogColor = new THREE.Color('#f3f6fb');
    this.scene.fog = new THREE.Fog(this.fogColor, 200, 400);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.wires = [];
    await this.buildStatic();
    this.buildCamera();
    stage.frameCallbacks.add((dt, now) => this.frame(dt, now));
    this.drag = attachOrbitDrag(stage.canvas, this.rig, { onStart: () => { this.userOrbit = true; } });
    this.setupPicking();
    this.update(this.reduced ? REP_END : 0, 0);
    stage.render();
    this.onReady?.();
  }

  // ------------------------------------------------------------ construction

  addWire(points, options, phase) {
    const wire = new Wire(points, { radius: .026, corner: .18, ...options });
    wire.userData.phase = phase;
    this.root.add(wire);
    this.wires.push(wire);
    wire.setProgress(0);
    return wire;
  }

  async buildStatic() {
    const root = this.root;
    const features = {};
    await Promise.all(ENCODERS.map(async enc => {
      try { features[enc.key] = await loadImage(enc.feature); } catch { features[enc.key] = null; }
    }));

    // ---- ground shadow plane for depth cues
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshBasicMaterial({
      map: canvasTexture(256, 256, (ctx, w) => {
        const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
        g.addColorStop(0, 'rgba(47,85,151,.10)');
        g.addColorStop(1, 'rgba(47,85,151,0)');
        ctx.fillStyle = g; ctx.fillRect(0, 0, w, w);
      }), transparent: true, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(0, F(0, 1010).y, 0);
    shadow.scale.set(.45, .3, 1);
    root.add(shadow);
    const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(90, 70), new THREE.ShadowMaterial({ color: '#1f3d73', opacity: .075 }));
    backdrop.position.set(0, 2, -2.1);
    backdrop.receiveShadow = true;
    root.add(backdrop);

    // ---- input observation card stack
    this.inputGroup = new THREE.Group();
    this.inputGroup.position.copy(F(95, 783, .2));
    root.add(this.inputGroup);
    this.inputCards = [];
    const oLabel = textPlane('oₜ', { height: .52, family: SERIF, italic: true, weight: 600, color: '#2f3a4c' });
    oLabel.position.copy(F(95, 846, .2));
    root.add(oLabel);
    this.oLabel = oLabel;

    // ---- encoders
    this.encoders = ENCODERS.map(enc => {
      const group = new THREE.Group();
      group.position.copy(F(326, enc.y, -.76));
      const casing = makeSlab({ w: 6, h: 2.66, d: 1.4, r: .2, stroke: enc.color, strokeWidth: .03,
        material: solidMaterial(enc.tint[0], { roughness: .55, clearcoat: .35 }), draw: encoderFace(enc, features[enc.key]), texW: 1400 });
      group.add(casing);
      const layers = enc.layers.map(y => {
        const slab = makeSlab({ w: 3.22, h: .46, d: .5, r: .1, stroke: enc.color, strokeWidth: .022,
          material: solidMaterial('#ffffff', { roughness: .4, emissive: enc.color, emissiveIntensity: 0 }),
          draw: encoderLayerFace(enc.color), texW: 700 });
        slab.position.copy(F(371, y, 0)).sub(group.position);
        slab.userData.baseZ = slab.position.z;
        group.add(slab);
        return slab;
      });
      const glow = glowSprite(enc.color, 7.5, { opacity: 0, additive: false });
      glow.position.set(0, 0, -.9);
      group.add(glow);
      root.add(group);
      return { ...enc, group, casing, layers, glow };
    });

    // ---- RR layers
    this.rr = LAYERS.map(layer => {
      const slab = makeSlab({ w: 6, h: 1.04, d: 1.0, r: .16, stroke: C.navy, strokeWidth: .035,
        material: solidMaterial(C.blue150, { roughness: .38, clearcoat: .8, emissive: C.blue400, emissiveIntensity: 0 }),
        draw: layerFace(layer.name), texW: 1500 });
      slab.position.copy(F(326, layer.y, 0));
      const glow = glowSprite(C.blue400, 6.5, { opacity: 0, additive: false });
      glow.position.set(0, 0, -.7);
      glow.scale.set(8, 2.4, 1);
      slab.add(glow);
      root.add(slab);
      return { ...layer, slab, glow };
    });

    // ---- attention residual groups: w → α
    this.alphas = Object.entries(ROWS).map(([key, y]) => {
      const group = new THREE.Group();
      group.position.copy(F(326, y, 0));
      const ring = new THREE.Mesh(new THREE.TorusGeometry(.27, .05, 16, 48), solidMaterial(C.red, { roughness: .35, clearcoat: .6 }));
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(.25, .25, .12, 40), solidMaterial('#fff7f5', { roughness: .5 }));
      disc.rotation.x = Math.PI / 2;
      const a = glyph('α', { height: .36 });
      a.position.z = .08;
      const w = makeSlab({ w: .58, h: .46, d: .3, r: .08, stroke: C.red, strokeWidth: .025, material: solidMaterial(C.red50, { roughness: .5 }) });
      w.position.copy(F(278, y, 0)).sub(group.position);
      const wg = glyph('w', { height: .3 });
      wg.position.copy(w.position).add(new THREE.Vector3(0, -.02, .17));
      const link = new Wire([F(293, y, 0).sub(group.position), F(312, y, 0).sub(group.position)], { color: C.red, radius: .022, arrow: false, head: 0 });
      const glow = glowSprite(C.redGlow, 2.2, { opacity: 0, additive: false });
      group.add(disc, ring, a, w, wg, link, glow);
      root.add(group);
      return { key, y, group, ring, disc, glow, w };
    });

    // ---- AttnRes inset card (connected to the topmost α group, as in the figure)
    const inset = new THREE.Group();
    const insetCard = texturedPlane(canvasTexture(560, 680, insetFace), 3.2, 3.88);
    insetCard.renderOrder = 6;
    const insetCaptionPlane = texturedPlane(canvasTexture(560, 64, insetCaption), 3.2, .37);
    insetCaptionPlane.position.y = -2.25;
    inset.add(insetCard, insetCaptionPlane);
    inset.position.copy(F(105, 300, .3));
    inset.rotation.y = .22;
    root.add(inset);
    this.inset = inset;
    const dotted = canvasTexture(400, 160, (ctx, w, h) => {
      roundRect(ctx, 6, 6, w - 12, h - 12, 22);
      ctx.setLineDash([12, 10]); ctx.lineWidth = 5; ctx.strokeStyle = '#9aa7bd'; ctx.stroke();
    });
    const groupFrame = texturedPlane(dotted, 2.0, .8);
    groupFrame.position.copy(F(304, 250, .32));
    root.add(groupFrame);
    this.insetLinks = [
      new Wire([F(262, 236, .3), F(170, 218, .3)], { color: '#9aa7bd', radius: .014, arrow: false, dash: .16, head: 0 }),
      new Wire([F(262, 266, .3), F(170, 386, .3)], { color: '#9aa7bd', radius: .014, arrow: false, dash: .16, head: 0 }),
    ];
    for (const wire of this.insetLinks) root.add(wire);

    // ---- tokens
    this.inTokens = TOKEN_X.map(x => {
      const token = makeToken('rep', TOKEN);
      token.position.copy(F(x, 485, 0));
      root.add(token);
      return token;
    });
    this.learnable = makeToken('learnable', TOKEN);
    this.learnable.position.copy(F(210, 485, 0));
    root.add(this.learnable);
    const learnLabel = textPlane('* learnable [CLS]', { height: .27, color: C.navy, weight: 600 });
    learnLabel.position.copy(F(115, 485, .3));
    root.add(learnLabel);
    this.learnLabel = learnLabel;

    this.outTokens = TOKEN_X.map(x => {
      const token = makeToken('rep', TOKEN);
      token.position.copy(F(x, 135, 0));
      root.add(token);
      return token;
    });
    this.cls = makeToken('cls', TOKEN);
    this.cls.position.copy(F(210, 135, 0));
    this.clsHome = this.cls.position.clone();
    root.add(this.cls);
    this.clsGlow = glowSprite(C.blue400, 3.5, { opacity: 0, additive: false });
    this.cls.add(this.clsGlow);
    this.clsRing = new THREE.Mesh(new THREE.TorusGeometry(.75, .009, 8, 128),
      new THREE.MeshBasicMaterial({ color: C.blue500, transparent: true, opacity: 0, toneMapped: false }));
    this.cls.add(this.clsRing);
    this.clsRing2 = this.clsRing.clone();
    this.clsRing2.material = this.clsRing.material.clone();
    this.cls.add(this.clsRing2);
    this.cls.traverse(node => { if (node.material) node.material.fog = false; });
    this.burst = makeBurst({ count: 320, color: C.blue500, radius: 6.5, size: 10 });
    this.burst2 = makeBurst({ count: 160, color: C.amber2, radius: 4.5, size: 8, seed: 9 });
    root.add(this.burst, this.burst2);
    const clsLabel = textPlane('[CLS]', { height: .32, family: MONO, weight: 500, color: C.navy });
    clsLabel.position.copy(F(210, 98, 0));
    root.add(clsLabel);
    const outLabel = textPlane('self-constructed representation tokens', { height: .3, color: C.slate, weight: 500 });
    outLabel.position.copy(F(349, 98, 0));
    root.add(outLabel);
    this.outLabels = [clsLabel, outLabel];

    // ---- wires (directions follow the figure)
    const gray = '#4a5568';
    // o → encoders
    this.inputWires = ENCODERS.map(enc => this.addWire(
      [F(148, 783), F(166, 783), F(166, enc.y), F(186, enc.y)], { color: gray, radius: .03 }, 'input'));
    // encoder layers → bus → up to the α rows (three segments each so the bus can grow upward)
    this.busWires = [];
    for (const enc of ENCODERS) {
      enc.layers.forEach((y, i) => {
        const x = enc.bus[i];
        const a = this.addWire([F(446, y), F(x, y), F(x, ROWS.a0)], { color: enc.color, arrow: false, flow: .9, flowGap: 2.1 }, 'tap');
        const b = this.addWire([F(x, ROWS.a0), F(x, ROWS.a1)], { color: enc.color, arrow: false, flow: .9, flowGap: 2.1 }, 'busB');
        const c = this.addWire([F(x, ROWS.a1), F(x, ROWS.a2)], { color: enc.color, arrow: false, flow: .9, flowGap: 2.1 }, 'busC');
        this.busWires.push({ enc: enc.key, i, a, b, c, x });
      });
    }
    // RR layer outputs also join the bus (blue lines in the figure)
    this.rrBus1 = this.addWire([F(465, 423), F(RR_BUS[1], 423), F(RR_BUS[1], ROWS.a1)], { color: C.navy, arrow: false, flow: .9, flowGap: 2.1 }, 'rr1');
    this.rrBus1b = this.addWire([F(RR_BUS[1], ROWS.a1), F(RR_BUS[1], ROWS.a2)], { color: C.navy, arrow: false, flow: .9, flowGap: 2.1 }, 'rr1b');
    this.rrBus2 = this.addWire([F(465, 307), F(RR_BUS[2], 307), F(RR_BUS[2], ROWS.a2)], { color: C.navy, arrow: false, flow: .9, flowGap: 2.1 }, 'rr2');
    // bus → α (red, pointing into α)
    this.redWires = Object.entries(ROWS).map(([key, y]) =>
      this.addWire([F(566, y), F(340, y)], { color: C.red, radius: .034, arrowScale: 1.1 }, key));
    // junction dots on the bus; each carries the depth weight α gives its source layer
    this.dots = [];
    const dotGeometry = new THREE.SphereGeometry(.07, 16, 12);
    const rowCount = { a0: 0, a1: 0, a2: 0 };
    const junction = (x, y, color, row, source) => {
      const dot = new THREE.Mesh(dotGeometry, new THREE.MeshBasicMaterial({ color, transparent: true, toneMapped: false }));
      dot.position.copy(F(x, y, 0));
      dot.userData.row = row;
      dot.userData.source = source;
      dot.userData.weight = DEPTH_WEIGHTS[row][rowCount[row]++] ?? .05;
      dot.scale.setScalar(0);
      const halo = glowSprite(color, .9, { opacity: 0, additive: false });
      dot.add(halo);
      dot.userData.halo = halo;
      root.add(dot);
      this.dots.push(dot);
    };
    for (const row of ['a0', 'a1', 'a2']) {
      for (const bw of this.busWires) {
        const e = ENCODERS.findIndex(enc => enc.key === bw.enc);
        junction(bw.x, ROWS[row], ENCODERS[e].color, row, { kind: 'enc', e, i: bw.i, wire: bw });
      }
    }
    junction(RR_BUS[1], ROWS.a1, C.navy, 'a1', { kind: 'rr', l: 0 });
    junction(RR_BUS[1], ROWS.a2, C.navy, 'a2', { kind: 'rr', l: 0 });
    junction(RR_BUS[2], ROWS.a2, C.navy, 'a2', { kind: 'rr', l: 1 });

    // the query's look down the stack: a scan bar sweeps every depth below the active α
    this.scan = new THREE.Group();
    const bar = new THREE.Mesh(new THREE.PlaneGeometry(9.6, .11), new THREE.MeshBasicMaterial({ color: C.redGlow, transparent: true, opacity: .9, toneMapped: false, depthWrite: false }));
    const haze = glowSprite(C.redGlow, 1, { opacity: .5, additive: false });
    haze.scale.set(11, 1.5, 1);
    this.scan.add(haze, bar);
    this.scan.position.set(F(330, 0).x, 0, .78);
    this.scan.visible = false;
    root.add(this.scan);
    this.depthLabel = textPlane('softmax over depth', { height: .3, color: C.red, weight: 700, italic: true });
    this.depthLabel.renderOrder = 14;
    root.add(this.depthLabel);

    // self-attention inside each RR layer: [CLS] + patch tokens attend to one another
    this.selfAttention = LAYERS.map((layer, l) => {
      const group = new THREE.Group();
      root.add(group);
      const minis = TOKEN_ALL_X.map((x, i) => {
        const token = makeToken(i === 0 ? 'cls' : 'rep', .5);
        token.position.copy(F(x, layer.y + 4, 1.3));
        token.userData.home = token.position.clone();
        token.scale.setScalar(.001);
        group.add(token);
        return token;
      });
      const arcs = [];
      const rand = (n => () => { n = (n * 9301 + 49297) % 233280; return n / 233280; })(17 + l * 31);
      let order = 0;
      for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) {
        const a = F(TOKEN_ALL_X[i], layer.y - 8, 1.42), b = F(TOKEN_ALL_X[j], layer.y - 8, 1.42);
        const span = j - i;
        // cubic arcs rise mostly upward so every pair stays distinguishable from the front
        const lift = .42 + .26 * span, fwd = .18 + .08 * span;
        const curve = new THREE.CubicBezierCurve3(a, a.clone().add(new THREE.Vector3(.12 * span, lift, fwd)),
          b.clone().add(new THREE.Vector3(-.12 * span, lift, fwd)), b);
        const cls = i === 0;
        const weight = cls ? .7 + .3 * rand() : .25 + .55 * rand();
        const wire = new Wire(curve, { color: cls ? C.navy : C.blue400, radius: cls ? .014 + .016 * weight : .008 + .012 * weight,
          arrow: false, flow: 1, flowGap: .9, head: 1, opacity: cls ? 1 : .85 });
        wire.userData.order = order++;
        wire.setProgress(0);
        group.add(wire);
        arcs.push(wire);
      }
      const label = textPlane('self-attention · [CLS] + patch tokens', { height: .28, color: C.navy, weight: 700 });
      label.position.copy(F(326, layer.y - 62, 2.6));
      label.renderOrder = 14;
      group.add(label);
      return { group, minis, arcs, label };
    });

    // α0 → patch tokens (the learnable [CLS] has no input arrow)
    this.tokenWires = TOKEN_X.map(x => this.addWire([F(326, 566), F(326, 524), F(x, 524), F(x, 505)], { color: gray }, 'tok'));
    // tokens → layer 1
    this.intoL1 = [210, ...TOKEN_X].map(x => this.addWire([F(x, 466), F(x, 449)], { color: gray, radius: .024 }, 'l1'));
    // layer → α → next layer
    this.stack = [
      this.addWire([F(326, 398), F(326, 379)], { color: gray }, 'l1a1'),
      this.addWire([F(326, 351), F(326, 333)], { color: gray }, 'a1l2'),
      this.addWire([F(326, 282), F(326, 264)], { color: gray }, 'l2a2'),
      this.addWire([F(326, 236), F(326, 217)], { color: gray }, 'a2l3'),
    ];

    // ---- decode stage: anchor above the token row
    this.gallery = new THREE.Group();
    root.add(this.gallery);
    this.G = new THREE.Vector3(0, 13.1, 3.4); // where the [CLS] settles when opened
    this.castShadows(root);
  }

  castShadows(node) {
    node.traverse(object => {
      if (!object.isMesh || object.userData.isOutline) return;
      const m = object.material;
      if (m?.isMeshPhysicalMaterial || object.userData.cast) object.castShadow = true;
    });
  }

  buildCamera() {
    const G = this.G;
    const shot = (t, [x0, y0, x1, y1], side = .2, up = .1, margin = 1.08) => {
      const a = F(x0, y0), b = F(x1, y1);
      const center = a.clone().add(b).multiplyScalar(.5);
      return fitShot(t, center, Math.abs(b.x - a.x), Math.abs(b.y - a.y), { side, up, margin, fov: 34 });
    };
    const gallery = (t) => fitShot(t, G.clone().add(new THREE.Vector3(0, 2.75, 0)), 19.6, 9.6, { side: 0, up: .12, margin: 1.0, fov: 34 });
    this.rig = new CameraRig(this.stage.camera, [
      shot(0, [20, 590, 470, 1000], -.62, -.04, 1.0),
      shot(1.6, [10, 570, 560, 1000], -.4, .1, 1.02),
      shot(3.4, [130, 520, 610, 1000], .42, .2, 1.0),
      shot(4.6, [140, 500, 640, 1000], .3, .14, 1.02),
      shot(6.0, [140, 380, 620, 600], .22, .12, 1.0),
      shot(7.6, [150, 300, 600, 520], .08, .1, 1.0),
      shot(9.2, [90, 320, 640, 1000], .3, .12, 1.0),
      shot(10.9, [150, 190, 600, 410], .08, .1, 1.0),
      shot(12.5, [40, 210, 650, 1000], .26, .1, 1.0),
      shot(14.1, [150, 70, 600, 300], .08, .1, 1.0),
      { t: 15.7, pos: [2.6, 11.0, 8.0], target: [-1.6, 10.3, .4], fov: 34 },
      { t: 16.9, pos: [.6, G.y + .8, G.z + 5.6], target: [0, G.y + .1, G.z], fov: 34 },
      gallery(18.7),
      gallery(REP_END),
    ]);
  }

  // ------------------------------------------------------------ case data

  async setCase(item) {
    await this.ready;
    const token = this.caseToken = Symbol('case');
    this.item = item;
    const frame = item.frames[0];
    const views = (frame.views?.length ? frame.views : item.views) || [{ key: 'front', url: item.input }];
    const selected = views.find(v => v.key === item.selectedCamera) || views[0];

    // input cards
    const images = await Promise.all(views.slice(0, 4).map(v => loadImage(v.url).catch(() => null)));
    if (token !== this.caseToken) return;
    for (const card of this.inputCards) { this.inputGroup.remove(card); card.material.map?.dispose(); card.geometry.dispose(); }
    this.inputCards = images.filter(Boolean).map((image, i, list) => {
      const ratio = image.naturalHeight / image.naturalWidth;
      const w = ratio > .9 ? 1.95 : 2.35, h = w * ratio;
      const card = texturedPlane(photoCardTexture(image, { w: 640 }), w, h, { depthWrite: true });
      card.castShadow = true;
      card.renderOrder = 8 - i;
      card.userData.index = list.length - 1 - i;
      this.inputGroup.add(card);
      return card;
    }).reverse();
    this.inputCards.forEach((card, i) => { card.userData.index = i; });

    await this.buildPanels(item, frame, selected);
    if (token !== this.caseToken) return;
    this.replay();
  }

  async buildPanels(item, frame, view) {
    for (const panel of this.panels) this.gallery.remove(panel.group);
    for (const trail of this.trails || []) this.gallery.remove(trail);
    this.panels = [];
    this.trails = [];
    const specs = [];
    if (frame.voxels && item.grid) specs.push({ id: 'occ', title: '3D occupancy', sub: '120 × 80 × 20 voxels' });
    if (view?.semantic || frame.semantic) specs.push({ id: 'semantic', title: 'Semantics', sub: '64 × 64 readout' });
    if (view?.depth || frame.depth) specs.push({ id: 'depth', title: 'Depth', sub: '32 × 32 inverse depth' });
    if (view?.box || frame.box) specs.push({ id: 'box', title: 'Target', sub: 'Target box on RGB' });
    if (frame.egoMotion?.length) specs.push({ id: 'ego', title: 'Ego motion', sub: 'Preceding 0.6 s' });
    const allowed = item.availableModalities;
    const list = specs.filter(s => !allowed || allowed.includes(s.id)).slice(0, 4);
    const n = list.length;
    const G = this.G;
    const built = await Promise.all(list.map((spec, i) => this.buildPanel(spec, item, frame, view).catch(error => {
      console.warn('Readout panel', spec.id, error);
      return null;
    })));
    const spread = { 1: [90], 2: [124, 56], 3: [148, 90, 32], 4: [154, 112, 68, 26] }[n] || [90];
    built.forEach((group, i) => {
      if (!group) return;
      const spec = list[i];
      const a = spread[i] * Math.PI / 180;
      const R = n >= 4 ? 6.3 : 5.4;
      const target = new THREE.Vector3(G.x + Math.cos(a) * R * 1.22, G.y + .7 + Math.sin(a) * R * .6, G.z + Math.abs(Math.cos(a)) * 1.4);
      group.userData.target = target;
      group.userData.rotY = -Math.cos(a) * .42;
      group.position.copy(G);
      group.scale.setScalar(.01);
      group.traverse(node => { if (node.material) node.material.fog = false; });
      this.gallery.add(group);
      const end = target.clone().add(new THREE.Vector3(0, -1.88, 0));
      const mid = G.clone().lerp(end, .5).add(new THREE.Vector3(0, -.15, 1.1));
      const trail = new Wire(new THREE.CatmullRomCurve3([G.clone(), mid, end]),
        { color: C.blue500, radius: .024, arrow: 'end', flow: 1, flowGap: 1.4, head: 1, arrowScale: .9 });
      trail.traverse(node => { if (node.material) node.material.fog = false; });
      trail.setProgress(0);
      this.gallery.add(trail);
      this.trails.push(trail);
      this.panels.push({ ...spec, group, trail, index: this.panels.length });
    });
  }

  async buildPanel(spec, item, frame, view) {
    const group = new THREE.Group();
    group.userData.modality = spec.id;
    const W = 3.6, H = 3.0;
    // card backdrop
    const backTex = canvasTexture(720, 600, (ctx, w, h) => {
      roundRect(ctx, 4, 4, w - 8, h - 8, 38);
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, '#ffffff');
      g.addColorStop(1, '#edf2fa');
      ctx.fillStyle = g; ctx.fill();
      ctx.lineWidth = 4; ctx.strokeStyle = C.blue300; ctx.stroke();
    });
    const slab = new THREE.Mesh(roundedBox(W, H, .16, .16, 4), solidMaterial('#f7f9fd', { roughness: .4, clearcoat: .8 }));
    slab.position.z = -.1;
    slab.castShadow = true;
    group.add(slab);
    const back = texturedPlane(backTex, W, H, { depthWrite: false });
    back.position.z = -.015;
    back.renderOrder = -2;
    group.add(back);
    const hit = new THREE.Mesh(new THREE.PlaneGeometry(W, H + .8), new THREE.MeshBasicMaterial({ visible: false }));
    hit.userData.panel = spec.id;
    group.add(hit);
    group.userData.hit = hit;
    const caption = captionCard(spec.title, spec.sub, { width: 2.5 });
    caption.position.set(0, H / 2 + .36, .06);
    group.add(caption);
    const content = new THREE.Group();
    group.add(content);
    group.userData.content = content;

    if (spec.id === 'occ') {
      const voxels = await loadVoxels(frame.voxels);
      const mesh = makeVoxelMesh(voxels, item.grid, { fit: 2.75, spread: 1.6 });
      mesh.material.transparent = false;
      const pivot = new THREE.Group();
      pivot.rotation.set(.5, -.7, 0);
      pivot.add(mesh);
      pivot.position.set(0, -.62, .45);
      const plate = new THREE.Mesh(new THREE.BoxGeometry(2.75 * 4 / 6 + .2, .05, 2.75 + .2), solidMaterial(C.blue100, { roughness: .6 }));
      plate.position.y = -.03;
      pivot.add(plate);
      content.add(pivot);
      group.userData.voxels = mesh;
      group.userData.spin = pivot;
      // robot marker at the origin of the robot frame
      const robot = new THREE.Mesh(new THREE.ConeGeometry(.08, .24, 3), new THREE.MeshBasicMaterial({ color: C.red, toneMapped: false }));
      robot.rotation.x = -Math.PI / 2;
      const s = mesh.userData.scale;
      const cx = (item.grid.min[0] + item.grid.max[0]) / 2, cy = (item.grid.min[1] + item.grid.max[1]) / 2;
      robot.position.set(cy * s, .05, cx * s);
      pivot.add(robot);
    } else if (spec.id === 'semantic') {
      const image = await loadImage(view?.semantic || frame.semantic);
      const layers = makeSemanticLayers(image, { width: 2.55, height: 2.55 });
      layers.position.set(0, .05, .25);
      layers.rotation.set(-.08, .38, 0);
      content.add(layers);
      group.userData.semantic = layers;
    } else if (spec.id === 'depth') {
      const image = await loadImage(view?.depth || frame.depth);
      const relief = makeDepthRelief(image, { width: 2.7, height: 2.7, amp: .8 });
      relief.rotation.set(-.78, 0, 0);
      relief.position.set(0, -.05, .45);
      const lines = new THREE.Mesh(relief.geometry, new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true, transparent: true, opacity: .14, toneMapped: false }));
      relief.add(lines);
      content.add(relief);
      group.userData.relief = relief;
    } else if (spec.id === 'box') {
      const url = view?.url || frame.scene;
      const image = await loadImage(url);
      const box = view?.box || frame.box;
      const ratio = image.naturalHeight / image.naturalWidth;
      const pw = 2.9, ph = pw * ratio;
      const photo = texturedPlane(photoCardTexture(image, { w: 720, borderPx: 8, radius: 22 }), pw, ph, { depthWrite: true });
      photo.position.z = .15;
      content.add(photo);
      if (box) {
        const frameTex = canvasTexture(720, Math.round(720 * ratio), (ctx, w, h) => {
          const [x0, y0, x1, y1] = box;
          ctx.setLineDash([22, 12]);
          ctx.lineWidth = 9;
          ctx.strokeStyle = C.box;
          roundRect(ctx, x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h, 10);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.box;
          roundRect(ctx, x0 * w, Math.max(0, y0 * h - 54), 170, 50, 10);
          ctx.fill();
          ctx.fillStyle = '#fff';
          ctx.font = `700 34px ${FONT}`;
          ctx.textBaseline = 'middle';
          ctx.fillText('target', x0 * w + 18, Math.max(0, y0 * h - 54) + 26);
        });
        const overlay = texturedPlane(frameTex, pw, ph);
        overlay.position.z = .5;
        overlay.renderOrder = 9;
        content.add(overlay);
        group.userData.boxOverlay = overlay;
      }
    } else if (spec.id === 'ego') {
      const plate = new THREE.Group();
      const grid = new THREE.GridHelper(2.8, 8, C.blue300, C.blue200);
      grid.material.transparent = true;
      grid.material.opacity = .9;
      plate.add(grid);
      const pts = frame.egoMotion.map(([x, y]) => [x, y]);
      const span = Math.max(.4, ...pts.map(p => Math.hypot(p[0], p[1])));
      const k = 1.9 / span;
      const curvePts = pts.map(([x, y]) => new THREE.Vector3(-y * k, .05, -x * k + .55));
      const curve = new THREE.CatmullRomCurve3(curvePts);
      const path = new Wire(curve, { color: C.red, radius: .045, arrow: 'end', head: 1, arrowScale: 1.1 });
      plate.add(path);
      for (const p of curvePts) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(.07, 16, 12), new THREE.MeshBasicMaterial({ color: '#2b2b2b', toneMapped: false }));
        dot.position.copy(p);
        plate.add(dot);
      }
      const robot = makeToken('cls', .26);
      robot.position.copy(curvePts.at(-1)).add(new THREE.Vector3(0, .14, 0));
      plate.add(robot);
      const tag = (text, p, color = C.slate) => {
        const label = textPlane(text, { height: .17, color, weight: 600 });
        label.position.copy(p).add(new THREE.Vector3(.3, .02, 0));
        label.rotation.x = -Math.PI / 2;
        plate.add(label);
      };
      tag('now', curvePts.at(-1), C.navy);
      tag('−0.6 s', curvePts[0]);
      plate.rotation.set(.95, -.25, 0);
      plate.position.set(0, -.2, .35);
      content.add(plate);
      group.userData.path = path;
    }
    return group;
  }

  // ------------------------------------------------------------ interaction

  setupPicking() {
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let down = null;
    const canvas = this.stage.canvas;
    const pick = event => {
      const rect = canvas.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, this.stage.camera);
      const hits = raycaster.intersectObjects(this.panels.map(p => p.group.userData.hit), false);
      return hits[0]?.object.userData.panel;
    };
    canvas.addEventListener('pointerdown', event => { down = [event.clientX, event.clientY]; });
    canvas.addEventListener('pointerup', event => {
      if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 6) return;
      if (this.t < P.burst[0] + 1) return;
      const id = pick(event);
      if (id) this.selectPanel(id, true);
    });
    canvas.addEventListener('pointermove', event => {
      if (this.t < P.burst[0] + 1) { canvas.style.cursor = ''; this.hover = null; return; }
      const id = pick(event);
      this.hover = id || null;
      canvas.style.cursor = id ? 'pointer' : '';
    });
  }

  selectPanel(id, fromScene = false) {
    const index = this.panels.findIndex(p => p.id === id);
    this.selected = id;
    if (fromScene) {
      this.focus = this.focus === index ? -1 : index;
      this.onFocusChange?.(this.focus >= 0 ? id : null);
      this.onSelectModality?.(id);
    }
    this.stage.start();
  }

  clearFocus() { this.focus = -1; this.onFocusChange?.(null); }

  // ------------------------------------------------------------ playback

  replay() {
    this.t = this.reduced ? REP_END : 0;
    this.idle = 0;
    this.playing = !this.reduced;
    this.focus = -1;
    this.rig.offset.yaw = 0;
    this.rig.offset.pitch = 0;
    this.stepIndex = -1;
    this.onFocusChange?.(null);
    this.stage.start();
  }
  pause() { this.playing = false; }
  play() { if (this.t >= REP_END) this.replay(); else this.playing = true; this.stage.start(); }
  seek(t) { this.t = clamp(t, 0, REP_END); this.stage.start(); }
  get duration() { return REP_END; }

  frame(dt, now) {
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t >= REP_END) { this.t = REP_END; this.playing = false; this.onEnd?.(); }
    }
    this.idle += dt;
    this.update(this.t, now);
  }

  // ------------------------------------------------------------ the timeline

  update(t, now = 0) {
    const time = now;
    const step = REP_STEPS.reduce((acc, s, i) => (t >= s.t ? i : acc), 0);
    if (step !== this.stepIndex) { this.stepIndex = step; this.onStep?.(step, REP_STEPS[step]); }
    this.onProgress?.(t / REP_END);

    // input cards fly in and fan into a stack
    const pi = seg(t, ...P.img);
    this.inputCards.forEach((card, i) => {
      const k = easeOut(seg(pi, i * .14, .55 + i * .14));
      const n = this.inputCards.length;
      const depth = n - 1 - i;
      card.position.set(lerp(-4.5, depth * -.16, k), lerp(1.6, depth * .16, k), lerp(-1.5, -depth * .18, k));
      card.rotation.set(0, lerp(.9, .22, k), lerp(.25, 0, k));
      card.material.opacity = k;
    });
    this.oLabel.material.opacity = seg(t, .9, 1.4);

    // o → encoders
    const pIn = seg(t, ...P.input);
    this.inputWires.forEach((wire, i) => { wire.setProgress(easeInOut(seg(pIn, i * .12, .76 + i * .12))); wire.tick(time); });

    // depth attention: each α's query looks down every layer below it
    const lit = new Map();
    const flowBoost = new Map();
    let scanY = null, scanOp = 0, labelRow = null, labelOp = 0;
    for (const [row, [a, b]] of Object.entries(ROW_PHASE)) {
      const sp = seg(t, a + .2, a + .85);
      const hold = Math.sin(Math.PI * seg(t, a + .45, b + .25));
      const rowY = F(0, ROWS[row]).y;
      let thisScan = null;
      if (sp > 0 && sp < 1) {
        thisScan = lerp(rowY - .3, F(0, 978).y, easeInOut(sp));
        scanY = thisScan;
        scanOp = Math.sin(Math.PI * sp);
      }
      if (hold > labelOp) { labelOp = hold; labelRow = row; }
      for (const dot of this.dots) {
        if (dot.userData.row !== row) continue;
        const src = dot.userData.source, w = dot.userData.weight;
        const key = src.kind === 'enc' ? `e${src.e}:${src.i}` : `r${src.l}`;
        const y = src.kind === 'enc' ? F(0, ENCODERS[src.e].layers[src.i]).y : F(0, LAYERS[src.l].y).y;
        const bump = thisScan === null ? 0 : Math.exp(-((thisScan - y) ** 2) / .16);
        lit.set(key, Math.max(lit.get(key) || 0, hold * w * 3.4 + bump * .55));
        dot.userData.k = hold;
        if (src.kind === 'enc') flowBoost.set(src.wire, Math.max(flowBoost.get(src.wire) || 0, hold * w * 6));
      }
    }
    this.scan.visible = scanY !== null;
    if (scanY !== null) {
      this.scan.position.y = scanY;
      this.scan.children.forEach(child => { child.material.opacity = (child.isSprite ? .6 : 1) * scanOp; });
    }
    this.depthLabel.visible = labelOp > .02;
    if (labelRow) {
      this.depthLabel.position.copy(F(640, ROWS[labelRow] - 22, .4));
      this.depthLabel.material.opacity = labelOp;
    }

    // encoders activate
    this.encoders.forEach((enc, e) => {
      const k = seg(t, P.enc[0] + e * .15, P.enc[1]);
      enc.glow.material.opacity = .22 * Math.sin(Math.PI * clamp(k * 1.1)) + (t > P.enc[1] ? .05 : 0);
      enc.layers.forEach((layer, i) => {
        const lk = seg(t, P.enc[0] + .25 + i * .22 + e * .12, P.enc[0] + .85 + i * .22 + e * .12);
        const att = Math.min(.75, lit.get(`e${e}:${i}`) || 0);
        layer.userData.body.material.emissiveIntensity = Math.max(.55 * Math.sin(Math.PI * lk) + (lk >= 1 ? .06 : 0), att);
        layer.position.z = layer.userData.baseZ + .14 * Math.sin(Math.PI * lk) + .16 * att;
      });
    });

    // feature taps grow up the bus row by row
    const tap = seg(t, ...P.tap);
    this.busWires.forEach((bw, n) => {
      const e = ENCODERS.findIndex(x => x.key === bw.enc);
      bw.a.setProgress(easeInOut(seg(tap, e * .12 + bw.i * .05, .62 + e * .12 + bw.i * .05)));
      bw.b.setProgress(easeInOut(seg(t, P.a1[0] + .1 + e * .06, P.a1[0] + .7 + e * .06)));
      bw.c.setProgress(easeInOut(seg(t, P.a2[0] + .1 + e * .06, P.a2[0] + .7 + e * .06)));
      const boost = .9 + (flowBoost.get(bw) || 0);
      for (const seg3 of [bw.a, bw.b, bw.c]) { seg3.material.uniforms.uFlow.value = boost; seg3.tick(time); }
    });
    this.rrBus1.setProgress(easeInOut(seg(t, P.a1[0], P.a1[0] + .7)));
    this.rrBus1b.setProgress(easeInOut(seg(t, P.a2[0] + .1, P.a2[0] + .7)));
    this.rrBus2.setProgress(easeInOut(seg(t, P.a2[0], P.a2[0] + .7)));
    for (const w of [this.rrBus1, this.rrBus1b, this.rrBus2]) w.tick(time);

    // red: bus → α
    const rowPhase = ROW_PHASE;
    this.redWires.forEach((wire, i) => {
      const [a] = Object.values(rowPhase)[i];
      wire.setProgress(easeInOut(seg(t, a, a + .5)));
      wire.tick(time);
    });
    for (const dot of this.dots) {
      const [a] = rowPhase[dot.userData.row];
      const k = dot.userData.k || 0;
      dot.scale.setScalar(backOut(seg(t, a, a + .3)) * (1 + 3.2 * dot.userData.weight * k));
      dot.userData.halo.material.opacity = Math.min(.9, dot.userData.weight * 5 * k);
    }
    this.alphas.forEach((alpha, i) => {
      const [, b] = Object.values(rowPhase)[i];
      const k = seg(t, b - .35, b + .25);
      const pulse = Math.sin(Math.PI * k);
      alpha.ring.scale.setScalar(1 + .4 * pulse);
      alpha.glow.material.opacity = .55 * pulse;
      alpha.disc.material.emissive?.set(C.red);
      if (alpha.disc.material.emissiveIntensity !== undefined) alpha.disc.material.emissiveIntensity = .5 * pulse;
      alpha.w.scale.setScalar(1 + .18 * pulse);
    });
    const insetPulse = Math.max(...this.alphas.map((_, i) => Math.sin(Math.PI * seg(t, Object.values(rowPhase)[i][1] - .5, Object.values(rowPhase)[i][1] + .4))));
    this.inset.scale.setScalar(1 + .04 * insetPulse);
    this.inset.position.z = .3 + .5 * insetPulse;

    // α0 → patch tokens, learnable [CLS] drops in
    const pt = seg(t, ...P.tok);
    this.tokenWires.forEach((wire, i) => { wire.setProgress(easeInOut(seg(pt, 0, .55 + i * .05))); wire.tick(time); });
    this.inTokens.forEach((token, i) => token.scale.setScalar(backOut(seg(pt, .45 + i * .07, .75 + i * .07))));
    const lk = easeOut(seg(t, P.tok[0] + .2, P.tok[1]));
    this.learnable.position.copy(F(210, 485, 0)).add(new THREE.Vector3(0, (1 - lk) * 2.2, 0));
    this.learnable.scale.setScalar(Math.max(.001, lk));
    this.learnable.rotation.y = (1 - lk) * 2.4;
    this.learnLabel.material.opacity = lk;

    // tokens → layer 1 → α1 → layer 2 → α2 → layer 3
    this.intoL1.forEach(wire => { wire.setProgress(easeInOut(seg(t, P.l1[0], P.l1[0] + .4))); wire.tick(time); });
    const stackPhase = [[P.a1[0], P.a1[0] + .4], [P.l2[0], P.l2[0] + .35], [P.a2[0], P.a2[0] + .4], [P.l3[0], P.l3[0] + .35]];
    this.stack.forEach((wire, i) => { wire.setProgress(easeInOut(seg(t, ...stackPhase[i]))); wire.tick(time); });
    this.rr.forEach((layer, l) => {
      const [s0, s1] = SA_PHASE[l];
      const sa = this.selfAttention[l];
      const appear = easeOut(seg(t, s0, s0 + .3));
      const leave = seg(t, s1 - .25, s1 + .1);
      const k = appear * (1 - leave);
      sa.minis.forEach((mini, i) => {
        const pop = backOut(seg(t, s0 + i * .03, s0 + .3 + i * .03));
        mini.scale.setScalar(Math.max(.001, pop * (1 - leave)));
        mini.visible = k > .01;
        mini.position.copy(mini.userData.home).add(new THREE.Vector3(0, 0, lerp(-1.1, 0, appear) - leave * 1.1));
      });
      sa.arcs.forEach(arc => {
        const o = arc.userData.order;
        // [CLS] arcs (the first five) draw first, then every patch-to-patch pair together
        const start = o < 5 ? s0 + .2 + o * .05 : s0 + .5 + (o - 5) * .02;
        arc.setProgress(easeInOut(seg(t, start, start + .35)) * (leave < 1 ? 1 : 0));
        arc.setOpacity((arc.userData.baseOpacity ??= arc.material.uniforms.uOpacity.value) * (1 - leave));
        arc.tick(time);
      });
      sa.label.visible = false;
      // the slab turns to glass so the tokens inside can be seen
      const body = layer.slab.userData.body, face = layer.slab.userData.face;
      face.material.opacity = 1 - .6 * k;
      // processing pulse when the layer hands its output up
      const pk = seg(t, s1 - .35, s1 + .35);
      const pulse = Math.sin(Math.PI * pk);
      const att = Math.min(.6, lit.get(`r${l}`) || 0);
      body.material.emissiveIntensity = Math.max(.55 * pulse, att);
      layer.glow.material.opacity = .45 * Math.max(pulse, att);
      layer.slab.scale.set(1 + .025 * pulse, 1 + .06 * pulse, 1 + .06 * pulse);
    });

    // outputs emerge from layer 3
    const po = seg(t, ...P.out);
    [this.cls, ...this.outTokens].forEach((token, i) => {
      const k = easeOut(seg(po, i * .08, .5 + i * .08));
      token.scale.setScalar(Math.max(.001, backOut(clamp(k))));
      if (token !== this.cls) token.position.copy(F(TOKEN_X[i - 1], 135, 0)).add(new THREE.Vector3(0, (k - 1) * .9, 0));
    });
    for (const label of this.outLabels) label.material.opacity = seg(po, .5, 1) * (1 - seg(t, P.focus[0], P.focus[0] + .6));

    // focus: the [CLS] lifts out of the diagram toward the camera
    const pf = easeInOut(seg(t, P.focus[0] + .2, P.focus[1]));
    if (po > 0) {
      const home = this.clsHome.clone().add(new THREE.Vector3(0, (easeOut(seg(po, 0, .5)) - 1) * .9, 0));
      const lifted = home.clone().lerp(this.G, pf);
      lifted.y += Math.sin(Math.PI * pf) * .9;
      this.cls.position.copy(lifted);
      this.cls.scale.setScalar(Math.max(.001, backOut(clamp(easeOut(seg(po, 0, .5))))) * (1 + .55 * pf));
    }
    this.cls.rotation.set(Math.sin(pf * Math.PI) * .4, pf * Math.PI * 2 + (t >= P.burst[0] ? (this.idle * .25) % (Math.PI * 2) * 0 : 0), 0);
    const charge = seg(t, P.focus[0] + .6, P.burst[0] + .3);
    const glowBase = t >= P.burst[0] ? .5 + .12 * Math.sin(time * 2.2) : 0;
    this.clsGlow.material.opacity = Math.max(.85 * Math.sin(Math.PI * charge), glowBase);
    this.clsGlow.scale.setScalar(3.2 + 3.4 * Math.max(charge, t >= P.burst[0] ? .7 : 0));
    const ringK = seg(t, P.burst[0] - .2, P.burst[0] + 1.2);
    for (const [ring, delay] of [[this.clsRing, 0], [this.clsRing2, .25]]) {
      const k = seg(ringK, delay, .75 + delay);
      ring.scale.setScalar(.6 + k * 3.2);
      ring.material.opacity = (1 - k) * (k > 0 ? .8 : 0);
      ring.visible = ring.material.opacity > .01;
      ring.lookAt(this.stage.camera.position);
    }

    for (const [b, a] of [[this.burst, 0], [this.burst2, .12]]) {
      b.position.copy(this.G);
      b.userData.setProgress(seg(t, P.burst[0] - .15 + a, P.burst[0] + 1.6 + a));
    }

    // fade the diagram behind the [CLS] with fog
    const fogK = easeInOut(seg(t, P.focus[0], P.focus[1]));
    const camDist = this.stage.camera.position.distanceTo(this.G);
    this.scene.fog.near = lerp(200, camDist + 1.5, fogK);
    this.scene.fog.far = lerp(400, camDist + 7.5, fogK);

    // decode burst
    const pb = seg(t, ...P.burst);
    this.panels.forEach((panel, i) => {
      const g = panel.group;
      const k = easeOut(seg(pb, i * .1, .45 + i * .1));
      const target = g.userData.target;
      const p = this.G.clone().lerp(target, k);
      p.y += Math.sin(Math.PI * k) * .7 + (k >= 1 ? Math.sin(this.idle * 1.1 + i * 1.7) * .06 : 0);
      g.position.copy(p);
      const hovered = this.hover === panel.id || this.selected === panel.id;
      const s = Math.max(.001, k) * 1.18 * (hovered && k >= 1 ? 1.045 : 1);
      g.scale.setScalar(lerp(g.scale.x, s, k >= 1 ? .2 : 1));
      g.rotation.y = g.userData.rotY * k + (k < 1 ? (1 - k) * 1.2 * (i % 2 ? -1 : 1) : 0);
      panel.trail.setProgress(easeInOut(seg(pb, i * .1, .35 + i * .1)));
      panel.trail.setOpacity(k >= 1 ? .45 : .9);
      panel.trail.tick(time);
      const inner = clamp((pb - (.25 + i * .1)) / .55);
      if (g.userData.voxels) g.userData.voxels.material.uniforms.uAssemble.value = inner * 1.15;
      if (g.userData.spin) g.userData.spin.rotation.y = -.62 + (k >= 1 ? Math.sin(this.idle * .35 + i) * .35 : 0);
      if (g.userData.relief) g.userData.relief.userData.setRise(easeOut(inner));
      if (g.userData.semantic) g.userData.semantic.userData.setExplode(lerp(3.2, .55 + Math.sin(this.idle * .9) * .15 * (k >= 1 ? 1 : 0), easeInOut(inner)));
      if (g.userData.path) g.userData.path.setProgress(easeInOut(inner));
      if (g.userData.boxOverlay) {
        g.userData.boxOverlay.material.opacity = inner;
        g.userData.boxOverlay.position.z = lerp(1.2, .3, easeOut(inner));
      }
      setGroupOpacity(g.userData.content, 1);
    });

    // camera: timeline shots, then optional focus on a single readout
    const aspect = this.stage.width / this.stage.height;
    this.rig.aspectFit = aspect < 1.25 ? Math.pow(1.6 / aspect, .85) : 1;
    const target = this.focus >= 0 ? 1 : 0;
    this.focusBlend = lerp(this.focusBlend, target, .08);
    const sway = t >= REP_END ? 1 : .4;
    this.rig.apply(t, time, sway);
    if (this.focusBlend > .002 && this.panels.length) {
      const panel = this.panels[Math.max(0, this.focus)] || this.panels[0];
      if (this.focus >= 0) this.lastFocus = panel;
      const focusPanel = this.focus >= 0 ? panel : this.lastFocus || panel;
      const p = focusPanel.group.position.clone();
      const camPos = p.clone().add(new THREE.Vector3(0, .25, 7.4 * this.rig.aspectFit));
      this.stage.camera.position.lerp(camPos, this.focusBlend);
      const look = this.rig._target.clone().lerp(p, this.focusBlend);
      this.stage.camera.lookAt(look);
    }
    this.drag?.setEnabled(t >= REP_END - .01);
  }
}
