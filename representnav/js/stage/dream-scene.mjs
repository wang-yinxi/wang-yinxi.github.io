// Section 02 · Latent chain-of-thought as a 3D stage.
// Follows the right panel of the paper's pipeline figure: instruction → Text Encoder → text tokens;
// observation → Representation Residuals → representation tokens; (latent_start); the LLM emits
// h1…h4, each fed back as the next input (dashed loops); (latent_end) → Planning Head → trajectory.
// Each h_k is supervised by the future [CLS]_{t+k} (L_pred, both directions in the figure), whose
// targets VISReg regularizes; decoded latents are post-hoc visualizations only.
import {
  THREE, C, FONT, SERIF, MONO, Stage, CameraRig, Wire, attachOrbitDrag, fontsReady,
  clamp, lerp, seg, easeOut, easeInOut, backOut,
  canvasTexture, texturedPlane, textPlane, roundRect, drawDrop, makeToken, makeSlab, makeCanvas, toTexture,
  solidMaterial, glassMaterial, glowSprite, setGroupOpacity, roundedBox, mulberry32,
  loadImage, photoCardTexture, loadVoxels, makeVoxelMesh, fitShot, makeBurst,
} from './core.mjs';

import { DREAM_STEPS, DREAM_END } from './meta.mjs';

export { DREAM_STEPS, DREAM_END };

const S = 48;
const D = (x, y, z = 0) => new THREE.Vector3((x - 750) / S, (575 - y) / S, z);
const TOKEN = .74;
const LATENT_X = [648, 745, 842, 939];
const END_X = 1036;
const SLOT_X = [745, 842, 939, END_X];
const TEXT_X = [69, 116, 162, 208];
const REP_X = [296, 342, 388, 435, 481, 528];
const IN_Y = 452, OUT_Y = 698, CLS_Y = 822, OBS_Y = 928;
const LLM = { x: 753, y: 576, w: 1447 / S, h: 158 / S, d: 2.4 };
const LLM_FRONT = LLM.d / 2;

const LOOP0 = 4.4, LOOP_DT = 1.65;
const loopT = k => LOOP0 + (k - 1) * LOOP_DT; // start of step k (1-based)

function cardFace({ title, sub, fill = '#f4f7fd', stroke = '#9fb4d8', color = C.navy, size = .34, drop = false, dashed = false, align = 'center' }) {
  return (ctx, w, h) => {
    roundRect(ctx, 5, 5, w - 10, h - 10, Math.min(30, h * .2));
    ctx.fillStyle = fill;
    ctx.fill();
    if (dashed) ctx.setLineDash([16, 11]);
    ctx.lineWidth = 6;
    ctx.strokeStyle = stroke;
    ctx.stroke();
    ctx.setLineDash([]);
    if (drop) drawDrop(ctx, 60, h / 2, h * .32);
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.font = `700 ${Math.round(h * size)}px ${FONT}`;
    const x = align === 'center' ? w / 2 + (drop ? 22 : 0) : 40;
    if (sub) {
      ctx.fillText(title, x, h * .4);
      ctx.font = `500 ${Math.round(h * size * .62)}px ${FONT}`;
      ctx.fillStyle = 'rgba(60,76,105,.8)';
      ctx.fillText(sub, x, h * .7);
    } else ctx.fillText(title, x, h / 2 + 2);
  };
}

function wrapText(ctx, text, maxWidth) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = '';
  for (const word of words) {
    const probe = line ? `${line} ${word}` : word;
    if (ctx.measureText(probe).width > maxWidth && line) { lines.push(line); line = word; } else line = probe;
  }
  if (line) lines.push(line);
  return lines;
}

function starField(count = 1400) {
  const rand = mulberry32(11);
  const positions = new Float32Array(count * 3), phase = new Float32Array(count), size = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (rand() - .5) * 120;
    positions[i * 3 + 1] = (rand() - .5) * 70;
    positions[i * 3 + 2] = -14 - rand() * 50;
    phase[i] = rand() * 6.28;
    size[i] = .6 + rand() * rand() * 3.2;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aPhase', new THREE.BufferAttribute(phase, 1));
  geometry.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  const material = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uPixel: { value: Math.min(2, window.devicePixelRatio || 1) } },
    vertexShader: `attribute float aPhase; attribute float aSize; uniform float uTime; uniform float uPixel; varying float vA;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.); gl_Position = projectionMatrix * mv;
      vA = .35 + .65 * (.5 + .5 * sin(uTime * (.6 + fract(aPhase) * 1.4) + aPhase));
      gl_PointSize = aSize * uPixel * (60. / -mv.z); }`,
    fragmentShader: `varying float vA; void main(){ vec2 c = gl_PointCoord - .5; float d = length(c);
      float a = smoothstep(.5, 0., d); gl_FragColor = vec4(vec3(.78,.86,1.), a * vA * .9); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

/** Thin additive lines inside the LLM glass: the context attending into the next latent. */
function attentionLines(fromXs, toX) {
  const positions = [];
  const colors = [];
  const top = D(0, 505).y, bottom = D(0, 648).y;
  for (const x of fromXs) {
    for (const dz of [-.35, 0, .35]) {
      const a = D(x, 0, dz), b = D(toX, 0, dz * .4);
      positions.push(a.x, top, a.z, b.x, bottom, b.z);
      colors.push(.62, .78, 1, .85, .93, 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  return new THREE.LineSegments(geometry, material);
}

export class DreamScene {
  constructor(host, { onStep, onSelectToken, onReady, emergeEl = null } = {}) {
    this.host = host;
    this.emergeEl = emergeEl;
    this.onStep = onStep;
    this.onSelectToken = onSelectToken;
    this.onReady = onReady;
    this.t = 0;
    this.idle = 0;
    this.playing = true;
    this.speed = 1;
    this.selected = 1;
    this.stepIndex = -1;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.ready = this.init();
  }

  async init() {
    await fontsReady();
    const stage = this.stage = new Stage(this.host, { dark: true, fov: 30, exposure: 1.05 });
    this.scene = stage.scene;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.stars = starField();
    this.scene.add(this.stars);
    this.wires = [];
    this.build();
    this.buildCamera();
    stage.frameCallbacks.add((dt, now) => this.frame(dt, now));
    this.drag = attachOrbitDrag(stage.canvas, this.rig, { maxYaw: .5, maxPitch: .25 });
    this.setupPicking();
    this.update(this.reduced ? DREAM_END : 0, 0);
    stage.render();
    this.onReady?.();
  }

  addWire(points, options) {
    const wire = new Wire(points, { radius: .03, corner: .2, ...options });
    wire.setProgress(0);
    this.root.add(wire);
    this.wires.push(wire);
    return wire;
  }

  build() {
    const root = this.root;
    const light = '#b9c8e4';

    // ---- instruction card (text drawn per case)
    this.instruction = texturedPlane(canvasTexture(8, 8, () => {}), 174 / S * 1.02, 178 / S);
    this.instruction.position.copy(D(139, 250, 0));
    root.add(this.instruction);

    this.textEncoder = makeSlab({ w: 174 / S, h: 48 / S, d: .55, r: .12, stroke: null, outline: false,
      material: solidMaterial('#f2f4f8', { roughness: .5 }), draw: cardFace({ title: 'Text Encoder', color: '#3a3a3a', stroke: '#8a8f98', size: .4 }), texW: 700 });
    this.textEncoder.position.copy(D(139, 386, 0));
    root.add(this.textEncoder);
    this.rrBox = makeSlab({ w: 267 / S, h: 48 / S, d: .55, r: .12, stroke: null, outline: false,
      material: solidMaterial(C.blue150, { roughness: .45 }), draw: cardFace({ title: 'Representation Residuals', fill: '#dfe9f8', stroke: C.blue500, size: .38 }), texW: 1000 });
    this.rrBox.position.copy(D(411, 386, 0));
    root.add(this.rrBox);

    // observation stack sits above the RR box
    this.obsGroup = new THREE.Group();
    this.obsGroup.position.copy(D(411, 238, .2));
    root.add(this.obsGroup);
    this.obsCards = [];

    this.wInstr = this.addWire([D(139, 340), D(139, 360)], { color: light });
    this.wObs = this.addWire([D(411, 318), D(411, 360)], { color: light });

    // ---- input row
    this.textTokens = TEXT_X.map(x => { const t = makeToken('text', TOKEN, { dark: true }); t.position.copy(D(x, IN_Y)); root.add(t); return t; });
    this.repTokens = REP_X.map(x => { const t = makeToken('rep', TOKEN, { dark: true }); t.position.copy(D(x, IN_Y)); root.add(t); return t; });
    this.start = makeToken('boundary', TOKEN, { dark: true });
    this.start.position.copy(D(648, IN_Y));
    root.add(this.start);
    this.endToken = makeToken('boundary', TOKEN, { dark: true });
    this.endToken.position.copy(D(END_X, IN_Y));
    root.add(this.endToken);
    this.slots = SLOT_X.slice(0, 3).map(x => {
      const t = makeToken('latent', TOKEN, { dark: true });
      t.position.copy(D(x, IN_Y));
      const glow = glowSprite('#8fb6ff', 1.8, { opacity: 0 });
      t.add(glow);
      t.userData.glow = glow;
      root.add(t);
      return t;
    });
    const mono = (text, x, y, opts = {}) => {
      const label = textPlane(text, { height: .3, family: MONO, weight: 500, color: '#c9d6f2', ...opts });
      label.position.copy(D(x, y, .1));
      root.add(label);
      return label;
    };
    this.startLabel = mono('(latent_start)', 610, 418);
    this.endLabel = mono('(latent_end)', 1118, IN_Y);

    // ---- LLM
    const llm = new THREE.Group();
    llm.position.copy(D(LLM.x, LLM.y, 0));
    const glass = new THREE.Mesh(roundedBox(LLM.w, LLM.h, LLM.d, .3, 6), new THREE.MeshPhysicalMaterial({
      color: '#9dbcf2', roughness: .12, metalness: 0, transparent: true, opacity: .16, clearcoat: 1, clearcoatRoughness: .1,
      emissive: '#2f5597', emissiveIntensity: .35, depthWrite: false, side: THREE.DoubleSide }));
    glass.renderOrder = 1;
    llm.add(glass);
    const faceTex = canvasTexture(3000, Math.round(3000 * LLM.h / LLM.w), (ctx, w, h) => {
      roundRect(ctx, 8, 8, w - 16, h - 16, 60);
      ctx.lineWidth = 8;
      ctx.strokeStyle = 'rgba(150,186,245,.95)';
      ctx.stroke();
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, 'rgba(120,160,235,.16)');
      g.addColorStop(1, 'rgba(60,100,190,.06)');
      ctx.fillStyle = g;
      ctx.fill();
      drawDrop(ctx, 90, 110, 70, '#e0796c');
      ctx.font = `700 ${Math.round(h * .3)}px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.shadowColor = 'rgba(140,180,255,.85)';
      ctx.shadowBlur = 26;
      ctx.fillStyle = '#eaf1ff';
      ctx.fillText('Large Language Model', w / 2, h / 2 + 6);
    });
    const face = texturedPlane(faceTex, LLM.w, LLM.h);
    face.position.z = LLM.d / 2 + .01;
    face.renderOrder = 6;
    llm.add(face);
    const backTex = canvasTexture(1500, Math.round(1500 * LLM.h / LLM.w), (ctx, w, h) => {
      roundRect(ctx, 6, 6, w - 12, h - 12, 30);
      ctx.lineWidth = 5;
      ctx.strokeStyle = 'rgba(150,186,245,.7)';
      ctx.stroke();
    });
    const back = texturedPlane(backTex, LLM.w, LLM.h, { opacity: .5 });
    back.position.z = -LLM.d / 2;
    llm.add(back);
    this.llmGlow = glowSprite('#4f7fe0', 1, { opacity: 0 });
    this.llmGlow.scale.set(34, 7, 1);
    this.llmGlow.position.z = -.6;
    llm.add(this.llmGlow);
    this.llm = llm;
    this.llmGlass = glass;
    root.add(llm);
    this.scan = new THREE.Mesh(new THREE.PlaneGeometry(.18, LLM.h * .92), new THREE.MeshBasicMaterial({
      color: '#bcd4ff', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
    this.scan.position.copy(D(0, LLM.y, LLM_FRONT + .05));
    root.add(this.scan);

    // attention line sets, one per autoregressive step (+ latent_end)
    const context = [...TEXT_X, ...REP_X, 648];
    this.attn = [1, 2, 3, 4, 5].map(k => {
      const from = [...context, ...SLOT_X.slice(0, k - 1)];
      const lines = attentionLines(from, k <= 4 ? LATENT_X[k - 1] : END_X);
      root.add(lines);
      return lines;
    });
    // context tokens sink into the LLM
    this.contextDown = [...TEXT_X, ...REP_X, 648].map(x => this.addWire([D(x, IN_Y + 18), D(x, 497)], { color: '#8fa4c8', radius: .02, arrowScale: .8 }));

    // ---- outputs
    this.outputs = LATENT_X.map((x, i) => {
      const t = makeToken('latent', TOKEN, { dark: true });
      t.position.copy(D(x, OUT_Y));
      const glow = glowSprite('#8fb6ff', 3, { opacity: 0 });
      t.add(glow);
      t.userData.glow = glow;
      t.userData.k = i + 1;
      root.add(t);
      const label = textPlane(`h${'₁₂₃₄'[i]}`, { height: .42, family: SERIF, italic: true, weight: 700, color: '#f2f6ff' });
      label.position.copy(D(x, OUT_Y, TOKEN / 2 + .03));
      label.renderOrder = 12;
      root.add(label);
      t.userData.label = label;
      const burst = makeBurst({ count: 120, color: '#9cc2ff', radius: 2.2, size: 7, additive: true, seed: 20 + i });
      burst.position.copy(t.position);
      root.add(burst);
      t.userData.burst = burst;
      return t;
    });
    this.endOut = makeToken('boundary', TOKEN, { dark: true });
    this.endOut.position.copy(D(END_X, OUT_Y));
    root.add(this.endOut);
    this.outDown = [...LATENT_X, END_X].map(x => this.addWire([D(x, 657), D(x, OUT_Y - 19)], { color: '#8fa4c8', radius: .02, arrowScale: .8 }));

    // ---- feedback loops: output k → input slot k+1, routed in front of the LLM
    this.loops = LATENT_X.map((x, i) => {
      const nx = SLOT_X[i];
      const curve = new THREE.CatmullRomCurve3([
        D(x + 19, OUT_Y, .1), D(x + 46, OUT_Y - 4, .8), D(x + 50, 640, LLM_FRONT + .55), D(x + 50, 520, LLM_FRONT + .55),
        D(x + 56, 430, .9), D(x + 80, 404, .3), D(nx - 8, 410, 0), D(nx, 430, 0),
      ], false, 'centripetal');
      const wire = new Wire(curve, { color: '#9cc2ff', radius: .03, dash: .22, speed: .8, arrow: 'end', head: 1 });
      wire.setProgress(0);
      root.add(wire);
      return wire;
    });

    // ---- future targets: observations → [CLS]_{t+k}; L_pred between h_k and [CLS]_{t+k}
    this.futureStacks = LATENT_X.map(x => {
      const group = new THREE.Group();
      group.position.copy(D(x, OBS_Y, -.2));
      root.add(group);
      return group;
    });
    this.targets = LATENT_X.map(x => {
      const t = makeToken('cls', TOKEN);
      t.position.copy(D(x, CLS_Y));
      t.traverse(n => { if (n.userData.isOutline) n.material.color.set('#9cc2ff'); });
      const glow = glowSprite('#5f8ff0', 2, { opacity: 0 });
      t.add(glow);
      t.userData.glow = glow;
      root.add(t);
      return t;
    });
    this.beams = LATENT_X.map(x => this.addWire([D(x, OBS_Y - 52), D(x, CLS_Y + 20)], { color: '#6f8fca', radius: .018, arrow: false, head: 0, flow: 1, flowGap: .8 }));
    this.lpred = LATENT_X.map(x => this.addWire([D(x, OUT_Y + 19), D(x, CLS_Y - 19)], { color: '#ef8b7c', radius: .026, dash: .13, speed: .5, arrow: 'both', head: 0 }));
    this.lpredLabel = textPlane('ℒpred', { height: .42, family: SERIF, italic: true, weight: 600, color: '#ffb2a6' });
    this.lpredLabel.position.copy(D(592, 762, .2));
    root.add(this.lpredLabel);
    this.clsLabel = textPlane('[CLS]  t+1 : t+K  · training targets', { height: .3, family: MONO, weight: 500, color: '#c9d6f2' });
    this.clsLabel.position.copy(D(793, 862, .3));
    root.add(this.clsLabel);
    this.obsLabel = textPlane('future observations  o t+1 : t+K', { height: .32, color: '#c9d6f2', weight: 500 });
    this.obsLabel.position.copy(D(793, 1012, .3));
    root.add(this.obsLabel);

    // ---- VISReg
    this.visregTex = { canvas: makeCanvas(1000, 470) };
    this.visregTex.texture = toTexture(this.visregTex.canvas);
    this.visreg = texturedPlane(this.visregTex.texture, 475 / S, 225 / S * 1.0);
    this.visreg.position.copy(D(1228, 918, 0));
    root.add(this.visreg);
    this.visregTitle = textPlane('ℒVISReg', { height: .36, family: SERIF, italic: true, weight: 600, color: '#ffb2a6' });
    this.visregTitle.position.copy(D(1228, 790, .2));
    root.add(this.visregTitle);
    this.wVisreg = this.addWire([D(958, CLS_Y), D(992, CLS_Y), D(992, 840), D(1000, 840)], { color: '#ef8b7c', radius: .024, dash: .12, speed: .5, arrowScale: .9 });

    // ---- heads
    const head = (title, x, y, w = 120) => {
      const slab = makeSlab({ w: w / S, h: 45 / S, d: .5, r: .1, stroke: null, outline: false, material: solidMaterial('#f4f7fd', { roughness: .45 }),
        draw: cardFace({ title, size: .4, stroke: C.blue500, color: C.navy }), texW: 600 });
      slab.position.copy(D(x, y));
      root.add(slab);
      return slab;
    };
    this.planning = head('Planning Head', 1206, 698);
    this.lmHead = head('LM Head', 1206, 770);
    this.ifNav = textPlane('if nav', { height: .26, italic: true, color: '#c9d6f2', weight: 500 });
    this.ifNav.position.copy(D(1118, 684, .2));
    root.add(this.ifNav);
    this.ifVqa = textPlane('if vqa', { height: .26, italic: true, color: '#c9d6f2', weight: 500 });
    this.ifVqa.position.copy(D(1118, 790, .2));
    root.add(this.ifVqa);
    this.wNav = this.addWire([D(1055, OUT_Y), D(1144, OUT_Y)], { color: light });
    this.wVqa = this.addWire([D(1100, OUT_Y), D(1100, 770), D(1144, 770)], { color: '#6f7f9f', corner: .15 });
    this.trajTex = { canvas: makeCanvas(900, 260) };
    this.trajTex.texture = toTexture(this.trajTex.canvas);
    this.traj = texturedPlane(this.trajTex.texture, 178 / S, 178 / S * 260 / 900 * 1.25);
    this.traj.position.copy(D(1374, 702, 0));
    root.add(this.traj);
    this.wTraj = this.addWire([D(1267, OUT_Y), D(1284, OUT_Y)], { color: light, radius: .026 });
    this.answer = texturedPlane(canvasTexture(900, 230, (ctx, w, h) => {
      roundRect(ctx, 5, 5, w - 10, h - 10, 26);
      ctx.fillStyle = 'rgba(244,247,253,.9)'; ctx.fill();
      ctx.lineWidth = 5; ctx.strokeStyle = '#9fb4d8'; ctx.stroke();
      ctx.fillStyle = '#56698f'; ctx.font = `500 46px ${FONT}`; ctx.fillText('Answer', 34, 64);
      ctx.fillStyle = '#2a2a2a'; ctx.font = `500 60px ${MONO}`; ctx.fillText('Toward you.', 34, 160);
      ctx.fillStyle = C.red; ctx.font = `italic 600 50px ${SERIF}`; ctx.fillText('ℒCE', w - 130, 64);
    }), 178 / S, 178 / S * 230 / 900);
    this.answer.position.copy(D(1374, 770, 0));
    root.add(this.answer);
    this.wAnswer = this.addWire([D(1267, 770), D(1284, 770)], { color: '#6f7f9f', radius: .026 });

    // ---- decoded latent representation, bottom left as in the figure (visualization only)
    this.decode = new THREE.Group();
    this.decode.position.copy(D(296, 830, 0));
    root.add(this.decode);
    const panelTex = canvasTexture(1600, 980, (ctx, w, h) => {
      roundRect(ctx, 6, 6, w - 12, h - 12, 40);
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, 'rgba(26,48,92,.8)');
      g.addColorStop(1, 'rgba(14,30,62,.8)');
      ctx.fillStyle = g; ctx.fill();
      ctx.setLineDash([22, 14]); ctx.lineWidth = 5; ctx.strokeStyle = 'rgba(156,194,255,.7)'; ctx.stroke();
    });
    this.decodeBack = texturedPlane(panelTex, 533 / S, 326 / S);
    this.decodeBack.position.z = -.3;
    this.decodeBack.renderOrder = -2;
    this.decode.add(this.decodeBack);
    const title = textPlane('Decoded Latent Representation', { height: .4, color: '#eef3ff', weight: 700 });
    title.position.set(-1.05, -3.85, 0);
    this.decode.add(title);
    const vis = textPlane('(visualization only)', { height: .28, color: '#a9bbe0', weight: 500 });
    vis.position.set(3.4, -3.87, 0);
    this.decode.add(vis);
    this.decodeTitle = [title, vis];
    this.decodeGlow = glowSprite('#4f7fe0', 1, { opacity: 0 });
    this.decodeGlow.scale.set(14, 9, 1);
    this.decodeGlow.position.z = -1;
    this.decode.add(this.decodeGlow);
    this.flash = glowSprite('#ffe9c2', 1, { opacity: 0 });
    this.flash.position.set(-1.7, .2, .6);
    this.decode.add(this.flash);
    this.worldPivot = new THREE.Group();
    this.worldPivot.position.set(-1.7, -.9, .5);
    this.decode.add(this.worldPivot);
    this.worlds = [];
    this.futureCards = [];
    this.thumbSets = [];
    this.thumbs = new THREE.Group();
    this.thumbs.position.set(3.7, .1, .2);
    this.decode.add(this.thumbs);
    this.timeChip = textPlane('h₁ · t + 0.4 s', { height: .36, family: MONO, weight: 500, color: '#ffffff', background: 'rgba(47,85,151,.92)', padX: 26, padY: 8, radius: 26 });
    this.timeChip.position.set(-3.6, 2.75, .3);
    this.decode.add(this.timeChip);
    this.chipCache = new Map();
    // "decode" arrows: each latent h_k points into the panel
    this.decodeLinks = LATENT_X.map(x => {
      const curve = new THREE.CatmullRomCurve3([D(x - 19, OUT_Y, .2), D(x - 50, OUT_Y + 6, 1.4), D(600, OUT_Y + 6, 1.5), D(570, 700, .6)], false, 'centripetal');
      const wire = new Wire(curve, { color: '#9cc2ff', radius: .024, dash: .16, speed: .7, arrow: 'end', head: 1, arrowScale: .9 });
      wire.setProgress(0);
      root.add(wire);
      return wire;
    });
    this.decodeWord = textPlane('decode', { height: .28, italic: true, color: '#c9d6f2', weight: 500 });
    this.decodeWord.position.copy(D(598, 676, 1.2));
    root.add(this.decodeWord);
    this.emergeBurst = makeBurst({ count: 300, color: '#cfe0ff', radius: 4.5, size: 8, additive: true, seed: 41 });
    this.emergeBurst.position.copy(this.decode.position).add(new THREE.Vector3(-1.7, 0, .8));
    root.add(this.emergeBurst);
  }

  buildCamera() {
    const shot = (t, target, offset, fov = 32) => ({ t, target, pos: target.map((v, i) => v + offset[i]), fov });
    this.shots = [
      shot(0, [-10.2, 5.0, 0], [-5.0, 1.9, 18.3]),
      shot(2.3, [-7.6, 3.6, 0], [-3.1, 4.0, 19.5]),
      shot(4.2, [-1.8, .8, 0], [4.2, 5.2, 23]),
      shot(loopT(2) + .2, [.8, -2.2, 0], [7.3, 3.1, 21.8]),
      shot(loopT(4) + 1.3, [3.2, -2.3, 0], [8.0, 2.6, 21.8]),
      shot(12.3, [9.6, -3.6, 0], [5.4, 2.4, 17.5]),
      shot(13.7, [9.0, -4.0, 0], [4.4, 2.4, 18.5]),
      shot(14.9, [-5.6, -4.4, 0], [-2.6, 2.6, 21]),
      shot(16.6, [-8.4, -5.0, 0], [-3.2, 2.2, 16.5]),
      shot(19.4, [-9.0, -5.2, 0], [-3.6, 2.2, 14.8]),
      shot(DREAM_END - 2.6, [-8.4, -5.0, 0], [-2.2, 2.0, 15.5]),
      shot(DREAM_END - .3, [-.2, -.6, 0], [0, 2.8, 35]),
      shot(DREAM_END, [-.2, -.6, 0], [0, 2.8, 35]),
    ];
    this.rig = new CameraRig(this.stage.camera, this.shots);
  }

  // ------------------------------------------------------------ per case

  drawInstruction(prompt, task, chars = Infinity) {
    if (!this.instructionCanvas) {
      this.instructionCanvas = makeCanvas(560, 572);
      const tex = toTexture(this.instructionCanvas);
      this.instruction.material.map?.dispose();
      this.instruction.material.map = tex;
      this.instruction.material.needsUpdate = true;
    }
    const ctx = this.instructionCanvas.getContext('2d');
    const w = 560, h = 572;
    ctx.clearRect(0, 0, w, h);
    roundRect(ctx, 5, 5, w - 10, h - 10, 26);
    ctx.fillStyle = 'rgba(244,247,253,.96)'; ctx.fill();
    ctx.setLineDash([14, 10]); ctx.lineWidth = 4; ctx.strokeStyle = '#8fa1c0'; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#6c7a93';
    ctx.font = `700 26px ${FONT}`;
    if ('letterSpacing' in ctx) ctx.letterSpacing = '3px';
    ctx.fillText('INSTRUCTION', 36, 64);
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    ctx.fillStyle = C.navy;
    ctx.font = `700 46px ${FONT}`;
    ctx.fillText(task, 36, 122);
    ctx.fillStyle = '#26324a';
    ctx.font = `500 42px ${FONT}`;
    const shown = prompt.slice(0, Math.max(0, Math.floor(chars)));
    const lines = wrapText(ctx, prompt, w - 72).slice(0, 7);
    let used = 0, caret = null;
    lines.forEach((line, i) => {
      const begin = used;
      used += line.length + 1;
      const part = shown.slice(begin, begin + line.length);
      const y = 192 + i * 56;
      ctx.fillText(part, 36, y);
      if (!caret && chars < prompt.length && shown.length <= begin + line.length) caret = [36 + ctx.measureText(part).width + 4, y];
    });
    if (caret) { ctx.fillStyle = C.blue500; ctx.fillRect(caret[0], caret[1] - 34, 4, 44); }
    this.instruction.material.map.needsUpdate = true;
  }

  async setCase(item) {
    await this.ready;
    const token = this.caseToken = Symbol('case');
    this.item = item;
    this.taskName = { VLN: 'VLN', PointGoal: 'PointGoal', Track: 'Track' }[item.task] || item.task;
    this.typed = -1;
    this.drawInstruction(item.prompt, this.taskName, 0);
    const viewsOf = frame => (frame.views?.length ? frame.views : [{ url: frame.scene }]).slice(0, 4);
    const [current, ...future] = await Promise.all(item.frames.map(frame => Promise.all(viewsOf(frame).map(v => loadImage(v.url).catch(() => null)))));
    if (token !== this.caseToken) return;
    const stack = (group, images, width, list) => {
      for (const child of [...group.children]) group.remove(child);
      list.length = 0;
      images.filter(Boolean).forEach((image, i, all) => {
        const ratio = image.naturalHeight / image.naturalWidth;
        const card = texturedPlane(photoCardTexture(image, { w: 420, borderPx: 7, radius: 20 }), width, width * ratio, { depthWrite: true });
        const depth = all.length - 1 - i;
        card.position.set(depth * -.14, depth * .12, -depth * .16);
        card.renderOrder = 4;
        group.add(card);
        list.push(card);
      });
    };
    stack(this.obsGroup, current, 2.2, this.obsCards);
    this.futureCards = future.map((images, k) => { const list = []; stack(this.futureStacks[k], images, 1.55, list); return list; });

    // dream world per future latent
    for (const w of this.worlds) if (w) this.worldPivot.remove(w);
    this.worlds = [];
    this.thumbSets = [];
    for (const child of [...this.thumbs.children]) this.thumbs.remove(child);
    const frames = item.frames.slice(1, 5);
    const decoded = await Promise.all(frames.map(async frame => {
      const view = frame.views?.find(v => v.key === item.selectedCamera) || frame.views?.[0] || {};
      const [rgb, depth, semantic] = await Promise.all([
        loadImage(view.url || frame.scene).catch(() => null),
        (view.latentDepth || frame.latentDepth) ? loadImage(view.latentDepth || frame.latentDepth).catch(() => null) : null,
        (view.latentSemantic || frame.latentSemantic) ? loadImage(view.latentSemantic || frame.latentSemantic).catch(() => null) : null,
      ]);
      const voxels = frame.latentVoxels && item.grid ? await loadVoxels(frame.latentVoxels).catch(() => null) : null;
      return { frame, view, rgb, depth, semantic, voxels, box: view.latentBox || frame.latentBox };
    }));
    if (token !== this.caseToken) return;
    decoded.forEach((d, k) => {
      let world = null;
      if (d.voxels) {
        world = makeVoxelMesh(d.voxels, item.grid, { fit: 5.4 });
        world.rotation.set(.5, -.75, 0);
        world.material.fog = false;
        world.material.transparent = false;
      } else if (d.rgb) {
        const ratio = d.rgb.naturalHeight / d.rgb.naturalWidth;
        world = new THREE.Group();
        const pw = 4.3, ph = pw * ratio;
        const photo = texturedPlane(photoCardTexture(d.rgb, { w: 640, borderPx: 8 }), pw, ph, { depthWrite: true });
        photo.position.y = 1.4;
        world.add(photo);
        if (d.box) {
          const overlay = texturedPlane(canvasTexture(640, Math.round(640 * ratio), (ctx, w, h) => {
            const [x0, y0, x1, y1] = d.box;
            ctx.setLineDash([20, 11]); ctx.lineWidth = 8; ctx.strokeStyle = C.box;
            roundRect(ctx, x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h, 8); ctx.stroke();
          }), pw, ph);
          overlay.position.set(0, 1.4, .2);
          world.add(overlay);
        }
        world.rotation.y = .25;
      }
      if (world) { world.visible = false; this.worldPivot.add(world); }
      this.worlds.push(world);
      const set = new THREE.Group();
      const cards = [['RGB (real future)', d.rgb, false], ['Depth (decoded)', d.depth, true], ['Semantics (decoded)', d.semantic, true]].filter(([, img]) => img);
      cards.forEach(([name, img, nearest], i) => {
        const tex = photoCardTexture(img, { w: 360, h: 300, borderPx: 6, radius: 18, fit: 'cover', stroke: '#9cc2ff' });
        if (nearest) { tex.magFilter = THREE.NearestFilter; }
        const card = texturedPlane(tex, 2.05, 1.71, { depthWrite: true });
        card.position.set(0, 2.05 - i * 2.15, 0);
        set.add(card);
        const label = textPlane(name, { height: .24, color: '#dbe6f7', weight: 600 });
        label.position.set(0, 2.05 - i * 2.15 - 1.03, 0);
        set.add(label);
      });
      set.visible = false;
      this.thumbs.add(set);
      this.thumbSets.push(set);
    });
    this.replay();
  }

  // ------------------------------------------------------------ interaction

  setupPicking() {
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const canvas = this.stage.canvas;
    let down = null;
    const pick = event => {
      const rect = canvas.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, this.stage.camera);
      const hit = raycaster.intersectObjects(this.outputs.map(o => o.userData.body), false)[0];
      return hit ? this.outputs.findIndex(o => o.userData.body === hit.object) + 1 : 0;
    };
    canvas.addEventListener('pointerdown', event => { down = [event.clientX, event.clientY]; });
    canvas.addEventListener('pointerup', event => {
      if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 6 || this.t < DREAM_END - .01) return;
      const k = pick(event);
      if (k) { this.setToken(k); this.onSelectToken?.(k); }
    });
    canvas.addEventListener('pointermove', event => {
      if (this.t < DREAM_END - .01) { canvas.style.cursor = ''; return; }
      canvas.style.cursor = pick(event) ? 'pointer' : '';
    });
  }

  setToken(k) {
    this.selected = clamp(Math.round(k), 1, 4);
    this.stage.start();
  }

  replay() {
    this.shownWorld = -1;
    this.assembleAt = null;
    this.t = this.reduced ? DREAM_END : 0;
    this.idle = 0;
    this.playing = !this.reduced;
    this.rig.offset.yaw = 0;
    this.rig.offset.pitch = 0;
    this.stepIndex = -1;
    this.stage.start();
  }
  pause() { this.playing = false; }
  play() { if (this.t >= DREAM_END) this.replay(); else this.playing = true; this.stage.start(); }
  seek(t) { this.t = clamp(t, 0, DREAM_END); this.stage.start(); }
  get duration() { return DREAM_END; }

  frame(dt, now) {
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t >= DREAM_END) { this.t = DREAM_END; this.playing = false; }
    }
    this.idle += dt;
    this.stars.material.uniforms.uTime.value = now;
    this.update(this.t, now);
  }

  chip(k) {
    if (!this.chipCache.has(k)) {
      const time = this.item?.timeBasis === 'video-frame' ? `frame ${this.item.frames[k]?.sourceFrameIndex ?? k}` : `t + ${(0.4 * k).toFixed(1)} s`;
      const mesh = textPlane(`h${'₁₂₃₄'[k - 1]} · ${time}`, { height: .36, family: MONO, weight: 500, color: '#ffffff', background: 'rgba(47,85,151,.92)', padX: 26, padY: 8, radius: 26 });
      this.chipCache.set(k, mesh.material.map);
    }
    return this.chipCache.get(k);
  }

  drawVisreg(time, k) {
    const { canvas, texture } = this.visregTex;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    roundRect(ctx, 5, 5, w - 10, h - 10, 30);
    ctx.fillStyle = 'rgba(244,247,253,.93)'; ctx.fill();
    ctx.lineWidth = 6; ctx.strokeStyle = '#d98579'; ctx.stroke();
    // (a) [CLS] embedding distribution with random projection directions
    const cx = 200, cy = 205, r = 120;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.lineWidth = 4; ctx.strokeStyle = '#6b7c99'; ctx.stroke();
    const rand = mulberry32(5);
    for (let i = 0; i < 18; i++) {
      const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * r * .82;
      const sx = lerp(1.6, 1, k), ox = lerp(28, 0, k);
      ctx.beginPath();
      ctx.arc(cx + ox + Math.cos(a) * d * sx * .7, cy + Math.sin(a) * d, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#dfe9f8'; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = C.blue500; ctx.stroke();
    }
    const dirs = [[-.95, -.3, '#d9534f'], [.7, -.75, '#e39b3b'], [.25, .95, '#2f9b87']];
    dirs.forEach(([dx, dy, color], i) => {
      const a = Math.atan2(dy, dx) + Math.sin(time * .6 + i) * .15;
      const ex = cx + Math.cos(a) * r * .95, ey = cy + Math.sin(a) * r * .95;
      const bx = cx - Math.cos(a) * r * .6, by = cy - Math.sin(a) * r * .6;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(ex, ey); ctx.lineWidth = 6; ctx.strokeStyle = color; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 22 * Math.cos(a - .4), ey - 22 * Math.sin(a - .4));
      ctx.lineTo(ex - 22 * Math.cos(a + .4), ey - 22 * Math.sin(a + .4));
      ctx.fillStyle = color; ctx.fill();
    });
    ctx.fillStyle = C.navy; ctx.font = `600 30px ${FONT}`; ctx.textAlign = 'center';
    ctx.fillText('(a) [CLS] embedding distribution', cx + 10, h - 30);
    // (b) match center, scale and shape along random univariate projections
    const gx = 400, gw = 460;
    const rows = [['center', '#c0504d', 90], ['scale', '#e39b3b', 205], ['shape', '#2f9b87', 320]];
    rows.forEach(([name, color, y], i) => {
      const gauss = (x, mu, s) => Math.exp(-((x - mu) ** 2) / (2 * s * s));
      ctx.beginPath();
      for (let p = 0; p <= 120; p++) {
        const x = p / 120, v = gauss(x, .5, .14);
        const px = gx + x * gw, py = y - v * 70;
        if (p === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.lineWidth = 4; ctx.strokeStyle = '#3b3b3b'; ctx.stroke();
      ctx.beginPath();
      for (let p = 0; p <= 120; p++) {
        const x = p / 120;
        let v;
        if (i === 0) v = gauss(x, lerp(.68, .5, k), .14);
        else if (i === 1) v = gauss(x, .5, lerp(.07, .14, k)) * lerp(1.15, 1, k);
        else v = lerp(.55 * gauss(x, .34, .07) + .55 * gauss(x, .66, .07), gauss(x, .5, .14), k);
        const px = gx + x * gw, py = y - v * 70;
        if (p === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.lineWidth = 6; ctx.strokeStyle = color; ctx.stroke();
      ctx.fillStyle = color; ctx.font = `600 28px ${FONT}`; ctx.textAlign = 'left';
      ctx.fillText(name, gx + gw + 14, y - 6);
    });
    ctx.fillStyle = C.navy; ctx.font = `600 28px ${FONT}`; ctx.textAlign = 'center';
    ctx.fillText('(b) match center, scale and shape', gx + gw / 2 + 30, h - 30);
    texture.needsUpdate = true;
  }

  drawTrajectory(p) {
    const { canvas, texture } = this.trajTex;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    roundRect(ctx, 5, 5, w - 10, h - 10, 26);
    ctx.fillStyle = 'rgba(232,240,252,.94)'; ctx.fill();
    ctx.lineWidth = 5; ctx.strokeStyle = C.blue500; ctx.stroke();
    ctx.fillStyle = '#56698f'; ctx.font = `500 44px ${FONT}`; ctx.textAlign = 'left';
    ctx.fillText('Trajectory', 30, 62);
    ctx.fillStyle = C.red; ctx.font = `italic 600 48px ${SERIF}`;
    ctx.fillText('ℒnav', w - 140, 62);
    const pts = Array.from({ length: 8 }, (_, i) => {
      const u = (i + 1) / 8;
      return [110 + u * 640, 200 - Math.sin(u * 2.2) * 70 + u * 12];
    });
    const n = Math.max(0, Math.min(8, p * 8));
    ctx.beginPath();
    ctx.moveTo(80, 212);
    for (let i = 0; i < Math.ceil(n); i++) ctx.lineTo(...pts[i]);
    ctx.lineWidth = 6; ctx.strokeStyle = C.navy; ctx.stroke();
    for (let i = 0; i < Math.floor(n); i++) {
      ctx.beginPath(); ctx.arc(...pts[i], 11, 0, Math.PI * 2);
      ctx.fillStyle = C.navy; ctx.fill();
    }
    texture.needsUpdate = true;
  }

  // ------------------------------------------------------------ the timeline

  update(t, now = 0) {
    const time = now;
    const step = DREAM_STEPS.reduce((acc, s, i) => (t >= s.t ? i : acc), 0);
    if (step !== this.stepIndex) { this.stepIndex = step; this.onStep?.(step, DREAM_STEPS[step]); }
    this.onProgress?.(t / DREAM_END);
    const pop = (obj, k) => { obj.scale.setScalar(Math.max(.001, backOut(clamp(k)))); obj.visible = k > 0; };

    // inputs
    const a = seg(t, .1, 1.1);
    this.instruction.material.opacity = easeOut(a);
    this.instruction.position.y = D(0, 250).y + (1 - easeOut(a)) * .8;
    this.obsCards.forEach((card, i) => {
      const k = easeOut(seg(t, .25 + i * .12, .95 + i * .12));
      card.material.opacity = k;
      card.position.x = (this.obsCards.length - 1 - i) * -.14 + (1 - k) * -2.5;
    });
    if (this.item) {
      const chars = Math.round(seg(t, .35, 1.5) * this.item.prompt.length);
      if (chars !== this.typed) { this.typed = chars; this.drawInstruction(this.item.prompt, this.taskName, chars >= this.item.prompt.length ? Infinity : chars); }
    }
    this.wInstr.setProgress(easeInOut(seg(t, 1.1, 1.6)));
    this.wObs.setProgress(easeInOut(seg(t, 1.2, 1.7)));
    const enc = seg(t, 1.5, 2.3);
    for (const box of [this.textEncoder, this.rrBox]) {
      box.scale.setScalar(Math.max(.001, backOut(enc)));
      box.userData.body.material.emissive?.set('#7fa6d8');
      box.userData.body.material.emissiveIntensity = .5 * Math.sin(Math.PI * seg(t, 1.8, 2.8));
    }
    this.textTokens.forEach((tok, i) => pop(tok, seg(t, 2.1 + i * .08, 2.5 + i * .08)));
    this.repTokens.forEach((tok, i) => pop(tok, seg(t, 2.2 + i * .08, 2.6 + i * .08)));
    pop(this.start, seg(t, 3.0, 3.4));
    this.startLabel.material.opacity = seg(t, 3.1, 3.5);

    // context into the LLM
    const into = seg(t, 3.5, 4.3);
    this.contextDown.forEach((w, i) => { w.setProgress(easeInOut(seg(into, i * .04, .5 + i * .04))); w.tick(time); });
    const llmWake = seg(t, 3.6, 4.6);
    this.llmGlow.material.opacity = .25 * easeOut(llmWake);

    // autoregressive loop
    let scanX = null;
    for (let k = 1; k <= 5; k++) {
      const t0 = k <= 4 ? loopT(k) : loopT(4) + LOOP_DT;
      const think = seg(t, t0, t0 + .7);
      const outX = k <= 4 ? LATENT_X[k - 1] : END_X;
      this.attn[k - 1].material.opacity = .85 * Math.sin(Math.PI * think);
      if (think > 0 && think < 1) scanX = outX;
      this.outDown[k - 1].setProgress(easeInOut(seg(t, t0 + .45, t0 + .75)));
      this.outDown[k - 1].tick(time);
      if (k <= 4) {
        const out = this.outputs[k - 1];
        const emerge = seg(t, t0 + .6, t0 + 1.0);
        pop(out, emerge);
        out.userData.burst.userData.setProgress(seg(t, t0 + .62, t0 + 1.7));
        out.userData.label.material.opacity = emerge;
        out.userData.label.visible = emerge > 0;
        const sel = t >= DREAM_END - .01 && this.selected === k ? 1 : 0;
        out.userData.glow.material.opacity = Math.max(1 * Math.sin(Math.PI * seg(t, t0 + .6, t0 + 1.4)), emerge >= 1 ? .42 + .16 * Math.sin(time * 2 + k) + sel * .45 : 0);
        out.userData.glow.scale.setScalar(2.8 + sel * 1.4 + 1.6 * Math.sin(Math.PI * seg(t, t0 + .6, t0 + 1.4)));
        // targets and L_pred appear with the latent
        const fk = seg(t, t0 + .7, t0 + 1.3);
        (this.futureCards?.[k - 1] || []).forEach((card, i) => { card.material.opacity = easeOut(seg(fk, i * .1, .6 + i * .1)); });
        this.beams[k - 1].setProgress(easeInOut(seg(t, t0 + 1.0, t0 + 1.3)));
        this.beams[k - 1].tick(time);
        const target = this.targets[k - 1];
        pop(target, seg(t, t0 + 1.1, t0 + 1.4));
        target.userData.glow.material.opacity = .7 * Math.sin(Math.PI * seg(t, t0 + 1.1, t0 + 1.9)) + (seg(t, t0 + 1.4, t0 + 1.5) >= 1 ? .15 : 0);
        this.lpred[k - 1].setProgress(easeInOut(seg(t, t0 + 1.2, t0 + 1.55)));
        this.lpred[k - 1].tick(time);
        // the latent is fed back as the next input
        const loop = this.loops[k - 1];
        loop.setProgress(easeInOut(seg(t, t0 + .95, t0 + 1.6)));
        loop.tick(time);
        if (k <= 3) {
          const slot = this.slots[k - 1];
          pop(slot, seg(t, t0 + 1.5, t0 + 1.75));
          slot.userData.glow.material.opacity = .8 * Math.sin(Math.PI * seg(t, t0 + 1.5, t0 + 2.3));
        } else {
          pop(this.endToken, seg(t, t0 + 1.5, t0 + 1.75));
          this.endLabel.material.opacity = seg(t, t0 + 1.55, t0 + 1.9);
        }
      } else {
        pop(this.endOut, seg(t, t0 + .6, t0 + 1.0));
      }
    }
    this.lpredLabel.material.opacity = seg(t, loopT(1) + 1.2, loopT(1) + 1.6);
    this.clsLabel.material.opacity = seg(t, loopT(1) + 1.2, loopT(1) + 1.6);
    this.obsLabel.material.opacity = seg(t, loopT(1) + .8, loopT(1) + 1.3);
    if (scanX !== null) {
      this.scan.visible = true;
      const k = scanX === END_X ? 5 : LATENT_X.indexOf(scanX) + 1;
      const t0 = k <= 4 ? loopT(k) : loopT(4) + LOOP_DT;
      const u = seg(t, t0, t0 + .7);
      this.scan.position.x = lerp(D(40, 0).x, D(scanX, 0).x, easeInOut(u));
      this.scan.material.opacity = .7 * Math.sin(Math.PI * u);
      this.scan.scale.x = 1 + 6 * Math.sin(Math.PI * u);
    } else this.scan.visible = false;
    const pulse = Math.max(0, ...this.attn.map(l => l.material.opacity)) / .85;
    this.llmGlass.material.emissiveIntensity = .3 + .45 * pulse;
    this.llmGlow.material.opacity = .25 * easeOut(llmWake) + .3 * pulse;

    // act: planning head
    const tAct = 11.6;
    this.wNav.setProgress(easeInOut(seg(t, tAct, tAct + .4)));
    this.wVqa.setProgress(easeInOut(seg(t, tAct + .1, tAct + .5)));
    this.ifNav.material.opacity = seg(t, tAct, tAct + .4);
    pop(this.planning, seg(t, tAct + .25, tAct + .6));
    pop(this.lmHead, seg(t, tAct + .3, tAct + .65));
    this.wTraj.setProgress(easeInOut(seg(t, tAct + .6, tAct + .85)));
    this.wAnswer.setProgress(easeInOut(seg(t, tAct + .65, tAct + .9)));
    const trajK = seg(t, tAct + .7, tAct + 1.8);
    this.traj.material.opacity = seg(t, tAct + .6, tAct + .9);
    if (this.lastTraj !== Math.round(trajK * 40)) { this.lastTraj = Math.round(trajK * 40); this.drawTrajectory(trajK); }

    // VISReg on the targets (training only)
    const tReg = 12.5;
    this.wVisreg.setProgress(easeInOut(seg(t, tReg, tReg + .4)));
    this.wVisreg.tick(time);
    const regK = easeInOut(seg(t, tReg + .5, tReg + 1.8));

    // training-only parts step back before the dream emerges
    const tBack = DREAM_END - 2.6;
    const f = 1 - .86 * easeInOut(seg(t, 13.7, 14.5)) * (1 - easeInOut(seg(t, tBack, tBack + 1.4)));
    this.lpred.forEach(w => w.setOpacity(f));
    this.beams.forEach(w => w.setOpacity(f));
    this.targets.forEach(target => {
      setGroupOpacity(target.userData.body, f);
      target.userData.glow.material.opacity *= f;
    });
    (this.futureCards || []).forEach(list => list.forEach(card => { card.material.opacity *= f; }));
    this.lpredLabel.material.opacity *= f;
    this.clsLabel.material.opacity *= f;
    this.obsLabel.material.opacity *= f;
    this.visreg.material.opacity = seg(t, tReg + .2, tReg + .6) * f;
    this.visregTitle.material.opacity = seg(t, tReg + .2, tReg + .6) * f;
    this.wVisreg.setOpacity(f);
    if (this.visreg.material.opacity > .01 && f > .5) this.drawVisreg(time, regK);
    setGroupOpacity(this.lmHead, .35 + .65 * f);
    this.answer.material.opacity = .65 * seg(t, tAct + .65, tAct + .95) * f;
    this.ifVqa.material.opacity = .6 * seg(t, tAct + .1, tAct + .5) * f;
    this.wVqa.setOpacity(.4 + .6 * f);
    this.wAnswer.setOpacity(.4 + .6 * f);

    // the dream emerges in the decoded-latent panel (bottom left, as in the figure)
    const tE = 14.2;
    const final = t >= DREAM_END - .01;
    const panelIn = easeOut(seg(t, tE - .2, tE + .7));
    this.decodeBack.material.opacity = panelIn;
    for (const label of this.decodeTitle) label.material.opacity = panelIn;
    this.decodeGlow.material.opacity = .25 * panelIn;
    this.decodeWord.material.opacity = seg(t, tE + .4, tE + .9);
    // the four decode arrows reach the panel, then the word "Emerge" appears and dissolves
    const textIn = easeOut(seg(t, tE + 1.1, tE + 1.8)), textOut = easeInOut(seg(t, tE + 2.6, tE + 3.2));
    if (this.emergeEl) {
      const op = textIn * (1 - textOut);
      const el = this.emergeEl;
      el.style.opacity = op.toFixed(3);
      el.style.visibility = op > .002 ? 'visible' : 'hidden';
      el.style.setProperty('--e-scale', (.84 + .16 * textIn + .1 * textOut).toFixed(4));
      el.style.setProperty('--e-blur', `${((1 - textIn) * 16 + textOut * 20).toFixed(2)}px`);
      el.style.setProperty('--e-track', `${((1 - textIn) * .22 + textOut * .12).toFixed(3)}em`);
      el.style.setProperty('--e-rise', `${(-textOut * 30).toFixed(1)}px`);
    }
    const gather = seg(t, tE + .9, tE + 2.7), release = seg(t, tE + 2.7, tE + 3.4);
    this.flash.material.opacity = .6 * Math.sin(Math.PI * .5 * gather) * (1 - release);
    this.flash.scale.setScalar(1 + 2.4 * gather + 4 * release);
    this.emergeBurst.userData.setProgress(seg(t, tE + 2.7, tE + 4.2));
    const W = 1.45;
    const startOf = k => tE + 3.1 + (k - 1) * W;
    let shown = 0;
    for (let k = 1; k <= 4; k++) if (t >= startOf(k) - .05) shown = k;
    if (final) shown = this.selected;
    if (shown !== this.shownWorld) {
      this.shownWorld = shown;
      if (!final && shown) { this.selected = shown; this.onSelectToken?.(shown); }
      this.assembleAt = final ? this.idle : null;
      const world = this.worlds[shown - 1];
      if (world?.geometry?.attributes?.aFrom) {
        // voxels fly in from where the decode arrow of h_k enters the panel
        this.decode.updateMatrixWorld(true);
        const source = world.worldToLocal(this.decodeLinks[shown - 1].curve.getPointAt(1).clone());
        const from = world.geometry.attributes.aFrom;
        for (let i = 0; i < from.count; i++) {
          const a = (i * 2.399963) % (Math.PI * 2), r = .1 + ((i * .618034) % 1) * .45;
          from.setXYZ(i, source.x + Math.cos(a) * r, source.y + ((i * .3819) % 1 - .5) * .4, source.z + Math.sin(a) * r);
        }
        from.needsUpdate = true;
      }
    }
    this.worlds.forEach((world, i) => {
      if (!world) return;
      const k = i + 1;
      world.visible = shown === k;
      // the previous moment dissolves while the next one assembles
      const handoff = !final && k < 4 ? (t - startOf(k + 1)) / .5 : -1;
      if (!world.visible && handoff >= 0 && handoff < 1 && world.material?.uniforms) {
        world.visible = true;
        world.material.uniforms.uAssemble.value = 1.25 * (1 - easeInOut(handoff));
        return;
      }
      if (world.visible && world.material?.uniforms) {
        const progress = this.assembleAt != null ? (this.idle - this.assembleAt) / 1.0 : (t - startOf(k)) / .95;
        world.material.uniforms.uAssemble.value = clamp(progress) * 1.25;
      } else if (world.visible) {
        const k2 = easeOut(clamp(this.assembleAt != null ? this.idle - this.assembleAt : (t - startOf(k)) / .6));
        world.scale.setScalar(Math.max(.001, k2));
      }
    });
    this.worldPivot.rotation.y = .16 * Math.sin(this.idle * .35);
    this.thumbSets.forEach((set, i) => {
      set.visible = shown === i + 1;
      if (set.visible) {
        const k2 = easeOut(clamp(this.assembleAt != null ? (this.idle - this.assembleAt) / .5 : (t - startOf(i + 1) - .25) / .5));
        set.scale.setScalar(Math.max(.001, k2));
      }
    });
    if (shown) {
      const map = this.chip(shown);
      if (this.timeChip.material.map !== map) { this.timeChip.material.map = map; this.timeChip.material.needsUpdate = true; }
    }
    this.timeChip.visible = shown > 0;
    this.decodeLinks.forEach((link, i) => {
      const k = i + 1;
      link.setProgress(easeInOut(seg(t, tE + .1 + i * .12, tE + .9 + i * .12)));
      link.setOpacity(final ? (this.selected === k ? 1 : .2) : (t < startOf(1) - .1 ? 1 : shown === k ? 1 : .3));
      link.tick(time);
      const own = Math.sin(Math.PI * seg(t, startOf(k) - .5, startOf(k) + .6));
      const out = this.outputs[i];
      if (t > tE) out.userData.glow.material.opacity = Math.max(out.userData.glow.material.opacity, own);
    });

    // camera
    const aspect = this.stage.width / this.stage.height;
    this.rig.aspectFit = aspect < 1.3 ? Math.pow(1.7 / aspect, .9) : 1;
    this.rig.apply(t, time, t >= DREAM_END ? 1 : .5);
    this.drag?.setEnabled(t >= DREAM_END - .01);
  }
}
