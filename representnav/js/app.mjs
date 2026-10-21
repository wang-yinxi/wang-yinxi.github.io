import { OccupancyViewer } from './occupancy.mjs';
import { ReadoutPanel } from './readouts.mjs';
import { MODALITIES, createState, reduceState, validateManifest } from './state.mjs';
import { availableModalities, decodedFrame, stateTime, historyWindow } from './case-readouts.mjs';
import { mountSparkles } from './sparkles.mjs';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const sections = ['representation', 'dream'];
let manifest, state, activeWorkspace = null;
const caseIds = [];
const workspaces = new Map();
const stages = new Map();
window.__stages = stages; // debugging aid: window.__stages.get('dream').scene.seek(t)

function announce(message) {
  const status = $('#page-status');
  status.textContent = message;
  status.classList.add('visible');
  clearTimeout(announce.timer);
  announce.timer = setTimeout(() => status.classList.remove('visible'), 3600);
}

function currentCase() {
  const item = manifest.cases.find(entry => entry.id === state.caseId);
  return { ...item, palette: manifest.palette };
}

// ------------------------------------------------------------------ explorer (readouts)

function dispatch(action, { fromScene = false } = {}) {
  const next = reduceState(state, action, caseIds);
  if (next === state) return;
  const oldCase = state.caseId;
  state = next;
  if (oldCase !== state.caseId) {
    for (const workspace of workspaces.values()) {
      workspace.occSignature = '';
      workspace.occRequestId = (workspace.occRequestId || 0) + 1;
      workspace.occFailed = false;
      workspace.view = 'orbit';
    }
    updateCaseTabs();
    for (const stage of stages.values()) stage.scene?.setCase(currentCase());
  }
  if (action.type === 'modality') {
    renderWorkspace(action.section);
    if (action.section === 'representation' && !fromScene) stages.get('representation')?.scene?.selectPanel(action.id);
  } else if (action.type === 'token') {
    renderWorkspace('dream');
    if (!fromScene) stages.get('dream')?.scene?.setToken(state.dream.frame);
  } else sections.forEach(renderWorkspace);
}

function updateCaseTabs() {
  $$('.case-tabs button').forEach(button => {
    const selected = button.dataset.case === state.caseId;
    button.setAttribute('aria-pressed', selected);
  });
  const item = currentCase();
  const views = item.frames[0].views?.length ? item.frames[0].views : item.views || [];
  for (const workspace of workspaces.values()) {
    workspace.camera = item.selectedCamera || views[0]?.key;
    workspace.cameraSelect.replaceChildren();
    for (const view of views) {
      const option = document.createElement('option');
      option.value = view.key;
      option.textContent = view.label || `${view.key?.[0]?.toUpperCase() || ''}${view.key?.slice(1) || 'View'}`;
      workspace.cameraSelect.append(option);
    }
    workspace.cameraSelect.value = workspace.camera;
  }
}

function showFallback(section, error) {
  const workspace = workspaces.get(section);
  workspace.occFailed = true;
  workspace.viewer?.setActive(false);
  workspace.canvas.hidden = true;
  workspace.fallback.hidden = false;
  workspace.loading.hidden = true;
  $('.occ-hint', workspace.element).textContent = 'Static preview · change views below';
  updateFallback(section);
  if (error) console.warn('Occupancy fallback:', error.message);
}

function updateFallback(section) {
  const workspace = workspaces.get(section);
  const frame = decodedFrame(currentCase().frames[state[section].frame], section);
  const source = frame.fallback?.[workspace.view] ?? frame.fallback?.orbit;
  if (source) {
    workspace.fallback.hidden = false;
    workspace.fallback.src = source;
    workspace.fallback.onerror = () => {
      workspace.fallback.hidden = true;
      workspace.loading.hidden = false;
      workspace.loading.textContent = 'This occupancy preview is unavailable. Choose another case or reload to retry.';
    };
  } else {
    workspace.loading.hidden = false;
    workspace.loading.textContent = 'Occupancy data is unavailable for this state.';
  }
}

async function renderOccupancy(section) {
  const workspace = workspaces.get(section);
  if (!workspace.visible || state[section].modality !== 'occ') return;
  if (activeWorkspace !== workspace) {
    activeWorkspace?.viewer?.setActive(false);
    activeWorkspace = workspace;
  }
  if (!workspace.viewer && !workspace.occFailed) {
    try {
      workspace.viewer = new OccupancyViewer(workspace.canvas, {
        onError: error => {
          workspace.occError = error;
          if (workspace.viewer?.getState().renderer !== 'webgl2') showFallback(section, error);
        },
        onViewChange: view => updateViewButtons(section, typeof view === 'string' ? view : view.view),
      });
      workspace.canvas.addEventListener('webglcontextrestored', () => {
        workspace.occFailed = false;
        workspace.occSignature = '';
        renderOccupancy(section).catch(error => showFallback(section, error));
      });
      if (workspace.viewer.getState().renderer === 'unavailable') showFallback(section);
    } catch (error) { showFallback(section, error); }
  }
  if (workspace.occFailed) { updateFallback(section); return; }
  workspace.viewer?.setActive(true);
  const signature = `${state.caseId}/${state[section].frame}`;
  if (workspace.occSignature === signature) return;
  const requestId = workspace.occRequestId = (workspace.occRequestId || 0) + 1;
  workspace.occSignature = signature;
  workspace.occError = null;
  workspace.loading.hidden = false;
  workspace.loading.textContent = 'Loading the scene…';
  const item = currentCase();
  const success = await workspace.viewer.setFrame(item, decodedFrame(item.frames[state[section].frame], section));
  if (workspace.occRequestId !== requestId || workspace.occSignature !== signature ||
      signature !== `${state.caseId}/${state[section].frame}`) return;
  workspace.loading.hidden = true;
  if (!success || workspace.viewer.getState().renderer !== 'webgl2') {
    showFallback(section, workspace.occError);
    return;
  }
  workspace.occFailed = false;
  workspace.canvas.hidden = false;
  workspace.fallback.hidden = true;
  updateViewButtons(section, workspace.view);
}

function renderWorkspace(section) {
  const workspace = workspaces.get(section);
  const viewState = state[section];
  const item = currentCase();
  const supported = availableModalities(item);
  if (!supported.includes(viewState.modality)) viewState.modality = supported[0];
  const sourceFrame = decodedFrame(item.frames[viewState.frame], section);
  const selectedView = sourceFrame.views?.find(view => view.key === workspace.camera);
  const frame = ['semantic', 'depth', 'box'].includes(viewState.modality) && selectedView
    ? { ...sourceFrame, scene: selectedView.url, semantic: selectedView.semantic || sourceFrame.semantic,
        depth: selectedView.depth || sourceFrame.depth, box: selectedView.box ?? null,
        boxEmbedded: selectedView.boxEmbedded ?? false }
    : sourceFrame;
  const modality = MODALITIES.find(m => m.id === viewState.modality);
  workspace.element.dataset.case = item.id;
  workspace.element.dataset.frame = viewState.frame;
  $('.readout-title', workspace.element).textContent = modality.name;
  $('.readout-dim', workspace.element).textContent = modality.id === 'occ'
    ? `Four-view volume · ${item.grid.max.map((n, i) => n - item.grid.min[i]).join(' × ')} m`
    : modality.dimension;
  workspace.cameraSelect.hidden = !['semantic', 'depth', 'box'].includes(modality.id) || sourceFrame.views?.length !== 4;
  for (const option of workspace.cameraSelect.options) {
    const view = sourceFrame.views?.find(entry => entry.key === option.value);
    option.disabled = modality.id !== 'box' && !view?.[modality.id];
  }
  $('.height-legend span:last-child', workspace.element).textContent = `${item.grid?.max?.[2] ?? 2} m`;
  $('.height-legend', workspace.element).hidden = modality.id !== 'occ' || workspace.view === 'top';
  $('.time-tag', workspace.element).textContent = section === 'representation' ? 'Current [CLS] · t = 0'
    : `h${['₀', '₁', '₂', '₃', '₄'][viewState.frame]} · ${stateTime(item, frame)}`;
  if (modality.id === 'ego') $('.readout-dim', workspace.element).textContent = `Past ego motion · ${historyWindow(frame, item.timeBasis)}`;
  const referenceTime = Number.isFinite(frame.referenceTime) ? frame.referenceTime : frame.time;
  const context = $('.frame-context', workspace.element);
  if (frame.temporalAvailable === false) {
    context.textContent = `No aligned RGB at +${frame.time.toFixed(1)} s · last available reference: +${referenceTime.toFixed(1)} s`;
  } else {
    context.textContent = Math.abs(referenceTime - frame.time) > .001
      ? `RGB reference +${referenceTime.toFixed(2)} s · nearest available source frame`
      : 'All readouts share the selected scene and time.';
  }
  $$('.modality-tabs button', workspace.element).forEach(button => {
    button.hidden = !supported.includes(button.dataset.modality);
    const selected = button.dataset.modality === modality.id;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', selected);
  });
  const isOccupancy = modality.id === 'occ';
  workspace.occContainer.hidden = !isOccupancy;
  workspace.panelContainer.hidden = isOccupancy;
  $('.view-buttons', workspace.element).hidden = !isOccupancy;
  if (isOccupancy) {
    workspace.panel.clear();
    renderOccupancy(section).catch(error => showFallback(section, error));
  } else {
    workspace.viewer?.setActive(false);
    if ((modality.id === 'ego' && frame.egoMotionAvailable === false) ||
        (modality.id === 'target' && item.hasTarget && frame.targetMotionAvailable === false)) {
      workspace.panel.clear();
      const empty = document.createElement('div');
      empty.className = 'readout-empty';
      const title = document.createElement('h4');
      title.textContent = 'Motion reference unavailable';
      const detail = document.createElement('p');
      const lastTime = frame.motionTimes?.at(-1);
      detail.textContent = Number.isFinite(lastTime)
        ? `No motion sample is available at +${frame.time.toFixed(1)} s. The source trajectory ends at +${lastTime.toFixed(1)} s.`
        : 'This RGB sequence does not include a recoverable trajectory. Explore the spatial readouts, or select another case.';
      empty.append(title, detail);
      workspace.panelContainer.append(empty);
    } else workspace.panel.render(item, frame, modality.id);
  }
  if (section === 'dream') {
    const ends = $$('.future-timeline > span', workspace.element);
    if (ends[0]) ends[0].textContent = stateTime(item, item.frames[1]);
    if (ends[1]) ends[1].textContent = stateTime(item, item.frames[4]);
    $$('.latent-token', workspace.element).forEach(button => {
      const selected = Number(button.dataset.token) === viewState.frame;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', selected);
      const label = button.querySelector('small');
      if (label) label.textContent = stateTime(item, item.frames[Number(button.dataset.token)]);
      button.setAttribute('aria-label', `Open latent h${button.dataset.token}, ${stateTime(item, item.frames[Number(button.dataset.token)])}`);
    });
    $('.future-timeline input', workspace.element).value = viewState.frame;
  }
}

function updateViewButtons(section, view) {
  const workspace = workspaces.get(section);
  workspace.view = view;
  $('.height-legend', workspace.element).hidden = view === 'top' || state[section].modality !== 'occ';
  $('.occ-hint', workspace.element).textContent = view === 'top'
    ? 'Bird’s-eye occupancy · dark obstacles / pale yellow free space'
    : '↔ Drag to rotate · Scroll to zoom';
  $$('[data-view]', workspace.element).forEach(button => {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active);
    if (button.dataset.view !== 'reset') button.setAttribute('aria-pressed', active);
  });
}

let futureTimer = 0;
function setFuturePlaying(playing) {
  clearInterval(futureTimer);
  futureTimer = 0;
  const button = $('.strip-play');
  if (playing) {
    if (state.dream.frame === 4) dispatch({ type: 'token', frame: 1 });
    futureTimer = setInterval(() => {
      if (state.dream.frame >= 4) { setFuturePlaying(false); return; }
      dispatch({ type: 'token', frame: state.dream.frame + 1 });
    }, 1300);
  }
  if (button) button.innerHTML = playing ? 'Ⅱ <span>Pause</span>' : '▶ <span>Play future</span>';
}

function setupWorkspace(section) {
  const element = $(`[data-workspace="${section}"]`);
  const workspace = {
    element, canvas: $('.occ-canvas', element), fallback: $('.occ-fallback', element),
    loading: $('.occ-loading', element), occContainer: $('.occ-container', element),
    panelContainer: $('.modality-container', element), view: 'orbit', visible: false,
  };
  workspace.cameraSelect = document.createElement('select');
  workspace.cameraSelect.className = 'camera-select';
  workspace.cameraSelect.setAttribute('aria-label', 'Camera view to decode');
  $('.readout-shell .readout-heading', element).append(workspace.cameraSelect);
  workspace.cameraSelect.addEventListener('change', event => { workspace.camera = event.target.value; renderWorkspace(section); });
  workspace.panel = new ReadoutPanel(workspace.panelContainer, { onError: () => announce('This readout could not be loaded. Select another readout or reload to retry.') });
  workspaces.set(section, workspace);
  const tabs = $('.modality-tabs', element);
  for (const modality of MODALITIES) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.modality = modality.id;
    button.setAttribute('aria-label', modality.name);
    button.setAttribute('aria-pressed', modality.id === 'occ');
    button.innerHTML = `<span class="modality-icon">${modality.icon}</span><span><strong>${modality.name}</strong></span>`;
    tabs.append(button);
  }
  element.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button || button.closest('.stage')) return;
    if (button.dataset.modality) dispatch({ type: 'modality', section, id: button.dataset.modality });
    if (button.dataset.token) { setFuturePlaying(false); dispatch({ type: 'token', frame: Number(button.dataset.token) }); }
    if (button.dataset.view) {
      const view = button.dataset.view === 'reset' ? 'orbit' : button.dataset.view;
      updateViewButtons(section, view);
      if (button.dataset.view === 'reset') workspace.viewer?.reset();
      else workspace.viewer?.setView(view);
      if (workspace.occFailed) updateFallback(section);
    }
    if (button.dataset.action === 'previous') { setFuturePlaying(false); dispatch({ type: 'token', frame: state.dream.frame - 1 }); }
    if (button.dataset.action === 'next') { setFuturePlaying(false); dispatch({ type: 'token', frame: state.dream.frame + 1 }); }
    if (button.dataset.action === 'play') setFuturePlaying(!futureTimer);
  });
  $('.future-timeline input', element)?.addEventListener('input', event => { setFuturePlaying(false); dispatch({ type: 'token', frame: Number(event.target.value) }); });
  const explorer = $('.explorer', element);
  new IntersectionObserver(entries => {
    for (const entry of entries) {
      workspace.visible = entry.isIntersecting;
      if (workspace.visible) renderOccupancy(section).catch(error => showFallback(section, error));
      else workspace.viewer?.setActive(false);
    }
  }, { threshold: 0.05 }).observe(explorer);
}

// ------------------------------------------------------------------ 3D stages

function setupStageUI(section, steps, duration) {
  const element = $(`[data-stage="${section}"]`);
  const caption = $('.stage-caption', element);
  const chapters = $('.timeline-chapters', element);
  const fill = $('.timeline-fill', element);
  const toggle = $('[data-action="toggle"]', element);
  const ui = { element, caption, fill, toggle, buttons: [] };
  steps.forEach((step, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.style.left = `${(step.t / duration) * 100}%`;
    button.setAttribute('aria-label', `Jump to ${step.label}`);
    button.innerHTML = `<span>${step.label}</span>`;
    button.addEventListener('click', () => {
      const scene = stages.get(section)?.scene;
      if (!scene) return;
      scene.seek(step.t + .01);
      scene.play();
      setPaused(ui, false);
    });
    chapters.append(button);
    ui.buttons.push(button);
  });
  ui.setStep = (index, step) => {
    caption.classList.add('is-changing');
    clearTimeout(ui.captionTimer);
    ui.captionTimer = setTimeout(() => {
      $('.step-index', caption).textContent = String(index + 1).padStart(2, '0');
      $('.step-label', caption).textContent = step.label;
      $('.step-text', caption).textContent = step.text;
      caption.classList.remove('is-changing');
    }, 180);
    ui.buttons.forEach((button, i) => {
      button.classList.toggle('is-done', i < index);
      button.classList.toggle('is-current', i === index);
    });
  };
  ui.setProgress = p => {
    fill.style.transform = `scaleX(${Math.max(0, Math.min(1, p))})`;
    element.classList.toggle('is-final', p >= .999);
    if (p >= .999 && !toggle.classList.contains('is-paused')) setPaused(ui, true);
  };
  toggle.addEventListener('click', () => {
    const scene = stages.get(section)?.scene;
    if (!scene) return;
    const paused = toggle.classList.contains('is-paused');
    if (paused) { scene.play(); setPaused(ui, false); } else { scene.pause(); setPaused(ui, true); }
  });
  $('[data-action="replay"]', element).addEventListener('click', () => {
    const scene = stages.get(section)?.scene;
    if (!scene) return;
    scene.replay();
    setPaused(ui, false);
  });
  $$('.case-tabs button', element).forEach(button => button.addEventListener('click', () => dispatch({ type: 'case', id: button.dataset.case })));
  return ui;
}

function setPaused(ui, paused) {
  ui.toggle.classList.toggle('is-paused', paused);
  ui.toggle.setAttribute('aria-label', paused ? 'Play animation' : 'Pause animation');
}

async function mountStage(section) {
  const record = stages.get(section);
  if (record.mounting) return;
  record.mounting = true;
  const host = $('.stage-host', record.ui.element);
  try {
    let scene;
    if (section === 'representation') {
      const { RepresentScene } = await import('./stage/represent-scene.mjs');
      scene = new RepresentScene(host, {
        onStep: (i, step) => record.ui.setStep(i, step),
        onSelectModality: id => {
          dispatch({ type: 'modality', section: 'representation', id }, { fromScene: true });
          announce(`Opened “${MODALITIES.find(m => m.id === id)?.name}” in the explorer below.`);
        },
      });
    } else {
      const { DreamScene } = await import('./stage/dream-scene.mjs');
      scene = new DreamScene(host, {
        emergeEl: $('.stage-emerge', record.ui.element),
        onStep: (i, step) => record.ui.setStep(i, step),
        onSelectToken: k => { setFuturePlaying(false); dispatch({ type: 'token', frame: k }, { fromScene: true }); },
      });
    }
    record.scene = scene;
    await scene.ready;
    let pending = 0;
    scene.onProgress = p => { pending = p; };
    scene.stage.frameCallbacks.add(() => record.ui.setProgress(pending));
    await scene.setCase(currentCase());
    if (section === 'dream') scene.setToken(state.dream.frame);
    if (reducedMotion.matches) setPaused(record.ui, true);
    // ?freeze=representation:6.3,dream:9 holds a frame (used for visual review).
    const freeze = new URLSearchParams(location.search).get('freeze');
    const hold = freeze?.split(',').map(part => part.split(':')).find(([name]) => name === section);
    if (hold) { scene.pause(); scene.seek(Number(hold[1]) || 0); setPaused(record.ui, true); }
  } catch (error) {
    console.error('3D stage unavailable:', error);
    $('.stage-fallback', record.ui.element).hidden = false;
    $('.stage-ui', record.ui.element).hidden = true;
  }
}

function setupStages() {
  const { REP_STEPS, REP_END, DREAM_STEPS, DREAM_END } = window.__stageMeta;
  stages.set('representation', { ui: setupStageUI('representation', REP_STEPS, REP_END) });
  stages.set('dream', { ui: setupStageUI('dream', DREAM_STEPS, DREAM_END) });
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const section = entry.target.dataset.stage;
      mountStage(section);
      observer.unobserve(entry.target);
    }
  }, { rootMargin: '500px 0px' });
  for (const record of stages.values()) observer.observe(record.ui.element);
}

// ------------------------------------------------------------------ page chrome

function setupTopbar() {
  const bar = $('[data-topbar]');
  const dream = $('#dream');
  const links = $$('.topbar nav a');
  const onScroll = () => {
    bar.classList.toggle('is-scrolled', scrollY > 8);
    const rect = dream.getBoundingClientRect();
    bar.classList.toggle('is-dark', rect.top < 64 && rect.bottom > 64);
  };
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  const targets = links.map(link => $(link.getAttribute('href'))).filter(Boolean);
  const spy = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach(link => link.classList.toggle('is-active', link.getAttribute('href') === `#${entry.target.id}`));
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  targets.forEach(target => spy.observe(target));
}

function setupReveal() {
  if (reducedMotion.matches) return;
  const items = $$('.teaser, .abstract, .overview-cards .idea, .pipeline-figure, .section-head, .video-card, .citation-box');
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.add('is-in'); observer.unobserve(entry.target); }
  }, { rootMargin: '0px 0px -8% 0px' });
  items.forEach(item => { item.classList.add('reveal'); observer.observe(item); });
}

function setupVideos() {
  const load = video => {
    const source = $('source', video);
    if (source.dataset.src) {
      source.src = source.dataset.src;
      delete source.dataset.src;
      video.load();
    }
  };
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) { load(entry.target); observer.unobserve(entry.target); }
    }
  }, { rootMargin: '350px' });
  const videos = $$('video');
  videos.forEach(video => {
    observer.observe(video);
    const select = $('select', video.closest('.video-card'));
    video.addEventListener('loadedmetadata', () => { video.playbackRate = Number(select.value); });
    select.addEventListener('change', () => { video.playbackRate = Number(select.value); });
    video.addEventListener('play', () => videos.forEach(other => { if (other !== video) other.pause(); }));
    video.addEventListener('error', () => {
      const media = video.closest('.video-media');
      if ($('.video-error', media)) return;
      const message = document.createElement('p');
      message.className = 'video-error';
      message.textContent = 'The video could not be loaded. ';
      const link = document.createElement('a');
      link.href = $('source', video).src || $('source', video).dataset.src;
      link.textContent = 'Open the video file ↗';
      message.append(link);
      media.append(message);
    });
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) videos.forEach(video => video.pause()); });
}

function setupPaper() {
  const dialog = $('#figure-dialog');
  let opener;
  $$('[data-figure]').forEach(button => button.addEventListener('click', () => {
    opener = button;
    $('#expanded-figure').src = button.dataset.figure;
    $('#expanded-figure').alt = $('img', button).alt;
    $('#figure-caption').textContent = button.dataset.caption;
    $('#figure-original').href = button.dataset.figure;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  }));
  $('#close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { document.body.classList.remove('dialog-open'); opener?.focus(); });
  $('#copy-citation').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('#citation-text').textContent);
      $('#copy-citation').textContent = 'Copied ✓';
      setTimeout(() => { $('#copy-citation').textContent = 'Copy citation'; }, 2500);
    } catch {
      const selection = getSelection();
      const range = document.createRange();
      range.selectNodeContents($('#citation-text'));
      selection.removeAllRanges();
      selection.addRange(range);
      announce('Citation selected. Press ⌘C / Ctrl+C to copy it.');
    }
  });
}

// ?only=dream renders a single section (used for visual review captures).
const only = new URLSearchParams(location.search).get('only');
if (only) $$('main > section, main > figure').forEach(node => { node.hidden = node.id !== only; });

setupPaper();
setupVideos();
setupTopbar();
setupReveal();
mountSparkles($('#dream .sparkle-canvas'), $('#dream-title'));
try {
  const [response, meta] = await Promise.all([fetch('data/cases.json'), import('./stage/meta.mjs')]);
  window.__stageMeta = meta;
  if (!response.ok) throw new Error(`Scene manifest returned ${response.status}`);
  manifest = validateManifest(await response.json());
  // Annotated paper panels are evidence, never unannotated camera input.
  manifest.cases = manifest.cases.filter(item => !item.frames[0].boxEmbedded);
  $$('[data-case]').forEach(button => { button.hidden = !manifest.cases.some(item => item.id === button.dataset.case); });
  caseIds.push(...manifest.cases.map(item => item.id));
  state = createState(caseIds);
  state.caseId = caseIds.includes(manifest.defaultCase) ? manifest.defaultCase : state.caseId;
  sections.forEach(setupWorkspace);
  updateCaseTabs();
  sections.forEach(renderWorkspace);
  setupStages();
  const jump = new URLSearchParams(location.search).get('scrollto');
  if (jump) document.getElementById(jump)?.scrollIntoView({ behavior: 'instant', block: 'start' });
} catch (error) {
  console.error('Interactive illustrations:', error);
  $$('.occ-loading').forEach(element => { element.textContent = 'The scene data could not be loaded. Please reload the page to retry.'; });
  announce('The interactive scene data could not be loaded. The method figures and videos are still available.');
}
