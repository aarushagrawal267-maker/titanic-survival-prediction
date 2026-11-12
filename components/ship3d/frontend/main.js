// ship3d — bidirectional Streamlit component (plain ES modules, no build step).
// Implements the Streamlit component protocol by hand, bootstraps the 3D
// scene once, and updates it in place on every re-render.
import * as THREE from 'three';
import { Stage, webglAvailable, CAMERA_PRESETS } from './scene.js';
import { createOcean, createIceberg } from './ocean.js';
import { buildShip } from './ship.js';
import {
  Zones, PassengerLights, Beacon, Timeline, ZONE_DEFS,
  TIMELINE_EVENTS, TIMELINE_END, captionFor, clockFor,
} from './overlays.js';

// ------------------------------------------------------------ protocol ---
const Streamlit = {
  send(type, data = {}) {
    try { window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*'); } catch (e) { console.warn('[ship3d] postMessage failed', e); }
  },
  ready() { this.send('streamlit:componentReady', { apiVersion: 1 }); },
  setHeight(height) { this.send('streamlit:setFrameHeight', { height }); },
  setValue(value) { this.send('streamlit:setComponentValue', { value, dataType: 'json' }); },
};

const $ = (id) => document.getElementById(id);
const app = $('app');
const hud = $('hud');
const tooltip = $('tooltip');

const state = {
  mode: 'hero',
  colorBy: 'actual',
  tod: 'night',
  camera: 'side',
  autoRotate: true,
  showHud: true,
  zonesOn: true,
  lightsOn: true,
  boatsOn: false,
  timelineOn: false,
  stats: null,
  height: null,
  first: true,
  hover: null, // {type:'zone', id} | {type:'pax', index}
  interacted: false,
};
const lastArgs = {};

let stage = null, ship = null, ocean = null, iceberg = null;
let zones = null, lights = null, beacon = null, timeline = null;
let glOK = false;

// --------------------------------------------------------------- build ---
function build() {
  stage = new Stage($('canvas-host'), { quality: 'auto' });
  const { scene, env } = stage;

  ocean = createOcean(stage);
  scene.add(ocean.mesh);
  iceberg = createIceberg(env);
  scene.add(iceberg.group);

  ship = buildShip(env);
  scene.add(ship.root);

  zones = new Zones(ship.root, env);
  lights = new PassengerLights(ship.root, env);
  beacon = new Beacon(ship.root, env);
  timeline = new Timeline(scene, ship, ocean);

  stage.onPreset = (p) => iceberg.applyPreset(p);
  stage._applyPreset(stage.cur);

  // until the user moves the camera, keep the preset framed for the current frame size
  let lastAspect = 0;
  stage.onResize = (w, h) => {
    const a = w / h;
    if (!state.first && !state.interacted && Math.abs(a - lastAspect) > 0.02) setCamera(state.camera, true);
    lastAspect = a;
  };
  stage.onUpdate(update);
  stage.controls.addEventListener('start', onFirstInteraction);
  wireHud();
  wirePointer();
  stage.start();
  glOK = true;
}

function showFallback() {
  $('fallback').hidden = false;
  hud.style.display = 'none';
}

// ---------------------------------------------------------- per frame ---
const tmpV = new THREE.Vector3();
function update(dt, t) {
  // timeline drives the ship's trim, depth and lights
  timeline.update(dt, t);
  const tl = timeline.state(t);
  const sailing = timeline.active ? 0 : 1;
  ocean.update(dt, sailing);
  ocean.uniforms.uHullFoam.value = tl.hullFoam;
  ocean.uniforms.uLights.value = tl.lights;

  // gentle bob & pitch with the swell
  const r = ship.root;
  r.position.y = 0.22 * Math.sin(t * 0.55) + tl.sink;
  r.rotation.z = 0.0035 * Math.sin(t * 0.42) - tl.pitch;
  r.rotation.x = 0.006 * Math.sin(t * 0.33 + 1.0);
  r.visible = tl.shipVisible;
  ship.setLights(tl.lights);
  ship.smoke.emission = tl.smoke;
  ship.smoke.wind = sailing;
  ship.update(dt);
  stage.shipBoxEnabled = !timeline.active;

  // overlay visibility (fade away while the timeline is running)
  const explorer = state.mode === 'explorer';
  // dim the data overlays while the lifeboats are highlighted so the outlines read clearly
  zones.setVisible(explorer && state.zonesOn && !timeline.active, state.boatsOn ? 0.25 : 1);
  lights.setVisible(explorer && state.lightsOn && !timeline.active, state.boatsOn ? 0.2 : 1);
  zones.update(dt);
  lights.update(dt);
  beacon.group.visible = state.mode === 'passenger' && !!beacon.data;
  if (beacon.group.visible) beacon.update(dt, t);

  // lifeboat highlight pulse
  ship.outline.visible = explorer && state.boatsOn;
  ship.markers.visible = ship.outline.visible && !timeline.active;
  if (ship.outline.visible) ship.outline.material.opacity = 0.45 + 0.45 * (0.5 + 0.5 * Math.sin(t * 3.2));

  // auto-rotate pauses while the pointer rests on something
  stage.controls.autoRotate = state.autoRotate && !state.hover && !state.dragging;

  if (timeline.playing || state.timelineOn) syncTimelineUi();
  projectLabels();
}

// ------------------------------------------------------ projected labels ---
const zoneTagEls = [];
function buildZoneTags() {
  const host = $('zone-tags');
  host.innerHTML = '';
  zoneTagEls.length = 0;
  for (const def of ZONE_DEFS) {
    for (const p of def.tags) {
      const el = document.createElement('div');
      el.className = 'zone-tag';
      el.setAttribute('role', 'button');
      el.tabIndex = 0;
      el.innerHTML = '<span class="dot"></span><span class="lbl"></span><span class="mono"></span>';
      // the tags are clickable shortcuts for their zone (easier than hitting the volume)
      el.addEventListener('pointerenter', (e) => { onFirstInteraction(); showZoneHover(def.id, e); });
      el.addEventListener('pointermove', (e) => showZoneHover(def.id, e));
      el.addEventListener('pointerleave', () => clearHover());
      el.addEventListener('click', () => sendZoneClick(def.id));
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sendZoneClick(def.id); } });
      host.appendChild(el);
      zoneTagEls.push({ el, id: def.id, local: new THREE.Vector3(...p) });
    }
  }
  refreshZoneTags();
}
function refreshZoneTags() {
  for (const z of zoneTagEls) {
    const info = zones.info(z.id);
    z.el.querySelector('.dot').style.background = info.color;
    z.el.querySelector('.lbl').textContent = info.short || info.label;
    z.el.querySelector('.mono').textContent = info.rate == null ? '' : `${Math.round(info.rate * 100)}%`;
  }
}

function toScreen(local, out) {
  tmpV.copy(local).applyMatrix4(ship.root.matrixWorld).project(stage.camera);
  out.x = (tmpV.x * 0.5 + 0.5) * stage.width;
  out.y = (-tmpV.y * 0.5 + 0.5) * stage.height;
  out.ok = tmpV.z < 1 && tmpV.z > -1;
  return out;
}

const scr = {};
function projectLabels() {
  const far = stage.camera.position.distanceTo(ship.root.position) > 700;
  const showTags = state.mode === 'explorer' && state.zonesOn && !timeline.active && zones.opacity > 0.2 && !far;
  for (const z of zoneTagEls) {
    if (!showTags) { z.el.style.display = 'none'; continue; }
    toScreen(z.local, scr);
    z.el.style.display = scr.ok ? '' : 'none';
    z.el.style.transform = `translate(${scr.x.toFixed(1)}px, ${scr.y.toFixed(1)}px) translate(-50%, -100%)`;
    z.el.style.opacity = state.hover?.type === 'zone' && state.hover.id !== z.id ? '0.45' : '1';
  }
  const bl = $('beacon-label');
  if (beacon.group.visible) {
    tmpV.copy(beacon.labelLocal).add(beacon.group.position);
    toScreen(tmpV, scr);
    bl.hidden = !scr.ok;
    bl.style.transform = `translate(${scr.x.toFixed(1)}px, ${scr.y.toFixed(1)}px) translate(-50%, -100%)`;
    const text = beacon.labelText();
    const tEl = bl.querySelector('.beacon-text');
    if (tEl.textContent !== text) tEl.textContent = text;
    bl.querySelector('.beacon-dot').style.color = '#' + beacon.color.getHexString(THREE.SRGBColorSpace);
  } else {
    bl.hidden = true;
  }
}

// ----------------------------------------------------------------- HUD ---
function setPressed(el, on) { el.setAttribute('aria-pressed', on ? 'true' : 'false'); }
function setChecked(group, value) {
  for (const b of group.querySelectorAll('button')) b.setAttribute('aria-checked', b.dataset.value === value ? 'true' : 'false');
}

function onFirstInteraction() {
  if (state.interacted) return;
  state.interacted = true;
  $('hint').classList.add('gone');
}

function setTod(v, instant = false) {
  if (!['night', 'dusk', 'day'].includes(v)) v = 'night';
  state.tod = v;
  stage.setTimeOfDay(v, instant);
  setChecked($('tod'), v);
}

function setCamera(v, instant = false) {
  if (!CAMERA_PRESETS[v]) v = 'side';
  state.camera = v;
  const resolved = state.mode === 'passenger' && v === 'side' ? 'quarter' : v;
  stage.flyTo(resolved, instant ? 0 : 1.2);
  $('view-label').textContent = CAMERA_PRESETS[v].label;
  for (const b of $('view-menu').querySelectorAll('button')) b.setAttribute('aria-selected', b.dataset.value === v ? 'true' : 'false');
}

function setAutoRotate(on) {
  state.autoRotate = !!on;
  setPressed($('rotate-btn'), state.autoRotate);
}

function setColorBy(v) {
  state.colorBy = lights.setColorBy(v);
  setChecked($('colorby'), state.colorBy);
  renderLegend();
}

function renderLegend() {
  const host = $('legend-lights');
  const items = lights.legend(state.colorBy);
  host.innerHTML = '';
  for (const it of items) {
    const el = document.createElement('span');
    el.className = 'legend-item';
    el.innerHTML = `<span class="dot" style="background:${it.hex};color:${it.hex}"></span><span></span>${it.n != null ? `<span class="n">${it.n.toLocaleString()}</span>` : ''}`;
    el.children[1].textContent = it.label;
    host.appendChild(el);
  }
  const hasRows = lights.rows.length > 0;
  for (const b of $('colorby').querySelectorAll('button')) {
    const needsPred = b.dataset.value === 'predicted' || b.dataset.value === 'correct';
    b.disabled = !hasRows || (needsPred && !lights.hasPred);
    b.title = b.disabled ? (hasRows ? 'No model predictions supplied' : 'No passenger data supplied') : '';
  }
  $('t-lights').disabled = !hasRows;
  $('legend-lights').style.display = hasRows ? '' : 'none';
  $('colorby').style.display = hasRows ? '' : 'none';
  $('colorby-panel').querySelector('.panel-title').style.display = hasRows ? '' : 'none';
}

const num = (v) => (v === null || v === undefined || v === '' || typeof v === 'boolean' ? NaN : +v);
function renderStats() {
  const el = $('stats-line');
  const s = state.stats;
  const parts = [];
  if (s && typeof s === 'object') {
    const total = num(s.total), surv = num(s.survived);
    if (Number.isFinite(total)) parts.push(`${total.toLocaleString()} passengers`);
    if (Number.isFinite(surv)) parts.push(`${surv.toLocaleString()} survived${Number.isFinite(total) && total > 0 ? ` (${((surv / total) * 100).toFixed(1)}%)` : ''}`);
    if (Number.isFinite(num(s.accuracy))) parts.push(`accuracy ${(num(s.accuracy) * 100).toFixed(1)}%`);
  }
  if (lights.rows.length && !(s && num(s.total) === lights.rows.length)) parts.push(`${lights.rows.length.toLocaleString()} lights`);
  el.textContent = parts.join(' · ');
}

function setTimelineOpen(on) {
  state.timelineOn = !!on;
  setPressed($('t-timeline'), state.timelineOn);
  $('timeline-card').hidden = !state.timelineOn;
  if (!state.timelineOn) timeline.reset();
  syncTimelineUi();
}

let lastCaption = '';
function syncTimelineUi() {
  const s = $('tl-slider');
  const t = timeline.t;
  if (document.activeElement !== s || timeline.playing) s.value = String(Math.round(t));
  s.style.setProperty('--fill', `${(t / TIMELINE_END) * 100}%`);
  $('tl-time').textContent = clockFor(t);
  $('tl-play').classList.toggle('playing', timeline.playing);
  $('tl-play').setAttribute('aria-label', timeline.playing ? 'Pause timeline' : 'Play timeline');
  const c = captionFor(t);
  const html = `<span class="t">${c.time}</span>${c.text}`;
  if (html !== lastCaption) { $('tl-caption').innerHTML = html; lastCaption = html; }
}

function wireHud() {
  // time of day
  $('tod').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b) setTod(b.dataset.value);
  });
  // camera presets dropdown
  const menu = $('view-menu');
  const btn = $('view-btn');
  const closeMenu = () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    btn.setAttribute('aria-expanded', menu.hidden ? 'false' : 'true');
  });
  menu.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    setCamera(b.dataset.value);
    closeMenu();
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('#view-dd')) closeMenu(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });

  $('rotate-btn').addEventListener('click', () => setAutoRotate(!state.autoRotate));
  $('colorby').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b && !b.disabled) setColorBy(b.dataset.value);
  });
  $('t-zones').addEventListener('click', () => { state.zonesOn = !state.zonesOn; setPressed($('t-zones'), state.zonesOn); });
  $('t-lights').addEventListener('click', () => { state.lightsOn = !state.lightsOn; setPressed($('t-lights'), state.lightsOn); });
  $('t-boats').addEventListener('click', () => {
    state.boatsOn = !state.boatsOn;
    setPressed($('t-boats'), state.boatsOn);
    $('lifeboat-card').hidden = !state.boatsOn;
  });
  $('t-timeline').addEventListener('click', () => setTimelineOpen(!state.timelineOn));

  // timeline
  const ticks = $('tl-ticks');
  for (const ev of TIMELINE_EVENTS) {
    const s = document.createElement('span');
    s.style.left = `${(ev.t / TIMELINE_END) * 100}%`;
    s.title = `${ev.time} — ${ev.text}`;
    ticks.appendChild(s);
  }
  $('tl-slider').addEventListener('input', (e) => {
    timeline.pause();
    timeline.setTime(+e.target.value);
    syncTimelineUi();
  });
  $('tl-play').addEventListener('click', () => {
    if (timeline.playing) timeline.pause(); else timeline.play();
    syncTimelineUi();
  });
  syncTimelineUi();
  buildZoneTags();
}

// ------------------------------------------------------------- pointer ---
function wirePointer() {
  const el = stage.renderer.domElement;
  let down = null;
  let pending = null;
  el.addEventListener('pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button };
    state.dragging = true;
    onFirstInteraction();
  });
  window.addEventListener('pointerup', (e) => {
    state.dragging = false;
    if (!down) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const quick = performance.now() - down.t < 600;
    if (down.button === 0 && moved < 6 && quick && e.target === el) handleClick(e);
    down = null;
  });
  el.addEventListener('pointermove', (e) => {
    const wasPending = !!pending;
    pending = e;
    if (wasPending) return; // one pick per animation frame
    requestAnimationFrame(() => { if (pending) hoverAt(pending); pending = null; });
  });
  el.addEventListener('pointerleave', () => { clearHover(); });
  el.addEventListener('wheel', onFirstInteraction, { passive: true });
}

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();

function localXY(e) {
  const r = stage.renderer.domElement.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function pickAt(e) {
  if (state.mode !== 'explorer' || timeline.active || !glOK) return null;
  const { x, y } = localXY(e);
  if (state.lightsOn && lights.points) {
    const i = lights.pick(x, y, stage.camera, stage.width, stage.height, 7);
    if (i >= 0) return { type: 'pax', index: i };
  }
  if (state.zonesOn && zones.opacity > 0.3) {
    ndc.set((x / stage.width) * 2 - 1, -(y / stage.height) * 2 + 1);
    raycaster.setFromCamera(ndc, stage.camera);
    const hit = raycaster.intersectObjects(zones.meshes, false)[0];
    if (hit) return { type: 'zone', id: hit.object.userData.zoneId };
  }
  return null;
}

function showZoneHover(id, e) {
  if (state.dragging) return;
  setHover({ type: 'zone', id }, e);
}

function sendZoneClick(id) {
  Streamlit.setValue({ event: 'zone_click', zone: id, nonce: Date.now() });
}

function hoverAt(e) {
  if (state.dragging) return;
  setHover(pickAt(e), e);
}

function setHover(h, e) {
  state.hover = h;
  zones.setHover(h?.type === 'zone' ? h.id : null);
  lights.setHover(h?.type === 'pax' ? h.index : null);
  stage.renderer.domElement.style.cursor = h ? 'pointer' : '';
  if (!h) { tooltip.hidden = true; return; }
  tooltip.innerHTML = h.type === 'zone' ? zoneTooltip(h.id) : paxTooltip(lights.rows[h.index]);
  tooltip.hidden = false;
  const { x, y } = localXY(e);
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  const W = app.clientWidth, H = app.clientHeight;
  const left = x + 14 + tw > W - 6 ? x - tw - 14 : x + 14;
  const top = y + 14 + th > H - 6 ? y - th - 14 : y + 14;
  tooltip.style.transform = 'none';
  tooltip.style.left = `${Math.max(6, left)}px`;
  tooltip.style.top = `${Math.max(6, top)}px`;
}

function clearHover() {
  state.hover = null;
  zones?.setHover(null);
  lights?.setHover(null);
  tooltip.hidden = true;
  if (stage) stage.renderer.domElement.style.cursor = '';
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function zoneTooltip(id) {
  const z = zones.info(id);
  const pct = z.rate == null ? null : Math.round(z.rate * 100);
  const rows = [];
  if (Number.isFinite(+z.count) && z.count !== null) rows.push(`<div class="tt-row"><span>Passengers</span><span class="v">${(+z.count).toLocaleString()}</span></div>`);
  if (Number.isFinite(+z.survived) && z.survived !== null) rows.push(`<div class="tt-row"><span>Survived</span><span class="v">${(+z.survived).toLocaleString()}</span></div>`);
  rows.push(`<div class="tt-row"><span>Survival rate</span><span class="v">${pct == null ? '—' : pct + '%'}</span></div>`);
  const bar = pct == null ? '' : `<div class="tt-bar"><i style="width:${pct}%;background:${z.color}"></i></div>`;
  return `<div class="tt-title"><span class="swatch" style="background:${z.color}"></span>${esc(z.label)}</div>${rows.join('')}${bar}<div class="tt-hint">Click to explore this class ›</div>`;
}

function paxTooltip(r) {
  if (!r) return '';
  const cat = lights.category(r);
  const outcome = (v, yes, no) => (v === null ? 'unknown' : v ? yes : no);
  const lines = [
    `<div class="tt-row"><span>Class</span><span class="v">${r.pclass}</span></div>`,
    `<div class="tt-row"><span>Actual</span><span class="v">${outcome(r.actual, 'survived', 'did not survive')}</span></div>`,
  ];
  if (r.pred !== null) lines.push(`<div class="tt-row"><span>Predicted</span><span class="v">${outcome(r.pred, 'survived', 'did not survive')}</span></div>`);
  if (r.prob !== null) lines.push(`<div class="tt-row"><span>Est. probability</span><span class="v">${Math.round(r.prob * 100)}%</span></div>`);
  return `<div class="tt-title"><span class="swatch" style="background:${cat.hex}"></span>Passenger #${esc(r.id)}</div>${lines.join('')}`;
}

function handleClick(e) {
  const h = pickAt(e);
  if (!h) return;
  if (h.type === 'zone') {
    sendZoneClick(h.id);
  } else if (h.type === 'pax') {
    const r = lights.rows[h.index];
    Streamlit.setValue({ event: 'passenger_click', passenger_id: r.id, pclass: r.pclass, nonce: Date.now() });
  }
}

// ---------------------------------------------------------------- mode ---
function applyMode(mode) {
  if (!['hero', 'explorer', 'passenger'].includes(mode)) mode = 'hero';
  state.mode = mode;
  app.dataset.mode = mode;
  clearHover();
  if (mode !== 'explorer') {
    setTimelineOpen(false);
    if (state.boatsOn) $('t-boats').click();
  }
  stage.controls.autoRotateSpeed = mode === 'hero' ? 0.28 : mode === 'passenger' ? 0.22 : 0.35;
}

// -------------------------------------------------------------- render ---
const sig = (v) => { try { return JSON.stringify(v ?? null); } catch (e) { return String(Math.random()); } };
function changed(args, key) {
  const s = sig(args[key]);
  if (lastArgs[key] === s) return false;
  lastArgs[key] = s;
  return true;
}

function onRender(args) {
  args = args && typeof args === 'object' ? args : {};
  let h = parseInt(args.height, 10);
  if (!Number.isFinite(h)) h = 560;
  h = Math.max(160, Math.min(2400, h));
  if (h !== state.height) { state.height = h; Streamlit.setHeight(h); }
  if (!glOK) return;

  const first = state.first;
  state.first = false;
  try {
    if (changed(args, 'quality')) stage.setQuality(args.quality || 'auto');
    const modeChanged = changed(args, 'mode');
    if (modeChanged) applyMode(args.mode || 'hero');
    if (changed(args, 'show_hud')) { state.showHud = args.show_hud !== false; hud.classList.toggle('hidden-hud', !state.showHud); }
    if (changed(args, 'zones')) { zones.setData(args.zones); refreshZoneTags(); }
    if (changed(args, 'passengers')) { lights.setData(args.passengers); setColorBy(state.colorBy); renderStats(); }
    if (changed(args, 'color_by')) setColorBy(args.color_by || 'actual');
    if (changed(args, 'stats')) { state.stats = args.stats; renderStats(); }
    if (changed(args, 'beacon')) beacon.setTarget(args.beacon);
    if (changed(args, 'time_of_day')) setTod(args.time_of_day || 'night', first);
    if (changed(args, 'auto_rotate')) setAutoRotate(args.auto_rotate !== false);
    // camera: re-apply on first render, when the arg changes, or when the mode changed
    const camChanged = changed(args, 'camera');
    if (first || camChanged || modeChanged) setCamera(args.camera || 'side', first);
    renderLegend();
    if (first) stage.renderOnce(0); // paint immediately
  } catch (e) {
    console.error('[ship3d] render error', e);
  }
}

window.addEventListener('message', (event) => {
  const d = event.data;
  if (!d || d.type !== 'streamlit:render') return;
  onRender(d.args);
});

// ----------------------------------------------------------- bootstrap ---
try {
  if (webglAvailable()) build();
  else showFallback();
} catch (e) {
  console.error('[ship3d] WebGL initialisation failed', e);
  glOK = false;
  showFallback();
}
Streamlit.ready();

// Small debugging / testing hook (harmless in production).
window.__ship3d = {
  get state() { return state; },
  get stage() { return stage; },
  get timeline() { return timeline; },
  setTimeline(t) { if (!timeline) return; if (!state.timelineOn) setTimelineOpen(true); timeline.pause(); timeline.setTime(t); syncTimelineUi(); },
  setColorBy: (v) => setColorBy(v),
  setTod: (v) => setTod(v),
  setCamera: (v, instant) => setCamera(v, instant),
  render: (args) => onRender(args),
  ready: () => glOK,
};
