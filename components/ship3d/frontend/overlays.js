// Data overlays: class zones, passenger lights, passenger beacon, lifeboat
// highlight and the historical timeline of the night of 14–15 April 1912.
import * as THREE from 'three';
import { shipHalfWidth, BOATS, lifeboatGeometry } from './ship.js';

// ------------------------------------------------------------- colours ---
// Validated colour-blind-safe data colours on the Midnight Atlantic navy.
export const HEX = {
  survived: '#3a8ee6',
  died: '#e8693a',
  aqua: '#1fa883',
  class1: '#c08a1c',
  class2: '#8a7ae6',
  class3: '#1fa883',
  slate: '#4a5878',
  unknown: '#7d8ca8',
  brass: '#e3b866',
};

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
/** Survival-rate scale: orange (0) → slate (0.5) → blue (1). Interpolated in sRGB, returns css hex. */
export function rateHex(rate) {
  if (rate == null || !Number.isFinite(+rate)) return HEX.slate;
  const r = Math.max(0, Math.min(1, +rate));
  const [a, b, k] = r < 0.5 ? [HEX.died, HEX.slate, r / 0.5] : [HEX.slate, HEX.survived, (r - 0.5) / 0.5];
  const A = hexToRgb(a), B = hexToRgb(b);
  const c = A.map((v, i) => Math.round(v + (B[i] - v) * k));
  return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
}
export const rateColor = (rate) => new THREE.Color(rateHex(rate));

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hashInt = (n) => (Math.imul((n | 0) ^ 0x9e3779b9, 2654435761) >>> 0);

// --------------------------------------------------------------- zones ---
// Approximate areas of accommodation (the dataset has no cabin locations):
// 1st = amidships upper decks, 2nd = aft upper/middle decks, 3rd = lower decks fore & aft.
export const ZONE_DEFS = [
  { id: 'class1', pclass: 1, label: 'First class', short: '1st', vols: [{ x0: -45, x1: 62, y0: 6, y1: 20.6 }], tags: [[8, 25, 0]] },
  { id: 'class2', pclass: 2, label: 'Second class', short: '2nd', vols: [{ x0: -92, x1: -45, y0: 3, y1: 17.8 }], tags: [[-68, 22, 0]] },
  { id: 'class3', pclass: 3, label: 'Third class', short: '3rd', vols: [{ x0: 76, x1: 118, y0: -3, y1: 12 }, { x0: -124, x1: -92, y0: -3, y1: 11 }], tags: [[97, 18, 0], [-108, 17, 0]] },
];

function zoneGeometry(v, pad = 0.8) {
  const NX = 30, NY = 14;
  const loopLen = 2 * (NY + 1);
  const pos = [];
  for (let i = 0; i <= NX; i++) {
    const x = v.x0 + ((v.x1 - v.x0) * i) / NX;
    const ring = [];
    for (let j = 0; j <= NY; j++) {
      const y = v.y0 + ((v.y1 - v.y0) * j) / NY;
      ring.push([x, y, Math.max(0.5, shipHalfWidth(x, y)) + pad]);
    }
    for (const [px, py, w] of ring) pos.push(px, py, w);
    for (let j = NY; j >= 0; j--) { const [px, py, w] = ring[j]; pos.push(px, py, -w); }
  }
  const idx = [];
  for (let i = 0; i < NX; i++) for (let k = 0; k < loopLen; k++) {
    const k2 = (k + 1) % loopLen;
    const a = i * loopLen + k, b = (i + 1) * loopLen + k, c = (i + 1) * loopLen + k2, d = i * loopLen + k2;
    idx.push(a, b, c, a, c, d);
  }
  // end caps (fan from section centroid)
  for (const [i, flip] of [[0, false], [NX, true]]) {
    let cx = 0, cy = 0, cz = 0;
    for (let k = 0; k < loopLen; k++) { const o = (i * loopLen + k) * 3; cx += pos[o]; cy += pos[o + 1]; cz += pos[o + 2]; }
    const ci = pos.length / 3;
    pos.push(cx / loopLen, cy / loopLen, cz / loopLen);
    for (let k = 0; k < loopLen; k++) {
      const a = i * loopLen + k, b = i * loopLen + ((k + 1) % loopLen);
      if (flip) idx.push(ci, b, a); else idx.push(ci, a, b);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function zoneMaterial(env) {
  return new THREE.ShaderMaterial({
    uniforms: { uTime: env.uTime, uColor: { value: new THREE.Color(HEX.slate) }, uOpacity: { value: 1 }, uHover: { value: 0 } },
    vertexShader: /* glsl */ `
      varying vec3 vN;
      varying vec3 vV;
      varying vec3 vLocal;
      void main() {
        vLocal = position;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vN = normalize(mat3(modelMatrix) * normal);
        vV = cameraPosition - wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uHover;
      uniform float uTime;
      varying vec3 vN;
      varying vec3 vV;
      varying vec3 vLocal;
      void main() {
        float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.0);
        float scan = smoothstep(0.86, 1.0, sin(vLocal.y * 1.3 - uTime * 1.7) * 0.5 + 0.5);
        float ribs = smoothstep(0.94, 1.0, abs(sin(vLocal.x * 0.32)));
        float a = (0.22 + 0.55 * fres + 0.14 * scan + 0.06 * ribs) * (0.85 + 0.75 * uHover);
        vec3 c = uColor * (1.05 + 0.7 * uHover + 0.35 * fres);
        gl_FragColor = vec4(c, clamp(a * uOpacity, 0.0, 0.92));
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    fog: false,
  });
}

export class Zones {
  constructor(parent, env) {
    this.group = new THREE.Group();
    this.group.name = 'zones';
    parent.add(this.group);
    this.meshes = [];
    this.byId = {};
    this.data = {};
    this.opacity = 0;
    this.targetOpacity = 0;
    this.hoverId = null;
    for (const def of ZONE_DEFS) {
      const mat = zoneMaterial(env);
      const entry = { def, mat, meshes: [], hover: 0 };
      for (const v of def.vols) {
        const m = new THREE.Mesh(zoneGeometry(v), mat);
        m.userData.zoneId = def.id;
        m.renderOrder = 4;
        this.group.add(m);
        this.meshes.push(m);
        entry.meshes.push(m);
      }
      this.byId[def.id] = entry;
    }
    this.group.visible = false;
  }

  /** zones: [{id,label,count,survived,rate}] (any may be missing). */
  setData(zones) {
    this.data = {};
    if (Array.isArray(zones)) {
      for (const z of zones) {
        if (!z || typeof z !== 'object') continue;
        let id = z.id;
        if (!this.byId[id] && [1, 2, 3, '1', '2', '3'].includes(z.pclass)) id = 'class' + z.pclass;
        if (!this.byId[id]) { console.warn('[ship3d] unknown zone id', z.id); continue; }
        let rate = z.rate;
        if ((rate == null || !Number.isFinite(+rate)) && Number.isFinite(+z.count) && +z.count > 0 && Number.isFinite(+z.survived)) rate = +z.survived / +z.count;
        this.data[id] = { ...z, id, rate: Number.isFinite(+rate) && rate !== null ? +rate : null };
      }
    }
    for (const id in this.byId) this.byId[id].mat.uniforms.uColor.value.copy(rateColor(this.data[id]?.rate));
  }

  info(id) {
    const def = this.byId[id]?.def;
    const d = this.data[id] || {};
    return { id, label: d.label || def?.label || id, short: def?.short, count: d.count, survived: d.survived, rate: d.rate, color: rateHex(d.rate) };
  }

  setVisible(on, level = 1) { this.targetOpacity = on ? level : 0; }
  setHover(id) { this.hoverId = id; }

  update(dt) {
    this.opacity += (this.targetOpacity - this.opacity) * Math.min(1, dt * 6);
    this.group.visible = this.opacity > 0.01;
    for (const id in this.byId) {
      const e = this.byId[id];
      e.hover += ((this.hoverId === id ? 1 : 0) - e.hover) * Math.min(1, dt * 10);
      e.mat.uniforms.uHover.value = e.hover;
      e.mat.uniforms.uOpacity.value = this.opacity;
    }
  }
}

// ---------------------------------------------------- passenger lights ---
function placePassenger(id, pclass) {
  const def = ZONE_DEFS[pclass - 1];
  const r = mulberry32(hashInt(id));
  const vol = def.vols.length > 1 ? (r() < 0.55 ? def.vols[0] : def.vols[1]) : def.vols[0];
  const x = vol.x0 + 2 + (vol.x1 - vol.x0 - 4) * r();
  const y = vol.y0 + 0.8 + (vol.y1 - vol.y0 - 1.8) * r();
  const w = Math.max(0.5, shipHalfWidth(x, y) * 0.82);
  const z = (r() * 2 - 1) * w;
  return [x, y, z];
}

const toBin = (v) => (v === null || v === undefined || v === '' || Number.isNaN(+v) ? null : (+v >= 0.5 ? 1 : 0));

export class PassengerLights {
  constructor(parent, env) {
    this.parent = parent;
    this.env = env;
    this.rows = [];
    this.points = null;
    this.colorBy = 'actual';
    this.opacity = 0;
    this.targetOpacity = 0;
    this.hasPred = false;
    this.hasActual = false;
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uTime: env.uTime, uViewportH: env.uViewportH, uPixelRatio: env.uPixelRatio, uOpacity: { value: 0 }, uHover: { value: -1 } },
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform float uViewportH;
        uniform float uPixelRatio;
        uniform float uHover;
        attribute float aSeed;
        attribute float aIndex;
        varying vec3 vColor;
        varying float vHot;
        void main() {
          vColor = color;
          vHot = 1.0 - step(0.5, abs(aIndex - uHover));
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float tw = 0.88 + 0.12 * sin(uTime * 2.1 + aSeed * 40.0);
          float px = 2.6 * uViewportH * projectionMatrix[1][1] * 0.5 / -mv.z;
          gl_PointSize = clamp(px, 4.0 * uPixelRatio, 13.0 * uPixelRatio) * tw * (1.0 + vHot * 1.1);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uOpacity;
        varying vec3 vColor;
        varying float vHot;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float core = smoothstep(0.30, 0.12, d);
          float halo = smoothstep(0.5, 0.0, d);
          float a = (core + halo * halo * 0.55) * uOpacity;
          vec3 c = mix(vColor, vec3(1.0), core * 0.18 + vHot * 0.35);
          gl_FragColor = vec4(c, a);
          #include <colorspace_fragment>
        }`,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      depthTest: false, // the lights sit inside the hull: always draw them on top
      fog: false,
    });
  }

  setData(rows) {
    if (this.points) {
      this.parent.remove(this.points);
      this.points.geometry.dispose();
      this.points = null;
    }
    this.rows = [];
    if (!Array.isArray(rows)) rows = [];
    for (const row of rows) {
      try {
        let id, pclass, actual, pred, prob;
        if (Array.isArray(row)) [id, pclass, actual, pred, prob] = row;
        else if (row && typeof row === 'object') ({ passenger_id: id, pclass, actual, predicted: pred, prob } = row);
        pclass = parseInt(pclass, 10);
        if (![1, 2, 3].includes(pclass)) continue;
        id = Number.isFinite(+id) ? +id : this.rows.length + 1;
        const p = Number.isFinite(+prob) && prob !== null && prob !== '' ? +prob : null;
        this.rows.push({ id, pclass, actual: toBin(actual), pred: toBin(pred), prob: p, pos: placePassenger(id, pclass) });
      } catch (e) {
        console.warn('[ship3d] bad passenger row', row, e);
      }
    }
    this.hasPred = this.rows.some((r) => r.pred !== null);
    this.hasActual = this.rows.some((r) => r.actual !== null);
    if (!this.rows.length) return;
    const n = this.rows.length;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const index = new Float32Array(n);
    this.rows.forEach((r, i) => { pos.set(r.pos, i * 3); seed[i] = (hashInt(r.id) % 1000) / 1000; index[i] = i; });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    g.setAttribute('aIndex', new THREE.BufferAttribute(index, 1));
    this.points = new THREE.Points(g, this.mat);
    this.points.renderOrder = 6;
    this.points.frustumCulled = false;
    this.parent.add(this.points);
    this.setColorBy(this.colorBy);
  }

  /** Category for a row under the current colouring: {key, hex, label}. */
  category(r, mode = this.colorBy) {
    switch (mode) {
      case 'class':
        return { key: 'c' + r.pclass, hex: HEX['class' + r.pclass], label: ['1st class', '2nd class', '3rd class'][r.pclass - 1] };
      case 'predicted':
        if (r.pred === null) return { key: 'na', hex: HEX.unknown, label: 'No prediction' };
        return r.pred ? { key: 'p1', hex: HEX.survived, label: 'Predicted survived' } : { key: 'p0', hex: HEX.died, label: 'Predicted did not survive' };
      case 'correct':
        if (r.pred === null || r.actual === null) return { key: 'na', hex: HEX.unknown, label: 'Outcome unknown' };
        return r.pred === r.actual ? { key: 'ok', hex: HEX.survived, label: 'Correct prediction' } : { key: 'bad', hex: HEX.died, label: 'Wrong prediction' };
      case 'actual':
      default:
        if (r.actual === null) return { key: 'na', hex: HEX.unknown, label: 'Outcome unknown' };
        return r.actual ? { key: 's1', hex: HEX.survived, label: 'Survived' } : { key: 's0', hex: HEX.died, label: 'Did not survive' };
    }
  }

  /** Ordered legend entries with counts for the current colouring. */
  legend(mode = this.colorBy) {
    const order = { class: ['c1', 'c2', 'c3', 'na'], actual: ['s1', 's0', 'na'], predicted: ['p1', 'p0', 'na'], correct: ['ok', 'bad', 'na'] }[mode] || [];
    const map = {};
    for (const r of this.rows) {
      const c = this.category(r, mode);
      (map[c.key] ||= { ...c, n: 0 }).n++;
    }
    if (!this.rows.length) {
      const fake = {
        class: [[HEX.class1, '1st class'], [HEX.class2, '2nd class'], [HEX.class3, '3rd class']],
        actual: [[HEX.survived, 'Survived'], [HEX.died, 'Did not survive']],
        predicted: [[HEX.survived, 'Predicted survived'], [HEX.died, 'Predicted did not survive']],
        correct: [[HEX.survived, 'Correct prediction'], [HEX.died, 'Wrong prediction']],
      }[mode] || [];
      return fake.map(([hex, label]) => ({ hex, label, n: null }));
    }
    return order.filter((k) => map[k]).map((k) => map[k]);
  }

  setColorBy(mode) {
    if (!['class', 'actual', 'predicted', 'correct'].includes(mode)) mode = 'actual';
    if ((mode === 'predicted' || mode === 'correct') && this.rows.length && !this.hasPred) mode = 'actual';
    this.colorBy = mode;
    if (!this.points) return mode;
    const col = this.points.geometry.attributes.color;
    const c = new THREE.Color();
    this.rows.forEach((r, i) => { c.set(this.category(r, mode).hex); col.setXYZ(i, c.r, c.g, c.b); });
    col.needsUpdate = true;
    return mode;
  }

  setVisible(on, level = 1) { this.targetOpacity = on ? level : 0; }

  setHover(i) { this.mat.uniforms.uHover.value = i == null ? -1 : i; }

  /** Nearest light to a screen point (px) within `radius` px, or -1. */
  pick(mx, my, camera, w, h, radius = 9) {
    if (!this.points || this.opacity < 0.5) return -1;
    const m = this.points.matrixWorld;
    const v = new THREE.Vector3();
    let best = -1, bestD = radius * radius;
    for (let i = 0; i < this.rows.length; i++) {
      v.fromArray(this.rows[i].pos).applyMatrix4(m).project(camera);
      if (v.z > 1) continue;
      const dx = (v.x * 0.5 + 0.5) * w - mx, dy = (-v.y * 0.5 + 0.5) * h - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  update(dt) {
    this.opacity += (this.targetOpacity - this.opacity) * Math.min(1, dt * 6);
    this.mat.uniforms.uOpacity.value = this.opacity;
    if (this.points) this.points.visible = this.opacity > 0.01;
  }
}

// -------------------------------------------------------------- beacon ---
const BEACON_SPOTS = { 1: [6, 20.1, 0], 2: [-62, 20.1, 0], 3: [100, 13.2, 0] };

export class Beacon {
  constructor(parent, env) {
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.color = new THREE.Color(HEX.slate);
    this.targetColor = this.color.clone();
    this.targetPos = new THREE.Vector3(...BEACON_SPOTS[1]);
    this.hasTarget = false;
    this.labelLocal = new THREE.Vector3(0, 36, 0);
    this.data = null;
    const H = 70;
    const beamGeo = new THREE.CylinderGeometry(1.6, 3.6, H, 32, 1, true);
    beamGeo.translate(0, H / 2, 0);
    this.beamMat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: this.color }, uTime: env.uTime, uOpacity: { value: 1 } },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        varying float vFacing;
        void main() {
          vUv = uv;
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vec3 n = normalize(mat3(modelMatrix) * normal);
          vFacing = abs(dot(n, normalize(cameraPosition - wp.xyz)));
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        uniform float uTime;
        uniform float uOpacity;
        varying vec2 vUv;
        varying float vFacing;
        void main() {
          float fall = pow(1.0 - vUv.y, 1.6);
          float shimmer = 0.85 + 0.15 * sin(vUv.y * 40.0 - uTime * 4.0);
          float a = fall * pow(vFacing, 1.5) * shimmer * 0.85 * uOpacity;
          gl_FragColor = vec4(uColor * 1.7, a);
          #include <colorspace_fragment>
        }`,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    this.beam = new THREE.Mesh(beamGeo, this.beamMat);
    this.beam.renderOrder = 7;
    this.coreMat = new THREE.MeshBasicMaterial({ color: this.color.clone(), toneMapped: false });
    this.core = new THREE.Mesh(new THREE.SphereGeometry(2.6, 20, 12), this.coreMat);
    this.core.position.y = 2.2;
    this.rings = [0, 1].map(() => {
      const m = new THREE.Mesh(new THREE.RingGeometry(0.7, 1.0, 64), new THREE.MeshBasicMaterial({
        color: this.color.clone(), transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false, blending: THREE.AdditiveBlending,
      }));
      m.rotation.x = -Math.PI / 2;
      m.position.y = 0.3;
      m.renderOrder = 7;
      return m;
    });
    this.group.add(this.beam, this.core, ...this.rings);
    this.group.position.copy(this.targetPos);
  }

  /** beacon: {pclass, prob, label}. */
  setTarget(b) {
    this.data = b && typeof b === 'object' ? b : null;
    if (!this.data) { this.group.visible = false; return; }
    const pc = [1, 2, 3].includes(parseInt(b.pclass, 10)) ? parseInt(b.pclass, 10) : 3;
    this.targetPos.set(...BEACON_SPOTS[pc]);
    const prob = Number.isFinite(+b.prob) && b.prob !== null ? Math.max(0, Math.min(1, +b.prob)) : null;
    this.prob = prob;
    this.targetColor.copy(rateColor(prob));
    if (!this.hasTarget) {
      this.group.position.copy(this.targetPos);
      this.color.copy(this.targetColor);
      this.hasTarget = true;
    }
  }

  labelText() {
    if (!this.data) return '';
    const label = this.data.label || 'Your passenger';
    return this.prob == null ? label : `${label} · ${Math.round(this.prob * 100)}% est. survival`;
  }

  update(dt, t) {
    const k = 1 - Math.exp(-dt * 3.2);
    this.group.position.lerp(this.targetPos, k);
    this.color.lerp(this.targetColor, k);
    this.coreMat.color.copy(this.color).multiplyScalar(2.4);
    this.rings.forEach((r, i) => {
      const ph = ((t * 0.5) + i * 0.5) % 1;
      r.scale.setScalar(3 + ph * 20);
      r.material.color.copy(this.color).multiplyScalar(2.0);
      r.material.opacity = (1 - ph) * (1 - ph);
    });
    const s = 1 + 0.12 * Math.sin(t * 3.0);
    this.core.scale.setScalar(s);
  }
}

// ------------------------------------------------------------ timeline ---
// Minutes after 23:40 on 14 April 1912 (approximate, from historical accounts).
export const TIMELINE_END = 160;
export const TIMELINE_EVENTS = [
  { t: 0, time: '23:40', text: 'The Titanic strikes an iceberg on her starboard bow.' },
  { t: 25, time: '00:05', text: 'Captain Smith orders the lifeboats to be uncovered.' },
  { t: 65, time: '00:45', text: 'The first lifeboat, No. 7, is lowered.' },
  { t: 145, time: '02:05', text: 'The last lifeboat, Collapsible D, is lowered.' },
  { t: 158, time: '02:18', text: "The ship's lights fail." },
  { t: 160, time: '02:20', text: 'The Titanic sinks beneath the surface.' },
];
const BETWEEN = [
  [25, 'Water floods the forward compartments; the ship settles slightly by the bow.'],
  [65, 'Crew prepare the boats as the bow slowly settles.'],
  [145, 'Lifeboats are lowered and pull away from the ship.'],
  [158, 'The bow slips under as the stern begins to rise.'],
];

export function clockFor(t) {
  const m = (23 * 60 + 40 + Math.round(t)) % (24 * 60);
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

export function captionFor(t) {
  let ev = TIMELINE_EVENTS[0];
  for (const e of TIMELINE_EVENTS) if (t >= e.t) ev = e;
  // between milestones, describe what is happening (but the milestone itself shows for a few minutes)
  if (t - ev.t > 6) {
    for (const [until, text] of BETWEEN) if (t < until) return { time: clockFor(t), text, milestone: false };
  }
  return { time: ev.time, text: ev.text, milestone: true };
}

function interp(t, pts, easeLast = false) {
  if (t <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i][0]) {
      let k = (t - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
      if (easeLast && i === pts.length - 1) k = k * k;
      else k = k * k * (3 - 2 * k);
      return pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * k;
    }
  }
  return pts[pts.length - 1][1];
}

// launch order: first No. 7 (starboard forward), last Collapsible D (port forward)
const LAUNCH_ORDER = [11, 3, 10, 2, 9, 1, 12, 4, 13, 5, 14, 6, 15, 7, 8, 0, 18, 19, 17, 16];
const LAUNCH_T = {};
LAUNCH_ORDER.forEach((b, i) => { LAUNCH_T[b] = 65 + (80 * i) / (LAUNCH_ORDER.length - 1); });

export class Timeline {
  constructor(scene, ship, ocean) {
    this.ship = ship;
    this.ocean = ocean;
    this.t = 0;
    this.playing = false;
    this.duration = 30; // seconds for a full play-through
    this.aboard = BOATS.map(() => true);
    const mat = new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.6 });
    this.water = new THREE.InstancedMesh(lifeboatGeometry(), mat, BOATS.length);
    this.water.frustumCulled = false;
    this.hidden = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < BOATS.length; i++) this.water.setMatrixAt(i, this.hidden);
    scene.add(this.water);
    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._e = new THREE.Euler();
  }

  get active() { return this.t > 0.001; }

  setTime(t) { this.t = Math.max(0, Math.min(TIMELINE_END, +t || 0)); }

  play() {
    if (this.t >= TIMELINE_END) this.t = 0;
    this.playing = true;
  }

  pause() { this.playing = false; }

  reset() { this.playing = false; this.t = 0; }

  /** Ship pose / lights for the current time (all neutral at t = 0). */
  state(clock) {
    const t = this.t;
    const pitchDeg = interp(t, [[0, 0], [25, 0.3], [65, 1.0], [120, 3.0], [145, 5.5], [155, 9.5], [158, 12], [160, 16]]);
    const sink = interp(t, [[0, 0], [65, -0.8], [120, -2.6], [145, -5], [155, -10], [158, -16], [160, -118]], true);
    let lights = 1;
    if (t >= 158) lights = 0;
    else if (t > 150) {
      const fade = 1 - ((t - 150) / 8) * 0.55;
      const flick = Math.sin(clock * 23.0) * Math.sin(clock * 7.3 + 1.0) > -0.15 ? 1 : 0.15;
      lights = fade * flick;
    }
    return {
      pitch: (pitchDeg * Math.PI) / 180,
      sink,
      lights,
      shipVisible: t < TIMELINE_END - 0.001,
      hullFoam: t < 158 ? 1 : Math.max(0, 1 - (t - 158) / 1.5),
      smoke: t <= 0.001 ? 1 : Math.max(0, 1 - t / 45),
    };
  }

  update(dt, clock) {
    if (this.playing) {
      this.t = Math.min(TIMELINE_END, this.t + (dt * TIMELINE_END) / this.duration);
      if (this.t >= TIMELINE_END) this.playing = false;
    }
    // lifeboats: aboard until launch time, then on the water drifting away
    for (let i = 0; i < BOATS.length; i++) {
      const lt = LAUNCH_T[i];
      const launched = this.t >= lt;
      if (launched === this.aboard[i]) {
        this.aboard[i] = !launched;
        this.ship.setBoatAboard(i, !launched);
      }
      if (!launched) { this.water.setMatrixAt(i, this.hidden); continue; }
      const b = BOATS[i];
      const age = this.t - lt; // minutes on the water
      const r = mulberry32(i * 977 + 3);
      const drift = 3 + 60 * (1 - Math.exp(-age / 18)) + age * 0.5;
      const x = b.x + (r() - 0.5) * 30 * (1 - Math.exp(-age / 25));
      const z = b.side * (16 + drift + r() * 6);
      const y = this.ocean.heightAt(x, z, clock) * 0.85 + 0.25;
      const appear = Math.min(1, age / 2.5);
      this._e.set(Math.sin(clock * 0.9 + i) * 0.05, (r() - 0.5) * 0.9, Math.sin(clock * 0.7 + i * 2) * 0.04);
      this._q.setFromEuler(this._e);
      const h = b.kind === 'collapsible' ? 0.55 : 1.0;
      const w = b.kind === 'cutter' ? 1.2 : 1.35;
      this._m.compose(new THREE.Vector3(x, y, z), this._q, new THREE.Vector3((b.len / 2) * appear, h * appear, w * appear));
      this.water.setMatrixAt(i, this._m);
    }
    this.water.instanceMatrix.needsUpdate = true;
  }
}

