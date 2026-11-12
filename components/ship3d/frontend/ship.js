// Procedural RMS Titanic, built entirely in code from lofted / extruded
// geometry (1 scene unit = 1 metre, ~269 m long). Ship-local frame:
// +X = bow, +Y = up (waterline at y = 0), +Z = port side.
import * as THREE from 'three';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const KEEL = -10.5;

// ------------------------------------------------------------- hull form ---
/** Height of the hull's top edge (sheer line) along the length. */
export function sheer(x) {
  const u = clamp(x, -134.5, 134.5) / 134.5;
  return 11.2 + (u > 0 ? 3.4 : 1.6) * u * u;
}
/** Stem: slightly raked. */
export function bowX(y) { return 133.2 + y * 0.09; }
/** Counter stern: overhangs above the waterline, cut away underneath. */
export function sternX(y) { return y >= 1.5 ? -134.3 : -134.3 + (1.5 - y) * 1.15; }

/** Half-breadth of the hull at (x, y). */
export function halfBeam(x, y) {
  const bx = bowX(y), sx = sternX(y);
  if (x >= bx || x <= sx) return 0;
  const Lb = 70 - (y - KEEL) * 0.5; // finer entry low down, fuller (flared) at deck
  const db = (bx - x) / Lb;
  const fb = db < 1 ? 1 - Math.pow(1 - db, 2.2) : 1;
  const Ls = 22 + Math.max(0, 1.5 - y) * 2.2; // long, fine run under the counter
  const ds = (x - sx) / Ls;
  const fs = ds < 1 ? Math.sqrt(Math.max(0, 1 - (1 - ds) * (1 - ds))) : 1;
  const t = clamp((y - KEEL) / 5, 0, 1);
  const bilge = 0.5 + 0.5 * Math.sqrt(1 - (1 - t) * (1 - t));
  return 14.1 * fb * fs * bilge;
}

/** Width envelope for the whole ship (hull + superstructure); used for class zones. */
export function shipHalfWidth(x, y) {
  const s = sheer(x);
  if (y <= s) return halfBeam(clamp(x, sternX(y) + 0.01, bowX(y) - 0.01), y);
  if (x > -84 && x < 78) {
    if (y <= 14.6) return 13.9;
    if (y <= 17.4) return 12.9;
    return 7.8;
  }
  return halfBeam(x, s);
}

function buildHullGeometry() {
  const NS = 300, NR = 34;
  const pos = [];
  const idx = [];
  const rowLen = NR + 2;
  for (const side of [1, -1]) {
    const base = pos.length / 3;
    for (let i = 0; i <= NS; i++) {
      // sine spacing: stations crowd toward bow and stern where curvature is high
      const s = (i / NS) * 2 - 1;
      const x = 134.7 * Math.sin((s * Math.PI) / 2);
      const top = sheer(x);
      const xk = clamp(x, sternX(KEEL), bowX(KEEL));
      pos.push(xk, KEEL, 0); // keel centreline closes the bottom
      for (let j = 0; j <= NR; j++) {
        const v = j / NR;
        const y = KEEL + (top - KEEL) * v;
        const xe = clamp(x, sternX(y), bowX(y));
        pos.push(xe, y, side * halfBeam(xe, y));
      }
    }
    for (let i = 0; i < NS; i++) for (let j = 0; j < rowLen - 1; j++) {
      const a = base + i * rowLen + j, b = base + (i + 1) * rowLen + j, c = b + 1, d = a + 1;
      if (side > 0) idx.push(a, b, c, a, c, d); else idx.push(a, c, b, a, d, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function buildDeckGeometry() {
  const NS = 200;
  const pos = [];
  const idx = [];
  for (let i = 0; i <= NS; i++) {
    const s = (i / NS) * 2 - 1;
    const x = 134.7 * Math.sin((s * Math.PI) / 2);
    const y = sheer(x) - 0.02;
    const xe = clamp(x, sternX(y), bowX(y));
    const hb = halfBeam(xe, y);
    pos.push(xe, y, hb, xe, y, -hb);
  }
  for (let i = 0; i < NS; i++) {
    const a = i * 2, d = i * 2 + 1, b = (i + 1) * 2, c = (i + 1) * 2 + 1;
    idx.push(a, b, d, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Hull paint scheme in the fragment shader: red below the waterline, black topsides, gold sheer line. */
function hullMaterial() {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.35 });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vLocal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocal = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vLocal;')
      .replace('vec4 diffuseColor = vec4( diffuse, opacity );', /* glsl */ `
        float ux = clamp(vLocal.x, -134.5, 134.5) / 134.5;
        float shr = 11.2 + (ux > 0.0 ? 3.4 : 1.6) * ux * ux;
        vec3 hc = vec3(0.016, 0.017, 0.021);
        hc = mix(hc, vec3(0.28, 0.04, 0.03), 1.0 - smoothstep(-0.32, -0.18, vLocal.y));
        float gold = smoothstep(shr - 1.32, shr - 1.24, vLocal.y) * (1.0 - smoothstep(shr - 1.06, shr - 0.98, vLocal.y));
        hc = mix(hc, vec3(0.62, 0.43, 0.14), gold);
        float plate = smoothstep(0.965, 0.985, fract(vLocal.x / 9.0)) * (1.0 - gold);
        hc *= 1.0 - plate * 0.35;
        vec4 diffuseColor = vec4(hc, opacity);`);
  };
  return m;
}

// ------------------------------------------------------- superstructure ---
function roundedPlan(x0, x1, hw, rf, rb) {
  const s = new THREE.Shape();
  s.moveTo(x0 + rb, -hw);
  s.lineTo(x1 - rf, -hw);
  if (rf > 0) s.quadraticCurveTo(x1, -hw, x1, -hw + rf);
  s.lineTo(x1, hw - rf);
  if (rf > 0) s.quadraticCurveTo(x1, hw, x1 - rf, hw);
  s.lineTo(x0 + rb, hw);
  if (rb > 0) s.quadraticCurveTo(x0, hw, x0, hw - rb);
  s.lineTo(x0, -hw + rb);
  if (rb > 0) s.quadraticCurveTo(x0, -hw, x0 + rb, -hw);
  return s;
}

function block(x0, x1, hw, y0, y1, rf, rb, mats) {
  const g = new THREE.ExtrudeGeometry(roundedPlan(x0, x1, hw, rf, rb), { depth: y1 - y0, bevelEnabled: false, curveSegments: 8 });
  g.rotateX(-Math.PI / 2); // extrusion (+z) becomes +y
  g.translate(0, y0, 0);
  return new THREE.Mesh(g, mats);
}

// --------------------------------------------------------------- helpers ---
function placeRaked(obj, x, y, rake) {
  obj.position.set(x, y, 0);
  obj.rotation.z = rake;
}
const RAKE = 0.075; // funnels & masts lean aft

export const FUNNEL_X = [46, 22, -2, -26];
const FUNNEL_BASE = 17.0;
const FUNNEL_H = 26.5;
const FUNNEL_RX = 3.75;
const FUNNEL_RZ = 2.95;

/** Top centre of each funnel in ship-local coordinates (for smoke). */
export function funnelTop(i) {
  const x = FUNNEL_X[i];
  return new THREE.Vector3(x - Math.sin(RAKE) * FUNNEL_H, FUNNEL_BASE + Math.cos(RAKE) * FUNNEL_H, 0);
}

// ------------------------------------------------------------ lifeboats ---
// 14 standard boats + 2 emergency cutters in davits (8 a side), plus 4 collapsibles.
export const BOATS = (() => {
  const list = [];
  const xs = [57, 48, 39, 30, -40, -49, -58, -67];
  for (const side of [1, -1]) {
    xs.forEach((x, k) => list.push({ x, y: 19.2, z: side * 12.0, len: k === 0 ? 7.8 : 9.0, kind: k === 0 ? 'cutter' : 'standard', side }));
  }
  list.push({ x: 53.5, y: 17.9, z: 9.6, len: 8.2, kind: 'collapsible', side: 1 });
  list.push({ x: 53.5, y: 17.9, z: -9.6, len: 8.2, kind: 'collapsible', side: -1 });
  list.push({ x: 50.5, y: 20.35, z: 4.6, len: 8.2, kind: 'collapsible', side: 1 });
  list.push({ x: 50.5, y: 20.35, z: -4.6, len: 8.2, kind: 'collapsible', side: -1 });
  return list;
})();

export function lifeboatGeometry() {
  const g = new THREE.SphereGeometry(1, 18, 10);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i);
    if (y > 0) p.setY(i, y * 0.22); // flattened, canvas-covered top
  }
  g.computeVertexNormals();
  return g;
}

export function boatMatrix(b, out = new THREE.Matrix4(), extraScale = 1) {
  const h = b.kind === 'collapsible' ? 0.55 : 1.0;
  const w = b.kind === 'cutter' ? 1.2 : 1.35;
  return out.compose(
    new THREE.Vector3(b.x, b.y, b.z),
    new THREE.Quaternion(),
    new THREE.Vector3((b.len / 2) * extraScale, h * extraScale, w * extraScale),
  );
}

// -------------------------------------------------------------- windows ---
function windowMaterial(env, uPower) {
  const m = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = env.uTime;
    sh.uniforms.uPower = uPower;
    sh.uniforms.uWindowGlow = env.uWindowGlow;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uTime;\nvarying float vFlick;')
      .replace('#include <begin_vertex>', /* glsl */ `#include <begin_vertex>
        float iid = float(gl_InstanceID);
        vFlick = 0.9 + 0.1 * sin(uTime * (1.1 + mod(iid, 5.0) * 0.41) + iid * 1.618);`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uPower;\nuniform float uWindowGlow;\nvarying float vFlick;')
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb *= vFlick * uPower * uWindowGlow;');
  };
  m.customProgramCacheKey = () => 'ship3d-window';
  return m;
}

function buildWindows(env, uPower) {
  const rnd = mulberry32(42);
  const port = []; // hull portholes
  const rect = []; // superstructure windows
  const lamps = []; // deck lamps (tiny spheres)
  const dummy = new THREE.Object3D();

  const warm = () => {
    const off = rnd() < 0.12;
    const k = off ? 0.06 : 0.95 + rnd() * 0.6;
    const c = new THREE.Color().setHSL(0.09 + rnd() * 0.035, 0.85, 0.62);
    return c.multiplyScalar(k);
  };

  // portholes along the hull on four decks
  for (const y of [2.8, 5.1, 7.4, 9.5]) {
    for (let x = -124; x <= 124; x += 2.3) {
      if (rnd() < 0.08) continue; // irregular gaps
      if (y > sheer(x) - 1.6) continue;
      const hb = halfBeam(x, y);
      if (hb < 5) continue;
      const slope = (halfBeam(x + 0.5, y) - halfBeam(x - 0.5, y));
      for (const side of [1, -1]) {
        dummy.position.set(x, y, side * (hb + 0.06));
        // orient outward along the local surface normal
        const n = new THREE.Vector3(-slope, 0, side).normalize();
        dummy.lookAt(dummy.position.clone().add(n));
        dummy.updateMatrix();
        port.push({ m: dummy.matrix.clone(), c: warm() });
      }
    }
  }
  // B deck (black plating) large windows, A deck promenade, boat-deck houses
  const rows = [
    { y: 12.7, x0: -80, x1: 74, step: 3.2, z: 13.95 },
    { y: 16.0, x0: -76, x1: 66, step: 2.9, z: 12.95 },
    { y: 18.8, x0: -70, x1: 56, step: 3.6, z: 7.85 },
  ];
  for (const r of rows) {
    for (let x = r.x0; x <= r.x1; x += r.step) {
      if (rnd() < 0.06) continue;
      for (const side of [1, -1]) {
        dummy.position.set(x, r.y, side * r.z);
        dummy.rotation.set(0, side > 0 ? 0 : Math.PI, 0);
        dummy.updateMatrix();
        rect.push({ m: dummy.matrix.clone(), c: warm() });
      }
    }
  }
  // bridge / wheelhouse front windows
  for (let z = -5; z <= 5; z += 1.4) {
    dummy.position.set(64.06, 19.3, z);
    dummy.rotation.set(0, Math.PI / 2, 0);
    dummy.updateMatrix();
    rect.push({ m: dummy.matrix.clone(), c: warm().multiplyScalar(0.6) });
  }
  // superstructure fronts (A deck + B deck) facing forward
  for (const [fx, fy, hw] of [[70.02, 16.0, 9], [78.02, 12.7, 10]]) {
    for (let z = -hw; z <= hw; z += 2.4) {
      dummy.position.set(fx, fy, z);
      dummy.rotation.set(0, Math.PI / 2, 0);
      dummy.updateMatrix();
      rect.push({ m: dummy.matrix.clone(), c: warm() });
    }
  }
  // deck lamps along the boat deck edge + masthead / stern lights
  for (let x = -76; x <= 64; x += 10) {
    for (const side of [1, -1]) lamps.push({ p: [x, 18.3, side * 12.75], c: new THREE.Color('#ffd9a0').multiplyScalar(3) });
  }

  const mat = windowMaterial(env, uPower);
  const make = (geo, list) => {
    const im = new THREE.InstancedMesh(geo, mat, Math.max(1, list.length));
    list.forEach((w, i) => { im.setMatrixAt(i, w.m); im.setColorAt(i, w.c); });
    im.count = list.length;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    return im;
  };
  const portholes = make(new THREE.CircleGeometry(0.34, 10), port);
  const rects = make(new THREE.PlaneGeometry(1.15, 0.95), rect);
  const lampList = lamps.map((l) => {
    dummy.position.set(...l.p); dummy.rotation.set(0, 0, 0); dummy.updateMatrix();
    return { m: dummy.matrix.clone(), c: l.c };
  });
  const lampMesh = make(new THREE.SphereGeometry(0.28, 8, 6), lampList);
  return { portholes, rects, lampMesh, count: port.length + rect.length };
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- smoke ---
class Smoke {
  constructor(env) {
    this.perFunnel = 46;
    this.N = this.perFunnel * 4;
    this.rnd = mulberry32(7);
    this.parts = [];
    const pos = new Float32Array(this.N * 3);
    const size = new Float32Array(this.N);
    const alpha = new Float32Array(this.N);
    const seed = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) {
      const f = Math.floor(i / this.perFunnel);
      const p = { f, age: 0, life: 1, pos: new THREE.Vector3(), vel: new THREE.Vector3(), alive: false };
      // stagger initial ages so plumes are already formed on first frame
      this._spawn(p, 1);
      p.age = this.rnd() * p.life;
      p.pos.addScaledVector(p.vel, p.age);
      this.parts.push(p);
      seed[i] = this.rnd();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aAlpha', new THREE.BufferAttribute(alpha, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    this.geo = g;
    const mat = new THREE.ShaderMaterial({
      uniforms: { uSmokeColor: env.uSmokeColor, uViewportH: env.uViewportH, uTime: env.uTime },
      vertexShader: /* glsl */ `
        uniform float uViewportH;
        attribute float aSize;
        attribute float aAlpha;
        attribute float aSeed;
        varying float vAlpha;
        varying float vSeed;
        void main() {
          vAlpha = aAlpha;
          vSeed = aSeed;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(aSize * uViewportH * projectionMatrix[1][1] * 0.5 / -mv.z, 1.0, 512.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uSmokeColor;
        uniform float uTime;
        varying float vAlpha;
        varying float vSeed;
        void main() {
          vec2 q = gl_PointCoord - 0.5;
          float d = length(q);
          float a = smoothstep(0.5, 0.05, d);
          float wob = 0.75 + 0.25 * sin(q.x * 9.0 + vSeed * 30.0 + uTime * 0.6) * sin(q.y * 7.0 - vSeed * 20.0);
          gl_FragColor = vec4(uSmokeColor, a * a * wob * vAlpha);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true,
      depthWrite: false,
      fog: false,
    });
    this.points = new THREE.Points(g, mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 2;
    this.emission = 1;
    this.wind = 1;
  }

  _spawn(p, wind) {
    const top = funnelTop(p.f);
    p.pos.set(top.x + (this.rnd() - 0.5) * 3, top.y - 0.5, (this.rnd() - 0.5) * 2.4);
    p.vel.set(-2.0 - wind * (7 + this.rnd() * 3), 2.4 + this.rnd() * 1.2, (this.rnd() - 0.5) * 1.4);
    p.life = 9 + this.rnd() * 5;
    p.age = 0;
    // the 4th funnel was mostly a ventilator: barely any smoke
    p.weight = p.f === 3 ? 0.25 : 1;
    p.alive = true;
  }

  update(dt) {
    const pos = this.geo.attributes.position.array;
    const size = this.geo.attributes.aSize.array;
    const alpha = this.geo.attributes.aAlpha.array;
    for (let i = 0; i < this.N; i++) {
      const p = this.parts[i];
      p.age += dt;
      if (p.age >= p.life) {
        if (this.rnd() < this.emission) this._spawn(p, this.wind);
        else { p.alive = false; p.age = 0; p.life = 0.5 + this.rnd(); }
      }
      p.pos.addScaledVector(p.vel, dt);
      p.vel.y *= 1 - 0.08 * dt;
      const k = p.age / p.life;
      pos[i * 3] = p.pos.x; pos[i * 3 + 1] = p.pos.y; pos[i * 3 + 2] = p.pos.z;
      size[i] = 7 + k * 34;
      alpha[i] = p.alive ? Math.min(1, p.age * 0.6) * (1 - k) * (1 - k) * 0.3 * p.weight : 0;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.aSize.needsUpdate = true;
    this.geo.attributes.aAlpha.needsUpdate = true;
  }
}

// ================================================================= ship ===
export function buildShip(env) {
  const root = new THREE.Group();   // moved by bobbing / timeline
  root.name = 'ship';

  const uPower = { value: 1 };     // window light power (timeline: lights fail)

  // materials
  const hullMat = hullMaterial();
  const black = new THREE.MeshStandardMaterial({ color: 0x0a0b0e, roughness: 0.42, metalness: 0.35 });
  const white = new THREE.MeshStandardMaterial({ color: 0xe9e3d6, roughness: 0.62, metalness: 0.0 });
  const wood = new THREE.MeshStandardMaterial({ color: 0x7d6a52, roughness: 0.9 });
  const buff = new THREE.MeshStandardMaterial({ color: 0xc9a96e, roughness: 0.6, metalness: 0.05 });
  const funnelBlack = new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.5, metalness: 0.2 });
  const gold = new THREE.MeshStandardMaterial({ color: 0xc9a24e, roughness: 0.35, metalness: 0.6 });
  const mastMat = new THREE.MeshStandardMaterial({ color: 0x8a6a45, roughness: 0.7 });
  const domeMat = new THREE.MeshStandardMaterial({ color: 0x2a2a30, roughness: 0.2, metalness: 0.1, emissive: new THREE.Color('#ffc77a'), emissiveIntensity: 0.9 });

  // hull + main deck
  const hull = new THREE.Mesh(buildHullGeometry(), hullMat);
  const deck = new THREE.Mesh(buildDeckGeometry(), wood);
  root.add(hull, deck);

  // superstructure (ExtrudeGeometry groups: 0 = caps (deck tops), 1 = sides)
  const parts = new THREE.Group();
  root.add(parts);
  parts.add(block(-84, 78, 13.9, 10.4, 14.6, 7, 4, [wood, black]));      // B deck, black plated
  parts.add(block(-83.9, 77.9, 13.95, 14.15, 14.4, 7, 4, [gold, gold]));  // gold rail stripe
  parts.add(block(-80, 70, 12.9, 14.6, 17.4, 6, 3, [wood, white]));      // A deck / promenade
  parts.add(block(-72, 58, 7.6, 17.4, 20.0, 2.5, 1.5, [white, white])); // boat-deck houses
  parts.add(block(58, 64, 6.0, 17.4, 20.6, 1.2, 0, [white, white]));    // wheelhouse / bridge
  parts.add(block(60.5, 63, 13.6, 19.7, 20.1, 0, 0, [white, white]));   // bridge wings
  parts.add(block(-128, -112, 5.5, 12.4, 14.8, 2.5, 3, [white, white])); // poop deckhouse
  parts.add(block(-121.4, -119.6, 13.2, 16.0, 16.5, 0, 0, [white, white])); // docking bridge
  parts.add(block(108, 114, 4, 13.8, 15.6, 1, 1, [white, white]));     // forecastle house
  // posts for docking bridge
  for (const z of [-12.5, 12.5]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 3.8, 6), white);
    post.position.set(-120.5, 14.3, z);
    parts.add(post);
  }

  // first-class grand staircase glass domes
  for (const x of [34, -52]) {
    const dome = new THREE.Mesh(new THREE.SphereGeometry(2.8, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2), domeMat);
    dome.scale.set(1.25, 0.7, 1);
    dome.position.set(x, 20.0, 0);
    parts.add(dome);
  }

  // ventilator cowls on the boat deck
  const cowlGeo = new THREE.CylinderGeometry(0.55, 0.65, 2.6, 10);
  const cowlTopGeo = new THREE.SphereGeometry(0.8, 10, 8);
  for (const x of [52, 40, 28, 10, -8, -18, -38, -48, -62]) {
    for (const side of [1, -1]) {
      const c = new THREE.Mesh(cowlGeo, buff);
      c.position.set(x, 18.7, side * 9.3);
      const t = new THREE.Mesh(cowlTopGeo, buff);
      t.position.set(x + 0.3, 20.1, side * 9.3);
      parts.add(c, t);
    }
  }

  // funnels
  const funnelGeo = new THREE.CylinderGeometry(1, 1, FUNNEL_H, 40, 1, true);
  funnelGeo.translate(0, FUNNEL_H / 2, 0);
  const bandGeo = new THREE.CylinderGeometry(1.015, 1.015, 4.3, 40, 1, true);
  bandGeo.translate(0, FUNNEL_H - 2.15, 0);
  const capGeo = new THREE.CircleGeometry(1, 40);
  capGeo.rotateX(-Math.PI / 2);
  capGeo.translate(0, FUNNEL_H - 1.4, 0);
  const lipGeo = new THREE.TorusGeometry(1.0, 0.06, 6, 40);
  lipGeo.rotateX(Math.PI / 2);
  lipGeo.translate(0, FUNNEL_H, 0);
  const pipeGeo = new THREE.CylinderGeometry(0.32, 0.32, FUNNEL_H * 0.95, 8);
  pipeGeo.translate(0, FUNNEL_H * 0.475 + 0.5, 0);
  FUNNEL_X.forEach((x) => {
    const f = new THREE.Group();
    const scale = new THREE.Vector3(FUNNEL_RX, 1, FUNNEL_RZ);
    for (const [geo, mat] of [[funnelGeo, buff], [bandGeo, funnelBlack], [capGeo, funnelBlack], [lipGeo, funnelBlack]]) {
      const m = new THREE.Mesh(geo, mat);
      m.scale.copy(scale);
      f.add(m);
    }
    const pipe = new THREE.Mesh(pipeGeo, buff);
    pipe.position.set(FUNNEL_RX + 0.25, 0, 0);
    f.add(pipe);
    placeRaked(f, x, FUNNEL_BASE, RAKE);
    parts.add(f);
  });

  // masts (raked aft like the funnels) + crow's nest
  const mastTops = [];
  for (const [x, h] of [[98, 49], [-104, 44]]) {
    const base = sheer(x) - 0.2;
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.55, h, 8), mastMat);
    m.geometry.translate(0, h / 2, 0);
    placeRaked(m, x, base, RAKE);
    parts.add(m);
    mastTops.push(new THREE.Vector3(x - Math.sin(RAKE) * h, base + Math.cos(RAKE) * h, 0));
    if (x > 0) {
      const nest = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 1.4, 10), white);
      nest.position.set(x - Math.sin(RAKE) * 28, base + Math.cos(RAKE) * 28, 0);
      parts.add(nest);
    }
  }

  // rigging & Marconi aerial (thin lines)
  const rig = [];
  const [fore, main] = mastTops;
  const L = (a, b) => rig.push(a.x, a.y, a.z, b.x, b.y, b.z);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  L(fore, V(bowX(15) - 0.5, 15.0, 0));
  L(main, V(-133.5, sheer(-133) + 0.3, 0));
  for (const side of [1, -1]) {
    for (const dx of [-4, 0, 4]) {
      L(fore, V(96 + dx, sheer(96 + dx), side * (halfBeam(96 + dx, sheer(96 + dx)) - 0.3)));
      L(main, V(-104 + dx, sheer(-104 + dx), side * (halfBeam(-104 + dx, sheer(-104)) - 0.3)));
    }
    L(V(fore.x, fore.y - 1, side * 0.9), V(main.x, main.y - 1, side * 0.9)); // aerial wires
  }
  // railings along the sheer and the boat deck
  const railY = 1.1;
  for (const side of [1, -1]) {
    let prev = null;
    for (let i = 0; i <= 120; i++) {
      const s = (i / 120) * 2 - 1;
      const x = 134.0 * Math.sin((s * Math.PI) / 2);
      const y = sheer(x) + railY;
      const p = V(clamp(x, sternX(y), bowX(sheer(x))), y, side * (halfBeam(clamp(x, sternX(sheer(x)), bowX(sheer(x))), sheer(x)) - 0.1));
      if (prev) L(prev, p);
      prev = p;
    }
    L(V(-80, 17.4 + railY, side * 12.8), V(66, 17.4 + railY, side * 12.8));
  }
  const rigGeo = new THREE.BufferGeometry();
  rigGeo.setAttribute('position', new THREE.Float32BufferAttribute(rig, 3));
  const rigging = new THREE.LineSegments(rigGeo, new THREE.LineBasicMaterial({ color: 0x8a93a6, transparent: true, opacity: 0.22 }));
  parts.add(rigging);

  // lifeboats (instanced so the timeline can lower them one by one)
  const boatGeo = lifeboatGeometry();
  const boatMat = new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.55 });
  const boats = new THREE.InstancedMesh(boatGeo, boatMat, BOATS.length);
  const tmpM = new THREE.Matrix4();
  BOATS.forEach((b, i) => boats.setMatrixAt(i, boatMatrix(b, tmpM)));
  boats.instanceMatrix.needsUpdate = true;
  root.add(boats);

  // pulsing outline for the "Lifeboats" toggle (inverted-hull technique)
  const outlineMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#e3b866').multiplyScalar(3.0), side: THREE.BackSide, transparent: true, opacity: 0.9, toneMapped: false, depthWrite: false });
  const outline = new THREE.InstancedMesh(boatGeo, outlineMat, BOATS.length);
  BOATS.forEach((b, i) => outline.setMatrixAt(i, boatMatrix(b, tmpM, 1.45)));
  outline.instanceMatrix.needsUpdate = true;
  outline.visible = false;
  outline.renderOrder = 3;
  root.add(outline);

  // brass ring markers above each boat so the highlight reads from far away
  const markerPos = new Float32Array(BOATS.length * 3);
  BOATS.forEach((b, i) => markerPos.set([b.x, b.y + 3.2, b.z], i * 3));
  const markerGeo = new THREE.BufferGeometry();
  markerGeo.setAttribute('position', new THREE.BufferAttribute(markerPos, 3));
  const markerMat = new THREE.ShaderMaterial({
    uniforms: { uTime: env.uTime, uPixelRatio: env.uPixelRatio, uOpacity: { value: 1 } },
    vertexShader: /* glsl */ `
      uniform float uTime;
      uniform float uPixelRatio;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = (13.0 + 3.0 * sin(uTime * 3.2)) * uPixelRatio;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uOpacity;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float ring = smoothstep(0.5, 0.42, d) * smoothstep(0.26, 0.34, d);
        float ctr = smoothstep(0.16, 0.08, d);
        gl_FragColor = vec4(vec3(0.89, 0.72, 0.40) * 1.6, (ring + ctr) * uOpacity);
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    toneMapped: false,
    fog: false,
  });
  const markers = new THREE.Points(markerGeo, markerMat);
  markers.renderOrder = 8;
  markers.frustumCulled = false;
  markers.visible = false;
  outline.add(markers); // shown/hidden together with the outline

  // windows & lamps
  const win = buildWindows(env, uPower);
  root.add(win.portholes, win.rects, win.lampMesh);

  // masthead light & stern light (bright points for bloom)
  const navMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#fff4e0').multiplyScalar(4), toneMapped: false });
  const masthead = new THREE.Mesh(new THREE.SphereGeometry(0.45, 8, 6), navMat);
  masthead.position.set(fore.x + 0.4, fore.y - 12, 0);
  const sternLight = new THREE.Mesh(new THREE.SphereGeometry(0.4, 8, 6), navMat);
  sternLight.position.set(-133.8, sheer(-134) + 1.6, 0);
  root.add(masthead, sternLight);

  // warm interior glow lighting the superstructure a little
  const glowLights = [];
  for (const [x, y, z] of [[30, 16, 18], [-30, 16, -18], [-40, 16, 18], [40, 16, -18]]) {
    const l = new THREE.PointLight(0xffb066, 18, 60, 1.6);
    l.position.set(x, y, z);
    root.add(l);
    glowLights.push(l);
  }

  // smoke
  const smoke = new Smoke(env);
  root.add(smoke.points);

  const hiddenM = new THREE.Matrix4().makeScale(0, 0, 0);

  return {
    root,
    smoke,
    boats,
    outline,
    markers,
    uPower,
    windowCount: win.count,
    /** 0..1 – window/lamp brightness (lights failing at 02:18). */
    setLights(power) {
      uPower.value = power;
      domeMat.emissiveIntensity = 0.9 * power;
      navMat.color.set('#fff4e0').multiplyScalar(4 * Math.max(power, 0.0));
      for (const l of glowLights) l.intensity = 18 * power;
    },
    /** Hide/show lifeboat i in its davits (launched boats move to the water). */
    setBoatAboard(i, aboard) {
      boats.setMatrixAt(i, aboard ? boatMatrix(BOATS[i], tmpM) : hiddenM);
      outline.setMatrixAt(i, aboard ? boatMatrix(BOATS[i], tmpM, 1.45) : hiddenM);
      boats.instanceMatrix.needsUpdate = true;
      outline.instanceMatrix.needsUpdate = true;
    },
    update(dt) { smoke.update(dt); },
  };
}
