// Stage: renderer, camera, controls, sky (gradient + stars + milky way + aurora),
// lights, fog, time-of-day presets, quality handling, resize and pause logic.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ---------------------------------------------------------------------------
// Shared GLSL: sky colour as a function of view direction. Used by the sky dome
// and by the ocean (reflections + horizon fog) so the horizon blends seamlessly.
// ---------------------------------------------------------------------------
export const SKY_GLSL = /* glsl */ `
uniform vec3 uZenith;
uniform vec3 uMid;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uSunGlow;
vec3 skyColor(vec3 dir) {
  float hp = max(dir.y, 0.0);
  vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 0.22, hp));
  c = mix(c, uZenith, smoothstep(0.16, 0.95, hp));
  vec3 L = normalize(uSunDir);
  float sd = max(dot(dir, L), 0.0);
  vec2 hd = normalize(dir.xz + 1e-5);
  vec2 hl = normalize(L.xz + 1e-5);
  float horiz = pow(max(dot(hd, hl), 0.0), 3.0) * exp(-hp * 7.0);
  c += uSunColor * (pow(sd, 7.0) * 0.30 + pow(sd, 80.0) * 0.55 + horiz * 0.45) * uSunGlow;
  return c;
}
`;

export const NOISE_GLSL = /* glsl */ `
float hash12(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash13(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 x) {
  vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float fbm(vec3 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
`;

// ---------------------------------------------------------------------------
// Time-of-day presets (colours are sRGB hex; converted to linear by THREE.Color)
// ---------------------------------------------------------------------------
const PRESETS = {
  // Historically a clear, moonless, starlit night: no moon disc, faint horizon glow.
  night: {
    zenith: '#010309', mid: '#061331', horizon: '#1b3263',
    sunDir: [-0.42, 0.05, -1.0], sunColor: '#5d78b8', sunGlow: 0.55, sunDisc: 0.0,
    stars: 1.0, milky: 1.0, aurora: 1.0,
    deep: '#00030a', shallow: '#071b38', foam: '#4d5d7d',
    specPow: 70, specStrength: 0.45, fogDensity: 0.00085,
    hemiSky: '#4664a8', hemiGround: '#04060c', hemiInt: 0.7,
    keyColor: '#a8bde8', keyInt: 1.05, keyDir: [0.45, 0.75, 0.6],
    rimColor: '#6f8fd6', rimInt: 0.9,
    windowGlow: 1.0, smoke: '#36405a', bloom: 0.55, ice: 0.32, iceGlow: 0.22,
  },
  // Dusk: low sun/moon near the horizon with a strong glint path.
  dusk: {
    zenith: '#0d1838', mid: '#3e3a6e', horizon: '#e48a5e',
    sunDir: [-0.45, 0.045, -1.0], sunColor: '#ffb77a', sunGlow: 1.0, sunDisc: 1.0,
    stars: 0.28, milky: 0.0, aurora: 0.0,
    deep: '#07122a', shallow: '#2a3460', foam: '#c2b0aa',
    specPow: 260, specStrength: 3.2, fogDensity: 0.0008,
    hemiSky: '#8a86c4', hemiGround: '#1a1018', hemiInt: 0.7,
    keyColor: '#ffc59a', keyInt: 0.9, keyDir: [0.35, 0.55, 0.75],
    rimColor: '#ffa36b', rimInt: 1.6,
    windowGlow: 0.8, smoke: '#5a4e57', bloom: 0.6, ice: 0.12, iceGlow: 0.10,
  },
  day: {
    zenith: '#1f5aa8', mid: '#4f8ccc', horizon: '#c4d9ec',
    sunDir: [0.35, 0.62, -0.7], sunColor: '#fff2d6', sunGlow: 0.6, sunDisc: 1.0,
    stars: 0.0, milky: 0.0, aurora: 0.0,
    deep: '#062c4f', shallow: '#1c6c8d', foam: '#f2f6fa',
    specPow: 400, specStrength: 4.0, fogDensity: 0.0005,
    hemiSky: '#cfe3ff', hemiGround: '#2a3a4a', hemiInt: 1.1,
    keyColor: '#fff4e0', keyInt: 2.4, keyDir: [0.35, 0.62, -0.7],
    rimColor: '#bcd6ff', rimInt: 0.5,
    windowGlow: 0.3, smoke: '#7d7a78', bloom: 0.25, ice: 0.04, iceGlow: 0.0,
  },
};

const COLOR_KEYS = ['zenith', 'mid', 'horizon', 'sunColor', 'deep', 'shallow', 'foam', 'hemiSky', 'hemiGround', 'keyColor', 'rimColor', 'smoke'];
const NUM_KEYS = ['sunGlow', 'sunDisc', 'stars', 'milky', 'aurora', 'specPow', 'specStrength', 'fogDensity', 'hemiInt', 'keyInt', 'rimInt', 'windowGlow', 'bloom', 'ice', 'iceGlow'];
const VEC_KEYS = ['sunDir', 'keyDir'];

function parsePreset(p) {
  const out = {};
  for (const k of COLOR_KEYS) out[k] = new THREE.Color(p[k]);
  for (const k of NUM_KEYS) out[k] = p[k];
  for (const k of VEC_KEYS) out[k] = new THREE.Vector3(...p[k]).normalize();
  return out;
}
const PARSED = Object.fromEntries(Object.entries(PRESETS).map(([k, v]) => [k, parsePreset(v)]));

// ---------------------------------------------------------------------------
// Camera presets (ship lies along +X, bow at +X, waterline y = 0, port side +Z)
// ---------------------------------------------------------------------------
// Offsets are tuned for a 2:1 frame; narrower frames pull the camera back.
export const CAMERA_PRESETS = {
  side:    { pos: [-22, 34, 258],  target: [0, 16, 0], label: 'Side' },
  bow:     { pos: [215, 30, 92],   target: [30, 16, 0], label: 'Bow' },
  stern:   { pos: [-215, 40, -95], target: [-25, 16, 0], label: 'Stern' },
  top:     { pos: [0, 430, 18],    target: [0, 0, 0], label: 'Top' },
  iceberg: { pos: [540, 30, 120],  target: [0, 16, 0], label: 'From the iceberg', maxScale: 1.12 },
  quarter: { pos: [165, 62, 205],  target: [-8, 14, 0], label: 'Three-quarter' },
};

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && c.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

export function pickAutoQuality() {
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 8;
  const small = Math.min(window.screen?.width || 1280, window.screen?.height || 800) < 600;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent || '');
  return (cores < 4 || mem < 4 || small || mobile) ? 'low' : 'high';
}

export class Stage {
  constructor(host, { quality = 'auto' } = {}) {
    this.host = host;
    this.qualitySetting = quality;
    this.quality = quality === 'auto' ? pickAutoQuality() : quality;
    this.updaters = [];
    this.time = 0;
    this.running = false;
    this.visible = true;
    this.onScreen = true;
    this.cameraFlight = null;
    this.shipBox = new THREE.Box3(new THREE.Vector3(-142, -14, -19), new THREE.Vector3(142, 47, 19));
    this.shipBoxEnabled = true;

    // --- renderer -----------------------------------------------------------
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(this._dpr());
    this.renderer.toneMapping = THREE.NeutralToneMapping; // keeps data colours faithful
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute('tabindex', '0');

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x000000, 0.0009);
    this.camera = new THREE.PerspectiveCamera(38, 2, 1, 22000);
    this.camera.position.set(...CAMERA_PRESETS.side.pos);

    // --- shared uniforms (referenced by many materials) --------------------
    this.env = {
      uTime: { value: 0 },
      uZenith: { value: new THREE.Color() },
      uMid: { value: new THREE.Color() },
      uHorizon: { value: new THREE.Color() },
      uSunDir: { value: new THREE.Vector3(0, 0.1, -1) },
      uSunColor: { value: new THREE.Color() },
      uSunGlow: { value: 1 },
      uSunDisc: { value: 0 },
      uStars: { value: 1 },
      uMilky: { value: 1 },
      uAurora: { value: 1 },
      uDeep: { value: new THREE.Color() },
      uShallow: { value: new THREE.Color() },
      uFoamColor: { value: new THREE.Color() },
      uSpecPow: { value: 100 },
      uSpecStrength: { value: 1 },
      uFogDensity: { value: 0.0009 },
      uWindowGlow: { value: 1 },
      uSmokeColor: { value: new THREE.Color() },
      uPixelRatio: { value: this.renderer.getPixelRatio() },
      uViewportH: { value: 500 },
    };

    // --- controls -----------------------------------------------------------
    const c = new OrbitControls(this.camera, this.renderer.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.07;
    c.rotateSpeed = 0.55;
    c.zoomSpeed = 0.9;
    c.panSpeed = 0.7;
    c.minDistance = 45;
    c.maxDistance = 950;
    c.maxPolarAngle = 1.5; // never look from below the waterline
    c.autoRotateSpeed = 0.35;
    c.target.set(...CAMERA_PRESETS.side.target);
    this.controls = c;

    this._buildSky();
    this._buildLights();

    // --- time of day --------------------------------------------------------
    this.tod = 'night';
    this.cur = parsePreset(PRESETS.night);
    this.todFrom = null;
    this.todTo = null;
    this.todK = 1;
    this._applyPreset(this.cur);

    // --- post-processing ----------------------------------------------------
    this.composer = null;
    this.bloom = null;
    this._setupComposer();

    // --- resize / visibility -----------------------------------------------
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(host);
    this.resize();
    document.addEventListener('visibilitychange', () => {
      this.visible = document.visibilityState !== 'hidden';
      this._syncRunning();
    });
    if ('IntersectionObserver' in window) {
      this._io = new IntersectionObserver((entries) => {
        for (const e of entries) this.onScreen = e.isIntersecting;
        this._syncRunning();
      });
      this._io.observe(host);
    }

    // adaptive quality (auto only): drop to low if frames are slow
    this._perf = { frames: 0, acc: 0, checked: false };
    this._loop = this._loop.bind(this);
    this._clock = new THREE.Clock();
  }

  _dpr() {
    const dpr = window.devicePixelRatio || 1;
    return this.quality === 'high' ? Math.min(dpr, 2) : 1;
  }

  setQuality(q) {
    this.qualitySetting = q || 'auto';
    const resolved = this.qualitySetting === 'auto' ? pickAutoQuality() : this.qualitySetting;
    if (resolved === this.quality) return;
    this.quality = resolved;
    this._perf = { frames: 0, acc: 0, checked: false };
    this.renderer.setPixelRatio(this._dpr());
    this.env.uPixelRatio.value = this.renderer.getPixelRatio();
    this._setupComposer();
    this.resize();
  }

  _setupComposer() {
    if (this.quality !== 'high') {
      if (this.composer) { this.composer.dispose?.(); }
      this.composer = null;
      this.bloom = null;
      return;
    }
    if (this.composer) return;
    try {
      const size = this.renderer.getSize(new THREE.Vector2());
      const comp = new EffectComposer(this.renderer);
      comp.addPass(new RenderPass(this.scene, this.camera));
      this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x || 1, size.y || 1), this.cur.bloom, 0.4, 0.85);
      comp.addPass(this.bloom);
      comp.addPass(new OutputPass());
      this.composer = comp;
    } catch (e) {
      console.warn('[ship3d] bloom unavailable, continuing without it', e);
      this.composer = null;
      this.bloom = null;
    }
  }

  // ------------------------------------------------------------------ sky ---
  _buildSky() {
    this.skyGroup = new THREE.Group();
    this.scene.add(this.skyGroup);

    const skyMat = new THREE.ShaderMaterial({
      uniforms: this.env,
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() {
          vDir = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uSunDisc;
        uniform float uMilky;
        uniform float uAurora;
        varying vec3 vDir;
        ${SKY_GLSL}
        ${NOISE_GLSL}
        void main() {
          vec3 dir = normalize(vDir);
          vec3 col = skyColor(dir);
          float sd = max(dot(dir, normalize(uSunDir)), 0.0);
          // sun / low moon disc with soft limb
          col += uSunColor * smoothstep(0.99935, 0.99975, sd) * uSunDisc * 5.0;
          // Milky Way: a tilted, mottled band with dark dust lanes
          if (uMilky > 0.001) {
            vec3 bn = normalize(vec3(0.42, 0.55, -0.72));
            float d = dot(dir, bn);
            float band = exp(-d * d * 12.0);
            float n = fbm(dir * 5.0);
            float lanes = smoothstep(0.42, 0.72, fbm(dir * 12.0 + 3.1)) * exp(-d * d * 40.0);
            float mw = band * (0.25 + 0.75 * n) * (1.0 - 0.7 * lanes);
            col += vec3(0.34, 0.38, 0.56) * mw * 0.10 * uMilky * smoothstep(0.0, 0.3, dir.y);
          }
          // Faint aurora low over the horizon (survivors reported one that night)
          if (uAurora > 0.001) {
            float az = atan(dir.x, -dir.z);
            float h = dir.y;
            float curtain = fbm(vec3(az * 2.5, uTime * 0.02, 0.0));
            float rays = 0.55 + 0.45 * sin(az * 70.0 + fbm(vec3(az * 6.0, uTime * 0.04, 1.0)) * 8.0);
            float top = 0.10 + curtain * 0.22;
            float band = smoothstep(0.015, 0.07, h) * (1.0 - smoothstep(top * 0.6, top, h));
            float sector = smoothstep(1.5, 0.2, abs(az - 1.9));
            vec3 ac = mix(vec3(0.12, 0.85, 0.55), vec3(0.42, 0.30, 0.85), smoothstep(0.04, 0.24, h));
            col += ac * band * sector * rays * curtain * 0.07 * uAurora;
          }
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
          gl_FragColor.rgb += (hash12(gl_FragCoord.xy) - 0.5) / 255.0; // de-band
        }`,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      fog: false,
    });
    const sky = new THREE.Mesh(new THREE.SphereGeometry(9000, 48, 24), skyMat);
    sky.renderOrder = -10;
    sky.frustumCulled = false;
    this.skyGroup.add(sky);

    // Stars: per-star size, colour and twinkle phase.
    const N = this.quality === 'high' ? 5200 : 2600;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const size = new Float32Array(N);
    const phase = new Float32Array(N);
    let s = 1234567;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const tint = [new THREE.Color('#cdd9ff'), new THREE.Color('#ffffff'), new THREE.Color('#ffe2c0'), new THREE.Color('#bcd0ff')];
    for (let i = 0; i < N; i++) {
      // uniform on the sphere, keep only above (slightly below) the horizon
      let y = rnd() * 1.06 - 0.06;
      const th = rnd() * Math.PI * 2;
      const r = Math.sqrt(1 - y * y);
      pos.set([Math.cos(th) * r * 8200, y * 8200, Math.sin(th) * r * 8200], i * 3);
      const t = tint[Math.floor(rnd() * tint.length)];
      col.set([t.r, t.g, t.b], i * 3);
      const m = rnd();
      size[i] = 0.9 + Math.pow(m, 6) * 3.6;
      phase[i] = rnd();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aPhase', new THREE.BufferAttribute(phase, 1));
    const starMat = new THREE.ShaderMaterial({
      uniforms: this.env,
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform float uStars;
        uniform float uPixelRatio;
        attribute float aSize;
        attribute float aPhase;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          vColor = color;
          float tw = 0.6 + 0.4 * sin(uTime * (0.8 + aPhase * 3.2) + aPhase * 60.0);
          float h = normalize(position).y;
          vAlpha = tw * smoothstep(-0.01, 0.12, h) * uStars;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = aSize * uPixelRatio * (0.85 + 0.3 * tw);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vColor * a * a * vAlpha * 1.3, 1.0);
          #include <colorspace_fragment>
        }`,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    const stars = new THREE.Points(g, starMat);
    stars.renderOrder = -9;
    stars.frustumCulled = false;
    this.skyGroup.add(stars);
  }

  _buildLights() {
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x000000, 0.6);
    this.key = new THREE.DirectionalLight(0xffffff, 1);
    this.rim = new THREE.DirectionalLight(0xffffff, 1);
    this.scene.add(this.hemi, this.key, this.rim, this.key.target, this.rim.target);
  }

  // ---------------------------------------------------------- time of day ---
  setTimeOfDay(name, instant = false) {
    if (!PARSED[name]) name = 'night';
    if (name === this.tod && !this.todTo) return;
    this.tod = name;
    if (instant) {
      this.cur = parsePreset(PRESETS[name]);
      this.todTo = null;
      this._applyPreset(this.cur);
      return;
    }
    this.todFrom = this._snapshot();
    this.todTo = PARSED[name];
    this.todK = 0;
  }

  _snapshot() {
    const o = {};
    for (const k of COLOR_KEYS) o[k] = this.cur[k].clone();
    for (const k of NUM_KEYS) o[k] = this.cur[k];
    for (const k of VEC_KEYS) o[k] = this.cur[k].clone();
    return o;
  }

  _blendTod(dt) {
    if (!this.todTo) return;
    this.todK = Math.min(1, this.todK + dt / 1.6);
    const k = easeInOutCubic(this.todK);
    const a = this.todFrom, b = this.todTo, c = this.cur;
    for (const key of COLOR_KEYS) c[key].copy(a[key]).lerp(b[key], k);
    for (const key of NUM_KEYS) {
      // fog density & spec power blend better geometrically
      if (key === 'specPow' || key === 'fogDensity') c[key] = a[key] * Math.pow(b[key] / a[key], k);
      else c[key] = a[key] + (b[key] - a[key]) * k;
    }
    for (const key of VEC_KEYS) c[key].copy(a[key]).lerp(b[key], k).normalize();
    this._applyPreset(c);
    if (this.todK >= 1) this.todTo = null;
  }

  _applyPreset(p) {
    const e = this.env;
    e.uZenith.value.copy(p.zenith);
    e.uMid.value.copy(p.mid);
    e.uHorizon.value.copy(p.horizon);
    e.uSunDir.value.copy(p.sunDir);
    e.uSunColor.value.copy(p.sunColor);
    e.uSunGlow.value = p.sunGlow;
    e.uSunDisc.value = p.sunDisc;
    e.uStars.value = p.stars;
    e.uMilky.value = p.milky;
    e.uAurora.value = p.aurora;
    e.uDeep.value.copy(p.deep);
    e.uShallow.value.copy(p.shallow);
    e.uFoamColor.value.copy(p.foam);
    e.uSpecPow.value = p.specPow;
    e.uSpecStrength.value = p.specStrength;
    e.uFogDensity.value = p.fogDensity;
    e.uWindowGlow.value = p.windowGlow;
    e.uSmokeColor.value.copy(p.smoke);
    this.scene.fog.color.copy(p.horizon);
    this.scene.fog.density = p.fogDensity;
    this.hemi.color.copy(p.hemiSky);
    this.hemi.groundColor.copy(p.hemiGround);
    this.hemi.intensity = p.hemiInt;
    this.key.color.copy(p.keyColor);
    this.key.intensity = p.keyInt;
    this.key.position.copy(p.keyDir).multiplyScalar(500);
    this.rim.color.copy(p.rimColor);
    this.rim.intensity = p.rimInt;
    this.rim.position.copy(p.sunDir).multiplyScalar(500);
    if (this.bloom) this.bloom.strength = p.bloom;
    if (this.onPreset) this.onPreset(p);
  }

  // --------------------------------------------------------------- camera ---
  /** Distance multiplier so the whole ship fits narrow / tall frames. */
  _fitScale() {
    const aspect = this.camera.aspect || 2;
    return THREE.MathUtils.clamp(2 / aspect, 0.85, 2.4);
  }

  presetPose(name) {
    const p = CAMERA_PRESETS[name] || CAMERA_PRESETS.side;
    const target = new THREE.Vector3(...p.target);
    const off = new THREE.Vector3(...p.pos).sub(target);
    let k = name === 'top' ? Math.max(1, this._fitScale() * 0.75) : this._fitScale();
    if (p.maxScale) k = Math.min(k, p.maxScale);
    off.multiplyScalar(k);
    return { pos: target.clone().add(off), target };
  }

  flyTo(name, duration = 1.2) {
    const to = this.presetPose(name);
    if (duration <= 0) {
      this.camera.position.copy(to.pos);
      this.controls.target.copy(to.target);
      this.controls.update();
      return;
    }
    const s0 = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const s1 = new THREE.Spherical().setFromVector3(to.pos.clone().sub(to.target));
    let dTheta = s1.theta - s0.theta;
    while (dTheta > Math.PI) dTheta -= Math.PI * 2;
    while (dTheta < -Math.PI) dTheta += Math.PI * 2;
    this.cameraFlight = {
      t: 0, duration, s0, s1, dTheta,
      tg0: this.controls.target.clone(), tg1: to.target,
    };
  }

  _updateFlight(dt) {
    const f = this.cameraFlight;
    if (!f) return;
    f.t = Math.min(1, f.t + dt / f.duration);
    const k = easeInOutCubic(f.t);
    const sph = new THREE.Spherical(
      f.s0.radius + (f.s1.radius - f.s0.radius) * k,
      f.s0.phi + (f.s1.phi - f.s0.phi) * k,
      f.s0.theta + f.dTheta * k,
    );
    const tg = f.tg0.clone().lerp(f.tg1, k);
    this.controls.target.copy(tg);
    this.camera.position.copy(tg).add(new THREE.Vector3().setFromSpherical(sph));
    this.camera.lookAt(tg);
    if (f.t >= 1) { this.cameraFlight = null; this.controls.update(0); }
  }

  /** Keep the camera out of the hull/superstructure and above the water. */
  _constrainCamera() {
    const t = this.controls.target;
    t.x = THREE.MathUtils.clamp(t.x, -450, 600);
    t.y = THREE.MathUtils.clamp(t.y, 0, 60);
    t.z = THREE.MathUtils.clamp(t.z, -420, 420);
    const cam = this.camera.position;
    if (this.shipBoxEnabled && this.shipBox.containsPoint(cam)) {
      const off = cam.clone().sub(t);
      if (off.lengthSq() < 1) off.set(0, 0.3, 1);
      for (let i = 0; i < 40 && this.shipBox.containsPoint(cam); i++) {
        off.multiplyScalar(1.06);
        cam.copy(t).add(off);
      }
    }
    if (cam.y < 3) cam.y = 3;
  }

  // ---------------------------------------------------------------- loop ---
  onUpdate(fn) { this.updaters.push(fn); }

  resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.env.uViewportH.value = h * this.renderer.getPixelRatio();
    this.env.uPixelRatio.value = this.renderer.getPixelRatio();
    if (this.composer) {
      this.composer.setPixelRatio(this.renderer.getPixelRatio());
      this.composer.setSize(w, h);
    }
    this.width = w;
    this.height = h;
    if (this.onResize) this.onResize(w, h);
  }

  start() {
    this._wanted = true;
    this._syncRunning();
  }

  _syncRunning() {
    const should = this._wanted && this.visible && this.onScreen;
    if (should && !this.running) {
      this.running = true;
      this._clock.getDelta();
      requestAnimationFrame(this._loop);
    } else if (!should) {
      this.running = false;
    }
  }

  _loop() {
    if (!this.running) return;
    requestAnimationFrame(this._loop);
    const dt = Math.min(this._clock.getDelta(), 0.1);
    this.renderOnce(dt);
    this._watchPerf(dt);
  }

  renderOnce(dt = 1 / 60) {
    this.time += dt;
    this.env.uTime.value = this.time;
    this._blendTod(dt);
    for (const fn of this.updaters) {
      try { fn(dt, this.time); } catch (e) { console.error('[ship3d] update error', e); }
    }
    if (this.cameraFlight) this._updateFlight(dt);
    else this.controls.update(dt);
    this._constrainCamera();
    this.skyGroup.position.copy(this.camera.position);
    if (this.composer) this.composer.render(dt);
    else this.renderer.render(this.scene, this.camera);
  }

  _watchPerf(dt) {
    if (this.qualitySetting !== 'auto' || this.quality !== 'high' || this._perf.checked) return;
    const p = this._perf;
    p.frames++;
    if (p.frames < 60) return; // warm-up (shader compile etc.)
    p.acc += dt;
    if (p.frames >= 240) {
      p.checked = true;
      const avg = p.acc / 180;
      if (avg > 1 / 32) {
        console.info('[ship3d] low frame rate detected, switching to low quality');
        this.quality = 'low';
        this.renderer.setPixelRatio(1);
        this._setupComposer();
        this.resize();
      }
    }
  }
}
