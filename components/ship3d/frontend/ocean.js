// Animated ocean (Gerstner swells in the vertex shader + procedural detail
// normals, fresnel sky reflection, glint path, crest/wake foam, horizon fog)
// and the iceberg.
import * as THREE from 'three';
import { SKY_GLSL } from './scene.js';

// Hull half-breadth at the waterline (GLSL copy of ship.js halfBeam(x, 0)).
const HULL_GLSL = /* glsl */ `
float hullHalfBeam0(float x) {
  float bx = 133.2;
  float sx = -132.575;
  if (x >= bx || x <= sx) return 0.0;
  float db = (bx - x) / 64.75;
  float fb = db < 1.0 ? 1.0 - pow(1.0 - db, 2.2) : 1.0;
  float ds = (x - sx) / 25.3;
  float fs = ds < 1.0 ? sqrt(max(1.0 - (1.0 - ds) * (1.0 - ds), 0.0)) : 1.0;
  return 14.1 * fb * fs;
}
`;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Tileable detail normal map generated procedurally (no image files shipped).
 * R,G = surface slope (x,z), B = height (used as foam / sparkle noise).
 */
function makeNormalTexture(N = 256) {
  const rnd = mulberry32(1912);
  const h = new Float32Array(N * N);
  const octaves = [
    { cells: 6, amp: 1.0 }, { cells: 12, amp: 0.55 }, { cells: 24, amp: 0.3 },
    { cells: 48, amp: 0.16 }, { cells: 96, amp: 0.08 },
  ];
  for (const o of octaves) {
    const C = o.cells;
    const grid = new Float32Array(C * C);
    for (let i = 0; i < grid.length; i++) grid[i] = rnd() * 2 - 1;
    for (let y = 0; y < N; y++) {
      const fy = (y / N) * C; const iy = Math.floor(fy); let ty = fy - iy; ty = ty * ty * (3 - 2 * ty);
      for (let x = 0; x < N; x++) {
        const fx = (x / N) * C; const ix = Math.floor(fx); let tx = fx - ix; tx = tx * tx * (3 - 2 * tx);
        const a = grid[(iy % C) * C + (ix % C)], b = grid[(iy % C) * C + ((ix + 1) % C)];
        const c = grid[((iy + 1) % C) * C + (ix % C)], d = grid[((iy + 1) % C) * C + ((ix + 1) % C)];
        h[y * N + x] += o.amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty);
      }
    }
  }
  // a few periodic directional ripples so it reads as water, not clouds
  const ripples = [[5, 2, 0.35], [-3, 7, 0.22], [9, -4, 0.15], [2, 13, 0.1], [17, 6, 0.06]];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let s = 0;
    for (const [kx, ky, a] of ripples) s += a * Math.sin((2 * Math.PI * (kx * x + ky * y)) / N + kx * 1.3);
    h[y * N + x] += s;
  }
  let mn = Infinity, mx = -Infinity;
  for (const v of h) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const data = new Uint8Array(N * N * 4);
  const strength = 9.0;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const hx = (h[y * N + ((x + 1) % N)] - h[y * N + ((x - 1 + N) % N)]) / (mx - mn);
    const hz = (h[((y + 1) % N) * N + x] - h[((y - 1 + N) % N) * N + x]) / (mx - mn);
    data[i * 4 + 0] = Math.max(0, Math.min(255, 128 + hx * strength * 127));
    data[i * 4 + 1] = Math.max(0, Math.min(255, 128 + hz * strength * 127));
    data[i * 4 + 2] = Math.round(((h[i] - mn) / (mx - mn)) * 255);
    data[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Polar grid: dense near the ship, sparse toward the horizon. */
function radialGrid(R, rings, segs) {
  const pos = new Float32Array((rings + 1) * segs * 3);
  for (let i = 0; i <= rings; i++) {
    const r = R * Math.pow(i / rings, 2.3);
    for (let j = 0; j < segs; j++) {
      const th = (j / segs) * Math.PI * 2;
      pos.set([Math.cos(th) * r, 0, Math.sin(th) * r], (i * segs + j) * 3);
    }
  }
  const idx = [];
  for (let i = 0; i < rings; i++) for (let j = 0; j < segs; j++) {
    const a = i * segs + j, b = i * segs + ((j + 1) % segs);
    const c = (i + 1) * segs + ((j + 1) % segs), d = (i + 1) * segs + j;
    idx.push(a, b, d, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

// direction (x,z), wavelength (m), amplitude (m). The night was calm: gentle swell.
const WAVES = [
  [1.0, 0.25, 190, 0.75],
  [0.7, -0.7, 115, 0.45],
  [0.2, 1.0, 72, 0.3],
  [-0.6, 0.8, 44, 0.18],
  [0.9, 0.45, 27, 0.1],
];

export function createOcean(stage) {
  const low = stage.quality !== 'high';
  const geo = radialGrid(6500, low ? 130 : 190, low ? 200 : 300);
  const tex = makeNormalTexture(256);
  tex.anisotropy = Math.min(8, stage.renderer.capabilities.getMaxAnisotropy());

  const uniforms = {
    ...stage.env,
    uNormalMap: { value: tex },
    uWaves: { value: WAVES.map(([x, z, L, A]) => { const d = new THREE.Vector2(x, z).normalize(); return new THREE.Vector4(d.x, d.y, L, A); }) },
    uSteep: { value: 0.55 },
    uFlow: { value: 0 },
    uSail: { value: 1 },
    uHullFoam: { value: 1 },
    uLights: { value: 1 },
    uShip: { value: new THREE.Vector3() },
    uIce: { value: new THREE.Vector3(470, 40, 240) }, // x, radius, z
  };

  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: /* glsl */ `
      uniform float uTime;
      uniform float uFlow;
      uniform float uSteep;
      uniform vec4 uWaves[5];
      varying vec3 vWorld;
      varying vec3 vNormalW;
      varying float vHeight;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vec2 p = wp.xz + vec2(uFlow, 0.0);
        float dist = length(wp.xz - cameraPosition.xz);
        float atten = 1.0 - smoothstep(1400.0, 4200.0, dist);
        vec3 disp = vec3(0.0);
        float nx = 0.0, ny = 0.0, nz = 0.0;
        for (int i = 0; i < 5; i++) {
          vec4 w = uWaves[i];
          float k = 6.2831853 / w.z;
          float A = w.w * atten * (w.z < 50.0 ? 1.0 - smoothstep(500.0, 1400.0, dist) : 1.0);
          float c = sqrt(9.8 / k);
          float f = k * (dot(w.xy, p) - c * uTime);
          float Q = uSteep / (k * w.w * 5.0);
          float S = sin(f), C = cos(f);
          disp.x += Q * A * w.x * C;
          disp.z += Q * A * w.y * C;
          disp.y += A * S;
          float WA = k * A;
          nx += w.x * WA * C;
          nz += w.y * WA * C;
          ny += Q * WA * S;
        }
        wp.xyz += disp;
        vHeight = disp.y;
        vNormalW = normalize(vec3(-nx, 1.0 - ny, -nz));
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform float uFlow;
      uniform sampler2D uNormalMap;
      uniform vec3 uDeep;
      uniform vec3 uShallow;
      uniform vec3 uFoamColor;
      uniform float uSpecPow;
      uniform float uSpecStrength;
      uniform float uFogDensity;
      uniform float uSail;
      uniform float uHullFoam;
      uniform float uLights;
      uniform float uWindowGlow;
      uniform vec3 uShip;
      uniform vec3 uIce;
      varying vec3 vWorld;
      varying vec3 vNormalW;
      varying float vHeight;
      ${SKY_GLSL}
      ${HULL_GLSL}
      void main() {
        vec3 toCam = cameraPosition - vWorld;
        float dist = length(toCam);
        vec3 V = toCam / dist;
        vec2 uv = vWorld.xz + vec2(uFlow, 0.0);
        vec4 t1 = texture2D(uNormalMap, uv / 173.0 + vec2(uTime * 0.0040, uTime * 0.0023));
        vec4 t2 = texture2D(uNormalMap, uv / 61.0 + vec2(-uTime * 0.0061, uTime * 0.0047));
        vec4 t3 = texture2D(uNormalMap, uv / 19.0 + vec2(uTime * 0.0110, -uTime * 0.0090));
        float near = 1.0 - smoothstep(150.0, 1100.0, dist);
        // wind patches: large slow-moving areas of rougher / calmer water
        float windPatch = smoothstep(0.25, 0.75, texture2D(uNormalMap, uv / 1900.0 + vec2(uTime * 0.0007, 0.0)).b);
        float rough = mix(0.55, 1.15, windPatch);
        vec2 s = ((t1.rg * 2.0 - 1.0) * 0.5 + (t2.rg * 2.0 - 1.0) * 0.38 * rough + (t3.rg * 2.0 - 1.0) * 0.22 * near * rough);
        // flatten detail toward the horizon to avoid sparkle aliasing
        s *= mix(0.35, 1.0, 1.0 - smoothstep(600.0, 3500.0, dist));
        vec3 N = normalize(vNormalW + vec3(-s.x, 0.0, -s.y));

        float NdV = max(dot(N, V), 0.0);
        float fres = 0.02 + 0.98 * pow(1.0 - NdV, 5.0);
        vec3 R = reflect(-V, N);
        R.y = abs(R.y);
        vec3 refl = skyColor(R);
        float spec = pow(max(dot(R, normalize(uSunDir)), 0.0), uSpecPow) * uSpecStrength;

        // body colour: darker looking straight down, lighter through crests
        float crest = clamp(vHeight * 0.45 + 0.5, 0.0, 1.0);
        vec3 body = mix(uDeep, uShallow, crest * crest * 0.85 + (1.0 - NdV) * 0.12);
        vec3 col = mix(body, refl, fres);
        col += uSunColor * spec;

        // --- foam: crests, hull contact / bow wave, stern wake, Kelvin arms
        vec2 p = vWorld.xz - uShip.xz;
        float hb = hullHalfBeam0(p.x);
        float e = abs(p.y) - hb;
        // fine foam texture near the camera, broad soft texture further away
        float foamFine = smoothstep(0.40, 0.72, t2.b * 0.55 + t3.b * 0.45);
        float foamBroad = smoothstep(0.30, 0.80, t1.b * 0.5 + t2.b * 0.5);
        float foamTex = mix(foamBroad, foamFine, near);
        float foam = smoothstep(1.0, 1.6, vHeight) * foamTex * 0.35;
        float onHull = step(-133.0, p.x) * step(p.x, 133.5);
        float bowF = smoothstep(55.0, 128.0, p.x);
        foam += onHull * exp(-max(e, 0.0) / (1.4 + 3.5 * bowF * uSail)) * (0.30 + 0.70 * bowF * uSail) * uHullFoam * (0.45 + 0.8 * foamTex);
        float dd = -129.0 - p.x;
        if (dd > 0.0) {
          float w = 10.0 + dd * 0.075;
          float core = 1.0 - smoothstep(w * 0.3, w, abs(p.y));
          float fade = exp(-dd / 700.0) * uSail * uHullFoam;
          col = mix(col, uShallow * 1.5 + uFoamColor * 0.08, core * fade * 0.45);
          foam += core * fade * (0.25 + 0.6 * foamTex) * 0.9;
        }
        float db = 133.0 - p.x;
        if (db > 0.0) {
          float arm = 7.0 + db * 0.34;
          float wdt = 1.4 + db * 0.016;
          float ea = (abs(p.y) - arm) / wdt;
          float m = exp(-ea * ea) * exp(-db / 380.0) * uSail * uHullFoam;
          foam += m * (0.3 + 0.7 * foamTex) * 0.4;
        }
        // lapping foam around the iceberg
        float ei = length(vWorld.xz - uIce.xz) - uIce.y;
        foam += exp(-abs(ei) / 5.0) * foamTex * 0.5;

        // warm window light spilling onto the water beside the hull
        float spill = exp(-max(e, 0.0) / 30.0) * smoothstep(140.0, 95.0, abs(p.x)) * uLights * uWindowGlow * uHullFoam;
        // glints of the lit portholes: sparse warm sparkles on wave faces near the hull
        float sparkle = smoothstep(0.62, 0.9, t3.b * 0.6 + t2.b * 0.4) * near;
        col += vec3(1.0, 0.78, 0.45) * spill * (0.006 + 0.11 * sparkle);

        col = mix(col, uFoamColor, clamp(foam, 0.0, 1.0));

        // fog toward a horizon colour that exactly matches the sky dome
        float f = 1.0 - exp(-pow(uFogDensity * dist, 2.0));
        vec3 fogC = skyColor(normalize(vec3(-V.x, 0.0, -V.z)));
        col = mix(col, fogC, f);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    side: THREE.DoubleSide,
    fog: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 0;

  return {
    mesh,
    uniforms,
    /** Analytic swell height at a world xz (CPU copy of the vertex shader, no Gerstner xz shift). */
    heightAt(x, z, t) {
      const px = x + uniforms.uFlow.value;
      let y = 0;
      for (const w of uniforms.uWaves.value) {
        const k = (Math.PI * 2) / w.z;
        y += w.w * Math.sin(k * (w.x * px + w.y * z - Math.sqrt(9.8 / k) * t));
      }
      return y;
    },
    update(dt, sail) {
      uniforms.uFlow.value += dt * 7.5 * sail; // the sea slides aft while sailing
      uniforms.uSail.value = sail;
    },
  };
}

// ---------------------------------------------------------------- iceberg ---
function noise3(x, y, z) {
  // cheap smooth-ish deterministic noise for vertex displacement
  return (
    Math.sin(x * 1.7 + y * 0.9) * 0.5 + Math.sin(y * 2.3 - z * 1.3 + 1.1) * 0.3 +
    Math.sin(z * 3.1 + x * 2.2 + 2.3) * 0.2 + Math.sin((x + y + z) * 5.3) * 0.08
  );
}

export function createIceberg(env) {
  const g = new THREE.IcosahedronGeometry(1, 3);
  const p = g.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = noise3(v.x * 2.1, v.y * 2.1, v.z * 2.1);
    let r = 1 + 0.28 * n;
    v.multiplyScalar(r);
    // peaked, asymmetric top; broad base
    if (v.y > 0) {
      v.y *= 1.0 + 0.9 * Math.max(0, Math.sin(v.x * 2.0 + 0.6)) + 0.5 * Math.max(0, v.z);
      v.x *= 1 - 0.25 * v.y; v.z *= 1 - 0.22 * v.y;
    } else {
      v.y *= 0.7;
    }
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals(); // non-indexed => faceted, crystalline look
  g.scale(48, 30, 38);

  const mat = new THREE.MeshStandardMaterial({
    color: 0xdceaf7, roughness: 0.42, metalness: 0.0, flatShading: true,
    emissive: new THREE.Color('#7fa8d6'), emissiveIntensity: 0.3,
  });
  const berg = new THREE.Mesh(g, mat);
  berg.position.set(470, -4, 240);
  berg.rotation.y = 0.7;

  // pale glow halo (additive sprite with a procedural radial gradient)
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const ctx = cv.getContext('2d');
  const grd = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.35, 'rgba(255,255,255,0.35)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 128, 128);
  const glowTex = new THREE.CanvasTexture(cv);
  glowTex.colorSpace = THREE.SRGBColorSpace;
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex, color: 0x9cc4ff, transparent: true, opacity: 0.2,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
  }));
  glow.scale.set(240, 150, 1);
  glow.position.set(470, 22, 240);

  const group = new THREE.Group();
  group.add(berg, glow);
  return {
    group,
    applyPreset(p) {
      mat.emissiveIntensity = p.ice;
      glow.material.opacity = p.iceGlow;
      glow.visible = p.iceGlow > 0.005;
    },
  };
}
