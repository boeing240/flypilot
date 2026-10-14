// 3D view of a Grand Prix: the same recorded runs, with the fly clearly visible at the wheel of its car.
// Loaded by index.html as an ES module; talks to the page through window.View3D (see the bottom).
// World: x along the strip (metres), y up, z across (lane 1 on the left when looking down the strip).
import * as THREE from "./vendor/three.module.min.js";

const LANE_W = 3.0, CW = 1316, CH = 560;
let renderer = null, scene = null, camera = null, cars = [], field = null, shared = null, smoke = null, trackGroup = null, failed = false;
const look = new THREE.Vector3(), camGoal = new THREE.Vector3(), lookGoal = new THREE.Vector3();
const cam = { shot: "", since: 0, init: false };
const clamp01 = (x) => Math.max(0, Math.min(1, x));

function canvasTex(w, h, draw, repeat) {
  const c = document.createElement("canvas"); c.width = w; c.height = h; draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  return t;
}
const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, ...o });

/* ------------------------------------------------------------------ shared geometry and materials */
function buildShared() {
  const g = {};
  g.box = new THREE.BoxGeometry(1, 1, 1);
  g.sphere = new THREE.SphereGeometry(1, 28, 18);
  g.ico = new THREE.IcosahedronGeometry(1, 2);
  g.cyl = new THREE.CylinderGeometry(1, 1, 1, 24);
  g.cone = new THREE.ConeGeometry(1, 1, 16);
  g.dome = new THREE.SphereGeometry(1, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2);
  g.torus = new THREE.TorusGeometry(1, 0.16, 8, 28);
  g.wheelRot = new THREE.CylinderGeometry(1, 1, 1, 28).rotateX(Math.PI / 2);        // axis along z
  g.plane = new THREE.PlaneGeometry(1, 1);
  g.m = {
    brown: mat(0x5a4630, { roughness: 0.7 }), brownDark: mat(0x3b2f22, { roughness: 0.8 }), tan: mat(0xd8b078, { roughness: 0.7 }),
    eye: mat(0xd31a2e, { roughness: 0.25, emissive: 0x5a0610, flatShading: true }), glint: new THREE.MeshBasicMaterial({ color: 0xffffff }),
    tire: mat(0x15181f, { roughness: 0.9 }), rim: mat(0xc7ced9, { roughness: 0.3, metalness: 0.7 }), dark: mat(0x0e121a, { roughness: 0.6 }),
    white: mat(0xf2f6fc, { roughness: 0.4 }), gold: mat(0xffd700, { roughness: 0.25, metalness: 0.6, emissive: 0x4a3a00 }),
    wing: new THREE.MeshStandardMaterial({ color: 0xcfe6ff, roughness: 0.2, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }),
    glass: new THREE.MeshStandardMaterial({ color: 0x9fd2ff, roughness: 0.05, transparent: true, opacity: 0.18, depthWrite: false }),
  };
  return g;
}
const liveryCache = new Map();
function livery(color) {
  if (!liveryCache.has(color)) liveryCache.set(color, { body: mat(color, { roughness: 0.32, metalness: 0.25 }), dark: mat(new THREE.Color(color).multiplyScalar(0.55), { roughness: 0.4, metalness: 0.2 }),
    helmet: mat(color, { roughness: 0.25, metalness: 0.3 }) });
  return liveryCache.get(color);
}
const plateCache = new Map();
function plateMaterial(number, color) {
  const key = number + color;
  if (!plateCache.has(key)) plateCache.set(key, new THREE.MeshBasicMaterial({ map: canvasTex(128, 96, (c, w, h) => {
    c.fillStyle = "#f7f9fc"; c.beginPath(); c.roundRect(4, 4, w - 8, h - 8, 16); c.fill();
    c.lineWidth = 7; c.strokeStyle = color; c.stroke();
    c.fillStyle = "#0b0f17"; c.font = "800 62px Consolas, monospace"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(String(number), w / 2, h / 2 + 4);
  }), transparent: true }));
  return plateCache.get(key);
}

function part(geo, material, pos, scale, rot) {
  const m = new THREE.Mesh(geo, material);
  m.position.set(...pos); m.scale.set(...(Array.isArray(scale) ? scale : [scale, scale, scale]));
  if (rot) m.rotation.set(...rot);
  return m;
}
function limb(a, b, r, material) {                                   // a cylinder between two points
  const d = new THREE.Vector3().subVectors(b, a), m = new THREE.Mesh(shared.cyl, material);
  m.position.copy(a).addScaledVector(d, 0.5); m.scale.set(r, d.length(), r);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return m;
}

/* ------------------------------------------------------------------ the fly (the star of the show) */
function buildFly(color, legend) {
  const S = shared, M = S.m, L = livery(color), grp = new THREE.Group(), body = new THREE.Group();
  grp.add(body);
  body.add(part(S.sphere, M.brown, [-0.5, 0.12, 0], [0.42, 0.32, 0.34]));                    // thorax
  body.add(part(S.sphere, M.brownDark, [-1.05, 0.02, 0], [0.62, 0.3, 0.3]));                 // abdomen
  for (let i = 0; i < 3; i++) body.add(part(S.sphere, M.tan, [-0.95 - i * 0.2, 0.02, 0], [0.04, 0.31, 0.31]));   // pale bands
  body.add(part(S.sphere, M.brownDark, [0.12, 0.24, 0], [0.3, 0.32, 0.34]));                  // head
  for (const sz of [-1, 1]) {
    body.add(part(S.ico, M.eye, [0.2, 0.3, sz * 0.22], [0.2, 0.25, 0.2]));                   // compound eyes (faceted)
    body.add(part(S.sphere, M.glint, [0.32, 0.4, sz * 0.2], 0.045));                          // glint
    body.add(limb(new THREE.Vector3(0.3, 0.52, sz * 0.07), new THREE.Vector3(0.52, 0.78, sz * 0.16), 0.016, M.brownDark));   // antennae
    body.add(part(S.sphere, M.brownDark, [0.54, 0.8, sz * 0.17], [0.07, 0.035, 0.05]));
    const arm = limb(new THREE.Vector3(-0.2, 0.05, sz * 0.2), new THREE.Vector3(0.48, 0.4, sz * 0.22), 0.028, M.brownDark);   // arms to the wheel
    body.add(arm);
  }
  body.add(part(S.sphere, M.tan, [0.4, 0.2, 0], [0.1, 0.12, 0.12]));                          // face
  body.add(part(S.cone, M.brownDark, [0.5, 0.1, 0], [0.04, 0.16, 0.04], [0, 0, -2.2]));       // proboscis
  const helmet = new THREE.Group(); helmet.position.set(0.1, 0.33, 0);
  helmet.add(part(S.dome, legend ? M.gold : L.helmet, [0, 0, 0], [0.4, 0.34, 0.4], [0, 0, 0.12]));
  helmet.add(part(S.box, M.white, [0, 0.335, 0], [0.62, 0.02, 0.07], [0, 0, 0.12]));          // racing stripe
  if (legend) for (let i = 0; i < 5; i++) { const a = (i / 5) * Math.PI * 2; helmet.add(part(S.cone, M.gold, [Math.cos(a) * 0.17, 0.44, Math.sin(a) * 0.17], [0.05, 0.14, 0.05])); }
  body.add(helmet);
  const wheel = new THREE.Group(); wheel.position.set(0.55, 0.34, 0);                      // the steering wheel
  wheel.add(part(S.torus, M.dark, [0, 0, 0], [0.22, 0.22, 0.22], [0, Math.PI / 2, 0]));
  body.add(wheel);
  const wings = [-1, 1].map((sz) => {
    const h = new THREE.Group(); h.position.set(-0.55, 0.38, sz * 0.12);
    h.add(part(S.sphere, M.wing, [-0.1, 0, sz * 0.5], [0.62, 0.012, 0.3]));
    h.userData.sz = sz; body.add(h); return h;
  });
  grp.userData = { wings, wheel, body };
  return grp;
}

/* ------------------------------------------------------------------ the car */
function buildCar(c) {
  const S = shared, M = S.m, L = livery(c.legend ? 0xffd700 : c.color), grp = new THREE.Group(), body = new THREE.Group();
  grp.add(body);
  body.add(part(S.box, L.body, [0, 0.58, 0], [4.3, 0.34, 1.25]));                              // tub
  body.add(part(S.cone, L.body, [2.75, 0.58, 0], [0.5, 1.7, 0.5], [0, 0, -Math.PI / 2]));       // nose
  body.add(part(S.box, M.dark, [-0.25, 0.86, 0], [1.9, 0.5, 1.0]));                              // cockpit tub
  for (const sz of [-1, 1]) {
    body.add(part(S.box, L.dark, [0.9, 0.62, sz * 0.78], [1.9, 0.4, 0.34]));                      // side pods
    body.add(part(S.plane, plateMaterial(c.number, c.color), [0.6, 0.84, sz * 0.97], [0.8, 0.58, 1], [0, sz > 0 ? 0 : Math.PI, 0]));
    body.add(part(S.box, M.dark, [-2.15, 1.05, sz * 0.8], [0.12, 0.75, 0.1]));                   // wing supports
    body.add(part(S.box, L.dark, [-2.15, 1.5, sz * 1.02], [0.7, 0.4, 0.06]));                     // end plates
  }
  body.add(part(S.box, M.white, [-2.15, 1.42, 0], [0.62, 0.07, 2.0]));                            // rear wing
  body.add(part(S.box, M.dark, [-2.05, 0.55, 0], [0.5, 0.3, 0.7]));                               // engine
  body.add(part(S.cyl, M.rim, [-2.35, 0.5, 0.25], [0.07, 0.7, 0.07], [0, 0, Math.PI / 2]));       // exhausts
  body.add(part(S.cyl, M.rim, [-2.35, 0.5, -0.25], [0.07, 0.7, 0.07], [0, 0, Math.PI / 2]));
  const wheels = [];
  const mk = (x, z, r, w, front) => {
    const wg = new THREE.Group(); wg.position.set(x, r, z);
    const spin = new THREE.Group(); wg.add(spin);
    spin.add(part(S.wheelRot, M.tire, [0, 0, 0], [r, r, w]));
    spin.add(part(S.wheelRot, M.rim, [0, 0, Math.sign(z) * w * 0.52], [r * 0.62, r * 0.62, 0.02]));
    spin.add(part(S.box, M.dark, [0, 0, Math.sign(z) * w * 0.54], [r * 1.2, r * 0.12, 0.02]));          // a spoke, so the spin shows
    body.add(wg); wheels.push({ wg, spin, r, front });
  };
  mk(-1.5, 0.95, 0.52, 0.62, false); mk(-1.5, -0.95, 0.52, 0.62, false); mk(1.9, 0.82, 0.38, 0.32, true); mk(1.9, -0.82, 0.38, 0.32, true);
  const fly = buildFly(c.color, c.legend); fly.position.set(-0.15, 1.0, 0); fly.scale.setScalar(1.5);
  body.add(fly);
  body.add(part(S.box, M.glass, [0.15, 1.12, 0], [0.04, 0.5, 0.95], [0, 0, -0.5]));            // a small windscreen
  const shadow = new THREE.Mesh(S.plane, new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; shadow.scale.set(5.2, 2.6, 1); shadow.position.y = 0.02;
  grp.add(shadow);
  grp.userData = { body, fly, wheels, shadow, lastX: 0, spin: 0 };
  return grp;
}

/* ------------------------------------------------------------------ the strip */
function disposeTrack() {
  if (!trackGroup) return;
  scene.remove(trackGroup);
  trackGroup.traverse((o) => { if (o.geometry && !Object.values(shared).includes(o.geometry)) o.geometry.dispose(); });
  trackGroup = null;
}
function buildTrack(n, L) {
  disposeTrack();
  const g = trackGroup = new THREE.Group(), S = shared, W = n * LANE_W, END = L + 120;
  const asphalt = canvasTex(256, 256, (c, w, h) => {
    c.fillStyle = "#2a2f3a"; c.fillRect(0, 0, w, h);
    for (let i = 0; i < 2600; i++) { const v = 36 + Math.random() * 26; c.fillStyle = `rgb(${v},${v + 3},${v + 10})`; c.fillRect(Math.random() * w, Math.random() * h, 2, 2); }
  }, true); asphalt.repeat.set((END + 40) / 8, W / 8);
  const road = new THREE.Mesh(S.plane, new THREE.MeshStandardMaterial({ map: asphalt, roughness: 0.92 }));
  road.rotation.x = -Math.PI / 2; road.scale.set(END + 40, W, 1); road.position.set((END - 40) / 2 - 0, 0, 0); g.add(road);
  const grass = new THREE.Mesh(S.plane, new THREE.MeshStandardMaterial({ color: 0x16301f, roughness: 1 }));
  grass.rotation.x = -Math.PI / 2; grass.scale.set(900, 400, 1); grass.position.set(L / 2, -0.04, 0); g.add(grass);
  const line = new THREE.MeshBasicMaterial({ color: 0xe9eef6 });
  for (let i = 0; i <= n; i++) {                                                    // lane lines
    const l = new THREE.Mesh(S.plane, line); l.rotation.x = -Math.PI / 2; l.scale.set(END + 40, 0.09, 1); l.position.set((END - 40) / 2, 0.012, (i - n / 2) * LANE_W); g.add(l);
  }
  const startLine = new THREE.Mesh(S.plane, line); startLine.rotation.x = -Math.PI / 2; startLine.scale.set(0.35, W, 1); startLine.position.set(0, 0.014, 0); g.add(startLine);
  const chequer = canvasTex(64, 8, (c, w, h) => { for (let x = 0; x < 16; x++) for (let y = 0; y < 2; y++) { c.fillStyle = (x + y) % 2 ? "#0b0f17" : "#f4f7fb"; c.fillRect(x * 4, y * 4, 4, 4); } });
  chequer.magFilter = THREE.NearestFilter;
  const fin = new THREE.Mesh(S.plane, new THREE.MeshBasicMaterial({ map: chequer })); fin.rotation.x = -Math.PI / 2; fin.scale.set(1.6, W, 1); fin.position.set(L, 0.015, 0); g.add(fin);
  // barriers with red/white stripes, distance boards, grandstands
  const stripes = canvasTex(64, 16, (c, w, h) => { for (let x = 0; x < 8; x++) { c.fillStyle = x % 2 ? "#e8eef7" : "#d6455a"; c.fillRect(x * 8, 0, 8, h); } }, true);
  stripes.repeat.set((END + 40) / 3, 1);
  const crowd = canvasTex(512, 64, (c, w, h) => {
    c.fillStyle = "#151b2b"; c.fillRect(0, 0, w, h);
    for (let i = 0; i < 700; i++) { c.fillStyle = `hsl(${Math.random() * 360},65%,${45 + Math.random() * 25}%)`; c.fillRect(Math.random() * w, Math.random() * h, 3, 4); }
  }, true); crowd.repeat.set((END + 40) / 40, 1);
  for (const sz of [-1, 1]) {
    const wall = new THREE.Mesh(S.box, new THREE.MeshStandardMaterial({ map: stripes, roughness: 0.7 })); wall.scale.set(END + 40, 0.9, 0.5); wall.position.set((END - 40) / 2, 0.45, sz * (W / 2 + 0.7)); g.add(wall);
    const stand = new THREE.Mesh(S.box, new THREE.MeshStandardMaterial({ map: crowd, roughness: 1, emissive: 0x222233, emissiveMap: crowd })); stand.scale.set(L + 60, 7, 9);
    stand.position.set(L / 2, 3.5, sz * (W / 2 + 12)); stand.rotation.x = sz * -0.18; g.add(stand);
    for (let x = 0; x <= L; x += 50) {
      const t = canvasTex(128, 64, (c, w, h) => { c.fillStyle = "#0b0f17"; c.fillRect(0, 0, w, h); c.strokeStyle = "#ffd700"; c.lineWidth = 4; c.strokeRect(3, 3, w - 6, h - 6);
        c.fillStyle = "#fff"; c.font = "800 34px Consolas, monospace"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(x + " m", w / 2, h / 2 + 2); });
      const b = new THREE.Mesh(S.plane, new THREE.MeshBasicMaterial({ map: t })); b.scale.set(3.6, 1.8, 1); b.position.set(x, 2.2, sz * (W / 2 + 1.4)); b.rotation.y = sz > 0 ? Math.PI : 0; g.add(b);
      g.add(part(S.cyl, mat(0x39425a), [x, 1.1, sz * (W / 2 + 1.4)], [0.06, 2.2, 0.06]));
    }
    for (let x = -20; x <= L + 80; x += 45) {                                       // flood-light poles
      g.add(part(S.cyl, mat(0x39425a), [x, 6, sz * (W / 2 + 3)], [0.15, 12, 0.15]));
      g.add(part(S.sphere, new THREE.MeshBasicMaterial({ color: 0xfff0c8 }), [x, 12.2, sz * (W / 2 + 3)], 0.55));
    }
  }
  // the start gantry and the finish arch
  g.add(part(S.box, mat(0x252c3d), [-4, 6.2, 0], [0.5, 0.5, W + 4]));
  g.add(part(S.box, mat(0x252c3d), [-4, 3.1, -(W / 2 + 1.8)], [0.4, 6.2, 0.4])); g.add(part(S.box, mat(0x252c3d), [-4, 3.1, W / 2 + 1.8], [0.4, 6.2, 0.4]));
  const lights = [];
  for (let i = 0; i < 4; i++) { const lm = new THREE.MeshBasicMaterial({ color: 0x1d2739 }); const lamp = part(S.sphere, lm, [-4.3, 5.4, (i - 1.5) * 1.1], 0.42); g.add(lamp); lights.push(lm); }
  g.userData.lights = lights;
  const fsign = canvasTex(512, 96, (c, w, h) => { c.fillStyle = "#0b0f17"; c.fillRect(0, 0, w, h); c.fillStyle = "#ffd700"; c.font = "800 62px Segoe UI, Arial"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("FINISH", w / 2, h / 2 + 3); });
  const sign = new THREE.Mesh(S.plane, new THREE.MeshBasicMaterial({ map: fsign, side: THREE.DoubleSide })); sign.scale.set(14, 2.6, 1); sign.position.set(L, 6.4, 0); g.add(sign);
  g.add(part(S.box, mat(0x252c3d), [L, 4.8, -(W / 2 + 1.5)], [0.4, 9.6, 0.4])); g.add(part(S.box, mat(0x252c3d), [L, 4.8, W / 2 + 1.5], [0.4, 9.6, 0.4])); g.add(part(S.box, mat(0x252c3d), [L, 9.5, 0], [0.4, 0.4, W + 3.4]));
  scene.add(g);
  return g;
}

/* ------------------------------------------------------------------ smoke from spinning wheels */
function buildSmoke() {
  const N = 720, geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  const tex = canvasTex(64, 64, (c, w, h) => { const gr = c.createRadialGradient(32, 32, 2, 32, 32, 30); gr.addColorStop(0, "rgba(255,255,255,.9)"); gr.addColorStop(1, "rgba(255,255,255,0)"); c.fillStyle = gr; c.fillRect(0, 0, w, h); });
  const m = new THREE.PointsMaterial({ size: 1.6, map: tex, transparent: true, opacity: 0.45, depthWrite: false, color: 0xd9dfe9 });
  const pts = new THREE.Points(geo, m); pts.frustumCulled = false;
  return { pts, N, life: new Float32Array(N), vel: new Float32Array(N * 3), head: 0 };
}
function puff(x, y, z, vx) {
  const s = smoke, i = (s.head = (s.head + 1) % s.N), p = s.pts.geometry.attributes.position.array;
  p[i * 3] = x + (Math.random() - 0.5) * 0.4; p[i * 3 + 1] = y + Math.random() * 0.3; p[i * 3 + 2] = z + (Math.random() - 0.5) * 1.2;
  s.vel[i * 3] = vx * 0.1 - Math.random() * 1.5; s.vel[i * 3 + 1] = 0.6 + Math.random() * 0.8; s.vel[i * 3 + 2] = (Math.random() - 0.5) * 0.8; s.life[i] = 1;
}
function stepSmoke(dt) {
  const s = smoke, p = s.pts.geometry.attributes.position.array;
  for (let i = 0; i < s.N; i++) {
    if (s.life[i] <= 0) { p[i * 3 + 1] = -50; continue; }
    s.life[i] -= dt * 0.9; p[i * 3] += s.vel[i * 3] * dt; p[i * 3 + 1] += s.vel[i * 3 + 1] * dt; p[i * 3 + 2] += s.vel[i * 3 + 2] * dt;
  }
  s.pts.geometry.attributes.position.needsUpdate = true;
}

/* ------------------------------------------------------------------ set-up */
function init(canvas) {
  if (renderer || failed) return !!renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5) * Math.max(0.75, Math.min(1.4, Math.min(innerWidth / 1920, innerHeight / 1080) * 1.2)));
    renderer.setSize(CW, CH, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    scene = new THREE.Scene();
    const sky = canvasTex(8, 256, (c, w, h) => { const gr = c.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, "#070b1b"); gr.addColorStop(0.55, "#1b2547"); gr.addColorStop(0.86, "#5a3b64"); gr.addColorStop(1, "#d98a68"); c.fillStyle = gr; c.fillRect(0, 0, w, h); });
    scene.background = sky; scene.fog = new THREE.Fog(0x241f3d, 90, 420);
    camera = new THREE.PerspectiveCamera(42, CW / CH, 0.3, 1200);
    scene.add(new THREE.HemisphereLight(0xb7c8ff, 0x2a2438, 0.95));
    const sun = new THREE.DirectionalLight(0xfff0d8, 2.1); sun.position.set(-30, 50, -25); scene.add(sun);
    const rim = new THREE.DirectionalLight(0x6f8cff, 0.8); rim.position.set(40, 20, 30); scene.add(rim);
    shared = buildShared(); smoke = buildSmoke(); scene.add(smoke.pts);
    return true;
  } catch (e) {
    failed = true; renderer = null; console.error("3D view unavailable:", e);
    return false;
  }
}

function setField(key, list, n, L) {
  if (field && field.key === key) return;
  for (const c of cars) scene.remove(c.obj);
  cars = list.map((c) => { const obj = buildCar(c); scene.add(obj); return { obj, lane: c.lane, number: c.number }; });
  buildTrack(n, L);
  field = { key, n, L, W: n * LANE_W };
  cam.init = false;
}

/* ------------------------------------------------------------------ camera director */
function director(st, dt, pack, winner) {
  const { L, W } = field, t = st.simT - st.tGreen, xl = pack.x, zc = pack.z, sp = pack.spread;
  let shot, P = camGoal, T = lookGoal;
  if (st.simT < st.tGreen + 0.2) {                                              // grid: slow push-in from behind and above
    shot = "grid"; P.set(-15 + Math.min(st.simT, 8) * 0.5, 6.5, 2); T.set(14, 1.1, 0);
  } else if (st.tWin !== null && st.simT >= st.tWin && winner) {                // the winner: a slow orbit
    shot = "winner"; const a = 0.9 + (st.simT - st.tWin) * 0.55;
    P.set(winner.x - 2 + Math.cos(a) * 6.5, 2.3, winner.z + Math.sin(a) * 6.5); T.set(winner.x, 1.45, winner.z);
  } else if (xl > L - 42) {                                                      // finish line, facing the fly faces
    shot = "finish"; P.set(L + 3.0, 1.9, THREE.MathUtils.clamp(zc * 0.8, -W * 0.3, W * 0.3)); T.set(L - 14, 1.3, zc * 0.6);
  } else if (t < 2.4) {                                                          // launch: a broadcast angle from the side
    shot = "launch"; P.set(xl - 4, 5.5, -W / 2 - 6); T.set(xl + 4, 0.9, -W * 0.1);
  } else {
    const idx = Math.floor((t - 2.4) / 1.9) % 3;                                // three shots in rotation, a couple of seconds each
    shot = ["hero", "chase", "side"][idx];
    if (idx === 0) { P.set(xl + 9 + sp * 0.4, 2.1 + sp * 0.05, zc + 3.5 + sp * 0.25); T.set(xl - 1, 1.25, zc); }            // ahead of the leaders, looking back at the faces
    else if (idx === 1) { P.set(xl - 6.5 - sp * 0.5, 2.9 + sp * 0.12, zc); T.set(xl + 6, 1.0, zc); }                         // above and behind: the wings from the top
    else { P.set(xl + 3, 4.2, -W / 2 - 8.5); T.set(xl - 1, 1.1, zc); }                                              // beside the pack, above the wall
  }
  const k = cam.init ? 1 - Math.exp(-dt * (shot === cam.shot ? 3.2 : 2.0)) : 1;
  camera.position.lerp(P, k); look.lerp(T, k); camera.lookAt(look);
  cam.shot = shot; cam.init = true;
}

/* ------------------------------------------------------------------ one frame */
const v3 = new THREE.Vector3();
function render(st) {
  // st: {simT, dt, tGreen, tWin, anim:{p}|null, cars:[{lane,x,y,v,g,s,th,st,sh,yaw,done,status,since}]}
  if (!renderer || !field) return [];
  const { n, L } = field, dt = Math.min(0.1, Math.max(0.001, st.dt || 0.016));
  let leader = null, winner = null, lx = -1e9;
  const rolls = st.anim ? (i) => -(1 - clamp01((st.anim.p - 0.15 - i * 0.02) / 0.35)) * 40 : () => 0;
  const labels = [];
  st.cars.forEach((cs, i) => {
    const c = cars[i]; if (!c) return;
    const u = c.obj.userData, coast = cs.done ? cs.vEnd * 2.2 * (1 - Math.exp(-cs.since / 2.2)) : 0;
    const x = cs.x + coast + rolls(i), z = (cs.lane - (n + 1) / 2) * LANE_W + cs.y;
    c.x = x; c.z = z;
    if (cs.status === "finished" && cs.pos === 1) winner = c;
    if (x > lx) { lx = x; leader = c; }
    const crash = cs.done && cs.status === "crash", yaw = crash ? Math.min(cs.since * 1.2, 0.95) * (cs.lane % 2 ? 1 : -1) : -cs.yaw;
    c.obj.position.set(x, 0, z); c.obj.rotation.y = yaw;
    const accel = THREE.MathUtils.clamp((cs.v - (c.prevV ?? cs.v)) / dt, -30, 40); c.prevV = cs.v;
    u.body.rotation.z = THREE.MathUtils.lerp(u.body.rotation.z, -accel * 0.0035 - (cs.done ? 0 : 0.0), 0.25);
    u.body.rotation.x = crash ? Math.min(cs.since * 0.5, 0.25) : cs.st * -0.05;
    for (const w of u.wheels) { w.spin.rotation.z = -(cs.x + coast) / w.r; if (w.front) w.wg.rotation.y = -cs.st * 0.45; }
    // the fly: wings beat faster with speed, hands turn the wheel, a gear change makes it lean forward
    const f = u.fly, fu = f.userData, beat = Math.sin(st.simT * (24 + cs.v * 0.55) + i);
    fu.wings.forEach((h) => { h.rotation.x = h.userData.sz * (0.55 + 0.55 * beat); h.rotation.y = -0.2 * h.userData.sz * 0; });
    fu.wheel.rotation.z = -cs.st * 1.1; fu.body.rotation.z = cs.sh ? -0.12 : cs.v > 1 ? -0.04 : 0;
    fu.body.position.y = Math.sin(st.simT * 9 + i) * 0.01;
    u.shadow.position.x = 0;
    if (cs.s > 0.12 && !cs.done && Math.random() < 0.9) { puff(x - 2.4, 0.15, z + 0.9, cs.v); puff(x - 2.4, 0.15, z - 0.9, cs.v); }
    if (crash && cs.since < 1.6) puff(x, 0.4, z, 0);
    v3.set(x, 2.75, z).project(camera);                                         // where the HUD puts the name tag
    labels.push({ lane: cs.lane, sx: (v3.x * 0.5 + 0.5) * CW, sy: (-v3.y * 0.5 + 0.5) * CH, front: v3.z < 1 && v3.z > -1, d: camera.position.distanceTo(c.obj.position) });
  });
  const lights = trackGroup && trackGroup.userData.lights;
  if (lights) lights.forEach((m, i) => { const green = st.simT >= st.tGreen; const on = i === 3 ? green : !green && !st.anim; m.color.set(on ? (i === 3 ? 0x35e07a : 0xffb020) : 0x1d2739); });
  const front = cars.filter((c) => c.x !== undefined).sort((a, b) => b.x - a.x).slice(0, 3);
  const zs = front.map((c) => c.z), pack = front.length ? { x: front[0].x, z: zs.reduce((a, b) => a + b, 0) / zs.length, spread: Math.max(...zs) - Math.min(...zs) } : { x: 0, z: 0, spread: 0 };
  director(st, dt, pack, winner);
  stepSmoke(dt);
  renderer.render(scene, camera);
  return labels;
}

window.View3D = { init, setField, render, get ready() { return !!renderer; }, get failed() { return failed; },
  debug: () => ({ cam: camera.position.toArray().map((v) => +v.toFixed(1)), look: look.toArray().map((v) => +v.toFixed(1)), shot: cam.shot, cars: cars.map((c) => [+(c.x ?? 0).toFixed(1), +(c.z ?? 0).toFixed(1)]) }) };
window.dispatchEvent(new Event("view3d-loaded"));
