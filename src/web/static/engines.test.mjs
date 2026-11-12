// Node test: the JS engines reproduce the Python results in tests/fixtures/engine_cases.json
// (written by scripts/make_engine_fixture.py). Run: node src/web/static/engines.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { detectFrame } from './detect_engine.js';
import { linkLap } from './link_engine.js';
import { ensemble, perTrack } from './analysis_engine.js';
import { detectionScores, trackingScores } from './score_engine.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const fx = JSON.parse(readFileSync(join(ROOT, 'tests', 'fixtures', 'engine_cases.json'), 'utf8'));
let checks = 0;
const fail = msg => { throw new Error(msg); };
function close(a, b, path, rtol = 1e-9, atol = 1e-12) {
  checks++;
  if (a === null || b === null || a === undefined || b === undefined) {
    const na = a == null || (typeof a === 'number' && Number.isNaN(a)), nb = b == null || (typeof b === 'number' && Number.isNaN(b));
    if (na !== nb) fail(`${path}: ${a} vs ${b}`);
    return;
  }
  if (typeof b === 'number') {
    if (!(Math.abs(a - b) <= atol + rtol * Math.abs(b))) fail(`${path}: JS ${a} vs Python ${b}`);
  } else if (typeof b === 'string' || typeof b === 'boolean') {
    if (a !== b) fail(`${path}: JS ${a} vs Python ${b}`);
  } else if (Array.isArray(b)) {
    if (!Array.isArray(a) || a.length !== b.length) fail(`${path}: length ${a && a.length} vs ${b.length}`);
    b.forEach((v, i) => close(a[i], v, `${path}[${i}]`, rtol, atol));
  } else {
    for (const k of Object.keys(b)) close(a[k], b[k], `${path}.${k}`, rtol, atol);
  }
}

// detection: x, y exact; intensity exact; snr float32
const [T, H, W] = fx.movie.shape;
const raw = readFileSync(join(ROOT, fx.movie.file));
const movie = new Uint16Array(raw.buffer, raw.byteOffset, raw.length / 2);
fx.detect_cases.forEach((c, ci) => c.frames.forEach((f, k) => {
  const got = detectFrame(Float32Array.from(movie.subarray(f * H * W, (f + 1) * H * W)), H, W, c.params);
  close(got, c.expected[k], `detect[${ci}] frame ${f}`, 1e-6);
}));
console.log(`detection: ${fx.detect_cases.length} parameter sets x ${fx.detect_cases[0].frames.length} frames ok`);

close(detectionScores(fx.particles, fx.truth), fx.detection_scores, 'detection_scores');

const fitTol = { brownian: 1e-9, anomalous: 1e-9, directed: 1e-5 };
fx.link_cases.forEach((c, ci) => {
  const tracks = linkLap(fx.particles, c.params);
  close(tracks, c.tracks, `link[${ci}].tracks`);
  const rows = tracks.map(t => perTrack(t, fx.pixel_size, fx.time_interval));
  rows.forEach((r, i) => {
    const e = c.per_track[i];
    for (const k of ['track_id', 'length', 'motion_class', 'directionality', 'confinement_level', 'confined', 'confined_legacy', 'D'])
      close(r[k], e[k], `link[${ci}].per_track[${i}].${k}`, 1e-8);
    close(r.alpha, e.alpha, `link[${ci}].per_track[${i}].alpha`, 1e-8);
  });
  const ens = ensemble(tracks, fx.pixel_size, fx.time_interval, 50);
  close(ens.msd, c.ensemble.msd, `link[${ci}].msd`);
  for (const m of Object.keys(c.ensemble.fits)) {
    const e = c.ensemble.fits[m], g = ens.fits[m];
    for (const k of Object.keys(e)) if (typeof e[k] === 'number') close(g[k], e[k], `link[${ci}].fits.${m}.${k}`, fitTol[m], 1e-7);
  }
  close(trackingScores(tracks, fx.truth), c.tracking_scores, `link[${ci}].tracking_scores`);
  console.log(`link+analysis case ${ci} (${JSON.stringify(c.params)}): ${tracks.length} tracks ok`);
});
console.log(`engines: ${checks} checks ok`);
