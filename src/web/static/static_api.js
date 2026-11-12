// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs what the
// browser can with the JS ports of the repo's code (node-tested against Python): pixel-level wavelet detection, LAP
// linking, the MSD analysis and classification, and the ground-truth scores. Everything else needs the local app.
import { detectMovie } from './detect_engine.js';
import { linkLap } from './link_engine.js';
import { ensemble, perTrack } from './analysis_engine.js';
import { detectionScores, trackingScores } from './score_engine.js';
import { getMovie } from './data.js';

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body) {
    if (clean === 'demo/run') return run(body);
    throw new Error('Running the pipeline on your own files ' + NEEDS_SERVER + '.');
  }
  return file(clean);
}

const same = (a, b) => Object.keys({ ...a, ...b }).every(k => a[k] === b[k]);

async function particlesFor(detector, dp, info) {
  if (detector === 'wavelet' && dp.subpixel === false) {
    const m = await getMovie();
    return { particles: detectMovie(m.data, m.shape, dp), computed_in: 'browser' };
  }
  const defaults = detector === 'wavelet' ? { ...info.defaults.detection } : {};
  delete defaults.method;
  if (!same({ ...dp, subpixel: undefined }, { ...defaults, subpixel: undefined }))
    throw new Error(`Changing the ${detector} detector's parameters ${NEEDS_SERVER}. On this site the wavelet detector runs in ` +
      'your browser with sub-pixel refinement off; the other detectors are shown with their default parameters.');
  const src = detector === 'wavelet' ? await file('demo/pipeline') : await file(`demo/detections/${detector}`);
  return { particles: src.particles, computed_in: 'build+browser' };
}

async function run({ detector = 'wavelet', det_params = {}, link_params = {} }) {
  const [info, meta] = await Promise.all([file('info'), file('demo/meta')]);
  const { particles, computed_in } = await particlesFor(detector, det_params, info);
  const lp = { ...info.defaults.tracking, ...link_params };
  const tracks = linkLap(particles, lp);
  const ps = meta.pixel_size, dt = meta.time_interval;
  const rows = tracks.map(t => perTrack(t, ps, dt));
  const class_counts = {};
  rows.forEach(r => { class_counts[r.motion_class] = (class_counts[r.motion_class] || 0) + 1; });
  return {
    detector, det_params, link_params: lp, computed_in,
    units: { length: 'um', time: 's', pixel_size_um: ps, time_interval_s: dt },
    particles, tracks, per_track: rows, class_counts,
    ensemble: ensemble(tracks, ps, dt, info.defaults.analysis.msd_lag_max),
    detection_scores: detectionScores(particles, meta.ground_truth),
    tracking_scores: trackingScores(tracks, meta.ground_truth),
  };
}
