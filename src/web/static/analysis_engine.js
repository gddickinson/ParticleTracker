// Port of the track analysis (src/analysis/msd.py, motion.py, confinement.py, msd_classifier.py) and
// of the web layer's per_track() / ensemble() (src/web/analysis.py). Tracks are {frames, x, y} in pixels and frames.
// Fits: 'brownian' and 'anomalous' are numpy.polyfit straight lines; 'directed' (MSD = 4Dt + (vt)^2, D, v >= 0) is
// solved exactly as non-negative least squares in (4D, v^2), the same optimum scipy's curve_fit converges to; the
// 'confined' fit (nonlinear) runs only in Python. Tested against Python in engines.test.mjs.

// frame_msd: MSD at frame lags 1..maxLag from all pairs exactly `lag` frames apart -> {msd, n} (NaN where none).
export function frameMsd(frames, xs, ys, maxLag, scale = 1) {
  const msd = new Array(maxLag).fill(NaN), n = new Array(maxLag).fill(0);
  if (!frames.length || maxLag < 1) return { msd, n };
  const idx = frames.map((f, i) => i).sort((a, b) => frames[a] - frames[b]);
  const f0 = frames[idx[0]], span = frames[idx[idx.length - 1]] - f0;
  const X = new Array(span + 1).fill(NaN), Y = new Array(span + 1).fill(NaN);
  for (let k = idx.length - 1; k >= 0; k--) { const i = idx[k]; X[frames[i] - f0] = xs[i] * scale; Y[frames[i] - f0] = ys[i] * scale; }
  for (let lag = 1; lag <= Math.min(maxLag, span); lag++) {
    let s = 0, c = 0;
    for (let i = 0; i + lag <= span; i++) {
      const d = (X[i + lag] - X[i]) ** 2 + (Y[i + lag] - Y[i]) ** 2;
      if (Number.isFinite(d)) { s += d; c++; }
    }
    if (c) { msd[lag - 1] = s / c; n[lag - 1] = c; }
  }
  return { msd, n };
}

// calculate_msd(tracks, max_lag, time_interval, pixel_size): ensemble over tracks of >= 5 points.
export function calculateMsd(tracks, maxLag, timeInterval = null, pixelSize = null) {
  const lags = Array.from({ length: maxLag }, (_, i) => i + 1), dt = timeInterval ?? 1, scale = pixelSize ?? 1;
  const units = { length: pixelSize == null ? 'px' : 'um', time: timeInterval == null ? 'frame' : 's' };
  const all = [];
  for (const t of tracks) {
    if (t.frames.length < 5) continue;
    const r = frameMsd(t.frames, t.x, t.y, maxLag, scale);
    if (r.n.some(v => v > 0)) all.push(r.msd);
  }
  const time_lags = lags.map(l => l * dt);
  if (!all.length) return { lags, time_lags, mean_msd: lags.map(() => 0), std_msd: lags.map(() => 0),
    n_tracks_per_lag: lags.map(() => 0), n_tracks: 0, units };
  const mean = [], std = [], nper = [];
  for (let k = 0; k < maxLag; k++) {
    const v = all.map(m => m[k]).filter(Number.isFinite);
    nper.push(v.length);
    if (!v.length) { mean.push(NaN); std.push(NaN); continue; }
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    mean.push(m);
    std.push(Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length));
  }
  return { lags, time_lags, mean_msd: mean, std_msd: std, n_tracks_per_lag: nper, n_tracks: all.length, units };
}

function line(x, y) {   // least-squares y = a x + b
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const a = sxy / sxx;
  return [a, my - a * mx];
}
function r2(obs, pred) {
  const m = obs.reduce((s, v) => s + v, 0) / obs.length;
  let res = 0, tot = 0;
  for (let i = 0; i < obs.length; i++) { res += (obs[i] - pred[i]) ** 2; tot += (obs[i] - m) ** 2; }
  return tot > 0 ? 1 - res / tot : 0;
}

// fit_diffusion_model(msd, model) for 'brownian', 'anomalous', 'directed'.
export function fitModel(msd, model) {
  const t = [], m = [];
  msd.mean_msd.forEach((v, i) => { if (v > 0 && Number.isFinite(v)) { t.push(msd.time_lags[i]); m.push(v); } });
  if (t.length < 3) return { success: false, message: 'Insufficient data for fitting' };
  if (model === 'brownian') {
    const [slope, intercept] = line(t, m);
    return { success: true, model, diffusion_coefficient: slope / 4, slope, intercept, r_squared: r2(m, t.map(v => slope * v + intercept)) };
  }
  if (model === 'anomalous') {
    const lt = t.map(Math.log), lm = m.map(Math.log), [alpha, l4d] = line(lt, lm);
    return { success: true, model, diffusion_coefficient: Math.exp(l4d) / 4, alpha,
      motion_type: alpha < 0.9 ? 'subdiffusive' : alpha > 1.1 ? 'superdiffusive' : 'normal_diffusion',
      r_squared: r2(lm, lt.map(v => alpha * v + l4d)) };
  }
  if (model === 'directed') {
    // minimise sum (a t + b t^2 - m)^2 with a, b >= 0  (a = 4D, b = v^2)
    let s11 = 0, s12 = 0, s22 = 0, y1 = 0, y2 = 0;
    t.forEach((v, i) => { s11 += v * v; s12 += v ** 3; s22 += v ** 4; y1 += v * m[i]; y2 += v * v * m[i]; });
    let a = (y1 * s22 - y2 * s12) / (s11 * s22 - s12 * s12), b = (y2 * s11 - y1 * s12) / (s11 * s22 - s12 * s12);
    if (!(a >= 0 && b >= 0)) {
      const cands = [[Math.max(0, y1 / s11), 0], [0, Math.max(0, y2 / s22)]];
      const sse = ([p, q]) => t.reduce((s, v, i) => s + (p * v + q * v * v - m[i]) ** 2, 0);
      [a, b] = sse(cands[0]) <= sse(cands[1]) ? cands[0] : cands[1];
    }
    const D = a / 4, vel = Math.sqrt(b);
    return { success: true, model, diffusion_coefficient: D, velocity: vel, r_squared: r2(m, t.map(v => 4 * D * v + (vel * v) ** 2)) };
  }
  return { success: false, message: `Model '${model}' runs in Python only` };
}

export function directionality(tr) {
  const n = tr.x.length;
  if (n < 3) return 0;
  const net = Math.sqrt((tr.x[n - 1] - tr.x[0]) ** 2 + (tr.y[n - 1] - tr.y[0]) ** 2);
  let path = 0;
  for (let i = 0; i < n - 1; i++) path += Math.sqrt((tr.x[i + 1] - tr.x[i]) ** 2 + (tr.y[i + 1] - tr.y[i]) ** 2);
  return path > 0 ? net / path : 0;
}

// Saxton/Simson confinement level (confinement.py): 2.5117 D t / R^2 - 0.2048, NaN if undefined.
export const CONFINEMENT_LEVEL_THRESHOLD = 3.16;
export function confinementLevel(tr, fitLags = 4) {
  if (tr.frames.length < 2) return NaN;
  const { msd, n } = frameMsd(tr.frames, tr.x, tr.y, fitLags);
  const lt = [], lm = [];
  n.forEach((c, i) => { if (c > 0) { lt.push(i + 1); lm.push(msd[i]); } });
  if (lt.length < Math.min(3, fitLags)) return NaN;
  const D = line(lt, lm)[0] / 4;
  const idx = tr.frames.map((f, i) => i).sort((a, b) => tr.frames[a] - tr.frames[b]);
  const x0 = tr.x[idx[0]], y0 = tr.y[idx[0]];
  const T = tr.frames[idx[idx.length - 1]] - tr.frames[idx[0]];
  let r2max = 0;
  for (const i of idx) r2max = Math.max(r2max, (tr.x[i] - x0) ** 2 + (tr.y[i] - y0) ** 2);
  if (!Number.isFinite(D) || r2max === 0) return NaN;
  return 2.5117 * D * T / r2max - 0.2048;
}

// Original confinement rule, kept for comparison: Rg < 2 px in > 70 % of 10-point windows.
export function confinedLegacy(tr, win = 10) {
  const n = tr.x.length;
  if (n < win) return false;
  let c = 0;
  for (let i = 0; i + win <= n; i++) {
    let mx = 0, my = 0;
    for (let k = i; k < i + win; k++) { mx += tr.x[k]; my += tr.y[k]; }
    mx /= win; my /= win;
    let s = 0;
    for (let k = i; k < i + win; k++) s += (tr.x[k] - mx) ** 2 + (tr.y[k] - my) ** 2;
    if (Math.sqrt(s / win) < 2) c++;
  }
  return c / (n - win + 1) > 0.7;
}

// analysis/msd_classifier.py: confinement level, drift test, anomalous exponent (see that module for the rules).
export const MSD_CLASSIFIER_DEFAULTS = { min_length: 10, fit_fraction: 0.25, max_fit_lag: 10, alpha_sub: 0.6,
  alpha_super: 1.4, drift_p: 0.01, confinement_fit_lags: 2, confinement_threshold: 2.0, alpha_classes: 'motion' };

export const alphaFitLags = (n, frac = 0.25, cap = 10) => Math.min(Math.max(Math.trunc(n * frac), 4), cap);

export function anomalousAlpha(tr, maxLag) {
  const { msd, n } = frameMsd(tr.frames, tr.x, tr.y, maxLag);
  const lx = [], ly = [];
  n.forEach((c, i) => { if (c > 0 && msd[i] > 0) { lx.push(Math.log(i + 1)); ly.push(Math.log(msd[i])); } });
  return lx.length < 3 ? NaN : line(lx, ly)[0];
}

export function driftTest(tr) {
  const m = tr.frames.length;
  if (m < 3) return { z2: NaN, p: NaN };
  const idx = tr.frames.map((f, i) => i).sort((a, b) => tr.frames[a] - tr.frames[b]);
  const f = idx.map(i => Math.round(tr.frames[i])), x = idx.map(i => tr.x[i]), y = idx.map(i => tr.y[i]);
  const T = f[m - 1] - f[0], nx = x[m - 1] - x[0], ny = y[m - 1] - y[0];
  const steps = [];
  for (let i = 0; i + 1 < m; i++) { const dt = f[i + 1] - f[i]; if (dt > 0) steps.push([x[i + 1] - x[i], y[i + 1] - y[i], dt]); }
  if (steps.length < 2 || T <= 0) return { z2: NaN, p: NaN };
  const vx = nx / T, vy = ny / T;
  let s = 0;
  for (const [dx, dy, dt] of steps) s += ((dx - dt * vx) ** 2 + (dy - dt * vy) ** 2) / dt;
  const s2 = s / (2 * (steps.length - 1));
  if (!(s2 > 0)) return { z2: NaN, p: NaN };
  const z2 = (nx * nx + ny * ny) / (T * s2);
  return { z2, p: Math.exp(-z2 / 2) };
}

export function msdTrackFeatures(tr, opts = {}) {
  const o = { ...MSD_CLASSIFIER_DEFAULTS, ...opts }, n = tr.frames.length;
  const K = alphaFitLags(n, o.fit_fraction, o.max_fit_lag);
  const d = n >= 3 ? driftTest(tr) : { z2: NaN, p: NaN };
  return { length: n, alpha_fit_lags: K, alpha: n >= 4 ? anomalousAlpha(tr, K) : NaN, drift_z2: d.z2, drift_p: d.p,
    confinement_level: n >= 2 ? confinementLevel(tr, o.confinement_fit_lags) : NaN };
}

export function classifyFeatures(f, opts = {}) {
  const o = { ...MSD_CLASSIFIER_DEFAULTS, ...opts }, motion = o.alpha_classes === 'motion';
  if (f.length < o.min_length) return 'insufficient_data';
  if (Number.isFinite(f.confinement_level) && f.confinement_level > o.confinement_threshold) return 'confined';
  if (Number.isFinite(f.drift_p) && f.drift_p < o.drift_p) return 'directed';
  if (Number.isFinite(f.alpha) && f.alpha < o.alpha_sub) return motion ? 'confined' : 'subdiffusive';
  if (Number.isFinite(f.alpha) && f.alpha > o.alpha_super) return motion ? 'directed' : 'superdiffusive';
  return 'brownian';
}

export const classifyTrack = (tr, opts = {}) => classifyFeatures(msdTrackFeatures(tr, opts), opts);

// web/analysis.per_track
export function perTrack(tr, pixelSize, timeInterval) {
  const n = tr.frames.length, f = msdTrackFeatures(tr), ok = n >= MSD_CLASSIFIER_DEFAULTS.min_length;
  const row = { track_id: tr.track_id, length: n, motion_class: classifyFeatures(f),
    alpha: ok && Number.isFinite(f.alpha) ? f.alpha : null, alpha_fit_lags: f.alpha_fit_lags,
    drift_p: ok && Number.isFinite(f.drift_p) ? f.drift_p : null };
  if (n >= 5) {
    const br = fitModel(calculateMsd([tr], Math.max(3, Math.min(n >> 1, 20)), timeInterval, pixelSize), 'brownian');
    row.D = br.success ? br.diffusion_coefficient : null;
  }
  row.directionality = directionality(tr);
  const L = ok ? f.confinement_level : NaN;
  row.confinement_level = Number.isFinite(L) ? L : null;
  row.confined = Number.isFinite(L) && L > MSD_CLASSIFIER_DEFAULTS.confinement_threshold;
  row.confined_legacy = confinedLegacy(tr);
  return row;
}

// web/analysis.ensemble without the 'confined' model
export function ensemble(tracks, pixelSize, timeInterval, maxLag = 50) {
  const msd = calculateMsd(tracks, maxLag, timeInterval, pixelSize), fits = {};
  for (const m of ['brownian', 'anomalous', 'directed']) fits[m] = fitModel(msd, m);
  fits.confined = { success: false, message: "The confined model is fitted by scipy's curve_fit: Python only" };
  return { msd, fits, units: msd.units };
}
