// Port of src/web/scoring.py: detection and tracking scores against the simulated ground truth (pixels).
export const DET_TOL = 2.0, TRACK_TOL = 2.0, RECOVERED = 0.8;

function pairs(dets, truth, tol) {
  const cand = [];
  dets.forEach(([dx, dy], i) => truth.forEach(([tx, ty], j) => {
    const d = Math.hypot(dx - tx, dy - ty);
    if (d <= tol) cand.push([d, i, j]);
  }));
  cand.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const ui = new Set(), uj = new Set(), out = [];
  for (const [d, i, j] of cand) if (!ui.has(i) && !uj.has(j)) { ui.add(i); uj.add(j); out.push(d); }
  return out;
}

export function detectionScores(particles, truth, tol = DET_TOL) {
  let tp = 0, nDet = 0, nTrue = 0, sq = 0;
  particles.forEach((frame, t) => {
    const dets = frame.map(p => [p[0], p[1]]), tru = truth.map(g => [g.x[t], g.y[t]]);
    const d = pairs(dets, tru, tol);
    tp += d.length; sq += d.reduce((s, v) => s + v * v, 0); nDet += dets.length; nTrue += tru.length;
  });
  const precision = nDet ? tp / nDet : 0, recall = nTrue ? tp / nTrue : 0;
  return { tolerance_px: tol, n_detected: nDet, n_true: nTrue, true_positives: tp, false_positives: nDet - tp,
    false_negatives: nTrue - tp, precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0,
    rmse_px: tp ? Math.sqrt(sq / tp) : null };
}

function nearestTruth(x, y, t, truth, tol) {
  let best = null, bd = tol;
  truth.forEach((g, k) => {
    if (t < g.x.length) { const d = Math.hypot(x - g.x[t], y - g.y[t]); if (d <= bd) { best = k; bd = d; } }
  });
  return best;
}

export function trackingScores(tracks, truth, tol = TRACK_TOL) {
  const follow = truth.map(() => new Array(tracks.length).fill(0)), perTrack = [];
  tracks.forEach((tr, j) => {
    const counts = new Map();
    tr.frames.forEach((f, i) => {
      const k = nearestTruth(tr.x[i], tr.y[i], f, truth, tol);
      if (k !== null) { counts.set(k, (counts.get(k) || 0) + 1); follow[k][j]++; }
    });
    if (counts.size) {
      const k = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a - b)[0];
      perTrack.push({ track_id: tr.track_id, particle: k, purity: counts.get(k) / tr.frames.length });
    } else perTrack.push({ track_id: tr.track_id, particle: null, purity: 0 });
  });
  const perParticle = truth.map((g, k) => ({ particle: k, coverage: g.x.length ? (tracks.length ? Math.max(...follow[k]) : 0) / g.x.length : 0,
    tracks: perTrack.filter(p => p.particle === k).length }));
  const cov = perParticle.map(p => p.coverage);
  return { tolerance_px: tol, n_tracks: tracks.length, n_true: truth.length, recovered: cov.filter(c => c >= RECOVERED).length,
    recovered_threshold: RECOVERED, mean_coverage: truth.length ? cov.reduce((a, b) => a + b, 0) / truth.length : 0,
    unmatched_tracks: perTrack.filter(p => p.particle === null).length, per_track: perTrack, per_particle: perParticle };
}
