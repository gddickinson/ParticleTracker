// Port of the LAP tracker (src/tracking/lap_tracker.py: frame-to-frame LAP, then gap closing) and the pipeline's
// min_track_length filter (ParticleTracker.track). Tested against Python in engines.test.mjs.
const BIG = 1e12;   // "forbidden" in the augmented matrix, as solve_augmented_lap

// Minimum-cost assignment of a square cost matrix (rows of numbers): Hungarian method with potentials
// (shortest augmenting paths). Returns col[row].
export function hungarian(a) {
  const n = a.length, INF = Infinity;
  const u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0], row = a[i0 - 1];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = row[j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const col = new Int32Array(n);
  for (let j = 1; j <= n; j++) if (p[j]) col[p[j] - 1] = j - 1;
  return col;
}

// solve_augmented_lap(cost, death, birth): real (row, col) links of the U-track augmented problem.
export function solveAugmentedLap(cost, deathCost, birthCost) {
  const nr = cost.length, nc = nr ? cost[0].length : 0;
  if (!nr || !nc) return [];
  const n = nr + nc, a = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) a[i][j] = Number.isFinite(cost[i][j]) ? cost[i][j] : BIG;
  for (let i = 0; i < nr; i++) for (let j = 0; j < nr; j++) a[i][nc + j] = i === j ? deathCost : BIG;
  for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) a[nr + i][j] = i === j ? birthCost : BIG;
  const col = hungarian(a), out = [];
  for (let r = 0; r < nr; r++) if (col[r] < nc && Number.isFinite(cost[r][col[r]])) out.push([r, col[r]]);
  return out;
}

const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
const sortT = d => d.sort((p, q) => p.t - q.t);
const endOf = tr => tr.dets[tr.dets.length - 1].t;

export const LINK_DEFAULTS = { max_displacement: 5.0, gap_closing: 3, min_track_length: 5,
  max_gap_displacement: 10.0, motion_model: 'brownian', brownian_search_radius_multiplier: 2.0 };

function frameToFrame(byFrame, frames, p) {
  const tracks = [];
  const start = d => tracks.push({ track_id: tracks.length, dets: [d], gaps: [] });
  byFrame.get(frames[0]).forEach(start);
  const maxCost = p.max_displacement * 2;
  for (let i = 0; i < frames.length - 1; i++) {
    const cur = frames[i], next = byFrame.get(frames[i + 1]);
    const active = tracks.filter(t => endOf(t) === cur);
    if (!active.length) { next.forEach(start); continue; }
    if (!next.length) continue;
    const cost = active.map(t => {
      const last = t.dets[t.dets.length - 1];
      return next.map(d => {
        const r = p.motion_model === 'brownian'
          ? p.max_displacement * p.brownian_search_radius_multiplier * Math.sqrt(d.t - last.t) : p.max_displacement;
        const dd = dist(last, d);
        return dd < r ? dd : Infinity;
      });
    });
    const links = solveAugmentedLap(cost, maxCost * 0.5, maxCost * 0.5).filter(([r, c]) => cost[r][c] < maxCost);
    for (const [r, c] of links) { const t = tracks[active[r].track_id]; t.dets.push(next[c]); sortT(t.dets); }
    const assigned = new Set(links.map(l => l[1]));
    next.forEach((d, j) => { if (!assigned.has(j)) start(d); });
  }
  return tracks;
}

function gapClose(segs, p) {
  const ends = [], starts = [];
  segs.forEach((s, i) => { if (s.dets.length) { ends.push([i, s.dets[s.dets.length - 1]]); starts.push([i, s.dets[0]]); } });
  if (!ends.length || !starts.length) return segs;
  let any = false;
  const cost = ends.map(([i, e]) => starts.map(([j, s]) => {
    if (i === j) return Infinity;
    const gap = s.t - e.t;
    if (gap > 0 && gap <= p.gap_closing) {
      const d = Math.sqrt((e.x - s.x) ** 2 + (e.y - s.y) ** 2);
      if (d < p.max_gap_displacement * Math.sqrt(gap)) { any = true; return d + gap * 0.5; }
    }
    return Infinity;
  }));
  if (!any) return segs;
  const maxGap = p.max_gap_displacement * 2, next = new Map();
  for (const [r, c] of solveAugmentedLap(cost, maxGap * 0.5, maxGap * 0.5)) if (cost[r][c] < maxGap) next.set(ends[r][0], starts[c][0]);
  const continued = new Set(next.values()), out = [];
  segs.forEach((tr, i) => {
    if (continued.has(i)) return;
    let j = i;
    while (next.has(j)) {
      const nx = segs[next.get(j)];
      for (let g = endOf(segs[j]) + 1; g < nx.dets[0].t; g++) if (!tr.gaps.includes(g)) tr.gaps.push(g);
      for (const g of nx.gaps) if (!tr.gaps.includes(g)) tr.gaps.push(g);
      tr.gaps.sort((a, b) => a - b);
      for (const d of nx.dets) tr.dets.push(d);
      sortT(tr.dets);
      j = next.get(j);
    }
    out.push(tr);
  });
  return out;
}

// particles: per frame [[x, y, ...], ...] -> tracks [{track_id, frames, x, y, gaps}] (ParticleTracker.track('lap')).
export function linkLap(particles, params = {}) {
  const p = { ...LINK_DEFAULTS, ...params };
  const byFrame = new Map();
  particles.forEach((f, t) => f.forEach(q => {
    if (!byFrame.has(t)) byFrame.set(t, []);
    byFrame.get(t).push({ x: q[0], y: q[1], t });
  }));
  if (!byFrame.size) return [];
  const frames = [...byFrame.keys()].sort((a, b) => a - b);
  let tracks = frameToFrame(byFrame, frames, p);
  tracks = gapClose(tracks, p);   // merge_splits defaults to True, so gap closing always runs
  const minLen = p.min_track_length || 0;
  return tracks.filter(t => !minLen || t.dets.length >= minLen).map(t => ({
    track_id: t.track_id, frames: t.dets.map(d => d.t), x: t.dets.map(d => d.x), y: t.dets.map(d => d.y), gaps: t.gaps }));
}
