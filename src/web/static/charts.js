// Small SVG charts (no library): horizontal bars and a line chart.
import { esc } from './core.js';
import { fmt } from './data.js';

// rows: [{label, value, color}]
export function bars(rows, { width = 520, row = 24, unit = '' } = {}) {
  const max = Math.max(1e-12, ...rows.map(r => r.value)), lw = 120, h = rows.length * row + 6;
  return `<svg class="chart" viewBox="0 0 ${width} ${h}" role="img" aria-label="${esc(rows.map(r => `${r.label} ${fmt(r.value)}`).join(', '))}">
  ${rows.map((r, i) => {
    const w = (width - lw - 70) * r.value / max, y = i * row + 4;
    return `<text x="0" y="${y + row * 0.62}">${esc(r.label)}</text>
    <rect x="${lw}" y="${y}" width="${Math.max(1, w)}" height="${row - 8}" rx="2" fill="${r.color || 'var(--accent)'}"></rect>
    <text x="${lw + w + 6}" y="${y + row * 0.62}">${fmt(r.value)}${esc(unit)}</text>`;
  }).join('')}</svg>`;
}

// series: [{points: [[x, y]], color, label}]
export function lines(series, { width = 520, height = 220, xlabel = '', ylabel = '', logx = false, logy = false } = {}) {
  const tx = v => (logx ? Math.log10(v) : v), ty = v => (logy ? Math.log10(v) : v);
  const pts = series.flatMap(s => s.points).filter(([x, y]) => (!logx || x > 0) && (!logy || y > 0));
  if (!pts.length) return '<p class="mute">No data to plot.</p>';
  const xs = pts.map(p => tx(p[0])), ys = pts.map(p => ty(p[1]));
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) y1 = y0 + 1;
  const L = 52, B = 34, T = 10, R = 12;
  const X = v => L + (tx(v) - x0) / (x1 - x0) * (width - L - R), Y = v => height - B - (ty(v) - y0) / (y1 - y0) * (height - B - T);
  const tick = v => (logx || logy ? fmt(10 ** v, 2) : fmt(v, 1));
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ylabel)} against ${esc(xlabel)}">
  <line class="axis" x1="${L}" y1="${height - B}" x2="${width - R}" y2="${height - B}"></line>
  <line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${height - B}"></line>
  <text x="${L}" y="${height - B + 14}">${logx ? fmt(10 ** x0, 2) : fmt(x0, 1)}</text>
  <text x="${width - R}" y="${height - B + 14}" text-anchor="end">${logx ? fmt(10 ** x1, 2) : fmt(x1, 1)}</text>
  <text x="${L - 4}" y="${height - B}" text-anchor="end">${logy ? fmt(10 ** y0, 2) : tick(y0)}</text>
  <text x="${L - 4}" y="${T + 9}" text-anchor="end">${logy ? fmt(10 ** y1, 2) : tick(y1)}</text>
  <text x="${(L + width - R) / 2}" y="${height - 4}" text-anchor="middle">${esc(xlabel)}</text>
  <text x="12" y="${(T + height - B) / 2}" transform="rotate(-90 12 ${(T + height - B) / 2})" text-anchor="middle">${esc(ylabel)}</text>
  ${series.map(s => `<polyline fill="none" stroke="${s.color || 'var(--accent)'}" stroke-width="1.6" points="${s.points
    .filter(([x, y]) => (!logx || x > 0) && (!logy || y > 0)).map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ')}"></polyline>`).join('')}
  </svg>`;
}

// Mean squared displacement of one track for lags 1..maxLag (in frames, from consecutive points), for display.
export function msd(t, maxLag = 15) {
  const out = [], n = t.x.length;
  for (let lag = 1; lag < Math.min(n, maxLag); lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += (t.x[i + lag] - t.x[i]) ** 2 + (t.y[i + lag] - t.y[i]) ** 2;
    out.push([lag, s / (n - lag)]);
  }
  return out;
}
