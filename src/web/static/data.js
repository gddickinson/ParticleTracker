// Shared data for the pages: project info, the demo movie (raw uint16 from static/demo/movie.u16), the current run
// (detector + linking parameters, shared by the Detect, Track and Analyse pages), colours and number formatting.
import { api, esc } from './core.js';

export const getInfo = () => api('/info');
export const getMeta = () => api('/demo/meta');

let moviePromise = null;
// The demo movie as { data: Uint16Array (T*H*W, row-major), shape: [T, H, W], lo, hi } (lo/hi: display range).
export function getMovie() {
  if (!moviePromise) moviePromise = (async () => {
    const meta = await getMeta();
    const r = await fetch(meta.file);
    if (!r.ok) throw new Error(`demo movie: ${r.status}`);
    const data = new Uint16Array(await r.arrayBuffer());
    if (data.length !== meta.shape.reduce((a, b) => a * b, 1)) throw new Error('demo movie has the wrong size');
    return { data, shape: meta.shape, lo: meta.display[0], hi: meta.display[1] };
  })();
  return moviePromise;
}

// The run the pages show: null = the default pipeline (precomputed); else the body for POST /demo/run.
let current = null, currentResult = null, currentKey = '';
export const runSettings = () => current;
export async function getRun(settings = current) {
  const key = JSON.stringify(settings);
  current = settings;
  if (currentResult && key === currentKey) return currentResult;
  const r = settings ? await api('/demo/run', { body: settings }) : await api('/demo/pipeline');
  if (JSON.stringify(current) === key) { currentResult = r; currentKey = key; }
  return r;
}
export const lastRun = () => currentResult;

export const COLORS = { brownian: '#5ce1ff', directed: '#ff4fd8', confined: '#ffd166', subdiffusive: '#ff9a5c',
  superdiffusive: '#c18bff', insufficient_data: '#5c637a', none: '#5c637a' };
export const color = t => COLORS[t] || '#5c637a';
export const cls = t => `<span class="cls" style="--c:${color(t)}">${esc(String(t).replace('_', ' '))}</span>`;
export const legend = types => `<div class="legend">${types.map(cls).join('')}</div>`;

const TRACK_COLS = ['#5ce1ff', '#ffd166', '#ff4fd8', '#9bff6b', '#c18bff', '#ff9a5c', '#7aa8ff', '#38f5c0'];
export const trackColor = id => TRACK_COLS[id % TRACK_COLS.length];

export function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (typeof v !== 'number') return esc(String(v));
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : v.toFixed(d);
}
export const pct = v => (v == null ? '–' : `${(100 * v).toFixed(1)}%`);

// Badge saying where a result was computed.
export function where(r) {
  if (r.computed_in === 'browser') return '<span class="badge js">computed in your browser</span>';
  if (r.computed_in === 'build+browser') return '<span class="badge py">detections: Python (site build)</span> <span class="badge js">linking + analysis: your browser</span>';
  return '<span class="badge py">computed by the Python code</span>';
}
