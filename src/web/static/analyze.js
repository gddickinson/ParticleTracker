// Analyse: ensemble MSD with the diffusion-model fits, the MSD classifier's class for every track compared with the
// simulated motion, and the confinement test (Saxton/Simson level vs the original radius-of-gyration rule).
import { $, app, esc, route } from './core.js';
import { cls, color, fmt, getMeta, getRun, runSettings, where } from './data.js';
import { bars, lines } from './charts.js';
import { calculateMsd } from './analysis_engine.js';

const ALL_CLASSES = ['brownian', 'directed', 'confined', 'subdiffusive', 'superdiffusive', 'insufficient_data'];
const TYPES = ['brownian', 'directed', 'confined'];
const MODEL_COL = { brownian: color('brownian'), anomalous: color('subdiffusive'), directed: color('directed'), confined: color('confined') };

route('analyse', async () => {
  const meta = await getMeta();
  const r = await getRun(runSettings());
  const u = r.units, ps = u.pixel_size_um, dt = u.time_interval_s;
  const match = new Map(r.tracking_scores.per_track.map(p => [p.track_id, p.particle]));
  const truthType = id => { const k = match.get(id); return k == null ? null : meta.ground_truth[k].motion_type; };
  const ens = r.ensemble, F = ens.fits;
  const L = ens.units.length === 'um' ? 'µm' : 'px';

  // ensemble MSD + model curves
  const pts = ens.msd.time_lags.map((t, i) => [t, ens.msd.mean_msd[i]]).filter(([, m]) => m > 0);
  const curve = f => pts.map(([t]) => [t, f(t)]);
  const series = [{ points: pts, color: '#ffffff', label: 'mean MSD' }];
  if (F.brownian?.success) series.push({ points: curve(t => F.brownian.slope * t + F.brownian.intercept), color: color('brownian'), label: 'brownian' });
  if (F.anomalous?.success) series.push({ points: curve(t => 4 * F.anomalous.diffusion_coefficient * t ** F.anomalous.alpha), color: color('subdiffusive'), label: 'anomalous' });
  if (F.directed?.success) series.push({ points: curve(t => 4 * F.directed.diffusion_coefficient * t + (F.directed.velocity * t) ** 2), color: color('directed'), label: 'directed' });
  if (F.confined?.success) series.push({ points: curve(t => F.confined.confinement_area * (1 - Math.exp(-4 * F.confined.diffusion_coefficient * t / F.confined.confinement_area))), color: color('confined'), label: 'confined' });

  // sub/superdiffusive only appear with alpha_classes='anomalous': show their columns only when used
  const CLASSES = ALL_CLASSES.filter(c => !['subdiffusive', 'superdiffusive'].includes(c) || r.class_counts[c]);
  const conf = TYPES.map(t => CLASSES.map(c => r.per_track.filter(p => truthType(p.track_id) === t && p.motion_class === c).length));
  const noPart = CLASSES.map(c => r.per_track.filter(p => truthType(p.track_id) === null && p.motion_class === c).length);
  const long = r.per_track.filter(p => p.length >= 10);
  const confRows = TYPES.map(t => { const ts = long.filter(p => truthType(p.track_id) === t);
    return [t, ts.length, ts.filter(p => p.confined).length, ts.filter(p => p.confined_legacy).length]; });
  const fitRow = (m, f) => !f ? '' : f.success
    ? `<tr><td><span class="cls" style="--c:${MODEL_COL[m]}">${m}</span></td><td>${fmt(f.diffusion_coefficient, 4)}</td>
      <td>${m === 'anomalous' ? fmt(f.alpha) : '–'}</td><td>${m === 'directed' ? fmt(f.velocity, 4) : '–'}</td>
      <td>${m === 'confined' ? fmt(f.confinement_area, 4) : '–'}</td><td>${fmt(f.r_squared)}</td></tr>`
    : `<tr><td>${esc(m)}</td><td colspan="5" class="mute">${esc(f.message || 'fit failed')}</td></tr>`;

  app().innerHTML = `
  <h1>Analyse</h1>
  <p class="lead">Mean squared displacement (<code>calculate_msd</code>, lags in frames), the four diffusion models of
    <code>fit_diffusion_model</code>, and the MSD classifier (<code>classify_track_motion(method='msd')</code>) on the
    ${r.tracks.length} tracks of the current run. Lengths use the demo's assumed scale of ${ps} µm per pixel and ${dt} s per frame.</p>
  <p>${where(r)}</p>
  <div class="two">
    <div class="panel"><h3>Ensemble MSD (${r.ensemble.msd.n_tracks} tracks)</h3>
      ${lines(series, { xlabel: 'time lag (s)', ylabel: `MSD (${L}²)`, logx: true, logy: true })}
      <div class="legend">${series.map(s => `<span class="cls" style="--c:${s.color}">${esc(s.label)}</span>`).join('')}</div></div>
    <div class="panel"><h3>Fits</h3><div class="scroll"><table class="tbl"><thead><tr><th scope="col">model</th><th scope="col">D (${L}²/s)</th>
      <th scope="col">α</th><th scope="col">v (${L}/s)</th><th scope="col">area (${L}²)</th><th scope="col">R²</th></tr></thead>
      <tbody>${['brownian', 'anomalous', 'directed', 'confined'].map(m => fitRow(m, F[m])).join('')}</tbody></table></div>
      <p class="hint">The ensemble mixes the three simulated motion types, so no single model describes it (the confined fit's
        plateau runs off to an implausible area); at long lags each track contributes few pairs, so the mean is noisy. Per-track classes are below.</p>
      <h3>Classes</h3>${bars(CLASSES.filter(c => r.class_counts[c]).map(c => ({ label: c.replace('_', ' '), value: r.class_counts[c], color: color(c) })))}</div>
  </div>
  <h2>Assigned class vs simulated motion</h2>
  <div class="scroll"><table class="tbl conf"><thead><tr><th scope="col">simulated \\ assigned</th>${CLASSES.map(c => `<th scope="col">${cls(c)}</th>`).join('')}</tr></thead>
    <tbody>${TYPES.map((t, i) => `<tr><th scope="row">${cls(t)}</th>${conf[i].map((n, j) => `<td class="${n ? (CLASSES[j] === t ? 'hit' : 'miss') : ''}">${n}</td>`).join('')}</tr>`).join('')}
    <tr><th scope="row" class="mute">on no particle</th>${noPart.map(n => `<td>${n}</td>`).join('')}</tr></tbody></table></div>
  <p class="hint">Each track is matched to the simulated particle most of its points sit on (within 2 px). The classifier
    checks, in order: confinement level L &gt; 2 → confined; drift test p &lt; 0.01 → directed; α &lt; 0.6 → confined,
    α &gt; 1.4 → directed (α from a log-log fit of the track's MSD over lags 1 to a quarter of its length, max 10); else
    brownian. Each test flags about 1–2 % of free-diffusion tracks of 50 points. The thresholds are options; the
    confusion matrices before and after the 2026-11 fix, on thousands of simulated tracks, are in
    <code>docs/msd_classifier.md</code>.</p>
  <h2>Confinement test</h2>
  <div class="scroll"><table class="tbl"><thead><tr><th scope="col">simulated</th><th scope="col">tracks ≥ 10 points</th>
    <th scope="col">confined: Saxton/Simson level &gt; 2</th><th scope="col">confined: original rule (Rg &lt; 2 px)</th></tr></thead>
    <tbody>${confRows.map(([t, n, a, b]) => `<tr><td>${cls(t)}</td><td>${n}</td><td>${a}</td><td>${b}</td></tr>`).join('')}</tbody></table></div>
  <p class="hint">The original rule (still available as <code>detect_confinement(method='rg_legacy')</code>) called a track confined when
    its radius of gyration stayed under 2 pixels in most 10-point windows, whatever the particle's speed or the pixel size,
    so slow free diffusion was confined too. The level L = 2.5117·D·t/R² − 0.2048 (Saxton 1993; Simson et al. 1995), with D from
    the MSD slope between lags 1 and 2, is dimensionless.</p>
  <h2>Tracks</h2>
  <div class="two"><div id="tt" class="scroll tall"></div><div class="panel" id="one"><p class="mute">Select a track to see its MSD.</p></div></div>`;

  $('#tt').innerHTML = `<table class="tbl"><thead><tr><th scope="col">track</th><th scope="col">points</th><th scope="col">particle</th>
    <th scope="col">simulated</th><th scope="col">assigned</th><th scope="col">α</th><th scope="col">D (${L}²/s)</th>
    <th scope="col">net/path</th><th scope="col">drift p</th><th scope="col">level L</th></tr></thead><tbody>
    ${r.per_track.map(p => { const k = match.get(p.track_id); return `<tr tabindex="0" data-id="${p.track_id}"><td>${p.track_id}</td><td>${p.length}</td>
      <td>${k == null ? '–' : '#' + k}</td><td>${k == null ? '–' : cls(meta.ground_truth[k].motion_type)}</td><td>${cls(p.motion_class)}</td>
      <td>${fmt(p.alpha)}</td><td>${fmt(p.D, 4)}</td><td>${fmt(p.directionality)}</td><td>${p.drift_p == null ? '–' : p.drift_p.toExponential(1)}</td><td>${fmt(p.confinement_level, 2)}</td></tr>`; }).join('')}</tbody></table>`;
  const pick = id => {
    const t = r.tracks.find(x => x.track_id === id), p = r.per_track.find(x => x.track_id === id);
    $('#tt').querySelectorAll('tr').forEach(tr => tr.classList.toggle('sel', +tr.dataset.id === id));
    const m = calculateMsd([t], Math.max(4, p.alpha_fit_lags), dt, ps);
    $('#one').innerHTML = `<h3>Track ${id}: ${cls(p.motion_class)}</h3>
      ${lines([{ points: m.time_lags.map((x, i) => [x, m.mean_msd[i]]).filter(([, y]) => y > 0), color: color(p.motion_class) }],
        { xlabel: 'time lag (s)', ylabel: `MSD (${L}²)`, logx: true, logy: true })}
      <p class="hint">Lags 1–${Math.max(4, p.alpha_fit_lags)} frames: the range the classifier fits α over.</p>`;
  };
  $('#tt').querySelectorAll('tr[data-id]').forEach(tr => {
    tr.addEventListener('click', () => pick(+tr.dataset.id));
    tr.addEventListener('keydown', e => { if (e.key === 'Enter') pick(+tr.dataset.id); });
  });
});
