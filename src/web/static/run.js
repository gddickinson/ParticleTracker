// Your data: the pipeline on an uploaded image stack (local app only): detection, LAP linking, MSD analysis.
import { $, api, app, download, esc, isStatic, needsServer, route } from './core.js';
import { bars } from './charts.js';
import { cls, color, fmt, getInfo } from './data.js';
import { Viewer } from './viewer.js';

const SRC = { file: 'read from the file', argument: 'typed in the form', default: 'not in the file: ImageStack default' };

route('run', async () => {
  const info = await getInfo();
  const d = info.defaults;
  const num = (id, label, v, hint = '') => `<div><label for="${id}">${label}</label><input id="${id}" type="number" step="any" value="${v}">${hint}</div>`;
  app().innerHTML = `
  <h1>Your data</h1>
  <p class="lead">Upload a single-channel time series (TIFF / OME-TIFF stack, or a .npy array) and run the pipeline:
    detection, LAP linking and the MSD analysis, as <code>ParticleTracker</code> does.</p>
  ${isStatic() ? needsServer('Analysing your own files') : ''}
  <form id="f" class="panel params">
    <label for="file">Image stack (frames × height × width)</label>
    <input id="file" type="file" accept=".tif,.tiff,.npy" required>
    <fieldset><legend>Physical scale</legend><div class="grid2">
      ${num('ps', 'Pixel size (µm)', '', '<p class="hint">Empty: use the value stored in the file (ImageJ/OME).</p>')}
      ${num('dt', 'Frame interval (s)', '', '<p class="hint">Empty: use the value stored in the file.</p>')}</div>
      <p class="hint">If neither the form nor the file gives a value, the ImageStack defaults (0.1 µm, 0.1 s) are used and the result says so.</p></fieldset>
    <fieldset><legend>Detection</legend><div class="grid2">
      <div><label for="det">Detector</label><select id="det">${info.detectors.map(m => `<option>${m}</option>`).join('')}</select></div>
      ${num('sigma', 'Wavelet sigma (px)', d.detection.sigma)}${num('thr', 'Wavelet threshold (noise SDs)', d.detection.threshold)}
      ${num('md', 'Wavelet min. spacing (px)', d.detection.min_distance)}</div>
      <p class="hint">The three wavelet parameters apply to the wavelet detector; the others use their own defaults.</p></fieldset>
    <fieldset><legend>Linking (LAP)</legend><div class="grid2">
      ${num('maxd', 'Max displacement (px/frame)', d.tracking.max_displacement)}${num('gap', 'Gap closing (frames)', d.tracking.gap_closing)}
      ${num('minl', 'Minimum track length', d.tracking.min_track_length)}</div></fieldset>
    <button type="submit" ${isStatic() ? 'disabled' : ''}>Run</button>
  </form>
  <div id="out" aria-live="polite"></div>`;

  $('#f').addEventListener('submit', async e => {
    e.preventDefault();
    const v = id => $('#' + id).value.trim();
    const det = v('det');
    const p = { detector: det, det_params: det === 'wavelet' ? { sigma: +v('sigma'), threshold: +v('thr'), min_distance: +v('md') } : {},
      link_params: { max_displacement: +v('maxd'), gap_closing: +v('gap'), min_track_length: +v('minl') } };
    if (v('ps')) p.pixel_size = +v('ps');
    if (v('dt')) p.time_interval = +v('dt');
    const fd = new FormData();
    fd.append('file', $('#file').files[0]);
    fd.append('parameters', JSON.stringify(p));
    $('#out').innerHTML = '<p class="mute">Running in Python...</p>';
    try { show(await api('/run', { body: fd })); } catch (err) { $('#out').innerHTML = `<p class="err">${esc(err.message)}</p>`; }
  });
});

function show(r) {
  const [, H, W] = r.shape, u = r.units, f = r.ensemble.fits;
  $('#out').innerHTML = `<h2>${esc(r.file)}</h2>
    <p class="mute">${r.shape.join(' × ')} · ${fmt(u.pixel_size_um, 4)} µm/px (${esc(SRC[r.scale_source.pixel_size] || r.scale_source.pixel_size)}) ·
      ${fmt(u.time_interval_s, 4)} s/frame (${esc(SRC[r.scale_source.time_interval] || r.scale_source.time_interval)})</p>
    <div class="two"><div><div id="pv"></div><p class="hint">All tracks (pixels), coloured by assigned class.</p></div>
    <div class="panel"><dl class="kv"><dt>detections</dt><dd>${fmt(r.particles.reduce((s, x) => s + x.length, 0))}</dd>
      <dt>tracks</dt><dd>${fmt(r.tracks.length)}</dd>
      <dt>ensemble D (brownian fit)</dt><dd>${f.brownian.success ? fmt(f.brownian.diffusion_coefficient, 4) + ' µm²/s' : '–'}</dd>
      <dt>ensemble α</dt><dd>${f.anomalous.success ? fmt(f.anomalous.alpha) : '–'}</dd></dl>
      ${bars(Object.entries(r.class_counts).map(([c, n]) => ({ label: c.replace('_', ' '), value: n, color: color(c) })))}
      <button type="button" id="csv" class="alt">Download tracks (CSV)</button></div></div>
    <div class="scroll"><table class="tbl"><thead><tr><th scope="col">track</th><th scope="col">points</th><th scope="col">class</th>
      <th scope="col">α</th><th scope="col">D (µm²/s)</th><th scope="col">level L</th></tr></thead><tbody>
      ${r.per_track.slice(0, 300).map(p => `<tr><td>${p.track_id}</td><td>${p.length}</td><td>${cls(p.motion_class)}</td><td>${fmt(p.alpha)}</td>
        <td>${fmt(p.D, 4)}</td><td>${fmt(p.confinement_level, 2)}</td></tr>`).join('')}</tbody></table></div>`;
  const v = new Viewer($('#pv'), [H, W], Math.max(1, Math.round(560 / Math.max(H, W))));
  const cls_of = new Map(r.per_track.map(p => [p.track_id, p.motion_class]));
  v.overlay({ tracks: r.tracks, colorOf: t => color(cls_of.get(t.track_id)), label: `${r.tracks.length} tracks` });
  $('#csv').addEventListener('click', () => {
    const rows = ['# x, y in pixels (pixel centres); x_um, y_um = pixels x ' + u.pixel_size_um + ' um; time_s = frame x ' + u.time_interval_s + ' s',
      'track_id,frame,x,y,x_um,y_um,time_s,motion_class'];
    for (const t of r.tracks) t.frames.forEach((fr, i) => rows.push([t.track_id, fr, t.x[i], t.y[i], t.x[i] * u.pixel_size_um,
      t.y[i] * u.pixel_size_um, fr * u.time_interval_s, cls_of.get(t.track_id)].join(',')));
    download(r.file.replace(/\.[^.]+$/, '') + '_tracks.csv', rows.join('\n') + '\n', 'text/csv');
  });
}
