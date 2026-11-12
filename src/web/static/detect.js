// Detect: the demo movie with the detections of the chosen detector, its parameters and the scores against the
// simulated ground truth; a comparison of the four detectors at their default parameters.
import { $, api, app, esc, go, isStatic, route, slider } from './core.js';
import { fmt, getInfo, getMeta, getMovie, getRun, pct, runSettings, where } from './data.js';
import { moviePlayer } from './player.js';

const WAVELET = [['sigma', 'Wavelet scale sigma (px)', 0.8, 3, 0.1], ['threshold', 'Threshold (noise standard deviations)', 1, 8, 0.25],
  ['min_distance', 'Minimum peak spacing (px)', 1, 6, 1]];

route('detect', async () => {
  const [info, meta, movie] = await Promise.all([getInfo(), getMeta(), getMovie()]);
  const def = { ...info.defaults.detection }; delete def.method;
  const cur = runSettings();
  const st = { detector: cur?.detector || 'wavelet', dp: { ...def, subpixel: true, ...(cur?.det_params || {}) }, res: null };
  app().innerHTML = `
  <h1>Detect</h1>
  <p class="lead">The demo movie: ${meta.n_frames} frames of ${meta.frame_shape.join(' × ')} px simulated by the repo's own
    <code>create_synthetic_data</code> (seed ${meta.seed}) with ${meta.n_particles} particles whose true positions are known.
    Circles are detections, red crosses the true positions.</p>
  <div class="two">
    <div id="pl"></div>
    <div>
      <div class="panel params"><fieldset><legend>Detector</legend>
        <div class="seg" role="group" aria-label="Detector">${info.detectors.map(d => `<button type="button" data-det="${d}">${d}</button>`).join('')}</div>
        <div id="dp"></div></fieldset>
        <button type="button" id="use" class="alt">Track with these detections →</button></div>
      <div class="panel" id="sc" aria-live="polite"></div>
    </div>
  </div>
  <h2>The four detectors at their default parameters</h2><div id="cmp" class="scroll"></div>
  <p class="hint">A detection is a true positive when it lies within 2 px of a true position in the same frame (each
    true position used once, closest pairs first). RMSE is the localisation error of those pairs.</p>`;

  const player = moviePlayer($('#pl'), movie, {
    checks: [['det', 'detections', true], ['truth', 'true positions', true]],
    overlay: (t, show) => ({ frame: t, particles: st.res && show.det ? st.res.particles[t] : [],
      truth: show.truth ? meta.ground_truth.map(g => ({ x: g.x[t], y: g.y[t] })) : [],
      label: `Frame ${t}: ${st.res ? st.res.particles[t].length : 0} detections, ${meta.n_particles} true particles` }),
  });

  const panel = () => {
    app().querySelectorAll('[data-det]').forEach(b => b.classList.toggle('on', b.dataset.det === st.detector));
    const host = $('#dp'); host.innerHTML = '';
    if (st.detector !== 'wavelet') {
      host.innerHTML = `<p class="hint">${esc(st.detector)} with its default parameters${isStatic() ? ' (run by the Python code when the site was built; other parameters need the local app)' : ''}.</p>`;
      return;
    }
    for (const [k, label, min, max, step] of WAVELET)
      slider(host, { id: k, label, min, max, step, value: st.dp[k], fmt: v => fmt(v, step < 1 ? 2 : 0) }, v => {
        st.dp[k] = v;
        if (isStatic()) { st.dp.subpixel = false; $('#sp').checked = false; }
        schedule();
      });
    host.insertAdjacentHTML('beforeend', `<label class="checks"><input type="checkbox" id="sp" ${st.dp.subpixel ? 'checked' : ''}>
      sub-pixel 2D Gaussian refinement${isStatic() ? ' (Python: on this site only with the default parameters)' : ''}</label>
      <p class="hint">${isStatic() ? 'With refinement off, the detector runs in your browser (JavaScript port, positions at peak pixels).' : ''}</p>`);
    $('#sp').addEventListener('change', e => {
      st.dp.subpixel = e.target.checked;
      if (isStatic() && st.dp.subpixel) { Object.assign(st.dp, def); panel(); }
      schedule();
    });
  };

  let timer = 0, seq = 0;
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 250); };
  const settings = () => {
    const isDef = st.detector === 'wavelet' && st.dp.subpixel && Object.keys(def).every(k => st.dp[k] === def[k]);
    return isDef ? null : { detector: st.detector, det_params: st.detector === 'wavelet' ? { ...st.dp } : {},
      link_params: runSettings()?.link_params || {} };
  };
  async function run() {
    const my = ++seq;
    $('#sc').innerHTML = '<p class="mute">Detecting...</p>';
    try {
      const r = await getRun(settings());
      if (my !== seq || !$('#sc')) return;   // superseded, or the page was left
      st.res = r;
      const d = r.detection_scores;
      $('#sc').innerHTML = `<h3>Against the ground truth</h3><p>${where(r)}</p><dl class="kv">
        <dt>detections</dt><dd>${fmt(d.n_detected)} (true ${fmt(d.n_true)})</dd>
        <dt>true positives</dt><dd>${fmt(d.true_positives)}</dd><dt>false positives</dt><dd>${fmt(d.false_positives)}</dd>
        <dt>missed</dt><dd>${fmt(d.false_negatives)}</dd><dt>precision · recall</dt><dd>${pct(d.precision)} · ${pct(d.recall)}</dd>
        <dt>F1</dt><dd>${fmt(d.f1)}</dd><dt>RMSE</dt><dd>${fmt(d.rmse_px)} px</dd></dl>`;
      player.draw();
    } catch (e) { if (my === seq && $('#sc')) $('#sc').innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  app().querySelectorAll('[data-det]').forEach(b => b.addEventListener('click', () => { st.detector = b.dataset.det; panel(); run(); }));
  $('#use').addEventListener('click', () => go('#/track'));
  panel(); run();

  const rows = await Promise.all(info.detectors.map(d => api(`/demo/detections/${d}`)));
  if (!$('#cmp')) return;   // the page was left while loading
  $('#cmp').innerHTML = `<table class="tbl"><thead><tr><th scope="col">detector</th><th scope="col">detections</th><th scope="col">true pos.</th>
    <th scope="col">false pos.</th><th scope="col">missed</th><th scope="col">precision</th><th scope="col">recall</th><th scope="col">RMSE (px)</th></tr></thead><tbody>
    ${rows.map(r => { const d = r.detection_scores; return `<tr><td>${esc(r.detector)}</td><td>${fmt(d.n_detected)}</td><td>${fmt(d.true_positives)}</td>
      <td>${fmt(d.false_positives)}</td><td>${fmt(d.false_negatives)}</td><td>${pct(d.precision)}</td><td>${pct(d.recall)}</td><td>${fmt(d.rmse_px)}</td></tr>`; }).join('')}
    </tbody></table>`;
});
