// Track: LAP linking of the current detections (frame-to-frame assignment, then gap closing), the tracks over the
// movie, and how well they follow the simulated particles.
import { $, app, esc, route, slider } from './core.js';
import { cls, fmt, getInfo, getMeta, getMovie, getRun, pct, runSettings, trackColor, where } from './data.js';
import { moviePlayer } from './player.js';

const LINK = [['max_displacement', 'Max displacement (px / frame)', 1, 15, 0.5], ['gap_closing', 'Gap closing (frames)', 0, 6, 1],
  ['min_track_length', 'Minimum track length (points)', 2, 20, 1]];

route('track', async () => {
  const [info, meta, movie] = await Promise.all([getInfo(), getMeta(), getMovie()]);
  const cur = runSettings();
  const st = { link: { ...info.defaults.tracking, ...(cur?.link_params || {}) }, res: null };
  app().innerHTML = `
  <h1>Track</h1>
  <p class="lead">The LAP tracker (<code>src/tracking/lap_tracker.py</code>, after U-track) links the detections from the
    Detect page${cur ? ` (${esc(cur.detector)}, your parameters)` : ' (wavelet detector, default parameters)'}: an optimal
    assignment between consecutive frames, then a second assignment that closes gaps of a few frames. Tracks shorter than the minimum length are dropped.</p>
  <div class="two">
    <div id="pl"></div>
    <div>
      <div class="panel params"><fieldset><legend>Linking</legend><div id="lp"></div>
        <p class="hint">A link costs its distance; ending a track or starting one costs max displacement each, so links longer than
          2 × max displacement are never made. Gaps cost distance + 0.5 per missing frame (up to 10 px × √gap).</p></fieldset></div>
      <div class="panel" id="sc" aria-live="polite"></div>
    </div>
  </div>
  <h2>Simulated particles</h2><div id="pp" class="scroll"></div>`;

  const player = moviePlayer($('#pl'), movie, {
    checks: [['tracks', 'tracks', true], ['det', 'detections', false], ['truth', 'true positions', false], ['full', 'whole tracks', true]],
    overlay: (t, show) => ({ frame: show.full ? null : t, tail: 15, particles: st.res && show.det ? st.res.particles[t] : [],
      tracks: st.res && show.tracks ? st.res.tracks : [], colorOf: tr => trackColor(tr.track_id),
      truth: show.truth ? meta.ground_truth.map(g => ({ x: g.x[t], y: g.y[t] })) : [],
      label: `Frame ${t}: ${st.res ? st.res.tracks.length : 0} tracks` }),
  });
  for (const [k, label, min, max, step] of LINK)
    slider($('#lp'), { id: k, label, min, max, step, value: st.link[k], fmt: v => fmt(v, step < 1 ? 1 : 0) }, v => { st.link[k] = v; schedule(); });

  let timer = 0, seq = 0;
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 200); };
  async function run() {
    const my = ++seq;
    $('#sc').innerHTML = '<p class="mute">Linking...</p>';
    const lp = Object.fromEntries(LINK.map(([k]) => [k, st.link[k]]));
    const changed = LINK.some(([k]) => st.link[k] !== info.defaults.tracking[k]);
    const base = runSettings();
    const settings = !base && !changed ? null
      : { detector: base?.detector || 'wavelet', det_params: base?.det_params || {}, link_params: lp };
    try {
      const r = await getRun(settings);
      if (my !== seq || !$('#sc')) return;   // superseded, or the page was left
      st.res = r;
      const s = r.tracking_scores;
      $('#sc').innerHTML = `<h3>Against the ground truth</h3><p>${where(r)}</p><dl class="kv">
        <dt>tracks</dt><dd>${fmt(s.n_tracks)} (true particles ${fmt(s.n_true)})</dd>
        <dt>particles recovered</dt><dd>${fmt(s.recovered)} / ${fmt(s.n_true)} (one track follows ≥ ${pct(s.recovered_threshold)} of its frames)</dd>
        <dt>mean coverage</dt><dd>${pct(s.mean_coverage)}</dd>
        <dt>tracks on no particle</dt><dd>${fmt(s.unmatched_tracks)}</dd></dl>
        <p class="hint">A track point follows a particle when it is within ${s.tolerance_px} px of its true position.</p>`;
      $('#pp').innerHTML = `<table class="tbl"><thead><tr><th scope="col">particle</th><th scope="col">simulated motion</th>
        <th scope="col">coverage by its best track</th><th scope="col">tracks on it</th></tr></thead><tbody>
        ${s.per_particle.map(p => `<tr><td>#${p.particle}</td><td>${cls(meta.ground_truth[p.particle].motion_type)}</td>
          <td>${pct(p.coverage)}</td><td>${p.tracks}</td></tr>`).join('')}</tbody></table>`;
      player.draw();
    } catch (e) { if (my === seq && $('#sc')) $('#sc').innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
  run();
});
