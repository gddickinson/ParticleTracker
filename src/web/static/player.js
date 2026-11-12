// Movie player: the demo movie in a Viewer with a frame slider, Play/Pause and overlay check boxes.
// overlay(frame, show) returns the Viewer.overlay options for that frame.
import { animate, esc } from './core.js';
import { Viewer } from './viewer.js';

export function moviePlayer(host, movie, { checks = [], overlay = () => ({}), scale = 5 } = {}) {
  const [T, H, W] = movie.shape;
  const show = Object.fromEntries(checks.map(([k, , on]) => [k, !!on]));
  host.innerHTML = `<div class="vw"></div>
    <div class="player"><button type="button" class="play">Play</button>
      <label class="mono">frame <b class="frn">0</b> / ${T - 1}<input class="fr" type="range" min="0" max="${T - 1}" value="0" aria-label="Frame"></label></div>
    <div class="checks">${checks.map(([k, label]) => `<label><input type="checkbox" data-k="${k}" ${show[k] ? 'checked' : ''}> ${esc(label)}</label>`).join('')}</div>`;
  const view = new Viewer(host.querySelector('.vw'), [H, W], scale);
  const st = { frame: 0, playing: false };
  const draw = () => {
    const t = st.frame;
    view.drawFrame(movie.data.subarray(t * H * W, (t + 1) * H * W), movie.lo, movie.hi);
    view.overlay(overlay(t, show));
    host.querySelector('.frn').textContent = t;
    host.querySelector('.fr').value = t;
  };
  host.querySelector('.fr').addEventListener('input', e => { st.frame = +e.target.value; draw(); });
  const btn = host.querySelector('.play');
  btn.addEventListener('click', () => { st.playing = !st.playing; btn.textContent = st.playing ? 'Pause' : 'Play'; });
  host.querySelectorAll('[data-k]').forEach(c => c.addEventListener('change', () => { show[c.dataset.k] = c.checked; draw(); }));
  let last = 0;
  animate(t => { if (st.playing && t - last > 120) { last = t; st.frame = (st.frame + 1) % T; draw(); } });
  draw();
  return { draw, show };
}
