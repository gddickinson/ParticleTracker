// Entry point: lazy page modules, the mode label, a subtle diffusing-spots background, then the router.
import { $, isStatic, lazy, route, start, app } from './core.js';

const PAGES = { 'home.js': ['home'], 'detect.js': ['detect'], 'track.js': ['track'], 'analyze.js': ['analyse'],
  'run.js': ['run'], 'docs.js': ['methods', 'about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: detection (pixel level), linking and analysis run in your browser; sub-pixel fits and the other detectors were run by the Python code at build time'
  : 'local app: everything runs in Python on this computer';

// Background: faint Gaussian spots diffusing, like a fluorescence movie. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const ps = Array.from({ length: 20 }, () => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 60) return;
    last = t;
    g.fillStyle = 'rgba(5,6,12,.22)'; g.fillRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      p.x = (p.x + (Math.random() - 0.5) * 2.4 + canvas.width) % canvas.width;
      p.y = (p.y + (Math.random() - 0.5) * 2.4 + canvas.height) % canvas.height;
      const gr = g.createRadialGradient(p.x, p.y, 0, p.x, p.y, 9);
      gr.addColorStop(0, 'rgba(92,225,255,.2)'); gr.addColorStop(1, 'rgba(92,225,255,0)');
      g.fillStyle = gr; g.fillRect(p.x - 9, p.y - 9, 18, 18);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
