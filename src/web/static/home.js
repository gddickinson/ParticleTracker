// Home: hero, real counts from the demo and the pipeline, and cards to the pages.
import { app, isStatic, route } from './core.js';
import { fmt, getInfo, getRun, legend, pct } from './data.js';

route('home', async () => {
  const [info, run] = await Promise.all([getInfo(), getRun(null)]);
  const d = info.demo, ds = run.detection_scores, ts = run.tracking_scores;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Fluorescence microscopy · single-particle tracking</p>
    <h1 class="glow">ParticleTracker</h1>
    <p class="lead">A Python framework for particle <b>detection</b>, sub-pixel <b>localisation</b>, <b>tracking</b> and motion
      <b>analysis</b>. Try the pipeline on a movie simulated by the repo's own generator, where the true path and motion
      type of every particle are known, so every step can be checked.</p>
    <a class="btn" href="#/detect">Detect</a><a class="btn alt" href="#/track">Track</a><a class="btn alt" href="#/analyse">Analyse</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${d.n_frames}</b><span>frames of ${d.frame_shape.join(' × ')} px in the simulated demo</span></div>
    <div class="tile"><b>${d.n_particles}</b><span>particles with known paths</span></div>
    <div class="tile"><b>${pct(ds.recall)}</b><span>of true positions detected (default wavelet detector, RMSE ${fmt(ds.rmse_px, 2)} px)</span></div>
    <div class="tile"><b>${ts.recovered}/${ts.n_true}</b><span>particles followed by one track for ≥ 80 % of the movie (LAP tracker)</span></div>
    <div class="tile"><b>${info.detectors.length}</b><span>detectors: ${info.detectors.join(', ')}</span></div>
  </div>
  <p class="mute">Simulated motion in the demo (4 particles each):</p>${legend(['brownian', 'directed', 'confined'])}
  ${isStatic() ? `<div class="callout">This is the static showcase. Pixel-level wavelet detection, LAP linking, the MSD analysis
    and classification and the scores against the ground truth run in your browser (JavaScript ports tested against the
    Python code). Sub-pixel Gaussian refinement, the other detectors and the confined-diffusion fit were run by the Python
    code when the site was built. Your own files need the local app (see About).</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/detect"><b>Detect</b><span>Play the demo with detections and true positions; tune the wavelet detector and compare the four detectors (precision, recall, localisation error).</span></a>
    <a class="card" href="#/track"><b>Track</b><span>Link detections with the LAP tracker (frame-to-frame assignment and gap closing) and see how completely each simulated particle is followed.</span></a>
    <a class="card" href="#/analyse"><b>Analyse</b><span>Ensemble MSD and four diffusion models, the MSD classifier against the simulated motion, and the confinement test.</span></a>
    <a class="card" href="#/run"><b>Your data</b><span>Run the pipeline on your own TIFF stack, with the pixel size and frame interval read from the file.</span></a>
    <a class="card" href="#/methods"><b>Methods</b><span>What each step computes in this code, with formulas and defaults.</span></a>
    <a class="card" href="#/about"><b>About</b><span>Desktop GUI, Python API, web app, what is implemented and credits.</span></a>
  </div>`;
});
