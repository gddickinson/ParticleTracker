// Methods (what each step computes in this code) and About.
import { app, isStatic, route } from './core.js';
import { getInfo } from './data.js';

route('methods', async () => {
  const d = (await getInfo()).defaults;
  app().innerHTML = `
  <h1>Methods</h1>
  <p class="lead">What each step computes in this repository's code, with its defaults (<code>ParticleTracker._default_config</code>).
    Positions are in pixels with the pixel-centre convention (x = column, y = row; a spot at x = 10 is centred on column 10).</p>
  <h2>Detection</h2><div class="grid wide">
    <div class="card"><b>wavelet</b> (default) <span class="badge js">pixel-level in your browser</span><span>The frame is convolved with a
      2D Mexican hat (negative Laplacian of a Gaussian, scale sigma = ${d.detection.sigma} px, zero-sum). Noise = 1.4826 × the median
      absolute deviation of the filtered image; local maxima at least ${d.detection.min_distance} px apart and from the border above
      ${d.detection.threshold} × noise are kept (scikit-image <code>peak_local_max</code>). Each is refined by fitting a 2D Gaussian
      in a 7 × 7 window (scipy <code>curve_fit</code>); if the fit moves more than 3 px the peak pixel is kept.</span></div>
    <div class="card"><b>log</b> / <b>dog</b><span>scikit-image <code>blob_log</code> / <code>blob_dog</code> (sigma 1–3 px, threshold 0.1)
      on the frame rescaled to 0–1. Positions are whole pixels; intensity is the pixel value.</span></div>
    <div class="card"><b>gaussian</b><span>Local maxima of the raw frame (at least 5 px apart, above 3 × noise), each fitted with a
      2D Gaussian in a 9 × 9 window. Fits whose centre leaves the window are discarded. On noisy frames it keeps many noise
      peaks (see the comparison on Detect).</span></div>
    <div class="card"><b>deep</b><span>Listed in the factory but not implemented (<code>src/models/</code> is empty); it raises NotImplementedError.</span></div>
  </div>
  <h2>Linking</h2><div class="grid wide">
    <div class="card"><b>LAP tracker</b> <span class="badge js">also in your browser</span><span>After U-track (Jaqaman et al. 2008).
      Frame to frame: an optimal assignment (Hungarian algorithm) where linking costs the distance (allowed below
      2 × max displacement × √Δt for Brownian motion), and ending or starting a track costs max displacement each
      (default ${d.tracking.max_displacement} px). Then gap closing: segment ends are joined to segment starts up to
      ${d.tracking.gap_closing} frames later, cost distance + 0.5 × gap (allowed below 10 px × √gap). Tracks shorter
      than ${d.tracking.min_track_length} points are dropped.</span></div>
    <div class="card"><b>nearest</b><span>Greedy nearest-neighbour baseline: each track takes its nearest detection in the next frame
      within max displacement if that detection is still free (otherwise the track ends); no gap closing. Not used in the web app.</span></div>
  </div>
  <h2>Analysis</h2><div class="grid wide">
    <div class="card"><b>MSD</b><span>For every track, the mean of squared displacements over all pairs of points exactly k frames
      apart (k = 1 … ${d.analysis.msd_lag_max}); the ensemble MSD averages the tracks that have that lag. Positions × pixel size
      give µm², lags × frame interval give s.</span></div>
    <div class="card"><b>Diffusion models</b><span>brownian MSD = 4Dt (straight line with intercept) · anomalous MSD = 4Dt<sup>α</sup>
      (line in log-log) · directed MSD = 4Dt + (vt)² · confined MSD = A(1 − e<sup>−4Dt/A</sup>) (scipy curve_fit; Python only).</span></div>
    <div class="card"><b>Confinement level</b><span>L = 2.5117·D·t/R² − 0.2048 (Saxton 1993, Biophys. J. 64:1766; Simson, Sheets &amp;
      Jacobson 1995, Biophys. J. 69:989): D from the slope of the MSD over lags 1–4 frames, t the duration, R the largest
      distance from the start. Confined when L &gt; 3.16. Dimensionless, so independent of speed and units.</span></div>
    <div class="card"><b>MSD classifier</b><span>Tracks of at least 10 points: confined if L &gt; 3.16; else subdiffusive if α &lt; 0.85,
      superdiffusive if α &gt; 1.15 (α from the track's MSD up to half its length, max 20 lags); else directed if net/total path
      length &gt; 0.7 or the directed fit's R² beats the Brownian fit's by 0.1; else brownian. The 'ml' and 'hmm' methods
      are placeholders that return the same classes.</span></div>
  </div>
  <h2>The demo movie</h2>
  <p><code>src/core/simulation.py</code>: particles move in pixels per frame (Brownian steps N(0, 2D) per axis; directed adds a
    constant drift; confined rejects steps leaving a disc around the start), are drawn as Gaussian spots (PSF sigma) at
    their sub-pixel positions, with background and Gaussian noise. The demo parameters are listed on About.</p>`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">ParticleTracker is a Python framework for detecting, localising and tracking particles in fluorescence microscopy
    movies and analysing their motion: a Python API (<code>ParticleTracker</code>), a PyQt5 desktop GUI, and this web app.</p>
  <div class="grid wide">
    <div class="card"><b>Desktop GUI</b><span><code>pip install -r requirements.txt</code>, then <code>python run_gui.py</code>: load a TIFF
      stack or generate synthetic data, choose detector and tracker parameters, run, view tracks and the MSD, export CSV/HDF5.</span></div>
    <div class="card"><b>Python API</b><span><code>from src.core.pipeline import ParticleTracker</code> · <code>load_data</code> (pixel size and
      frame interval from the TIFF's ImageJ/OME metadata) · <code>detect</code> · <code>track</code> · <code>analyze_msd</code> ·
      <code>fit_diffusion</code> · <code>classify_motion</code> · <code>export</code>. See <code>examples/example_workflow.py</code>.</span></div>
    <div class="card"><b>Web app</b><span><code>pip install -r requirements-web.txt</code>, then <code>python run_web.py</code> and open
      http://127.0.0.1:8211. Everything runs in Python there, including your own files${isStatic() ? '; this static site runs the parts it can in your browser' : ''}.</span></div>
  </div>
  <h2>Implemented and not (yet)</h2>
  <p>Implemented: wavelet, LoG, DoG and Gaussian-fit detection with sub-pixel Gaussian refinement; LAP and nearest-neighbour
    tracking with gap closing; MSD, diffusion fits, velocity, directionality, confinement and MSD-based classification;
    CSV/HDF5/MAT/TrackMate-XML export; synthetic movies. Not implemented: deep-learning detection or tracking, STORM/PALM
    reconstruction, drift correction, 3D tracking, multiple hypothesis tracking (see ROADMAP.md).</p>
  <h2>Demo movie</h2>
  <p>Seed ${info.demo.seed}; ${info.demo.n_frames} frames of ${info.demo.frame_shape.join(' × ')} px; ${info.demo.n_particles} particles
    (4 brownian, 4 directed, 4 confined); D = 0.3 px²/frame, drift 0.4 px/frame, confinement radius 2 px, PSF sigma 1.5 px,
    intensity 1000, background 100, noise SD 10; ${info.demo.pixel_size} µm/px and ${info.demo.time_interval} s/frame assumed. The seed was
    chosen so that particles stay at least 6 px apart and away from the frame edge.</p>
  <h2>Credits</h2>
  <p>From the README: the project builds on ThunderSTORM (Ovesný et al., 2014), U-track (Jaqaman et al., 2008; Roudot et al., 2023),
    trackpy (Soft-Matter community) and deep-learning tracking methods (Spilger et al., 2021). Confinement criterion: Saxton (1993)
    and Simson, Sheets &amp; Jacobson (1995). MIT licence. Built with AI assistance from Claude (Anthropic).</p>
  <p><a href="https://github.com/gddickinson/ParticleTracker">Source on GitHub</a></p>`;
});
