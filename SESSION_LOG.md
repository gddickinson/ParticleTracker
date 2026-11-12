# Session log

## 2026-11-12: web interface, static site and bug fixes (Claude, branch `web-interface`)

### Bugs fixed (each with a regression test that failed first; separate "Fix ..." commits)
1. Detectors could not be created: unused `import pywt` broke the default detector without PyWavelets;
   `create_detector('dog'/'gaussian')` imported modules that do not exist; 'deep' now raises NotImplementedError.
2. Wavelet detector kernel used the 1D Mexican hat formula (negative sum in 2D): bright spots gave a negative
   response and the default detector reported noise. Now the 2D form, zero-sum, symmetric grid.
3. LAP tracker never linked anything with 2+ particles (birth/death blocks had 0 off-diagonal costs), so the
   pipeline's min_track_length filter removed every track. Gap-closing chains could also duplicate tracks.
4. MSD: pairs searched by row lag (valid pairs across gaps lost); short tracks contributed MSD = 0 to the
   ensemble mean at long lags. Now `frame_msd` + NaN-aware mean.
5. Confinement: fixed 2 px radius-of-gyration rule called slow free diffusion confined (100 % at 0.25 px²/frame)
   and fast corralled motion free. Now the Saxton/Simson level (as particle_tracking / particle-tracker); old rule
   kept as `method='rg_legacy'`. Changes classification results.
6. Units never reached the analysis: MSD/D in px² labelled µm² in the GUI; classify_motion without frame interval;
   TIFF loader ignored the file's pixel size/frame interval (now read, source recorded in `scale_source`);
   `load_project` called a missing method.
7. GUI runs failed ('relative import beyond top-level package') when launched with run_gui.py; Tracks tab drew nothing.
8. `_extract_track_features` (method 'ml') crashed on NumPy 2 (np.cross of 2D vectors).
9. GaussianFitDetector crashed (negative positions) or returned positions outside the image on noise peaks.
10. `detect(method='log'/'dog')` found nothing: the wavelet defaults (threshold 3.0) leaked into every method.
11. examples/example_workflow.py imported a non-existent `particletracker` package; velocity printed px/s as µm/s.
12. pyproject.toml build backend did not exist (`pip install -e .` failed); TECHNICAL.md/README claimed
    unimplemented features (deep learning, STORM/PALM, MHT, 3D) — corrected.
Also: simulator renders at sub-pixel positions (was truncated to whole pixels), seeded, returns ground truth,
per-particle motion types (moved to src/core/simulation.py).

### Web interface (src/web, port 8211)
Pages Home, Detect, Track, Analyse, Your data (local only), Methods, About. Demo movie simulated by the repo's own
simulator (seed 220, 50×112×112, 1.25 MB uint16). JS ports (wavelet detection with subpixel off, LAP linking with a
Hungarian solver, MSD analysis/classifier, scoring) match Python in `engines.test.mjs` (7873 checks). Precomputed:
sub-pixel wavelet detections, LoG/DoG/Gaussian detections, confined-model fit. The 4 IC235 TIFFs in data/ (61 MB)
are not used by the site.

### Open (needs George's decision)
- MSD classifier calls simulated directed motion (with diffusion) 'superdiffusive' (α > 1.15 checked before
  directionality) and two of four confined particles 'subdiffusive' (confinement level 2.8 < 3.16).
- `pipeline.py` calls `logging.basicConfig(level=INFO)` at import, which configures logging for any program that imports it.
- NearestNeighborTracker ignores its documented `gap_closing` parameter.

## 2026-11-12 (follow-up): George's decisions on the open items (Claude, branch `web-interface`)
1. **MSD classifier fixed** (`src/analysis/msd_classifier.py`; old rule kept as `method='msd_legacy'`). Why it failed,
   measured on thousands of simulated tracks (`scripts/msd_classifier_study.py` -> `docs/msd_classifier.md`): alpha was
   fitted over lags up to n/2 (max 20), so ~20 % of free-diffusion tracks fell outside 0.85-1.15; alpha > 1.15 was
   'superdiffusive' before directed motion was tested; the confinement level used lags 1-4, where a small corral is
   already saturating. Now: confinement level (lags 1-2) > 2, drift test (chi-square, p < 0.01), alpha over lags
   1..clip(n/4, 4, 10) with 0.6 / 1.4 -> confined / directed (or sub/superdiffusive with `alpha_classes='anomalous'`).
   All are options (`config['classification']`, `classify_motion(**options)`). Demo-parameter tracks: recall
   brownian 58 -> 96 %, directed 0 -> 79 %, confined 12 -> 92 %; end to end on 11 demo-like movies 41/0/16 % ->
   88/65/84 %. `classify_motion_type` now uses the same classifier. JS port + fixture (72 tracks x 3 option sets).
   Changes classification results (approved: it fixes the problem).
