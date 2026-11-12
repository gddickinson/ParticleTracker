# ParticleTracker -- Interface Map

## Project Layout

```
ParticleTracker/
  run_gui.py              -- Entry point to launch GUI (src/ on sys.path: packages import as core, detection, ...)
  run_web.py              -- Starts the web app (uvicorn, 127.0.0.1:8211)
  requirements-web.txt    -- Pinned web/test dependencies
  setup.py                -- Legacy setuptools config
  pyproject.toml          -- Modern packaging config
  requirements.txt        -- Pinned dependency versions
  src/
    __init__.py            -- Package root, re-exports core classes
    core/
      data_structures.py   -- ImageStack, Detection, Track, TrackingResult (from_dataframe),
                              create_synthetic_data (thin wrapper -> simulation.py)
      simulation.py        -- create_synthetic_data (seeded, sub-pixel rendering, motion_types,
                              return_ground_truth), simulate_trajectories, render_frame
      pipeline.py          -- ParticleTracker (main high-level API); _sibling() imports that work
                              both as src.core and as top-level core
    detection/
      detector_factory.py  -- BaseDetector (ABC), create_detector() factory
      wavelet_detector.py  -- WaveletDetector (Mexican hat / Morlet wavelet detection)
      log_detector.py      -- LoGDetector, DoGDetector, GaussianFitDetector
    tracking/
      tracker_factory.py   -- BaseTracker (ABC), create_tracker() factory
      lap_tracker.py       -- LAPTracker (frame-to-frame LAP + gap closing), solve_augmented_lap
      nearest_tracker.py   -- NearestNeighborTracker (simple baseline)
    analysis/
      msd.py               -- frame_msd (pairs exactly k frames apart; gaps allowed)
      confinement.py       -- Saxton/Simson confinement_level, initial_slope_D
      motion.py            -- calculate_msd (pixel_size/time_interval -> units), fit_diffusion_model
                              (adds units), calculate_velocity, calculate_directionality,
                              detect_confinement (method 'saxton' | 'rg_legacy'), classify_motion_type
      classification.py    -- classify_track_motion, batch_classify_tracks,
                              get_classification_summary, _extract_track_features
    utils/
      io.py                -- load_image_stack, save_image_stack, export_results
                              (CSV, HDF5, MAT, TrackMate XML)
      visualization.py     -- plot_tracks, plot_msd, plot_track_statistics,
                              plot_trajectory_classification, create_movie
    gui/
      main.py              -- ParticleTrackerGUI (PyQt5), TrackingWorker thread
    web/                   -- web interface (FastAPI + plain ES modules, no build step)
      app.py               -- routes under /api, STATIC_ENDPOINTS (exported by scripts/build_static.py)
      demo.py              -- demo movie (seed 220, static/demo/movie.u16), info/meta, detections(), run(), pipeline()
      analysis.py          -- runs ParticleTracker steps; per_track(), ensemble(), analyse() -> compact JSON
      scoring.py           -- detection_scores, tracking_scores against the simulated ground truth
      upload.py            -- run_upload: pipeline on an uploaded TIFF/.npy (local app only)
      jsonify.py           -- numpy -> JSON
      static/              -- index.html, core.js (kit), main.js, data.js, static_api.js, player.js, viewer.js,
                              charts.js, pages (home, detect, track, analyze, run, docs) and JS ports:
                              detect_engine.js (wavelet, subpixel off), link_engine.js (LAP + Hungarian),
                              analysis_engine.js (MSD, fits, confinement, classifier), score_engine.js;
                              engines.test.mjs (node, vs tests/fixtures/engine_cases.json)
  scripts/
    build_static.py        -- static site export (site/)
    make_demo.py           -- writes src/web/static/demo/movie.u16 from the simulator
    make_engine_fixture.py -- writes tests/fixtures/engine_cases.json (Python results for the JS ports)
    screenshots.py         -- Playwright smoke test + README screenshots (docs/img)
  tests/
    test_data_structures.py
    test_wavelet_detector.py
    test_lap_tracker.py
    test_motion.py
    test_detector_factory.py, test_lap_linking.py, test_msd_frames.py, test_confinement_level.py,
    test_units.py, test_gui_launch_path.py, test_simulation.py, test_track_features.py,
    test_pipeline_defaults.py  -- regression tests for the 2026-11-12 fixes
    test_web.py            -- web API, static build, fixture current, node engine tests
```

## Module Dependencies

- `core.pipeline` depends on: `core.data_structures`, `detection`, `tracking`, `analysis`, `utils`
- `detection.*` depends on: `core.data_structures`
- `tracking.*` depends on: `core.data_structures`
- `analysis.*` depends on: `core.data_structures`
- `utils.*` depends on: `core.data_structures`
- `gui.main` depends on: `core.pipeline`, `core.data_structures`
- `web.*` depends on: `core.pipeline`, `core.simulation`, `analysis.*`, `utils.io`; the JS ports in `web/static`
  mirror `detection.wavelet_detector` (subpixel off), `tracking.lap_tracker`, `analysis.*` and `web.scoring`

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `ParticleTracker` | `core/pipeline.py` | Main API: load, detect, track, analyze, export |
| `ImageStack` | `core/data_structures.py` | Image data container (T,Y,X) or (T,Z,Y,X) |
| `Detection` | `core/data_structures.py` | Single particle detection with position and properties |
| `Track` | `core/data_structures.py` | Linked trajectory of detections |
| `TrackingResult` | `core/data_structures.py` | Collection of tracks + metadata |
| `BaseDetector` | `detection/detector_factory.py` | ABC for all detectors |
| `BaseTracker` | `tracking/tracker_factory.py` | ABC for all trackers |
| `WaveletDetector` | `detection/wavelet_detector.py` | Wavelet-based particle detection |
| `LAPTracker` | `tracking/lap_tracker.py` | LAP-based tracking (U-track style) |
