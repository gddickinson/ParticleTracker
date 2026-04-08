# ParticleTracker -- Interface Map

## Project Layout

```
ParticleTracker/
  run_gui.py              -- Entry point to launch GUI
  setup.py                -- Legacy setuptools config
  pyproject.toml          -- Modern packaging config
  requirements.txt        -- Pinned dependency versions
  src/
    __init__.py            -- Package root, re-exports core classes
    core/
      data_structures.py   -- ImageStack, Detection, Track, TrackingResult, create_synthetic_data
      pipeline.py          -- ParticleTracker (main high-level API)
    detection/
      detector_factory.py  -- BaseDetector (ABC), create_detector() factory
      wavelet_detector.py  -- WaveletDetector (Mexican hat / Morlet wavelet detection)
      log_detector.py      -- LoGDetector, DoGDetector, GaussianFitDetector
    tracking/
      tracker_factory.py   -- BaseTracker (ABC), create_tracker() factory
      lap_tracker.py       -- LAPTracker (Hungarian algorithm, gap closing)
      nearest_tracker.py   -- NearestNeighborTracker (simple baseline)
    analysis/
      motion.py            -- calculate_msd, fit_diffusion_model, calculate_velocity,
                              calculate_directionality, detect_confinement, classify_motion_type
      classification.py    -- classify_track_motion, batch_classify_tracks,
                              get_classification_summary, _extract_track_features
    utils/
      io.py                -- load_image_stack, save_image_stack, export_results
                              (CSV, HDF5, MAT, TrackMate XML)
      visualization.py     -- plot_tracks, plot_msd, plot_track_statistics,
                              plot_trajectory_classification, create_movie
    gui/
      main.py              -- ParticleTrackerGUI (PyQt5), TrackingWorker thread
  tests/
    test_data_structures.py
    test_wavelet_detector.py
    test_lap_tracker.py
    test_motion.py
```

## Module Dependencies

- `core.pipeline` depends on: `core.data_structures`, `detection`, `tracking`, `analysis`, `utils`
- `detection.*` depends on: `core.data_structures`
- `tracking.*` depends on: `core.data_structures`
- `analysis.*` depends on: `core.data_structures`
- `utils.*` depends on: `core.data_structures`
- `gui.main` depends on: `core.pipeline`, `core.data_structures`, `utils.visualization`

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
