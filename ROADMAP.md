# ParticleTracker -- Roadmap

## Current State
A well-structured fluorescence microscopy particle detection and tracking framework with modular architecture under `src/` (core, detection, tracking, analysis, gui, utils). Core components exist: wavelet and LoG detectors, LAP and nearest-neighbor trackers, MSD motion analysis, track classification, and a PyQt GUI. The codebase uses good patterns (factory classes, dataclasses, type hints) but has no tests, bare `except` clauses in `pipeline.py` for imports, and several README-advertised features (deep learning, 3D tracking, STORM/PALM) are not yet implemented.

## Short-term Improvements
- [x] Replace bare `except` with `except ImportError` in `src/core/pipeline.py` (lines 13-22)
- [x] Add a `tests/` directory with unit tests for `data_structures.py`, `wavelet_detector.py`, `lap_tracker.py`, and `motion.py`
- [x] Add `requirements.txt` pinned versions (currently unpinned)
- [ ] Add input validation in `ParticleTracker.load_data()` for unsupported file formats
- [x] Add docstrings to `detector_factory.py` and `tracker_factory.py` factory methods
- [x] Create an `INTERFACE.md` navigation map per project conventions
- [x] Add `__all__` exports to each `__init__.py`

## Feature Enhancements
- [ ] Implement DoG (Difference of Gaussians) detector listed in README but missing from `src/detection/`
- [ ] Add batch processing mode to `pipeline.py` for multi-file analysis
- [ ] Implement drift correction for super-resolution workflows
- [ ] Add CSV/HDF5 export options in `src/utils/io.py` (currently basic)
- [ ] Add track filtering by length, displacement, and quality score in the GUI
- [ ] Implement 3D tracking support in `lap_tracker.py` (currently 2D only)
- [ ] Add progress callbacks/bars for long-running detection and tracking

## Long-term Vision
- [ ] Implement CNN-based detection model under `src/models/` (currently empty)
- [ ] Add LSTM-based tracking for dense particle fields
- [ ] Build STORM/PALM reconstruction pipeline with blinking statistics
- [ ] GPU acceleration via CuPy or PyTorch for wavelet transforms and LAP solving
- [ ] Plugin architecture for custom detection/tracking algorithms
- [ ] REST API for headless batch processing on HPC clusters
- [ ] Integration with napari for modern visualization

## Technical Debt
- [ ] `src/gui/main.py` likely exceeds 500 lines -- split into separate widget modules
- [x] Eliminate duplicate import fallback pattern (bare except) across all modules
- [ ] Add type stubs or mypy configuration for static analysis
- [ ] Add CI/CD with GitHub Actions (lint, test, build)
- [x] Add `pyproject.toml` alongside `setup.py` for modern packaging
- [ ] Profile `lap_tracker.py` cost matrix construction for large particle counts
