"""
Main pipeline for particle detection and tracking.

This module provides the high-level API for the ParticleTracker system,
orchestrating detection, tracking, and analysis workflows.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging

try:
    from .data_structures import (
        ImageStack, Detection, Track, TrackingResult,
        create_synthetic_data
    )
except ImportError:
    from core.data_structures import (
        ImageStack, Detection, Track, TrackingResult,
        create_synthetic_data
    )

def _sibling(name: str):
    """
    Import a sibling subpackage module ('detection.detector_factory', ...)
    whether this package is imported as 'src.core' (tests, examples) or as
    top-level 'core' (run_gui.py and the 'particletracker' console script put
    src/ on sys.path). Imported lazily to avoid circular imports.
    """
    import importlib
    try:
        return importlib.import_module('..' + name, __package__)
    except (ImportError, TypeError):
        return importlib.import_module(name)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParticleTracker:
    """
    Main interface for particle detection and tracking.

    This class provides a high-level API for the complete workflow:
    1. Load microscopy data
    2. Detect particles in each frame
    3. Link detections into trajectories
    4. Analyze motion and behavior
    5. Export and visualize results

    Example:
        >>> tracker = ParticleTracker()
        >>> tracker.load_data('movie.tif')
        >>> detections = tracker.detect(method='wavelet')
        >>> tracks = tracker.track(method='lap')
        >>> tracker.analyze_msd(tracks)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize tracker.

        Args:
            config: Configuration dictionary with default parameters
        """
        self.config = config or self._default_config()
        self.image_stack: Optional[ImageStack] = None
        self.detections: List[Detection] = []
        self.tracking_result: Optional[TrackingResult] = None

        logger.info(f"ParticleTracker initialized")

    @staticmethod
    def _default_config() -> Dict:
        """Return default configuration."""
        return {
            'detection': {
                'method': 'wavelet',
                'sigma': 1.5,
                'threshold': 3.0,
                'min_distance': 3
            },
            'tracking': {
                'method': 'lap',
                'max_displacement': 5.0,
                'gap_closing': 3,
                'min_track_length': 5
            },
            'analysis': {
                'msd_lag_max': 50,
                'diffusion_model': 'brownian'
            },
            # MSD classifier options (analysis/msd_classifier.py)
            'classification': dict(_sibling('analysis.msd_classifier').MSD_CLASSIFIER_DEFAULTS)
        }

    def load_data(self, filename: Union[str, Path], **kwargs) -> ImageStack:
        """
        Load microscopy image data.

        Args:
            filename: Path to image file (TIFF, HDF5, etc.)
            **kwargs: Additional parameters (pixel_size, time_interval, etc.)

        Returns:
            ImageStack object
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        logger.info(f"Loading data from {filename}")

        # Import here to avoid circular dependencies
        load_image_stack = _sibling('utils.io').load_image_stack

        self.image_stack = load_image_stack(filename, **kwargs)
        logger.info(f"Loaded {self.image_stack.n_frames} frames, "
                   f"shape: {self.image_stack.shape}")

        return self.image_stack

    def create_synthetic(self, n_frames: int = 100,
                        n_particles: int = 10,
                        **kwargs) -> ImageStack:
        """
        Generate synthetic test data.

        Args:
            n_frames: Number of time frames
            n_particles: Number of particles
            **kwargs: Additional parameters

        Returns:
            ImageStack with synthetic data
        """
        logger.info(f"Generating synthetic data: {n_frames} frames, "
                   f"{n_particles} particles")

        self.image_stack = create_synthetic_data(
            n_frames=n_frames,
            n_particles=n_particles,
            **kwargs
        )

        return self.image_stack

    def detect(self, method: str = 'wavelet',
               frames: Optional[List[int]] = None,
               **params) -> List[Detection]:
        """
        Detect particles in loaded image stack.

        Args:
            method: Detection algorithm ('wavelet', 'log', 'dog', 'deep')
            frames: List of frame indices to process (None = all)
            **params: Method-specific parameters

        Returns:
            List of Detection objects
        """
        if self.image_stack is None:
            raise ValueError("No data loaded. Use load_data() or create_synthetic()")

        logger.info(f"Running {method} detection...")

        # Import detector
        create_detector = _sibling('detection.detector_factory').create_detector

        # Merge config with provided params. The configured parameters belong
        # to the configured method; another method uses its own defaults
        # (e.g. LoG/DoG 'threshold' is on a 0-1 scale, the wavelet's is in
        # noise standard deviations).
        configured = self.config['detection']
        if configured.get('method', 'wavelet') == method:
            config = configured.copy()
        else:
            config = {}
        config.update(params)
        config['method'] = method

        detector = create_detector(config)

        # Detect in all or subset of frames
        if frames is None:
            frames = range(self.image_stack.n_frames)

        all_detections = []

        for t in frames:
            frame = self.image_stack.get_frame(t)
            frame_detections = detector.detect(frame, frame_index=t)
            all_detections.extend(frame_detections)

            if (t + 1) % 10 == 0:
                logger.info(f"Processed {t + 1}/{len(frames)} frames, "
                           f"found {len(frame_detections)} detections")

        self.detections = all_detections
        logger.info(f"Total detections: {len(all_detections)}")

        return all_detections

    def track(self, method: str = 'lap',
              detections: Optional[List[Detection]] = None,
              **params) -> TrackingResult:
        """
        Link detections into trajectories.

        Args:
            method: Tracking algorithm ('lap', 'mht', 'deep', 'nearest')
            detections: List of detections (None = use self.detections)
            **params: Method-specific parameters

        Returns:
            TrackingResult object
        """
        if detections is None:
            if not self.detections:
                raise ValueError("No detections available. Run detect() first.")
            detections = self.detections

        logger.info(f"Running {method} tracking on {len(detections)} detections...")

        # Import tracker
        create_tracker = _sibling('tracking.tracker_factory').create_tracker

        # Merge config with provided params
        config = self.config['tracking'].copy()
        config.update(params)
        config['method'] = method

        tracker = create_tracker(config)

        # Run tracking
        self.tracking_result = tracker.track(detections)

        logger.info(f"Found {self.tracking_result.n_tracks} tracks")

        # Apply filtering if specified
        min_length = config.get('min_track_length', None)
        if min_length:
            before = self.tracking_result.n_tracks
            self.tracking_result = self.tracking_result.filter_tracks(
                min_length=min_length
            )
            logger.info(f"Filtered tracks: {before} -> {self.tracking_result.n_tracks}")

        return self.tracking_result

    def analyze_msd(self, tracks: Optional[Union[TrackingResult, List[Track]]] = None,
                    max_lag: Optional[int] = None) -> Dict:
        """
        Calculate mean squared displacement.

        Args:
            tracks: TrackingResult or list of tracks (None = use self.tracking_result)
            max_lag: Maximum lag time for MSD calculation

        Returns:
            Dictionary with MSD analysis results
        """
        if tracks is None:
            if self.tracking_result is None:
                raise ValueError("No tracks available. Run track() first.")
            tracks = self.tracking_result.tracks
        elif isinstance(tracks, TrackingResult):
            tracks = tracks.tracks

        logger.info(f"Calculating MSD for {len(tracks)} tracks...")

        calculate_msd = _sibling('analysis.motion').calculate_msd

        if max_lag is None:
            max_lag = self.config['analysis']['msd_lag_max']

        # Positions are in pixels: scale by the stack's pixel size (um) so
        # MSDs are in um^2 and D in um^2/s (results carry 'units').
        msd_results = calculate_msd(tracks, max_lag=max_lag,
                                    time_interval=self.image_stack.time_interval,
                                    pixel_size=self.image_stack.pixel_size)

        return msd_results

    def fit_diffusion(self, msd_results: Dict,
                      model: str = 'brownian') -> Dict:
        """
        Fit diffusion model to MSD data.

        Args:
            msd_results: Output from analyze_msd()
            model: Diffusion model ('brownian', 'directed', 'confined', 'anomalous')

        Returns:
            Dictionary with fitted parameters
        """
        logger.info(f"Fitting {model} diffusion model...")

        fit_diffusion_model = _sibling('analysis.motion').fit_diffusion_model

        fit_results = fit_diffusion_model(msd_results, model=model)

        return fit_results

    def classify_motion(self, tracks: Optional[Union[TrackingResult, List[Track]]] = None,
                       method: str = 'msd', **options) -> Dict:
        """
        Classify motion type for each track.

        Args:
            tracks: TrackingResult or list of tracks
            method: Classification method ('msd', 'msd_legacy', 'ml', 'hmm')
            **options: MSD classifier options, overriding config['classification']
                (see analysis/msd_classifier.py for the options and defaults)

        Returns:
            Dictionary mapping track_id to motion class
        """
        if tracks is None:
            if self.tracking_result is None:
                raise ValueError("No tracks available.")
            tracks = self.tracking_result.tracks
        elif isinstance(tracks, TrackingResult):
            tracks = tracks.tracks

        logger.info(f"Classifying motion for {len(tracks)} tracks using {method}")

        classify_track_motion = _sibling('analysis.classification').classify_track_motion

        opts = {} if method == 'msd_legacy' else {**self.config.get('classification', {}), **options}
        classifications = {}
        for track in tracks:
            motion_class = classify_track_motion(
                track, method=method,
                time_interval=self.image_stack.time_interval if self.image_stack else 1.0,
                **opts)
            classifications[track.track_id] = motion_class

        return classifications

    def visualize(self, output: Optional[str] = None,
                  show: bool = True,
                  **kwargs):
        """
        Visualize tracking results.

        Args:
            output: Output filename for saving (None = don't save)
            show: Whether to display plot
            **kwargs: Visualization options
        """
        logger.info("Generating visualization...")

        plot_tracks = _sibling('utils.visualization').plot_tracks

        plot_tracks(
            image_stack=self.image_stack,
            tracking_result=self.tracking_result,
            output=output,
            show=show,
            **kwargs
        )

    def export(self, filename: str, format: str = 'csv',
               include_analysis: bool = True):
        """
        Export results to file.

        Args:
            filename: Output filename
            format: File format ('csv', 'hdf5', 'mat', 'xml')
            include_analysis: Include analysis results
        """
        if self.tracking_result is None:
            raise ValueError("No tracking results to export.")

        logger.info(f"Exporting results to {filename} ({format})")

        export_results = _sibling('utils.io').export_results

        export_results(
            tracking_result=self.tracking_result,
            filename=filename,
            format=format,
            include_analysis=include_analysis
        )

    def save_project(self, filename: str):
        """
        Save entire project (data + results) to file.

        Args:
            filename: Output filename (.h5)
        """
        import h5py

        logger.info(f"Saving project to {filename}")

        with h5py.File(filename, 'w') as f:
            # Save image stack
            if self.image_stack is not None:
                f.create_dataset('images', data=self.image_stack.data)
                f.attrs['pixel_size'] = self.image_stack.pixel_size
                f.attrs['time_interval'] = self.image_stack.time_interval

            # Save tracking results
            if self.tracking_result is not None:
                df = self.tracking_result.to_dataframe()
                f.create_dataset('tracks', data=df.to_records(index=False))

            # Save config
            import json
            f.attrs['config'] = json.dumps(self.config)

        logger.info("Project saved successfully")

    @classmethod
    def load_project(cls, filename: str) -> 'ParticleTracker':
        """
        Load project from file.

        Args:
            filename: Project file (.h5)

        Returns:
            ParticleTracker instance with loaded data
        """
        import h5py
        import json
        import pandas as pd

        logger.info(f"Loading project from {filename}")

        with h5py.File(filename, 'r') as f:
            # Load config
            config = json.loads(f.attrs['config'])
            tracker = cls(config=config)

            # Load image stack
            if 'images' in f:
                data = f['images'][:]
                tracker.image_stack = ImageStack(
                    data=data,
                    pixel_size=f.attrs['pixel_size'],
                    time_interval=f.attrs['time_interval']
                )

            # Load tracking results
            if 'tracks' in f:
                df = pd.DataFrame(f['tracks'][:])
                tracker.tracking_result = TrackingResult.from_dataframe(df)

        logger.info("Project loaded successfully")
        return tracker

    def __repr__(self) -> str:
        """String representation."""
        status = []
        if self.image_stack:
            status.append(f"Data: {self.image_stack.n_frames} frames")
        if self.detections:
            status.append(f"{len(self.detections)} detections")
        if self.tracking_result:
            status.append(f"{self.tracking_result.n_tracks} tracks")

        return f"ParticleTracker({', '.join(status) if status else 'empty'})"
