"""
Core data structures for particle tracking.

This module defines the fundamental data structures used throughout the
ParticleTracker package for representing images, detections, and tracks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ImageStack:
    """
    Container for microscopy image stack data.
    
    Attributes:
        data: 3D/4D numpy array (T, [Z], Y, X) or (T, Y, X)
        metadata: Dictionary containing acquisition parameters
        pixel_size: Physical pixel size in micrometers
        time_interval: Time between frames in seconds
        channels: Number of channels (default 1)
    """
    data: np.ndarray
    metadata: Dict = field(default_factory=dict)
    pixel_size: float = 0.1  # micrometers
    time_interval: float = 0.1  # seconds
    channels: int = 1
    
    def __post_init__(self):
        """Validate and normalize data."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
            
        # Ensure at least 3D
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, ...]
            
    @property
    def shape(self) -> Tuple:
        """Return shape of image stack."""
        return self.data.shape
    
    @property
    def n_frames(self) -> int:
        """Return number of time frames."""
        return self.shape[0]
    
    @property
    def is_3d(self) -> bool:
        """Check if data is 3D (has Z dimension)."""
        return len(self.shape) == 4
    
    def get_frame(self, t: int, channel: int = 0) -> np.ndarray:
        """
        Get a single frame from the stack.
        
        Args:
            t: Time index
            channel: Channel index (for multi-channel data)
            
        Returns:
            2D or 3D numpy array
        """
        if channel != 0 and self.channels > 1:
            # Handle multi-channel indexing here
            raise NotImplementedError("Multi-channel support coming soon")
        return self.data[t]
    
    def get_substack(self, t_start: int, t_end: int) -> 'ImageStack':
        """Extract a temporal subset of the stack."""
        return ImageStack(
            data=self.data[t_start:t_end],
            metadata=self.metadata.copy(),
            pixel_size=self.pixel_size,
            time_interval=self.time_interval,
            channels=self.channels
        )


@dataclass
class Detection:
    """
    Single particle detection at one time point.
    
    Attributes:
        x, y: Position in pixels
        z: Z position for 3D data
        t: Time frame index
        intensity: Integrated intensity
        sigma_x, sigma_y: PSF width estimates
        sigma_z: Z PSF width for 3D
        uncertainty: Localization uncertainty (nm)
        chi_squared: Goodness of fit metric
        snr: Signal-to-noise ratio
    """
    x: float
    y: float
    z: Optional[float] = None
    t: int = 0
    intensity: float = 0.0
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    sigma_z: Optional[float] = None
    uncertainty: float = 0.0
    chi_squared: float = 0.0
    snr: float = 0.0
    
    def __post_init__(self):
        """Validate detection parameters."""
        if self.x < 0 or self.y < 0:
            raise ValueError("Coordinates must be non-negative")
        if self.z is not None and self.z < 0:
            raise ValueError("Z coordinate must be non-negative")
    
    @property
    def position(self) -> np.ndarray:
        """Return position as array."""
        if self.z is not None:
            return np.array([self.x, self.y, self.z])
        return np.array([self.x, self.y])
    
    @property
    def is_3d(self) -> bool:
        """Check if detection is 3D."""
        return self.z is not None
    
    def distance_to(self, other: 'Detection') -> float:
        """
        Calculate Euclidean distance to another detection.
        
        Args:
            other: Another Detection object
            
        Returns:
            Distance in pixels
        """
        if self.is_3d and other.is_3d:
            return np.sqrt((self.x - other.x)**2 + 
                          (self.y - other.y)**2 +
                          (self.z - other.z)**2)
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Track:
    """
    Particle trajectory composed of linked detections.
    
    Attributes:
        detections: List of Detection objects
        track_id: Unique track identifier
        start_frame: First frame of track
        end_frame: Last frame of track
        gap_frames: Frames where detection failed
        quality_score: Trackability/quality metric
    """
    
    def __init__(self, track_id: int = 0):
        """
        Initialize empty track.
        
        Args:
            track_id: Unique identifier for this track
        """
        self.track_id = track_id
        self.detections: List[Detection] = []
        self.gap_frames: List[int] = []
        self.quality_score: float = 0.0
        self._properties: Dict = {}
        
    def add_detection(self, detection: Detection):
        """Add a detection to the track."""
        self.detections.append(detection)
        self.detections.sort(key=lambda d: d.t)
        
    def add_gap(self, frame: int):
        """Mark a frame as a gap (missing detection)."""
        if frame not in self.gap_frames:
            self.gap_frames.append(frame)
            self.gap_frames.sort()
    
    @property
    def length(self) -> int:
        """Return number of detections in track."""
        return len(self.detections)
    
    @property
    def duration(self) -> int:
        """Return temporal span of track in frames."""
        if not self.detections:
            return 0
        return self.end_frame - self.start_frame + 1
    
    @property
    def start_frame(self) -> int:
        """Return first frame of track."""
        if not self.detections:
            return -1
        return min(d.t for d in self.detections)
    
    @property
    def end_frame(self) -> int:
        """Return last frame of track."""
        if not self.detections:
            return -1
        return max(d.t for d in self.detections)
    
    @property
    def positions(self) -> np.ndarray:
        """Return array of positions (N x 2 or N x 3)."""
        if not self.detections:
            return np.array([])
        return np.array([d.position for d in self.detections])
    
    @property
    def times(self) -> np.ndarray:
        """Return array of time indices."""
        return np.array([d.t for d in self.detections])
    
    @property
    def is_3d(self) -> bool:
        """Check if track is 3D."""
        if not self.detections:
            return False
        return self.detections[0].is_3d
    
    def get_detection_at_frame(self, frame: int) -> Optional[Detection]:
        """Get detection at specific frame, or None if gap."""
        for detection in self.detections:
            if detection.t == frame:
                return detection
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert track to pandas DataFrame.
        
        Returns:
            DataFrame with columns: frame, x, y, [z], intensity, etc.
        """
        data = {
            'frame': [d.t for d in self.detections],
            'x': [d.x for d in self.detections],
            'y': [d.y for d in self.detections],
            'intensity': [d.intensity for d in self.detections],
            'uncertainty': [d.uncertainty for d in self.detections],
            'track_id': [self.track_id] * len(self.detections)
        }
        
        if self.is_3d:
            data['z'] = [d.z for d in self.detections]
            
        return pd.DataFrame(data)
    
    def set_property(self, key: str, value):
        """Set custom property."""
        self._properties[key] = value
        
    def get_property(self, key: str, default=None):
        """Get custom property."""
        return self._properties.get(key, default)


class TrackingResult:
    """
    Container for complete tracking results.
    
    Attributes:
        tracks: List of Track objects
        detections: List of all Detection objects
        metadata: Analysis metadata and parameters
    """
    
    def __init__(self):
        """Initialize empty tracking result."""
        self.tracks: List[Track] = []
        self.detections: List[Detection] = []
        self.metadata: Dict = {}
        
    def add_track(self, track: Track):
        """Add a track to results."""
        self.tracks.append(track)
        
    def add_detections(self, detections: List[Detection]):
        """Add detections to results."""
        self.detections.extend(detections)
        
    @property
    def n_tracks(self) -> int:
        """Return number of tracks."""
        return len(self.tracks)
    
    @property
    def n_detections(self) -> int:
        """Return total number of detections."""
        return len(self.detections)
    
    def get_tracks_by_frame(self, frame: int) -> List[Track]:
        """Get all tracks active at a specific frame."""
        return [t for t in self.tracks 
                if t.start_frame <= frame <= t.end_frame]
    
    def filter_tracks(self, min_length: int = None, 
                      min_duration: int = None,
                      quality_threshold: float = None) -> 'TrackingResult':
        """
        Filter tracks based on criteria.
        
        Args:
            min_length: Minimum number of detections
            min_duration: Minimum temporal span
            quality_threshold: Minimum quality score
            
        Returns:
            New TrackingResult with filtered tracks
        """
        filtered = TrackingResult()
        filtered.metadata = self.metadata.copy()
        
        for track in self.tracks:
            keep = True
            if min_length is not None and track.length < min_length:
                keep = False
            if min_duration is not None and track.duration < min_duration:
                keep = False
            if quality_threshold is not None and track.quality_score < quality_threshold:
                keep = False
                
            if keep:
                filtered.add_track(track)
                
        return filtered
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all tracks to single DataFrame.
        
        Returns:
            DataFrame with all track data
        """
        if not self.tracks:
            return pd.DataFrame()
            
        dfs = [track.to_dataframe() for track in self.tracks]
        return pd.concat(dfs, ignore_index=True)
    
    def save(self, filename: str, format: str = 'csv'):
        """
        Save results to file.
        
        Args:
            filename: Output filename
            format: File format ('csv', 'hdf5', 'pickle')
        """
        df = self.to_dataframe()
        
        if format == 'csv':
            df.to_csv(filename, index=False)
        elif format == 'hdf5':
            df.to_hdf(filename, key='tracks', mode='w')
        elif format == 'pickle':
            df.to_pickle(filename)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, filename: str, format: str = 'csv') -> 'TrackingResult':
        """
        Load results from file.
        
        Args:
            filename: Input filename
            format: File format
            
        Returns:
            TrackingResult object
        """
        if format == 'csv':
            df = pd.read_csv(filename)
        elif format == 'hdf5':
            df = pd.read_hdf(filename, key='tracks')
        elif format == 'pickle':
            df = pd.read_pickle(filename)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'TrackingResult':
        """Rebuild tracks from a DataFrame with track_id, frame, x, y
        (optional z, intensity, uncertainty) columns."""
        result = cls()

        for track_id, group in df.groupby('track_id'):
            track = Track(track_id=int(track_id))

            for _, row in group.iterrows():
                z = row.get('z', None)
                detection = Detection(
                    x=float(row['x']),
                    y=float(row['y']),
                    z=None if z is None or pd.isna(z) else float(z),
                    t=int(row['frame']),
                    intensity=float(row.get('intensity', 0.0)),
                    uncertainty=float(row.get('uncertainty', 0.0))
                )
                track.add_detection(detection)

            result.add_track(track)

        return result


def create_synthetic_data(n_frames: int = 100, 
                         n_particles: int = 10,
                         frame_shape: Tuple[int, int] = (256, 256),
                         noise_level: float = 10.0,
                         **kwargs) -> ImageStack:
    """
    Generate synthetic particle movie for testing.
    
    Args:
        n_frames: Number of time frames
        n_particles: Number of particles
        frame_shape: (height, width) of each frame
        noise_level: Standard deviation of Gaussian noise
        **kwargs: Additional parameters (diffusion_coeff, psf_sigma, etc.)
        
    Returns:
        ImageStack with synthetic data
    """
    from scipy.ndimage import gaussian_filter
    
    # Extract parameters
    diffusion_coeff = kwargs.get('diffusion_coeff', 1.0)
    psf_sigma = kwargs.get('psf_sigma', 1.5)
    intensity = kwargs.get('intensity', 1000.0)
    
    # Initialize stack
    data = np.zeros((n_frames, *frame_shape), dtype=np.float32)
    
    # Generate random walk trajectories
    positions = np.random.rand(n_particles, 2) * np.array(frame_shape)
    
    for t in range(n_frames):
        # Update positions (Brownian motion)
        positions += np.random.randn(n_particles, 2) * np.sqrt(2 * diffusion_coeff)
        
        # Keep particles in frame
        positions = np.clip(positions, [0, 0], frame_shape)
        
        # Render particles as Gaussians
        for pos in positions:
            y, x = int(pos[0]), int(pos[1])
            if 0 <= y < frame_shape[0] and 0 <= x < frame_shape[1]:
                data[t, y, x] += intensity
                
        # Apply Gaussian PSF
        data[t] = gaussian_filter(data[t], sigma=psf_sigma)
        
        # Add noise
        data[t] += np.random.randn(*frame_shape) * noise_level
        
    return ImageStack(
        data=data,
        pixel_size=0.1,
        time_interval=0.1,
        metadata={'synthetic': True, 'n_particles': n_particles}
    )
