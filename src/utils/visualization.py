"""
Visualization utilities for particle tracking results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, List, Tuple
from pathlib import Path

try:
    from ..core.data_structures import ImageStack, Track, TrackingResult
except ImportError:
    from core.data_structures import ImageStack, Track, TrackingResult


def plot_tracks(image_stack: Optional[ImageStack] = None,
                tracking_result: Optional[TrackingResult] = None,
                frame: Optional[int] = None,
                tracks_to_plot: Optional[List[int]] = None,
                color_by: str = 'track_id',
                show_detections: bool = True,
                show_tails: bool = True,
                tail_length: int = 10,
                figsize: Tuple[int, int] = (12, 10),
                output: Optional[str] = None,
                show: bool = True,
                **kwargs):
    """
    Plot tracking results overlaid on image.

    Args:
        image_stack: ImageStack to display as background
        tracking_result: TrackingResult with tracks
        frame: Frame to display (None = maximum projection)
        tracks_to_plot: List of track IDs to plot (None = all)
        color_by: Color coding ('track_id', 'velocity', 'length')
        show_detections: Show detection points
        show_tails: Show trajectory tails
        tail_length: Length of tail in frames
        figsize: Figure size
        output: Output filename (None = don't save)
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot background image
    if image_stack is not None:
        if frame is None:
            # Maximum intensity projection
            img = np.max(image_stack.data, axis=0)
        else:
            img = image_stack.get_frame(frame)

        ax.imshow(img, cmap='gray', interpolation='nearest')

    # Plot tracks
    if tracking_result is not None:
        tracks = tracking_result.tracks

        if tracks_to_plot is not None:
            tracks = [t for t in tracks if t.track_id in tracks_to_plot]

        # Determine colors
        if color_by == 'track_id':
            colors = _generate_colors(len(tracks))
            track_colors = {t.track_id: colors[i] for i, t in enumerate(tracks)}
        elif color_by == 'velocity':
            # Calculate mean velocity for each track
            from ..analysis.motion import calculate_velocity
            velocities = [calculate_velocity(t)['mean_velocity'] for t in tracks]
            norm_velocities = (velocities - np.min(velocities)) / (np.max(velocities) - np.min(velocities) + 1e-10)
            track_colors = {t.track_id: plt.cm.viridis(norm_velocities[i])
                          for i, t in enumerate(tracks)}
        elif color_by == 'length':
            lengths = [t.length for t in tracks]
            norm_lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths) + 1e-10)
            track_colors = {t.track_id: plt.cm.plasma(norm_lengths[i])
                          for i, t in enumerate(tracks)}
        else:
            track_colors = {t.track_id: 'cyan' for t in tracks}

        # Plot each track
        for track in tracks:
            color = track_colors[track.track_id]

            # Get positions
            positions = track.positions

            if len(positions) < 2:
                continue

            # Plot trajectory
            if frame is None or show_tails:
                # Full trajectory or tail
                if frame is not None and show_tails:
                    # Only show tail up to current frame
                    end_idx = sum(1 for d in track.detections if d.t <= frame)
                    start_idx = max(0, end_idx - tail_length)
                    positions_to_plot = positions[start_idx:end_idx]
                else:
                    positions_to_plot = positions

                if len(positions_to_plot) > 1:
                    ax.plot(positions_to_plot[:, 0], positions_to_plot[:, 1],
                           color=color, alpha=0.6, linewidth=2)

            # Plot detections
            if show_detections:
                if frame is None:
                    # All detections
                    ax.scatter(positions[:, 0], positions[:, 1],
                             c=[color], s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
                else:
                    # Only detections in current frame
                    frame_dets = [d for d in track.detections if d.t == frame]
                    if frame_dets:
                        for det in frame_dets:
                            circle = patches.Circle((det.x, det.y), 3,
                                                   color=color, alpha=0.8,
                                                   edgecolor='white', linewidth=1)
                            ax.add_patch(circle)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Particle Tracks (Frame {frame if frame is not None else "All"})')
    ax.axis('equal')

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_msd(msd_results: dict,
             fit_results: Optional[dict] = None,
             figsize: Tuple[int, int] = (10, 6),
             output: Optional[str] = None,
             show: bool = True):
    """
    Plot mean squared displacement.

    Args:
        msd_results: Output from calculate_msd()
        fit_results: Optional fit results from fit_diffusion_model()
        figsize: Figure size
        output: Output filename
        show: Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)

    time_lags = msd_results['time_lags']
    mean_msd = msd_results['mean_msd']
    std_msd = msd_results['std_msd']

    # Plot MSD with error bars
    ax.errorbar(time_lags, mean_msd, yerr=std_msd,
               fmt='o', label='Data', capsize=3)

    # Plot fit if provided
    if fit_results and fit_results.get('success', False):
        model = fit_results['model']

        if model == 'brownian':
            D = fit_results['diffusion_coefficient']
            fit_msd = 4 * D * time_lags
            label = f'Brownian (D = {D:.3f} μm²/s)'
        elif model == 'directed':
            D = fit_results['diffusion_coefficient']
            v = fit_results['velocity']
            fit_msd = 4 * D * time_lags + (v * time_lags)**2
            label = f'Directed (D = {D:.3f}, v = {v:.3f})'
        elif model == 'anomalous':
            D = fit_results['diffusion_coefficient']
            alpha = fit_results['alpha']
            fit_msd = 4 * D * time_lags**alpha
            label = f'Anomalous (D = {D:.3f}, α = {alpha:.3f})'
        else:
            fit_msd = None
            label = None

        if fit_msd is not None:
            ax.plot(time_lags, fit_msd, 'r-', linewidth=2, label=label)

    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('MSD (μm²)')
    ax.set_title(f'Mean Squared Displacement (n = {msd_results["n_tracks"]} tracks)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_track_statistics(tracking_result: TrackingResult,
                          figsize: Tuple[int, int] = (15, 5),
                          output: Optional[str] = None,
                          show: bool = True):
    """
    Plot summary statistics of tracks.

    Args:
        tracking_result: TrackingResult
        figsize: Figure size
        output: Output filename
        show: Whether to display
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    tracks = tracking_result.tracks

    # Track lengths
    lengths = [t.length for t in tracks]
    axes[0].hist(lengths, bins=30, edgecolor='black')
    axes[0].set_xlabel('Track length (detections)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Track Length Distribution')
    axes[0].grid(True, alpha=0.3)

    # Track durations
    durations = [t.duration for t in tracks]
    axes[1].hist(durations, bins=30, edgecolor='black', color='orange')
    axes[1].set_xlabel('Track duration (frames)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Track Duration Distribution')
    axes[1].grid(True, alpha=0.3)

    # Detection intensities
    all_intensities = []
    for track in tracks:
        all_intensities.extend([d.intensity for d in track.detections])

    axes[2].hist(all_intensities, bins=50, edgecolor='black', color='green')
    axes[2].set_xlabel('Intensity')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Detection Intensity Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_classification(tracking_result: TrackingResult,
                                   classifications: dict,
                                   figsize: Tuple[int, int] = (12, 10),
                                   output: Optional[str] = None,
                                   show: bool = True):
    """
    Plot trajectories colored by motion classification.

    Args:
        tracking_result: TrackingResult
        classifications: Dictionary mapping track_id to motion class
        figsize: Figure size
        output: Output filename
        show: Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each class
    class_colors = {
        'brownian': 'blue',
        'directed': 'red',
        'confined': 'green',
        'subdiffusive': 'purple',
        'superdiffusive': 'orange',
        'insufficient_data': 'gray'
    }

    # Plot tracks
    for track in tracking_result.tracks:
        motion_class = classifications.get(track.track_id, 'insufficient_data')
        color = class_colors.get(motion_class, 'gray')

        positions = track.positions
        if len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1],
                   color=color, alpha=0.6, linewidth=1.5,
                   label=motion_class if motion_class not in ax.get_legend_handles_labels()[1] else "")

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Trajectories by Motion Classification')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def _generate_colors(n: int) -> List:
    """Generate n distinct colors."""
    if n <= 10:
        return sns.color_palette("tab10", n)
    else:
        return sns.color_palette("husl", n)


def create_movie(image_stack: ImageStack,
                tracking_result: TrackingResult,
                output: str,
                fps: int = 10,
                tail_length: int = 10,
                **kwargs):
    """
    Create movie with tracks overlaid.

    Args:
        image_stack: Image stack
        tracking_result: Tracking results
        output: Output filename (.mp4, .avi, .gif)
        fps: Frames per second
        tail_length: Length of trajectory tail
        **kwargs: Additional plotting parameters
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine writer based on extension
    ext = Path(output).suffix.lower()
    if ext == '.gif':
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=1800)

    def update(frame):
        ax.clear()

        # Show image
        img = image_stack.get_frame(frame)
        ax.imshow(img, cmap='gray')

        # Plot tracks with tails
        plot_tracks(
            image_stack=None,
            tracking_result=tracking_result,
            frame=frame,
            show_detections=True,
            show_tails=True,
            tail_length=tail_length,
            show=False,
            **kwargs
        )

        ax.set_title(f'Frame {frame}')

    anim = FuncAnimation(fig, update, frames=image_stack.n_frames, repeat=True)
    anim.save(output, writer=writer)
    plt.close()
