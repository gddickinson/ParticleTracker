"""
Example workflow demonstrating ParticleTracker capabilities.

This script shows how to:
1. Generate or load microscopy data
2. Detect particles
3. Track particles
4. Analyze motion
5. Classify trajectories
6. Visualize and export results
"""

import numpy as np
import sys
from pathlib import Path

# Make the repository root importable (the package is 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ParticleTracker
from src.core.data_structures import create_synthetic_data


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
    print("=" * 80)
    print("ParticleTracker Example Workflow")
    print("=" * 80)
    
    # =========================================================================
    # 1. Load or Generate Data
    # =========================================================================
    print("\n1. Generating synthetic test data...")
    
    tracker = ParticleTracker()
    
    # Generate synthetic data with multiple particles undergoing Brownian motion
    image_stack = tracker.create_synthetic(
        n_frames=100,
        n_particles=20,
        frame_shape=(256, 256),
        diffusion_coeff=1.0,
        psf_sigma=1.5,
        intensity=1000.0,
        noise_level=50.0
    )
    
    print(f"   Generated {image_stack.n_frames} frames")
    print(f"   Frame shape: {image_stack.shape[1:]}")
    
    # Alternatively, load real data:
    # image_stack = tracker.load_data('path/to/microscopy_data.tif',
    #                                  pixel_size=0.1,  # micrometers
    #                                  time_interval=0.1)  # seconds
    
    # =========================================================================
    # 2. Detect Particles
    # =========================================================================
    print("\n2. Detecting particles...")
    
    # Try different detection methods:
    
    # Method 1: Wavelet detection (ThunderSTORM-style)
    detections_wavelet = tracker.detect(
        method='wavelet',
        sigma=1.5,
        threshold=3.0,
        min_distance=5
    )
    print(f"   Wavelet detector found {len(detections_wavelet)} detections")
    
    # Method 2: LoG detection (fast)
    # detections = tracker.detect(
    #     method='log',
    #     min_sigma=1.0,
    #     max_sigma=3.0,
    #     threshold=0.05
    # )
    
    # Method 3: DoG detection (faster)
    # detections = tracker.detect(
    #     method='dog',
    #     min_sigma=1.0,
    #     max_sigma=3.0,
    #     threshold=0.05
    # )
    
    # =========================================================================
    # 3. Track Particles
    # =========================================================================
    print("\n3. Tracking particles...")
    
    # Method 1: LAP tracker (U-track style, most robust)
    tracking_result = tracker.track(
        method='lap',
        max_displacement=5.0,
        gap_closing=3,
        merge_splits=True,
        min_track_length=10
    )
    print(f"   LAP tracker found {tracking_result.n_tracks} tracks")
    
    # Method 2: Nearest neighbor (faster, simpler)
    # tracking_result = tracker.track(
    #     method='nearest',
    #     max_displacement=5.0
    # )
    
    # =========================================================================
    # 4. Analyze Motion
    # =========================================================================
    print("\n4. Analyzing particle motion...")
    
    # Calculate Mean Squared Displacement
    msd_results = tracker.analyze_msd(max_lag=30)
    print(f"   Calculated MSD for {msd_results['n_tracks']} tracks")
    
    # Fit diffusion models
    brownian_fit = tracker.fit_diffusion(msd_results, model='brownian')
    if brownian_fit['success']:
        print(f"   Brownian diffusion coefficient: {brownian_fit['diffusion_coefficient']:.4f} μm²/s")
        print(f"   R² = {brownian_fit['r_squared']:.4f}")
    
    directed_fit = tracker.fit_diffusion(msd_results, model='directed')
    if directed_fit['success']:
        print(f"   Directed motion - D: {directed_fit['diffusion_coefficient']:.4f}, "
              f"v: {directed_fit['velocity']:.4f}")
    
    anomalous_fit = tracker.fit_diffusion(msd_results, model='anomalous')
    if anomalous_fit['success']:
        print(f"   Anomalous diffusion - α: {anomalous_fit['alpha']:.4f}")
        print(f"   Motion type: {anomalous_fit['motion_type']}")
    
    # =========================================================================
    # 5. Classify Trajectories
    # =========================================================================
    print("\n5. Classifying trajectories...")
    
    classifications = tracker.classify_motion(method='msd')
    
    # Count each class
    from collections import Counter
    class_counts = Counter(classifications.values())
    
    print("   Motion classification summary:")
    for motion_class, count in class_counts.items():
        print(f"      {motion_class}: {count}")
    
    # =========================================================================
    # 6. Visualize Results
    # =========================================================================
    print("\n6. Creating visualizations...")
    
    # Plot tracks overlaid on image
    from src.utils.visualization import (
        plot_tracks, plot_msd, plot_track_statistics,
        plot_trajectory_classification
    )
    
    print("   - Generating track overlay plot...")
    plot_tracks(
        image_stack=image_stack,
        tracking_result=tracking_result,
        color_by='track_id',
        show_detections=True,
        show_tails=True,
        tail_length=20,
        output='example_tracks.png',
        show=False
    )
    
    print("   - Generating MSD plot...")
    plot_msd(
        msd_results=msd_results,
        fit_results=brownian_fit,
        output='example_msd.png',
        show=False
    )
    
    print("   - Generating statistics plots...")
    plot_track_statistics(
        tracking_result=tracking_result,
        output='example_statistics.png',
        show=False
    )
    
    print("   - Generating classification plot...")
    plot_trajectory_classification(
        tracking_result=tracking_result,
        classifications=classifications,
        output='example_classification.png',
        show=False
    )
    
    # =========================================================================
    # 7. Export Results
    # =========================================================================
    print("\n7. Exporting results...")
    
    # Export to CSV
    tracker.export('example_results.csv', format='csv')
    print("   Exported to example_results.csv")
    
    # Export to HDF5
    tracker.export('example_results.h5', format='hdf5')
    print("   Exported to example_results.h5")
    
    # Export to MATLAB
    tracker.export('example_results.mat', format='mat')
    print("   Exported to example_results.mat")
    
    # Save entire project
    tracker.save_project('example_project.h5')
    print("   Saved project to example_project.h5")
    
    # =========================================================================
    # 8. Advanced: Per-Track Analysis
    # =========================================================================
    print("\n8. Per-track analysis examples...")
    
    from src.analysis.motion import (
        calculate_velocity, calculate_directionality,
        detect_confinement
    )
    
    # Analyze first track in detail
    if tracking_result.tracks:
        example_track = tracking_result.tracks[0]
        
        print(f"\n   Track {example_track.track_id}:")
        print(f"      Length: {example_track.length} detections")
        print(f"      Duration: {example_track.duration} frames")
        
        velocity = calculate_velocity(example_track, time_interval=image_stack.time_interval)
        # calculate_velocity works in pixels: scale by the pixel size for um/s
        print(f"      Mean velocity: {velocity['mean_velocity'] * image_stack.pixel_size:.3f} μm/s")
        
        directionality = calculate_directionality(example_track)
        print(f"      Directionality: {directionality['directionality_ratio']:.3f}")
        
        confinement = detect_confinement(example_track)
        print(f"      Confined: {confinement['is_confined']}")
    
    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total frames analyzed: {image_stack.n_frames}")
    print(f"Total detections: {len(tracker.detections)}")
    print(f"Total tracks: {tracking_result.n_tracks}")
    print(f"Mean track length: {np.mean([t.length for t in tracking_result.tracks]):.1f}")
    print(f"Longest track: {max([t.length for t in tracking_result.tracks])} detections")
    print("\nResults saved:")
    print("  - example_tracks.png")
    print("  - example_msd.png")
    print("  - example_statistics.png")
    print("  - example_classification.png")
    print("  - example_results.csv")
    print("  - example_results.h5")
    print("  - example_results.mat")
    print("  - example_project.h5")
    print("=" * 80)


if __name__ == '__main__':
    main()
