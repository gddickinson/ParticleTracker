# ParticleTracker Quick Start Guide

## Installation

```bash
cd ParticleTracker
pip install -r requirements.txt
pip install -e .
```

## 5-Minute Tutorial

### 1. Import and Create Tracker

```python
from src.core.pipeline import ParticleTracker

tracker = ParticleTracker()
```

### 2. Load Data (or Generate Test Data)

```python
# Option A: Load your data
tracker.load_data('your_movie.tif', 
                  pixel_size=0.1,      # micrometers
                  time_interval=0.1)    # seconds

# Option B: Generate synthetic data for testing
tracker.create_synthetic(n_frames=100, 
                        n_particles=20)
```

### 3. Detect Particles

```python
# Wavelet detection (most accurate)
detections = tracker.detect(
    method='wavelet',
    sigma=1.5,        # PSF size
    threshold=3.0     # SNR threshold
)

print(f"Found {len(detections)} particles")
```

### 4. Track Particles

```python
# LAP tracking (most robust)
result = tracker.track(
    method='lap',
    max_displacement=5.0,  # pixels
    gap_closing=3,         # frames
    min_track_length=10    # filter short tracks
)

print(f"Created {result.n_tracks} tracks")
```

### 5. Analyze Motion

```python
# Calculate MSD
msd = tracker.analyze_msd(max_lag=30)

# Fit diffusion model
fit = tracker.fit_diffusion(msd, model='brownian')
print(f"Diffusion coefficient: {fit['diffusion_coefficient']:.4f} μm²/s")

# Classify motion types
classifications = tracker.classify_motion()
```

### 6. Visualize and Export

```python
# Create visualization
tracker.visualize(output='my_tracks.png', show=True)

# Export results
tracker.export('results.csv', format='csv')
tracker.save_project('my_analysis.h5')
```

## Complete Example Script

```python
from src.core.pipeline import ParticleTracker

# Initialize
tracker = ParticleTracker()

# Generate test data
tracker.create_synthetic(n_frames=50, n_particles=15)

# Detect
detections = tracker.detect(method='wavelet', sigma=1.5, threshold=3.0)
print(f"Detections: {len(detections)}")

# Track  
tracks = tracker.track(method='lap', max_displacement=5.0)
print(f"Tracks: {tracks.n_tracks}")

# Analyze
msd = tracker.analyze_msd()
diffusion = tracker.fit_diffusion(msd, model='brownian')
print(f"D = {diffusion['diffusion_coefficient']:.4f} μm²/s")

# Visualize
tracker.visualize(output='tracks.png')

# Export
tracker.export('results.csv')

print("Done!")
```

## Running the Examples

```bash
# Run the comprehensive example
cd ParticleTracker
python examples/example_workflow.py
```

This will:
1. Generate synthetic data
2. Detect and track particles
3. Perform motion analysis
4. Create visualizations
5. Export results to multiple formats

## Using the GUI

```bash
# Launch the GUI (requires PyQt5)
python -m src.gui.main
```

The GUI provides:
- Visual data loading
- Interactive parameter tuning
- Real-time visualization
- One-click exports

## Common Workflows

### Workflow 1: Quick Analysis

```python
tracker = ParticleTracker()
tracker.load_data('movie.tif')
tracker.detect(method='dog')  # Fast
tracker.track(method='nearest')  # Fast
tracker.export('results.csv')
```

### Workflow 2: High-Precision STORM Analysis

```python
tracker = ParticleTracker()
tracker.load_data('storm_data.tif')
detections = tracker.detect(
    method='wavelet',
    sigma=1.2,
    threshold=5.0,
    subpixel=True
)
tracks = tracker.track(
    method='lap',
    max_displacement=2.0,
    gap_closing=5
)
msd = tracker.analyze_msd()
tracker.visualize(output='storm_tracks.png')
```

### Workflow 3: Batch Processing

```python
import glob

tracker = ParticleTracker()

for movie_file in glob.glob('data/*.tif'):
    print(f"Processing {movie_file}...")
    
    tracker.load_data(movie_file)
    tracker.detect(method='log')
    result = tracker.track(method='lap')
    
    output_name = movie_file.replace('.tif', '_results.csv')
    tracker.export(output_name)
```

## Customization

### Custom Detection Parameters

```python
# Adjust for your specific microscopy setup
detections = tracker.detect(
    method='wavelet',
    sigma=2.0,           # Larger particles
    threshold=4.0,       # More stringent
    min_distance=10      # Well-separated
)
```

### Custom Tracking Parameters

```python
# For fast-moving particles
tracks = tracker.track(
    method='lap',
    max_displacement=10.0,     # Larger search
    gap_closing=1,             # Less gap tolerance
    motion_model='directed'    # Directional motion
)
```

## Troubleshooting

### Too Many/Few Detections?

```python
# Too many: increase threshold
detections = tracker.detect(threshold=5.0)  # More stringent

# Too few: decrease threshold
detections = tracker.detect(threshold=2.0)  # More sensitive
```

### Poor Tracking?

```python
# Increase search radius
tracks = tracker.track(max_displacement=10.0)

# Enable gap closing
tracks = tracker.track(gap_closing=5)

# Try different method
tracks = tracker.track(method='nearest')  # Simpler
```

### Memory Issues?

```python
# Process subset of frames
tracker.detect(frames=range(0, 100))  # First 100 frames

# Use faster method
tracker.detect(method='dog')  # Less memory
```

## Getting Help

- Check the full documentation: `docs/TECHNICAL.md`
- Run example script: `python examples/example_workflow.py`
- Read docstrings: `help(tracker.detect)`

## Next Steps

1. Try with your own data
2. Experiment with different parameters
3. Explore motion classification
4. Create custom visualizations
5. Integrate into your analysis pipeline

Happy tracking!
