"""
Synthetic particle movies with ground truth (the repo's test-data simulator).

create_synthetic_data() renders particles as Gaussian spots (PSF sigma) on
a dark background with additive Gaussian noise. Each particle moves by one of
three simple models, in pixels per frame:

- 'brownian': steps N(0, 2 D) per axis (D in px^2/frame).
- 'directed': the Brownian step plus a constant drift `velocity` px/frame in
  a random direction.
- 'confined': Brownian steps rejected when they would leave a disc of radius
  `confinement_radius` px around the start position.

Positions use the pixel-centre convention: x is the column, y the row, and a
particle at x = 10.0 sits in the middle of column 10. Each particle's
intensity is split bilinearly over the four nearest pixels before the PSF
blur, so the spot is centred on the sub-pixel position.

Particles that leave the frame are clamped to its edge (as the original
generator did).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from .data_structures import ImageStack
except ImportError:
    from core.data_structures import ImageStack

MOTION_TYPES = ('brownian', 'directed', 'confined')


def _step(rng, n, D):
    return rng.standard_normal((n, 2)) * np.sqrt(2 * D)


def simulate_trajectories(n_frames: int, n_particles: int, frame_shape: Tuple[int, int],
                          rng, diffusion_coeff: float = 1.0,
                          motion_types: Optional[Sequence[str]] = None,
                          velocity: float = 0.5, confinement_radius: float = 3.0,
                          margin: float = 0.0) -> List[Dict]:
    """Ground-truth trajectories: list of {track_id, motion_type, frames, x, y} (pixels)."""
    h, w = frame_shape
    types = list(motion_types) if motion_types is not None else ['brownian'] * n_particles
    if len(types) != n_particles or any(t not in MOTION_TYPES for t in types):
        raise ValueError(f"motion_types needs {n_particles} entries from {MOTION_TYPES}")
    lo = np.array([margin, margin])
    hi = np.array([h - 1 - margin, w - 1 - margin])
    start = lo + rng.random_sample((n_particles, 2)) * (hi - lo)       # (y, x)
    angle = rng.random_sample(n_particles) * 2 * np.pi
    drift = velocity * np.stack([np.sin(angle), np.cos(angle)], axis=1)
    pos = start.copy()
    path = np.zeros((n_frames, n_particles, 2))
    for t in range(n_frames):
        step = _step(rng, n_particles, diffusion_coeff)
        for i, kind in enumerate(types):
            if kind == 'directed':
                step[i] += drift[i]
            elif kind == 'confined':
                for _ in range(100):
                    if np.hypot(*(pos[i] + step[i] - start[i])) <= confinement_radius:
                        break
                    step[i] = _step(rng, 1, diffusion_coeff)[0]
                else:
                    step[i] = 0.0
        pos = np.clip(pos + step, [0, 0], [h - 1, w - 1])
        path[t] = pos
    return [{'track_id': i, 'motion_type': types[i], 'frames': list(range(n_frames)),
             'x': path[:, i, 1].tolist(), 'y': path[:, i, 0].tolist()} for i in range(n_particles)]


def render_frame(xy: np.ndarray, frame_shape: Tuple[int, int], intensity: float,
                 psf_sigma: float) -> np.ndarray:
    """Noise-free frame: bilinear splat of each (x, y) then a Gaussian PSF blur."""
    h, w = frame_shape
    img = np.zeros(frame_shape, dtype=np.float64)
    for x, y in xy:
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < h and 0 <= xx < w:
                    img[yy, xx] += intensity * wy * wx
    return gaussian_filter(img, sigma=psf_sigma)


def create_synthetic_data(n_frames: int = 100,
                          n_particles: int = 10,
                          frame_shape: Tuple[int, int] = (256, 256),
                          noise_level: float = 10.0,
                          return_ground_truth: bool = False,
                          **kwargs) -> Union[ImageStack, Tuple[ImageStack, List[Dict]]]:
    """
    Generate a synthetic particle movie for testing.

    Args:
        n_frames: Number of time frames
        n_particles: Number of particles
        frame_shape: (height, width) of each frame
        noise_level: Standard deviation of the additive Gaussian noise
        return_ground_truth: also return the true trajectories
        **kwargs: diffusion_coeff (px^2/frame, 1.0), psf_sigma (px, 1.5),
            intensity (1000), motion_types (one per particle, default all
            'brownian'), velocity (px/frame, 0.5), confinement_radius (px,
            3.0), margin (px from the edge for start positions, 0),
            background (added offset, 0), seed (None: numpy's global
            generator, as before), pixel_size (um, 0.1), time_interval (s, 0.1)

    Returns:
        ImageStack (float32), or (ImageStack, ground_truth list) when
        return_ground_truth is True.
    """
    seed = kwargs.get('seed')
    rng = np.random.RandomState(seed) if seed is not None else np.random.mtrand._rand
    psf_sigma = kwargs.get('psf_sigma', 1.5)
    intensity = kwargs.get('intensity', 1000.0)
    background = kwargs.get('background', 0.0)

    truth = simulate_trajectories(
        n_frames, n_particles, frame_shape, rng,
        diffusion_coeff=kwargs.get('diffusion_coeff', 1.0),
        motion_types=kwargs.get('motion_types'),
        velocity=kwargs.get('velocity', 0.5),
        confinement_radius=kwargs.get('confinement_radius', 3.0),
        margin=kwargs.get('margin', 0.0))

    data = np.zeros((n_frames, *frame_shape), dtype=np.float32)
    for t in range(n_frames):
        xy = [(tr['x'][t], tr['y'][t]) for tr in truth]
        data[t] = render_frame(xy, frame_shape, intensity, psf_sigma) + background
        data[t] += rng.standard_normal(frame_shape) * noise_level

    stack = ImageStack(
        data=data,
        pixel_size=kwargs.get('pixel_size', 0.1),
        time_interval=kwargs.get('time_interval', 0.1),
        metadata={'synthetic': True, 'n_particles': n_particles, 'seed': seed,
                  'diffusion_coeff_px2_per_frame': kwargs.get('diffusion_coeff', 1.0),
                  'psf_sigma_px': psf_sigma, 'noise_level': noise_level,
                  'motion_types': [tr['motion_type'] for tr in truth]}
    )
    return (stack, truth) if return_ground_truth else stack
