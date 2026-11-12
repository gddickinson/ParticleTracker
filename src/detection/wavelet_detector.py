"""
Wavelet-based particle detection.

This detector uses wavelet transforms for multi-scale particle detection,
inspired by ThunderSTORM's B-spline wavelet approach.
"""

import numpy as np
from typing import List, Tuple
from scipy import ndimage

try:
    from .detector_factory import BaseDetector
    from ..core.data_structures import Detection
except ImportError:
    from detector_factory import BaseDetector
    from core.data_structures import Detection

class WaveletDetector(BaseDetector):
    """
    Wavelet-transform based particle detector.

    Uses continuous wavelet transform (Mexican hat or similar) to detect
    particles at multiple scales, providing robust detection even in
    noisy images.

    Config parameters:
        sigma: Scale parameter for wavelet (default: 1.5)
        threshold: Detection threshold in standard deviations (default: 3.0)
        min_distance: Minimum distance between particles (default: 3)
        wavelet: Wavelet type ('mexican_hat', 'morlet') (default: 'mexican_hat')
        subpixel: Whether to perform sub-pixel localization (default: True)
    """

    def __init__(self, config: dict):
        """Initialize wavelet detector."""
        super().__init__(config)

        self.sigma = config.get('sigma', 1.5)
        self.threshold = config.get('threshold', 3.0)
        self.min_distance = config.get('min_distance', 3)
        self.wavelet_type = config.get('wavelet', 'mexican_hat')
        self.subpixel = config.get('subpixel', True)

    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect particles using wavelet transform.

        Args:
            image: 2D image array
            frame_index: Frame time index

        Returns:
            List of Detection objects
        """
        # Ensure float type
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Apply wavelet transform
        wavelet_image = self._apply_wavelet_transform(image)

        # Estimate noise level
        noise_std = self._estimate_noise(wavelet_image)
        threshold_value = noise_std * self.threshold

        # Find peaks
        peaks = self._find_peaks(wavelet_image, threshold_value, self.min_distance)

        if len(peaks) == 0:
            return []

        # Create detections
        detections = []

        for peak in peaks:
            y, x = peak

            # Sub-pixel refinement
            if self.subpixel:
                refined_pos, fit_params = self._refine_position(image, x, y)
                x, y = refined_pos
                intensity = fit_params.get('amplitude', image[int(y), int(x)])
                sigma_x = fit_params.get('sigma_x', self.sigma)
                sigma_y = fit_params.get('sigma_y', self.sigma)
                chi_squared = fit_params.get('chi_squared', 0.0)
            else:
                intensity = image[int(y), int(x)]
                sigma_x = sigma_y = self.sigma
                chi_squared = 0.0

            # Calculate SNR
            snr = intensity / noise_std if noise_std > 0 else 0

            # Estimate uncertainty (Mortensen formula)
            # σ^2 = (s^2 + a^2/12) / N + (8πs^4b^2) / (a^2N^2)
            # Simplified version for now
            n_photons = intensity
            uncertainty = sigma_x / np.sqrt(max(n_photons, 1)) if n_photons > 0 else 10.0

            detection = Detection(
                x=float(x),
                y=float(y),
                t=frame_index,
                intensity=float(intensity),
                sigma_x=float(sigma_x),
                sigma_y=float(sigma_y),
                uncertainty=float(uncertainty),
                chi_squared=float(chi_squared),
                snr=float(snr)
            )

            detections.append(detection)

        return detections

    def _apply_wavelet_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply wavelet transform for particle detection.

        Args:
            image: Input image

        Returns:
            Wavelet-transformed image
        """
        if self.wavelet_type == 'mexican_hat':
            # Mexican hat (Ricker) wavelet - second derivative of Gaussian
            return self._mexican_hat_transform(image, self.sigma)
        elif self.wavelet_type == 'morlet':
            return self._morlet_transform(image, self.sigma)
        else:
            # Default to LoG (approximates Mexican hat)
            from scipy.ndimage import gaussian_laplace
            return -gaussian_laplace(image, self.sigma)

    def _mexican_hat_transform(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Mexican hat wavelet transform.

        The 2D Mexican hat is the negative Laplacian of a Gaussian. It enhances
        bright blob-like structures of width ~sigma and gives zero response to
        a flat background.
        """
        kernel = self._mexican_hat_kernel(sigma)
        return ndimage.convolve(image, kernel, mode='reflect')

    @staticmethod
    def _mexican_hat_kernel(sigma: float) -> np.ndarray:
        """
        Odd-sized, symmetric 2D Mexican hat kernel
        (1 - r^2 / (2 sigma^2)) exp(-r^2 / (2 sigma^2)), made exactly zero-sum
        (truncation) and scaled to unit absolute sum.
        """
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        half = size // 2
        y, x = np.ogrid[-half:half + 1, -half:half + 1]
        r2 = x * x + y * y
        kernel = (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        return kernel / np.abs(kernel).sum()

    def _morlet_transform(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Morlet wavelet transform."""
        # Simplified implementation using complex Morlet
        from scipy.signal import cwt, morlet2

        # Create 1D Morlet and apply separably (approximation)
        wavelet = morlet2(M=int(6*sigma), s=sigma, w=5)

        # Apply along both axes
        result = ndimage.convolve1d(image, wavelet.real, axis=0, mode='reflect')
        result = ndimage.convolve1d(result, wavelet.real, axis=1, mode='reflect')

        return np.abs(result)

    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise standard deviation using robust median estimator.

        Uses median absolute deviation (MAD) for robust noise estimation.
        """
        # Use MAD estimator
        median = np.median(image)
        mad = np.median(np.abs(image - median))

        # Convert MAD to std (for Gaussian noise)
        sigma = 1.4826 * mad

        return sigma

    def _refine_position(self, image: np.ndarray,
                        x: float, y: float,
                        window_size: int = 7) -> Tuple[Tuple[float, float], dict]:
        """
        Refine particle position using 2D Gaussian fitting.

        Args:
            image: Image array
            x, y: Initial position estimate
            window_size: Size of fitting window

        Returns:
            Tuple of ((refined_x, refined_y), fit_parameters)
        """
        # Extract region around peak
        half_window = window_size // 2

        x_int, y_int = int(round(x)), int(round(y))

        # Ensure within bounds
        y_min = max(0, y_int - half_window)
        y_max = min(image.shape[0], y_int + half_window + 1)
        x_min = max(0, x_int - half_window)
        x_max = min(image.shape[1], x_int + half_window + 1)

        region = image[y_min:y_max, x_min:x_max].copy()

        if region.size == 0:
            return (x, y), {}

        # Fit 2D Gaussian
        try:
            fit_params = self._fit_gaussian_2d(region)

            # Convert to image coordinates
            refined_x = x_min + fit_params['x0']
            refined_y = y_min + fit_params['y0']

            # Sanity check - don't move too far from initial position
            if abs(refined_x - x) > half_window or abs(refined_y - y) > half_window:
                refined_x, refined_y = x, y

            return (refined_x, refined_y), fit_params

        except Exception:
            # If fitting fails, return original position
            return (x, y), {}

    def _fit_gaussian_2d(self, region: np.ndarray) -> dict:
        """
        Fit 2D Gaussian to image region.

        Returns:
            Dictionary with fitted parameters
        """
        from scipy.optimize import curve_fit

        # Create coordinate grids
        y, x = np.mgrid[:region.shape[0], :region.shape[1]]

        # Initial parameter guess
        amplitude = region.max() - region.min()
        offset = region.min()
        x0 = region.shape[1] / 2
        y0 = region.shape[0] / 2
        sigma = self.sigma

        # Flatten for fitting
        x_data = np.vstack([x.ravel(), y.ravel()])
        z_data = region.ravel()

        # Define 2D Gaussian function
        def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
            x, y = xy
            return (amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                        ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset)

        try:
            # Fit
            popt, pcov = curve_fit(
                gaussian_2d, x_data, z_data,
                p0=[amplitude, x0, y0, sigma, sigma, offset],
                maxfev=1000
            )

            amplitude, x0, y0, sigma_x, sigma_y, offset = popt

            # Calculate chi-squared
            fitted = gaussian_2d(x_data, *popt)
            residuals = z_data - fitted
            chi_squared = np.sum(residuals**2) / (len(residuals) - len(popt))

            return {
                'amplitude': amplitude,
                'x0': x0,
                'y0': y0,
                'sigma_x': abs(sigma_x),
                'sigma_y': abs(sigma_y),
                'offset': offset,
                'chi_squared': chi_squared
            }

        except Exception as e:
            # Return default if fit fails
            return {
                'amplitude': amplitude,
                'x0': x0,
                'y0': y0,
                'sigma_x': sigma,
                'sigma_y': sigma,
                'offset': offset,
                'chi_squared': 999.0
            }
