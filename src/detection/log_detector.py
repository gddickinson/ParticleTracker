"""
Laplacian of Gaussian (LoG) and Difference of Gaussians (DoG) detectors.

These are fast, classical blob detection methods widely used in microscopy.
"""

import numpy as np
from typing import List
from scipy import ndimage
from skimage.feature import blob_log, blob_dog

try:
    from .detector_factory import BaseDetector
    from ..core.data_structures import Detection
except ImportError:
    from detector_factory import BaseDetector
    from core.data_structures import Detection


class LoGDetector(BaseDetector):
    """
    Laplacian of Gaussian detector.

    Fast blob detection using LoG filter. Excellent for well-separated
    particles with consistent size.

    Config parameters:
        min_sigma: Minimum sigma for LoG (default: 1.0)
        max_sigma: Maximum sigma for LoG (default: 3.0)
        num_sigma: Number of scales (default: 10)
        threshold: Detection threshold (default: 0.1)
        overlap: Maximum overlap between blobs (default: 0.5)
    """

    def __init__(self, config: dict):
        """Initialize LoG detector."""
        super().__init__(config)

        self.min_sigma = config.get('min_sigma', 1.0)
        self.max_sigma = config.get('max_sigma', 3.0)
        self.num_sigma = config.get('num_sigma', 10)
        self.threshold = config.get('threshold', 0.1)
        self.overlap = config.get('overlap', 0.5)

    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect particles using LoG.

        Args:
            image: 2D image array
            frame_index: Frame time index

        Returns:
            List of Detection objects
        """
        # Ensure float type and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize image
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

        # Detect blobs
        blobs = blob_log(
            image_norm,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        )

        # Convert to Detection objects
        detections = []
        noise_std = self._estimate_noise(image)

        for blob in blobs:
            y, x, sigma = blob

            # Get intensity at this location
            intensity = float(image[int(y), int(x)])
            snr = intensity / noise_std if noise_std > 0 else 0

            # Estimate uncertainty
            n_photons = max(intensity, 1)
            uncertainty = sigma / np.sqrt(n_photons)

            detection = Detection(
                x=float(x),
                y=float(y),
                t=frame_index,
                intensity=intensity,
                sigma_x=float(sigma),
                sigma_y=float(sigma),
                uncertainty=float(uncertainty),
                snr=float(snr)
            )

            detections.append(detection)

        return detections

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise using MAD."""
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        return 1.4826 * mad


class DoGDetector(BaseDetector):
    """
    Difference of Gaussians detector.

    Even faster than LoG, good for quick detection tasks.

    Config parameters:
        min_sigma: Minimum sigma for DoG (default: 1.0)
        max_sigma: Maximum sigma for DoG (default: 3.0)
        sigma_ratio: Ratio between sigmas (default: 1.6)
        threshold: Detection threshold (default: 0.1)
        overlap: Maximum overlap between blobs (default: 0.5)
    """

    def __init__(self, config: dict):
        """Initialize DoG detector."""
        super().__init__(config)

        self.min_sigma = config.get('min_sigma', 1.0)
        self.max_sigma = config.get('max_sigma', 3.0)
        self.sigma_ratio = config.get('sigma_ratio', 1.6)
        self.threshold = config.get('threshold', 0.1)
        self.overlap = config.get('overlap', 0.5)

    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect particles using DoG.

        Args:
            image: 2D image array
            frame_index: Frame time index

        Returns:
            List of Detection objects
        """
        # Ensure float type and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

        # Detect blobs
        blobs = blob_dog(
            image_norm,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            sigma_ratio=self.sigma_ratio,
            threshold=self.threshold,
            overlap=self.overlap
        )

        # Convert to Detection objects
        detections = []
        noise_std = self._estimate_noise(image)

        for blob in blobs:
            y, x, sigma = blob

            intensity = float(image[int(y), int(x)])
            snr = intensity / noise_std if noise_std > 0 else 0

            n_photons = max(intensity, 1)
            uncertainty = sigma / np.sqrt(n_photons)

            detection = Detection(
                x=float(x),
                y=float(y),
                t=frame_index,
                intensity=intensity,
                sigma_x=float(sigma),
                sigma_y=float(sigma),
                uncertainty=float(uncertainty),
                snr=float(snr)
            )

            detections.append(detection)

        return detections

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise using MAD."""
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        return 1.4826 * mad


class GaussianFitDetector(BaseDetector):
    """
    Direct Gaussian fitting detector.

    Performs local maximum finding followed by Gaussian fitting.
    Most accurate but slower than LoG/DoG.

    Config parameters:
        sigma: Expected PSF sigma (default: 1.5)
        threshold: Detection threshold (default: 3.0)
        min_distance: Minimum distance between peaks (default: 5)
        fit_window: Window size for fitting (default: 9)
    """

    def __init__(self, config: dict):
        """Initialize Gaussian fit detector."""
        super().__init__(config)

        self.sigma = config.get('sigma', 1.5)
        self.threshold = config.get('threshold', 3.0)
        self.min_distance = config.get('min_distance', 5)
        self.fit_window = config.get('fit_window', 9)

    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect particles using Gaussian fitting.

        Args:
            image: 2D image array
            frame_index: Frame time index

        Returns:
            List of Detection objects
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Estimate noise
        noise_std = self._estimate_noise(image)
        threshold_value = noise_std * self.threshold

        # Find local maxima
        peaks = self._find_peaks(image, threshold_value, self.min_distance)

        # Fit Gaussian to each peak
        detections = []

        for peak in peaks:
            y, x = peak

            # Fit Gaussian
            fit_result = self._fit_gaussian(image, x, y)

            if fit_result is not None:
                detection = Detection(
                    x=float(fit_result['x0']),
                    y=float(fit_result['y0']),
                    t=frame_index,
                    intensity=float(fit_result['amplitude']),
                    sigma_x=float(fit_result['sigma_x']),
                    sigma_y=float(fit_result['sigma_y']),
                    uncertainty=float(fit_result.get('uncertainty', 10.0)),
                    chi_squared=float(fit_result.get('chi_squared', 0.0)),
                    snr=float(fit_result['amplitude'] / noise_std if noise_std > 0 else 0)
                )
                detections.append(detection)

        return detections

    def _fit_gaussian(self, image: np.ndarray, x: float, y: float) -> dict:
        """Fit 2D Gaussian to region around (x, y)."""
        from scipy.optimize import curve_fit

        half_window = self.fit_window // 2

        x_int, y_int = int(round(x)), int(round(y))

        # Extract region
        y_min = max(0, y_int - half_window)
        y_max = min(image.shape[0], y_int + half_window + 1)
        x_min = max(0, x_int - half_window)
        x_max = min(image.shape[1], x_int + half_window + 1)

        region = image[y_min:y_max, x_min:x_max].copy()

        if region.size < 9:  # Too small to fit
            return None

        # Create coordinates
        y_coords, x_coords = np.mgrid[:region.shape[0], :region.shape[1]]

        # Initial guess
        amplitude = region.max() - region.min()
        offset = region.min()
        x0 = region.shape[1] / 2
        y0 = region.shape[0] / 2

        # Flatten
        x_data = np.vstack([x_coords.ravel(), y_coords.ravel()])
        z_data = region.ravel()

        # Gaussian function
        def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
            x, y = xy
            return (amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                        ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset)

        try:
            popt, pcov = curve_fit(
                gaussian_2d, x_data, z_data,
                p0=[amplitude, x0, y0, self.sigma, self.sigma, offset],
                maxfev=500
            )

            amplitude, x0_fit, y0_fit, sigma_x, sigma_y, offset = popt

            # Convert back to image coordinates
            x0_img = x_min + x0_fit
            y0_img = y_min + y0_fit

            # Calculate uncertainty
            n_photons = max(amplitude, 1)
            uncertainty = max(sigma_x, sigma_y) / np.sqrt(n_photons)

            # Calculate chi-squared
            fitted = gaussian_2d(x_data, *popt)
            residuals = z_data - fitted
            chi_squared = np.sum(residuals**2) / max(len(residuals) - len(popt), 1)

            return {
                'x0': x0_img,
                'y0': y0_img,
                'amplitude': amplitude,
                'sigma_x': abs(sigma_x),
                'sigma_y': abs(sigma_y),
                'offset': offset,
                'uncertainty': uncertainty,
                'chi_squared': chi_squared
            }

        except Exception:
            return None

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise using MAD."""
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        return 1.4826 * mad
