"""
ParticleTracker: Advanced Particle Detection and Tracking for Fluorescence Microscopy

This package provides comprehensive tools for detecting, localizing, and tracking
particles in fluorescence microscopy images, with support for conventional, TIRF,
and super-resolution (STORM/PALM) microscopy.
"""

__version__ = "1.0.0"
__author__ = "ParticleTracker Development Team"
__license__ = "MIT"

from .core.pipeline import ParticleTracker
from .core.data_structures import (
    ImageStack,
    Detection,
    Track,
    TrackingResult
)

__all__ = [
    'ParticleTracker',
    'ImageStack',
    'Detection',
    'Track',
    'TrackingResult',
]
