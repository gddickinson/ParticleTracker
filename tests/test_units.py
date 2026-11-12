"""Regression tests: physical units reach the analysis.

Before the fix ParticleTracker.analyze_msd passed only the frame interval, so MSDs and D were in px^2 and px^2/s while
the GUI labelled them um^2; classify_motion passed no frame interval; the TIFF loader ignored the pixel size and
frame interval stored in the file and silently used 0.1 um / 0.1 s; load_project called a method that does not exist.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection, ImageStack, Track, TrackingResult  # noqa: E402
from src.core.pipeline import ParticleTracker  # noqa: E402
from src.analysis.motion import calculate_msd, fit_diffusion_model  # noqa: E402
from src.utils.io import load_image_stack  # noqa: E402


def _line_track(n=20):
    t = Track(0)
    for f in range(n):
        t.add_detection(Detection(x=10 + f, y=10, t=f))
    return t


def test_msd_in_microns_when_pixel_size_given():
    r = calculate_msd([_line_track()], max_lag=3, time_interval=0.05, pixel_size=0.2)
    assert np.allclose(r["mean_msd"], np.array([1, 4, 9]) * 0.04)
    assert r["units"] == {"length": "um", "time": "s"}
    fit = fit_diffusion_model(r, "brownian")
    assert fit["units"]["diffusion_coefficient"] == "um^2/s"


def test_msd_in_pixels_by_default():
    r = calculate_msd([_line_track()], max_lag=3)
    assert r["units"] == {"length": "px", "time": "frame"}


def test_pipeline_passes_pixel_size_and_interval():
    pt = ParticleTracker()
    pt.image_stack = ImageStack(np.zeros((20, 8, 8)), pixel_size=0.2, time_interval=0.05)
    res = TrackingResult()
    res.add_track(_line_track())
    pt.tracking_result = res
    r = pt.analyze_msd(max_lag=3)
    assert np.allclose(r["mean_msd"], np.array([1, 4, 9]) * 0.04)
    assert np.allclose(r["time_lags"], [0.05, 0.1, 0.15])


def test_tiff_scale_read_from_file(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    f = tmp_path / "movie.tif"
    tifffile.imwrite(f, np.zeros((4, 16, 16), np.uint16), imagej=True, resolution=(1 / 0.16, 1 / 0.16),
                     metadata={"unit": "micron", "finterval": 0.05, "axes": "TYX"})
    s = load_image_stack(f)
    assert np.isclose(s.pixel_size, 0.16) and np.isclose(s.time_interval, 0.05)
    assert s.metadata["scale_source"]["pixel_size"] == "file"
    s2 = load_image_stack(f, pixel_size=0.1)
    assert s2.pixel_size == 0.1 and s2.metadata["scale_source"]["pixel_size"] == "argument"


def test_tiff_without_scale_says_default(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((3, 8, 8), np.uint16), photometric="minisblack")
    s = load_image_stack(f)
    assert s.metadata["scale_source"] == {"pixel_size": "default", "time_interval": "default"}


def test_project_round_trip(tmp_path):
    pytest.importorskip("h5py")
    pt = ParticleTracker()
    pt.image_stack = ImageStack(np.zeros((20, 8, 8), np.float32), pixel_size=0.2, time_interval=0.05)
    res = TrackingResult()
    res.add_track(_line_track())
    pt.tracking_result = res
    pt.save_project(str(tmp_path / "p.h5"))
    back = ParticleTracker.load_project(str(tmp_path / "p.h5"))
    assert back.tracking_result.n_tracks == 1 and back.tracking_result.tracks[0].length == 20
    assert back.image_stack.pixel_size == 0.2
