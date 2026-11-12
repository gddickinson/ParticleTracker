"""Regression test: the pipeline works when imported the way run_gui.py and the 'particletracker' console script
import it (src/ on sys.path, top-level package 'core').

Before the fix ParticleTracker.detect/track/analyze_msd used 'from ..detection ...' style imports, which fail with
'attempted relative import beyond top-level package' there, so the GUI could load data but every run failed.
Runs in a subprocess so the two import styles do not mix in one interpreter.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

CODE = """
import sys
sys.path.insert(0, %r)
import numpy as np
from core.pipeline import ParticleTracker
pt = ParticleTracker()
np.random.seed(0)
pt.create_synthetic(n_frames=12, n_particles=4, frame_shape=(64, 64))
pt.detect(method='wavelet')
pt.track(method='lap', min_track_length=3)
r = pt.analyze_msd(max_lag=3)
pt.classify_motion()
print('OK', r['units']['length'])
"""


def test_pipeline_runs_with_src_on_path():
    r = subprocess.run([sys.executable, "-c", CODE % str(SRC)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK um" in r.stdout
