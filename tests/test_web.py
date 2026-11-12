"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- ParticleTracker-specific tests -------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402


def test_demo_movie_matches_simulator():
    from src.web import demo
    movie, _ = demo.simulate()
    assert np.max(np.abs(demo.to_uint16(movie).astype(int) - demo.movie().astype(int))) <= 1


@pytest.mark.skipif(bool(os.environ.get("CI")), reason=(
    "regenerating the fixture runs detection, Gaussian subpixel fitting and LAP linking; libm differences on the Linux "
    "CI runner move fitted positions beyond float tolerance (the node JS-vs-fixture tests still run)"))
def test_engine_fixture_current():
    """tests/fixtures/engine_cases.json still matches the Python code (numbers with a tolerance: libm differs)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    new = json.loads(json.dumps(make_engine_fixture.build()))
    old = json.loads((ROOT / "tests" / "fixtures" / "engine_cases.json").read_text())

    def same(a, b, path=""):
        if isinstance(b, dict):
            assert isinstance(a, dict) and a.keys() == b.keys(), path
            for k in b:
                same(a[k], b[k], f"{path}.{k}")
        elif isinstance(b, list):
            assert isinstance(a, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        elif isinstance(b, float) and not isinstance(b, bool):
            assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
        else:
            assert a == b, path
    same(new, old)


def test_pipeline_demo_tracks_every_particle():
    r = client.get("/api/demo/pipeline").json()
    assert r["tracking_scores"]["recovered"] == r["tracking_scores"]["n_true"] == 12
    assert r["detection_scores"]["recall"] > 0.95
    assert r["units"]["pixel_size_um"] == 0.1 and r["ensemble"]["units"] == {"length": "um", "time": "s"}


def test_demo_run_post():
    r = client.post("/api/demo/run", json={"detector": "wavelet", "det_params": {"subpixel": False, "threshold": 4},
                                           "link_params": {"max_displacement": 3, "gap_closing": 2}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["det_params"]["subpixel"] is False and j["link_params"]["gap_closing"] == 2 and j["tracks"]


def test_demo_run_rejects_bad_input():
    assert client.post("/api/demo/run", json={"detector": "nope"}).status_code == 422
    assert client.post("/api/demo/run", json={"det_params": {"sigma": "x"}}).status_code == 422


def test_upload_tiff_reads_scale(tmp_path):
    import tifffile
    from src.web import demo
    f = tmp_path / "m.tif"
    tifffile.imwrite(f, demo.movie()[:12], imagej=True, resolution=(1 / 0.16, 1 / 0.16),
                     metadata={"unit": "micron", "finterval": 0.05, "axes": "TYX"})
    params = {"link_params": {"min_track_length": 5}}
    r = client.post("/api/run", files={"file": ("m.tif", f.read_bytes(), "image/tiff")},
                    data={"parameters": json.dumps(params)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["scale_source"] == {"pixel_size": "file", "time_interval": "file"}
    assert j["units"]["pixel_size_um"] == pytest.approx(0.16) and len(j["tracks"]) >= 10


def test_upload_rejects_other_files():
    r = client.post("/api/run", files={"file": ("a.png", b"x", "image/png")}, data={"parameters": "{}"})
    assert r.status_code == 422
