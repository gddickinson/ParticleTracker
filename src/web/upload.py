"""The pipeline on an uploaded image stack (local web app only).

The stack is read with the repo's own loader (src/utils/io.load_image_stack), which takes the pixel size and frame
interval from the file's ImageJ/OME metadata when present; values typed in the form override them. The response says
where each scale value came from.
"""
import json
import tempfile
from pathlib import Path

from ..utils.io import load_image_stack
from . import analysis as A

MAX_FRAMES = 2000
SUFFIXES = (".tif", ".tiff", ".npy")


class RunError(ValueError):
    pass


def run_upload(filename, data, parameters_json):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUFFIXES:
        raise RunError("upload a .tif/.tiff stack (or a .npy array)")
    try:
        p = json.loads(parameters_json or "{}")
    except json.JSONDecodeError:
        raise RunError("parameters must be JSON")
    scale = {k: float(p[k]) for k in ("pixel_size", "time_interval") if p.get(k) not in (None, "")}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / ("upload" + suffix)
        f.write_bytes(data)
        try:
            stack = load_image_stack(f, **scale)
        except Exception as e:  # unreadable file
            raise RunError(f"could not read the stack: {e}")
    if stack.data.ndim != 3:
        raise RunError(f"expected a single-channel time series (T, Y, X); got shape {list(stack.data.shape)}")
    if stack.n_frames > MAX_FRAMES:
        raise RunError(f"{stack.n_frames} frames; the web app accepts up to {MAX_FRAMES}")
    source = stack.metadata.get("scale_source", {k: "argument" if k in scale else "default"
                                                 for k in ("pixel_size", "time_interval")})
    pt = A.make_tracker(stack.data, stack.pixel_size, stack.time_interval)
    detector = p.get("detector", "wavelet")
    if detector not in ("wavelet", "log", "dog", "gaussian"):
        raise RunError("unknown detector")
    dp = {k: float(v) for k, v in (p.get("det_params") or {}).items()}
    if detector == "wavelet":
        dp = {**{k: v for k, v in A.defaults()["detection"].items() if k != "method"}, **dp}
    if "min_distance" in dp:
        dp["min_distance"] = int(dp["min_distance"])
    A.detect(pt, detector, dp)
    lp = {k: float(v) for k, v in (p.get("link_params") or {}).items()}
    for k in ("gap_closing", "min_track_length"):
        if k in lp:
            lp[k] = int(lp[k])
    out = A.analyse(pt, lp)
    out.update({"file": Path(filename).name, "shape": list(stack.data.shape), "detector": detector,
                "det_params": dp, "scale_source": source, "computed_in": "python"})
    return A.finish(out, 6)
