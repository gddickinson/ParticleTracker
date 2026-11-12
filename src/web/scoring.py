"""Scores against the simulated ground truth (web layer; mirrored exactly by static/score_engine.js).

Positions in pixels, pixel-centre convention (as the detectors and the simulator).

- detection_scores: per frame, detections and true positions are paired greedily by increasing distance (each used
  once) up to `tol` px; TP / FP / FN, precision, recall, F1 and the RMS localisation error of the pairs.
- tracking_scores: every point of a predicted track is assigned to the nearest true particle within `tol` px in that
  frame (or none). A track follows the particle most of its points are assigned to (its purity is that fraction). A
  particle's coverage is the largest number of frames any single track follows it, divided by the frames it exists.
"""
import math

DET_TOL = 2.0
TRACK_TOL = 2.0
RECOVERED = 0.8


def _pairs(dets, truth, tol):
    cand = []
    for i, (dx, dy) in enumerate(dets):
        for j, (tx, ty) in enumerate(truth):
            d = math.hypot(dx - tx, dy - ty)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    used_i, used_j, out = set(), set(), []
    for d, i, j in cand:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append(d)
    return out


def detection_scores(particles, truth, tol=DET_TOL):
    """particles: per frame [[x, y, ...], ...]; truth: [{x: [...], y: [...]}] (one value per frame)."""
    tp = n_det = n_true = 0
    sq = 0.0
    for t, frame in enumerate(particles):
        dets = [(p[0], p[1]) for p in frame]
        tru = [(g["x"][t], g["y"][t]) for g in truth]
        d = _pairs(dets, tru, tol)
        tp += len(d)
        sq += sum(v * v for v in d)
        n_det += len(dets)
        n_true += len(tru)
    precision = tp / n_det if n_det else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tolerance_px": tol, "n_detected": n_det, "n_true": n_true, "true_positives": tp,
            "false_positives": n_det - tp, "false_negatives": n_true - tp, "precision": precision,
            "recall": recall, "f1": f1, "rmse_px": math.sqrt(sq / tp) if tp else None}


def _nearest_truth(x, y, t, truth, tol):
    best, bd = None, tol
    for k, g in enumerate(truth):
        if t < len(g["x"]):
            d = math.hypot(x - g["x"][t], y - g["y"][t])
            if d <= bd:
                best, bd = k, d
    return best


def tracking_scores(tracks, truth, tol=TRACK_TOL):
    """tracks: [{track_id, frames, x, y}]; returns per-track and per-particle matches and summary numbers."""
    n_true = len(truth)
    follow = [[0] * len(tracks) for _ in range(n_true)]   # frames track j sits on particle k
    per_track = []
    for j, tr in enumerate(tracks):
        counts = {}
        for f, x, y in zip(tr["frames"], tr["x"], tr["y"]):
            k = _nearest_truth(x, y, f, truth, tol)
            if k is not None:
                counts[k] = counts.get(k, 0) + 1
                follow[k][j] += 1
        if counts:
            k = min(counts, key=lambda c: (-counts[c], c))
            per_track.append({"track_id": tr["track_id"], "particle": k, "purity": counts[k] / len(tr["frames"])})
        else:
            per_track.append({"track_id": tr["track_id"], "particle": None, "purity": 0.0})
    per_particle = []
    for k, g in enumerate(truth):
        n = len(g["x"])
        best = max(follow[k]) if tracks else 0
        pieces = sum(1 for p in per_track if p["particle"] == k)
        per_particle.append({"particle": k, "coverage": best / n if n else 0.0, "tracks": pieces})
    cov = [p["coverage"] for p in per_particle]
    return {"tolerance_px": tol, "n_tracks": len(tracks), "n_true": n_true,
            "recovered": sum(1 for c in cov if c >= RECOVERED), "recovered_threshold": RECOVERED,
            "mean_coverage": sum(cov) / n_true if n_true else 0.0,
            "unmatched_tracks": sum(1 for p in per_track if p["particle"] is None),
            "per_track": per_track, "per_particle": per_particle}
