"""Web interface for ParticleTracker (FastAPI app + static front end).

- app.py       routes under /api, STATIC_ENDPOINTS for the static export
- demo.py      the simulated demo movie and its precomputed analyses
- analysis.py  runs the repo's pipeline steps and builds compact JSON
- scoring.py   detection / tracking scores against the simulated ground truth
- upload.py    the pipeline on an uploaded TIFF stack (local app only)
- jsonify.py   numpy -> JSON conversion
- static/      front end (ES modules, JS ports of detection, linking and analysis)
"""
