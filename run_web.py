"""Start the ParticleTracker web app on http://127.0.0.1:8211 (local only).

Run: python run_web.py   (needs requirements-web.txt)
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8211)
