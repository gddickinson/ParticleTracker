"""Write the web demo movie (src/web/static/demo/movie.u16) from the repo's simulator.

Run: python scripts/make_demo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.web import demo  # noqa: E402


def main():
    movie, _ = demo.simulate()
    demo.MOVIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    demo.to_uint16(movie).tofile(demo.MOVIE_FILE)
    print(f"wrote {demo.MOVIE_FILE} ({demo.MOVIE_FILE.stat().st_size / 1e6:.2f} MB, shape {movie.shape})")


if __name__ == "__main__":
    main()
