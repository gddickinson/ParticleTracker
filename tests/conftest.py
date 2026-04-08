"""Pytest configuration and fixtures for ParticleTracker tests."""

import sys
from pathlib import Path

# Ensure the src directory is importable as a package
# We import via the top-level package from src/__init__.py
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))
