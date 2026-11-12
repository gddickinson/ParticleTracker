"""Regression test: importing the library must not configure logging.

pipeline.py used to call logging.basicConfig(level=INFO) at import, which attached a handler to the root logger and
turned on INFO output for every program that imported ParticleTracker (and made the program's own later basicConfig
calls do nothing). Logging is now configured only by the entry points (GUI main(), example workflow).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = """
import logging
import src
from src.core.pipeline import ParticleTracker
ParticleTracker()
root = logging.getLogger()
print(len(root.handlers), logging.getLevelName(root.level))
"""


def test_import_leaves_root_logger_alone():
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0", "WARNING"]
    assert "ParticleTracker initialized" not in out.stderr


def test_gui_entry_point_configures_logging():
    src = (ROOT / "src" / "gui" / "main.py").read_text()
    main = src[src.index("def main("):]
    assert "logging.basicConfig(" in main
