#!/usr/bin/env python3
"""
Launcher script to run ParticleTracker GUI without installation.

This script adds the src directory to Python's path and launches the GUI.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Check for PyQt5
try:
    import PyQt5
    print("✓ PyQt5 found")
except ImportError:
    print("❌ PyQt5 not found. Install with:")
    print("   pip install PyQt5 pyqtgraph")
    sys.exit(1)

# Check for other key dependencies
missing_deps = []
try:
    import numpy
    print("✓ NumPy found")
except ImportError:
    missing_deps.append("numpy")

try:
    import scipy
    print("✓ SciPy found")
except ImportError:
    missing_deps.append("scipy")

try:
    import matplotlib
    print("✓ Matplotlib found")
except ImportError:
    missing_deps.append("matplotlib")

try:
    import skimage
    print("✓ scikit-image found")
except ImportError:
    missing_deps.append("scikit-image")

try:
    import tifffile
    print("✓ tifffile found")
except ImportError:
    missing_deps.append("tifffile")

if missing_deps:
    print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
    print("\nInstall with:")
    print(f"   pip install {' '.join(missing_deps)}")
    sys.exit(1)

print("\n" + "="*60)
print("Launching ParticleTracker GUI...")
print("="*60 + "\n")

# Import and run GUI
try:
    # Import main function
    import importlib.util
    spec = importlib.util.spec_from_file_location("gui_main", src_path / "gui" / "main.py")
    gui_module = importlib.util.module_from_spec(spec)
    sys.modules["gui_main"] = gui_module
    spec.loader.exec_module(gui_module)

    # Run main function
    gui_module.main()
except Exception as e:
    print(f"\n❌ Error launching GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
