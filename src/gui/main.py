"""
Main GUI application for ParticleTracker.

This provides a user-friendly interface for:
- Loading microscopy data
- Configuring detection and tracking parameters
- Running analysis
- Visualizing results
- Exporting data
"""

import sys
from pathlib import Path
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
        QDoubleSpinBox, QGroupBox, QTextEdit, QTabWidget,
        QProgressBar, QMessageBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Install with: pip install PyQt5")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from ..core.pipeline import ParticleTracker
    from ..core.data_structures import ImageStack, TrackingResult
except ImportError:
    from core.pipeline import ParticleTracker
    from core.data_structures import ImageStack, TrackingResult


class TrackingWorker(QThread):
    """Worker thread for running tracking analysis."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, tracker: ParticleTracker, config: dict):
        super().__init__()
        self.tracker = tracker
        self.config = config

    def run(self):
        """Run tracking pipeline."""
        try:
            # Detection
            self.progress.emit(25)
            detections = self.tracker.detect(**self.config['detection'])

            # Tracking
            self.progress.emit(50)
            result = self.tracker.track(**self.config['tracking'])

            # Analysis
            self.progress.emit(75)
            msd = self.tracker.analyze_msd()

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class ParticleTrackerGUI(QMainWindow):
    """Main GUI window for ParticleTracker."""

    def __init__(self):
        super().__init__()

        self.tracker = ParticleTracker()
        self.tracking_result: Optional[TrackingResult] = None

        self.init_ui()

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle('ParticleTracker - Advanced Particle Tracking')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 2)

    def create_control_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Load Data section
        load_group = QGroupBox("1. Load Data")
        load_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Image Stack")
        self.load_btn.clicked.connect(self.load_data)
        load_layout.addWidget(self.load_btn)

        self.synthetic_btn = QPushButton("Generate Synthetic Data")
        self.synthetic_btn.clicked.connect(self.generate_synthetic)
        load_layout.addWidget(self.synthetic_btn)

        self.data_label = QLabel("No data loaded")
        load_layout.addWidget(self.data_label)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # Detection parameters
        detection_group = QGroupBox("2. Detection Parameters")
        detection_layout = QVBoxLayout()

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.detection_method = QComboBox()
        self.detection_method.addItems(['wavelet', 'log', 'dog', 'gaussian'])
        method_layout.addWidget(self.detection_method)
        detection_layout.addLayout(method_layout)

        # Sigma
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.5, 10.0)
        self.sigma_spin.setValue(1.5)
        self.sigma_spin.setSingleStep(0.1)
        sigma_layout.addWidget(self.sigma_spin)
        detection_layout.addLayout(sigma_layout)

        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 10.0)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(self.threshold_spin)
        detection_layout.addLayout(threshold_layout)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # Tracking parameters
        tracking_group = QGroupBox("3. Tracking Parameters")
        tracking_layout = QVBoxLayout()

        # Method
        track_method_layout = QHBoxLayout()
        track_method_layout.addWidget(QLabel("Method:"))
        self.tracking_method = QComboBox()
        self.tracking_method.addItems(['lap', 'nearest'])
        track_method_layout.addWidget(self.tracking_method)
        tracking_layout.addLayout(track_method_layout)

        # Max displacement
        max_disp_layout = QHBoxLayout()
        max_disp_layout.addWidget(QLabel("Max Displacement:"))
        self.max_disp_spin = QDoubleSpinBox()
        self.max_disp_spin.setRange(1.0, 50.0)
        self.max_disp_spin.setValue(5.0)
        max_disp_layout.addWidget(self.max_disp_spin)
        tracking_layout.addLayout(max_disp_layout)

        # Gap closing
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Gap Closing:"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 10)
        self.gap_spin.setValue(3)
        gap_layout.addWidget(self.gap_spin)
        tracking_layout.addLayout(gap_layout)

        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)

        # Run button
        self.run_btn = QPushButton("Run Tracking")
        self.run_btn.clicked.connect(self.run_tracking)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Export section
        export_group = QGroupBox("4. Export Results")
        export_layout = QVBoxLayout()

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_results('csv'))
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)

        self.export_hdf5_btn = QPushButton("Export HDF5")
        self.export_hdf5_btn.clicked.connect(lambda: self.export_results('hdf5'))
        self.export_hdf5_btn.setEnabled(False)
        export_layout.addWidget(self.export_hdf5_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

        return panel

    def create_visualization_panel(self) -> QWidget:
        """Create right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Tabs for different views
        self.tabs = QTabWidget()

        # Image tab
        self.image_canvas = FigureCanvas(plt.Figure(figsize=(8, 8)))
        self.tabs.addTab(self.image_canvas, "Image")

        # Tracks tab
        self.tracks_canvas = FigureCanvas(plt.Figure(figsize=(8, 8)))
        self.tabs.addTab(self.tracks_canvas, "Tracks")

        # MSD tab
        self.msd_canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        self.tabs.addTab(self.msd_canvas, "MSD Analysis")

        layout.addWidget(self.tabs)

        return panel

    def load_data(self):
        """Load microscopy data from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Image Stack", "",
            "Image Files (*.tif *.tiff *.h5 *.hdf5)"
        )

        if filename:
            try:
                self.tracker.load_data(filename)
                self.log(f"Loaded: {filename}")
                self.data_label.setText(f"Loaded: {Path(filename).name}")
                self.run_btn.setEnabled(True)
                self.update_image_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def generate_synthetic(self):
        """Generate synthetic test data."""
        try:
            self.tracker.create_synthetic(n_frames=50, n_particles=15)
            self.log("Generated synthetic data: 50 frames, 15 particles")
            self.data_label.setText("Synthetic data generated")
            self.run_btn.setEnabled(True)
            self.update_image_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate data: {str(e)}")

    def run_tracking(self):
        """Run detection and tracking."""
        self.log("Starting tracking...")
        self.run_btn.setEnabled(False)

        # Collect parameters
        config = {
            'detection': {
                'method': self.detection_method.currentText(),
                'sigma': self.sigma_spin.value(),
                'threshold': self.threshold_spin.value()
            },
            'tracking': {
                'method': self.tracking_method.currentText(),
                'max_displacement': self.max_disp_spin.value(),
                'gap_closing': self.gap_spin.value()
            }
        }

        # Run in worker thread
        self.worker = TrackingWorker(self.tracker, config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.error.connect(self.on_tracking_error)
        self.worker.start()

    def on_tracking_finished(self, result):
        """Handle completion of tracking."""
        self.tracking_result = result
        self.log(f"Tracking complete: {result.n_tracks} tracks found")
        self.run_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_hdf5_btn.setEnabled(True)

        # Update visualizations
        self.update_tracks_display()
        self.update_msd_display()

    def on_tracking_error(self, error_msg):
        """Handle tracking error."""
        QMessageBox.critical(self, "Tracking Error", error_msg)
        self.log(f"Error: {error_msg}")
        self.run_btn.setEnabled(True)

    def update_image_display(self):
        """Update image visualization."""
        if self.tracker.image_stack is None:
            return

        # Show maximum projection
        img = np.max(self.tracker.image_stack.data, axis=0)

        ax = self.image_canvas.figure.clear()
        ax = self.image_canvas.figure.add_subplot(111)
        ax.imshow(img, cmap='gray')
        ax.set_title('Maximum Intensity Projection')
        self.image_canvas.draw()

    def update_tracks_display(self):
        """Draw the tracks over the maximum-intensity projection."""
        ax = self.tracks_canvas.figure.clear()
        ax = self.tracks_canvas.figure.add_subplot(111)
        if self.tracker.image_stack is not None:
            ax.imshow(np.max(self.tracker.image_stack.data, axis=0), cmap='gray')
        if self.tracking_result is not None:
            for track in self.tracking_result.tracks:
                pos = track.positions
                ax.plot(pos[:, 0], pos[:, 1], '-', lw=1)
            ax.set_title(f'{self.tracking_result.n_tracks} tracks (pixels)')
        self.tracks_canvas.draw()

    def update_msd_display(self):
        """Update MSD visualization."""
        if self.tracking_result is None:
            return

        # Calculate and plot MSD
        msd_results = self.tracker.analyze_msd()

        ax = self.msd_canvas.figure.clear()
        ax = self.msd_canvas.figure.add_subplot(111)

        ax.errorbar(msd_results['time_lags'], msd_results['mean_msd'],
                   yerr=msd_results['std_msd'], fmt='o-')
        units = msd_results.get('units', {'length': 'px', 'time': 'frame'})
        ax.set_xlabel(f"Time lag ({units['time']})")
        ax.set_ylabel(f"MSD ({units['length']}²)")
        ax.set_title('Mean Squared Displacement')
        ax.grid(True, alpha=0.3)

        self.msd_canvas.draw()

    def export_results(self, format: str):
        """Export tracking results."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "",
            f"{format.upper()} Files (*.{format})"
        )

        if filename:
            try:
                self.tracker.export(filename, format=format)
                self.log(f"Exported results to {filename}")
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)


def main():
    """Main entry point for GUI application."""
    if not PYQT_AVAILABLE:
        print("PyQt5 is required for the GUI. Install with:")
        print("  pip install PyQt5 pyqtgraph")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = ParticleTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
