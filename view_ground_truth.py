"""Launcher for the fish EOD/video ground-truth viewer.

Usage:
    python view_ground_truth.py VIDEO_PATH CSV_PATH

VIDEO_PATH   the tank recording (.avi/.mp4)
CSV_PATH     the matching identity-labelled Data_4fish_*_no_ref_per.csv
             (must have a ``discharging_fish`` column with A/B/C/D labels)
"""

import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

from gui.ground_truth_viewer import GroundTruthViewer


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    video_path, csv_path = sys.argv[1], sys.argv[2]

    for path, label in ((video_path, "Video"), (csv_path, "CSV")):
        if not os.path.exists(path):
            print(f"{label} file not found: {path}")
            sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Ground Truth Viewer")
    app.setStyle("Fusion")
    pg.setConfigOption("background", "#0d0d2a")
    pg.setConfigOption("foreground", "#e0e0e0")
    pg.setConfigOption("antialias", True)
    pg.setConfigOption("imageAxisOrder", "row-major")

    win = GroundTruthViewer(video_path, csv_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
