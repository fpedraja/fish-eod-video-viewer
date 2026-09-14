"""Ground-truth viewer: video (with pose overlay) synced to the per-fish EOD
pulse timeline, for visually checking fish-ID <-> EOD assignment.

Loads the *_no_ref_per.csv files from Assigned_ground_truth_SLEAP, which are
EOD-event indexed (one row per detected pulse) and already carry the
ground-truth identity of the discharging fish (``discharging_fish``: A-D),
alongside each fish's pose (held constant across all pulses of the same
video frame). Frames with no pulse have no row at all, so pose for those
frames is taken from the nearest frame that does have one.
"""

from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

BODYPARTS = ["mouth", "head", "middle", "tail"]
FISH_LETTERS = ["A", "B", "C", "D"]
FISH_COLORS = ["#e06c75", "#98c379", "#61afef", "#e5c07b"]
_SCORE_THRESH = 0.2


def _label_color(text: str) -> QColor:
    """Deterministic color per label text (stable across sessions/reloads)."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 360
    return QColor.fromHsv(h, 170, 235)


class _AnnotateViewBox(pg.ViewBox):
    """ViewBox that, when annotate_mode is on, turns a left-drag into a
    time-range selection (instead of panning) and emits it on release."""

    dragFinished = pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotate_mode = False
        self._temp_region = None

    def mouseDragEvent(self, ev, axis=None):
        if not self.annotate_mode:
            super().mouseDragEvent(ev, axis=axis)
            return
        ev.accept()
        p0 = self.mapSceneToView(ev.buttonDownScenePos())
        p1 = self.mapSceneToView(ev.scenePos())
        x0, x1 = sorted((p0.x(), p1.x()))
        if self._temp_region is None:
            self._temp_region = pg.LinearRegionItem(
                brush=pg.mkBrush(255, 255, 255, 70), movable=False)
            self.addItem(self._temp_region)
        self._temp_region.setRegion((x0, x1))
        if ev.isFinish():
            self.removeItem(self._temp_region)
            self._temp_region = None
            self.dragFinished.emit(x0, x1)


def _pose_columns() -> list[str]:
    cols = []
    for letter in FISH_LETTERS:
        for bp in BODYPARTS:
            cols += [f"{bp}_x_{letter}", f"{bp}_y_{letter}", f"{bp}_score_{letter}"]
    return cols


class GroundTruthData:
    """Loads the pose + timing + identity columns of an assigned Data_4fish
    CSV (skips the 250 raw-waveform-snippet columns, which this viewer
    doesn't need)."""

    def __init__(self, csv_path: str):
        pose_cols = _pose_columns()
        usecols = ["Frame", "RelativeTime", "discharging_fish"] + pose_cols
        dtype = {"Frame": "int32", "RelativeTime": "float64",
                 "discharging_fish": "category"}
        dtype.update({c: "float32" for c in pose_cols})

        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype)
        df = df.rename(columns={"RelativeTime": "Time"})

        self.event_times = df["Time"].to_numpy()
        self.event_frames = df["Frame"].to_numpy()
        self.event_fish = (
            df["discharging_fish"]
            .astype(pd.CategoricalDtype(categories=FISH_LETTERS))
            .cat.codes.to_numpy()
        )  # -1 for any row whose label isn't in FISH_LETTERS

        self.fish_event_times = {
            letter: np.sort(self.event_times[self.event_fish == i])
            for i, letter in enumerate(FISH_LETTERS)
        }

        time_by_frame = df.groupby("Frame")["Time"].median()
        self._known_frames = time_by_frame.index.to_numpy()
        self._known_times = time_by_frame.to_numpy()

        pose_df = df.drop_duplicates(subset="Frame", keep="first").sort_values("Frame")
        self._pose_frames = pose_df["Frame"].to_numpy()
        pose_arr = pose_df[pose_cols].to_numpy(dtype=np.float32)
        self._pose = pose_arr.reshape(len(pose_df), len(FISH_LETTERS), len(BODYPARTS), 3)

    def _nearest_index(self, frames: np.ndarray, query_frame: int) -> int:
        i = np.searchsorted(frames, query_frame)
        if i <= 0:
            return 0
        if i >= len(frames):
            return len(frames) - 1
        before, after = frames[i - 1], frames[i]
        return i - 1 if (query_frame - before) <= (after - query_frame) else i

    def pose_for_video_frame(self, video_frame_idx: int) -> np.ndarray:
        """Returns (n_fish, n_bodyparts, 3) array of x, y, score."""
        query_frame = video_frame_idx + 1  # CSV Frame is 1-indexed
        j = self._nearest_index(self._pose_frames, query_frame)
        return self._pose[j]

    def time_for_video_frame(self, video_frame_idx: int) -> float:
        query_frame = video_frame_idx + 1
        return float(np.interp(query_frame, self._known_frames, self._known_times))

    def nearest_video_frame_for_time(self, t: float) -> int:
        i = np.searchsorted(self._known_times, t)
        if i <= 0:
            j = 0
        elif i >= len(self._known_times):
            j = len(self._known_times) - 1
        else:
            before, after = self._known_times[i - 1], self._known_times[i]
            j = i - 1 if (t - before) <= (after - t) else i
        return int(self._known_frames[j]) - 1


class VideoPanel(QWidget):
    """Video frame display with per-fish keypoint/skeleton overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        layout.addWidget(self.plot)

        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        self._skeleton_curves = []
        self._joint_scatters = []
        for color in FISH_COLORS:
            curve = pg.PlotCurveItem(pen=pg.mkPen(color, width=2))
            self.plot.addItem(curve)
            self._skeleton_curves.append(curve)

            scatter = pg.ScatterPlotItem(
                size=8, brush=pg.mkBrush(color), pen=pg.mkPen("k", width=0.5))
            self.plot.addItem(scatter)
            self._joint_scatters.append(scatter)

    def set_frame(self, frame_rgb: np.ndarray, pose: np.ndarray) -> None:
        self.img_item.setImage(frame_rgb, autoLevels=False, levels=(0, 255))
        for fid in range(len(FISH_LETTERS)):
            xs, ys, scores = pose[fid, :, 0], pose[fid, :, 1], pose[fid, :, 2]
            valid = scores > _SCORE_THRESH
            if valid.sum() >= 2:
                self._skeleton_curves[fid].setData(xs[valid], ys[valid])
            else:
                self._skeleton_curves[fid].setData([], [])
            self._joint_scatters[fid].setData(xs[valid], ys[valid])


class EODPanel(QWidget):
    """Per-fish EOD pulse timeline: scrolling window synced to the video's
    current time, plus a full-session overview strip to jump around."""

    frameJumpRequested = pyqtSignal(int)
    annotationsChanged = pyqtSignal()

    def __init__(self, data: GroundTruthData, parent=None):
        super().__init__(parent)
        self._data = data
        self._mode = "dots"
        self._current_time = 0.0
        self.annotations: list[dict] = []
        self._overview_ann_items: list = []
        self._build_ui()
        self._apply_mode()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.addWidget(QLabel("EOD view:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Event times (dots)", "Instantaneous frequency"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)

        top.addWidget(QLabel("Window (s):"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.5, 300.0)
        self.window_spin.setValue(10.0)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.valueChanged.connect(lambda *_: self.set_time(self._current_time))
        top.addWidget(self.window_spin)

        top.addSpacing(20)
        top.addWidget(QLabel("Fish:"))
        for letter, color in zip(FISH_LETTERS, FISH_COLORS):
            tag = QLabel(f"● {letter}")
            tag.setStyleSheet(f"color: {color}; font-weight: bold;")
            top.addWidget(tag)

        top.addSpacing(20)
        self.add_label_btn = QPushButton("🏷 Add Label")
        self.add_label_btn.setCheckable(True)
        self.add_label_btn.setToolTip(
            "When on, drag a range on the timeline below to tag it "
            "with a behavioral label.")
        self.add_label_btn.toggled.connect(self._on_toggle_add_label)
        top.addWidget(self.add_label_btn)

        top.addStretch()
        layout.addLayout(top)

        self._yrange_row = QWidget()
        yrow = QHBoxLayout(self._yrange_row)
        yrow.setContentsMargins(0, 0, 0, 0)
        self.lock_y_check = QCheckBox("Lock Y range (Hz)")
        self.lock_y_check.setStyleSheet("color:#abb2bf;")
        self.lock_y_check.stateChanged.connect(self._update_freq_yrange)
        yrow.addWidget(self.lock_y_check)
        yrow.addWidget(QLabel("Min:"))
        self.ymin_spin = QDoubleSpinBox()
        self.ymax_spin = QDoubleSpinBox()
        for sp in (self.ymin_spin, self.ymax_spin):
            sp.setRange(0.0, 100000.0)
            sp.setSingleStep(1.0)
            sp.setDecimals(1)
            sp.valueChanged.connect(self._update_freq_yrange)
        self.ymax_spin.setValue(100.0)
        yrow.addWidget(self.ymin_spin)
        yrow.addWidget(QLabel("Max:"))
        yrow.addWidget(self.ymax_spin)

        yrow.addSpacing(20)
        yrow.addWidget(QLabel("Max plausible (Hz):"))
        self.max_plausible_spin = QDoubleSpinBox()
        self.max_plausible_spin.setRange(1.0, 2000.0)
        self.max_plausible_spin.setSingleStep(5.0)
        self.max_plausible_spin.setDecimals(0)
        self.max_plausible_spin.setValue(170.0)
        self.max_plausible_spin.setToolTip(
            "Pulses implying a higher single-fish discharge rate than this "
            "are flagged (✕) as likely EOD→fish assignment errors.")
        self.max_plausible_spin.valueChanged.connect(self._update_suspect_markers)
        yrow.addWidget(self.max_plausible_spin)

        yrow.addStretch()
        self._yrange_row.setVisible(False)
        layout.addWidget(self._yrange_row)

        self._ann_vb = _AnnotateViewBox()
        self.main_plot = pg.PlotWidget(viewBox=self._ann_vb)
        self.main_plot.showGrid(x=True, y=True, alpha=0.2)
        self.main_plot.setLabel("bottom", "Time (s)")
        self.main_plot.setMenuEnabled(False)
        self._ann_vb.dragFinished.connect(self._on_label_drag)
        layout.addWidget(self.main_plot, stretch=3)

        self.ann_track = pg.PlotWidget()
        self.ann_track.setMaximumHeight(46)
        self.ann_track.setMenuEnabled(False)
        self.ann_track.hideAxis("bottom")
        self.ann_track.getAxis("left").setStyle(showValues=False)
        self.ann_track.setXLink(self.main_plot)
        self.ann_track.setYRange(-1, 1)
        layout.addWidget(self.ann_track, stretch=0)

        self.overview_plot = pg.PlotWidget()
        self.overview_plot.setMaximumHeight(70)
        self.overview_plot.getAxis("left").setStyle(showValues=False)
        self.overview_plot.setLabel("bottom", "Session time (s)")
        self.overview_plot.setMenuEnabled(False)
        layout.addWidget(self.overview_plot, stretch=1)

        self._marker = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#ff5555", width=2))
        self._threshold_line = pg.InfiniteLine(
            pos=170, angle=0,
            pen=pg.mkPen("#888888", width=1, style=Qt.DashLine))
        self._suspect_scatter = pg.ScatterPlotItem(
            symbol="x", size=10, pen=pg.mkPen("#ffffff", width=2), brush=None)
        self._freq_cache: dict[str, tuple] = {}

        self._dot_items = []
        self._freq_items = []
        for letter, color in zip(FISH_LETTERS, FISH_COLORS):
            self._dot_items.append(
                pg.ScatterPlotItem(size=6, brush=pg.mkBrush(color), pen=None))
            self._freq_items.append(pg.PlotDataItem(pen=pg.mkPen(color, width=1)))

        all_t = self._data.event_times
        if len(all_t):
            self.overview_plot.setXRange(all_t.min(), all_t.max(), padding=0.02)
        self.overview_plot.setYRange(-1, len(FISH_LETTERS))
        for i, (letter, color) in enumerate(zip(FISH_LETTERS, FISH_COLORS)):
            t = self._data.fish_event_times[letter]
            sc = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(color), pen=None)
            sc.setData(t, np.full(len(t), i, dtype=float))
            self.overview_plot.addItem(sc)

        self._region = pg.LinearRegionItem(brush=pg.mkBrush(97, 175, 239, 60))
        self.overview_plot.addItem(self._region)
        self._region.sigRegionChanged.connect(self._on_region_dragged)

    # ------------------------------------------------------------------
    # Behavioral labels
    # ------------------------------------------------------------------

    def _on_toggle_add_label(self, checked: bool) -> None:
        self._ann_vb.annotate_mode = checked
        self.main_plot.viewport().setCursor(
            Qt.CrossCursor if checked else Qt.ArrowCursor)

    def _on_label_drag(self, x0: float, x1: float) -> None:
        if abs(x1 - x0) < 0.02:
            return  # accidental click, not a real drag
        existing = sorted({a["text"] for a in self.annotations})
        text, ok = QInputDialog.getItem(
            self, "Add behavioral label",
            "Label (pick an existing one or type a new one):",
            existing, 0, True)
        if ok and text.strip():
            self.add_annotation(x0, x1, text.strip())

    def add_annotation(self, start: float, end: float, text: str) -> None:
        self.annotations.append({"start": float(start), "end": float(end), "text": text})
        self.annotations.sort(key=lambda a: a["start"])
        self._redraw_annotations()
        self.annotationsChanged.emit()

    def remove_annotation(self, index: int) -> None:
        if 0 <= index < len(self.annotations):
            del self.annotations[index]
            self._redraw_annotations()
            self.annotationsChanged.emit()

    def _redraw_annotations(self) -> None:
        self.ann_track.clear()
        for item in self._overview_ann_items:
            self.overview_plot.removeItem(item)
        self._overview_ann_items.clear()

        for a in self.annotations:
            color = _label_color(a["text"])
            brush = pg.mkBrush(color.red(), color.green(), color.blue(), 100)
            pen = pg.mkPen(color, width=1)
            center = (a["start"] + a["end"]) / 2.0

            region = pg.LinearRegionItem((a["start"], a["end"]), movable=False,
                                          brush=brush, pen=pen)
            region.setZValue(-10)
            self.ann_track.addItem(region)
            txt = pg.TextItem(a["text"], color="#0d0d2a", anchor=(0.5, 0.5))
            txt.setPos(center, 0)
            self.ann_track.addItem(txt)

            ov_region = pg.LinearRegionItem((a["start"], a["end"]), movable=False,
                                             brush=brush, pen=pen)
            ov_region.setZValue(-10)
            self.overview_plot.addItem(ov_region)
            self._overview_ann_items.append(ov_region)

    def load_annotations(self, path: str) -> None:
        self.annotations = []
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                self.annotations = df[["start", "end", "text"]].to_dict("records")
                self.annotations.sort(key=lambda a: a["start"])
            except Exception:
                self.annotations = []
        self._redraw_annotations()

    def save_annotations(self, path: str) -> None:
        pd.DataFrame(self.annotations, columns=["start", "end", "text"]).to_csv(
            path, index=False)

    def _apply_mode(self) -> None:
        self.main_plot.clear()
        self.main_plot.addItem(self._marker)
        if self._mode == "dots":
            for i, letter in enumerate(FISH_LETTERS):
                t = self._data.fish_event_times[letter]
                item = self._dot_items[i]
                item.setData(t, np.full(len(t), i, dtype=float))
                self.main_plot.addItem(item)
            self.main_plot.setLabel("left", "Fish")
            self.main_plot.getAxis("left").setTicks(
                [[(i, letter) for i, letter in enumerate(FISH_LETTERS)]])
            self.main_plot.setYRange(-0.5, len(FISH_LETTERS) - 0.5)
        else:
            all_freqs = []
            self._freq_cache = {}
            for i, letter in enumerate(FISH_LETTERS):
                t = self._data.fish_event_times[letter]
                item = self._freq_items[i]
                if len(t) > 1:
                    isi = np.diff(t)
                    freq = np.where(isi > 0, 1.0 / isi, 0.0)
                    item.setData(t[1:], freq)
                    all_freqs.append(freq)
                    self._freq_cache[letter] = (t[1:], freq)
                else:
                    item.setData([], [])
                    self._freq_cache[letter] = (np.array([]), np.array([]))
                self.main_plot.addItem(item)
            self.main_plot.addItem(self._threshold_line)
            self.main_plot.addItem(self._suspect_scatter)
            self.main_plot.setLabel("left", "Instantaneous freq (Hz)")
            self.main_plot.getAxis("left").setTicks(None)
            if not self.lock_y_check.isChecked():
                if all_freqs:
                    hi = np.percentile(np.concatenate(all_freqs), 99)
                else:
                    hi = 1.0
                for sp, val in ((self.ymin_spin, 0.0), (self.ymax_spin, max(hi * 1.1, 1.0))):
                    sp.blockSignals(True)
                    sp.setValue(val)
                    sp.blockSignals(False)
            self._update_freq_yrange()
            self._update_suspect_markers()

    def _update_freq_yrange(self) -> None:
        if self._mode != "freq":
            return
        ymin, ymax = self.ymin_spin.value(), self.ymax_spin.value()
        if ymax <= ymin:
            return
        self.main_plot.setYRange(ymin, ymax, padding=0)

    def _update_suspect_markers(self) -> None:
        if self._mode != "freq":
            return
        thresh = self.max_plausible_spin.value()
        self._threshold_line.setPos(thresh)
        xs, ys = [], []
        for letter in FISH_LETTERS:
            t, freq = self._freq_cache.get(letter, (np.array([]), np.array([])))
            bad = freq > thresh
            if bad.any():
                xs.append(t[bad])
                ys.append(freq[bad])
        if xs:
            self._suspect_scatter.setData(np.concatenate(xs), np.concatenate(ys))
        else:
            self._suspect_scatter.setData([], [])

    def _on_mode_changed(self, idx: int) -> None:
        self._mode = "dots" if idx == 0 else "freq"
        self._yrange_row.setVisible(self._mode == "freq")
        self._apply_mode()
        self.set_time(self._current_time)

    def set_time(self, t: float) -> None:
        self._current_time = t
        self._marker.setPos(t)
        half = self.window_spin.value() / 2.0
        self.main_plot.setXRange(t - half, t + half, padding=0)
        self._region.blockSignals(True)
        self._region.setRegion((t - half, t + half))
        self._region.blockSignals(False)

    def _on_region_dragged(self) -> None:
        lo, hi = self._region.getRegion()
        center = (lo + hi) / 2.0
        frame = self._data.nearest_video_frame_for_time(center)
        self.frameJumpRequested.emit(frame)


class GroundTruthViewer(QMainWindow):
    def __init__(self, video_path: str, csv_path: str):
        super().__init__()
        self.setWindowTitle(f"Ground Truth Viewer — {os.path.basename(video_path)}")
        self.resize(1400, 950)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 50.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        loading_label = QLabel(f"Loading {os.path.basename(csv_path)} …")
        loading_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(loading_label)
        QApplication.processEvents()

        self.data = GroundTruthData(csv_path)
        self._annotations_path = os.path.splitext(csv_path)[0] + "_annotations.csv"

        self._current_frame = -1
        self._build_ui()
        self._apply_stylesheet()

        self.eod_panel.load_annotations(self._annotations_path)
        self._refresh_annotation_list()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.goto_frame(0)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        self.video_panel = VideoPanel()
        splitter.addWidget(self.video_panel)
        self.eod_panel = EODPanel(self.data)
        self.eod_panel.frameJumpRequested.connect(self.goto_frame)
        self.eod_panel.annotationsChanged.connect(self._on_annotations_changed)
        splitter.addWidget(self.eod_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        h_split = QSplitter(Qt.Horizontal)
        h_split.addWidget(splitter)
        h_split.addWidget(self._build_annotation_list_panel())
        h_split.setStretchFactor(0, 4)
        h_split.setStretchFactor(1, 1)
        root.addWidget(h_split, stretch=1)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.n_frames - 1))
        self.frame_slider.valueChanged.connect(self.goto_frame)
        controls.addWidget(self.frame_slider, stretch=1)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, max(0, self.n_frames - 1))
        self.frame_spin.valueChanged.connect(self.goto_frame)
        controls.addWidget(self.frame_spin)

        self.time_label = QLabel("t = 0.000 s")
        self.time_label.setMinimumWidth(220)
        controls.addWidget(self.time_label)

        root.addLayout(controls)

    def _build_annotation_list_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(280)
        v = QVBoxLayout(panel)
        v.addWidget(QLabel("Behavioral labels"))

        self.ann_list = QListWidget()
        self.ann_list.itemDoubleClicked.connect(self._on_annotation_double_clicked)
        v.addWidget(self.ann_list, stretch=1)

        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected_annotation)
        v.addWidget(del_btn)
        return panel

    def _refresh_annotation_list(self) -> None:
        self.ann_list.clear()
        for a in self.eod_panel.annotations:
            item = QListWidgetItem(f'{a["start"]:.2f}s – {a["end"]:.2f}s   {a["text"]}')
            self.ann_list.addItem(item)

    def _on_annotations_changed(self) -> None:
        self._refresh_annotation_list()
        self.eod_panel.save_annotations(self._annotations_path)

    def _on_annotation_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.ann_list.row(item)
        a = self.eod_panel.annotations[row]
        self.goto_frame(self.data.nearest_video_frame_for_time(a["start"]))

    def _delete_selected_annotation(self) -> None:
        row = self.ann_list.currentRow()
        if row >= 0:
            self.eod_panel.remove_annotation(row)

    def goto_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(self.n_frames - 1, int(frame_idx)))
        if frame_idx != self._current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap.read()
        if not ret:
            return
        self._current_frame = frame_idx

        frame_rgb = frame_bgr[:, :, ::-1]
        pose = self.data.pose_for_video_frame(frame_idx)
        self.video_panel.set_frame(frame_rgb, pose)

        t = self.data.time_for_video_frame(frame_idx)
        self.eod_panel.set_time(t)
        self.time_label.setText(f"t = {t:.3f} s   frame {frame_idx}/{self.n_frames - 1}")

        for w in (self.frame_slider, self.frame_spin):
            w.blockSignals(True)
            w.setValue(frame_idx)
            w.blockSignals(False)

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(max(1, int(1000 / self.fps)))
            self.play_btn.setText("Pause")

    def _advance_frame(self) -> None:
        nxt = self._current_frame + 1
        if nxt >= self.n_frames:
            self.timer.stop()
            self.play_btn.setText("Play")
            return
        self.goto_frame(nxt)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0d0d2a; color: #e0e0e0; }
            QPushButton {
                background-color: #1e2a3a; border: 1px solid #2a4a6a;
                border-radius: 4px; color: #e0e0e0; padding: 4px 10px;
            }
            QPushButton:hover { background-color: #2a3a5a; }
            QPushButton:pressed { background-color: #1a2a4a; }
            QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background-color: #1a1a3a; border: 1px solid #2a2a5a;
                border-radius: 3px; color: #e0e0e0; padding: 2px 4px;
            }
            QListWidget::item:selected { background-color: #2a4a6a; }
            QSlider::groove:horizontal { background: #1a1a3a; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #61afef; width: 12px; margin: -5px 0; border-radius: 6px;
            }
        """)
