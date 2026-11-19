"""Main PyQt6 window for the tactical-gesture recognition system.

Horizontal split: video pane on the left, recognition panel (big label
+ probability bars + command log) on the right, bottom toolbar.
Dark theme via stylesheet. The heavy work (capture + inference) runs in
:class:`gui.inference_worker.InferenceWorker` on a :class:`QThread`;
this module only does the UI.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import config
from gui.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)


# Local mirror of tools.generate_audio.GESTURE_PHRASES (and the same
# mapping that lives in src/evaluate.py). Kept here so this module
# has no tools/ dependency. Keep in sync if class names change.
GESTURE_UKRAINIAN: dict[str, str] = {
    "danger":   "Увага",
    "forward":  "Уперед",
    "come":     "До мене",
    "down":     "Лягти",
    "attack":   "Атакувати",
    "encircle": "Оточити",
    "disperse": "Розсипатися",
    "position": "Позиція",
    "mine":     "Міна",
    "ambush":   "Засідка",
}


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QLabel {
    color: #e0e0e0;
}
QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 6px 14px;
    border-radius: 3px;
}
QPushButton:hover:!disabled {
    background-color: #3d3d3d;
}
QPushButton:checked {
    background-color: #4CAF50;
    color: #1e1e1e;
}
QPushButton:disabled {
    color: #666;
    border-color: #333;
}
QComboBox {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px 8px;
    min-width: 70px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #e0e0e0;
    selection-background-color: #4CAF50;
    selection-color: #1e1e1e;
}
QListWidget {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #555;
}
QProgressBar {
    background-color: #2d2d2d;
    border: 1px solid #555;
    text-align: center;
    color: #e0e0e0;
    height: 16px;
    border-radius: 2px;
}
QProgressBar::chunk {
    background-color: #757575;
}
QStatusBar {
    background-color: #252525;
    color: #b0b0b0;
}
"""


def detect_cameras(max_index: int = 5) -> list[int]:
    """Probe ``cv2.VideoCapture`` for camera indices 0..max_index-1.

    Returns the indices that opened successfully. On Windows each probe
    via ``CAP_DSHOW`` is ~0.5–1 s — cheap enough for a one-time UI init.
    """
    available: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


class MainWindow(QMainWindow):
    """Top-level window with video pane, recognition panel, and toolbar."""

    start_capture_requested = pyqtSignal(int)
    stop_capture_requested = pyqtSignal()
    audio_toggled = pyqtSignal(bool)

    def __init__(
        self,
        model_checkpoint_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the UI, apply theme, and spin up the worker thread."""
        super().__init__(parent)
        self.setWindowTitle("Система розпізнавання тактичних жестів")
        self.resize(1280, 720)
        self.setMinimumSize(1024, 600)

        self._test_audio_index = 0
        self._test_audio_timer = QTimer(self)
        self._test_audio_timer.setSingleShot(True)
        self._test_audio_timer.timeout.connect(self._play_next_test_audio)

        self._build_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        self._setup_worker(model_checkpoint_path)
        self._set_initial_state()

    def _build_ui(self) -> None:
        """Construct widgets and layouts."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        split = QHBoxLayout()
        outer.addLayout(split, stretch=1)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(QSize(640, 480))
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; border: 1px solid #444;"
        )
        split.addWidget(self.video_label, stretch=6)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(12, 8, 12, 8)
        split.addWidget(right_widget, stretch=4)

        self.recognized_label = QLabel("Готово до запуску")
        self.recognized_label.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #757575;"
        )
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("font-size: 13pt; color: #999;")
        right.addWidget(self.recognized_label)
        right.addWidget(self.confidence_label)
        right.addSpacing(8)

        probs_header = QLabel("<b>Ймовірності:</b>")
        probs_header.setStyleSheet("font-size: 11pt;")
        right.addWidget(probs_header)

        self.prob_bars: list[QProgressBar] = []
        for name in config.GESTURES:
            row = QHBoxLayout()
            label = QLabel(GESTURE_UKRAINIAN.get(name, name))
            label.setMinimumWidth(130)
            label.setStyleSheet("color: #c0c0c0;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            row.addWidget(label)
            row.addWidget(bar, stretch=1)
            right.addLayout(row)
            self.prob_bars.append(bar)

        right.addSpacing(8)
        log_header = QLabel("<b>Журнал команд:</b>")
        log_header.setStyleSheet("font-size: 11pt;")
        right.addWidget(log_header)

        self.log_widget = QListWidget()
        self.log_widget.setFont(QFont("Consolas", 10))
        self.log_widget.setStyleSheet(
            "QListWidget { background-color: #252525; padding: 4px; }"
        )
        right.addWidget(self.log_widget, stretch=1)

        toolbar = QHBoxLayout()
        outer.addLayout(toolbar)

        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)

        toolbar.addWidget(QLabel("Камера:"))
        self.camera_combo = QComboBox()
        cameras = detect_cameras()
        if not cameras:
            self.camera_combo.addItem("Не знайдено", None)
        else:
            for idx in cameras:
                self.camera_combo.addItem(str(idx), idx)
        toolbar.addWidget(self.camera_combo)

        self.audio_toggle = QPushButton("🔊 Звук увімкнено")
        self.audio_toggle.setCheckable(True)
        self.audio_toggle.setChecked(True)
        toolbar.addWidget(self.audio_toggle)

        self.test_audio_button = QPushButton("Тест звуку")
        toolbar.addWidget(self.test_audio_button)

        self.clear_log_button = QPushButton("Очистити журнал")
        toolbar.addWidget(self.clear_log_button)

        toolbar.addStretch()

        self.statusBar().showMessage("Очікування старту")

    def _setup_worker(self, checkpoint_path: Path | None) -> None:
        """Create worker, move it to its own QThread, wire signals."""
        self.worker = InferenceWorker(checkpoint_path=checkpoint_path)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)

        # Worker → main (queued by default across threads).
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.prediction_ready.connect(self.on_prediction_ready)
        self.worker.raw_probs_ready.connect(self.on_raw_probs_ready)
        self.worker.status_message.connect(self.statusBar().showMessage)
        self.worker.error_occurred.connect(self.on_error)

        # Main → worker.
        self.start_capture_requested.connect(self.worker.start_capture)
        self.stop_capture_requested.connect(self.worker.stop_capture)
        self.audio_toggled.connect(self.worker.set_audio_enabled)

        self.thread.started.connect(self.worker.initialize)
        self.thread.start()

        # Button wiring.
        self.start_button.clicked.connect(self.on_start_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.audio_toggle.toggled.connect(self.on_audio_toggled)
        self.test_audio_button.clicked.connect(self.on_test_audio_clicked)
        self.clear_log_button.clicked.connect(self.log_widget.clear)

    def _set_initial_state(self) -> None:
        """Apply the idle visual state."""
        self.stop_button.setEnabled(False)
        if not detect_cameras():
            # If camera detection returned nothing, the start button
            # would have nothing useful to do. Leave the Stop disabled.
            self.start_button.setEnabled(False)
            self.statusBar().showMessage(
                "Жодної камери не знайдено. Підключіть вебкамеру і перезапустіть."
            )

    # ----- Slots ------------------------------------------------------

    def on_start_clicked(self) -> None:
        idx = self.camera_combo.currentData()
        if idx is None:
            QMessageBox.warning(
                self, "Камера", "Не вибрано доступну камеру."
            )
            return
        self.start_capture_requested.emit(int(idx))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.recognized_label.setText("Очікування жесту…")
        self.recognized_label.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #c0c0c0;"
        )
        self.confidence_label.setText("")

    def on_stop_clicked(self) -> None:
        self.stop_capture_requested.emit()
        # Also flip the worker's flag directly — guarantees the loop
        # exits even before its event queue is drained.
        self.worker.running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.recognized_label.setText("Зупинено")
        self.recognized_label.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #757575;"
        )
        self.confidence_label.setText("")

    def on_audio_toggled(self, checked: bool) -> None:
        self.audio_toggle.setText(
            "🔊 Звук увімкнено" if checked else "🔇 Звук вимкнено"
        )
        self.audio_toggled.emit(checked)

    def on_test_audio_clicked(self) -> None:
        """Cycle through all 10 WAVs with 1.5 s gaps via a QTimer chain."""
        if self.worker.audio_player is None:
            QMessageBox.warning(
                self,
                "Аудіо",
                "Аудіо плеєр не ініціалізовано — перевірте assets/audio/.",
            )
            return
        self._test_audio_index = 0
        self._play_next_test_audio()

    def _play_next_test_audio(self) -> None:
        if (
            self.worker.audio_player is None
            or self._test_audio_index >= len(config.GESTURES)
        ):
            self.statusBar().showMessage("Тест звуку завершено.")
            return
        idx = self._test_audio_index
        name = config.GESTURES[idx]
        try:
            self.worker.audio_player.play_by_index(idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Помилка тесту звуку для %s: %s", name, exc)
        self.statusBar().showMessage(
            f"Тест звуку: {GESTURE_UKRAINIAN.get(name, name)} ({idx+1}/10)"
        )
        self._test_audio_index += 1
        self._test_audio_timer.start(1500)

    def on_frame_ready(self, frame_bgr: np.ndarray) -> None:
        """Display the captured BGR frame in the video pane."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(
            rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def on_prediction_ready(
        self, class_name: str, confidence: float, smoothed_probs: np.ndarray
    ) -> None:
        """Flip the big label green and append to the log."""
        ua_name = GESTURE_UKRAINIAN.get(class_name, class_name)
        self.recognized_label.setText(f"Розпізнано: {ua_name.upper()}")
        self.recognized_label.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #4CAF50;"
        )
        self.confidence_label.setText(f"({confidence * 100:.0f}%)")

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp}  {ua_name:<14} ({confidence * 100:.0f}%)"
        self.log_widget.insertItem(0, entry)
        while self.log_widget.count() > 10:
            self.log_widget.takeItem(self.log_widget.count() - 1)

    def on_raw_probs_ready(self, probs: np.ndarray) -> None:
        """Update the 10 probability bars; argmax bar is green."""
        top_idx = int(np.argmax(probs))
        for i, bar in enumerate(self.prob_bars):
            color = "#4CAF50" if i == top_idx else "#757575"
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: #2d2d2d; border: 1px solid #555;"
                f" text-align: center; color: #e0e0e0; height: 16px; }}"
                f"QProgressBar::chunk {{ background-color: {color}; }}"
            )
            bar.setValue(int(round(float(probs[i]) * 100)))

    def on_error(self, message: str) -> None:
        """Show a critical dialog and reset run controls."""
        logger.error("Worker error: %s", message)
        QMessageBox.critical(self, "Помилка", message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.recognized_label.setText("Помилка")
        self.recognized_label.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #e57373;"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        """Stop the worker, quit the thread, then accept close."""
        logger.info("MainWindow.closeEvent: завершення воркера...")
        self.worker.running = False
        self.stop_capture_requested.emit()
        # Give the loop a moment to break out before quitting the thread.
        self.thread.quit()
        if not self.thread.wait(2000):
            logger.warning("Воркер не зупинився за 2 с — terminate().")
            self.thread.terminate()
            self.thread.wait()
        # Release audio.
        try:
            self.worker.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Помилка shutdown воркера: %s", exc)
        event.accept()
