"""QObject worker that runs capture + inference on a background thread.

Owned by :class:`gui.main_window.MainWindow`. Constructed on the main
thread, then ``moveToThread()``-ed onto a :class:`QThread` whose
``started`` signal fires :py:meth:`InferenceWorker.initialize` — the
expensive setup (TimeSformer download, image processor, audio preload)
runs there so the UI thread stays responsive.

The hot loop (capture → buffer → inference → trigger) lives in
:py:meth:`_run_loop`. It calls ``QCoreApplication.processEvents()`` each
iteration so queued cross-thread slot invocations (e.g. ``stop_capture``)
get a chance to run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal, pyqtSlot

import config
from src.audio import AudioPlayer
from src.model import TimeSformerClassifier
from src.postprocessing import TriggerPostProcessor
from src.preprocessing import ClipPreprocessor
from src.video_buffer import VideoBuffer

logger = logging.getLogger(__name__)


class InferenceWorker(QObject):
    """Capture-and-infer loop running on a worker thread.

    Signals are emitted with queued connections to the main (UI) thread
    when the worker has been moved off the main thread.

    Attributes:
        running: Set to ``False`` from any thread to stop the loop.
            Plain attribute write — Python's GIL makes it atomic.
        audio_enabled: Toggled via :py:meth:`set_audio_enabled`.
    """

    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float, np.ndarray)
    raw_probs_ready = pyqtSignal(np.ndarray)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Store config; do NOT load the model here (wrong thread)."""
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.device: torch.device | None = None
        self.model: TimeSformerClassifier | None = None
        self.video_buffer: VideoBuffer | None = None
        self.preprocessor: ClipPreprocessor | None = None
        self.postprocessor: TriggerPostProcessor | None = None
        self.audio_player: AudioPlayer | None = None
        self.camera: cv2.VideoCapture | None = None
        self.running: bool = False
        self.audio_enabled: bool = True
        self._initialized: bool = False

    @pyqtSlot()
    def initialize(self) -> None:
        """Load model and helpers on the worker thread.

        Connected to ``QThread.started`` so it runs once the worker
        thread's event loop is up.
        """
        try:
            self.status_message.emit("Завантаження моделі...")
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model = TimeSformerClassifier(device=self.device)
            if self.checkpoint_path is not None and Path(self.checkpoint_path).is_file():
                self.model.load_checkpoint(self.checkpoint_path)
                logger.info("Завантажено чекпойнт: %s", self.checkpoint_path)
            else:
                if self.checkpoint_path is not None:
                    logger.warning(
                        "Чекпойнт не знайдено: %s — використовую випадкові ваги.",
                        self.checkpoint_path,
                    )

            self.video_buffer = VideoBuffer()
            self.preprocessor = ClipPreprocessor()
            self.postprocessor = TriggerPostProcessor()

            try:
                self.audio_player = AudioPlayer()
            except FileNotFoundError as exc:
                logger.warning("Аудіо вимкнено: %s", exc)
                self.audio_player = None

            self._initialized = True
            self.status_message.emit("Готово до запуску")
            logger.info("InferenceWorker ініціалізовано (device=%s)", self.device)
        except Exception as exc:
            logger.exception("Помилка ініціалізації")
            self.error_occurred.emit(f"Помилка ініціалізації: {exc}")

    @pyqtSlot(int)
    def start_capture(self, camera_index: int) -> None:
        """Open the camera and enter the capture loop on this thread."""
        if not self._initialized:
            self.error_occurred.emit("Воркер ще не ініціалізовано.")
            return
        if self.running:
            logger.warning("start_capture викликано, але цикл вже працює.")
            return

        try:
            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.error_occurred.emit(
                    f"Не вдалося відкрити камеру з індексом {camera_index}."
                )
                self.camera = None
                return

            assert self.video_buffer is not None
            assert self.postprocessor is not None
            self.video_buffer.clear()
            self.postprocessor.reset()
            self.running = True
            self.status_message.emit(f"Запис з камери {camera_index}")
            logger.info("Старт capture (camera=%d).", camera_index)
            self._run_loop()
        except Exception as exc:
            logger.exception("Помилка під час старту запису")
            self.error_occurred.emit(f"Помилка: {exc}")
        finally:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.running = False
            self.status_message.emit("Зупинено")

    def _run_loop(self) -> None:
        """Inner capture-inference loop. Called only from start_capture."""
        assert self.camera is not None
        assert self.video_buffer is not None
        assert self.preprocessor is not None
        assert self.model is not None
        assert self.postprocessor is not None

        frames_seen = 0
        consecutive_failures = 0
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self.error_occurred.emit(
                            "Камера не повертає кадри (30 невдалих читань поспіль)."
                        )
                        break
                    QCoreApplication.processEvents()
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0

                self.frame_ready.emit(frame)
                self.video_buffer.add_frame(frame)
                frames_seen += 1

                if (
                    self.video_buffer.is_ready()
                    and frames_seen % config.WINDOW_STEP_FRAMES == 0
                ):
                    clip = self.video_buffer.sample_clip()
                    if clip is None:
                        continue
                    pixel_values = self.preprocessor(clip).to(self.device)
                    probs_tensor = self.model.predict_proba(pixel_values)
                    probs = probs_tensor.squeeze(0).cpu().numpy()
                    self.raw_probs_ready.emit(probs)

                    trigger_idx = self.postprocessor.update(probs)
                    if trigger_idx is not None:
                        smoothed = self.postprocessor.smoothed_probs
                        if smoothed is None:
                            smoothed = probs
                        confidence = float(smoothed[trigger_idx])
                        class_name = config.GESTURES[trigger_idx]
                        self.prediction_ready.emit(
                            class_name, confidence, smoothed
                        )
                        if self.audio_enabled and self.audio_player is not None:
                            self.audio_player.play_by_index(trigger_idx)
                        logger.info(
                            "Тригер: %s (%.3f)", class_name, confidence
                        )

                QCoreApplication.processEvents()
                time.sleep(0.005)
            except Exception as exc:
                logger.exception("Виняток у циклі захоплення")
                self.error_occurred.emit(f"Помилка у циклі: {exc}")
                break

    @pyqtSlot()
    def stop_capture(self) -> None:
        """Signal the capture loop to exit on its next iteration."""
        if self.running:
            logger.info("stop_capture: зупиняю цикл.")
        self.running = False

    @pyqtSlot(bool)
    def set_audio_enabled(self, enabled: bool) -> None:
        """Toggle audio playback on trigger events."""
        self.audio_enabled = enabled
        logger.info("Аудіо %s.", "увімкнено" if enabled else "вимкнено")

    @pyqtSlot()
    def shutdown(self) -> None:
        """Release the audio player. Call before quitting the thread."""
        self.running = False
        if self.audio_player is not None:
            try:
                self.audio_player.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Помилка при закритті AudioPlayer: %s", exc)
            self.audio_player = None
        logger.info("InferenceWorker завершено.")
