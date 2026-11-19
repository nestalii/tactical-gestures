"""Streaming inference loop.

Stitches :mod:`src.video_buffer`, :mod:`src.preprocessing`,
:mod:`src.model`, and :mod:`src.postprocessing` into a single
:class:`StreamingInference` consumed by:

(a) the GUI worker (:mod:`gui.inference_worker`), which wraps an
    instance in a ``QThread`` and feeds it webcam frames;
(b) the CLI: ``python -m src.inference --video FILE`` or
    ``python -m src.inference --webcam INDEX``.

Inference fires once every ``config.WINDOW_STEP_FRAMES`` frames after the
buffer has filled, per thesis formula (3.15). Triggers are debounced by
:class:`src.postprocessing.TriggerPostProcessor` per formulas
(3.16)–(3.19).
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import config
from src.audio import AudioPlayer
from src.model import TimeSformerClassifier
from src.postprocessing import TriggerPostProcessor
from src.preprocessing import ClipPreprocessor
from src.utils import setup_logging
from src.video_buffer import VideoBuffer

logger = logging.getLogger(__name__)


class StreamingInference:
    """End-to-end streaming inference pipeline.

    Attributes:
        device: Target device for the model and intermediate tensors.
        buffer: Ring buffer of recent BGR frames.
        preprocessor: BGR-clip → normalized tensor converter.
        classifier: TimeSformer wrapper.
        postprocessor: EMA / threshold / cooldown gate.
        frames_seen: Total frames pushed since construction or
            :py:meth:`reset`.
        last_inference_latency_ms: Latency of the most recent
            forward+softmax, in milliseconds; ``None`` until the first
            inference runs.
        last_preprocess_latency_ms: Latency of the most recent
            preprocessing pass, in milliseconds; ``None`` until the
            first inference runs.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Initialize all sub-components and optionally load weights.

        Args:
            device: Target ``torch.device``; defaults to CUDA if
                available, otherwise CPU.
            checkpoint_path: Optional path to fine-tuned weights from
                :py:meth:`src.model.TimeSformerClassifier.save_checkpoint`.
                If ``None``, the model retains a freshly initialized
                K=10 head — predictions will be near-random, useful only
                for pipeline smoke tests.
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.buffer = VideoBuffer()
        self.preprocessor = ClipPreprocessor()
        self.classifier = TimeSformerClassifier(device=self.device)
        if checkpoint_path is not None:
            self.classifier.load_checkpoint(checkpoint_path)
        self.postprocessor = TriggerPostProcessor()

        self.frames_seen: int = 0
        self.last_inference_latency_ms: float | None = None
        self.last_preprocess_latency_ms: float | None = None

        self._stop_event = threading.Event()

    def push_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[int | None, dict[str, Any]]:
        """Push one frame and possibly return a trigger.

        Args:
            frame_bgr: Array of shape ``(H_raw, W_raw, 3)``, dtype
                ``uint8``, BGR.

        Returns:
            ``(trigger_idx, info)``. ``trigger_idx`` is the gesture
            class index to fire audio for, or ``None`` on most frames.
            ``info`` carries telemetry useful to a GUI: ``frames_seen``,
            ``warmup``, ``did_infer``, ``last_inference_latency_ms``,
            ``last_preprocess_latency_ms``, ``smoothed_probs``.
        """
        self.buffer.add_frame(frame_bgr)
        self.frames_seen += 1

        if not self.buffer.is_ready():
            return None, self._info(warmup=True, did_infer=False)

        if self.frames_seen % config.WINDOW_STEP_FRAMES != 0:
            return None, self._info(warmup=False, did_infer=False)

        clip = self.buffer.sample_clip()
        assert clip is not None  # is_ready() guarantees this

        t0 = time.perf_counter()
        pixel_values = self.preprocessor(clip).to(self.device)
        t1 = time.perf_counter()
        probs_tensor = self.classifier.predict_proba(pixel_values)
        t2 = time.perf_counter()

        self.last_preprocess_latency_ms = (t1 - t0) * 1000.0
        self.last_inference_latency_ms = (t2 - t1) * 1000.0

        probs_np = probs_tensor.squeeze(0).cpu().numpy()
        trigger_idx = self.postprocessor.update(probs_np)

        logger.debug(
            "frame=%d preprocess_ms=%.1f inference_ms=%.1f trigger=%s",
            self.frames_seen,
            self.last_preprocess_latency_ms,
            self.last_inference_latency_ms,
            trigger_idx,
        )

        return trigger_idx, self._info(warmup=False, did_infer=True)

    def _info(self, *, warmup: bool, did_infer: bool) -> dict[str, Any]:
        """Build the telemetry dict returned by :py:meth:`push_frame`."""
        return {
            "frames_seen": self.frames_seen,
            "warmup": warmup,
            "did_infer": did_infer,
            "last_inference_latency_ms": self.last_inference_latency_ms,
            "last_preprocess_latency_ms": self.last_preprocess_latency_ms,
            "smoothed_probs": self.postprocessor.smoothed_probs,
        }

    def run_video_file(
        self,
        path: Path,
        audio: AudioPlayer | None = None,
    ) -> list[dict[str, Any]]:
        """Run the pipeline end-to-end on a saved video file.

        Args:
            path: Path to a video file readable by OpenCV.
            audio: Optional player. If supplied, its
                :py:meth:`src.audio.AudioPlayer.play_by_index` is called
                for each trigger.

        Returns:
            A list of trigger events
            ``{frame_idx, time_s, class_idx, class_name, confidence}``
            in the order they fired.

        Raises:
            FileNotFoundError: If ``path`` does not point at a file.
            RuntimeError: If OpenCV cannot open the file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Відеофайл не знайдено: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV не зміг відкрити: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or float(config.NATIVE_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Відкрито відео %s: %d кадрів, %.2f FPS", path, total, fps
        )

        events: list[dict[str, Any]] = []
        total_inference_ms = 0.0
        total_inferences = 0
        max_inference_ms = 0.0
        t_start = time.monotonic()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                trig, info = self.push_frame(frame)
                if info["did_infer"]:
                    ms = info["last_inference_latency_ms"] or 0.0
                    total_inference_ms += ms
                    total_inferences += 1
                    if ms > max_inference_ms:
                        max_inference_ms = ms
                if trig is not None:
                    probs = info["smoothed_probs"]
                    confidence = (
                        float(probs[trig]) if probs is not None else 0.0
                    )
                    event = {
                        "frame_idx": self.frames_seen,
                        "time_s": self.frames_seen / fps,
                        "class_idx": int(trig),
                        "class_name": config.GESTURES[trig],
                        "confidence": confidence,
                    }
                    events.append(event)
                    logger.info(
                        "Тригер на кадрі %d (t=%.2f с): %s (%.3f)",
                        event["frame_idx"],
                        event["time_s"],
                        event["class_name"],
                        confidence,
                    )
                    if audio is not None:
                        audio.play_by_index(int(trig))
        finally:
            cap.release()

        wall_clock_s = time.monotonic() - t_start
        mean_inference_ms = (
            total_inference_ms / total_inferences if total_inferences else 0.0
        )
        logger.info(
            "Підсумок: %d кадрів, %d тригер(ів), "
            "середня latency=%.1f мс, макс. latency=%.1f мс, "
            "час обробки=%.2f с",
            self.frames_seen,
            len(events),
            mean_inference_ms,
            max_inference_ms,
            wall_clock_s,
        )
        return events

    def run_webcam(
        self,
        camera_index: int = 0,
        audio: AudioPlayer | None = None,
    ) -> None:
        """Run the pipeline from a webcam until :py:meth:`stop` is set.

        Uses ``cv2.CAP_DSHOW`` on Windows per CLAUDE.md §3.

        Args:
            camera_index: 0-based webcam index.
            audio: Optional player; if supplied,
                :py:meth:`src.audio.AudioPlayer.play_by_index` is called
                per trigger.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"Не вдалося відкрити вебкамеру з індексом {camera_index}"
            )
        logger.info(
            "Вебкамеру %d відкрито; натисніть Ctrl-C для виходу.",
            camera_index,
        )

        target_period = 1.0 / config.NATIVE_FPS
        loop_start = time.monotonic()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Не вдалось прочитати кадр з вебкамери.")
                    continue
                trig, _info = self.push_frame(frame)
                if trig is not None:
                    logger.info(
                        "Тригер: %s (кадр %d)",
                        config.GESTURES[trig],
                        self.frames_seen,
                    )
                    if audio is not None:
                        audio.play_by_index(int(trig))

                # Throttle only when the read came back faster than the
                # camera's nominal FPS (can happen with virtual cameras
                # or buffered reads). On a real webcam, cap.read() is
                # already rate-limited and the sleep below is skipped.
                now = time.monotonic()
                elapsed = now - loop_start
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
                loop_start = time.monotonic()
        except KeyboardInterrupt:
            logger.info("Перервано користувачем (Ctrl-C).")
        finally:
            cap.release()
            logger.info(
                "Вебкамеру закрито. Усього кадрів: %d.", self.frames_seen
            )

    def stop(self) -> None:
        """Signal the webcam loop to exit cleanly."""
        self._stop_event.set()

    def reset(self) -> None:
        """Reset all stateful components and counters."""
        self.buffer.clear()
        self.postprocessor.reset()
        self.frames_seen = 0
        self.last_inference_latency_ms = None
        self.last_preprocess_latency_ms = None
        self._stop_event.clear()
        logger.info("StreamingInference скинуто.")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the streaming-inference entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Tactical Gesture Recognition — streaming inference CLI"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--video",
        type=Path,
        help="Path to a video file to run inference on.",
    )
    source.add_argument(
        "--webcam",
        type=int,
        metavar="INDEX",
        help="Webcam index to capture from (0 = default camera).",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Play the recognized gesture's audio cue.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a fine-tuned weights file (.pt).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force a device; default auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    setup_logging()

    if args.video is None and args.webcam is None:
        print("Use --video PATH or --webcam INDEX")
        return

    device = torch.device(args.device) if args.device else None
    si = StreamingInference(device=device, checkpoint_path=args.checkpoint)

    audio: AudioPlayer | None = None
    if args.audio:
        try:
            audio = AudioPlayer()
        except FileNotFoundError as exc:
            logger.warning("Аудіо вимкнено: %s", exc)
            audio = None

    try:
        if args.video is not None:
            si.run_video_file(args.video, audio=audio)
        else:
            si.run_webcam(args.webcam, audio=audio)
    finally:
        if audio is not None:
            audio.close()


if __name__ == "__main__":
    main()
