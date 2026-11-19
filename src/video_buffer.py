"""Thread-safe ring buffer of recent webcam frames.

The buffer spans exactly ``window_s`` seconds at the camera's native frame
rate (defaults from :mod:`config`). Frames are stored in OpenCV-native
BGR ``uint8`` form; conversion to RGB float tensors is the job of
:mod:`src.preprocessing`, not this module.

A capture thread calls :py:meth:`VideoBuffer.add_frame` while an inference
thread periodically calls :py:meth:`VideoBuffer.sample_clip`; a
``threading.Lock`` guards every read and write. Locks are held only long
enough to mutate or snapshot the deque — never across ``numpy`` work.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

import config

logger = logging.getLogger(__name__)


class VideoBuffer:
    """Ring buffer of the most recent raw BGR frames.

    Attributes:
        maxlen: Buffer capacity in frames (= ``round(window_s * fps)``).
        clip_length: Number of frames returned by :py:meth:`sample_clip`
            (= ``config.CLIP_LENGTH``).
    """

    def __init__(
        self,
        window_s: float | None = None,
        fps: int | None = None,
        clip_length: int | None = None,
    ) -> None:
        """Build an empty buffer sized for ``window_s * fps`` frames.

        Args:
            window_s: Temporal window length in seconds. Defaults to
                ``config.TEMPORAL_WINDOW_S`` (2.0 s).
            fps: Camera native frame rate. Defaults to
                ``config.NATIVE_FPS`` (30).
            clip_length: T, the number of frames returned by
                :py:meth:`sample_clip`. Defaults to ``config.CLIP_LENGTH``
                (8).

        Raises:
            ValueError: If the resulting capacity is smaller than
                ``clip_length`` (uniform sampling would yield duplicates).
        """
        window_s = config.TEMPORAL_WINDOW_S if window_s is None else window_s
        fps = config.NATIVE_FPS if fps is None else fps
        self.clip_length: int = (
            config.CLIP_LENGTH if clip_length is None else clip_length
        )

        self.maxlen: int = int(round(window_s * fps))
        if self.maxlen < self.clip_length:
            raise ValueError(
                f"Buffer capacity {self.maxlen} is smaller than clip length "
                f"{self.clip_length}; choose a longer window or a higher FPS."
            )

        self._frames: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

        logger.info(
            "Ініціалізовано VideoBuffer: %d кадрів (%.2f с @ %d FPS), T=%d",
            self.maxlen,
            window_s,
            fps,
            self.clip_length,
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """Append a raw BGR frame to the buffer.

        Args:
            frame: Array of shape ``(H_raw, W_raw, 3)`` and dtype
                ``uint8`` (OpenCV BGR).

        Raises:
            ValueError: If ``frame.dtype`` is not ``uint8`` or
                ``frame.ndim`` is not 3.
        """
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Очікувано dtype=uint8, отримано {frame.dtype}"
            )
        if frame.ndim != 3:
            raise ValueError(
                f"Очікувано ndim=3 (H, W, C), отримано ndim={frame.ndim} "
                f"з shape={frame.shape}"
            )

        with self._lock:
            self._frames.append(frame)

    def is_ready(self) -> bool:
        """Return ``True`` once the buffer is full.

        Returns:
            ``True`` iff the buffer holds ``maxlen`` frames; only then is
            :py:meth:`sample_clip` allowed to return a tensor.
        """
        with self._lock:
            return len(self._frames) == self.maxlen

    def __len__(self) -> int:
        """Current number of frames stored (0 .. ``maxlen``)."""
        with self._lock:
            return len(self._frames)

    def sample_clip(self) -> np.ndarray | None:
        """Uniformly sample ``T`` frames from the full window.

        Indices are produced by ``np.linspace(0, N-1, T).astype(int)`` so
        the first sampled frame is always the oldest in the buffer and
        the last is always the newest, with step ≈ ``(N-1)/(T-1)``
        between adjacent samples.

        Returns:
            Array of shape ``(T, H_raw, W_raw, 3)`` and dtype ``uint8``,
            or ``None`` if the buffer is not yet full.
        """
        with self._lock:
            if len(self._frames) < self.maxlen:
                return None
            snapshot = list(self._frames)

        idx = np.linspace(0, self.maxlen - 1, self.clip_length).astype(int)
        return np.stack([snapshot[i] for i in idx], axis=0)

    def clear(self) -> None:
        """Empty the buffer."""
        with self._lock:
            self._frames.clear()
        logger.info("Буфер очищено.")


if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()

    buf = VideoBuffer()
    print(f"maxlen           = {buf.maxlen}")
    print(f"clip_length      = {buf.clip_length}")
    print(f"is_ready() empty = {buf.is_ready()}")
    print(f"len() empty      = {len(buf)}")

    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(60):
        buf.add_frame(synthetic)

    print(f"is_ready() full  = {buf.is_ready()}")
    print(f"len() full       = {len(buf)}")

    clip = buf.sample_clip()
    assert clip is not None, "sample_clip returned None on a full buffer"
    print(f"clip.shape       = {clip.shape}")
    print(f"clip.dtype       = {clip.dtype}")
    assert clip.shape == (8, 480, 640, 3), clip.shape
    assert clip.dtype == np.uint8, clip.dtype

    buf.clear()
    print(f"len() cleared    = {len(buf)}")
    assert len(buf) == 0

    print("Smoke test PASSED")
