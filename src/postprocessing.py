"""Temporal post-processing for streaming inference.

Implements thesis §3.5 formulas (3.15)–(3.19) end-to-end:

    (3.15) sliding-window clip definition (lives in the inference loop)
    (3.16) p̂_n = α · p_n + (1 − α) · p̂_{n−1}
    (3.17) ĉ_n = argmax_c p̂_n,  ĉ_conf = max_c p̂_n
    (3.18) trigger gate: ĉ_conf > θ_conf
    (3.19) cooldown gate: real-time gap since previous trigger ≥ τ_cd

Cooldown is measured in seconds via :func:`time.monotonic`, never in
frame counts: GUI pauses, slow inference steps and dropped frames must
not corrupt the debounce. Frame indices are deliberately NOT used for
this purpose.

The EMA buffer is stateful and the cooldown timer is global, so the
class is guarded by a :class:`threading.Lock` for safe use across the
capture thread, the inference thread, and the GUI thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

import config

logger = logging.getLogger(__name__)


class TriggerPostProcessor:
    """EMA + threshold + cooldown gate over per-clip probabilities.

    Attributes:
        alpha: EMA coefficient α in (0, 1]. ``alpha = 1`` disables
            smoothing (raw probabilities pass through).
        threshold: Confidence threshold θ_conf in (0, 1).
        cooldown_seconds: τ_cd, minimum wall-clock seconds between two
            successive triggers; ``0`` disables the cooldown.
        num_classes: K, expected length of every probability vector
            passed to :py:meth:`update`.
    """

    def __init__(
        self,
        alpha: float = config.EMA_ALPHA,
        threshold: float = config.CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = config.COOLDOWN_SECONDS,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        """Construct and validate the post-processor.

        Args:
            alpha: EMA coefficient α, in (0, 1].
            threshold: Confidence threshold θ_conf in (0, 1).
            cooldown_seconds: Minimum interval in seconds between two
                consecutive triggers; must be ``>= 0``.
            num_classes: Expected length of every probability vector.

        Raises:
            ValueError: If any parameter is outside its admissible range.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Очікувано 0 < alpha <= 1, отримано {alpha}")
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"Очікувано 0 < threshold < 1, отримано {threshold}"
            )
        if cooldown_seconds < 0.0:
            raise ValueError(
                f"Очікувано cooldown_seconds >= 0, отримано {cooldown_seconds}"
            )

        self.alpha = alpha
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.num_classes = num_classes

        self._smoothed: np.ndarray | None = None
        self._last_trigger_time: float = -math.inf
        self._lock = threading.Lock()

        logger.info(
            "TriggerPostProcessor: α=%.2f, θ_conf=%.2f, τ_cd=%.2f с, K=%d",
            alpha,
            threshold,
            cooldown_seconds,
            num_classes,
        )

    def update(self, probs: np.ndarray) -> int | None:
        """Update the smoother and decide whether to fire a trigger now.

        Atomically applies formulas (3.16)–(3.19): bootstrap or EMA
        update, then argmax / max, threshold gate, and real-time
        cooldown gate.

        Args:
            probs: Per-clip probability vector of shape
                ``(num_classes,)``, dtype floating, summing to 1
                (tolerance ``1e-3``).

        Returns:
            Class index in ``[0, num_classes)`` if a trigger should
            fire now, otherwise ``None`` (confidence below threshold or
            still inside the cooldown window).

        Raises:
            ValueError: If ``probs`` has the wrong shape, dtype, or
                does not sum to 1.
        """
        if probs.shape != (self.num_classes,):
            raise ValueError(
                f"Очікувано shape=({self.num_classes},), отримано "
                f"{probs.shape}"
            )
        if not np.issubdtype(probs.dtype, np.floating):
            raise ValueError(
                f"Очікувано dtype з плаваючою комою, отримано {probs.dtype}"
            )
        total = float(probs.sum())
        if abs(total - 1.0) >= 1e-3:
            raise ValueError(
                f"Очікувано sum(probs)≈1, отримано {total:.6f}"
            )

        with self._lock:
            # (3.16) Bootstrap on the very first call; otherwise EMA recurrence.
            if self._smoothed is None:
                self._smoothed = probs.copy()
            else:
                self._smoothed = (
                    self.alpha * probs + (1.0 - self.alpha) * self._smoothed
                )

            # (3.17) argmax + max over the smoothed vector.
            candidate = int(np.argmax(self._smoothed))
            confidence = float(self._smoothed[candidate])

            logger.debug(
                "update: candidate=%d (%s), confidence=%.3f",
                candidate,
                config.GESTURES[candidate],
                confidence,
            )

            # (3.18) Confidence gate.
            if confidence < self.threshold:
                return None

            # (3.19) Real-time cooldown gate.
            now = time.monotonic()
            if now - self._last_trigger_time < self.cooldown_seconds:
                return None

            self._last_trigger_time = now

        # Released the lock before the user-facing INFO log to keep the
        # critical section as short as possible.
        logger.info(
            "Спрацював тригер: %s (впевненість=%.3f)",
            config.GESTURES[candidate],
            confidence,
        )
        return candidate

    def reset(self) -> None:
        """Clear the smoother state and reset the cooldown timer.

        Thread-safe. Use when the GUI restarts the stream.
        """
        with self._lock:
            self._smoothed = None
            self._last_trigger_time = -math.inf
        logger.info("Стан пост-обробника скинуто.")

    @property
    def smoothed_probs(self) -> np.ndarray | None:
        """Snapshot of the current smoothed probability vector.

        Returns:
            A copy of the smoothed vector (so the caller can safely
            read it outside the lock), or ``None`` if
            :py:meth:`update` has not been called yet.
        """
        with self._lock:
            if self._smoothed is None:
                return None
            return self._smoothed.copy()


if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()

    K = config.NUM_CLASSES

    def make_probs(peak_idx: int, peak_value: float) -> np.ndarray:
        """Probability vector with one peak and a uniform tail."""
        tail = (1.0 - peak_value) / (K - 1)
        p = np.full(K, tail, dtype=np.float32)
        p[peak_idx] = peak_value
        return p

    pp = TriggerPostProcessor()

    triggers: list[tuple[int, float, int]] = []
    t0 = time.monotonic()
    for i in range(20):
        step = i + 1
        if step <= 5:
            probs = make_probs(0, 0.85)
        elif step <= 10:
            probs = make_probs(0, 0.60)
        else:
            probs = make_probs(3, 0.80)

        result = pp.update(probs)
        elapsed = time.monotonic() - t0
        snap = pp.smoothed_probs
        assert snap is not None
        conf = float(snap.max())
        name = config.GESTURES[result] if result is not None else None
        print(
            f"step={step:2d}  t={elapsed:5.2f}s  conf={conf:.3f}  "
            f"trigger={result}  name={name}"
        )
        if result is not None:
            triggers.append((step, elapsed, result))
        if step < 20:
            time.sleep(0.5)

    assert triggers, "Жодного тригера не спрацювало."
    assert triggers[0][0] <= 5, (
        f"Перший тригер на кроці {triggers[0][0]}; очікувано в перших 5."
    )
    for (s1, t1, _), (s2, t2, _) in zip(triggers, triggers[1:]):
        gap = t2 - t1
        assert gap >= 2.0 - 0.05, (
            f"Інтервал між тригерами {gap:.3f}с менший за cooldown."
        )
    class3 = [(s, t, c) for s, t, c in triggers if c == 3]
    assert class3, "Тригер для класу 3 (down) не спрацював."
    assert class3[0][0] > 11, (
        f"Перший тригер для класу 3 на кроці {class3[0][0]}; "
        f"очікувано після згладжування (> 11)."
    )
    print("Smoke test PASSED")
