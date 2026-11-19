"""Non-blocking WAV playback for recognized gestures.

Uses ``pygame.mixer`` because it ships pre-built Windows wheels and does
not require a C++ toolchain. All ten gesture WAVs are pre-loaded at
construction time so the inference hot path never touches disk; missing
files surface immediately, not at trigger time.

Audio files are expected at ``config.AUDIO_DIR`` and named
``{index:02d}_{gesture}.wav``, e.g. ``00_stop.wav`` (see CLAUDE.md §4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pygame

import config

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Loads and plays pre-recorded gesture commands without blocking.

    The mixer is initialized once in ``__init__`` and torn down by
    ``close()``. ``play`` and ``play_by_index`` return immediately;
    overlapping calls are routed to free channels by ``pygame.mixer``.

    Attributes:
        audio_dir: Directory containing the ``NN_<gesture>.wav`` files.
        gestures: Class names in fixed order (index 0..K-1).
        sounds: Mapping ``gesture_name -> pygame.mixer.Sound`` for
            preloaded clips.
    """

    def __init__(
        self,
        audio_dir: Path | None = None,
        gestures: list[str] | None = None,
    ) -> None:
        """Initialize the mixer and preload all gesture WAVs.

        Args:
            audio_dir: Directory holding the WAV files. Defaults to
                ``config.AUDIO_DIR``.
            gestures: Ordered list of gesture names. Defaults to
                ``config.GESTURES``.

        Raises:
            FileNotFoundError: If any expected WAV is missing.
            pygame.error: If the audio device cannot be opened.
        """
        self.audio_dir = audio_dir if audio_dir is not None else config.AUDIO_DIR
        self.gestures = gestures if gestures is not None else config.GESTURES

        try:
            pygame.mixer.init()
        except pygame.error as exc:
            logger.error("Не вдалося ініціалізувати pygame.mixer: %s", exc)
            raise

        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self._preload()

    def _preload(self) -> None:
        """Load every gesture WAV into memory.

        Raises:
            FileNotFoundError: If a WAV expected by index/name is absent.
        """
        for idx, name in enumerate(self.gestures):
            path = self.audio_dir / f"{idx:02d}_{name}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            try:
                self.sounds[name] = pygame.mixer.Sound(str(path))
            except pygame.error as exc:
                logger.error("Не вдалося завантажити %s: %s", path, exc)
                raise

        logger.info("Preloaded %d gesture sounds from %s", len(self.sounds), self.audio_dir)

    def play(self, gesture: str) -> None:
        """Play the audio for ``gesture`` without blocking.

        Args:
            gesture: One of the names in ``self.gestures``.
        """
        sound = self.sounds.get(gesture)
        if sound is None:
            logger.warning("Невідомий жест для відтворення: %s", gesture)
            return
        try:
            sound.play()
        except pygame.error as exc:
            logger.error("Помилка відтворення %s: %s", gesture, exc)

    def play_by_index(self, class_idx: int) -> None:
        """Play the audio for class index ``class_idx``.

        Args:
            class_idx: Zero-based class index in ``self.gestures``.
        """
        if not 0 <= class_idx < len(self.gestures):
            logger.warning("Індекс класу поза межами: %d", class_idx)
            return
        self.play(self.gestures[class_idx])

    def close(self) -> None:
        """Release the mixer. Safe to call multiple times."""
        try:
            pygame.mixer.quit()
        except pygame.error as exc:
            logger.warning("Помилка під час закриття mixer: %s", exc)

    def __enter__(self) -> AudioPlayer:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
