"""Generate offline-TTS WAV files for the 10 tactical gestures.

Produces one ``{idx:02d}_{gesture}.wav`` per class in
``config.AUDIO_DIR`` (or any directory passed via ``--output``). The
filenames match what :class:`src.audio.AudioPlayer` expects.

The TTS engine is :mod:`pyttsx3` (offline, ships pre-built on Windows
via SAPI5).

Voice-selection policy (strict):

1. If a Ukrainian SAPI5 voice is installed, use it. Done.
2. Otherwise, inspect the system's default voice:
   - If it is on :data:`BLACKLISTED_VOICE_SUBSTRINGS` (Russian /
     Belarusian SAPI5 voices), **abort with exit code 2**. These voices
     mispronounce Ukrainian phonemes (e.g. Й → "и краткое"), which is
     unacceptable for a thesis about Ukrainian Armed Forces gestures.
     Generation is refused — the user must either install a Ukrainian
     voice or record the WAVs manually.
   - Otherwise (typically an English default voice), proceed with a
     WARN that the pronunciation will be poor.

Run::

    python -m tools.generate_audio              # write only missing files
    python -m tools.generate_audio --overwrite  # rewrite all 10
    python -m tools.generate_audio --output PATH
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyttsx3

import config
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Index here matches config.GESTURES order — do not reorder.
# Source: CLAUDE.md §2 "Classes" — Ukrainian Armed Forces tactical signals
# from the "Пантерівець" training video series.
GESTURE_PHRASES: dict[str, str] = {
    "danger":   "Увага",
    "forward":  "Уперед",
    "come":     "До мене",
    "down":     "Лягти",
    "attack":   "Атакувати",
    "encircle": "Оточити",
    "disperse": "Розсипатися",
    "position": "Позиція противника",
    "mine":     "Міна",
    "ambush":   "Засідка",
}

# Threshold below which a generated WAV is treated as "suspicious"
# (e.g. an English voice silently skipping the Cyrillic phrase).
MIN_FILE_SIZE_BYTES = 1024

# Voices we refuse to fall back to: they mispronounce Ukrainian phonemes
# in ways that are unacceptable for a thesis about Ukrainian Armed
# Forces tactical gestures. Belarusian voices share the same problem.
BLACKLISTED_VOICE_SUBSTRINGS: tuple[str, ...] = (
    "russian",
    "ru-ru",
    "irina",
    "pavel",
    "be-by",
    "belarusian",
)

UKRAINIAN_VOICE_SUBSTRINGS: tuple[str, ...] = ("uk-ua", "ukrainian", "ukr")


def _voice_text(voice: object) -> tuple[str, str]:
    """Return the lower-cased ``(name, id)`` of a SAPI5 voice descriptor."""
    name = (getattr(voice, "name", "") or "").lower()
    vid = (getattr(voice, "id", "") or "").lower()
    return name, vid


def _select_voice(engine: pyttsx3.Engine) -> tuple[str | None, str]:
    """Pick a Ukrainian voice; refuse to fall back to a blacklisted one.

    Args:
        engine: A live pyttsx3 engine.

    Returns:
        ``(voice_id, status)`` where ``status`` is one of:

        - ``"ukrainian"`` — found a Ukrainian voice; ``voice_id`` is its
          id, suitable for ``engine.setProperty('voice', voice_id)``.
        - ``"no_ukrainian_no_fallback"`` — no Ukrainian voice, but the
          system's default voice is acceptable (not blacklisted).
          ``voice_id`` is ``None``: use whatever pyttsx3 picks. Caller
          should warn the user that pronunciation will be poor.
        - ``"blacklisted_default"`` — no Ukrainian voice AND the default
          is on :data:`BLACKLISTED_VOICE_SUBSTRINGS` (e.g. Irina).
          ``voice_id`` is ``None`` and the caller MUST refuse to
          proceed.
    """
    voices = engine.getProperty("voices") or []

    for voice in voices:
        name, vid = _voice_text(voice)
        if any(n in name or n in vid for n in UKRAINIAN_VOICE_SUBSTRINGS):
            return voice.id, "ukrainian"

    default_id = engine.getProperty("voice")
    default_voice = next((v for v in voices if v.id == default_id), None)
    if default_voice is not None:
        name, vid = _voice_text(default_voice)
        if any(b in name or b in vid for b in BLACKLISTED_VOICE_SUBSTRINGS):
            return None, "blacklisted_default"

    return None, "no_ukrainian_no_fallback"


def generate_audio_files(
    output_dir: Path = config.AUDIO_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Generate one WAV per gesture using offline TTS.

    Args:
        output_dir: Destination directory. Created if missing.
        overwrite: If ``False``, existing files are kept and skipped.

    Returns:
        The list of every WAV path either written or already present.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    voice_id, status = _select_voice(engine)

    if status == "blacklisted_default":
        logger.error(
            "Система має російський SAPI5-голос (Irina), але не має українського.\n"
            "Згенеровані файли будуть звучати російською — це неприйнятно для\n"
            "цього проєкту. Дії:\n"
            "(1) встановити український голос: Settings → Time & language →\n"
            "    Language & region → Українська → Options → Speech → Download;\n"
            "АБО\n"
            "(2) записати 10 WAV-файлів власним голосом і помістити їх у\n"
            "    assets/audio/ з іменами 00_stop.wav ... 09_enemy.wav.\n"
            "Скрипт не виконує генерацію за російським голосом."
        )
        sys.exit(2)

    if status == "ukrainian":
        engine.setProperty("voice", voice_id)
        logger.info("Знайдено українську TTS-голос: %s", voice_id)
    else:  # "no_ukrainian_no_fallback"
        logger.warning(
            "Українську TTS-голос не знайдено; системний default — "
            "англомовний голос. Вимова кирилиці буде обмеженою. Щоб "
            "встановити українську: Settings → Time & Language → "
            "Language & region → Add language → Ukrainian → Speech."
        )

    summary: list[tuple[int, str, str, float, str]] = []
    written: list[Path] = []

    for idx, gesture in enumerate(config.GESTURES):
        phrase = GESTURE_PHRASES[gesture]
        filename = f"{idx:02d}_{gesture}.wav"
        path = output_dir / filename

        if path.exists() and not overwrite:
            size_kb = path.stat().st_size / 1024.0
            logger.info(
                "Пропущено (вже існує): %s (%.1f KB)", filename, size_kb
            )
            summary.append((idx, gesture, filename, size_kb, "skipped"))
            written.append(path)
            continue

        engine.save_to_file(phrase, str(path))
        engine.runAndWait()

        if not path.exists():
            logger.error("TTS не створив файл: %s", path)
            summary.append((idx, gesture, filename, 0.0, "failed"))
            continue

        size_bytes = path.stat().st_size
        size_kb = size_bytes / 1024.0
        if size_bytes < MIN_FILE_SIZE_BYTES:
            logger.warning(
                "Підозріло малий WAV: %s (%d байт). Можливо, голос не "
                "зміг прочитати фразу. Перевірте файл вручну.",
                filename,
                size_bytes,
            )
        logger.info(
            'Створено: %s (%.1f KB) — "%s"', filename, size_kb, phrase
        )
        summary.append((idx, gesture, filename, size_kb, "created"))
        written.append(path)

    print()
    header = f"{'idx':>3}  {'gesture':<10}  {'filename':<22}  {'size_kb':>8}  status"
    print(header)
    print("-" * len(header))
    for idx, gesture, filename, size_kb, status in summary:
        print(
            f"{idx:>3}  {gesture:<10}  {filename:<22}  "
            f"{size_kb:>8.1f}  {status}"
        )
    print()
    print(f"Усього файлів: {len(written)} у {output_dir}")

    return written


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the audio-generation entry point."""
    parser = argparse.ArgumentParser(
        description="Generate offline TTS WAV files for tactical gestures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing WAV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.AUDIO_DIR,
        help="Output directory (default: config.AUDIO_DIR).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    setup_logging()
    generate_audio_files(output_dir=args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
