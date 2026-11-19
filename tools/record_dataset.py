"""Interactive webcam recorder for the tactical-gesture dataset.

Workflow:

1. Pick a gesture class (CLI ``--class`` or interactive prompt).
2. A live preview window opens. Press **SPACE** to start a recording,
   hold the gesture for ~2–3 seconds, press **SPACE** again to stop.
3. The clip is saved to ``data/<class_name>/clip_<NNN>.mp4`` with
   ``NNN`` auto-incremented per class.
4. Press digit keys **0–9** to switch the active class on the fly.
5. Press **ESC** to quit; a per-class summary is printed.

OpenCV's ``cv2.putText`` does not render Cyrillic reliably, so the
on-screen overlay is drawn through PIL with a TrueType font from
``C:\\Windows\\Fonts``. The recorded frames themselves are unchanged —
the overlay is composited only into the preview.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from src.utils import setup_logging

logger = logging.getLogger(__name__)


CLIP_PATTERN = re.compile(r"^clip_(\d{3})\.mp4$")
DEFAULT_RESOLUTION = "640x480"
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _resolve_font(size: int) -> ImageFont.ImageFont:
    """Find a Windows font capable of rendering Cyrillic."""
    candidates = (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning(
        "Не знайдено системних шрифтів (arial/tahoma/segoeui); "
        "використовується default-шрифт PIL — українська буде дрібною."
    )
    return ImageFont.load_default()


def _get_font(size: int) -> ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = _resolve_font(size)
    return _FONT_CACHE[size]


def draw_ukrainian_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_size: int = 24,
) -> np.ndarray:
    """Draw Cyrillic text on a BGR frame using PIL.

    Args:
        frame: BGR uint8 frame of shape ``(H, W, 3)``.
        text: Text to draw (may contain Cyrillic).
        position: ``(x, y)`` top-left corner in pixels.
        color: BGR tuple, e.g. ``(0, 255, 0)``.
        font_size: Font size in pixels.

    Returns:
        New BGR frame with the text composited on top.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    pil_color = (color[2], color[1], color[0])
    draw.text(position, text, fill=pil_color, font=_get_font(font_size))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _next_clip_index(class_dir: Path) -> int:
    """Return the next free ``clip_NNN.mp4`` index in ``class_dir``."""
    if not class_dir.is_dir():
        return 0
    indices = [
        int(m.group(1))
        for p in class_dir.iterdir()
        if (m := CLIP_PATTERN.match(p.name)) is not None
    ]
    return max(indices, default=-1) + 1


def _prompt_class() -> int:
    """Ask the user in the terminal to pick a gesture class."""
    print("Доступні класи:")
    for i, name in enumerate(config.GESTURES):
        print(f"  {i}: {name}")
    while True:
        s = input("Виберіть номер класу (0-9): ").strip()
        try:
            idx = int(s)
        except ValueError:
            print("Невірний номер. Спробуйте ще раз.")
            continue
        if 0 <= idx < len(config.GESTURES):
            return idx
        print("Невірний номер. Спробуйте ще раз.")


def _parse_resolution(s: str) -> tuple[int, int]:
    """Parse ``"640x480"`` → ``(640, 480)``."""
    m = re.match(r"^(\d+)x(\d+)$", s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Невірний формат resolution: {s!r}, очікувано WIDTHxHEIGHT"
        )
    return int(m.group(1)), int(m.group(2))


def _print_summary(output_dir: Path) -> None:
    """Print a per-class clip-count table."""
    print()
    header = f"{'idx':>3}  {'class':<10}  {'clips':>5}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(config.GESTURES):
        class_dir = output_dir / name
        count = (
            sum(1 for p in class_dir.iterdir() if CLIP_PATTERN.match(p.name))
            if class_dir.is_dir()
            else 0
        )
        print(f"{i:>3}  {name:<10}  {count:>5}")


def run_recorder(
    camera: int,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
    initial_class_idx: int,
) -> None:
    """Run the interactive recording loop until the user presses ESC."""
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити вебкамеру {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    if (actual_w, actual_h) != (width, height) or abs(actual_fps - fps) > 0.5:
        logger.warning(
            "Камера повернула %dx%d @ %.1f FPS (запитано %dx%d @ %d FPS).",
            actual_w, actual_h, actual_fps, width, height, fps,
        )

    class_idx = initial_class_idx
    recording = False
    writer: cv2.VideoWriter | None = None
    record_start = 0.0
    record_frames = 0
    record_path: Path | None = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    window = "Tactical Gestures Dataset Recorder"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    logger.info("Активний клас: %s", config.GESTURES[class_idx])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Не вдалось прочитати кадр з вебкамери.")
                continue

            if recording and writer is not None:
                writer.write(frame)
                record_frames += 1

            class_name = config.GESTURES[class_idx]
            overlay = draw_ukrainian_text(
                frame, f"Клас: {class_name}", (10, 10),
                color=(0, 255, 0), font_size=28,
            )
            overlay = draw_ukrainian_text(
                overlay,
                "ПРОБІЛ — старт/стоп, 0-9 — клас, ESC — вихід",
                (10, 50), color=(255, 255, 255), font_size=18,
            )
            if recording:
                elapsed = time.perf_counter() - record_start
                overlay = draw_ukrainian_text(
                    overlay, f"ЗАПИС {elapsed:.1f}с",
                    (actual_w - 200, 10), color=(0, 0, 255), font_size=24,
                )
                cv2.circle(overlay, (actual_w - 30, 25), 12, (0, 0, 255), -1)

            cv2.imshow(window, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                if not recording:
                    class_dir = output_dir / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    next_idx = _next_clip_index(class_dir)
                    record_path = class_dir / f"clip_{next_idx:03d}.mp4"
                    writer = cv2.VideoWriter(
                        str(record_path), fourcc, fps, (actual_w, actual_h)
                    )
                    if not writer.isOpened():
                        logger.error(
                            "Не вдалося відкрити VideoWriter для %s",
                            record_path,
                        )
                        writer = None
                        continue
                    record_start = time.perf_counter()
                    record_frames = 0
                    recording = True
                    logger.info("Старт запису: %s", record_path)
                else:
                    duration = time.perf_counter() - record_start
                    if writer is not None:
                        writer.release()
                        writer = None
                    logger.info(
                        "Збережено: %s, тривалість %.2f с, кадрів %d",
                        record_path, duration, record_frames,
                    )
                    recording = False
            elif 0x30 <= key <= 0x39:  # '0' .. '9'
                new_idx = key - 0x30
                if new_idx >= len(config.GESTURES) or new_idx == class_idx:
                    continue
                if recording:
                    logger.warning(
                        "Запис триває; зміна класу можлива тільки після зупинки."
                    )
                else:
                    class_idx = new_idx
                    logger.info("Активний клас: %s", config.GESTURES[class_idx])
    finally:
        if recording and writer is not None:
            writer.release()
            logger.warning(
                "Перервано під час запису; неповний файл збережено: %s",
                record_path,
            )
        cap.release()
        cv2.destroyAllWindows()

    _print_summary(output_dir)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    parser = argparse.ArgumentParser(
        description="Запис відео-кліпів для датасету тактичних жестів.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Індекс вебкамери (за замовч. 0).",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        default=None,
        choices=config.GESTURES,
        help="Початковий клас жесту. Якщо не вказано — інтерактивний вибір.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATA_DIR,
        help="Каталог для збереження кліпів (за замовч. config.DATA_DIR).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.NATIVE_FPS,
        help="Цільова частота кадрів (за замовч. config.NATIVE_FPS).",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=_parse_resolution(DEFAULT_RESOLUTION),
        help="Роздільна здатність ширинаxвисота (за замовч. 640x480).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = _parse_args()

    if args.class_name is not None:
        initial_class_idx = config.GESTURES.index(args.class_name)
    else:
        initial_class_idx = _prompt_class()

    width, height = args.resolution
    run_recorder(
        camera=args.camera,
        output_dir=args.output,
        fps=args.fps,
        width=width,
        height=height,
        initial_class_idx=initial_class_idx,
    )


if __name__ == "__main__":
    main()
