"""Dataset health check for tactical-gesture clips.

Scans ``data/<class>/clip_NNN.mp4`` and produces:

- a per-class summary table (count, total / mean duration, problems);
- a list of problem clips (short / long / wrong FPS / unreadable / ...);
- a class-balance warning (if any class holds < 0.5 × mean count);
- a final verdict.

Designed as a pre-flight check before training. Uses OpenCV
(``cv2.VideoCapture``) rather than ``decord``: OpenCV is more
recovery-friendly on slightly malformed files and lets us mark them
``unreadable`` instead of crashing the report.

Run::

    python -m tools.verify_dataset                # default thresholds
    python -m tools.verify_dataset --strict       # exit 1 on any issue
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

import config
from src.utils import setup_logging

logger = logging.getLogger(__name__)

SIZE_FLOOR_BYTES = 50 * 1024
FPS_TOLERANCE = 2.0


@dataclass
class ClipReport:
    """Inspection result for one video clip."""

    path: Path
    class_name: str
    size_mb: float
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int
    status: str
    detail: str

    @property
    def is_problem(self) -> bool:
        """``True`` if this clip is anything other than ``ok``."""
        return self.status != "ok"


def inspect_clip(
    path: Path,
    data_dir: Path,
    min_duration: float,
    max_duration: float,
    expected_fps: float,
) -> ClipReport:
    """Open ``path`` and return a structured health report.

    Args:
        path: Absolute path to a ``clip_NNN.mp4`` file.
        data_dir: Dataset root, used to compute the relative path
            shown in the report.
        min_duration: Lower bound for duration in seconds.
        max_duration: Upper bound for duration in seconds.
        expected_fps: Nominal frame rate; mismatch tolerated up to
            ``FPS_TOLERANCE`` FPS.

    Returns:
        A :class:`ClipReport` capturing every metric plus a single
        ``status`` string in the documented vocabulary.
    """
    rel = path.relative_to(data_dir)
    class_name = path.parent.name
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    def _bad(status: str, detail: str) -> ClipReport:
        return ClipReport(
            path=rel, class_name=class_name, size_mb=size_mb,
            duration_s=0.0, fps=0.0, frame_count=0,
            width=0, height=0, status=status, detail=detail,
        )

    if size_bytes < SIZE_FLOOR_BYTES:
        return _bad("size_too_small", f"{size_bytes / 1024:.1f} KB, < 50 KB")

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return _bad("unreadable", "cv2 не може відкрити")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0:
            cap.release()
            return _bad("zero_frames", "FRAME_COUNT == 0")
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return _bad("unreadable", "не вдалося прочитати перший кадр")
    except cv2.error as exc:
        return _bad("unreadable", f"cv2.error: {exc}")

    duration_s = frame_count / fps if fps > 0 else 0.0

    if abs(fps - expected_fps) > FPS_TOLERANCE:
        status = "fps_mismatch"
        detail = f"{fps:.1f} FPS, очікувано {expected_fps:.0f}"
    elif duration_s < min_duration:
        status = "short"
        detail = f"{duration_s:.2f} с, < {min_duration}"
    elif duration_s > max_duration:
        status = "long"
        detail = f"{duration_s:.2f} с, > {max_duration}"
    else:
        status = "ok"
        detail = ""

    return ClipReport(
        path=rel, class_name=class_name, size_mb=size_mb,
        duration_s=duration_s, fps=fps, frame_count=frame_count,
        width=width, height=height, status=status, detail=detail,
    )


def run_verify(
    data_dir: Path,
    min_duration: float,
    max_duration: float,
    expected_fps: float,
    strict: bool,
) -> int:
    """Inspect every clip in ``data_dir`` and print a report.

    Returns:
        The shell exit code (0 on success; 1 in ``--strict`` mode if
        any problems or balance warnings were found).
    """
    if not data_dir.is_dir():
        print(
            "❌ Датасет порожній — спочатку зніміть кліпи через "
            "tools.record_dataset"
        )
        return 0

    clip_paths = sorted(data_dir.glob("**/clip_*.mp4"))
    if not clip_paths:
        print(
            "❌ Датасет порожній — спочатку зніміть кліпи через "
            "tools.record_dataset"
        )
        return 0

    logger.info("Знайдено %d кліпів у %s", len(clip_paths), data_dir)
    reports = [
        inspect_clip(p, data_dir, min_duration, max_duration, expected_fps)
        for p in clip_paths
    ]

    known_reports = [r for r in reports if r.class_name in config.GESTURES]
    stale_reports = [r for r in reports if r.class_name not in config.GESTURES]

    by_class: dict[str, list[ClipReport]] = {}
    for r in known_reports:
        by_class.setdefault(r.class_name, []).append(r)

    print()
    header = (
        f"{'Клас':<10}  {'Кліпів':>6}  {'Загальна тривалість (с)':>23}  "
        f"{'Середня (с)':>11}  {'Проблем':>7}"
    )
    bar = "─" * len(header)
    print(header)
    print(bar)

    total_clips = 0
    total_duration = 0.0
    total_problems = 0
    for name in config.GESTURES:
        items = by_class.get(name, [])
        count = len(items)
        total_dur = sum(r.duration_s for r in items)
        problems = sum(1 for r in items if r.is_problem)
        mean_str = f"{(total_dur / count):.2f}" if count else "—"
        print(
            f"{name:<10}  {count:>6}  {total_dur:>23.1f}  "
            f"{mean_str:>11}  {problems:>7}"
        )
        total_clips += count
        total_duration += total_dur
        total_problems += problems

    print(bar)
    print(
        f"{'УСЬОГО':<10}  {total_clips:>6}  {total_duration:>23.1f}  "
        f"{'—':>11}  {total_problems:>7}"
    )

    if stale_reports:
        stale_by_class: dict[str, list[ClipReport]] = {}
        for r in stale_reports:
            stale_by_class.setdefault(r.class_name, []).append(r)
        print()
        print(
            "⚠️ Невідомі класи (не у config.GESTURES; можливо, застаріли "
            "після перейменування):"
        )
        for name in sorted(stale_by_class):
            count = len(stale_by_class[name])
            print(f"  data/{name}/ — {count} кліп(ів)")

    problem_reports = [r for r in reports if r.is_problem]
    if problem_reports:
        print()
        print("⚠️ Проблемні кліпи:")
        for r in problem_reports:
            print(f"  {str(r.path):<40}  {r.status} ({r.detail})")

    counts = {name: len(by_class.get(name, [])) for name in config.GESTURES}
    mean_count = sum(counts.values()) / len(counts)
    threshold = 0.5 * mean_count
    balance_msgs: list[str] = []
    for name, count in counts.items():
        if count >= threshold:
            continue
        if count == 0:
            balance_msgs.append(f"'{name}' не має жодного кліпу.")
        else:
            balance_msgs.append(
                f"'{name}' має лише {count} кліпів (середнє "
                f"{mean_count:.1f}). Рекомендовано довести до "
                f"~{int(round(mean_count))} кліпів для збалансованого "
                f"тренування."
            )

    if balance_msgs:
        print()
        print("⚠️ Дисбаланс класів:")
        for w in balance_msgs:
            print(f"  {w}")

    print()
    if total_problems == 0 and not balance_msgs and not stale_reports:
        print("✅ Датасет готовий до тренування")
        return 0

    print(
        f"⚠️ Датасет має {total_problems} проблемний(их) кліпів — "
        f"рекомендовано виправити"
    )
    return 1 if strict else 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    parser = argparse.ArgumentParser(
        description="Перевірка цілісності датасету тактичних жестів.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Каталог з даними (за замовч. config.DATA_DIR).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Мінімальна допустима тривалість кліпу, с (за замовч. 1.0).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Максимальна допустима тривалість кліпу, с (за замовч. 5.0).",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=config.NATIVE_FPS,
        help="Очікувана частота кадрів (за замовч. config.NATIVE_FPS).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Вихідний код 1 якщо знайдено проблеми (для CI).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = _parse_args()
    code = run_verify(
        data_dir=args.data_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        expected_fps=args.expected_fps,
        strict=args.strict,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
