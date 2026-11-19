"""Latency benchmark for the TimeSformer forward pass.

Runs a warmup loop, then a measurement loop, against a fixed synthetic
input on the requested device. Per thesis §4.4.6, the result is the
expected end-to-end inference cost of the model (not including video
capture, preprocessing, or post-processing). CPU numbers represent a
portable / chest-camera-class machine; CUDA numbers are the Colab T4
ceiling.

Run::

    python -m tools.profile_inference                       # auto device
    python -m tools.profile_inference --device cpu          # force CPU
    python -m tools.profile_inference --device cuda         # force GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from src.model import TimeSformerClassifier
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _resolve_device(arg: str) -> torch.device:
    """Map ``cpu`` / ``cuda`` / ``auto`` to a ``torch.device``."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Запитано --device cuda, але CUDA недоступна на цій машині."
        )
    return torch.device(arg)


def _benchmark(
    model: TimeSformerClassifier,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    num_runs: int,
) -> list[float]:
    """Run warmup + measurement; return per-call timings in seconds.

    On CUDA, ``torch.cuda.synchronize()`` brackets each measured call so
    we time real compute rather than asynchronous kernel queueing.
    """
    model.model.eval()
    is_cuda = device.type == "cuda"
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()
        for _ in range(num_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def _compute_stats(times: list[float]) -> dict[str, float]:
    """Compute mean / std / median / p90 / p99 / min / max in seconds."""
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "p99_s": float(np.percentile(arr, 99)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _print_table(
    stats: dict[str, float],
    device: torch.device,
    num_runs: int,
    warmup: int,
) -> None:
    """Print the seven summary statistics in a Ukrainian-headered table."""
    print()
    print(
        f"Підсумок latency (device={device}, прогонів={num_runs}, "
        f"прогрів={warmup}):"
    )
    print("-" * 44)
    for label, key in (
        ("mean", "mean_s"),
        ("std", "std_s"),
        ("median", "median_s"),
        ("p90", "p90_s"),
        ("p99", "p99_s"),
        ("min", "min_s"),
        ("max", "max_s"),
    ):
        print(f"  {label:<7} {stats[key]:>10.4f} с")


def _save_json(
    output_path: Path,
    device: torch.device,
    num_runs: int,
    warmup: int,
    input_shape: tuple[int, ...],
    stats: dict[str, float],
    times: list[float],
    checkpoint: Path | None,
) -> None:
    """Persist all per-run timings + metadata so thesis §4.4.6 can cite them."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "num_runs": num_runs,
        "warmup": warmup,
        "input_shape": list(input_shape),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "per_run_times_s": times,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Збережено latency-звіт у %s", output_path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    parser = argparse.ArgumentParser(
        description="Бенчмарк latency forward-проходу TimeSformer.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Пристрій (за замовч. auto: cuda за наявності, інакше cpu).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Кількість вимірювальних прогонів (за замовч. 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Кількість прогрівальних прогонів (за замовч. 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Шлях до чекпойнта (опціонально). Якщо не задано — "
            "використовуються випадкові ваги: ми міряємо швидкість "
            "forward, а не точність."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROJECT_ROOT / "evaluation_results",
        help=(
            "Куди писати latency_{device}.json (за замовч. "
            "evaluation_results/ у корені проєкту)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = _parse_args()

    device = _resolve_device(args.device)
    logger.info("Пристрій: %s", device)

    if device.type == "cpu":
        logger.warning(
            "CPU-бенчмарк представляє очікувану latency на портативному "
            "пристрої (laptop / chest-camera-class), не на серверному "
            "GPU. Для T4-чисел запустіть з --device cuda."
        )

    model = TimeSformerClassifier(device=device)
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            logger.error("Чекпойнт не знайдено: %s", args.checkpoint)
            sys.exit(1)
        model.load_checkpoint(args.checkpoint)

    input_tensor = torch.randn(
        1,
        config.CLIP_LENGTH,
        config.NUM_CHANNELS,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )
    logger.info(
        "Запуск: warmup=%d, num_runs=%d, input_shape=%s",
        args.warmup, args.num_runs, tuple(input_tensor.shape),
    )

    times = _benchmark(model, input_tensor, device, args.warmup, args.num_runs)
    stats = _compute_stats(times)
    _print_table(stats, device, args.num_runs, args.warmup)

    output_path = args.output_dir / f"latency_{device.type}.json"
    _save_json(
        output_path, device, args.num_runs, args.warmup,
        tuple(input_tensor.shape), stats, times, args.checkpoint,
    )
    print()
    print(f"Збережено: {output_path}")


if __name__ == "__main__":
    main()
