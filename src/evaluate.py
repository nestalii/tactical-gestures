"""Evaluation script: produces every artefact needed for thesis §4.4.

Runs a trained checkpoint on the chosen split (default: ``test``) and
saves under ``--output-dir``:

- ``metrics_{subset}.json`` — full metrics dict (top-1 / top-2 accuracy,
  macro + per-class P/R/F1, raw confusion matrix);
- ``metrics_{subset}.csv``  — wide-format table for Word/LaTeX paste;
- ``training_curves.png``   — loss + accuracy curves (only if
  ``training_history.json`` sits next to the checkpoint);
- ``CM_counts.png``         — confusion matrix, raw counts;
- ``CM_normalized.png``     — confusion matrix, row-normalized;
- ``per_class_f1.png``      — horizontal bar chart sorted by F1.

Run::

    python -m src.evaluate                       # eval best_model.pt on test
    python -m src.evaluate --subset val          # sanity-check on val
    python -m src.evaluate --subset train        # sanity-check on train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from src.dataset import TacticalGestureDataset, load_split  # noqa: E402
from src.model import TimeSformerClassifier  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Mirror of tools.generate_audio.GESTURE_PHRASES, kept here so this
# module has no tools/ dependency. Keep in sync if class names ever
# change again.
GESTURE_UKRAINIAN: dict[str, str] = {
    "danger":   "Увага",
    "forward":  "Уперед",
    "come":     "До мене",
    "down":     "Лягти",
    "attack":   "Атакувати",
    "encircle": "Оточити",
    "disperse": "Розсипатися",
    "position": "Позиція",  # shortened from "Позиція противника" for plots
    "mine":     "Міна",
    "ambush":   "Засідка",
}


def _resolve_device(arg: str) -> torch.device:
    """Map ``cpu`` / ``cuda`` / ``auto`` to a ``torch.device``."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Запитано --device cuda, але CUDA недоступна на цій машині."
        )
    return torch.device(arg)


def _ukrainian_labels() -> list[str]:
    """Ukrainian label per class in ``config.GESTURES`` order."""
    return [GESTURE_UKRAINIAN[name] for name in config.GESTURES]


def _collect_predictions(
    model: TimeSformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference; return ``(all_logits (N, K), all_labels (N,))``."""
    model.model.eval()
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for clips, labels in tqdm(loader, desc="Eval", dynamic_ncols=True):
            clips = clips.to(device, non_blocking=True)
            labels_chunks.append(labels.clone())
            logits = model(clips)
            logits_chunks.append(logits.cpu())
    return (
        torch.cat(logits_chunks, dim=0),
        torch.cat(labels_chunks, dim=0),
    )


def _compute_metrics(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    subset: str,
) -> dict[str, object]:
    """Aggregate top-1 / top-2 / per-class metrics + confusion matrix."""
    k = all_logits.size(1)
    all_preds = all_logits.argmax(dim=-1)
    all_top2 = all_logits.topk(min(2, k), dim=-1).indices
    correct_top2 = (
        (all_labels.unsqueeze(-1) == all_top2).any(dim=-1).float().mean()
    )

    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()
    acc1 = float(accuracy_score(y_true, y_pred))
    acc2 = float(correct_top2.item())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(k)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))

    per_class = {
        config.GESTURES[i]: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    }
    return {
        "subset": subset,
        "n_samples": int(all_labels.numel()),
        "accuracy_top1": acc1,
        "accuracy_top2": acc2,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _save_metrics_json(metrics: dict[str, object], path: Path) -> None:
    """Write the metrics dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_metrics_csv(metrics: dict[str, object], path: Path) -> None:
    """Write a wide-format CSV: one row per class, plus macro + accuracy rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    per_class = metrics["per_class"]  # type: ignore[index]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "precision", "recall", "f1", "support"])
        total_support = 0
        for name in config.GESTURES:
            stats = per_class[name]  # type: ignore[index]
            writer.writerow([
                name,
                f"{stats['precision']:.4f}",
                f"{stats['recall']:.4f}",
                f"{stats['f1']:.4f}",
                int(stats["support"]),
            ])
            total_support += int(stats["support"])
        writer.writerow([
            "macro",
            f"{metrics['macro_precision']:.4f}",  # type: ignore[index]
            f"{metrics['macro_recall']:.4f}",  # type: ignore[index]
            f"{metrics['macro_f1']:.4f}",  # type: ignore[index]
            total_support,
        ])
        writer.writerow(
            ["accuracy_top1", "", "", f"{metrics['accuracy_top1']:.4f}", ""]  # type: ignore[index]
        )
        writer.writerow(
            ["accuracy_top2", "", "", f"{metrics['accuracy_top2']:.4f}", ""]  # type: ignore[index]
        )


def _plot_training_curves(history_path: Path, output_path: Path) -> None:
    """Two-panel figure: loss curves and accuracy curves vs epoch."""
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    best_epoch = payload.get("best_epoch")
    if not history:
        raise RuntimeError("training_history.json порожня")

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Втрата")
    ax_loss.set_title("Криві втрат")
    ax_loss.legend()
    ax_loss.grid(True, linestyle=":", alpha=0.5)

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set_xlabel("Епоха")
    ax_acc.set_ylabel("Точність")
    ax_acc.set_title("Криві точності")
    ax_acc.legend()
    ax_acc.grid(True, linestyle=":", alpha=0.5)

    if best_epoch:
        for ax in (ax_loss, ax_acc):
            ax.axvline(
                int(best_epoch), linestyle="--", color="gray",
                label=f"best epoch={best_epoch}",
            )

    fig.suptitle("Криві навчання")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    normalize: bool,
) -> None:
    """Render the confusion matrix as a seaborn heatmap (counts or row %)."""
    if normalize:
        cm_data = cm.astype(np.float64)
        row_sums = cm_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_data = cm_data / row_sums
        fmt = ".2f"
        title = "Матриця помилок (норм. за рядками)"
    else:
        cm_data = cm
        fmt = "d"
        title = "Матриця помилок (підрахунки)"

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        cm_data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=True, square=True, ax=ax,
    )
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_f1(
    per_class_f1: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """Horizontal bar chart of per-class F1, sorted descending."""
    order = np.argsort(per_class_f1)[::-1]
    sorted_labels = [labels[i] for i in order]
    sorted_f1 = per_class_f1[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_labels)))
    bars = ax.barh(range(len(sorted_labels)), sorted_f1, color=colors)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1")
    ax.set_title("F1 за класами")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    for bar, value in zip(bars, sorted_f1):
        ax.text(
            value + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    metrics: dict[str, object],
    saved: list[Path],
    output_dir: Path,
) -> None:
    """Pretty-print all metrics + list of saved artefacts."""
    per_class = metrics["per_class"]  # type: ignore[index]
    print()
    print("=" * 64)
    print(
        f"Результати оцінки на subset='{metrics['subset']}' "  # type: ignore[index]
        f"({metrics['n_samples']} зразків):"  # type: ignore[index]
    )
    print("=" * 64)
    print(f"  accuracy@1:        {metrics['accuracy_top1']:.4f}")  # type: ignore[index]
    print(f"  accuracy@2:        {metrics['accuracy_top2']:.4f}")  # type: ignore[index]
    print(f"  macro precision:   {metrics['macro_precision']:.4f}")  # type: ignore[index]
    print(f"  macro recall:      {metrics['macro_recall']:.4f}")  # type: ignore[index]
    print(f"  macro F1:          {metrics['macro_f1']:.4f}")  # type: ignore[index]
    print()
    print(f"{'class':<10}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>7}")
    print("-" * 42)
    for name in config.GESTURES:
        stats = per_class[name]  # type: ignore[index]
        print(
            f"{name:<10}  {stats['precision']:>6.4f}  {stats['recall']:>6.4f}  "
            f"{stats['f1']:>6.4f}  {int(stats['support']):>7}"
        )
    print()
    print(f"Артефакти у {output_dir}/:")
    for p in saved:
        print(f"  - {p.name}")
    print()
    print(f"Результати збережено у {output_dir}/")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    default_workers = 0 if sys.platform == "win32" else 2
    parser = argparse.ArgumentParser(
        description="Оцінка чекпойнта на test/val/train split-і.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINTS_DIR / "best_model.pt",
        help=(
            "Шлях до чекпойнта моделі (за замовч. "
            "config.CHECKPOINTS_DIR/best_model.pt)."
        ),
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=config.PROJECT_ROOT / "data_split.json",
        help=(
            "Шлях до JSON зі split-розбиттям (за замовч. "
            "data_split.json у корені проєкту)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROJECT_ROOT / "evaluation_results",
        help=(
            "Куди зберігати метрики, CSV та графіки (за замовч. "
            "evaluation_results/ у корені проєкту)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Розмір батчу (за замовч. 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help=(
            "Кількість DataLoader-воркерів (за замовч. 0 на Windows, "
            "2 на інших ОС)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help=(
            "Пристрій для інференсу (за замовч. auto: cuda за наявності, "
            "інакше cpu)."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["test", "val", "train"],
        default="test",
        help=(
            "Яку підмножину оцінювати. 'test' — головна метрика для "
            "тези; 'train'/'val' — діагностика."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = _parse_args()

    if not args.checkpoint.is_file():
        logger.error("Чекпойнт не знайдено: %s", args.checkpoint)
        sys.exit(1)
    if not args.split_path.is_file():
        logger.error(
            "Файл розбіття не знайдено: %s. Спочатку запустіть "
            "python -m src.train, щоб створити split.",
            args.split_path,
        )
        sys.exit(1)

    device = _resolve_device(args.device)
    logger.info("Пристрій: %s", device)

    split = load_split(args.split_path)
    if args.subset not in split:
        logger.error(
            "Невідомий subset: %s. Доступні: %s",
            args.subset, sorted(split.keys()),
        )
        sys.exit(1)
    paths, labels = split[args.subset]
    if not paths:
        logger.error(
            "Підмножина '%s' порожня — нічого оцінювати.", args.subset,
        )
        sys.exit(1)

    dataset = TacticalGestureDataset(
        clip_paths=paths, labels=labels, augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TimeSformerClassifier(device=device)
    model.load_checkpoint(args.checkpoint)

    all_logits, all_labels = _collect_predictions(model, loader, device)
    metrics = _compute_metrics(all_logits, all_labels, subset=args.subset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    metrics_json = args.output_dir / f"metrics_{args.subset}.json"
    _save_metrics_json(metrics, metrics_json)
    saved.append(metrics_json)

    metrics_csv = args.output_dir / f"metrics_{args.subset}.csv"
    _save_metrics_csv(metrics, metrics_csv)
    saved.append(metrics_csv)

    history_path = args.checkpoint.parent / "training_history.json"
    if history_path.is_file():
        curves_path = args.output_dir / "training_curves.png"
        try:
            _plot_training_curves(history_path, curves_path)
            saved.append(curves_path)
        except Exception as exc:  # noqa: BLE001 — plotting must be resilient
            logger.warning("Не вдалось згенерувати training_curves.png: %s", exc)
    else:
        logger.warning(
            "training_history.json не знайдено поряд з чекпойнтом — "
            "пропускаю training_curves.png."
        )

    cm_array = np.array(metrics["confusion_matrix"])  # type: ignore[index]
    uk_labels = _ukrainian_labels()
    for name, normalize in (("CM_counts.png", False), ("CM_normalized.png", True)):
        out_path = args.output_dir / name
        try:
            _plot_confusion_matrix(cm_array, uk_labels, out_path, normalize=normalize)
            saved.append(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не вдалось згенерувати %s: %s", name, exc)

    per_class_f1 = np.array(
        [metrics["per_class"][n]["f1"] for n in config.GESTURES]  # type: ignore[index]
    )
    f1_path = args.output_dir / "per_class_f1.png"
    try:
        _plot_per_class_f1(per_class_f1, uk_labels, f1_path)
        saved.append(f1_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалось згенерувати per_class_f1.png: %s", exc)

    _print_summary(metrics, saved, args.output_dir)


if __name__ == "__main__":
    main()
