"""5-fold stratified cross-validation for the TimeSformer classifier.

Trains 5 independent models on disjoint 80/20 splits and aggregates
accuracy with mean ± std plus a 95% CI. Reuses the per-epoch training
loop, optimizer setup, AMP, early stopping and checkpoint helpers from
:mod:`src.train`; only the fold-driven outer loop and the cross-fold
aggregation live here.

Runtime estimate: ~5 × (20 epochs) on a T4 ≈ 1.5–2 hours total for the
full dataset. Fits in a free Colab session (90-min idle disconnect,
12-hour hard cap).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import t as scipy_t  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from src.dataset import discover_clips  # noqa: E402
from src.model import TimeSformerClassifier  # noqa: E402
from src.train import (  # noqa: E402
    _build_dataloaders,
    _resolve_device,
    _save_checkpoint,
    _train_one_epoch,
    _validate,
    set_seed,
)
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

N_SPLITS = 5


def _reseed(seed: int) -> None:
    """Reset RNGs at fold boundaries without re-logging the determinism warning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_fold_indices(labels: list[int], n_splits: int, seed: int):
    """Yield (fold_idx, train_indices, val_indices) per StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(labels))
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dummy_x, labels)):
        yield fold_idx, train_idx, val_idx


def _eval_fold(
    model: TimeSformerClassifier,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Run inference and compute metrics on the fold's validation split."""
    model.model.eval()
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for clips, labels in val_loader:
            clips = clips.to(device, non_blocking=True)
            labels_chunks.append(labels.clone())
            logits_chunks.append(model(clips).cpu())
    all_logits = torch.cat(logits_chunks, dim=0)
    all_labels = torch.cat(labels_chunks, dim=0)
    y_pred = all_logits.argmax(dim=-1).numpy()
    y_true = all_labels.numpy()

    accuracy = float((y_pred == y_true).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            config.GESTURES[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }


def _run_one_fold(
    fold_idx: int,
    train_paths: list[Path],
    train_labels: list[int],
    val_paths: list[Path],
    val_labels: list[int],
    *,
    args: argparse.Namespace,
    device: torch.device,
    hyperparameters: dict,
) -> dict:
    """Train one fold from scratch; save artefacts; return val metrics."""
    fold_dir = args.output_dir / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    _reseed(args.seed)

    split = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
    }
    train_loader, val_loader = _build_dataloaders(
        split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        pin_memory=(device.type == "cuda"),
    )

    model = TimeSformerClassifier(device=device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            head_lr=config.HEAD_LR, encoder_lr=config.ENCODER_LR,
        ),
        weight_decay=config.WEIGHT_DECAY,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    n_train, n_val = len(train_paths), len(val_paths)
    total = n_train + n_val
    ratios = {"train": n_train / total, "val": n_val / total}

    best_path = fold_dir / "best_model.pt"
    history_path = fold_dir / "training_history.json"
    metrics_path = fold_dir / "metrics.json"

    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    try:
        for epoch in range(args.epochs):
            epoch_idx = epoch + 1
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                scaler, device, epoch_idx, args.epochs,
            )
            val_loss, val_acc = _validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Fold %d, Епоха %d: train_loss=%.4f, train_acc=%.4f, "
                "val_loss=%.4f, val_acc=%.4f, lr=%.2e",
                fold_idx + 1, epoch_idx,
                train_loss, train_acc, val_loss, val_acc, current_lr,
            )
            history.append({
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch_idx
                early_stop_counter = 0
                _save_checkpoint(
                    best_path, model,
                    epoch=epoch_idx, val_acc=val_acc, ratios=ratios,
                    seed=args.seed, hyperparameters=hyperparameters,
                    history=history,
                )
                logger.info(
                    "Fold %d: збережено найкращу модель val_acc=%.4f (епоха %d).",
                    fold_idx + 1, val_acc, epoch_idx,
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    logger.info(
                        "Fold %d: рання зупинка на епосі %d.",
                        fold_idx + 1, epoch_idx,
                    )
                    break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA OOM під час fold %d: %s. "
                "Спробуйте --batch-size 2 або 1.",
                fold_idx + 1, exc,
            )
            sys.exit(1)
        raise

    history_path.write_text(json.dumps({
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    # Reload best checkpoint for evaluation (may differ from last epoch)
    model.load_checkpoint(best_path)
    metrics = _eval_fold(model, val_loader, device, config.NUM_CLASSES)
    metrics["fold_idx"] = fold_idx + 1
    metrics["best_epoch"] = best_epoch
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return metrics


def _aggregate(fold_metrics: list[dict], num_classes: int) -> dict:
    """Compute mean / std / 95% CI across folds plus per-class spread."""
    n = len(fold_metrics)
    accs = np.array([m["accuracy"] for m in fold_metrics])
    macro_f1s = np.array([m["macro_f1"] for m in fold_metrics])

    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1)) if n > 1 else 0.0

    # t-distribution 95% CI — for n=5 this gives a wider, more honest
    # interval than the normal-approximation 1.96·SEM.
    if n > 1:
        t_crit = float(scipy_t.ppf(0.975, df=n - 1))
        sem = std_acc / np.sqrt(n)
        ci_lower = mean_acc - t_crit * sem
        ci_upper = mean_acc + t_crit * sem
    else:
        ci_lower = ci_upper = mean_acc

    per_class_f1 = np.array([
        [m["per_class"][config.GESTURES[c]]["f1"] for c in range(num_classes)]
        for m in fold_metrics
    ])

    return {
        "n_folds": n,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "mean_macro_f1": float(macro_f1s.mean()),
        "std_macro_f1": float(macro_f1s.std(ddof=1)) if n > 1 else 0.0,
        "mean_per_class_f1": {
            config.GESTURES[c]: {
                "mean_f1": float(per_class_f1[:, c].mean()),
                "std_f1": float(per_class_f1[:, c].std(ddof=1)) if n > 1 else 0.0,
            }
            for c in range(num_classes)
        },
    }


def _plot_summary(
    fold_metrics: list[dict],
    aggregate: dict,
    output_path: Path,
) -> None:
    """Bar chart of per-fold accuracy with mean line and ±1 σ band."""
    accs = [m["accuracy"] for m in fold_metrics]
    folds = [f"Fold {m['fold_idx']}" for m in fold_metrics]
    mean = aggregate["mean_accuracy"]
    std = aggregate["std_accuracy"]
    ci_lo = aggregate["ci_95_lower"]
    ci_hi = aggregate["ci_95_upper"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(folds, accs, color="#4C72B0", edgecolor="#2B4868")
    ax.axhline(
        mean, color="#DD8452", linestyle="--", label=f"Mean = {mean:.3f}",
    )
    ax.fill_between(
        [-0.5, len(folds) - 0.5],
        mean - std, mean + std,
        color="#DD8452", alpha=0.15, label=f"±1 σ ({std:.3f})",
    )
    ax.set_xlim(-0.5, len(folds) - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Fold")
    ax.set_title(
        f"5-fold cross-validation: {mean:.3f} ± {std:.3f} "
        f"(95% CI [{ci_lo:.3f}, {ci_hi:.3f}])"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.015,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(fold_metrics: list[dict], aggregate: dict) -> None:
    """Print per-fold and aggregate tables to stdout."""
    print()
    print("=" * 76)
    print("Підсумок 5-fold cross-validation")
    print("=" * 76)
    print()

    header = (
        f"{'Fold':>4}  {'n_train':>7}  {'n_val':>5}  "
        f"{'Accuracy':>8}  {'Macro F1':>8}  {'Best epoch':>10}"
    )
    print(header)
    print("-" * len(header))
    for m in fold_metrics:
        print(
            f"{m['fold_idx']:>4}  {m['n_train']:>7}  {m['n_val']:>5}  "
            f"{m['accuracy']:>8.4f}  {m['macro_f1']:>8.4f}  "
            f"{m['best_epoch']:>10}"
        )
    print("-" * len(header))
    print(
        f"{'mean':>4}  {'':>7}  {'':>5}  "
        f"{aggregate['mean_accuracy']:>8.4f}  "
        f"{aggregate['mean_macro_f1']:>8.4f}"
    )
    print(
        f"{'std':>4}  {'':>7}  {'':>5}  "
        f"{aggregate['std_accuracy']:>8.4f}  "
        f"{aggregate['std_macro_f1']:>8.4f}"
    )
    print()
    print(
        f"Aggregate accuracy: {aggregate['mean_accuracy']:.4f} ± "
        f"{aggregate['std_accuracy']:.4f}"
    )
    print(
        f"95% CI (t-distribution): "
        f"[{aggregate['ci_95_lower']:.4f}, {aggregate['ci_95_upper']:.4f}]"
    )
    print()

    print("F1 за класами (середнє ± стд по 5 folds):")
    pc = aggregate["mean_per_class_f1"]
    sorted_by_f1 = sorted(pc.items(), key=lambda kv: kv[1]["mean_f1"])
    print(f"{'class':<12}  {'mean F1':>8}  {'std F1':>8}")
    print("-" * 34)
    for name, stats in sorted_by_f1:
        print(
            f"{name:<12}  {stats['mean_f1']:>8.4f}  {stats['std_f1']:>8.4f}"
        )

    worst = sorted_by_f1[0]
    best = sorted_by_f1[-1]
    print()
    print(f"Найгірший клас: {worst[0]} (mean F1 = {worst[1]['mean_f1']:.4f})")
    print(f"Найкращий клас: {best[0]} (mean F1 = {best[1]['mean_f1']:.4f})")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Help text is in Ukrainian."""
    default_workers = 0 if sys.platform == "win32" else 2
    parser = argparse.ArgumentParser(
        description=(
            "5-fold stratified cross-validation для TimeSformer-класифікатора."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Каталог з даними (за замовч. config.DATA_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROJECT_ROOT / "kfold_results",
        help=(
            "Каталог для fold-чекпойнтів і підсумку "
            "(за замовч. kfold_results/ у корені проєкту)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Максимальна кількість епох на fold (за замовч. 20).",
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
            "Кількість DataLoader-воркерів "
            "(за замовч. 0 на Windows, 2 на інших ОС)."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help=(
            "Зупинити fold, якщо val_acc не покращується стільки епох "
            "поспіль (за замовч. 5)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed для StratifiedKFold і ініціалізації моделі (за замовч. 42)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Пристрій для тренування (за замовч. auto).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging()
    args = _parse_args()
    set_seed(args.seed)

    device = _resolve_device(args.device)
    logger.info("Пристрій: %s", device)
    logger.warning(
        "5-fold CV: ~%d × (до %d епох) на T4 ≈ 1.5-2 години для повного "
        "датасету. Впишемось у вільну сесію Colab (90-хв idle, 12-год hard cap).",
        N_SPLITS, args.epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = discover_clips(args.data_dir)
    if not paths:
        logger.error("Датасет порожній: %s", args.data_dir)
        sys.exit(1)

    n_total = len(paths)
    counts = Counter(labels)
    breakdown = ", ".join(
        f"{config.GESTURES[c]}={counts.get(c, 0)}"
        for c in range(config.NUM_CLASSES)
    )
    logger.info("Знайдено %d кліпів. Розподіл: %s", n_total, breakdown)

    min_count = min(counts.values()) if counts else 0
    if min_count < N_SPLITS:
        logger.error(
            "Найменший клас має лише %d кліпів — потрібно >= %d для %d-fold CV.",
            min_count, N_SPLITS, N_SPLITS,
        )
        sys.exit(1)

    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "head_lr": config.HEAD_LR,
        "encoder_lr": config.ENCODER_LR,
        "weight_decay": config.WEIGHT_DECAY,
        "label_smoothing": config.LABEL_SMOOTHING,
        "unfrozen_blocks": config.UNFROZEN_BLOCKS,
        "augment": True,
        "device": str(device),
    }

    fold_metrics: list[dict] = []
    t_start = datetime.now()

    for fold_idx, train_idx, val_idx in _make_fold_indices(
        labels, n_splits=N_SPLITS, seed=args.seed,
    ):
        logger.info("=" * 60)
        logger.info(
            "Fold %d/%d: n_train=%d, n_val=%d",
            fold_idx + 1, N_SPLITS, len(train_idx), len(val_idx),
        )
        logger.info("=" * 60)

        train_paths = [paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        metrics = _run_one_fold(
            fold_idx, train_paths, train_labels, val_paths, val_labels,
            args=args, device=device, hyperparameters=hyperparameters,
        )
        fold_metrics.append(metrics)

        elapsed = datetime.now() - t_start
        avg_per_fold = elapsed / (fold_idx + 1)
        remaining = avg_per_fold * (N_SPLITS - fold_idx - 1)
        logger.info(
            "Fold %d/%d завершено. Elapsed: %s. ETA до завершення: %s.",
            fold_idx + 1, N_SPLITS,
            timedelta(seconds=int(elapsed.total_seconds())),
            timedelta(seconds=int(remaining.total_seconds())),
        )

    aggregate = _aggregate(fold_metrics, config.NUM_CLASSES)

    summary = {
        "config": {
            "n_splits": N_SPLITS,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stop_patience": args.early_stop_patience,
            **{k: v for k, v in hyperparameters.items()
               if k not in {"epochs", "batch_size"}},
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_total_clips": n_total,
        "per_fold": [
            {
                "fold_idx": m["fold_idx"],
                "best_val_acc": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "per_class_f1": {
                    n: stats["f1"] for n, stats in m["per_class"].items()
                },
                "n_train": m["n_train"],
                "n_val": m["n_val"],
                "best_epoch": m["best_epoch"],
            }
            for m in fold_metrics
        ],
        "aggregate": aggregate,
    }
    summary_path = args.output_dir / "cv_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Збережено підсумок у %s", summary_path)

    plot_path = args.output_dir / "cv_summary.png"
    try:
        _plot_summary(fold_metrics, aggregate, plot_path)
        logger.info("Збережено графік у %s", plot_path)
    except Exception as exc:  # noqa: BLE001 — plot failure must not kill the summary
        logger.warning("Не вдалось згенерувати cv_summary.png: %s", exc)

    _print_summary(fold_metrics, aggregate)


if __name__ == "__main__":
    main()
