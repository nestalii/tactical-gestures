"""Training script for :class:`src.model.TimeSformerClassifier`.

Designed for two execution contexts:

(a) Local Windows laptop (CPU): smoke-test that the loop runs end-to-end
    on a tiny dataset. Typical invocation::

        python -m src.train --epochs 1 --batch-size 2 --num-workers 0

(b) Google Colab T4 GPU: real training on the full dataset, with
    checkpoints + training history persisted to a path on Google Drive.

The script saves three artefacts under ``--checkpoint-dir``:

- ``best_model.pt``       — weights + training metadata at the epoch of best val_acc
- ``last_model.pt``       — weights at the final epoch (always)
- ``training_history.json`` — per-epoch losses and accuracies for thesis §4.4 plots
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import (
    TacticalGestureDataset,
    discover_clips,
    load_split,
    save_split,
    stratified_split,
)
from src.model import TimeSformerClassifier
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Make all RNGs deterministic (slower but reproducible).

    Affects :mod:`random`, :mod:`numpy`, :mod:`torch`, CUDA RNGs, and
    cuDNN. Logs a single WARN per call so the user is reminded that
    benchmark-mode kernel selection is disabled.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
        "Детермінований режим увімкнено (seed=%d): cudnn.benchmark=False — "
        "повільніше, але результати повторюються.",
        seed,
    )


def _resolve_device(arg: str) -> torch.device:
    """Map ``cpu`` / ``cuda`` / ``auto`` to a ``torch.device``."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Запитано --device cuda, але CUDA недоступна. "
            "Перевірте інсталяцію torch+cu*."
        )
    return torch.device(arg)


def _load_or_build_split(
    data_dir: Path,
    split_path: Path,
    seed: int,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Read the split JSON if it exists; otherwise discover + split + save."""
    if split_path.is_file():
        logger.info("Завантажую split з %s", split_path)
        return load_split(split_path)

    logger.info("Split-файл відсутній; будую новий з %s", data_dir)
    paths, labels = discover_clips(data_dir)
    if not paths:
        raise RuntimeError(
            "Тренувальна вибірка порожня — спочатку запишіть кліпи через "
            "tools.record_dataset."
        )
    split = stratified_split(paths, labels, seed=seed)
    save_split(split, split_path, seed=seed)
    return split


def _build_dataloaders(
    split: dict[str, tuple[list[Path], list[int]]],
    batch_size: int,
    num_workers: int,
    augment: bool,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Build the train and val DataLoaders. Test set is reserved for evaluate.py."""
    if num_workers > 0 and sys.platform == "win32":
        logger.warning(
            "Windows + num_workers=%d: можливі проблеми з pickling. "
            "Якщо тренування зависає — спробуйте --num-workers 0.",
            num_workers,
        )

    train_paths, train_labels = split["train"]
    val_paths, val_labels = split["val"]
    if not train_paths:
        raise RuntimeError(
            "Тренувальний підмножина порожня — перевірте data/ і split-файл."
        )

    train_ds = TacticalGestureDataset(
        clip_paths=train_paths, labels=train_labels, augment=augment,
    )
    val_ds = TacticalGestureDataset(
        clip_paths=val_paths, labels=val_labels, augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(
        "DataLoader: train=%d зразків (batches=%d), val=%d зразків (batches=%d).",
        len(train_ds), len(train_loader), len(val_ds), len(val_loader),
    )
    return train_loader, val_loader


def _save_checkpoint(
    path: Path,
    model: TimeSformerClassifier,
    *,
    epoch: int,
    val_acc: float,
    ratios: dict[str, float],
    seed: int,
    hyperparameters: dict[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    history: list[dict[str, float]] | None = None,
) -> None:
    """Save a training checkpoint with full metadata for later resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "state_dict": model.model.state_dict(),
        "config": {
            "model_name": model.model_name,
            "num_classes": model.num_classes,
            "unfrozen_blocks": model.unfrozen_blocks,
        },
        "training": {
            "epoch": epoch,
            "val_acc": val_acc,
            "ratios": ratios,
            "seed": seed,
            "hyperparameters": hyperparameters,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def _save_history(
    path: Path,
    history: list[dict[str, float]],
    best_epoch: int,
    best_val_acc: float,
) -> None:
    """Persist per-epoch training metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Збережено історію навчання у %s", path)


def _print_final_summary(
    history: list[dict[str, float]],
    best_epoch: int,
    best_val_acc: float,
) -> None:
    """Print the per-epoch summary table with a marker on the best epoch."""
    print()
    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}"
    print(header)
    print("-" * len(header))
    for h in history:
        marker = " *" if int(h["epoch"]) == best_epoch else ""
        print(
            f"{int(h['epoch']):>5}  {h['train_loss']:>10.4f}  "
            f"{h['train_acc']:>9.4f}  {h['val_loss']:>8.4f}  "
            f"{h['val_acc']:>7.4f}{marker}"
        )
    print()
    print(f"Best epoch: {best_epoch}, best val acc: {best_val_acc:.4f}")


def _train_one_epoch(
    model: TimeSformerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch_idx: int,
    epochs: int,
) -> tuple[float, float]:
    """Run one training epoch; return ``(avg_loss, accuracy)``."""
    model.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader, desc=f"Epoch {epoch_idx}/{epochs}", leave=False, dynamic_ncols=True,
    )
    for batch_idx, (clips, labels) in enumerate(pbar):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            batch_correct = int((preds == labels).sum().item())
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += batch_correct
        total += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_correct / batch_size:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if batch_idx % 10 == 0:
            logger.debug(
                "epoch=%d batch=%d lr=%.2e batch_loss=%.4f batch_acc=%.3f",
                epoch_idx, batch_idx, optimizer.param_groups[0]["lr"],
                loss.item(), batch_correct / batch_size,
            )

    return total_loss / total, correct / total


def _validate(
    model: TimeSformerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation pass; return ``(avg_loss, accuracy)``."""
    model.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)
    with torch.inference_mode():
        for clips, labels in pbar:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(clips)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((preds == labels).sum().item())
            total += batch_size
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total, correct / total


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training. Help text is in Ukrainian."""
    parser = argparse.ArgumentParser(
        description="Тренування TimeSformer на датасеті тактичних жестів.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Каталог з даними (за замовч. config.DATA_DIR).",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=config.PROJECT_ROOT / "data_split.json",
        help=(
            "Шлях до JSON зі split-розбиттям. Якщо існує — завантажується; "
            "якщо ні — будується і зберігається (за замовч. data_split.json "
            "у корені проєкту)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=config.CHECKPOINTS_DIR,
        help=(
            "Куди зберігати best_model.pt / last_model.pt / "
            "training_history.json (за замовч. config.CHECKPOINTS_DIR)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Максимальна кількість епох (за замовч. 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Розмір батчу (за замовч. 4). TimeSformer пам'ятежерний — "
            "знизьте до 2 або 1 при CUDA OOM."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help=(
            "Кількість DataLoader-воркерів (за замовч. 2 для Colab/Linux). "
            "На Windows використовуйте 0, щоб уникнути проблем з pickling."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help=(
            "Зупинити, якщо val_acc не покращується стільки епох поспіль "
            "(за замовч. 5)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Початкове значення RNG для відтворюваності (за замовч. 42).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help=(
            "Пристрій для тренування (за замовч. auto: cuda за наявності, "
            "інакше cpu)."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Шлях до чекпойнта для продовження тренування. Стан "
            "оптимізатора, шедулера та історії відновлюються."
        ),
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help=(
            "Вимкнути аугментацію тренувальної вибірки "
            "(тільки для дебагу)."
        ),
    )
    parser.set_defaults(augment=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = _parse_args()
    set_seed(args.seed)

    device = _resolve_device(args.device)
    logger.info("Пристрій: %s", device)

    split = _load_or_build_split(args.data_dir, args.split_path, seed=args.seed)
    total = sum(len(lbls) for _, lbls in split.values()) or 1
    ratios = {name: len(lbls) / total for name, (_, lbls) in split.items()}

    train_loader, val_loader = _build_dataloaders(
        split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
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

    start_epoch = 0
    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    if args.resume is not None:
        logger.info("Відновлення з чекпойнта: %s", args.resume)
        model.load_checkpoint(args.resume)
        payload = torch.load(args.resume, map_location=device)
        training_meta = payload.get("training", {})
        start_epoch = int(training_meta.get("epoch", 0))
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if "history" in payload:
            history = list(payload["history"])
            if history:
                best = max(history, key=lambda h: h["val_acc"])
                best_val_acc = float(best["val_acc"])
                best_epoch = int(best["epoch"])
        logger.info(
            "Відновлено. Продовжуємо з епохи %d, best_val_acc=%.4f.",
            start_epoch + 1, best_val_acc,
        )

    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "head_lr": config.HEAD_LR,
        "encoder_lr": config.ENCODER_LR,
        "weight_decay": config.WEIGHT_DECAY,
        "label_smoothing": config.LABEL_SMOOTHING,
        "augment": args.augment,
        "device": str(device),
    }

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best_model.pt"
    last_path = args.checkpoint_dir / "last_model.pt"
    history_path = args.checkpoint_dir / "training_history.json"

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_idx = epoch + 1
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                scaler, device, epoch_idx, args.epochs,
            )
            val_loss, val_acc = _validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Епоха %d: train_loss=%.4f, train_acc=%.4f, "
                "val_loss=%.4f, val_acc=%.4f, lr=%.2e",
                epoch_idx, train_loss, train_acc, val_loss, val_acc, current_lr,
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
                    optimizer=optimizer, scheduler=scheduler, history=history,
                )
                logger.info(
                    "Збережено найкращу модель: val_acc=%.4f (епоха %d).",
                    val_acc, epoch_idx,
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    logger.info(
                        "Рання зупинка: val_acc не покращується %d "
                        "епох поспіль.",
                        args.early_stop_patience,
                    )
                    break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA OOM під час навчання: %s. "
                "Рекомендовано знизити --batch-size до 2 або 1, або "
                "обрати меншу архітектуру.", exc,
            )
            sys.exit(1)
        raise

    _save_checkpoint(
        last_path, model,
        epoch=history[-1]["epoch"] if history else 0,
        val_acc=history[-1]["val_acc"] if history else 0.0,
        ratios=ratios, seed=args.seed, hyperparameters=hyperparameters,
        optimizer=optimizer, scheduler=scheduler, history=history,
    )
    _save_history(history_path, history, best_epoch, best_val_acc)

    if history:
        _print_final_summary(history, best_epoch, best_val_acc)
    else:
        logger.warning("Жодної епохи не виконано — нема що зберігати.")


if __name__ == "__main__":
    main()
