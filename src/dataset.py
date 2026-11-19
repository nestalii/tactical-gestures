"""PyTorch ``Dataset`` for tactical-gesture clips.

Layout assumed on disk::

    data/
      <class_name>/
        clip_000.mp4
        clip_001.mp4
        ...

where ``<class_name>`` is one of :data:`config.GESTURES`. Stale folders
(e.g. left over after a class rename) are detected and skipped with a
warning so they cannot silently contaminate training.

Two read paths: :mod:`decord` first (fast, batched random access),
falling back to :mod:`cv2` if it fails on a particular file. Augmentation
is applied per-clip with a **single** sampling of random parameters
that is then shared across all T frames — so temporal coherence is
preserved (the whole clip rotates by the same angle, etc.).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

import config
from src.preprocessing import ClipPreprocessor

logger = logging.getLogger(__name__)

try:
    import decord  # type: ignore[import-untyped]

    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False


def discover_clips(
    data_dir: Path = config.DATA_DIR,
) -> tuple[list[Path], list[int]]:
    """Walk ``data_dir`` and return ``(paths, labels)`` for every valid clip.

    Args:
        data_dir: Dataset root containing one folder per class.

    Returns:
        Parallel lists of absolute paths and integer class indices. Empty
        lists if ``data_dir`` is absent or has no clips in known classes.
    """
    if not data_dir.is_dir():
        return [], []

    name_to_idx = {name: i for i, name in enumerate(config.GESTURES)}
    paths: list[Path] = []
    labels: list[int] = []
    stale: dict[str, int] = {}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        clip_paths = sorted(class_dir.glob("clip_*.mp4"))
        if class_dir.name not in name_to_idx:
            if clip_paths:
                stale[class_dir.name] = len(clip_paths)
            continue
        idx = name_to_idx[class_dir.name]
        for p in clip_paths:
            paths.append(p.resolve())
            labels.append(idx)

    for name, count in sorted(stale.items()):
        logger.warning(
            "Пропущено застарілу папку: data/%s/ (%d кліп(ів)) — "
            "не у config.GESTURES",
            name,
            count,
        )

    if paths:
        counts = Counter(labels)
        breakdown = ", ".join(
            f"{config.GESTURES[i]}={counts.get(i, 0)}"
            for i in range(len(config.GESTURES))
        )
        logger.info(
            "Знайдено %d кліпів у %s. Розподіл за класами: %s",
            len(paths),
            data_dir,
            breakdown,
        )
    return paths, labels


def stratified_split(
    paths: list[Path],
    labels: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Stratified train/val/test split.

    Args:
        paths: List of clip paths.
        labels: Parallel list of class indices.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0 within 1e-6.
        seed: Fixed for reproducibility.

    Returns:
        ``{"train": (paths, labels), "val": ..., "test": ...}``.

    Raises:
        ValueError: If ratios do not sum to 1, or any class has < 1 sample.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Сума ratios має бути 1.0, отримано {total}")
    if not paths:
        raise ValueError("stratified_split: список paths порожній.")
    counts = Counter(labels)
    if min(counts.values()) < 1:
        raise ValueError("Деякі класи не мають жодного кліпу.")

    # Stratification feasibility. sklearn requires:
    #   (i)  min_count >= 2  (so each class can appear in both sides);
    #   (ii) every resulting subset has size >= n_classes (so one sample
    #        per class fits).
    n_total = len(labels)
    n_classes = len(set(labels))
    min_count = min(counts.values())
    n_test = int(round(n_total * test_ratio))
    n_temp = n_total - n_test
    inner_frac = val_ratio / (train_ratio + val_ratio)
    n_val = int(round(n_temp * inner_frac))
    n_train_remainder = n_temp - n_val

    use_stratify = (
        min_count >= 2
        and n_test >= n_classes
        and n_val >= n_classes
        and n_train_remainder >= n_classes
    )

    if not use_stratify:
        logger.warning(
            "Стратифікацію вимкнено: датасет занадто малий "
            "(%d зразків у %d класах; test=%d, val=%d, train=%d). "
            "sklearn вимагає, щоб кожна підмножина містила принаймні "
            "один зразок на клас. Рекомендовано >=7 кліпів на клас "
            "для стратифікованого розподілу при ratios 70/15/15. "
            "Використано випадковий seeded розподіл (seed=%d).",
            n_total, n_classes, n_test, n_val, n_train_remainder, seed,
        )

    stratify_first: list[int] | None = labels if use_stratify else None
    paths_temp, paths_test, labels_temp, labels_test = train_test_split(
        paths,
        labels,
        test_size=test_ratio,
        stratify=stratify_first,
        random_state=seed,
    )

    stratify_second = labels_temp if use_stratify else None
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_temp,
        labels_temp,
        test_size=inner_frac,
        stratify=stratify_second,
        random_state=seed,
    )

    split = {
        "train": (paths_train, labels_train),
        "val": (paths_val, labels_val),
        "test": (paths_test, labels_test),
    }
    for name, (_, lbls) in split.items():
        c = Counter(lbls)
        breakdown = ", ".join(
            f"{config.GESTURES[i]}={c.get(i, 0)}"
            for i in range(len(config.GESTURES))
        )
        logger.info("split[%s]: %d зразків. %s", name, len(lbls), breakdown)
    return split


def save_split(
    split: dict[str, tuple[list[Path], list[int]]],
    output_path: Path,
    seed: int | None = None,
) -> None:
    """Save a split to JSON with project-root-relative paths.

    Args:
        split: As returned by :func:`stratified_split`.
        output_path: Destination JSON file.
        seed: Optional seed used to produce the split (recorded as
            metadata; not used for verification).
    """
    root = config.PROJECT_ROOT.resolve()
    payload: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "splits": {},
    }
    total = sum(len(lbls) for _, lbls in split.values()) or 1
    payload["ratios"] = {
        name: round(len(lbls) / total, 4) for name, (_, lbls) in split.items()
    }
    for name, (paths, labels) in split.items():
        rel_paths = [
            Path(p).resolve().relative_to(root).as_posix() for p in paths
        ]
        class_counts = {
            config.GESTURES[i]: c
            for i, c in sorted(Counter(labels).items())
        }
        payload["splits"][name] = {  # type: ignore[index]
            "paths": rel_paths,
            "labels": labels,
            "class_counts": class_counts,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Збережено split у %s", output_path)


def load_split(
    input_path: Path,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Load a split previously saved by :func:`save_split`."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    root = config.PROJECT_ROOT.resolve()
    result: dict[str, tuple[list[Path], list[int]]] = {}
    for name, data in payload["splits"].items():
        paths = [(root / p).resolve() for p in data["paths"]]
        labels = list(data["labels"])
        result[name] = (paths, labels)
    logger.info("Завантажено split з %s", input_path)
    return result


def _build_augmentation_pipeline(image_size: int) -> v2.Compose:
    """Build the per-clip augmentation pipeline.

    All transforms accept a batched ``(T, 3, H, W)`` tensor and sample
    their random parameters **once per __call__**, so every frame in the
    clip is transformed identically (temporal coherence preserved). No
    horizontal/vertical flips — left/right and up/down are semantically
    meaningful for tactical gestures.
    """
    return v2.Compose(
        [
            v2.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0
            ),
            v2.RandomRotation(degrees=10),
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.95, 1.05),
                antialias=True,
            ),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        ]
    )


class TacticalGestureDataset(Dataset):
    """PyTorch Dataset for the tactical-gesture clip library.

    Reads each clip on demand inside :py:meth:`__getitem__` — does not
    cache any decoded frames in memory.

    Attributes:
        samples: ``[(absolute_path, class_idx), ...]``.
        clip_length: T, frames returned per sample.
        image_size: Spatial size used by the augmentation pipeline.
        augment: Whether augmentation is enabled.
        preprocessor: Shared :class:`ClipPreprocessor` instance.
    """

    def __init__(
        self,
        data_dir: Path = config.DATA_DIR,
        clip_paths: list[Path] | None = None,
        labels: list[int] | None = None,
        clip_length: int = config.CLIP_LENGTH,
        image_size: int = config.IMAGE_SIZE,
        augment: bool = False,
        model_name: str = config.MODEL_NAME,
    ) -> None:
        """Build the dataset in scan-mode or explicit-list mode."""
        if (clip_paths is None) != (labels is None):
            raise ValueError(
                "clip_paths та labels мають бути обидва None або обидва задані."
            )
        if clip_paths is None:
            paths, lbls = discover_clips(data_dir)
        else:
            paths, lbls = clip_paths, labels  # type: ignore[assignment]
            if len(paths) != len(lbls):
                raise ValueError(
                    f"len(clip_paths)={len(paths)} != len(labels)={len(lbls)}"
                )

        self.samples: list[tuple[Path, int]] = list(zip(paths, lbls))
        self.clip_length = clip_length
        self.image_size = image_size
        self.augment = augment
        self.preprocessor = ClipPreprocessor(model_name=model_name)
        self._aug_pipeline: v2.Compose | None = (
            _build_augmentation_pipeline(image_size) if augment else None
        )

        if self.samples:
            c = Counter(int(x[1]) for x in self.samples)
            breakdown = ", ".join(
                f"{config.GESTURES[i]}={c.get(i, 0)}"
                for i in range(len(config.GESTURES))
            )
            logger.info(
                "TacticalGestureDataset: %d зразків, augment=%s. %s",
                len(self.samples),
                augment,
                breakdown,
            )
        else:
            logger.warning("TacticalGestureDataset: жодного зразка не знайдено.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Read clip ``idx`` and return ``(tensor, label)``.

        Returns:
            ``clip_tensor`` of shape ``(T, C, H, W)``, dtype ``float32``,
            on CPU. Batching is the ``DataLoader``'s job.
            ``label`` is the class index ``0..K-1``.
        """
        path, label = self.samples[idx]
        frames_rgb = self._read_clip(path)

        if self.augment and self._aug_pipeline is not None:
            tensor = (
                torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).contiguous()
            )
            tensor = self._aug_pipeline(tensor)
            frames_rgb = tensor.permute(0, 2, 3, 1).contiguous().numpy()

        # ClipPreprocessor expects BGR uint8 (T, H, W, 3); flip last axis.
        frames_bgr = frames_rgb[..., ::-1]
        pixel_values = self.preprocessor(frames_bgr).squeeze(0)
        return pixel_values, int(label)

    def _read_clip(self, path: Path) -> np.ndarray:
        """Return RGB uint8 frames of shape ``(T, H, W, 3)``."""
        if _DECORD_AVAILABLE:
            try:
                return self._read_decord(path)
            except Exception as exc:  # noqa: BLE001 — fallback path
                logger.warning(
                    "decord не зміг прочитати %s: %s. Перехід на cv2.",
                    path,
                    exc,
                )
        return self._read_cv2(path)

    def _read_decord(self, path: Path) -> np.ndarray:
        vr = decord.VideoReader(str(path))
        n = len(vr)
        if n == 0:
            raise RuntimeError(f"{path}: 0 frames")
        indices = np.linspace(0, n - 1, self.clip_length).astype(int)
        frames = vr.get_batch(indices.tolist()).asnumpy()
        return frames  # (T, H, W, 3) RGB uint8

    def _read_cv2(self, path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 не зміг відкрити {path}")
        raw: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw.append(frame)
        finally:
            cap.release()
        if not raw:
            raise RuntimeError(f"{path}: 0 кадрів прочитано")
        indices = np.linspace(0, len(raw) - 1, self.clip_length).astype(int)
        bgr = np.stack([raw[int(i)] for i in indices])
        return np.ascontiguousarray(bgr[..., ::-1])  # BGR → RGB


if __name__ == "__main__":
    import tempfile

    from src.utils import setup_logging

    setup_logging()

    paths, labels = discover_clips()
    print(f"discover_clips: знайдено {len(paths)} кліпів")
    if not paths:
        print("Датасет порожній — пропускаю інші тести")
        raise SystemExit(0)

    per_class_counts = Counter(labels)
    missing = [
        config.GESTURES[i]
        for i in range(len(config.GESTURES))
        if per_class_counts.get(i, 0) < 1
    ]
    if missing:
        print(
            f"Не всі класи мають хоча б 1 кліп (відсутні: {missing}) — "
            f"пропускаю split-тести"
        )
        raise SystemExit(0)

    split = stratified_split(paths, labels, seed=42)
    for name, (subset_paths, subset_labels) in split.items():
        print(
            f"split[{name}]: {len(subset_paths)} зразків, "
            f"labels Counter={dict(Counter(subset_labels))}"
        )

    for name, augment in (("train", True), ("val", False), ("test", False)):
        subset_paths, subset_labels = split[name]
        if not subset_paths:
            print(f"  {name}: порожній, пропускаю __getitem__ тест")
            continue
        ds = TacticalGestureDataset(
            clip_paths=subset_paths,
            labels=subset_labels,
            augment=augment,
        )
        clip_tensor, label = ds[0]
        print(
            f"  {name}[0]: shape={tuple(clip_tensor.shape)}, "
            f"dtype={clip_tensor.dtype}, label={label} "
            f"({config.GESTURES[label]})"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = Path(f.name)
    try:
        save_split(split, tmp_path, seed=42)
        loaded = load_split(tmp_path)
        for name in ("train", "val", "test"):
            orig_paths, orig_labels = split[name]
            new_paths, new_labels = loaded[name]
            assert [str(p) for p in orig_paths] == [
                str(p) for p in new_paths
            ], f"paths mismatch in {name}"
            assert orig_labels == new_labels, f"labels mismatch in {name}"
        print("save_split + load_split roundtrip: OK")
    finally:
        tmp_path.unlink(missing_ok=True)

    print("Smoke test PASSED")
