"""TimeSformer wrapper for tactical-gesture classification.

Implements the transfer-learning recipe from thesis §3.4:

- start from Kinetics-400 pre-trained weights
  (``facebook/timesformer-base-finetuned-k400``);
- replace the K=400 classification head with a fresh
  ``num_classes`` head (``num_classes = NUM_CLASSES`` by default);
- freeze the first ``L − L'`` encoder blocks;
- train only the last ``L'`` encoder blocks, the final
  ``LayerNorm`` before the head, and the new head itself.

Also exposes :py:meth:`parameter_groups` for AdamW with discriminative
learning rates (head : encoder ≈ 10 : 1), and ``save_checkpoint`` /
``load_checkpoint`` with a config-mismatch guard so a stray K=10
checkpoint can never be silently loaded into a differently-configured
model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from transformers import TimesformerForVideoClassification

import config

logger = logging.getLogger(__name__)


class TimeSformerClassifier(nn.Module):
    """TimeSformer-Base fine-tuned for K tactical gestures.

    Attributes:
        model_name: HuggingFace identifier of the source checkpoint.
        num_classes: K, the number of output classes.
        unfrozen_blocks: L', the number of encoder blocks left trainable
            at the tail of the encoder.
        device: ``torch.device`` the model lives on.
        model: Underlying ``TimesformerForVideoClassification``.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        unfrozen_blocks: int = config.UNFROZEN_BLOCKS,
        device: torch.device | None = None,
    ) -> None:
        """Load the backbone, reinitialize the head, freeze the early blocks.

        Args:
            model_name: HuggingFace model identifier.
            num_classes: K, the number of gesture classes.
            unfrozen_blocks: L', number of encoder blocks at the tail
                of the encoder that stay trainable.
            device: Target device. Defaults to CUDA if available, else
                CPU.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.unfrozen_blocks = unfrozen_blocks
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=0.05,
            attention_probs_dropout_prob=0.05,
            drop_path_rate=0.05,
        )
        logger.info(
            "Завантажено TimeSformer (%s); голову переініціалізовано на %d класів",
            model_name,
            num_classes,
        )

        self._freeze_backbone(unfrozen_blocks)
        self.model.to(self.device)
        logger.info("Модель переміщено на пристрій: %s", self.device)

    def _freeze_backbone(self, unfrozen_blocks: int) -> None:
        """Freeze the first ``L − unfrozen_blocks`` encoder blocks.

        Strategy: freeze everything, then explicitly unfreeze (a) the
        last ``unfrozen_blocks`` encoder blocks, (b) the top-level
        ``LayerNorm`` that sits between the encoder and the head, and
        (c) the classifier head itself. This avoids fragile
        name-substring matching and is robust to future HF refactors.

        Args:
            unfrozen_blocks: Number of encoder blocks at the tail to
                keep trainable.

        Raises:
            ValueError: If ``unfrozen_blocks`` exceeds the number of
                encoder blocks in the model.
        """
        layers = self.model.timesformer.encoder.layer
        total_blocks = len(layers)
        if unfrozen_blocks > total_blocks:
            raise ValueError(
                f"unfrozen_blocks={unfrozen_blocks} перевищує загальну "
                f"кількість блоків {total_blocks}"
            )
        frozen_count = total_blocks - unfrozen_blocks

        for p in self.model.parameters():
            p.requires_grad = False

        for block in list(layers)[frozen_count:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in self.model.timesformer.layernorm.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            "Заморожено %d / %d блоків. Тренованих параметрів: %d (%.2f%%), "
            "заморожених: %d (%.2f%%), загалом: %d",
            frozen_count,
            total_blocks,
            trainable,
            100.0 * trainable / total,
            frozen,
            100.0 * frozen / total,
            total,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of clips.

        Args:
            pixel_values: Tensor of shape
                ``(B, T, C, H, W) = (B, CLIP_LENGTH, NUM_CHANNELS,
                IMAGE_SIZE, IMAGE_SIZE)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.

        Raises:
            ValueError: If ``pixel_values`` has the wrong rank or wrong
                trailing dimensions.
        """
        expected_tail = (
            config.CLIP_LENGTH,
            config.NUM_CHANNELS,
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        )
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Очікувано ndim=5 (B, T, C, H, W), отримано "
                f"ndim={pixel_values.ndim} з shape={tuple(pixel_values.shape)}"
            )
        if tuple(pixel_values.shape[1:]) != expected_tail:
            raise ValueError(
                f"Очікувано хвіст (T, C, H, W)={expected_tail}, "
                f"отримано {tuple(pixel_values.shape[1:])}"
            )

        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for a batch of clips.

        Wraps :py:meth:`forward` in ``eval()`` + ``inference_mode()`` and
        applies ``softmax(dim=-1)``.

        Args:
            pixel_values: Tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)`` with rows
            summing to 1.
        """
        self.model.eval()
        with torch.inference_mode():
            logits = self.forward(pixel_values)
            return torch.softmax(logits, dim=-1)

    def parameter_groups(
        self,
        head_lr: float = config.HEAD_LR,
        encoder_lr: float = config.ENCODER_LR,
    ) -> list[dict]:
        """Build optimizer parameter groups with discriminative LRs.

        Args:
            head_lr: Learning rate for the classification head
                (defaults to ``config.HEAD_LR``).
            encoder_lr: Learning rate for the unfrozen encoder blocks
                and the final ``LayerNorm`` (defaults to
                ``config.ENCODER_LR``).

        Returns:
            A list of two dicts ready to feed into ``torch.optim.AdamW``:
            ``[{"params": encoder_+_layernorm, "lr": encoder_lr},
              {"params": classifier_head, "lr": head_lr}]``.
        """
        layers = self.model.timesformer.encoder.layer
        frozen_count = len(layers) - self.unfrozen_blocks

        encoder_params: list[nn.Parameter] = []
        for block in list(layers)[frozen_count:]:
            encoder_params.extend(p for p in block.parameters() if p.requires_grad)
        encoder_params.extend(
            p for p in self.model.timesformer.layernorm.parameters() if p.requires_grad
        )

        head_params = [
            p for p in self.model.classifier.parameters() if p.requires_grad
        ]

        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def save_checkpoint(self, path: Path) -> None:
        """Save the model state and the configuration that produced it.

        Args:
            path: Destination file path. Parent directories are created
                if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "unfrozen_blocks": self.unfrozen_blocks,
            },
        }
        torch.save(payload, path)
        logger.info("Збережено чекпойнт у %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load weights from ``path`` after verifying the stored config.

        Args:
            path: Path to the checkpoint produced by
                :py:meth:`save_checkpoint`.

        Raises:
            ValueError: If the checkpoint's recorded ``num_classes`` or
                ``unfrozen_blocks`` does not match the current model's
                settings.
        """
        path = Path(path)
        payload = torch.load(path, map_location=self.device)
        ck_config = payload.get("config", {})
        for key, expected in (
            ("num_classes", self.num_classes),
            ("unfrozen_blocks", self.unfrozen_blocks),
        ):
            actual = ck_config.get(key)
            if actual != expected:
                raise ValueError(
                    f"Невідповідність конфігурації чекпойнта: {key}={actual}, "
                    f"очікувано {expected}"
                )
        self.model.load_state_dict(payload["state_dict"])
        logger.info("Завантажено чекпойнт з %s", path)


if __name__ == "__main__":
    import time

    from src.utils import setup_logging

    setup_logging()

    clf = TimeSformerClassifier(device=torch.device("cpu"))

    total = sum(p.numel() for p in clf.model.parameters())
    trainable = sum(p.numel() for p in clf.model.parameters() if p.requires_grad)
    print(f"total params      = {total:,}")
    print(f"trainable params  = {trainable:,}")
    print(f"trainable ratio   = {100.0 * trainable / total:.2f}%")
    print(f"device            = {next(clf.model.parameters()).device}")
    print(f"head out_features = {clf.model.classifier.out_features}")
    assert clf.model.classifier.out_features == 10

    x = torch.randn(1, 8, 3, 224, 224)
    t0 = time.perf_counter()
    logits = clf(x)
    dt = time.perf_counter() - t0
    print(f"forward latency   = {dt:.3f} s")
    print(f"logits.shape      = {tuple(logits.shape)}")
    print(f"logits.dtype      = {logits.dtype}")
    print(f"logits.min()      = {logits.min().item():.4f}")
    print(f"logits.max()      = {logits.max().item():.4f}")

    proba = clf.predict_proba(x)
    s = proba.sum(dim=-1)
    print(f"proba.sum(dim=-1) = {[f'{v:.6f}' for v in s.tolist()]}")
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    print("Smoke test PASSED")
