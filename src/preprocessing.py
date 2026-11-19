"""Preprocessing for TimeSformer input.

Wraps HuggingFace ``AutoImageProcessor`` so that the raw BGR ``uint8``
clip returned by :py:meth:`src.video_buffer.VideoBuffer.sample_clip`
becomes a normalized ``float32`` tensor of shape
``(1, T, 3, IMAGE_SIZE, IMAGE_SIZE)`` ready to feed
``TimesformerForVideoClassification`` as ``pixel_values=...``.

We never reimplement the normalization or resize manually: the
checkpoint's own processor knows the correct mean/std and resize policy
for its training distribution. Hardcoding ImageNet stats here would
silently break if the Kinetics-400 checkpoint ever switches to a
different normalization (which has happened in past HF releases).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoImageProcessor

import config

logger = logging.getLogger(__name__)


class ClipPreprocessor:
    """Turn raw BGR clips into model-ready ``pixel_values`` tensors.

    Attributes:
        model_name: HuggingFace identifier of the source checkpoint.
        processor: Cached ``AutoImageProcessor`` for that checkpoint.
    """

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        """Load the official image processor for ``model_name``.

        Args:
            model_name: HuggingFace model identifier (defaults to
                ``config.MODEL_NAME``).
        """
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        logger.info("Завантажено AutoImageProcessor для %s", model_name)

    def __call__(self, clip_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR uint8 clip into a normalized model-ready tensor.

        Args:
            clip_bgr: array of shape ``(T, H_raw, W_raw, 3)``, dtype
                ``uint8``, BGR channel order (OpenCV native). Typically
                ``T = 8``.

        Returns:
            Tensor of shape ``(1, T, 3, IMAGE_SIZE, IMAGE_SIZE)``, dtype
            ``float32``, on CPU. The leading 1 is the batch dim;
            downstream code is free to ``.to(device)`` it. The tensor is
            ready for ``TimesformerForVideoClassification.forward`` as
            ``pixel_values=...``.

        Raises:
            ValueError: If ``clip_bgr`` does not have the expected shape
                or dtype.
            RuntimeError: If the processor's output shape drifts from
                the expected ``(1, T, 3, IMAGE_SIZE, IMAGE_SIZE)``.
        """
        if clip_bgr.dtype != np.uint8:
            raise ValueError(
                f"Очікувано dtype=uint8, отримано {clip_bgr.dtype}"
            )
        if clip_bgr.ndim != 4:
            raise ValueError(
                f"Очікувано ndim=4 (T, H, W, C), отримано ndim={clip_bgr.ndim} "
                f"з shape={clip_bgr.shape}"
            )
        if clip_bgr.shape[0] != config.CLIP_LENGTH:
            raise ValueError(
                f"Очікувано T={config.CLIP_LENGTH} кадрів, отримано "
                f"{clip_bgr.shape[0]}"
            )
        if clip_bgr.shape[-1] != 3:
            raise ValueError(
                f"Очікувано 3 канали, отримано {clip_bgr.shape[-1]}"
            )

        # BGR → RGB by reversing the last axis: much cheaper than a
        # per-frame cv2.cvtColor loop. The view is non-contiguous but
        # the processor produces a contiguous copy internally when it
        # resizes and normalizes, so no explicit .copy() is needed here.
        clip_rgb = clip_bgr[..., ::-1]

        inputs = self.processor(list(clip_rgb), return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"]

        expected = (
            1,
            config.CLIP_LENGTH,
            3,
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        )
        if tuple(pixel_values.shape) != expected:
            logger.warning(
                "Несподівана форма pixel_values: %s, очікувано %s",
                tuple(pixel_values.shape),
                expected,
            )
            raise RuntimeError(
                f"AutoImageProcessor returned shape "
                f"{tuple(pixel_values.shape)}, expected {expected}"
            )

        return pixel_values


if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()

    pp = ClipPreprocessor()
    clip = np.random.randint(0, 256, (8, 480, 640, 3), dtype=np.uint8)
    tensor = pp(clip)
    print(f"tensor.shape  = {tuple(tensor.shape)}")
    print(f"tensor.dtype  = {tensor.dtype}")
    print(f"tensor.min()  = {tensor.min().item():.4f}")
    print(f"tensor.max()  = {tensor.max().item():.4f}")
    print(f"tensor.mean() = {tensor.mean().item():.4f}")

    assert tuple(tensor.shape) == (1, 8, 3, 224, 224), tensor.shape
    assert tensor.dtype == torch.float32, tensor.dtype
    print("Smoke test PASSED")
