from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


COMPARISON_SIZE = (512, 512)


def resize_for_comparison(img: Image.Image, size: tuple[int, int] = COMPARISON_SIZE) -> Image.Image:
    """Resize an image to a fixed size for SSIM comparison."""
    return img.convert("RGB").resize(size, Image.LANCZOS)


def compute_ssim(img_a: Image.Image, img_b: Image.Image, size: tuple[int, int] = COMPARISON_SIZE) -> float:
    """Compute SSIM between two PIL images after resizing to a common size.

    Returns a float in [0, 1] where 1 means identical.
    """
    a = np.array(resize_for_comparison(img_a, size))
    b = np.array(resize_for_comparison(img_b, size))
    score = ssim(a, b, channel_axis=2, data_range=255)
    return float(score)
