from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPModel, CLIPProcessor


COMPARISON_SIZE = (512, 512)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None


def _get_clip(device: str = "cuda"):
    """Lazy-load and cache the CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    return _clip_model, _clip_processor


def resize_for_comparison(img: Image.Image, size: tuple[int, int] = COMPARISON_SIZE) -> Image.Image:
    """Resize an image to a fixed size for SSIM comparison."""
    return img.convert("RGB").resize(size, Image.LANCZOS)


def compute_ssim(
    img_a: Image.Image,
    img_b: Image.Image,
    size: tuple[int, int] = COMPARISON_SIZE,
    ssim_weight: float = 0.5,
) -> float:
    """Compute visual similarity between two PIL images after resizing.

    Combines SSIM (structural) with a pixel-level similarity (1 - MAE/255)
    so the score is sensitive to both layout *and* color/styling differences.

    Returns a float in [0, 1] where 1 means identical.
    """
    a = np.array(resize_for_comparison(img_a, size), dtype=np.float64)
    b = np.array(resize_for_comparison(img_b, size), dtype=np.float64)
    ssim_score = ssim(a, b, channel_axis=2, data_range=255)
    pixel_score = 1.0 - np.mean(np.abs(a - b)) / 255.0
    return float(ssim_weight * ssim_score + (1.0 - ssim_weight) * pixel_score)


def compute_clip_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    device: str = "cuda",
) -> float:
    """Compute CLIP image-image cosine similarity.

    Returns a float in [0, 1] where 1 means perceptually identical.
    """
    model, processor = _get_clip(device)
    inputs = processor(images=[img_a.convert("RGB"), img_b.convert("RGB")], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    score = (feats[0] @ feats[1]).item()
    # Clamp to [0, 1] (cosine sim can be slightly negative for very different images)
    return float(max(0.0, score))
