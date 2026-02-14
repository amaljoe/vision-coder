from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

from PIL import Image

from vcoder.rendering.browser_pool import BrowserPool
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.utils.image_utils import compute_clip_similarity, compute_ssim

logger = logging.getLogger(__name__)


async def _render_html(html: str, pool: BrowserPool) -> Image.Image | None:
    """Render HTML and return as PIL Image, or None on failure."""
    png_bytes = await pool.render(html)
    if png_bytes is None:
        return None
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


async def _render_and_score(html: str, reference: Image.Image, pool: BrowserPool) -> float:
    """Render HTML and compute SSIM against the reference image."""
    rendered = await _render_html(html, pool)
    if rendered is None:
        return 0.0
    return compute_ssim(rendered, reference)


async def _compute_visual_rewards(
    completions: list[list[dict[str, str]]],
    images: list[Image.Image],
) -> list[float]:
    """Async batch computation of visual fidelity rewards."""
    pool = await BrowserPool.get_instance()
    tasks = []
    for completion, ref_image in zip(completions, images):
        text = completion[0]["content"]
        html_str = extract_html_from_completion(text)
        if html_str is None:
            tasks.append(None)
        else:
            tasks.append(asyncio.create_task(_render_and_score(html_str, ref_image, pool)))

    rewards = []
    for task in tasks:
        if task is None:
            rewards.append(0.0)
        else:
            try:
                score = await task
                rewards.append(score)
            except Exception:
                rewards.append(0.0)
    return rewards


def visual_fidelity_reward(
    completions: list[list[dict[str, str]]],
    image: list[Image.Image],
    **kwargs,
) -> list[float]:
    """Compute visual fidelity reward by rendering generated HTML and comparing to the input screenshot.

    Uses async Playwright rendering with a browser pool for concurrent rendering.

    Args:
        completions: Model completions (each is a list with one dict containing 'content')
        image: Reference screenshot images (one per completion)

    Returns:
        List of SSIM scores in [0, 1].
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _compute_visual_rewards(completions, image))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(_compute_visual_rewards(completions, image))
    except Exception:
        return [0.0] * len(completions)


async def _render_and_clip_score(html: str, reference: Image.Image, pool: BrowserPool) -> float:
    """Render HTML and compute CLIP similarity against the reference image."""
    rendered = await _render_html(html, pool)
    if rendered is None:
        return 0.0
    return compute_clip_similarity(rendered, reference)


async def _compute_clip_rewards(
    completions: list[list[dict[str, str]]],
    images: list[Image.Image],
) -> list[float]:
    """Async batch computation of CLIP-based visual rewards."""
    pool = await BrowserPool.get_instance()
    rendered_images: list[Image.Image | None] = []
    for completion in completions:
        text = completion[0]["content"]
        html_str = extract_html_from_completion(text)
        if html_str is None:
            rendered_images.append(None)
        else:
            rendered_images.append(await _render_html(html_str, pool))

    rewards = []
    for rendered, ref_image in zip(rendered_images, images):
        if rendered is None:
            rewards.append(0.0)
        else:
            try:
                rewards.append(compute_clip_similarity(rendered, ref_image))
            except Exception:
                rewards.append(0.0)
    return rewards


def clip_visual_reward(
    completions: list[list[dict[str, str]]],
    image: list[Image.Image],
    **kwargs,
) -> list[float]:
    """Compute visual reward using CLIP image-image similarity.

    Renders generated HTML and compares the screenshot to the reference image
    using CLIP embeddings (cosine similarity).

    Args:
        completions: Model completions (each is a list with one dict containing 'content')
        image: Reference screenshot images (one per completion)

    Returns:
        List of CLIP similarity scores in [0, 1].
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _compute_clip_rewards(completions, image))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(_compute_clip_rewards(completions, image))
    except Exception:
        return [0.0] * len(completions)
