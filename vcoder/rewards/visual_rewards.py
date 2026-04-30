from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Optional

from PIL import Image

from vcoder.rendering.browser_pool import BrowserPool
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.utils.image_utils import compute_clip_similarity, compute_ssim

logger = logging.getLogger(__name__)

# Persistent background event loop so BrowserPool (Playwright) is created once
# and reused across training steps. asyncio.run() creates/destroys a loop each
# call, breaking the cached Playwright browser objects in BrowserPool._instance.
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            BrowserPool._instance = None  # reset stale singleton if loop restarted
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True, name="reward-async")
            _bg_thread.start()
    return _bg_loop


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


def _run_in_bg(coro, timeout=300):
    """Submit a coroutine to the persistent background event loop and wait for the result."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def visual_fidelity_reward(
    completions: list[list[dict[str, str]]],
    image: list[Image.Image],
    **kwargs,
) -> list[float]:
    try:
        return _run_in_bg(_compute_visual_rewards(completions, image))
    except Exception as e:
        logger.warning(f"visual_fidelity_reward failed: {e}")
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
    tasks = []
    for completion in completions:
        text = completion[0]["content"]
        html_str = extract_html_from_completion(text)
        if html_str is None:
            tasks.append(None)
        else:
            tasks.append(asyncio.create_task(_render_html(html_str, pool)))

    rendered_images: list[Image.Image | None] = []
    for task in tasks:
        if task is None:
            rendered_images.append(None)
        else:
            try:
                rendered_images.append(await task)
            except Exception:
                rendered_images.append(None)

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
    try:
        return _run_in_bg(_compute_clip_rewards(completions, image))
    except Exception as e:
        logger.warning(f"clip_visual_reward failed: {e}")
        return [0.0] * len(completions)
