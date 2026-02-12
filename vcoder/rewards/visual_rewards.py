from __future__ import annotations

import asyncio
import io
from typing import Optional

from PIL import Image

from vcoder.rendering.browser_pool import BrowserPool
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.utils.image_utils import compute_ssim


async def _render_and_score(html: str, reference: Image.Image, pool: BrowserPool) -> float:
    """Render HTML and compute SSIM against the reference image."""
    png_bytes = await pool.render(html)
    if png_bytes is None:
        return 0.0
    rendered = Image.open(io.BytesIO(png_bytes)).convert("RGB")
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
