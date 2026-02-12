from __future__ import annotations

import asyncio
import io

from PIL import Image

from vcoder.rendering.browser_pool import BrowserPool


def _blank_white_image(width: int = 512, height: int = 512) -> Image.Image:
    """Return a blank white PIL image (used as fallback on render failure)."""
    return Image.new("RGB", (width, height), (255, 255, 255))


async def _render_html_async(html: str) -> Image.Image:
    """Render HTML to a PIL Image using the browser pool."""
    pool = await BrowserPool.get_instance()
    png_bytes = await pool.render(html)
    if png_bytes is None:
        return _blank_white_image()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_html_to_image(html: str) -> Image.Image:
    """Synchronous wrapper: render an HTML string to a PIL Image.

    Returns a blank white image on failure.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from within an existing event loop (e.g. Jupyter),
            # use nest_asyncio or run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _render_html_async(html))
                return future.result(timeout=15)
        else:
            return loop.run_until_complete(_render_html_async(html))
    except Exception:
        return _blank_white_image()
