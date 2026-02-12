from __future__ import annotations

import asyncio
import os
from typing import Optional

from playwright.async_api import async_playwright, Browser


class BrowserPool:
    """Singleton pool of Playwright browser pages for HTML rendering.

    Uses a semaphore to limit concurrent page renders.
    """

    _instance: Optional[BrowserPool] = None
    _lock = asyncio.Lock()

    def __init__(self, max_concurrent: int = 8):
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._playwright = None
        self._browser: Optional[Browser] = None

    @classmethod
    async def get_instance(cls, max_concurrent: int = 8) -> BrowserPool:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    pool = cls(max_concurrent)
                    await pool._start()
                    cls._instance = pool
        return cls._instance

    async def _start(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )

    async def render(self, html: str, width: int = 1280, height: int = 1024, timeout_ms: int = 5000) -> bytes | None:
        """Render HTML string to a PNG screenshot.

        Returns PNG bytes, or None on failure/timeout.
        """
        async with self._semaphore:
            page = None
            try:
                page = await self._browser.new_page(viewport={"width": width, "height": height})
                await page.set_content(html, wait_until="networkidle", timeout=timeout_ms)
                screenshot = await page.screenshot(type="png", full_page=False)
                return screenshot
            except Exception:
                return None
            finally:
                if page:
                    await page.close()

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        BrowserPool._instance = None
