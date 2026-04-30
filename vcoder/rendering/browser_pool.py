from __future__ import annotations

import asyncio
import os
from typing import Optional

# Playwright browser location — prefer /dev/shm/pw-browsers (fast SSD) if it
# exists, fall back to ~/playwright-browsers (home dir install on this cluster).
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    _shm_browsers = "/dev/shm/pw-browsers"
    _home_browsers = os.path.expanduser("~/playwright-browsers")
    if os.path.isdir(_shm_browsers):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _shm_browsers
    elif os.path.isdir(_home_browsers):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _home_browsers

# Chromium shared library dependencies not in system lib path on this cluster.
# /dev/shm/compat_libs contains symlinks to libnspr4, libnss3, libasound, libstdc++.
_COMPAT_LIBS = "/dev/shm/compat_libs"
if os.path.isdir(_COMPAT_LIBS):
    _cur = os.environ.get("LD_LIBRARY_PATH", "")
    if _COMPAT_LIBS not in _cur:
        os.environ["LD_LIBRARY_PATH"] = f"{_COMPAT_LIBS}:{_cur}" if _cur else _COMPAT_LIBS

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
