import asyncio, io, os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.expanduser("~/playwright-browsers")
from PIL import Image
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        p = await b.new_page(viewport={"width": 800, "height": 600})
        await p.set_content("<h1 style='color:red'>Playwright works!</h1>")
        png = await p.screenshot()
        await b.close()
        return Image.open(io.BytesIO(png)).size

size = asyncio.run(test())
print(f"OK: rendered image size = {size}")
