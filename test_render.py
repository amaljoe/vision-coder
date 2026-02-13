"""Quick test for rendering with styles."""
import asyncio
import io
from PIL import Image
from playwright.async_api import async_playwright
from vcoder.utils.image_utils import compute_ssim


HTML = """<html>
<head>
<style>
body { margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; justify-content: center; align-items: center; height: 100vh; font-family: Arial, sans-serif; }
.card { background: white; border-radius: 16px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; }
h1 { color: #333; margin-bottom: 10px; }
p { color: #666; }
</style>
</head>
<body><div class="card"><h1>Hello World</h1><p>Styled card component</p></div></body>
</html>"""


async def test():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    page = await browser.new_page(viewport={"width": 1280, "height": 1024})
    await page.set_content(HTML, wait_until="load", timeout=15000)
    await page.wait_for_timeout(1000)
    png = await page.screenshot(type="png", full_page=False, path="/tmp/test_styled.png")
    print(f"screenshot: {len(png)} bytes")
    rendered = Image.open(io.BytesIO(png)).convert("RGB")
    print(f"image size: {rendered.size}")

    # Compare with itself as sanity check
    score = compute_ssim(rendered, rendered)
    print(f"self-SSIM: {score}")

    await page.close()
    await browser.close()
    await pw.stop()


asyncio.run(test())
