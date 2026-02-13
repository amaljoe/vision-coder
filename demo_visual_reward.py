"""Streamlit demo for visual fidelity rewards (SSIM + CLIP)."""

import asyncio
import io

import streamlit as st
from PIL import Image

from playwright.async_api import async_playwright
from vcoder.utils.image_utils import compute_clip_similarity, compute_ssim


async def render_and_score(html: str, ref_image: Image.Image) -> tuple[Image.Image | None, float, float]:
    """Render HTML and compute both SSIM and CLIP scores."""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    try:
        page = await browser.new_page(viewport={"width": 1280, "height": 1024})
        await page.set_content(html, wait_until="load", timeout=15000)
        await page.wait_for_timeout(1000)
        png_bytes = await page.screenshot(type="png", full_page=False)
        await page.close()
    except Exception as e:
        await browser.close()
        await pw.stop()
        raise RuntimeError(f"Rendering failed: {e}")
    await browser.close()
    await pw.stop()
    rendered = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    ssim_score = compute_ssim(rendered, ref_image)
    clip_score = compute_clip_similarity(rendered, ref_image)
    return rendered, ssim_score, clip_score


st.set_page_config(page_title="Visual Fidelity Reward Demo", layout="wide")
st.title("Visual Fidelity Reward Demo")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Image")
    uploaded = st.file_uploader("Upload a reference screenshot", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        ref_image = Image.open(uploaded).convert("RGB")
        st.image(ref_image, use_container_width=True)

with col2:
    st.subheader("HTML + CSS Code")
    html_code = st.text_area(
        "Paste your HTML+CSS code here",
        height=400,
        placeholder="<html>\n<head><style>body { background: white; }</style></head>\n<body>Hello</body>\n</html>",
    )

if st.button("Compute Reward", type="primary", disabled=not (uploaded and html_code)):
    with st.spinner("Rendering HTML and computing rewards..."):
        rendered_image, ssim_score, clip_score = asyncio.run(render_and_score(html_code, ref_image))

    st.divider()

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.subheader("Rendered Output")
        if rendered_image:
            st.image(rendered_image, use_container_width=True)
        else:
            st.error("Rendering failed.")
    with res_col2:
        st.subheader("SSIM + MAE")
        st.metric(label="Score", value=f"{ssim_score:.4f}")
    with res_col3:
        st.subheader("CLIP Similarity")
        st.metric(label="Score", value=f"{clip_score:.4f}")
