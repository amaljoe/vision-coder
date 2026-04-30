"""Streamlit demo: screenshot → VisionCoder (grpo-clip) → rendered HTML + scores."""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image
from playwright.async_api import async_playwright

from vcoder.eval.generate_predictions import (
    DIRECT_PROMPT,
    MAX_TOKENS,
    TEMPERATURE,
    cleanup_response,
)
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.utils.image_utils import compute_clip_similarity, compute_ssim

TESTSET_DIR = Path("/home/compiling-ganesh/24m0797/workspace/Design2Code/testset_final_extracted")
SFT_MODEL = "/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip-sft-openenv"
GRPO_MODEL = "/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip/checkpoint-500"
NUM_BENCHMARK = 484


def _get_vllm_model(host: str, port: int) -> str:
    """Query VLLM /v1/models and return the first model id, or fall back to SFT model path."""
    try:
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=3)
        return r.json()["data"][0]["id"]
    except Exception:
        return SFT_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def call_vllm(img: Image.Image, host: str, port: int, model_id: str | None = None) -> str:
    """POST image to VLLM and return raw completion text."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_id or _get_vllm_model(host, port),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}},
                    {"type": "text", "text": DIRECT_PROMPT},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _render_html_async(html: str, width: int = 1280, height: int = 1024) -> Image.Image | None:
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    try:
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="networkidle", timeout=10000)
        png_bytes = await page.screenshot(type="png", full_page=False)
        await page.close()
    except Exception:
        await browser.close()
        await pw.stop()
        return None
    await browser.close()
    await pw.stop()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_html(html: str, ref_size: tuple[int, int] = (1280, 1024)) -> Image.Image | None:
    """Render HTML at the same viewport dimensions as the reference image."""
    return asyncio.run(_render_html_async(html, width=ref_size[0], height=ref_size[1]))


def score_response(raw_response: str, ref_image: Image.Image, rendered: Image.Image | None) -> dict:
    completions = [[{"content": raw_response}]]
    fmt = format_reward(completions)[0]
    val = html_validity_reward(completions)[0]
    if rendered is not None:
        ssim = compute_ssim(rendered, ref_image)
        clip = compute_clip_similarity(rendered, ref_image)
    else:
        ssim, clip = 0.0, 0.0
    return {"Format": fmt, "HTML Validity": val, "SSIM": ssim, "CLIP": clip}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="VisionCoder Demo", layout="wide")
st.title("VisionCoder Demo — Screenshot → HTML")

# Sidebar config
with st.sidebar:
    st.header("VLLM Config")
    vllm_host = st.text_input("Host", value="localhost")
    vllm_port = st.number_input("Port", value=8001, min_value=1, max_value=65535, step=1)
    active_model_id = _get_vllm_model(vllm_host, int(vllm_port))
    st.divider()
    model_short = active_model_id.split("/")[-1] if "/" in active_model_id else active_model_id
    st.caption(f"Active model: `{model_short}`")
    st.caption(f"`{active_model_id}`")

# Image input tabs
tab_upload, tab_bench = st.tabs(["Upload Image", "Design2Code Benchmark"])

ref_image: Image.Image | None = None

with tab_upload:
    uploaded = st.file_uploader("Upload a webpage screenshot", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        ref_image = Image.open(uploaded).convert("RGB")
        st.image(ref_image, caption="Uploaded image", use_container_width=True)

with tab_bench:
    idx = st.slider("Benchmark index", min_value=0, max_value=NUM_BENCHMARK - 1, value=0)
    png_path = TESTSET_DIR / f"{idx}.png"
    if png_path.exists():
        bench_img = Image.open(png_path).convert("RGB")
        st.image(bench_img, caption=f"Design2Code #{idx}", use_container_width=True)
    else:
        st.warning(f"Image not found: {png_path}")
        bench_img = None

# Determine active image based on which tab was interacted with last
# We use session_state to track the last used source
if "last_source" not in st.session_state:
    st.session_state.last_source = "bench"

# Determine the active reference image
active_ref: Image.Image | None
if uploaded:
    active_ref = ref_image
    st.session_state.last_source = "upload"
elif png_path.exists():
    active_ref = bench_img
    st.session_state.last_source = "bench"
else:
    active_ref = None

# Generate button
st.divider()
if st.button("Generate HTML", type="primary", disabled=active_ref is None):
    # Clear cached results on new generation
    for key in ("raw_response", "rendered", "scores", "html_code"):
        st.session_state.pop(key, None)

    with st.spinner("Calling VLLM... (may take up to 2 min)"):
        try:
            raw = call_vllm(active_ref, vllm_host, int(vllm_port), model_id=active_model_id)
            st.session_state["raw_response"] = raw
        except Exception as e:
            st.error(f"VLLM error: {e}")
            st.stop()

    html_code = cleanup_response(raw)
    st.session_state["html_code"] = html_code

    with st.spinner("Rendering HTML with Playwright..."):
        rendered = render_html(html_code, ref_size=active_ref.size)
        st.session_state["rendered"] = rendered

    with st.spinner("Computing scores..."):
        scores = score_response(raw, active_ref, rendered)
        st.session_state["scores"] = scores

# Display results if available
if "html_code" in st.session_state:
    st.subheader("Results")

    col_ref, col_render = st.columns(2)
    with col_ref:
        st.markdown("**Reference**")
        if active_ref:
            st.image(active_ref, use_container_width=True)
    with col_render:
        st.markdown("**Rendered Output**")
        rendered = st.session_state.get("rendered")
        if rendered:
            st.image(rendered, use_container_width=True)
        else:
            st.warning("Rendering failed or returned no output.")

    # Scores row
    if "scores" in st.session_state:
        scores = st.session_state["scores"]
        st.divider()
        st.subheader("Evaluation Scores")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Format Reward", f"{scores['Format']:.2f}", help="0.5 fenced block + 0.25 <html> + 0.25 </html>")
        m2.metric("HTML Validity", f"{scores['HTML Validity']:.2f}", help="Parseability + structure + tag diversity")
        m3.metric("SSIM", f"{scores['SSIM']:.4f}", help="Structural similarity between rendered and reference")
        m4.metric("CLIP", f"{scores['CLIP']:.4f}", help="CLIP cosine similarity between rendered and reference")

    # Generated HTML
    with st.expander("Generated HTML", expanded=False):
        st.code(st.session_state["html_code"], language="html")

    with st.expander("Raw model response", expanded=False):
        st.text(st.session_state.get("raw_response", ""))
