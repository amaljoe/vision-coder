"""Streamlit demo: compare vLLM-generated HTML from 1-2 models side-by-side."""

import asyncio
import base64
import io
from dataclasses import dataclass

import requests
import streamlit as st
from PIL import Image
from playwright.async_api import async_playwright

from vcoder.utils.image_utils import compute_clip_similarity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    host: str = "localhost"
    port: int = 8000
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_seq_len: int = 8192
    system_prompt: str = (
        "You are an expert frontend developer. Given a reference screenshot, "
        "reproduce the UI as a single self-contained HTML file with inline CSS. "
        "Return ONLY the HTML code, no explanation."
    )
    user_prompt: str = (
        "Reproduce the UI shown in this screenshot as a single self-contained "
        "HTML file with inline CSS. Match the layout, colors, and typography "
        "as closely as possible."
    )


DEFAULT_CONFIGS = [
    ModelConfig(),
    ModelConfig(
        port=8001, 
        model_name="outputs/vcoder-grpo-clip/checkpoint-500/", 
        user_prompt="Generate the HTML code that reproduces this website screenshot.",
        system_prompt="You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code with inline CSS that reproduces the visual layout. Output only the HTML code wrapped in ```html and ``` tags."
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def query_vllm(cfg: ModelConfig, ref_image: Image.Image) -> str:
    """Send the reference image to a vLLM OpenAI-compatible server and return the response text."""
    url = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
    data_url = _image_to_data_url(ref_image)

    messages = []
    if cfg.system_prompt.strip():
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": cfg.user_prompt},
        ],
    })

    payload = {
        "model": cfg.model_name,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_html(text: str) -> str:
    """Extract HTML from model output, stripping markdown fences if present."""
    # Try to extract from ```html ... ``` block
    import re
    m = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(<!DOCTYPE|<html)(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip()
    # If it already looks like raw HTML, return as-is
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return text.strip()
    return text.strip()


async def render_html(html: str) -> Image.Image | None:
    """Render HTML to a screenshot via Playwright."""
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
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="vLLM Inference Demo", layout="wide")
st.title("vLLM Inference Comparison")

# --- Reference image ---
st.subheader("Reference Image")
uploaded = st.file_uploader("Upload a reference screenshot", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    ref_image = Image.open(uploaded).convert("RGB")
    st.image(ref_image, width=400)

# --- Model config ---
num_models = st.radio("Number of models to compare", [1, 2], horizontal=True)

configs: list[ModelConfig] = []
cols = st.columns(num_models)
for i, col in enumerate(cols):
    with col:
        st.subheader(f"Model {i + 1}")
        d = DEFAULT_CONFIGS[i]
        host = st.text_input("Host", value=d.host, key=f"host_{i}")
        port = st.number_input("Port", value=d.port, min_value=1, max_value=65535, key=f"port_{i}")
        model_name = st.text_input("Model name", value=d.model_name, key=f"model_{i}")
        temperature = st.slider("Temperature", 0.0, 2.0, value=d.temperature, step=0.1, key=f"temp_{i}")
        max_tokens = st.number_input("Max tokens", value=d.max_tokens, min_value=128, max_value=32768, key=f"max_{i}")
        system_prompt = st.text_area("System prompt", value=d.system_prompt, height=100, key=f"sys_{i}")
        user_prompt = st.text_area("User prompt", value=d.user_prompt, height=80, key=f"usr_{i}")
        configs.append(ModelConfig(
            host=host,
            port=int(port),
            model_name=model_name,
            temperature=temperature,
            max_tokens=int(max_tokens),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ))

# --- Run inference ---
if st.button("Run Inference", type="primary", disabled=not uploaded):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict] = [{}] * len(configs)

    with st.spinner("Querying vLLM server(s)..."):
        def _run(idx: int, cfg: ModelConfig):
            raw = query_vllm(cfg, ref_image)
            html = extract_html(raw)
            rendered = asyncio.run(render_html(html))
            clip = compute_clip_similarity(rendered, ref_image) if rendered else 0.0
            return idx, raw, html, rendered, clip

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_run, i, c) for i, c in enumerate(configs)]
            for f in as_completed(futures):
                idx, raw, html, rendered, clip = f.result()
                results[idx] = {"raw": raw, "html": html, "rendered": rendered, "clip": clip}

    # --- Display results ---
    st.divider()
    result_cols = st.columns(len(configs))
    for i, col in enumerate(result_cols):
        r = results[i]
        with col:
            st.subheader(f"Model {i + 1}: {configs[i].model_name}")
            st.metric("CLIP Score", f"{r['clip']:.4f}")
            if r["rendered"]:
                st.image(r["rendered"], caption="Rendered output", use_container_width=True)
            else:
                st.error("Rendering failed.")
            with st.expander("Generated HTML", expanded=False):
                st.code(r["html"], language="html")
            with st.expander("Raw model output", expanded=False):
                st.text(r["raw"])
