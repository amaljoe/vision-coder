"""VisionCoder live streaming demo.

Gradio app that streams HTML generation from both VLLM servers simultaneously
and renders partial HTML in iframes as tokens arrive.

Features:
- Testset dropdown (first 50 images) or custom upload
- Both models stream concurrently via threads + queues
- Live iframe render updates on every token batch
- Handles unclosed tags, partial CSS, <think> blocks gracefully

Run:
    python3 -u vcoder/demo/app.py
"""

from __future__ import annotations

import base64
import html as html_module
import io
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import requests
from PIL import Image

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.expanduser("~/playwright-browsers")

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTSET_DIR = REPO_ROOT.parent / "Design2Code" / "testset_final_extracted"

BASE_CFG = {
    "port": 8000,
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "name": "Qwen3-VL-2B (base)",
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
FINETUNED_CFG = {
    "port": 8001,
    "model_id": "/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip/checkpoint-500",
    "name": "VCoder-GRPO-CLIP",
    "extra_body": {},
}

DIRECT_PROMPT = (
    "You are an expert web developer who specializes in HTML and CSS.\n"
    "A user will provide you with a screenshot of a webpage.\n"
    "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    "Include all CSS code in the HTML file itself.\n"
    'If it involves any images, use "rick.jpg" as the placeholder.\n'
    "Some images on the webpage are replaced with a blue rectangle as the placeholder, "
    'use "rick.jpg" for those as well.\n'
    "Do not hallucinate any dependencies to external files. "
    "You do not need to include JavaScript scripts for dynamic interactions.\n"
    "Pay attention to things like size, text, position, and color of all the elements, "
    "as well as the overall layout.\n"
    "Respond with the content of the HTML+CSS file:\n"
)

POLL_INTERVAL = 0.02   # seconds between queue drain / yield


# ---------------------------------------------------------------------------
# Partial HTML fixer
# ---------------------------------------------------------------------------

def fix_partial_html(raw: str) -> str:
    """Extract renderable HTML from a partial model output.

    Handles:
    - <think>...</think> prefix (finetuned model)
    - ```html markdown fences (complete or truncated)
    - Unclosed <style> blocks with partial CSS
    - Unclosed HTML tags (delegated to BeautifulSoup)
    - Bare HTML fragments without doctype/html tags
    """
    if not raw:
        return ""

    # 1. Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # 2. If we're still inside an unclosed <think>, nothing to render yet
    if "<think>" in text:
        return ""

    text = text.strip()

    # 3. Extract HTML: prefer content inside ```html fence
    html = ""
    fence_m = re.search(r"```(?:html)?\s*\n?", text, re.IGNORECASE)
    if fence_m:
        after = text[fence_m.end():]
        # Strip any closing ``` at the end (may be partial/absent mid-stream)
        after = re.sub(r"\s*```\s*$", "", after)
        html = after
    elif re.search(r"<!DOCTYPE|<html\b", text, re.IGNORECASE):
        # Raw HTML without a fence
        start = len(text)
        for pat in (r"<!doctype", r"<html"):
            idx = text.lower().find(pat)
            if idx != -1:
                start = min(start, idx)
        html = text[start:]
    else:
        return ""   # still in preamble / system text

    html = html.strip()
    if not html:
        return ""

    # 4. Close any open <style> block that has incomplete CSS
    #    (partial CSS in a <style> tag causes entire style to be dropped)
    open_styles = list(re.finditer(r"<style\b[^>]*>", html, re.IGNORECASE))
    close_styles = list(re.finditer(r"</style>", html, re.IGNORECASE))
    if len(open_styles) > len(close_styles):
        # Find where the last unclosed <style> tag ends
        last_open_end = open_styles[-1].end()
        css_fragment = html[last_open_end:]
        # Count unbalanced braces in the CSS fragment
        open_braces = css_fragment.count("{") - css_fragment.count("}")
        html += "\n" + ("}" * max(0, open_braces)) + "\n</style>"

    # 5. If there's no <html> or <!DOCTYPE>, wrap so the browser has a base
    if not re.search(r"<!DOCTYPE|<html\b", html, re.IGNORECASE):
        html = f"<!DOCTYPE html><html><head></head><body>{html}</body></html>"

    # 6. Use BeautifulSoup to auto-close unclosed HTML tags
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        result = str(soup)
        # BS4 can return empty string for truly bad input — fall back to raw
        return result if result.strip() else html
    except Exception:
        return html


# ---------------------------------------------------------------------------
# Iframe wrapper for gr.HTML
# ---------------------------------------------------------------------------

_WAITING_TMPL = """
<div style="display:flex;align-items:center;justify-content:center;
            height:580px;background:#f8f8f8;border:1px solid #e0e0e0;
            border-radius:8px;color:#aaa;font-family:sans-serif;font-size:14px;
            flex-direction:column;gap:8px;">
  <span style="font-size:28px">⏳</span>
  <span>{msg}</span>
</div>
"""

_ERROR_TMPL = """
<div style="display:flex;align-items:center;justify-content:center;
            height:580px;background:#fff5f5;border:1px solid #ffb3b3;
            border-radius:8px;color:#cc0000;font-family:sans-serif;font-size:13px;
            padding:16px;text-align:center;">
  <pre style="margin:0;white-space:pre-wrap">{msg}</pre>
</div>
"""


def make_iframe(html_content: str, label: str = "", error: str = "") -> str:
    """Return a gr.HTML-compatible string with either an iframe or a placeholder."""
    if error:
        return _ERROR_TMPL.format(msg=html_module.escape(error))
    if not html_content:
        return _WAITING_TMPL.format(msg=f"Generating {label}…" if label else "Waiting…")
    # srcdoc requires HTML-escaped content (quotes must be escaped)
    escaped = html_module.escape(html_content, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'width="100%" height="580px" '
        f'style="border:1px solid #e0e0e0;border-radius:8px;background:white;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


# ---------------------------------------------------------------------------
# VLLM streaming (runs in a thread)
# ---------------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def stream_vllm(cfg: dict, img: Image.Image, out_q: "queue.Queue[str | None]") -> None:
    """Stream delta tokens from VLLM into out_q. Puts None sentinel when done."""
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    data_url = _image_to_data_url(img)
    payload = {
        "model": cfg["model_id"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": DIRECT_PROMPT},
        ]}],
        "temperature": 0.0,
        "max_tokens": 7500,
        "stream": True,
    }
    payload.update(cfg.get("extra_body", {}))
    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 300)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data.strip() == b"[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            out_q.put(delta)
                    except (json.JSONDecodeError, KeyError):
                        pass
    except Exception as e:
        out_q.put(f"\n[Stream error: {e}]")
    finally:
        out_q.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Testset helpers
# ---------------------------------------------------------------------------

def _load_testset() -> tuple[list[str], dict[str, Path]]:
    if not TESTSET_DIR.exists():
        return [], {}
    pngs = sorted(
        [p for p in TESTSET_DIR.glob("*.png") if "_marker" not in p.name],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )[:50]
    labels = [f"testset/{p.stem}" for p in pngs]
    paths = {f"testset/{p.stem}": p for p in pngs}
    return labels, paths


TESTSET_LABELS, TESTSET_PATHS = _load_testset()


# ---------------------------------------------------------------------------
# Gradio generation function
# ---------------------------------------------------------------------------

def generate(source_type: str, testset_choice: str, uploaded_img):
    """Generator yielding (base_iframe, ft_iframe, base_text, ft_text, ref_img)."""
    import gradio as gr

    # Resolve image
    if source_type == "Testset":
        path = TESTSET_PATHS.get(testset_choice)
        if not path or not path.exists():
            yield (
                make_iframe("", error="No testset image selected."),
                make_iframe("", error="No testset image selected."),
                "", "", None,
            )
            return
        img = Image.open(path).convert("RGB")
    else:
        if uploaded_img is None:
            yield (
                make_iframe("", error="Please upload an image."),
                make_iframe("", error="Please upload an image."),
                "", "", None,
            )
            return
        img = uploaded_img if isinstance(uploaded_img, Image.Image) else Image.fromarray(uploaded_img)
        img = img.convert("RGB")

    # Start streaming threads
    base_q: queue.Queue = queue.Queue()
    ft_q: queue.Queue = queue.Queue()
    threading.Thread(target=stream_vllm, args=(BASE_CFG, img, base_q), daemon=True).start()
    threading.Thread(target=stream_vllm, args=(FINETUNED_CFG, img, ft_q), daemon=True).start()

    base_text = ""
    ft_text = ""
    base_done = False
    ft_done = False

    # Initial yield — show placeholders immediately
    yield (
        make_iframe("", BASE_CFG["name"]),
        make_iframe("", FINETUNED_CFG["name"]),
        "", "", img,
    )

    while not (base_done and ft_done):
        updated = False

        # Drain base queue
        try:
            while True:
                chunk = base_q.get_nowait()
                if chunk is None:
                    base_done = True
                    break
                base_text += chunk
                updated = True
        except queue.Empty:
            pass

        # Drain ft queue
        try:
            while True:
                chunk = ft_q.get_nowait()
                if chunk is None:
                    ft_done = True
                    break
                ft_text += chunk
                updated = True
        except queue.Empty:
            pass

        if updated:
            base_html = fix_partial_html(base_text)
            ft_html = fix_partial_html(ft_text)
            yield (
                make_iframe(base_html, BASE_CFG["name"]),
                make_iframe(ft_html, FINETUNED_CFG["name"]),
                base_text, ft_text, img,
            )
        else:
            time.sleep(POLL_INTERVAL)

    # Final render
    base_html = fix_partial_html(base_text)
    ft_html = fix_partial_html(ft_text)
    yield (
        make_iframe(base_html, BASE_CFG["name"]),
        make_iframe(ft_html, FINETUNED_CFG["name"]),
        base_text, ft_text, img,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _on_source_change(choice: str):
    import gradio as gr
    return gr.update(visible=(choice == "Testset")), gr.update(visible=(choice == "Upload"))


def _load_ref_image(label: str):
    path = TESTSET_PATHS.get(label)
    if path and path.exists():
        return Image.open(path).convert("RGB")
    return None


_CSS = """
.model-col-title {
    text-align: center;
    font-weight: 700;
    font-size: 15px;
    padding: 6px 0;
    background: #f0f4ff;
    border-radius: 6px 6px 0 0;
    border: 1px solid #d0d8ff;
    border-bottom: none;
    margin-bottom: 0;
}
"""


def build_ui():
    import gradio as gr

    with gr.Blocks(title="VisionCoder Demo") as demo:
        gr.Markdown(
            "# 🖼️ VisionCoder — Screenshot → HTML\n"
            "Stream HTML from both models simultaneously. "
            "The rendered preview updates live as tokens arrive."
        )

        with gr.Row():
            # Left: controls
            with gr.Column(scale=1, min_width=280):
                source_radio = gr.Radio(
                    ["Testset", "Upload"], value="Testset",
                    label="Image Source", interactive=True,
                )
                testset_dd = gr.Dropdown(
                    choices=TESTSET_LABELS,
                    value=TESTSET_LABELS[0] if TESTSET_LABELS else None,
                    label="Select testset image",
                    visible=True, interactive=True,
                )
                upload_box = gr.Image(
                    label="Upload image", type="pil",
                    visible=False, height=220,
                )
                gen_btn = gr.Button("⚡ Generate", variant="primary", size="lg")

            # Right: reference preview
            with gr.Column(scale=1, min_width=300):
                ref_img = gr.Image(
                    label="Reference", height=320, interactive=False,
                    value=_load_ref_image(TESTSET_LABELS[0]) if TESTSET_LABELS else None,
                )

        gr.Markdown("---")

        # Two-column render + text output
        with gr.Row(equal_height=False):
            with gr.Column():
                gr.HTML(f'<div class="model-col-title">Qwen3-VL-2B (base) — port 8000</div>')
                base_render = gr.HTML(make_iframe("", BASE_CFG["name"]))
                with gr.Accordion("Raw output", open=False):
                    base_raw = gr.Textbox(
                        label="", lines=10, max_lines=30,
                        interactive=False,
                    )

            with gr.Column():
                gr.HTML(f'<div class="model-col-title">VCoder-GRPO-CLIP — port 8001</div>')
                ft_render = gr.HTML(make_iframe("", FINETUNED_CFG["name"]))
                with gr.Accordion("Raw output", open=False):
                    ft_raw = gr.Textbox(
                        label="", lines=10, max_lines=30,
                        interactive=False,
                    )

        # Events
        source_radio.change(
            _on_source_change, inputs=[source_radio],
            outputs=[testset_dd, upload_box],
        )
        testset_dd.change(
            _load_ref_image, inputs=[testset_dd], outputs=[ref_img],
        )
        gen_btn.click(
            generate,
            inputs=[source_radio, testset_dd, upload_box],
            outputs=[base_render, ft_render, base_raw, ft_raw, ref_img],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=_CSS,
    )
