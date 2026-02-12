# Image-Conditioned Code Generation using Reinforcement Learning with Verifiable Rewards (RLVR)

## 📌 Overview

This project explores whether **Reinforcement Learning with Verifiable
Rewards (RLVR)** can improve **image-conditioned code generation**,
specifically:

> **Input:** UI screenshot or PDF page\
> **Output:** HTML + CSS that visually reproduces the input

Rather than relying purely on supervised fine-tuning (SFT), we train a
**vision-language model (VLM)** using rewards computed from
**automatically verifiable signals** such as rendering similarity and
structural correctness.

------------------------------------------------------------------------

## 🎯 Motivation

Recent work shows RLVR significantly improves reasoning in text-only
tasks (math, coding). However:

-   RLVR for **vision-conditioned generation** remains underexplored\
-   Practical demand for **UI/PDF → HTML/CSS** systems is growing\
-   These tasks require **visual understanding + structured generation**

This project investigates whether RLVR can:

✅ Improve layout fidelity\
✅ Reduce malformed HTML/CSS\
✅ Encourage structured reasoning\
✅ Work with minimal human annotation

------------------------------------------------------------------------

## 🧩 Problem Statement

We train a model that:

-   Accepts an **image** (UI mockup / PDF page)\
-   Generates **HTML + CSS**\
-   Receives rewards based on **verifiable similarity metrics**

**Goal:** Produce code that renders as close as possible to the input
image.

------------------------------------------------------------------------

## 🧠 Proposed Approach

### 1️⃣ Base Pipeline

1.  Input image → Vision-Language Model\
2.  Model generates HTML + CSS\
3.  Code is rendered (headless browser)\
4.  Reward computed via:
    -   Visual similarity
    -   Structural validity
    -   Layout consistency

------------------------------------------------------------------------

### 2️⃣ Reinforcement Learning with Verifiable Rewards

Instead of token-level supervision:

-   Rewards derived from automatic checkers\
-   No human scoring required\
-   Encourages global correctness rather than token matching

------------------------------------------------------------------------

## 🏆 Reward Design

Potential reward components:

### ✅ Visual Fidelity

-   SSIM / LPIPS / Pixel similarity\
-   Layout alignment

### ✅ HTML/CSS Validity

-   Syntax correctness\
-   Proper tag nesting

### ✅ Structural Accuracy

-   DOM tree similarity\
-   Layout block consistency\
-   Reading order correctness

### ✅ PDF-Specific Constraints

-   Table row/column preservation\
-   Bounding box alignment

------------------------------------------------------------------------

## 🧱 Candidate Base Models

-   Qwen 3 VL Thinking 4B\
-   Qwen 3 VL Instruct 8B\
-   Unsloth Devstral 24B\
-   DeepSeek VL2 27B

Selection criteria:

✔ Strong multimodal understanding\
✔ Efficient RL fine-tuning capability

------------------------------------------------------------------------

## 📚 Datasets

### 🔹 HuggingFaceM4/WebSight

UI screenshots ↔ HTML/CSS pairs

### 🔹 KingstarOMEGA/HTML-CSS-UI

HTML/CSS (renderable to UI)

### 🔹 Custom Dataset (Optional)

Generated via web scraping or rendering pipelines

**Note:** RL training only requires images (HTML optional).

------------------------------------------------------------------------

## 🧪 Baselines

1.  Base model (no tuning)\
2.  Supervised Fine-Tuning (SFT)\
3.  RLVR-trained model

Evaluation:

-   Rendering similarity\
-   HTML validity\
-   Structural correctness

------------------------------------------------------------------------

## 📏 Evaluation Metrics

### 🎨 Visual Metrics

-   SSIM\
-   LPIPS\
-   Pixel accuracy

### 🧱 Structural Metrics

-   DOM similarity\
-   Tag validity rate

### 📐 Layout Metrics

-   Block detection accuracy\
-   Reading order consistency

------------------------------------------------------------------------

## 🖥 Compute Resources

Training performed on:

-   **4 × NVIDIA A100 (80GB)**

Supports:

✔ RL fine-tuning\
✔ Rendering-based reward loops\
✔ Large VLM experimentation

------------------------------------------------------------------------

## ⚙️ Training Strategy

### Phase 1 --- (Optional) SFT

Train on UI ↔ HTML/CSS pairs

### Phase 2 --- RLVR

Reward-based optimization using rendering similarity

------------------------------------------------------------------------

## 🚀 Setup

``` bash
git clone https://github.com/<repo>/rlvr-ui-codegen.git
cd rlvr-ui-codegen
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

``` bash
python generate.py --model checkpoints/rlvr_model --image samples/ui_example.png
```

Output:

    output/
     ├── index.html
     ├── styles.css
     └── render.png

------------------------------------------------------------------------

## 🛣 Roadmap

-   [ ] Dataset curation\
-   [ ] SFT baseline\
-   [ ] Reward design\
-   [ ] RLVR training\
-   [ ] Evaluation framework\
-   [ ] Ablation studies

------------------------------------------------------------------------

## 👥 Team

**Amal Joe**\
**Job J**

------------------------------------------------------------------------

## 📖 References

DeepSeek-R1\
DeepSeekMath / DeepSeekMath-V2\
Infinity Parser (LayoutRL)\
Efficient Medical VIE via RL\
Pix2Struct\
Nougat

------------------------------------------------------------------------

## 💡 Key Research Questions

-   Does RLVR improve visual layout fidelity?\
-   Can rewards replace heavy supervision?\
-   Does RLVR reduce hallucinated elements?\
-   How stable is rendering-based RL training?

