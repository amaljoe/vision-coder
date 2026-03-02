"""Generate a small synthetic screenshot->HTML dataset.

Each sample is saved as:
    <output_dir>/sample_XXXX/source.html
    <output_dir>/sample_XXXX/render.png
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Callable

from vcoder.rendering.html_renderer import render_html_to_image

DEFAULT_OUTPUT_DIR = Path("outputs/synth_html")
DEFAULT_NUM_SAMPLES = 200
DEFAULT_SEED = 1337
PROGRESS_EVERY = 20
INDEX_FIELDS = ("sample_id", "template", "source_html", "render_png")
TemplateFn = Callable[[random.Random], str]


def _hex_color(rng: random.Random, lo: int = 40, hi: int = 220) -> str:
    return f"#{rng.randint(lo, hi):02x}{rng.randint(lo, hi):02x}{rng.randint(lo, hi):02x}"


def _dashboard_html(rng: random.Random) -> str:
    cards = []
    labels = ["Users", "Orders", "Revenue", "Tickets"]
    for label in labels:
        value = rng.randint(10, 9999)
        delta = rng.randint(-25, 45)
        sign = "+" if delta >= 0 else ""
        cards.append(
            f"""
            <div class="card">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="delta">{sign}{delta}%</div>
            </div>
            """
        )

    bg = _hex_color(rng, 230, 255)
    accent = _hex_color(rng)
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: {bg}; }}
    .wrap {{ padding: 28px; }}
    .title {{ font-size: 32px; font-weight: 700; color: #1b1f24; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
    .card {{ background: white; border-radius: 14px; padding: 18px; border-top: 4px solid {accent}; }}
    .label {{ color: #5b6470; font-size: 14px; }}
    .value {{ margin-top: 8px; font-size: 34px; font-weight: 700; color: #101419; }}
    .delta {{ margin-top: 6px; font-size: 14px; color: #3b6f3b; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Weekly Snapshot {rng.randint(1, 52)}</div>
    <div class="grid">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
""".strip()


def _pricing_html(rng: random.Random) -> str:
    prices = [rng.randint(9, 39), rng.randint(40, 99), rng.randint(100, 249)]
    accent = _hex_color(rng)
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7fb; }}
    .wrap {{ padding: 28px; }}
    .title {{ text-align: center; font-size: 34px; margin-bottom: 18px; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #d8dce6; border-radius: 16px; padding: 20px; text-align: center; }}
    .pro {{ border: 2px solid {accent}; transform: scale(1.03); }}
    .name {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
    .price {{ font-size: 42px; font-weight: 700; margin: 8px 0; }}
    .sub {{ color: #687181; margin-bottom: 10px; }}
    .btn {{ display: inline-block; margin-top: 10px; padding: 10px 16px; border-radius: 999px; color: white; background: {accent}; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Simple Pricing {rng.randint(2024, 2030)}</div>
    <div class="cards">
      <div class="card">
        <div class="name">Starter</div>
        <div class="price">${prices[0]}</div>
        <div class="sub">{rng.randint(3, 20)} projects</div>
        <div class="sub">{rng.randint(10, 200)} GB storage</div>
        <div class="btn">Choose</div>
      </div>
      <div class="card pro">
        <div class="name">Pro</div>
        <div class="price">${prices[1]}</div>
        <div class="sub">{rng.randint(21, 80)} projects</div>
        <div class="sub">{rng.randint(250, 900)} GB storage</div>
        <div class="btn">Choose</div>
      </div>
      <div class="card">
        <div class="name">Scale</div>
        <div class="price">${prices[2]}</div>
        <div class="sub">{rng.randint(81, 220)} projects</div>
        <div class="sub">{rng.randint(1, 8)} TB storage</div>
        <div class="btn">Choose</div>
      </div>
    </div>
  </div>
</body>
</html>
""".strip()


def _invoice_html(rng: random.Random) -> str:
    rows = []
    subtotal = 0
    for i in range(1, 5):
        qty = rng.randint(1, 6)
        unit = rng.randint(12, 160)
        line = qty * unit
        subtotal += line
        rows.append(
            f"<tr><td>Item {i}</td><td>{qty}</td><td>${unit}</td><td>${line}</td></tr>"
        )
    tax = round(subtotal * 0.08)
    total = subtotal + tax
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #eef2f7; }}
    .invoice {{ width: 920px; margin: 34px auto; background: white; border-radius: 14px; padding: 24px; }}
    .row {{ display: flex; justify-content: space-between; margin-bottom: 14px; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .id {{ color: #637184; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e4e9f2; text-align: left; }}
    .sum {{ margin-top: 14px; width: 320px; margin-left: auto; }}
    .sum div {{ display: flex; justify-content: space-between; margin: 6px 0; }}
    .grand {{ font-size: 26px; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="invoice">
    <div class="row"><div class="title">Invoice</div><div class="id">INV-{rng.randint(10000, 99999)}</div></div>
    <div class="row"><div>Issued: 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}</div><div>Due: 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}</div></div>
    <table>
      <thead><tr><th>Description</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <div class="sum">
      <div><span>Subtotal</span><span>${subtotal}</span></div>
      <div><span>Tax (8%)</span><span>${tax}</span></div>
      <div class="grand"><span>Grand Total</span><span>${total}</span></div>
    </div>
  </div>
</body>
</html>
""".strip()


def _leaderboard_html(rng: random.Random) -> str:
    rows = []
    for i in range(1, 7):
        score = rng.randint(1200, 9999)
        wins = rng.randint(1, 24)
        rows.append(f"<tr><td>{i}</td><td>Team {chr(64 + i)}</td><td>{wins}</td><td>{score}</td></tr>")

    accent = _hex_color(rng)
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0f1622; color: white; }}
    .wrap {{ width: 980px; margin: 24px auto; }}
    .title {{ font-size: 34px; font-weight: 700; margin-bottom: 10px; color: {accent}; }}
    .tag {{ margin-bottom: 12px; color: #afbbcf; }}
    table {{ width: 100%; border-collapse: collapse; background: #162235; border-radius: 12px; overflow: hidden; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid #223149; }}
    th {{ text-align: left; color: #8ea2c2; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Leaderboard Week {rng.randint(1, 52)}</div>
    <div class="tag">Updated at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}</div>
    <table>
      <thead><tr><th>#</th><th>Team</th><th>Wins</th><th>Score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
""".strip()


TEMPLATES: tuple[tuple[str, TemplateFn], ...] = (
    ("dashboard", _dashboard_html),
    ("pricing", _pricing_html),
    ("invoice", _invoice_html),
    ("leaderboard", _leaderboard_html),
)


def _build_html(sample_seed: int) -> tuple[str, str]:
    rng = random.Random(sample_seed)
    name, fn = TEMPLATES[rng.randrange(len(TEMPLATES))]
    return name, fn(rng)


def _write_sample(output_dir: Path, sample_id: str, template: str, html: str) -> dict[str, str]:
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    html_path = sample_dir / "source.html"
    png_path = sample_dir / "render.png"
    html_path.write_text(html, encoding="utf-8")
    render_html_to_image(html).save(png_path)

    return {
        "sample_id": sample_id,
        "template": template,
        "source_html": str(html_path),
        "render_png": str(png_path),
    }


def _write_index(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    index_path = output_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def generate_dataset(output_dir: Path, num_samples: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for idx in range(num_samples):
        sample_id = f"sample_{idx:04d}"
        template, html = _build_html(seed + idx)
        rows.append(_write_sample(output_dir, sample_id, template, html))

        if (idx + 1) % PROGRESS_EVERY == 0 or idx + 1 == num_samples:
            print(f"[generate] {idx + 1}/{num_samples}")

    index_path = _write_index(output_dir, rows)
    print(f"Saved dataset to: {output_dir}")
    print(f"Index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic HTML screenshot dataset.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    generate_dataset(args.output_dir, args.num_samples, args.seed)


if __name__ == "__main__":
    main()
