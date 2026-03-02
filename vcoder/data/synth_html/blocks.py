"""Reusable HTML blocks and shared base layout."""

from __future__ import annotations

import random

from .constants import BRAND_LOGO_LABELS, BRAND_PICTURE_LABELS, BULKY_TEXT_SNIPPETS, TASKS
from .styles import _style_tokens


def _bulk_statement_html(rng: random.Random) -> str:
    return (
        "<div class='split-2'>"
        "<div class='bulk-panel'>"
        "<div class='bulk-kicker'>Executive Summary</div>"
        f"<div class='bulk-number'>{rng.randint(120, 980)}</div>"
        f"<div class='bulk-note'>{rng.choice(BULKY_TEXT_SNIPPETS)}<br />"
        f"Cycle target moves by {rng.randint(2, 19)}% in next phase.</div>"
        "</div>"
        "<div class='bulk-panel'>"
        "<div class='bulk-kicker'>Narrative Block</div>"
        f"<div class='bulk-note'>{rng.choice(BULKY_TEXT_SNIPPETS)}<br />"
        f"Cross-team actions: {rng.choice(TASKS)}, {rng.choice(TASKS)}.</div>"
        "</div>"
        "</div>"
    )


def _nested_metrics_table(rng: random.Random) -> str:
    mode = rng.random()
    if mode < 0.38:
        return (
            "<div class='detail-card'>"
            "<div class='detail-title'>Runtime Snapshot</div>"
            f"<div class='detail-value'>{rng.randint(60, 390)} ms</div>"
            f"<div class='detail-note'>Error {rng.randint(1, 80) / 100:.2f}%<br />CPU {rng.randint(35, 96)}%</div>"
            "</div>"
        )
    if mode < 0.68:
        return (
            "<div class='detail-list'>"
            "<div class='detail-title'>Metric Set</div>"
            "<ul>"
            f"<li>P95: {rng.randint(70, 440)} ms</li>"
            f"<li>Error: {rng.randint(1, 80) / 100:.2f}%</li>"
            f"<li>CPU: {rng.randint(35, 96)}%</li>"
            "</ul>"
            "</div>"
        )
    rows = []
    rows.append(f"<tr><td>P95 latency</td><td>{rng.randint(70, 440)} ms</td></tr>")
    rows.append(f"<tr><td>Error rate</td><td>{rng.randint(1, 80) / 100:.2f}%</td></tr>")
    rows.append(f"<tr><td>CPU</td><td>{rng.randint(35, 96)}%</td></tr>")
    return (
        "<table class='inner-table'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _nested_timeline_table(rng: random.Random) -> str:
    mode = rng.random()
    if mode < 0.42:
        return (
            "<div class='detail-card'>"
            "<div class='detail-title'>Timeline Panel</div>"
            f"<div class='detail-value'>T+{rng.randint(1, 14)}</div>"
            f"<div class='detail-note'>{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d} checkpoint<br />"
            f"Window {rng.randint(2, 7)}h</div>"
            "</div>"
        )
    if mode < 0.70:
        return (
            "<div class='detail-list'>"
            "<div class='detail-title'>Runbook Steps</div>"
            "<ul>"
            f"<li>{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d} handoff</li>"
            f"<li>T+{rng.randint(1, 9)} verify</li>"
            f"<li>T+{rng.randint(10, 18)} closeout</li>"
            "</ul>"
            "</div>"
        )
    rows = []
    for _ in range(2):
        hour = rng.randint(0, 23)
        minute = rng.choice([0, 15, 30, 45])
        rows.append(f"<tr><td>{hour:02d}:{minute:02d}</td><td>T+{rng.randint(1, 11)}</td></tr>")
    return (
        "<table class='inner-table'>"
        "<thead><tr><th>Slot</th><th>Delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _branding_block_html(rng: random.Random, stamp_cycle: int, stamp_batch: int) -> tuple[str, str]:
    location = rng.choice(["top", "bottom"])
    align_mode = rng.choice(["left", "center", "justify"])
    size_mode = rng.choice(["sm", "md", "lg"])
    text_mode = rng.choice(["left", "center", "justify"])
    logo_a, logo_b = rng.sample(BRAND_LOGO_LABELS, 2)
    pic_a, pic_b = rng.sample(BRAND_PICTURE_LABELS, 2)

    html = (
        f"<div class='branding branding--{align_mode} branding--{size_mode}'>"
        f"<div class='branding-stamp align-{text_mode}'>Cycle {stamp_cycle} / Batch {stamp_batch}</div>"
        "<div class='branding-row'>"
        f"<div class='logo-box'>{logo_a}</div>"
        f"<div class='logo-box'>{logo_b}</div>"
        "</div>"
        "<div class='picture-row'>"
        f"<div class='picture-box lg'>{pic_a}</div>"
        f"<div class='picture-box sm'>{pic_b}</div>"
        "</div>"
        "</div>"
    )
    return location, html


def _simple_brand_block_html(rng: random.Random) -> tuple[str, str]:
    location = rng.choice(["top", "bottom"])
    align_mode = rng.choice(["left", "center", "justify"])
    text_mode = rng.choice(["left", "center", "justify"])
    size_mode = rng.choice(["sm", "md", "lg"])
    logo_a, logo_b = rng.sample(BRAND_LOGO_LABELS, 2)
    pic = rng.choice(BRAND_PICTURE_LABELS)
    align_css = {"left": "flex-start", "center": "center", "justify": "space-between"}[align_mode]
    size_map = {
        "sm": (88, 30, 144, 38),
        "md": (118, 40, 188, 50),
        "lg": (146, 52, 232, 64),
    }
    logo_w, logo_h, pic_w, pic_h = size_map[size_mode]

    html = (
        f"<div class='simple-brand align-{text_mode}'>"
        f"<div class='simple-brand-stamp'>Cycle {rng.randint(100, 999)} / Batch {rng.randint(10, 99)}</div>"
        f"<div class='simple-brand-row' style='justify-content:{align_css};'>"
        f"<div class='simple-logo' style='min-width:{logo_w}px;height:{logo_h}px;'>{logo_a}</div>"
        f"<div class='simple-logo' style='min-width:{logo_w}px;height:{logo_h}px;'>{logo_b}</div>"
        "</div>"
        f"<div class='simple-brand-row' style='justify-content:{align_css};'>"
        f"<div class='simple-pic' style='min-width:{pic_w}px;height:{pic_h}px;'>{pic}</div>"
        "</div>"
        "</div>"
    )
    return location, html


def _base_layout(
    title: str,
    subtitle: str,
    accent: str,
    body: str,
    stamp_cycle: int,
    stamp_batch: int,
    rng: random.Random,
    bg: str = "#edf1f7",
) -> str:
    branding_location, branding_html = _branding_block_html(rng, stamp_cycle, stamp_batch)
    branding_top = branding_html if branding_location == "top" else ""
    branding_bottom = branding_html if branding_location == "bottom" else ""
    hero_note_align = rng.choice(["left", "center", "justify"])
    tok = _style_tokens(rng)
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    :root {{
      --accent: {accent};
      --ink: #1a2433;
      --muted: #5e6b7e;
      --line: {tok['line_color']};
      --line-soft: {tok['line_soft']};
      --line-strong: {tok['line_strong']};
      --surface: #ffffff;
      --bg: {bg};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: {tok['font_family']};
      font-size: {tok['base_font_size']}px;
      color: var(--ink);
      background: var(--bg);
    }}
    .page {{
      width: {tok['page_width']}px;
      margin: {tok['page_margin']}px auto;
      background: var(--surface);
      border: {tok['page_border']}px {tok['page_style']} var(--line);
      border-radius: {tok['page_radius']}px;
      padding: {tok['page_pad_v']}px {tok['page_pad_h']}px {tok['page_pad_v'] + 2}px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      border-bottom: {tok['panel_border']}px {tok['rule_style']} var(--line);
      padding-bottom: {tok['header_pad_bottom']}px;
      margin-bottom: {tok['header_margin_bottom']}px;
    }}
    h1 {{
      margin: 0;
      font-size: {tok['h1_size']}px;
      line-height: 1.1;
      letter-spacing: {tok['h1_letter_spacing']};
      font-weight: {tok['h1_weight']};
    }}
    .subtitle {{
      margin-top: 4px;
      color: var(--muted);
      font-size: {tok['subtitle_size']}px;
      letter-spacing: {tok['subtitle_letter_spacing']};
    }}
    .align-left {{ text-align: left; }}
    .align-center {{ text-align: center; }}
    .align-justify {{ text-align: justify; }}
    .branding {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f6faff;
      padding: 8px 10px;
      margin: 6px 0 10px;
      display: grid;
      gap: {tok['branding_gap']}px;
    }}
    .branding-stamp {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    .branding-row, .picture-row {{
      display: flex;
      gap: {tok['branding_row_gap']}px;
      flex-wrap: wrap;
    }}
    .branding--left .branding-row,
    .branding--left .picture-row {{
      justify-content: flex-start;
    }}
    .branding--center .branding-row,
    .branding--center .picture-row {{
      justify-content: center;
    }}
    .branding--justify .branding-row,
    .branding--justify .picture-row {{
      justify-content: space-between;
    }}
    .logo-box {{
      min-width: {tok['logo_min_w']}px;
      height: {tok['logo_h']}px;
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: linear-gradient(135deg, #f6f9ff, #edf3ff);
      color: #4a5a74;
      font-size: {tok['logo_font']}px;
      font-weight: 700;
      letter-spacing: {tok['logo_letter_spacing']};
      display: flex;
      align-items: center;
      justify-content: center;
      text-transform: uppercase;
    }}
    .hero-row {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: {tok['hero_gap']}px;
      align-items: stretch;
      margin-bottom: {tok['hero_margin_bottom']}px;
    }}
    .hero-card {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      background: #f7faff;
      border-radius: {tok['hero_radius']}px;
      padding: {tok['hero_pad_v']}px {tok['hero_pad_h']}px;
    }}
    .bulky-label {{
      color: #5f6e82;
      font-size: {tok['th_font']}px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }}
    .bulky-text {{
      margin-top: 2px;
      font-size: {tok['bulky_size']}px;
      line-height: 0.95;
      font-weight: {tok['bulky_weight']};
      letter-spacing: {tok['bulky_letter_spacing']};
      color: #1f2f47;
    }}
    .break-note {{
      color: #3c4c65;
      font-size: {tok['break_note_size']}px;
      line-height: {tok['break_note_line']};
      font-weight: 600;
    }}
    .picture-box {{
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: repeating-linear-gradient(
        135deg,
        #f4f8ff,
        #f4f8ff {tok['stripe']}px,
        #ecf2fd {tok['stripe']}px,
        #ecf2fd {tok['stripe'] * 2}px
      );
      color: #4e5f7b;
      font-size: {tok['pic_font']}px;
      font-weight: 700;
      letter-spacing: {tok['pic_letter_spacing']};
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .picture-box.sm {{
      min-width: {tok['pic_sm_w']}px;
      min-height: {tok['pic_sm_h']}px;
    }}
    .picture-box.lg {{
      min-width: {tok['pic_lg_w']}px;
      min-height: {tok['pic_lg_h']}px;
    }}
    .branding--sm .logo-box {{
      min-width: 78px;
      height: 26px;
      font-size: 9px;
    }}
    .branding--sm .picture-box.sm {{
      min-width: 112px;
      min-height: 34px;
      font-size: 9px;
    }}
    .branding--sm .picture-box.lg {{
      min-width: 160px;
      min-height: 44px;
      font-size: 9px;
    }}
    .branding--lg .logo-box {{
      min-width: 128px;
      height: 40px;
      font-size: 11px;
    }}
    .branding--lg .picture-box.sm {{
      min-width: 178px;
      min-height: 52px;
      font-size: 11px;
    }}
    .branding--lg .picture-box.lg {{
      min-width: 248px;
      min-height: 70px;
      font-size: 11px;
    }}
    .layout {{
      display: grid;
      gap: {tok['layout_gap']}px;
    }}
    .split-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: {tok['split_gap']}px;
    }}
    .section-title {{
      font-size: {tok['section_size']}px;
      font-weight: 700;
      color: #253449;
      margin: {tok['section_margin_top']}px 0 {tok['section_margin_bottom']}px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fbfcff;
      border: {tok['table_border']}px {tok['table_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      overflow: hidden;
      font-size: {tok['table_font']}px;
    }}
    th {{
      background: #f2f6fd;
      color: #2f3f58;
      font-size: {tok['th_font']}px;
      text-transform: uppercase;
      letter-spacing: {tok['th_letter_spacing']};
    }}
    th, td {{
      border: {tok['table_border']}px {tok['table_style']} var(--line);
      padding: {tok['cell_pad']}px;
      vertical-align: top;
      text-align: left;
    }}
    .muted {{ color: var(--muted); }}
    .status {{
      display: inline-block;
      border-radius: {tok['pill_radius']}px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 700;
      color: #fff;
      background: var(--accent);
    }}
    .risk-high {{ background: #b42424; }}
    .risk-med {{ background: #c27712; }}
    .risk-low {{ background: #2a8550; }}
    ul {{
      margin: 0;
      padding-left: {tok['list_indent']}px;
    }}
    li {{ margin: {tok['list_item_margin']}px 0; }}
    .inner-table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: {tok['table_border']}px {tok['table_style']} var(--line-soft);
      font-size: {tok['detail_font']}px;
    }}
    .inner-table th, .inner-table td {{
      border: {tok['table_border']}px {tok['table_style']} var(--line-soft);
      padding: 4px 5px;
    }}
    .inner-table th {{
      background: #eef3fb;
      font-size: {max(9, int(tok['detail_font']) - 1)}px;
    }}
    .detail-card {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line-soft);
      border-radius: {tok['panel_radius']}px;
      background: #f7fbff;
      padding: 6px 7px;
    }}
    .detail-list {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line-soft);
      border-radius: {tok['panel_radius']}px;
      background: #fbfdff;
      padding: 5px 7px;
      font-size: {tok['detail_font']}px;
    }}
    .detail-title {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      color: #51627d;
      margin-bottom: 4px;
      font-weight: 700;
    }}
    .detail-value {{
      font-size: {tok['detail_value_size']}px;
      line-height: 0.95;
      color: #233956;
      font-weight: 800;
      margin-bottom: 3px;
    }}
    .detail-note {{
      color: #4f6078;
      font-size: {tok['detail_font']}px;
      line-height: {tok['bulk_note_line']};
      font-weight: 600;
    }}
    .bulk-panel {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f8fbff;
      padding: 8px 10px;
    }}
    .bulk-kicker {{
      color: #62718a;
      font-size: {tok['bulk_kicker_size']}px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      font-weight: 700;
    }}
    .bulk-number {{
      font-size: {tok['bulk_number_size']}px;
      line-height: 0.95;
      font-weight: 800;
      color: #23364f;
      margin: 2px 0 4px;
    }}
    .bulk-note {{
      color: #4c5b72;
      font-size: {tok['bulk_note_size']}px;
      line-height: {tok['bulk_note_line']};
      font-weight: 600;
    }}
    .foot-note {{
      margin-top: 8px;
      color: #6b778b;
      font-size: {tok['detail_font']}px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
      </div>
    </div>
    {branding_top}
    <div class="hero-row">
      <div class="hero-card">
        <div class="bulky-label">Portfolio Index</div>
        <div class="bulky-text">{stamp_cycle + stamp_batch}</div>
      </div>
      <div class="hero-card break-note align-{hero_note_align}">
        Regional planning cycle<br />
        includes legal, finance,<br />
        reliability, and brand ops.
      </div>
    </div>
    <div class="layout">
      {body}
    </div>
    {branding_bottom}
  </div>
</body>
</html>
""".strip()
