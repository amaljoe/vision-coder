"""Synthetic HTML template definitions."""

from __future__ import annotations

import random
from typing import Callable

from .blocks import (
    _base_layout,
    _bulk_statement_html,
    _nested_metrics_table,
    _nested_timeline_table,
    _simple_brand_block_html,
)
from .constants import CHECKS, MATERIALS, OWNERS, REGIONS, RISK_CODES, SERVICE_NAMES, TASKS
from .styles import _hex_color, _sample_list, _style_tokens

TemplateFn = Callable[[random.Random], str]


def _simple_dashboard_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    brand_pos, brand_html = _simple_brand_block_html(rng)
    brand_top = brand_html if brand_pos == "top" else ""
    brand_bottom = brand_html if brand_pos == "bottom" else ""
    tok = _style_tokens(rng)
    cards = []
    for label in ["Users", "Orders", "Revenue", "Tickets"]:
        cards.append(
            "<div class='card'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{rng.randint(20, 9800)}</div>"
            f"<div class='delta'>{rng.choice(['+', '-'])}{rng.randint(1, 42)}%</div>"
            "</div>"
        )

    table_rows = []
    for _ in range(4):
        table_rows.append(
            "<tr>"
            f"<td>{rng.choice(REGIONS)}</td>"
            f"<td>{rng.randint(60, 980)}</td>"
            f"<td>{rng.randint(4, 99)}%</td>"
            "</tr>"
        )

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    :root {{
      --line: {tok['line_color']};
      --line-soft: {tok['line_soft']};
      --line-strong: {tok['line_strong']};
    }}
    body {{ margin: 0; font-family: {tok['font_family']}; font-size: {tok['base_font_size']}px; background: #f1f5fb; }}
    .wrap {{ width: {tok['page_width']}px; margin: {tok['page_margin']}px auto; background: #fff; border: {tok['page_border']}px {tok['page_style']} var(--line); border-radius: {tok['page_radius']}px; padding: {tok['page_pad_v']}px {tok['page_pad_h']}px; }}
    .align-left {{ text-align: left; }}
    .align-center {{ text-align: center; }}
    .align-justify {{ text-align: justify; }}
    .simple-brand {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f7faff;
      padding: {tok['hero_pad_v']}px {tok['hero_pad_h']}px;
      margin-bottom: 10px;
      display: grid;
      gap: {tok['branding_gap']}px;
    }}
    .simple-brand-stamp {{ color: #4f607a; font-size: {tok['subtitle_size']}px; font-weight: {tok['h1_weight']}; letter-spacing: {tok['subtitle_letter_spacing']}; }}
    .simple-brand-row {{ display: flex; gap: {tok['branding_row_gap']}px; flex-wrap: wrap; }}
    .simple-logo, .simple-pic {{
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: repeating-linear-gradient(135deg, #f5f9ff, #f5f9ff {tok['stripe']}px, #edf3ff {tok['stripe']}px, #edf3ff {tok['stripe'] * 2}px);
      color: #4d5e78;
      font-size: {tok['logo_font']}px;
      font-weight: 700;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 8px;
      letter-spacing: {tok['logo_letter_spacing']};
    }}
    .top {{ display: flex; justify-content: space-between; gap: {tok['hero_gap']}px; margin-bottom: {tok['hero_margin_bottom']}px; }}
    .title {{ font-size: {tok['h1_size']}px; font-weight: {tok['h1_weight']}; letter-spacing: {tok['h1_letter_spacing']}; margin-bottom: 3px; }}
    .subtitle {{ font-size: {tok['subtitle_size']}px; line-height: {tok['break_note_line']}; color: #56657c; letter-spacing: {tok['subtitle_letter_spacing']}; }}
    .logo-row {{ display: flex; gap: {tok['branding_row_gap']}px; }}
    .picture-box {{
      width: {tok['pic_sm_w']}px;
      height: {tok['pic_sm_h']}px;
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: repeating-linear-gradient(135deg, #f5f9ff, #f5f9ff {tok['stripe']}px, #edf3ff {tok['stripe']}px, #edf3ff {tok['stripe'] * 2}px);
      color: #4d5e78;
      font-size: {tok['pic_font']}px;
      font-weight: 700;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: center;
      letter-spacing: {tok['pic_letter_spacing']};
    }}
    .hero {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f7faff;
      padding: {tok['hero_pad_v']}px {tok['hero_pad_h']}px;
      margin-bottom: 10px;
    }}
    .hero-label {{ color: #5b6b82; font-size: {tok['th_font']}px; text-transform: uppercase; letter-spacing: {tok['th_letter_spacing']}; }}
    .hero-value {{ font-size: {tok['bulky_size']}px; line-height: 0.95; font-weight: {tok['bulky_weight']}; letter-spacing: {tok['bulky_letter_spacing']}; color: #1f3048; }}
    .hero-note {{ color: #4a5972; font-size: {tok['break_note_size']}px; line-height: {tok['break_note_line']}; font-weight: 600; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: {tok['layout_gap']}px; margin-bottom: {tok['section_margin_bottom'] + 6}px; }}
    .card {{ border: {tok['panel_border']}px {tok['panel_style']} var(--line); border-top: {tok['panel_border'] + 2}px {tok['rule_style']} {accent}; border-radius: {tok['panel_radius']}px; padding: 10px; background: #fcfdff; }}
    .label {{ color: #57657d; font-size: {tok['th_font']}px; }}
    .value {{ font-size: {max(22, int(tok['bulky_size']) - 16)}px; font-weight: {tok['h1_weight']}; margin: 5px 0; }}
    .delta {{ color: #2f7e4f; font-size: {tok['subtitle_size']}px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: {tok['table_font']}px; }}
    th, td {{ border: {tok['table_border']}px {tok['table_style']} var(--line); padding: {tok['cell_pad']}px; text-align: left; }}
    th {{ background: #eff4fd; }}
  </style>
</head>
<body>
  <div class="wrap">
    {brand_top}
    <div class="top">
      <div>
        <div class="title">Weekly Snapshot {rng.randint(1, 52)}</div>
        <div class="subtitle">Commercial dashboard<br />with brand and partner placeholders.</div>
      </div>
      <div class="logo-row">
        <div class="picture-box">Company Logo</div>
        <div class="picture-box">Partner Logo</div>
      </div>
    </div>
    <div class="hero">
      <div class="hero-label">Commercial Index</div>
      <div class="hero-value">{rng.randint(1020, 9980)}</div>
      <div class="hero-note">Demand trend remains elevated<br />across primary sales channels.</div>
    </div>
    <div class="cards">{''.join(cards)}</div>
    <table>
      <thead><tr><th>Region</th><th>Active Accounts</th><th>Conversion</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    {brand_bottom}
  </div>
</body>
</html>
""".strip()


def _simple_pricing_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    brand_pos, brand_html = _simple_brand_block_html(rng)
    brand_top = brand_html if brand_pos == "top" else ""
    brand_bottom = brand_html if brand_pos == "bottom" else ""
    tok = _style_tokens(rng)
    prices = [rng.randint(9, 39), rng.randint(40, 99), rng.randint(100, 249)]
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    :root {{
      --line: {tok['line_color']};
      --line-soft: {tok['line_soft']};
      --line-strong: {tok['line_strong']};
    }}
    body {{ margin: 0; font-family: {tok['font_family']}; font-size: {tok['base_font_size']}px; background: #f6f7fb; }}
    .wrap {{ width: {tok['page_width']}px; margin: {tok['page_margin']}px auto; background: #fff; border: {tok['page_border']}px {tok['page_style']} var(--line); border-radius: {tok['page_radius']}px; padding: {tok['page_pad_v']}px {tok['page_pad_h']}px; }}
    .align-left {{ text-align: left; }}
    .align-center {{ text-align: center; }}
    .align-justify {{ text-align: justify; }}
    .simple-brand {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f8fbff;
      padding: {tok['hero_pad_v']}px {tok['hero_pad_h']}px;
      margin-bottom: 10px;
      display: grid;
      gap: {tok['branding_gap']}px;
    }}
    .simple-brand-stamp {{ color: #53637b; font-size: {tok['subtitle_size']}px; font-weight: {tok['h1_weight']}; letter-spacing: {tok['subtitle_letter_spacing']}; }}
    .simple-brand-row {{ display: flex; gap: {tok['branding_row_gap']}px; flex-wrap: wrap; }}
    .simple-logo, .simple-pic {{
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: repeating-linear-gradient(135deg, #f5f9ff, #f5f9ff {tok['stripe']}px, #edf3ff {tok['stripe']}px, #edf3ff {tok['stripe'] * 2}px);
      color: #4d5e78;
      font-size: {tok['logo_font']}px;
      font-weight: 700;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 8px;
      letter-spacing: {tok['logo_letter_spacing']};
    }}
    .top {{ display: flex; justify-content: space-between; align-items: center; gap: {tok['hero_gap']}px; margin-bottom: {tok['hero_margin_bottom']}px; }}
    .title {{ font-size: {tok['h1_size']}px; font-weight: {tok['h1_weight']}; letter-spacing: {tok['h1_letter_spacing']}; text-align: left; margin-bottom: 2px; }}
    .subtitle {{ color: #596882; font-size: {tok['subtitle_size']}px; line-height: {tok['break_note_line']}; letter-spacing: {tok['subtitle_letter_spacing']}; }}
    .logo-row {{ display: flex; gap: {tok['branding_row_gap']}px; }}
    .picture-box {{
      width: {tok['pic_sm_w']}px;
      height: {tok['pic_sm_h']}px;
      border: {tok['logo_border']}px {tok['logo_style']} var(--line-strong);
      border-radius: {tok['logo_radius']}px;
      background: repeating-linear-gradient(135deg, #f5f9ff, #f5f9ff {tok['stripe']}px, #edf3ff {tok['stripe']}px, #edf3ff {tok['stripe'] * 2}px);
      color: #4d5e78;
      font-size: {tok['pic_font']}px;
      font-weight: 700;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: center;
      letter-spacing: {tok['pic_letter_spacing']};
    }}
    .hero {{
      border: {tok['panel_border']}px {tok['panel_style']} var(--line);
      border-radius: {tok['panel_radius']}px;
      background: #f9fbff;
      padding: {tok['hero_pad_v']}px {tok['hero_pad_h']}px;
      margin-bottom: 10px;
    }}
    .hero-label {{ color: #627189; font-size: {tok['th_font']}px; text-transform: uppercase; letter-spacing: {tok['th_letter_spacing']}; }}
    .hero-value {{ font-size: {tok['bulky_size']}px; line-height: 0.95; font-weight: {tok['bulky_weight']}; letter-spacing: {tok['bulky_letter_spacing']}; color: #1e2f47; }}
    .hero-note {{ color: #4d5d76; font-size: {tok['break_note_size']}px; line-height: {tok['break_note_line']}; font-weight: 600; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: {tok['layout_gap']}px; }}
    .card {{ border: {tok['panel_border']}px {tok['panel_style']} var(--line); border-radius: {tok['panel_radius']}px; padding: 14px; background: #fcfdff; text-align: center; }}
    .pro {{ border: {tok['panel_border'] + 1}px {tok['rule_style']} {accent}; }}
    .name {{ font-size: {max(16, int(tok['h1_size']) - 14)}px; font-weight: {tok['h1_weight']}; }}
    .price {{ font-size: {max(30, int(tok['bulky_size']) - 6)}px; font-weight: {tok['bulky_weight']}; margin: 8px 0; }}
    .line {{ color: #5f6c80; margin: 5px 0; font-size: {tok['subtitle_size']}px; }}
    .btn {{ margin-top: 8px; display: inline-block; border-radius: {tok['pill_radius']}px; background: {accent}; color: #fff; padding: 8px 14px; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    {brand_top}
    <div class="top">
      <div>
        <div class="title">Simple Pricing {rng.randint(2026, 2032)}</div>
        <div class="subtitle">Plan lineup with logo placeholders<br />and bold conversion headline.</div>
      </div>
      <div class="logo-row">
        <div class="picture-box">Company Mark</div>
        <div class="picture-box">Reseller Mark</div>
      </div>
    </div>
    <div class="hero">
      <div class="hero-label">Quarterly ARR Projection</div>
      <div class="hero-value">${rng.randint(120, 980)}K</div>
      <div class="hero-note">Promotion burst starts next cycle<br />for Pro and Scale packages.</div>
    </div>
    <div class="cards">
      <div class="card">
        <div class="name">Starter</div>
        <div class="price">${prices[0]}</div>
        <div class="line">{rng.randint(3, 20)} projects</div>
        <div class="line">{rng.randint(10, 160)} GB storage</div>
        <div class="btn">Choose</div>
      </div>
      <div class="card pro">
        <div class="name">Pro</div>
        <div class="price">${prices[1]}</div>
        <div class="line">{rng.randint(21, 80)} projects</div>
        <div class="line">{rng.randint(180, 900)} GB storage</div>
        <div class="btn">Choose</div>
      </div>
      <div class="card">
        <div class="name">Scale</div>
        <div class="price">${prices[2]}</div>
        <div class="line">{rng.randint(81, 240)} projects</div>
        <div class="line">{rng.randint(1, 8)} TB storage</div>
        <div class="btn">Choose</div>
      </div>
    </div>
    {brand_bottom}
  </div>
</body>
</html>
""".strip()


def _governance_ledger_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    main_rows = []
    for _ in range(4):
        service = rng.choice(SERVICE_NAMES)
        owner = rng.choice(OWNERS)
        risk = rng.choice(["HIGH", "MED", "LOW"])
        risk_class = "risk-high" if risk == "HIGH" else ("risk-med" if risk == "MED" else "risk-low")
        controls = f"<ul>{_sample_list(rng, CHECKS)}</ul>"
        main_rows.append(
            "<tr>"
            f"<td>{service}<br /><span class='muted'>pod-{rng.randint(1, 9)}</span></td>"
            f"<td>{owner}<br /><span class='muted'>shift {rng.choice(['A', 'B', 'C'])}</span></td>"
            f"<td><span class='status {risk_class}'>{risk}</span></td>"
            f"<td>{controls}<div class='muted'>Target window<br />Q{rng.randint(1,4)} review</div></td>"
            f"<td>{_nested_metrics_table(rng)}</td>"
            "</tr>"
        )

    risk_rows = []
    for code in rng.sample(RISK_CODES, 3):
        risk_rows.append(
            "<tr>"
            f"<td>{code}</td>"
            f"<td>{rng.choice(REGIONS)}<br /><span class='muted'>zone-{rng.randint(1, 4)}</span></td>"
            f"<td>{rng.randint(1, 9)}</td>"
            f"<td>{rng.randint(3, 40)}%</td>"
            f"<td><ul>{_sample_list(rng, TASKS, 2, 3)}</ul></td>"
            "</tr>"
        )

    schedule_rows = []
    for _ in range(3):
        schedule_rows.append(
            "<tr>"
            f"<td>{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-2026<br /><span class='muted'>phase-{rng.randint(1, 3)}</span></td>"
            f"<td>{rng.choice(OWNERS)}<br /><span class='muted'>on-call</span></td>"
            f"<td>{_nested_timeline_table(rng)}</td>"
            "</tr>"
        )

    body = f"""
{_bulk_statement_html(rng)}
<div class="section-title">Control Ledger</div>
<table>
  <thead><tr><th>Service</th><th>Owner</th><th>Risk</th><th>Required Controls</th><th>Live Metrics</th></tr></thead>
  <tbody>{''.join(main_rows)}</tbody>
</table>
<div class="split-2">
  <div>
    <div class="section-title">Risk Register</div>
    <table>
      <thead><tr><th>Code</th><th>Region</th><th>Impact</th><th>Exposure</th><th>Mitigation Plan</th></tr></thead>
      <tbody>{''.join(risk_rows)}</tbody>
    </table>
  </div>
  <div>
    <div class="section-title">Audit Calendar</div>
    <table>
      <thead><tr><th>Date</th><th>Lead</th><th>Timeline</th></tr></thead>
      <tbody>{''.join(schedule_rows)}</tbody>
    </table>
  </div>
</div>
<div class="foot-note">All values are synthetic and regenerated per sample seed.</div>
"""
    return _base_layout(
        "Program Governance Ledger",
        "Nested controls, risk tables, and schedule details",
        accent,
        body,
        stamp_cycle=rng.randint(100, 999),
        stamp_batch=rng.randint(10, 99),
        rng=rng,
    )


def _supply_chain_grid_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    route_rows = []
    for _ in range(4):
        route_rows.append(
            "<tr>"
            f"<td>R-{rng.randint(110, 899)}</td>"
            f"<td>{rng.choice(REGIONS)}-{rng.choice(REGIONS)}<br /><span class='muted'>lane-{rng.randint(10, 99)}</span></td>"
            f"<td>{rng.randint(2, 14)} d<br /><span class='muted'>{rng.randint(1, 5)} checkpoints</span></td>"
            f"<td><ul>{_sample_list(rng, MATERIALS, 2, 4)}</ul></td>"
            f"<td>{_nested_timeline_table(rng)}</td>"
            "</tr>"
        )

    vendor_rows = []
    for idx in range(3):
        vendor_rows.append(
            "<tr>"
            f"<td>V-{idx + 1:02d}</td>"
            f"<td>{rng.choice(['NorthStar', 'DeltaFab', 'ApexRoute', 'TerraLoop', 'Crescent'])}<br /><span class='muted'>tier-{rng.randint(1, 3)}</span></td>"
            f"<td>{rng.randint(72, 99)}%</td>"
            f"<td>{_nested_metrics_table(rng)}</td>"
            "</tr>"
        )

    inventory_rows = []
    for material in rng.sample(MATERIALS, 4):
        inventory_rows.append(
            "<tr>"
            f"<td>{material}</td>"
            f"<td>{rng.randint(60, 1800)}</td>"
            f"<td>{rng.randint(7, 55)}<br /><span class='muted'>reorder @ {rng.randint(5, 25)}</span></td>"
            f"<td><ul>{_sample_list(rng, ['Buffer', 'Dual source', 'Expedite', 'Inspect'], 2, 3)}</ul></td>"
            "</tr>"
        )

    body = f"""
{_bulk_statement_html(rng)}
<div class="split-2">
  <div>
    <div class="section-title">Route Complexity Matrix</div>
    <table>
      <thead><tr><th>Route</th><th>Lane</th><th>SLA</th><th>Critical Materials</th><th>Window Detail</th></tr></thead>
      <tbody>{''.join(route_rows)}</tbody>
    </table>
  </div>
  <div>
    <div class="section-title">Vendor Stability Map</div>
    <table>
      <thead><tr><th>ID</th><th>Vendor</th><th>Fill Rate</th><th>Inline Scorecard</th></tr></thead>
      <tbody>{''.join(vendor_rows)}</tbody>
    </table>
  </div>
</div>
<div class="section-title">Buffer Inventory Table</div>
<table>
  <thead><tr><th>Material</th><th>Stock Units</th><th>Days Left</th><th>Response Pack</th></tr></thead>
  <tbody>{''.join(inventory_rows)}</tbody>
</table>
"""
    return _base_layout(
        "Supply Chain Control Grid",
        "Multiple table layouts with nested timing and metrics",
        accent,
        body,
        stamp_cycle=rng.randint(100, 999),
        stamp_batch=rng.randint(10, 99),
        rng=rng,
        bg="#edf3f1",
    )


def _incident_command_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    board_rows = []
    for _ in range(4):
        sev = rng.choice(["SEV-1", "SEV-2", "SEV-3"])
        sev_class = "risk-high" if sev == "SEV-1" else ("risk-med" if sev == "SEV-2" else "risk-low")
        board_rows.append(
            "<tr>"
            f"<td><span class='status {sev_class}'>{sev}</span></td>"
            f"<td>{rng.choice(SERVICE_NAMES)}<br /><span class='muted'>node-{rng.randint(10, 99)}</span></td>"
            f"<td>{rng.randint(4, 120)} min<br /><span class='muted'>owner ack {rng.randint(1, 12)}m</span></td>"
            f"<td>{_nested_timeline_table(rng)}</td>"
            f"<td><ul>{_sample_list(rng, ['Escalate', 'Trace', 'Rollback', 'Patch', 'Notify'], 2, 4)}</ul></td>"
            "</tr>"
        )

    queue_rows = []
    for _ in range(4):
        queue_rows.append(
            "<tr>"
            f"<td>Q-{rng.randint(1000, 9999)}</td>"
            f"<td>{rng.randint(8, 520)}</td>"
            f"<td>{rng.randint(1, 90)}%</td>"
            f"<td>{_nested_metrics_table(rng)}</td>"
            "</tr>"
        )

    ownership_rows = []
    for owner in rng.sample(OWNERS, 3):
        ownership_rows.append(
            "<tr>"
            f"<td>{owner}<br /><span class='muted'>rotation {rng.randint(1, 4)}</span></td>"
            f"<td>{rng.randint(2, 16)}</td>"
            f"<td><ul>{_sample_list(rng, TASKS, 2, 3)}</ul></td>"
            "</tr>"
        )

    body = f"""
{_bulk_statement_html(rng)}
<div class="section-title">Incident Escalation Board</div>
<table>
  <thead><tr><th>Severity</th><th>Service</th><th>Age</th><th>Response Timeline</th><th>Active Actions</th></tr></thead>
  <tbody>{''.join(board_rows)}</tbody>
</table>
<div class="split-2">
  <div>
    <div class="section-title">Queue Health Table</div>
    <table>
      <thead><tr><th>Queue</th><th>Depth</th><th>Drain</th><th>Resource Metrics</th></tr></thead>
      <tbody>{''.join(queue_rows)}</tbody>
    </table>
  </div>
  <div>
    <div class="section-title">Ownership Rotation</div>
    <table>
      <thead><tr><th>Lead</th><th>Open Tasks</th><th>Checklist</th></tr></thead>
      <tbody>{''.join(ownership_rows)}</tbody>
    </table>
  </div>
</div>
"""
    return _base_layout(
        "Incident Command Console",
        "Escalation board + queue metrics + ownership tables",
        accent,
        body,
        stamp_cycle=rng.randint(100, 999),
        stamp_batch=rng.randint(10, 99),
        rng=rng,
        bg="#f4f0ed",
    )


def _portfolio_planning_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    program_rows = []
    for _ in range(3):
        program_rows.append(
            "<tr>"
            f"<td>P-{rng.randint(10, 99)}</td>"
            f"<td>{rng.choice(['Atlas', 'Orion', 'Nimbus', 'Vector', 'Comet'])}<br /><span class='muted'>stream-{rng.randint(1, 7)}</span></td>"
            f"<td>{rng.randint(20, 95)}%</td>"
            f"<td>{rng.randint(2, 11)} / {rng.randint(12, 24)}</td>"
            f"<td>{_nested_metrics_table(rng)}</td>"
            "</tr>"
        )

    capacity_rows = []
    for team in ["Platform", "Infra", "Data", "Security", "QA"]:
        capacity_rows.append(
            "<tr>"
            f"<td>{team}</td>"
            f"<td>{rng.randint(4, 20)}</td>"
            f"<td>{rng.randint(55, 98)}%</td>"
            f"<td><ul>{_sample_list(rng, ['Backfill', 'Contractor', 'Cross-train', 'De-scope'], 2, 3)}</ul></td>"
            "</tr>"
        )

    milestone_rows = []
    for _ in range(4):
        milestone_rows.append(
            "<tr>"
            f"<td>M-{rng.randint(100, 799)}</td>"
            f"<td>{rng.randint(1, 12):02d}/2026<br /><span class='muted'>wk-{rng.randint(1, 52)}</span></td>"
            f"<td>{rng.choice(['Planned', 'At Risk', 'On Hold'])}</td>"
            f"<td>{_nested_timeline_table(rng)}</td>"
            "</tr>"
        )

    body = f"""
{_bulk_statement_html(rng)}
<div class="section-title">Program Budget Table</div>
<table>
  <thead><tr><th>Program</th><th>Name</th><th>Spend</th><th>Used / Planned</th><th>KPI Breakdown</th></tr></thead>
  <tbody>{''.join(program_rows)}</tbody>
</table>
<div class="split-2">
  <div>
    <div class="section-title">Team Capacity Matrix</div>
    <table>
      <thead><tr><th>Team</th><th>Headcount</th><th>Utilization</th><th>Actions</th></tr></thead>
      <tbody>{''.join(capacity_rows)}</tbody>
    </table>
  </div>
  <div>
    <div class="section-title">Milestone Dependency Table</div>
    <table>
      <thead><tr><th>ID</th><th>Target</th><th>Status</th><th>Window Detail</th></tr></thead>
      <tbody>{''.join(milestone_rows)}</tbody>
    </table>
  </div>
</div>
"""
    return _base_layout(
        "Portfolio Planning Board",
        "Dense multi-table layout with nested details per cell",
        accent,
        body,
        stamp_cycle=rng.randint(100, 999),
        stamp_batch=rng.randint(10, 99),
        rng=rng,
        bg="#eef2f8",
    )


def _research_registry_html(rng: random.Random) -> str:
    accent = _hex_color(rng)
    study_rows = []
    for _ in range(4):
        study_rows.append(
            "<tr>"
            f"<td>S-{rng.randint(101, 999)}</td>"
            f"<td>{rng.choice(['Vision', 'NLP', 'Robotics', 'Ranking', 'Safety'])}<br /><span class='muted'>track-{rng.randint(1, 5)}</span></td>"
            f"<td>{rng.randint(10, 88)}%</td>"
            f"<td><ul>{_sample_list(rng, ['Annotate', 'Train', 'Review', 'Re-run', 'Ablate'], 2, 4)}</ul></td>"
            f"<td>{_nested_metrics_table(rng)}</td>"
            "</tr>"
        )

    ethics_rows = []
    for _ in range(3):
        ethics_rows.append(
            "<tr>"
            f"<td>E-{rng.randint(20, 99)}</td>"
            f"<td>{rng.choice(['Privacy', 'Bias', 'Attribution', 'Safety'])}</td>"
            f"<td>{rng.choice(['Open', 'Mitigated', 'Deferred'])}</td>"
            f"<td>{_nested_timeline_table(rng)}</td>"
            "</tr>"
        )

    benchmark_rows = []
    for _ in range(4):
        benchmark_rows.append(
            "<tr>"
            f"<td>B-{rng.randint(200, 999)}</td>"
            f"<td>{rng.randint(50, 98)}.{rng.randint(0, 9)}</td>"
            f"<td>{rng.randint(20, 130)} ms</td>"
            f"<td>{rng.randint(1, 12)} GB</td>"
            "</tr>"
        )

    body = f"""
{_bulk_statement_html(rng)}
<div class="section-title">Study Registry</div>
<table>
  <thead><tr><th>Study</th><th>Domain</th><th>Progress</th><th>Open Work</th><th>Eval Snapshot</th></tr></thead>
  <tbody>{''.join(study_rows)}</tbody>
</table>
<div class="split-2">
  <div>
    <div class="section-title">Ethics Review Table</div>
    <table>
      <thead><tr><th>Case</th><th>Theme</th><th>Status</th><th>Review Timeline</th></tr></thead>
      <tbody>{''.join(ethics_rows)}</tbody>
    </table>
  </div>
  <div>
    <div class="section-title">Benchmark Matrix</div>
    <table>
      <thead><tr><th>Run</th><th>Score</th><th>Latency</th><th>Memory</th></tr></thead>
      <tbody>{''.join(benchmark_rows)}</tbody>
    </table>
  </div>
</div>
"""
    return _base_layout(
        "Research Operations Registry",
        "Tables with nested lists, nested metrics, and mixed layouts",
        accent,
        body,
        stamp_cycle=rng.randint(100, 999),
        stamp_batch=rng.randint(10, 99),
        rng=rng,
        bg="#f5f4ef",
    )


SIMPLE_TEMPLATES: tuple[tuple[str, TemplateFn], ...] = (
    ("simple_dashboard", _simple_dashboard_html),
    ("simple_pricing", _simple_pricing_html),
)

COMPLEX_TEMPLATES: tuple[tuple[str, TemplateFn], ...] = (
    ("governance_ledger", _governance_ledger_html),
    ("supply_chain_grid", _supply_chain_grid_html),
    ("incident_command", _incident_command_html),
    ("portfolio_planning", _portfolio_planning_html),
    ("research_registry", _research_registry_html),
)


def _build_html(sample_seed: int, template_group: str) -> tuple[str, str]:
    rng = random.Random(sample_seed)
    pool = SIMPLE_TEMPLATES if template_group == "simple" else COMPLEX_TEMPLATES
    name, fn = pool[rng.randrange(len(pool))]
    return name, fn(rng)
