"""Style/token helpers for synthetic HTML templates."""

from __future__ import annotations

import random


def _hex_color(rng: random.Random, lo: int = 40, hi: int = 220) -> str:
    return f"#{rng.randint(lo, hi):02x}{rng.randint(lo, hi):02x}{rng.randint(lo, hi):02x}"


def _sample_list(rng: random.Random, pool: list[str], min_items: int = 2, max_items: int = 4) -> str:
    count = rng.randint(min_items, min(max_items, len(pool)))
    chosen = rng.sample(pool, count)
    return "".join(f"<li>{item}</li>" for item in chosen)


def _style_tokens(rng: random.Random) -> dict[str, int | str]:
    page_style = rng.choice(["solid", "solid", "dashed", "dotted", "double"])
    panel_style = rng.choice(["solid", "solid", "dashed", "dotted", "double"])
    table_style = rng.choice(["solid", "solid", "dashed", "dotted"])
    logo_style = rng.choice(["solid", "dashed", "dotted", "double"])
    line_palette = ["#111111", "#202a3a", "#2f3b50", "#55657e", "#8da1bf", "#c9d4e8"]
    line_color = rng.choice(line_palette)
    line_soft = rng.choice(["#dfe6f3", "#d4deee", "#c9d6ea", "#bccce4"])
    line_strong = rng.choice(["#101010", "#1a2537", "#304159", "#516480"])

    def _bw(style: str, lo: int, hi: int) -> int:
        low = max(lo, 2 if style == "double" else lo)
        return rng.randint(low, hi)

    return {
        "font_family": rng.choice(
            [
                '"Trebuchet MS", Arial, sans-serif',
                '"Verdana", "Segoe UI", sans-serif',
                '"Tahoma", Arial, sans-serif',
                '"Gill Sans", "Trebuchet MS", sans-serif',
                '"Lucida Sans", "Verdana", sans-serif',
            ]
        ),
        "base_font_size": rng.randint(12, 16),
        "page_width": rng.randint(1080, 1360),
        "page_margin": rng.randint(10, 24),
        "page_pad_v": rng.randint(14, 24),
        "page_pad_h": rng.randint(14, 26),
        "header_pad_bottom": rng.randint(8, 14),
        "header_margin_bottom": rng.randint(10, 18),
        "branding_gap": rng.randint(5, 11),
        "branding_row_gap": rng.randint(6, 12),
        "layout_gap": rng.randint(8, 16),
        "split_gap": rng.randint(8, 16),
        "hero_gap": rng.randint(8, 16),
        "hero_margin_bottom": rng.randint(4, 12),
        "hero_pad_v": rng.randint(6, 12),
        "hero_pad_h": rng.randint(8, 14),
        "line_color": line_color,
        "line_soft": line_soft,
        "line_strong": line_strong,
        "page_style": page_style,
        "panel_style": panel_style,
        "table_style": table_style,
        "logo_style": logo_style,
        "rule_style": rng.choice(["solid", "dashed", "dotted"]),
        "page_border": _bw(page_style, 1, 4),
        "page_radius": rng.choice([0, 0, 4, 8, 12, 16, 20, 24]),
        "panel_border": _bw(panel_style, 1, 4),
        "panel_radius": rng.choice([0, 0, 4, 8, 12, 16]),
        "table_border": _bw(table_style, 1, 3),
        "table_font": rng.randint(11, 14),
        "th_font": rng.randint(10, 13),
        "th_letter_spacing": rng.choice(["0.2px", "0.3px", "0.4px", "0.5px", "0.7px"]),
        "cell_pad": rng.randint(5, 11),
        "logo_border": _bw(logo_style, 1, 4),
        "logo_radius": rng.choice([0, 0, 4, 8, 12, 14]),
        "logo_min_w": rng.randint(82, 128),
        "logo_h": rng.randint(26, 44),
        "logo_font": rng.randint(9, 12),
        "logo_letter_spacing": rng.choice(["0.2px", "0.35px", "0.45px", "0.6px"]),
        "pic_sm_w": rng.randint(118, 186),
        "pic_sm_h": rng.randint(34, 54),
        "pic_lg_w": rng.randint(170, 268),
        "pic_lg_h": rng.randint(48, 76),
        "pic_font": rng.randint(9, 12),
        "pic_letter_spacing": rng.choice(["0.3px", "0.45px", "0.6px", "0.75px"]),
        "hero_radius": rng.choice([0, 4, 8, 12, 16]),
        "stripe": rng.choice([6, 8, 10, 12]),
        "h1_size": rng.randint(28, 36),
        "h1_weight": rng.choice([600, 700, 800]),
        "h1_letter_spacing": rng.choice(["0px", "0.1px", "0.2px", "0.4px"]),
        "subtitle_size": rng.randint(12, 15),
        "subtitle_letter_spacing": rng.choice(["0px", "0.2px", "0.3px", "0.5px"]),
        "bulky_size": rng.randint(40, 58),
        "bulky_weight": rng.choice([700, 800, 900]),
        "bulky_letter_spacing": rng.choice(["0px", "0.2px", "0.3px", "0.5px"]),
        "break_note_size": rng.randint(11, 14),
        "break_note_line": rng.choice(["1.15", "1.2", "1.25", "1.3"]),
        "section_size": rng.randint(14, 18),
        "section_margin_top": rng.randint(2, 8),
        "section_margin_bottom": rng.randint(4, 10),
        "list_indent": rng.randint(14, 24),
        "list_item_margin": rng.randint(1, 4),
        "detail_font": rng.randint(10, 12),
        "detail_value_size": rng.randint(20, 30),
        "bulk_kicker_size": rng.randint(9, 12),
        "bulk_number_size": rng.randint(26, 50),
        "bulk_note_size": rng.randint(11, 14),
        "bulk_note_line": rng.choice(["1.15", "1.2", "1.25", "1.3"]),
        "pill_radius": rng.choice([0, 4, 10, 99]),
    }
