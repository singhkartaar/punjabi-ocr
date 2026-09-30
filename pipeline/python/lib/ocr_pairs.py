"""
Paragraph-level pairing and alignment between Gurbani stanzas (left column)
and Punjabi translation/commentary paragraphs (right column).

Ensures that:
1. Gurbani verse stanzas on the left are segmented at the paragraph level (by danda markers and vertical gaps).
2. Translation prose on the right is segmented at the paragraph level (by sentence terminators and vertical gaps).
3. Left and right paragraphs are correlated by vertical spatial alignment [y0, y1], preserving structural integrity.
4. Individual lines are cleanly kept under their respective paragraph blocks.
"""

from __future__ import annotations
import re
from typing import Any


def is_danda_ending(text: str) -> bool:
    """Check if Gurmukhi text ends in double danda or verse terminator."""
    t = text.rstrip(" _\t\r\n")
    if not t:
        return False
    # Check for ॥, ॥੧॥, ॥ ਰਹਾਉ ॥, etc.
    return bool(re.search(r"(॥|॥\s*\d+\s*॥|॥\s*ਰਹਾਉ\s*॥|[।॥]\s*$)", t))


def is_sentence_ending(text: str) -> bool:
    """Check if Punjabi prose line ends with a sentence/paragraph terminator."""
    t = text.rstrip(" _\t\r\n")
    if not t:
        return False
    return t.endswith(("।", "॥", ".", "\"", "”", "'", "!", "?", ")", "।॥"))


def segment_column_paragraphs(
    lines: list[dict[str, Any]],
    is_left: bool,
    min_gap: int = 30,
    punct_gap: int = 16
) -> list[list[dict[str, Any]]]:
    """
    Segment a column of lines into coherent paragraphs/stanzas.
    
    Args:
        lines: Sorted list of line dicts with 'bbox' [x0, y0, x1, y1] and 'text'.
        is_left: True if Gurbani verse (left column), False if translation (right column).
        min_gap: Gap in pixels that unconditionally splits paragraphs.
        punct_gap: Gap in pixels that splits paragraphs if previous line has ending punctuation.
    """
    if not lines:
        return []

    paragraphs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        current.append(line)
        if i + 1 >= len(lines):
            paragraphs.append(current)
            current = []
            break

        next_line = lines[i + 1]
        gap = next_line["bbox"][1] - line["bbox"][3]
        text = line.get("text", "").strip()

        if is_left:
            # Gurbani stanza ends if:
            # - ends with ॥ and has some vertical separation (>= punct_gap)
            # - or has a large gap (>= min_gap)
            ends_verse = is_danda_ending(text)
            if (ends_verse and gap >= punct_gap) or gap >= min_gap:
                paragraphs.append(current)
                current = []
        else:
            # Translation paragraph ends if:
            # - ends with punctuation and has moderate gap (>= punct_gap)
            # - or has a large gap (>= min_gap)
            ends_sent = is_sentence_ending(text)
            if (ends_sent and gap >= punct_gap) or gap >= min_gap:
                paragraphs.append(current)
                current = []

    if current:
        paragraphs.append(current)

    return paragraphs


def align_page_pairs(
    lines: list[dict[str, Any]],
    page_h: int = 3000,
    page_w: int = 2000
) -> dict[str, Any]:
    """
    Align a page into header, footer, footnotes, and paired Gurbani ↔ Translation blocks.
    
    Returns:
        dict with:
          - header: list[line]
          - footer: list[line]
          - footnotes: list[line]
          - is_two_col: bool
          - pairs: list[dict] where each dict has:
              pair_id: int
              y_range: [y0, y1]
              gurbani: dict(text, lines, bbox) or None
              translation: dict(text, lines, bbox)
    """
    # Separate zones
    header: list[dict[str, Any]] = []
    footer: list[dict[str, Any]] = []
    footnotes: list[dict[str, Any]] = []
    col0_lines: list[dict[str, Any]] = []
    col1_lines: list[dict[str, Any]] = []

    # Thresholds for headers and footers
    head_y_max = int(page_h * 0.08)
    foot_y_min = int(page_h * 0.92)

    for l in lines:
        b = l.get("bbox", [0, 0, 0, 0])
        col = l.get("col")
        zone = l.get("zone")

        if zone == "header" or (col is None and b[1] < head_y_max):
            header.append(l)
        elif zone == "footer" or (col is None and b[1] > foot_y_min and "ਪੰਨਾ" in l.get("text", "")):
            footer.append(l)
        elif zone == "footnote" or (b[1] > int(page_h * 0.88) and l.get("text", "").strip().startswith(("*", "੧", "੨", "੩", "੪", "੫", "੬", "੭", "੮", "੯", "।")) and col is None):
            footnotes.append(l)
        elif col == 1:
            col1_lines.append(l)
        elif col == 0:
            col0_lines.append(l)
        else:
            # Single-column body line
            col0_lines.append(l)

    col0_lines.sort(key=lambda x: x["bbox"][1])
    col1_lines.sort(key=lambda x: x["bbox"][1])

    is_two_col = len(col1_lines) >= 3

    if not is_two_col:
        # Single column page: group col0 into prose paragraphs
        paras0 = segment_column_paragraphs(col0_lines, is_left=False, min_gap=30, punct_gap=18)
        pairs = []
        for idx, p in enumerate(paras0):
            p_text = " ".join(l.get("text", "").strip() for l in p)
            b = [
                min(l["bbox"][0] for l in p),
                min(l["bbox"][1] for l in p),
                max(l["bbox"][2] for l in p),
                max(l["bbox"][3] for l in p),
            ]
            pairs.append({
                "pair_id": idx + 1,
                "y_range": [b[1], b[3]],
                "gurbani": None,
                "translation": {
                    "text": p_text,
                    "lines": p,
                    "bbox": b,
                    "kind": "prose"
                }
            })
        return {
            "header": header,
            "footer": footer,
            "footnotes": footnotes,
            "is_two_col": False,
            "pairs": pairs
        }

    # Two-column page:
    # 1. Segment col1 into translation paragraphs
    t_paras = segment_column_paragraphs(col1_lines, is_left=False, min_gap=32, punct_gap=16)

    # 2. Segment col0 into Gurbani stanzas
    g_stanzas = segment_column_paragraphs(col0_lines, is_left=True, min_gap=32, punct_gap=18)

    # 3. Correlate Gurbani stanzas and Translation paragraphs by vertical overlap
    g_assigned: list[list[list[dict[str, Any]]]] = [[] for _ in t_paras]
    for g in g_stanzas:
        gy0 = g[0]["bbox"][1]
        gy1 = g[-1]["bbox"][3]
        best_t = 0
        best_overlap = -999999
        for idx, t in enumerate(t_paras):
            ty0 = t[0]["bbox"][1]
            ty1 = t[-1]["bbox"][3]
            overlap = min(gy1, ty1) - max(gy0, ty0)
            if overlap < 0:
                dist = min(abs(gy0 - ty1), abs(ty0 - gy1))
                score = -dist
            else:
                score = overlap
            if score > best_overlap:
                best_overlap = score
                best_t = idx
        if t_paras:
            g_assigned[best_t].append(g)

    pairs = []
    for idx, (t, gs) in enumerate(zip(t_paras, g_assigned)):
        all_g_lines = [l for s in gs for l in s]
        t_text = " ".join(l.get("text", "").strip() for l in t)
        t_bbox = [
            min(l["bbox"][0] for l in t),
            min(l["bbox"][1] for l in t),
            max(l["bbox"][2] for l in t),
            max(l["bbox"][3] for l in t),
        ]

        if all_g_lines:
            g_text = " / ".join(l.get("text", "").strip() for l in all_g_lines)
            g_bbox = [
                min(l["bbox"][0] for l in all_g_lines),
                min(l["bbox"][1] for l in all_g_lines),
                max(l["bbox"][2] for l in all_g_lines),
                max(l["bbox"][3] for l in all_g_lines),
            ]
            g_data = {
                "text": g_text,
                "lines": all_g_lines,
                "bbox": g_bbox
            }
            combined_y0 = min(g_bbox[1], t_bbox[1])
            combined_y1 = max(g_bbox[3], t_bbox[3])
        else:
            g_data = None
            combined_y0 = t_bbox[1]
            combined_y1 = t_bbox[3]

        pairs.append({
            "pair_id": idx + 1,
            "y_range": [combined_y0, combined_y1],
            "gurbani": g_data,
            "translation": {
                "text": t_text,
                "lines": t,
                "bbox": t_bbox
            }
        })

    return {
        "header": header,
        "footer": footer,
        "footnotes": footnotes,
        "is_two_col": True,
        "pairs": pairs
    }
