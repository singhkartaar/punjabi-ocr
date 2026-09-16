"""
From recognised lines with boxes to the paragraph shape the writings pipeline reads.

lib/writings_pdf.py already knows how to turn positioned lines into
paragraphs: group_lines() by y, left_margin() for the indent a new paragraph
announces itself with, paragraphs() for the breaks (a style flip, a size
change, a wider gap, an indent). Those rules were calibrated in PDF points
on the essays and hold for a scanned book once the boxes are put in the same
units, so this module converts and hands over rather than re-deriving them.

Two things the essays got from the PDF have to come from ink here:

  bold      Gurbani is set bold in every Punjabi sample; italic shear played
            this role for the essays. Bold is read from stroke width: the mean
            of the distance transform over the ink skeleton of a line's box,
            which is robust to the line's length and to the page's darkness.
            A page's widths are bimodal when it carries both weights; Otsu's
            threshold splits them, and is accepted only when the two modes are
            BOLD_RATIO apart. A page without bold text has no split and no
            line is bold. Matched corpus text is the other bold signal, and
            the stronger one: a line found in the corpus is Gurbani whatever
            its stroke says.
  columns   per page, from lib/ocr_zones.page_columns(); the left column is
            read before the right, as read_pdf() does with its bands.
"""
from __future__ import annotations
import statistics

from lib.writings_pdf import group_lines, left_margin, paragraphs

POINTS_PER_PX = 72.0 / 300.0
# Measured on Santhya p.30 at 300 dpi: regular lines 4.04-4.20 px, bold verse
# lines 4.59-4.68, a partly bold line 4.45. The modes are close (the bold face
# is a heavier cut of the same design, not a different width class), so the
# ratio between them is small and the within-mode spread is what makes the
# split safe: 0.16 px inside regular, 0.4 px between the modes.
BOLD_RATIO = 1.06             # the bold mode must be this much wider than the regular one
MIN_WEIGHT_LINES = 3      # lines of each weight a page needs before its split is believed
MIN_INK = 30                  # px of ink for a stroke estimate to mean anything


def stroke_width(img, bbox: list) -> float | None:
    """Mean stroke width in px of the ink inside bbox, or None for an empty box."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = [int(v) for v in bbox]
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int((ink > 0).sum()) < MIN_INK:
        return None
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    # the skeleton is where the distance is a local maximum; a cheap proxy is
    # the ridge left by a one-pixel erosion
    eroded = cv2.erode(ink, np.ones((3, 3), np.uint8))
    ridge = (ink > 0) & (eroded == 0)
    core = dist[(ink > 0) & ~ridge]
    vals = core if core.size else dist[ink > 0]
    return float(2.0 * vals.mean()) if vals.size else None


def bold_split(widths: list[float]) -> float | None:
    """The stroke width above which a line is bold, or None (one weight only)."""
    ws = sorted(w for w in widths if w)
    if len(ws) < 4:
        return None
    best, best_var = None, None
    for i in range(2, len(ws) - 1):
        a, b = ws[:i], ws[i:]
        var = len(a) * statistics.pvariance(a) + len(b) * statistics.pvariance(b)
        if best_var is None or var < best_var:
            best, best_var = i, var
    lo, hi = ws[:best], ws[best:]
    # two weights need a few lines of each: on a 100 dpi scan (Ten Masters)
    # two faint lines made the "regular" group and the other 34 came out bold,
    # and half the book's prose was then set aside as quotation
    if len(lo) < MIN_WEIGHT_LINES or len(hi) < MIN_WEIGHT_LINES:
        return None
    if statistics.mean(hi) < BOLD_RATIO * statistics.mean(lo):
        return None
    return (max(lo) + min(hi)) / 2.0


def as_runs(lines: list[dict], page_h: int, scale: float = POINTS_PER_PX) -> list[dict]:
    """
    Merged lines -> the run dicts writings_pdf.group_lines expects:
    {"x", "y", "size", "italic", "text"} in points, y upward. "italic" carries
    what "bold or corpus-matched" means here: a quoted verse.
    """
    out = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        out.append({"x": x0 * scale, "y": (page_h - y1) * scale, "size": (y1 - y0) * scale,
                    "italic": bool(ln.get("bold")) or str(ln.get("kind", "")).startswith("gurbani"),
                    "text": ln.get("text", ""), "_line": ln})
    return out


def page_paragraphs(lines: list[dict], page_h: int, columns: list[tuple] | None) -> list[dict]:
    """
    Paragraphs of one page, left column then right. Each paragraph carries
    "lines": the merged line records it was built from, so a quote's
    line_ids travel with it.
    """
    runs = as_runs([ln for ln in lines if ln.get("zone", "body") == "body" and ln.get("text", "").strip()], page_h)
    if columns:
        bands = [(lo * POINTS_PER_PX, hi * POINTS_PER_PX) for lo, hi in columns]
        cols = [[r for r in runs if lo <= r["x"] < hi] for lo, hi in bands]
    else:
        cols = [runs]
    out = []
    for col in cols:
        if not col:
            continue
        grouped = group_lines(col)
        # group_lines keeps the run objects; carry our line records along
        for g in grouped:
            g["_lines"] = [r["_line"] for r in g["runs"]]
        sizes = [g["size"] for g in grouped if len(g["text"]) > 20]
        body = statistics.median(sizes) if sizes else (grouped[0]["size"] if grouped else 10.0)
        paras = paragraphs(grouped, body, left_margin(grouped))
        # paragraphs() concatenates text; map paragraphs back to their lines by
        # walking the grouped lines in order and matching accumulated text
        gi = 0
        for p in paras:
            members = []
            acc = ""
            while gi < len(grouped) and len(acc) < len(p["text"]):
                acc = (acc + " " + grouped[gi]["text"]).strip()
                members.extend(grouped[gi]["_lines"])
                gi += 1
            p["lines"] = members
            out.append(p)
    return out


def quote_meta(par: dict) -> dict:
    """line_ids and the majority shabad/ang of a quote paragraph's lines."""
    ids, shabads, angs, scores = [], [], [], []
    for ln in par.get("lines", []):
        for m in ln.get("matches", []) or ([ln["match"]] if ln.get("match") else []):
            ids.append(m["line_id"])
            shabads.append(m["shabad_id"])
            angs.append(m["ang"])
            scores.append(m["score"])
    if not ids:
        return {}
    return {"line_ids": ids, "shabad_id": statistics.mode(shabads), "ang": statistics.mode(angs),
            "match_score": round(min(scores), 3), "match_method": "ocr-corpus-match"}
